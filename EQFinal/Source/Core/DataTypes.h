#pragma once

#include "pch.h"

namespace EQ
{
	// Primitive data types for backtesting (these are all redefined in the engine cuda generated code)

	enum OrderType { BUY = 1, SELL = -1, HOLD = 0 };

	#define FLOAT_MAX 1000000.0f
	#define FLOAT_MIN -1000000.0f

	struct Databar
	{
		__host__ __device__ Databar() {}

		float open = 0.0f;
		float high = 0.0f;
		float low = 0.0f;
		float close = 0.0f;
		float volume = 0.0f;
	};

	struct Order
	{
		__host__ __device__ Order() {}

		OrderType type = HOLD;
		float entryPrice = 0.0f;
		float stoploss = 0.0f;
		float takeprofit = 0.0f;
		int shares = 0;
	};

	// When backtesting, main is used when there is no currently open trade/defined condition is false, and alt is used when that condition is true
	struct OrderCombination
	{
		Order main;
		Order alt;
	};

	struct AccountData
	{
		__host__ __device__ AccountData() {}

		// Equity = cash + (assets - liabilities)
		float equity = 0.0f; // Equity = Assets - Liabilities;
		float cash = 0.0f; // Cash is an asset
		float netAL = 0.0f; // Net assets - liabilities (excluding cash)

		int netShares = 0;
		float avgPrice = 0.0f;
		float stoploss = 0.0f; // Max loss that can be taken for current trade
		float takeprofit = 0.0f; // Max gain that can be taken for current trade
	};

	// For every time step through the backtest, AccountData is updated
	struct AccountDataTracker
	{
		__host__ __device__ AccountDataTracker() {}
		devec<AccountData> tracker;
		int stoplossesHit = 0;
		int takeprofitsHit = 0;
	};

	class Security
	{
	public:
		const char* ticker;
		std::vector<Databar> prices;

		Security() : ticker("XYZ") {}
		Security(const char* name) : ticker(name) {}

		void addDatabar(const Databar& db)
		{
			prices.push_back(db);
		}
	};

	struct Trade
	{
		float entryPrice = 0.0f;
		float exitPrice = 0.0f;
		int entryShares = 0;
		int exitShares = 0;
		float realizedPnL = 0.0f;
		float unrealizedPnL = 0.0f;
		float risk = 0.0f;
		bool isOpen = true;
		bool isShort = false;
		float stoploss = 0.0f;
		float takeprofit = 0.0f;

		__host__ __device__ Trade() {}

		__host__ __device__ void updateTrade(float price, int shares, float _stoploss = 0.0f, float _takeprofit = 0.0f)
		{
			entryPrice = ((entryPrice * entryShares) + (price * shares)) / (entryShares + shares);
			entryShares += shares;
			stoploss = _stoploss;
			takeprofit = _takeprofit;
		}

		__host__ __device__ void closeTrade(float price, int shares)
		{
			exitPrice = price;
			exitShares += shares;

			if (isShort)
				realizedPnL += (entryPrice - exitPrice) * shares;
			else
				realizedPnL += (exitPrice - entryPrice) * shares;

			if (exitShares >= entryShares)
				isOpen = false;
		}

		__host__ __device__ void updateUnrealizedPnL(float currentPrice)
		{
			if (isShort)
				unrealizedPnL = (entryPrice - currentPrice) * entryShares;
			else
				unrealizedPnL = (currentPrice - entryPrice) * entryShares;
		}
	};

	class TradeTracker
	{
	public:
		devec<Trade> completedTrades;
		Trade currentTrade;

		__host__ __device__ TradeTracker()
		{
			currentTrade = Trade();
			currentTrade.isOpen = false;
		}

		__host__ __device__ void update(float price, const AccountData& previousData, const AccountData& currentData, float stoploss = 0.0f, float takeprofit = 0.0f)
		{
			int prevShares = previousData.netShares;
			int currShares = currentData.netShares;
			int shareChange = currShares - prevShares;

			if ((prevShares > 0 && currShares < prevShares) || (prevShares < 0 && currShares > prevShares))
			{
				closeTrade(price, (abs(shareChange) >= abs(prevShares) ? abs(prevShares) : abs(shareChange)));

				if (currShares * prevShares < 0)
				{
					int remainingShares = abs(currShares);
					bool isShort = currShares < 0;
					openTrade(price, remainingShares, isShort, stoploss, takeprofit);
				}
			}
			else if (shareChange != 0)
			{
				if (prevShares == 0)
				{
					bool isShort = currShares < 0;
					openTrade(price, abs(currShares), isShort, stoploss, takeprofit);
				}
				else if (currShares > 0 && prevShares > 0)
				{
					currentTrade.updateTrade(price, shareChange, stoploss, takeprofit);
				}
				else if (currShares < 0 && prevShares < 0)
				{
					currentTrade.updateTrade(price, -shareChange, stoploss, takeprofit);
				}
			}

			updateUnrealizedPnL(currentData);
		}

	private:
		__host__ __device__ void openTrade(float price, int shares, bool isShort, float stoploss = 0.0f, float takeprofit = 0.0f)
		{
			currentTrade = Trade();
			currentTrade.entryPrice = price;
			currentTrade.entryShares = shares;
			currentTrade.isShort = isShort;
			currentTrade.stoploss = stoploss;
			currentTrade.takeprofit = takeprofit;
		}

		__host__ __device__ void closeTrade(float price, int shares)
		{
			if (currentTrade.isOpen)
			{
				currentTrade.closeTrade(price, shares);
				if (!currentTrade.isOpen)
					completedTrades.push_back(currentTrade);
			}
		}

		__host__ __device__ void updateUnrealizedPnL(const AccountData& currentData)
		{
			if (currentTrade.isOpen)
				currentTrade.updateUnrealizedPnL(currentData.equity / abs(currentData.netShares));
		}
	};

	struct Parameter
	{
		std::string type;
		std::string name;
		std::variant<
			std::string,
			std::pair<std::string, std::string>,
			std::vector<std::string>
		> value;
	};
}