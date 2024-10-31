<predef>

parameter<int>("openLength") = optimize<5, 50>;
parameter<int>("closeLength") = optimize<[10, 15, 20]>;

source(
	name = "Donchian Channel Breakout"
	target = "TSLA",
	interval = "1H",
	start = "2023-02-12",
	end = "2024-02-12",
	batch = 30
);

engine(
	engine = GRID,
	folds = 3
);

settings(
	capital = 100000.0f,
	slippage = RANDOM,
	commission = 0.005f
);

</predef>

<intern>

int testAddTwoNumbers(int a, int b)
{
	return a + b;
}

</intern>

<extern>

System()
{
	float high = 0.0f;
	float low = 100000.0f;
	float fivebarHigh = 0.0f;
	float fivebarLow = 100000.0f;

	float maxLoss = 1000.0f;
	float positionSize = 1 * accountData.equity; // Always trade 10% of account size

	OrderCombination result;
	Order openOrder;
	Order closeOrder;

	if (index <= openLength || index <= closeLength)
	{
		openOrder.type = HOLD;
		closeOrder.type = HOLD;

		result.main = openOrder;
		result.alt = closeOrder;

		return result;
	}

	for (int i = 1; i <= openLength; ++i)
	{
		high = (high > data[index - i].close ? high : data[index - i].close);
		low = (low < data[index - i].low ? low : data[index - i].low);
	}

	for (int i = 1; i <= closeLength; ++i)
	{
		fivebarHigh = (fivebarHigh > data[index - i].close ? fivebarHigh : data[index - i].close);
		fivebarLow = (fivebarLow < data[index - i].low ? fivebarLow : data[index - i].low);
	}

	// If no current open position

	openOrder.entryPrice = data[index].close;
	openOrder.shares = (positionSize / openOrder.entryPrice);

	if (data[index].close > high)
		openOrder.type = BUY;
	else if (data[index].close < low)
		openOrder.type = SELL;
	else
		openOrder.type = HOLD;

	openOrder.stoploss = maxLoss;

	// If there is already an open position
	closeOrder.entryPrice = data[index].close;
	//closeOrder.shares = (positionSize / openOrder.entryPrice);
	closeOrder.shares = abs(accountData.netShares);
	if (data[index].close > fivebarHigh && accountData.netShares < 0)
		closeOrder.type = BUY;
	else if (data[index].close < fivebarLow && accountData.netShares > 0)
		closeOrder.type = SELL;
	else
		closeOrder.type = HOLD;

	result.main = openOrder;
	result.alt = closeOrder;

	return result;
}

</extern>