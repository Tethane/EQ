#pragma once

#include "pch.h"
#include "Engine/Backtesting.h"
#include <curl/curl.h>


namespace EQ
{
	// Retrieves data from Alpha Vantage
	// TODO: Send trade requests to Interactive Brokers for live trading
	class Connector
	{
	public:
		void query(const Backtest& backtest, devec<Databar>& data);

	private:
		std::string httpRequest(const std::string& url);

		std::map<std::string, std::map<std::string, double>> parse(const std::string& jsonData);

		bool parseDate(const std::string& date, struct std::tm& dateStruct);

		bool isBetween(const std::string& date, const std::string& startDate, const std::string& endDate);

	private:
		std::string key = "0MGUOODFZE54PIO1";
		std::string baseURL = "https://www.alphavantage.co/query";
	};

	size_t WriteCallback(void* contents, size_t size, size_t nmemb, std::string* data);
}