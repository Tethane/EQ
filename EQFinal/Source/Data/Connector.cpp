#include "pch.h"
#include "Connector.h"
using namespace EQ;

void Connector::query(const Backtest& backtest, devec<Databar>& data)
{
	std::string ticker = backtest.ticker;
	std::string startDate = backtest.startDate;
	std::string endDate = backtest.endDate;

	std::string url = baseURL + "?function=TIME_SERIES_DAILY&symbol=" + ticker + "&apikey=" + key + "&outputsize=full";

	std::string jsonData = httpRequest(url);
	auto rawData = parse(jsonData);

	struct tm t = {};
	parseDate(startDate, t);
	time_t start = mktime(&t);

	parseDate(endDate, t);
	time_t end = mktime(&t);

	for (time_t date = start; date <= end; date += 86400)
	{
		char buffer[11];
		strftime(buffer, 11, "%Y-%m-%d", localtime(&date));
		std::string dateStr(buffer);
		if (rawData.find(dateStr) != rawData.end())
		{
			Databar db;
			for (const auto& [key, value] : rawData.at(dateStr))
			{
				if (key == "1. open")
					db.open = value;
				else if (key == "2. high")
					db.high = value;
				else if (key == "3. low")
					db.low = value;
				else if (key == "4. close")
					db.close = value;
				else if (key == "5. volume")
					db.volume = value;
			}
			data.push_back(db);
		}
	}
}

size_t EQ::WriteCallback(void* contents, size_t size, size_t nmemb, std::string* data)
{
	data->append((char*)contents, size * nmemb);
	return size * nmemb;
}

std::string Connector::httpRequest(const std::string& url)
{
	CURL* curl;
	CURLcode res;
	std::string readBuffer;

	curl = curl_easy_init();
	if (curl)
	{
		curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
		curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
		curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);
		// Set timeout options
		curl_easy_setopt(curl, CURLOPT_TIMEOUT, 10L);          // 10 seconds overall timeout
		curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 5L);    // 5 seconds connection timeout

		// Use HTTP/1.1 for better compatibility in some cases
		curl_easy_setopt(curl, CURLOPT_HTTP_VERSION, CURL_HTTP_VERSION_1_1);

		// Uncomment for verbose output for debugging purposes
		// curl_easy_setopt(curl, CURLOPT_VERBOSE, 1L);

		res = curl_easy_perform(curl);
		if (res != CURLE_OK) {
			std::cerr << "curl_easy_perform() failed: " << curl_easy_strerror(res) << std::endl;
		}
		curl_easy_cleanup(curl);
	}

	return readBuffer;
}

std::map<std::string, std::map<std::string, double>> Connector::parse(const std::string& jsonData)
{
	std::map<std::string, std::map<std::string, double>> parsedData;

	try {
		auto jsonResponse = json::parse(jsonData); // Parse JSON data

		// Access the Time Series (Daily) data
		const auto& timeSeries = jsonResponse["Time Series (Daily)"];

		for (const auto& entry : timeSeries.items()) {
			const std::string& date = entry.key(); // The date is the key
			const auto& values = entry.value(); // The values for the date

			// Store OHLCV values
			std::map<std::string, double> ohlcv;
			ohlcv["1. open"] = std::stod(values["1. open"].get<std::string>());
			ohlcv["2. high"] = std::stod(values["2. high"].get<std::string>());
			ohlcv["3. low"] = std::stod(values["3. low"].get<std::string>());
			ohlcv["4. close"] = std::stod(values["4. close"].get<std::string>());
			ohlcv["5. volume"] = std::stod(values["5. volume"].get<std::string>());

			parsedData[date] = ohlcv; // Add the date and values to the map
		}
	}
	catch (json::parse_error& e) {
		std::cerr << "JSON parse error: " << e.what() << std::endl;
	}
	catch (json::type_error& e) {
		std::cerr << "JSON type error: " << e.what() << std::endl;
	}

	return parsedData;
}

bool Connector::parseDate(const std::string& date, struct std::tm& dateStruct)
{
	std::istringstream ss(date);
	ss >> std::get_time(&dateStruct, "%Y-%m-%d");
	return !ss.fail();
}

bool Connector::isBetween(const std::string& date, const std::string& startDate, const std::string& endDate)
{
	return date >= startDate && date <= endDate;
}