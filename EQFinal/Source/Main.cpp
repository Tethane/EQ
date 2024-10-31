#include "pch.h"

#include "App/Application.h"
#include "Engine/Backtesting.h"

void displayMenu(int& choice)
{
	std::cout << "\nOptions:\n";
	std::cout << "1. Start new backtest\n";
	std::cout << "2. View previous backtest\n";
	std::cout << "3. Exit\n";
	std::cout << "Enter (number): ";
	std::cin >> choice;
}

void displayBacktest(const Backtest& backtest)
{
	std::cout << "\n\nBacktest Complete\n";
	std::cout << "================\n";
	std::cout << "System Name: " << backtest.name << "\n";
	std::cout << "Backtest ID: " << backtest.id << "\n";
	std::cout << "Tested on Target (Ticker): " << backtest.ticker << "\n";
	std::cout << "Backtest Period Start: " << backtest.startDate << "\n";
	std::cout << "Backtest Period End: " << backtest.endDate << "\n";
	std::cout << "================\n";
	std::cout << "Parameter Definitions:\n";
	for (const auto& param : backtest.parameterDefinitions)
	{
		std::cout << "Parameter Name: " << param.name << "\n";
	}
	std::cout << "\nOptimal Parameter Set\n";
	for (size_t i = 0; i < backtest.optimalParameterSet.size(); ++i)
	{
		std::cout << backtest.parameterDefinitions[i].name << ": " << backtest.optimalParameterSet[i] << "\n";
	}

	float maxPNL = -100000.0f;

	for (int i = 0; i < backtest.resultsByParameterSet.size(); ++i)
	{
		auto results = backtest.resultsByParameterSet[i];
		std::cout << "Results for param combination: " << i << "\n";
		std::cout << "Starting Equity: " << results.startEquity << " End Equity: " << results.endEquity << "\n";
		std::cout << "Total PNL: " << results.totalPNL << " Max Drawdown: " << results.maxDrawdown << "\n";
		std::cout << "Sharpe Ratio: " << results.sharpeRatio << " Sortino Ratio: " << results.sortinoRatio << "\n";
		std::cout << "Total Trades: " << results.totalTrades << " Winners: " << results.winners << " Losers: " << results.losers << " Success Rate: " << results.successRate << "\n";
		std::cout << "Best Trade: " << results.bestTrade << " Worst Trade: " << results.worstTrade << "\n";
		std::cout << "Stoplosses Hit: " << results.stoplossesHit << " Take Profits Hit: " << results.takeprofitsHit << "\n";
		std::cout << "\n";
		std::cout << "\n";

		maxPNL = max(maxPNL, results.totalPNL);
	}

	int bestIndex = 0;

	for (int i = 0; i < backtest.resultsByParameterSet.size(); ++i)
	{
		if (backtest.resultsByParameterSet[i].sharpeRatio == maxPNL)
		{
			bestIndex = i;
			break;
		}
	}

	std::cout << "\n==============\n";
	std::cout << "Results: " << std::endl;
	std::cout << "Best Parameter Combination at Index: " << bestIndex << std::endl;
	std::cout << "Max Profit: " << maxPNL << std::endl;
}

int main(int argc, char* argv)
{
	auto app = EQ::Application::instance();

	app->init();

	app->activateEngine();

	int commandId = 0;
	bool isProcessing = false;

	EQ::Ptr<EQ::User> mainUser = std::make_shared<EQ::User>();

	while (app->getEngine()->isRunning())
	{
		int choice;
		
		if (isProcessing)
		{
			std::this_thread::sleep_for(std::chrono::seconds(1));
			std::cout << "\nProcessing...\n";
			continue;
		}
		else
		{
			displayMenu(choice);
		}
		
		switch (choice)
		{
		case 1:
		{
			// C:\Users\Owner\source\repos\EQFinal\EQFinal\Tests\breakout.cu
			std::string codePath;

			std::cout << "Enter path to strategy code file: ";
			std::cin >> codePath;

			std::string source;
			EQ::readTxtFile(codePath, source);

			EQ::Ptr<EQ::EngineCommand> command = std::make_shared<EQ::EngineCommand>(source, mainUser, commandId);

			command->callback = [&](const std::string& result) {
				isProcessing = false;
				Backtest backtestResult = Backtest::deserialize(result);
				displayBacktest(backtestResult);
			};

			app->addEngineBacktestCommand(command);

			isProcessing = true;

			std::cout << "Processing backtest with backtest ID: " << commandId << " and code at: " << codePath << "\n";
			std::cout << "Results will be available shortly...\n";
			break;
		}
		case 2:
		{
			std::cout << "Enter backtest ID: ";
			int id;
			std::cin >> id;
			Backtest backtest = mainUser->getBacktest(id);

			displayBacktest(backtest);
			break;
		}
		case 3:
		{
			std::cout << "Exiting Program...\n";
			app->deactivateEngine();
			break;
		}
		}

	}

	app->destroy();

	return 0;
}