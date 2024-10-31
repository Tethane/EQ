#pragma once

#include "pch.h"
#include "Commands/Command.h"
#include "Data/Connector.h"
#include "Data/FileIO.h"

namespace EQ
{
	class Engine
	{
	public:
		Engine()
		{
			m_connector = std::make_shared<Connector>();
		}

		~Engine()
		{

		}

		void activate();
		void deactivate();

		void addCommand(Ptr<EngineCommand>);
		void removeCommand(int id);
		
		bool isRunning() const { return m_running.load(); }
		bool isProcessing() const { return m_processing.load(); }

	private:

		void runMainThread();
		void processCommands();

		void insertAfterMarker(std::string& code, const std::string& marker, const std::string& insertion);
		std::string generateCudaFile(std::string& internScript, std::string& externScript);
		void generateCombinations(const std::vector<std::vector<int>>& inputs, std::vector<int>& currentCombination, int depth, std::vector<std::vector<int>>& result);

		void backtest();

	private:
		std::thread m_hostThread; // I/O

		std::vector<std::thread> m_workerThreads; // Live trading

		std::thread m_backThread; // Backtesting thread

		std::atomic<bool> m_running;
		std::atomic<bool> m_processing;

		Ptr<Connector> m_connector;

		Queue<Ptr<EngineCommand>> m_commandQueue;
		Ptr<EngineCommand> m_currentCommand;
		std::mutex m_mutex;
	};
}