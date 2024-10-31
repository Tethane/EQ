#pragma once

#include "pch.h"
#include "Data/FileIO.h"
#include "Engine/Backtesting.h"

namespace EQ
{
	// Trading System
	class System
	{
	public:
		System() {}

		void parse(const std::string&);

		void addBacktest(int id);
		void removeBacktest(int id);

		std::string& getIntern() { return m_internScript; }
		std::string& getExtern() { return m_externScript; }
		std::string& getPredef() { return m_predefScript; }
		std::string& name() { return m_name; }
		void setName(const std::string& name) { m_name = name; }
		Backtest& getCurrentBacktest() { return m_backtests.back(); }
		std::vector<Backtest>& backtests() { return m_backtests; }

	private:
		void readMetadataParameters(const std::vector<Parameter>& params);
		void readMetadataBlocks(const std::vector<Block>& blocks);

	private:
		std::vector<Parameter> m_params;
		std::vector<ExecutableParameter> m_execParams;

		std::vector<Backtest> m_backtests;

		std::string m_internScript;
		std::string m_externScript;
		std::string m_predefScript;
		std::string m_name;
	};
}