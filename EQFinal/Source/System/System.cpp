#include "pch.h"
#include "System.h"

using namespace EQ;

void System::readMetadataParameters(const std::vector<Parameter>& params)
{
	for (const auto& param : params)
	{
		ExecutableParameter p;

		p.name = param.name;
		p.type = param.type;

		std::visit([&](auto&& value) {
			using T = std::decay_t<decltype(value)>;

			if constexpr (std::is_same_v<T, std::string>)
			{
				p.value = std::stoi(value);
			}
			else if constexpr (std::is_same_v<T, std::pair<std::string, std::string>>)
			{
				int minVal = std::stoi(value.first);
				int maxVal = std::stoi(value.second);

				p.value = std::make_pair(minVal, maxVal);
			}
			else if constexpr (std::is_same_v<T, std::vector<std::string>>)
			{
				std::vector<int> vals;
				for (const auto& val : value)
				{
					vals.push_back(std::stoi(val));
				}
				p.value = vals;
			}
			}, param.value);

		m_execParams.push_back(p);
		m_backtests.back().parameterDefinitions.push_back(p);
	}
}

void System::readMetadataBlocks(const std::vector<Block>& blocks)
{
	for (auto& block : blocks)
	{
		for (auto& kv : block.keyValues)
		{
			if (kv.first == "start")
			{
				m_backtests.back().startDate = kv.second;
			}
			else if (kv.first == "end")
			{
				m_backtests.back().endDate = kv.second;
			}
			else if (kv.first == "interval")
			{
				m_backtests.back().interval = kv.second;
			}
			else if (kv.first == "batch")
			{
				m_backtests.back().batch = std::stoi(kv.second);
			}
			else if (kv.first == "target")
			{
				m_backtests.back().ticker = kv.second;
			}
			else if (kv.first == "folds")
			{
				m_backtests.back().folds = std::stoi(kv.second);
			}
			else if (kv.first == "engine")
			{
				m_backtests.back().engine = [](const std::string& s) -> int {
					if (s == "GRID")
						return 0;
					else if (s == "RANDOM")
						return 1;
					else if (s == "BAYESIAN")
						return 2;
					else
						return 3;
					}(kv.second);
			}
			else if (kv.first == "commission")
			{
				m_backtests.back().commission = std::stof(kv.second);
			}
			else if (kv.first == "slippage")
			{
				m_backtests.back().slippage = [](const std::string& s) -> int {
					if (s == "RANDOM")
						return 0;
					else if (s == "FIXED_PERFECT")
						return 1;
					else if (s == "FIXED_AVERAGE")
						return 2;
					else
						return 3;
					}(kv.second);
			}
			else if (kv.first == "capital")
			{
				m_backtests.back().capital = std::stof(kv.second);
			}
			else if (kv.first == "name")
			{
				m_backtests.back().name = kv.second;
			}
		}
	}
}

void System::parse(const std::string& source)
{
	try
	{

		// 1. Get the predef script in continuous form

		extractBlocks(source, "<predef>", "</predef>", m_predefScript);

		removeWhitespace(m_predefScript);

		std::vector<Token> tokens = tokenize(m_predefScript);
		std::vector<Block> blocks;
		std::vector<Parameter> parameters;


		parseTokens(tokens, blocks, parameters);

		// 2. Read the metadata from the predef section

		readMetadataParameters(parameters);

		// readMetadataBlocks(blocks);

		readMetadataBlocks(blocks);

		// 3. Get the intern and extern scripts in continuous form

		extractBlocks(source, "<intern>", "</intern>", m_internScript);
		extractBlocks(source, "<extern>", "</extern>", m_externScript);

		// 4. Process the intern and extern scripts

		addDeviceHostQualifiers(m_internScript);

		// Replace the system()

		std::string signature = "System()";
		size_t pos = m_externScript.find(signature);
		if (pos == std::string::npos)
			throw std::runtime_error("Function System() signature not found in extern script");

		m_externScript.replace(
			pos,
			signature.length(),
			"__device__ OrderCombination system(const devec<Databar>& data, int index, const devec<int>& parameters, const AccountData& accountData)");

		// 5. Add the parameters to the start of the main function

		insertParameterDefinitions(m_externScript, parameters);
	}
	catch (const std::exception& e)
	{
		std::cerr << "Error: " << e.what() << "\n";
	}
}

void System::addBacktest(int id)
{
	Backtest backtest;
	backtest.id = id;
	m_backtests.push_back(backtest);
}

void System::removeBacktest(int id)
{
	for (size_t i = 0; i < m_backtests.size(); ++i)
	{
		if (m_backtests[i].id == id)
		{
			m_backtests.erase(m_backtests.begin() + i);
		}
	}
}