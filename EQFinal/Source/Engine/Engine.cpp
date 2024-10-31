#include "pch.h"
#include "Engine.h"

using namespace EQ;

void Engine::activate()
{
	m_running = true;
	m_processing = false;
	m_hostThread = std::thread(&Engine::runMainThread, this);
}

void Engine::deactivate()
{
	m_running = false;

	// Join all of the worker threads

	for (size_t i = 0; i < m_workerThreads.size(); ++i)
	{
		if (m_workerThreads[i].joinable())
		{
			m_workerThreads[i].join();
		}
	}

	// Clear the command queue and join the background thread

	for (auto command : m_commandQueue.queue)
	{
		removeCommand(command->id);
	}

	if (m_backThread.joinable())
	{
		m_backThread.join();
	}

	m_processing = false;

	// Join the main thread

	if (m_hostThread.joinable())
	{
		m_hostThread.join();
	}
}

void Engine::addCommand(Ptr<EngineCommand> engineCommand)
{
	std::lock_guard<std::mutex> lock(m_mutex);
	m_commandQueue.push(engineCommand);
}

void Engine::removeCommand(int id)
{
	std::lock_guard<std::mutex> lock(m_mutex);
	for (auto it = m_commandQueue.queue.begin(); it != m_commandQueue.queue.end(); ++it)
	{
		if ((*it)->id == id)
		{
			(*it)->cancelled.store(true);
			m_commandQueue.queue.erase(it);
			return;
		}
	}
	if (m_currentCommand && m_currentCommand->id == id)
	{
		m_currentCommand->cancelled.store(true);
	}
}

void Engine::setBacktestProcessedCallback(BacktestProcessedCallback callback)
{
	onBacktestProcessed = callback;
}


void Engine::runMainThread()
{
	while (m_running)
	{
		auto frameStartTime = std::chrono::steady_clock::now();

		if (!m_commandQueue.empty() && !m_processing.load())
		{
			m_processing.store(true);

			if (m_backThread.joinable())
			{
				m_backThread.join();
			}

			m_backThread = std::thread(&Engine::processCommands, this);
		}

		/*
		if (!m_portfolios.empty())
		{
			int i = 0;
			for (auto it = m_portfolios.begin(); it != m_portfolios.end(); ++it)
			{
				// If thread is joinable, then work is complete
				if (m_workerThreads[i].joinable())
				{
					m_workerThreads[i].join();
				}
				else
				{
					m_workerThreads[i] = std::thread(&Engine::trade, this, std::ref((*it).second));
				}
				++i;
			}
		}
		std::cerr << "Running on main thread" << std::endl;
		*/

		std::this_thread::sleep_until(frameStartTime + std::chrono::seconds(1));
	}
}

void Engine::processCommands()
{
	while (true)
	{
		if (m_commandQueue.empty())
		{
			m_processing = false;
			break;
		}

		Ptr<EngineCommand> command = m_commandQueue.pop();
		m_currentCommand = command;

		backtest();

		// Backtest over, print the backtest through callback

		Backtest result = m_currentCommand->user->getBacktest(m_currentCommand->id);
		std::string source = result.serialize();

		if (onBacktestProcessed)
		{
			onBacktestProcessed(source);
		}
	}
}

void Engine::backtest()
{
	// 1. Parse the source code to get the backtest metadata information

	System system;
	system.addBacktest(m_currentCommand->id);
	system.parse(m_currentCommand->source); // Parse the system script + metadata

	// 2. Connector queries data for the engine

	Backtest backtest = system.getCurrentBacktest();

	devec<Databar> data;
	m_connector->query(backtest, data);

	// 3. Generate Parameter Combinations (currently brute force, we are going to fix this)

	std::vector<std::vector<int>> paramSets;

	for (const auto& param : backtest.parameterDefinitions)
	{
		std::vector<int> pSet;

		std::visit([&](auto&& value) {
			using T = std::decay_t<decltype(value)>;

			if constexpr (std::is_same_v<T, int>)
			{
				pSet.push_back(value);
			}
			else if constexpr (std::is_same_v<T, std::pair<int, int>>)
			{
				for (size_t i = value.first; i <= value.second; ++i)
				{
					pSet.push_back(i);
				}
			}
			else if constexpr (std::is_same_v<T, std::vector<int>>)
			{
				pSet = value;
			}
			}, param.value);
	}

	std::vector<std::vector<int>> combinations;
	std::vector<int> currentCombination(paramSets.size());

	generateCombinations(paramSets, currentCombination, 0, combinations);

	size_t rows = combinations.size();
	size_t cols = combinations[0].size();
	size_t totalSize = rows * cols;

	devec<int> paramCombinations;

	for (size_t i = 0; i < rows; ++i)
	{
		for (size_t j = 0; j < cols; ++j)
		{
			paramCombinations.push_back(combinations[i][j]);
		}
	}

	// 4. Setup AggregatePerformanceData array

	devec<AggregatePerformanceData> results;
	results.allocate(paramCombinations.size());

	// 5. Setup AccountData

	AccountData accountData;
	accountData.equity = backtest.capital;
	accountData.cash = accountData.equity;
	accountData.netAL = 0.0f;

	// 6. Generate the CUDA file

	std::string source = generateCudaFile(system.getIntern(), system.getExtern());

	// 7. Dynamic Compilation of the CUDA

	try
	{
		//	7.1. Create Nvrtc Program
		nvrtcProgram prog;
		cudaCheckNvrtc(nvrtcCreateProgram(&prog, source.c_str(), "eqCudaBacktest.cu", 0, nullptr, nullptr));

		const char* options[] =
		{
			"--relocatable-device-code=true",
			"--gpu-architecture=compute_86"
		};

		//	7.2. Compile Program and PTX
		nvrtcResult compileResult = nvrtcCompileProgram(prog, 2, options);
		if (compileResult != NVRTC_SUCCESS)
		{
			size_t logSize;
			cudaCheckNvrtc(nvrtcGetProgramLogSize(prog, &logSize));
			std::vector<char> log(logSize);
			cudaCheckNvrtc(nvrtcGetProgramLog(prog, log.data()));
			std::stringstream errMsg;
			errMsg << "Compilation failed with log: " << log.data() << "\n";
			throw std::runtime_error(errMsg.str());
		}

		size_t ptxSize;
		cudaCheckNvrtc(nvrtcGetPTXSize(prog, &ptxSize));
		std::vector<char> ptx(ptxSize);
		cudaCheckNvrtc(nvrtcGetPTX(prog, ptx.data()));
		cudaCheckNvrtc(nvrtcDestroyProgram(&prog));

		//	7.3. Initialize Driver API
		cudaCheck(cuInit(0));

		CUdevice cuDevice;
		cudaCheck(cuDeviceGet(&cuDevice, 0));
		CUcontext cuContext;
		cudaCheck(cuCtxCreate(&cuContext, 0, cuDevice));

		//	7.4. Create Link State (sneaky link)
		CUlinkState linkState;
		CUjit_option jitOptions[] = { CU_JIT_OPTIMIZATION_LEVEL, CU_JIT_LOG_VERBOSE };
		void* jitOptionValues[] = { (void*)4, (void*)1 };

		cudaCheck(cuLinkCreate(2, jitOptions, jitOptionValues, &linkState));

		const char* cudaPath = std::getenv("CUDA_PATH");
		if (!cudaPath)
		{
			throw std::runtime_error("CUDA_PATH environment variable is not set");
		}

		std::string devRtLibPath = std::string(cudaPath) + "\\lib\\x64\\cudadevrt.lib";

		cudaCheck(cuLinkAddFile(linkState, CU_JIT_INPUT_LIBRARY, devRtLibPath.c_str(), 0, 0, 0));

		//	7.5 Add PTX Data to Linker
		cudaCheck(cuLinkAddData(linkState, CU_JIT_INPUT_PTX, (void*)ptx.data(), ptxSize, "eqCudaBacktest.ptx", 0, 0, 0));

		//	7.6. Complete Linking Process
		void* cubinOut;
		size_t cubinSize;
		cudaCheck(cuLinkComplete(linkState, &cubinOut, &cubinSize));

		//	7.7. Load Linked Module
		CUmodule cuModule;
		cudaCheck(cuModuleLoadData(&cuModule, cubinOut));
		cuLinkDestroy(linkState);

		//	7.8. Get Kernel Object
		CUfunction mainKernel;
		cudaCheck(cuModuleGetFunction(&mainKernel, cuModule, "parameterOptimizationKernel"));


		// 8. Run the Kernel
		int threadsPerBlock = 256;
		int blocksPerGrid = (combinations.size() + threadsPerBlock - 1) / threadsPerBlock;

		Databar* dataPtr = data.data();
		int dataLength = data.size();
		int* paramCombinationsPtr = paramCombinations.data();
		AggregatePerformanceData* resultsPtr = results.data();

		void* args[] = {
			&dataPtr,
			&dataLength,
			&paramCombinationsPtr,
			&cols,
			&rows,
			&resultsPtr,
			&accountData
		};

		cudaCheck(cuLaunchKernel(
			mainKernel,
			blocksPerGrid, 1, 1,
			threadsPerBlock, 1, 1,
			0, 0,
			args, 0
		));

		cudaCheck(cuCtxSynchronize());

		// 9. Process Results and Push to User's Backtests
		Backtest finalResults = backtest;

		// Populate results
		int bestIndex = 0;
		float maxPNL = -1000000.0f;
		std::vector<int> bestParams;

		for (size_t i = 0; i < combinations.size(); ++i)
		{
			finalResults.resultsByParameterSet.push_back(results[i]);
			maxPNL = max(maxPNL, results[i].totalPNL);
		}

		for (size_t i = 0; i < combinations.size(); ++i)
		{
			if (results[i].totalPNL == maxPNL)
			{
				bestIndex = i;
				break;
			}
		}

		finalResults.optimalParameterSet = combinations[bestIndex];

		m_currentCommand->user->addBacktest(system, finalResults);
	}
	catch (const std::exception& e)
	{
		std::cerr << "Error: " << e.what() << std::endl;
	}
}

void Engine::insertAfterMarker(std::string& code, const std::string& marker, const std::string& insertion)
{
	size_t pos = code.find(marker);
	if (pos == std::string::npos)
	{
		std::cerr << "Marker not found in boilerplate: " << marker << std::endl;
		return;
	}

	pos = code.find('\n', pos);
	if (pos == std::string::npos)
	{
		std::cerr << "Newline not found after markerr: " << marker << std::endl;
		return;
	}

	code.insert(pos + 1, insertion + "\n");
}

std::string Engine::generateCudaFile(std::string& internScript, std::string& externScript)
{
	std::string code;
	readTxtFile("generation.txt", code);

	insertAfterMarker(code, "// Intern", internScript);
	insertAfterMarker(code, "// Extern", externScript);

	return code;
}

void Engine::generateCombinations(const std::vector<std::vector<int>>& inputs, std::vector<int>& currentCombination, int depth, std::vector<std::vector<int>>& result)
{
	if (depth == inputs.size())
	{
		result.push_back(currentCombination);
		return;
	}

	for (int i = 0; i < inputs[depth].size(); ++i)
	{
		currentCombination[depth] = inputs[depth][i];
		generateCombinations(inputs, currentCombination, depth + 1, result);
	}
}