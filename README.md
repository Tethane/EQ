# Equilibrium

An algotrading backtesting engine designed for programmers

## Mission:

This is a prototype project looking for continued development in the next few years to "equalize" the markets. Automated trading has taken over the bulk of the global markets' volume, especially with the development of high frequency trading systems to provide liquidity. However, as a consequence of this paradigm shift, traders are resorting to computerized strategies more than ever to keep up. Technical analysis and chart watching is becoming a science of the past.   

As a non-institutional investor, access to alpha-generating systems is extremely hard to find. Algorithmic traders often have to build their own testing platforms from scratch and design complex individualized components to design, test, and deploy their systems into the live market.

For algorithmic traders who write systems, often they must design testing and deployment systems tailored to the individual system, rather than have an abstract engine to facilitate and encapsulate this process. This reduces the flexibility and speed of development.

This process is a major deterrent for most traders considering computerizing their strategies. At the very least, understanding algorithmic trading often requires studying a lot of programming and math related to quantitative finance. In essence, there are high barriers to entry for flexible, rapid development of automated trading systems.

My goal is to build an engine that will abstract the challenging part of algorithmic trading and allow traders of all types to write their own systems, as complicated as they want, with minimal programming experience. All they need to know is the logic of their system. 

(I'm also learning along the way)

## What
- Unmatched flexibility when it comes to designing structured programmable trading systems. Using a modified version of C++ called EQC++, a trader can write GPU accelerated code without any underlying knowledge of Equilibrium's architecture.
- User can write any position-sizing algorithm they want.
- Write your own parameterized trading systems
- Parameter optimization (currently only using GPU-accelerated exhaustive search)
- Backtest your system against any set of data for a valid ticker and valid start and end dates (daily data only, because I'm a free API user)
- For each parameter combination, backtest data is returned. This can be useful to a trader for finding useful patterns in successful and unsuccessful parameter combinations, for instance identifying areas of "system stability," where successful parameter combinations are clustered, so if the market dynamics change slightly, little impact to the overall performance is expected.
- Console interface for interacting with the engine.
- NO exposure of the engine to the user. User only needs to know the logic of their system.
- Uses Nvidia CUDA for GPU acceleration
- User systems are compiled dynamically and run on the user's local machine (must have an Nvidia GPU supporting CUDA)
- Multiple trading systems can be queued for backtesting
- Multi-threading for asynchronous task processing and responsiveness to user input.
- Provides a universal data structure `devec<typename T>` (device-vector) that uses CUDA's Unified Memory for interaction between the host-side and device-side code. To the client, a `devec` can be used almost exactly the same way as a `std::vector`. When writing EQC++ code, you must use `devec` instead of `std::vector`

## What to look forward to
- Bug fixes
- Integration with EQUIX, the UI/UX C# WPF application that is being developed on the side for easier use of the engine.
- Integration of BoTorch (through Python bindings) for Bayesian optimization as an alternative to GPU-accelerated exhaustive search for parameter optimization.
- Integration with EQ Portfolio Optimizer. Provide various stocks OR backtested systems, and the engine will find the "optimal" portfolio weightings based on Modern Portfolio Theory (risk, reward)

## Requirements / External Libraries Used
- Nlohmann's Json for Modern C++ (included as json.hpp in EQFinal\External)
- Libcurl (for connections to the API)

Everything else was made from scratch.

## Build

Beware, this project has not yet been tested on another machine. (I know that is "Well, it works on my machine," but unfortunately I don't have anybody with a Nvidia GPU to test this project with)

I built this project in Visual Studio 2022 Community.

There are a few instances of using absolute paths that you may need to change for your specific system. In the function in `Engine.cpp`
```cpp
std::string Engine::generateCudaFile(std::string& internScript, std::string& externScript)
{
	std::string code;
	readTxtFile("C:\\Users\\Owner\\source\\repos\\EQFinal\\EQFinal\\Source\\Engine\\generation.txt", code);

	insertAfterMarker(code, "// Intern", internScript);
	insertAfterMarker(code, "// Extern", externScript);

	return code;
}
```

Build with x64 to Debug or Release and it goes to EQFinal/Build/(Platform)/(Configuration)

## EQC++

EQC++ is my version of a scripting language that I created that is compatible with the Nvidia CUDA GPU architecture. 

It is syntactically similar to C++. One of key features of EQC++ is its ability to be written as if it can only be run on the CPU, but it can also run on the GPU for parallelized testing. This allows EQ scripts to be rapidly tested on optimized hardware, a crucial aspect for systematic trading development.

There are three sections of an EQC++ program: predef, intern, and extern.

- `predef` enclosed by the markup style tags `<predef>` and `</predef>` the code in here does not have C++ syntax, but instead provides definitions for the metadata of the trading system. Here is where you define the strategy parameters. 
- To define a parameter, there are three ways
	- By value: `parameter<int>("NAME_OF_PARAMETER") = optimize<5>;`
	- By values: `parameter<int>("NAME") = optimize<[1, 2, 3, 4]>;`
	- By lower and upper bound: `parameter<int>("NAME") = optimize<1, 100>;`

```cpp
<predef>

parameter<int>("openLength") = optimize<5, 50>;
parameter<int>("closeLength") = optimize<[10, 15, 20]>;

source(
	name = "Example: Donchian Channel Breakout"
	target = "TSLA",
	interval = "1D",
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
```

- `intern` enclosed by the tags `<intern>` and `</intern>`. This is where you would write helper functions to be used in the main trading system. This looks like normal C++ code.

```cpp
<intern>

int sumDevec(const devec<int>& data)
{
	int sum = 0;
	for(size_t i = 0; i < data.size(); ++i)
	{
		sum += data[i]; // Data is stored in Unified Memory if created on host or global memory if created on device
	}
	return sum;
}
</intern>
```

- `extern` enclosed by the tags `<extern>` and `</extern>`. This is where you would write the main trading system. You must write your system within an explicit function called `System()` which has no return value (syntactically, but technically it does)
- However, your `System()` must return an `OrderCombination` 
- In your system, you can use the parameters you defined in the `predef` section as normal variables. You do not have to redefine them here.

```cpp
<extern>

System()
{
	// Write your strategy here and don't forget to return an OrderCombination
}

</extern>
```

