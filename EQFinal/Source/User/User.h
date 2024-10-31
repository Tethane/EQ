#pragma once

#include "pch.h"

#include "Engine/Backtesting.h"
#include "System/System.h"

namespace EQ
{
	// User (for future use in the C# WPF Application when multiple users could use the same machine)
	class User
	{
	public:
		User() {}

		void addBacktest(System& system, const Backtest& data);
		void deleteBacktest(int id);
		Backtest& getBacktest(int id);

		void addSystem(const System& system);
		void deleteSystem(const std::string& name);
		System getSystem(const std::string& name);

	private:
		std::vector<System> m_systems; // Backtests are stored within systems
	};
}