#include "pch.h"
#include "User.h"
using namespace EQ;

void User::addBacktest(System& system, const Backtest& data)
{
	for (auto& sys : m_systems)
	{
		if (sys.name() == data.name)
		{
			sys.removeBacktest(data.id);
			sys.addBacktest(data.id);
			return;
		}
	}
	system.setName(data.name);
	system.removeBacktest(data.id);
	system.addBacktest(data.id);

	addSystem(system);
}


void User::deleteBacktest(int id)
{
	for (auto& system : m_systems)
	{
		for (auto& backtest : system.backtests())
		{
			if (backtest.id == id)
			{
				system.removeBacktest(id);
			}
		}
	}
}

Backtest& User::getBacktest(int id)
{
	for (auto& system : m_systems)
	{
		for (auto& backtest : system.backtests())
		{
			if (backtest.id == id)
			{
				return backtest;
			}
		}
	}

	Backtest backtest;
	return backtest;
}

void User::addSystem(const System& system)
{
	m_systems.push_back(system);
}

void User::deleteSystem(const std::string& name)
{
	for (size_t i = 0; i < m_systems.size(); ++i)
	{
		if (m_systems[i].name() == name)
		{
			m_systems.erase(m_systems.begin() + i);
		}
	}
}

System User::getSystem(const std::string& name)
{
	for (size_t i = 0; i < m_systems.size(); ++i)
	{
		if (m_systems[i].name() == name)
		{
			return m_systems[i];
		}
	}
	System system;
	return system;
}