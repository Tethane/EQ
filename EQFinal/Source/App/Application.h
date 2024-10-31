#pragma once

#include "pch.h"
#include "Engine/Engine.h"
#include "Commands/Command.h"

namespace EQ
{
	// Singleton designed for use in multi-threaded application (when integrating with the C++/CLI and C# WPF application in the future)
	class Application
	{
	public:
		Application(const Application&) = delete;
		Application& operator=(const Application&) = delete;

		static Ptr<Application> instance();
		void init() const;
		void destroy() const;

		Ptr<User> activeUser() const;

		void activateEngine();
		void deactivateEngine();

		Ptr<Engine> getEngine();

		void addEngineBacktestCommand(Ptr<EngineCommand>);
		void removeEngineBacktestCommand(int);

		~Application()
		{

		}
	private:
		Application()
		{
			m_engine = std::make_shared<Engine>();
			m_activeUser = std::make_shared<User>();
		}

	private:

		Ptr<Engine> m_engine;
		Ptr<User> m_activeUser;

		static std::shared_ptr<Application> s_instance;
		static std::mutex s_mutex;
	};
}