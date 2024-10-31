#include "pch.h"
#include "Application.h"

using namespace EQ;

Ptr<Application> Application::s_instance = nullptr;
std::mutex Application::s_mutex;

Ptr<Application> Application::instance()
{
	std::lock_guard<std::mutex> lock(s_mutex);
	if (!s_instance)
	{
		s_instance = Ptr<Application>(new Application(), [](Application* ptr) {
			delete ptr;
			});
	}
	return s_instance;
}

void Application::destroy() const
{
	std::lock_guard<std::mutex> lock(s_mutex);

	s_instance.reset();
}

void Application::init() const
{
	// Doesn't do anything except initialize by calling
}

Ptr<User> Application::activeUser() const
{
	return m_activeUser;
}

void Application::addEngineBacktestCommand(Ptr<EngineCommand> engineCommand)
{
	std::lock_guard<std::mutex> lock(s_mutex);
	m_engine->addCommand(engineCommand);
}

void Application::removeEngineBacktestCommand(int engineCommandId)
{
	std::lock_guard<std::mutex> lock(s_mutex);
	m_engine->removeCommand(engineCommandId);
}

void Application::activateEngine()
{
	m_engine->activate();
}

void Application::deactivateEngine()
{
	m_engine->deactivate();
}

Ptr<Engine> Application::getEngine()
{
	return m_engine;
}