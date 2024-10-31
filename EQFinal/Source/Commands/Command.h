#pragma once

#include "pch.h"

#include "User/User.h"
#include "Engine/Backtesting.h"

namespace EQ
{
	struct EngineCommand
	{
	public:

		EngineCommand(const std::string& _source, Ptr<User> _user)
			: source(_source), user(_user), id(0), cancelled(false) {}

		EngineCommand(const std::string& _source, Ptr<User> _user, int _id)
			: source(_source), user(_user), id(_id), cancelled(false) {}


		std::string source;
		Ptr<User> user;
		int id;
		std::atomic<bool> cancelled;
	};
}