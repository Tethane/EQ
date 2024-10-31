#pragma once

#include "pch.h"

namespace EQ
{
	// Thread-safe queue for backtest command processing in the multi-threaded application
	template<typename T>
	class Queue
	{
	public:
		void push(const T& value)
		{
			std::lock_guard<std::mutex> lock(m_mutex);
			this->queue.push_back(value);
			m_condition.notify_one();
		}

		T pop()
		{
			std::unique_lock<std::mutex> lock(m_mutex);
			m_condition.wait(lock, [this] {return !this->queue.empty(); });
			T value = this->queue.front();
			this->queue.pop_front();
			return value;
		}

		bool empty() const
		{
			std::lock_guard<std::mutex> lock(m_mutex);
			return this->queue.empty();
		}

		~Queue()
		{
			if constexpr (std::is_pointer<T>::value)
			{
				while (!empty())
				{
					delete pop();
				}
			}
		}

	public:

		std::deque<T> queue;
	private:
		mutable std::mutex m_mutex;

		std::condition_variable m_condition;
	};

	template<typename T>
	using Ptr = std::shared_ptr<T>;
}