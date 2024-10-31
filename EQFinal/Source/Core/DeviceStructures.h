#pragma once

#include "pch.h"

namespace EQ
{
	template<typename T>
	class devec {
	private:
		T* m_dataManaged; // Data stored in unified memory (i.e created on host and used in device code)
		T* m_data; // Data stored in device memory (i.e created on device and used in device code)
		size_t m_size;
		size_t m_capacity;
		bool isManaged; // Is the devec managed or unmanaged

	public:
		__host__ __device__ devec() : m_size(0), m_capacity(0), m_data(nullptr), m_dataManaged(nullptr)
		{
#ifndef __CUDA_ARCH__
			isManaged = true;
#else
			isManaged = false;
#endif
			allocate(m_capacity);
		}

		__host__ __device__ devec(T* rawDataPtr, int n) : m_size(0), m_capacity(0), m_data(nullptr), m_dataManaged(nullptr)
		{
			m_size = n;
			m_capacity = n + 1;

#ifndef __CUDA_ARCH__
			isManaged = true;
			allocate(m_capacity);
			m_dataManaged = rawDataPtr;
#else
			isManaged = false;
			allocate(m_capacity);
			m_data = rawDataPtr;
#endif

		}


		__host__ __device__ ~devec()
		{
			if (m_data)
				cudaFree(m_data);
			if (m_dataManaged)
				cudaFree(m_dataManaged);
		}

		__host__ __device__ T& operator[](size_t idx)
		{
			if (isManaged)
				return m_dataManaged[idx];
			else
				return m_data[idx];
		}

		__host__ __device__ const T& operator[](size_t idx) const
		{
			if (isManaged)
				return m_dataManaged[idx];
			else
				return m_data[idx];
		}

		__host__ __device__ size_t size() const
		{
			return m_size;
		}

		__host__ __device__ size_t capacity() const
		{
			return m_capacity;
		}

		__host__ __device__ T* data() const
		{
			if (isManaged)
				return m_dataManaged;
			else
				return m_data;
		}

		__host__ __device__ void push_back(const T& value)
		{
			if (m_size == m_capacity)
				allocate(m_capacity == 0 ? 1 : m_capacity * 2);

			if (isManaged)
				m_dataManaged[m_size] = value;
			else
				m_data[m_size] = value;

			m_size++;
		}

		__host__ __device__ void pop_back()
		{
			if (m_size > 0)
				--m_size;
		}

		__host__ __device__ void erase(size_t index)
		{
			if (index < m_size)
			{

				if (isManaged)
#ifndef __CUDA_ARCH__
					cudaMemcpy(m_dataManaged + index, m_dataManaged + index + 1, (m_size - index - 1) * sizeof(T), cudaMemcpyDefault);
#endif
				else
					memcpy(m_data + index, m_data + index + 1, (m_size - index - 1) * sizeof(T));

				--m_size;
			}
		}

		__host__ __device__ void resize(size_t newSize)
		{
			if (newSize > m_capacity)
				allocate(newSize);
			m_size = newSize;
		}

		__host__ __device__ void allocate(size_t newCapacity)
		{
			if (newCapacity <= m_capacity)
				return;

			T* newData = nullptr;


			if (isManaged)
			{
#ifndef __CUDA_ARCH__
				cudaError_t err = cudaMallocManaged(&newData, newCapacity * sizeof(T));
				if (err != cudaSuccess)
				{
					printf("Critical Error: Device cudaMallocManaged failed\n");
					return;
				}

				if (m_dataManaged && m_size > 0)
					cudaMemcpy(newData, m_dataManaged, m_size * sizeof(T), cudaMemcpyDefault);

				if (m_dataManaged)
					cudaFree(m_dataManaged);

				m_dataManaged = newData;
#endif
			}
			else
			{
				newData = (T*)malloc(newCapacity * sizeof(T));
				if (!newData)
				{
					printf("Critical Error: Device malloc failed\n");
					return;
				}

				if (m_data && m_size > 0)
					memcpy(newData, m_data, m_size * sizeof(T));

				if (m_data)
					free(m_data);

				m_data = newData;
			}
			m_capacity = newCapacity;
		}
	};
}