#pragma once


enum Level { TRACE = 1, DEBUG, WARNING, CRITICAL };

#ifdef _DEBUG
#define LOG(level, msg) std::cerr << "[" << __FUNCTION__ "() Line " << __LINE__ << "] " << level << ": " << msg << "\n";
#else
#define LOG(level, msg)
#endif

#define cudaCheck(call)	do {																				\
	CUresult err = call;																					\
	if (err != CUDA_SUCCESS)																				\
	{																										\
		const char* errStr;																					\
		cuGetErrorString(err, &errStr);																		\
		std::stringstream errMsg;																			\
		errMsg << "CUDA Error: " << errStr << " in file " << __FILE__ << " at line " << __LINE__ << "\n";	\
		throw std::runtime_error(errMsg.str());																\
	}																										\
} while(0)

#define cudaCheckNvrtc(call) do {																								\
	nvrtcResult err = call;																										\
	if (err != NVRTC_SUCCESS)																									\
	{																															\
		std::stringstream errMsg;																								\
		errMsg << "NVRTC error: " << nvrtcGetErrorString(err) << " in file " << __FILE__ << " at line " << __LINE__ << "\n";	\
		throw std::runtime_error(errMsg.str());																					\
	}																															\
} while(0)