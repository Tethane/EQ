#pragma once

// CUDA Libraries

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda.h"
#include "nvrtc.h"

// #include "device_functions.h"

// STL Common

#include <iostream>
#include <fstream>
#include <algorithm>
#include <memory>
#include <chrono>
#include <optional>
#include <functional>
#include <utility>
#include <thread>
#include <mutex>
#include <future>
#include <condition_variable>
#include <stdexcept>
#include <cctype>
#include <sstream>
#include <variant>
#include <regex>

#include <limits>
#include <cassert>
#include <exception>

#include <queue>
#include <deque>
#include <vector>
#include <string>
#include <bitset>
#include <unordered_set>
#include <unordered_map>

#include <random>
#include <cmath>

// <matplotlibcpp.h>
// namespace plt = matplotlibcpp;

#include "External/json.h"
using json = nlohmann::json;

// EQ Common

#include "Core/Debug.h"
#include "Core/DeviceStructures.h"
#include "Core/Structures.h"
#include "Core/DataTypes.h"