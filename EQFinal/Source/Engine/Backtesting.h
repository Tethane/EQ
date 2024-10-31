#pragma once

#include "pch.h"

struct ExecutableParameter;
struct AggregatePerformanceData;
struct Backtest;

void to_json(json& j, const ExecutableParameter& param);

void from_json(const json& j, ExecutableParameter& param);

void to_json(json& j, const AggregatePerformanceData& data);

void from_json(const json& j, AggregatePerformanceData& data);

void to_json(json& j, const Backtest& bt);

void from_json(const json& j, Backtest& bt);

struct ExecutableParameter
{
	std::string type;
	std::string name;
	std::variant<
		int,
		std::pair<int, int>,
		std::vector<int>
	> value;
};

struct AggregatePerformanceData
{
	__host__ __device__ AggregatePerformanceData() {}

	float startEquity = 0.0f;
	float endEquity = 0.0f;

	float totalPNL = 0.0f;
	float maxDrawdown = 0.0f;
	float sharpeRatio = 0.0f;
	float sortinoRatio = 0.0f;

	float successRate = 0.0f;
	int totalTrades = 0;
	int winners = 0;
	int losers = 0;
	float bestTrade = 0.0f;
	float worstTrade = 0.0f;

	int stoplossesHit = 0;
	int takeprofitsHit = 0;
};

struct Backtest
{
public:
	std::string ticker;
	std::string startDate;
	std::string endDate;
	std::string interval;
	int batch;
	int engine;
	int folds;
	float capital;
	int slippage;
	float commission;
	std::vector<ExecutableParameter> parameterDefinitions;
	std::vector<AggregatePerformanceData> resultsByParameterSet;
	std::vector<int> optimalParameterSet;
	int id;
	std::string name;

public:
	std::string serialize() const {
		json j = *this;
		return j.dump();
	}

	// Deserialize from JSON string
	static Backtest deserialize(const std::string& jsonString) {
		json j = json::parse(jsonString);
		return j.get<Backtest>();
	}
};

