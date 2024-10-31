#include "pch.h"
#include "Backtesting.h"
using namespace EQ;

// Serialization and Deserialization Methods for Passing Data Through Callbacks

void to_json(json& j, const ExecutableParameter& param) {
	j = json{
		{"type", param.type},
		{"name", param.name}
	};

	std::visit([&](auto&& arg) {
		j["value"] = arg;
		}, param.value);
}

void from_json(const json& j, ExecutableParameter& param) {
	j.at("type").get_to(param.type);
	j.at("name").get_to(param.name);

	if (j["value"].is_number_integer()) {
		param.value = j["value"].get<int>();
	}
	else if (j["value"].is_array()) {
		if (j["value"].size() == 2) {
			param.value = std::make_pair(
				j["value"][0].get<int>(),
				j["value"][1].get<int>()
			);
		}
		else {
			param.value = j["value"].get<std::vector<int>>();
		}
	}
}

void to_json(json& j, const AggregatePerformanceData& data) {
	j = json{
		{"startEquity", data.startEquity},
		{"endEquity", data.endEquity},
		{"totalPNL", data.totalPNL},
		{"maxDrawdown", data.maxDrawdown},
		{"sharpeRatio", data.sharpeRatio},
		{"sortinoRatio", data.sortinoRatio},
		{"successRate", data.successRate},
		{"totalTrades", data.totalTrades},
		{"winners", data.winners},
		{"losers", data.losers},
		{"bestTrade", data.bestTrade},
		{"worstTrade", data.worstTrade},
		{"stoplossesHit", data.stoplossesHit},
		{"takeprofitsHit", data.takeprofitsHit}
	};
}

void from_json(const json& j, AggregatePerformanceData& data) {
	j.at("startEquity").get_to(data.startEquity);
	j.at("endEquity").get_to(data.endEquity);
	j.at("totalPNL").get_to(data.totalPNL);
	j.at("maxDrawdown").get_to(data.maxDrawdown);
	j.at("sharpeRatio").get_to(data.sharpeRatio);
	j.at("sortinoRatio").get_to(data.sortinoRatio);
	j.at("successRate").get_to(data.successRate);
	j.at("totalTrades").get_to(data.totalTrades);
	j.at("winners").get_to(data.winners);
	j.at("losers").get_to(data.losers);
	j.at("bestTrade").get_to(data.bestTrade);
	j.at("worstTrade").get_to(data.worstTrade);
	j.at("stoplossesHit").get_to(data.stoplossesHit);
	j.at("takeprofitsHit").get_to(data.takeprofitsHit);
}

void to_json(json& j, const Backtest& bt) {
	j = json{
		{"ticker", bt.ticker},
		{"startDate", bt.startDate},
		{"endDate", bt.endDate},
		{"interval", bt.interval},
		{"batch", bt.batch},
		{"engine", bt.engine},
		{"folds", bt.folds},
		{"capital", bt.capital},
		{"slippage", bt.slippage},
		{"commission", bt.commission},
		{"parameterDefinitions", bt.parameterDefinitions},
		{"resultsByParameterSet", bt.resultsByParameterSet},
		{"optimalParameterSet", bt.optimalParameterSet},
		{"id", bt.id},
		{"name", bt.name}
	};
}

void from_json(const json& j, Backtest& bt) {
	j.at("ticker").get_to(bt.ticker);
	j.at("startDate").get_to(bt.startDate);
	j.at("endDate").get_to(bt.endDate);
	j.at("interval").get_to(bt.interval);
	j.at("batch").get_to(bt.batch);
	j.at("engine").get_to(bt.engine);
	j.at("folds").get_to(bt.folds);
	j.at("capital").get_to(bt.capital);
	j.at("slippage").get_to(bt.slippage);
	j.at("commission").get_to(bt.commission);
	j.at("parameterDefinitions").get_to(bt.parameterDefinitions);
	j.at("resultsByParameterSet").get_to(bt.resultsByParameterSet);
	j.at("optimalParameterSet").get_to(bt.optimalParameterSet);
	j.at("id").get_to(bt.id);
	j.at("name").get_to(bt.name);
}