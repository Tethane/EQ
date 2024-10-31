#pragma once

#include "pch.h"

namespace EQ
{
    std::vector<std::string> split(const std::string& s, char delimiter);

    void readTxtFile(const std::string& filepath, std::string& source);

    void readCsvFile(const std::string& filepath, std::vector<Security>& securities);

    void extractBlocks(const std::string& text, const std::string& startTag, const std::string& endTag, std::string& output);

    void removeWhitespace(std::string& input);

    enum TokenType {
        IDENTIFIER,
        STRING,
        NUMBER,
        SYMBOL,
        KEYWORD
    };

    struct Token {
        TokenType type;
        std::string value;
    };

    std::vector<Token> tokenize(const std::string& input);

    void printTokens(const std::vector<Token>& tokens);

    using KeyValuePairs = std::unordered_map<std::string, std::string>;

    struct Block {
        std::string name;
        KeyValuePairs keyValues;
    };

    Block parseBlock(const std::vector<Token>& tokens, size_t& i);

    void parseTokens(const std::vector<Token>& tokens, std::vector<Block>& blocks, std::vector<Parameter>& parameters);

    std::string parseList(const std::vector<Token>& tokens, size_t& i);

    Parameter parseParameter(const std::vector<Token>& tokens, size_t& i);

    void printBlock(const Block& block);

    void printParameter(const Parameter& param);

    void addDeviceHostQualifiers(std::string& source);

    void insertParameterDefinitions(std::string& source, const std::vector<Parameter>& parameters);
}
