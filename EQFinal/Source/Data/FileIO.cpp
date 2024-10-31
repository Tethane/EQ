#include "pch.h"
#include "FileIO.h"

namespace EQ
{
    std::vector<std::string> split(const std::string& s, char delimiter)
    {
        std::vector<std::string> tokens;
        std::string token;
        std::istringstream tokenStream(s);
        while (std::getline(tokenStream, token, delimiter))
        {
            tokens.push_back(token);
        }
        return tokens;
    }

    void readTxtFile(const std::string& filepath, std::string& source)
    {
        std::ifstream file(filepath, std::ios::in | std::ios::binary);
        if (!file.is_open())
        {
            std::cerr << "Failed to open file: " << filepath << std::endl;
            return;
        }
        std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        source = content;
    }

    void readCsvFile(const std::string& filepath, std::vector<Security>& securities)
    {
        std::ifstream file(filepath.c_str());
        if (!file.is_open())
        {
            std::cerr << "Error opening data file: " << filepath << std::endl;
            return;
        }

        std::string line;

        std::getline(file, line);
        std::vector<std::string> headers = split(line, ',');
        std::unordered_map<std::string, Security> stockMap;
        for (size_t i = 1; i < headers.size(); i += 5)
        {
            std::string ticker = headers[i];
            stockMap[ticker] = Security(ticker.c_str());
        }

        std::getline(file, line);
        std::getline(file, line);

        int lineCount = 0;
        while (std::getline(file, line))
        {
            ++lineCount;
            std::vector<std::string> tokens = split(line, ',');
            size_t index = 1;
            for (auto& pair : stockMap)
            {
                try
                {
                    Databar db;
                    db.close = std::stof(tokens[index]);
                    db.high = std::stof(tokens[index + 1]);
                    db.low = std::stof(tokens[index + 2]);
                    db.open = std::stof(tokens[index + 3]);
                    db.volume = std::stof(tokens[index + 4]);

                    pair.second.addDatabar(db);
                }
                catch (const std::exception& e)
                {
                    std::cerr << "Error parsing line: " << lineCount << ": " << e.what() << std::endl;
                    break;
                }

                index += 5;
            }
        }

        file.close();

        for (auto& pair : stockMap)
        {
            securities.push_back(pair.second);
        }
    }

    void extractBlocks(const std::string& text, const std::string& startTag, const std::string& endTag, std::string& output) {
        size_t start = 0;

        while ((start = text.find(startTag, start)) != std::string::npos) {
            start += startTag.length();
            size_t end = text.find(endTag, start);
            if (end == std::string::npos) {
                std::cerr << "Error: No matching closing tag for " << startTag << "\n";
                break;
            }
            output += text.substr(start, end - start) + "\n";
            start = end + endTag.length();
        }
    }

    void removeWhitespace(std::string& input) {
        std::string result;
        result.reserve(input.length()); // Reserve memory to avoid multiple allocations

        for (char ch : input) {
            if (!std::isspace(static_cast<unsigned char>(ch))) {
                result += ch;
            }
        }

        input = result;
    }

    std::vector<Token> tokenize(const std::string& input) {
        std::vector<Token> tokens;
        size_t i = 0;

        while (i < input.length()) {
            if (std::isalpha(input[i]) || input[i] == '_') {  // Identifier or Keyword
                size_t start = i;
                while (i < input.length() && (std::isalnum(input[i]) || input[i] == '_')) i++;
                std::string identifier = input.substr(start, i - start);

                // Check if it's a keyword
                if (identifier == "parameter" || identifier == "optimize") {
                    tokens.push_back({ KEYWORD, identifier });
                }
                else {
                    tokens.push_back({ IDENTIFIER, identifier });
                }
            }
            else if (input[i] == '"') {  // String
                size_t start = ++i;
                while (i < input.length() && input[i] != '"') i++;
                tokens.push_back({ STRING, input.substr(start, i - start) });
                i++;  // Skip the closing quote
            }
            else if (std::isdigit(input[i]) || input[i] == '.' || input[i] == '-') {  // Number (including negative)
                size_t start = i;
                while (i < input.length() && (std::isdigit(input[i]) || input[i] == '.' || input[i] == 'f')) i++;
                tokens.push_back({ NUMBER, input.substr(start, i - start) });
            }
            else if (std::ispunct(input[i])) {  // Symbol
                tokens.push_back({ SYMBOL, std::string(1, input[i]) });
                i++;
            }
            else {
                i++;  // Skip any other characters (though ideally, these should have been removed)
            }
        }

        return tokens;
    }

    void printTokens(const std::vector<Token>& tokens) {
        for (const auto& token : tokens) {
            std::string typeStr;
            switch (token.type) {
            case IDENTIFIER: typeStr = "IDENTIFIER"; break;
            case STRING: typeStr = "STRING"; break;
            case NUMBER: typeStr = "NUMBER"; break;
            case SYMBOL: typeStr = "SYMBOL"; break;
            case KEYWORD: typeStr = "KEYWORD"; break;
            }
            std::cout << typeStr << ": " << token.value << "\n";
        }
    }

    std::string parseList(const std::vector<Token>& tokens, size_t& i) {
        std::string listValue;
        if (tokens[i].value != "[") {
            throw std::runtime_error("Expected '[' to start list");
        }
        listValue += tokens[i].value;
        i++;

        while (i < tokens.size() && tokens[i].value != "]") {
            listValue += tokens[i].value;
            i++;
        }

        if (tokens[i].value != "]") {
            throw std::runtime_error("Expected ']' to close list");
        }
        listValue += tokens[i].value;
        i++; // Move past ']'

        return listValue;
    }

    Block parseBlock(const std::vector<Token>& tokens, size_t& i) {
        Block block;

        if (tokens[i].type != IDENTIFIER) {
            throw std::runtime_error("Expected block name");
        }

        block.name = tokens[i].value;
        i++;  // Move past the block name

        if (tokens[i].value != "(") {
            throw std::runtime_error("Expected '(' after block name");
        }
        i++;  // Move past '('

        while (i < tokens.size() && tokens[i].value != ")") {
            if (tokens[i].type != IDENTIFIER) {
                throw std::runtime_error("Expected identifier in key-value pair");
            }

            std::string key = tokens[i].value;
            i++;  // Move past the key

            if (tokens[i].value != "=") {
                throw std::runtime_error("Expected '=' in key-value pair");
            }
            i++;  // Move past '='

            std::string value;

            if (tokens[i].type == STRING || tokens[i].type == NUMBER || tokens[i].type == IDENTIFIER) {
                value = tokens[i].value;
                i++;  // Move past the value
            }
            else if (tokens[i].value == "[") {
                value = parseList(tokens, i);  // Handle list parsing
            }
            else {
                throw std::runtime_error("Expected value in key-value pair");
            }

            block.keyValues[key] = value;

            if (tokens[i].value == ",") {
                i++;  // Move past ',' to the next key-value pair
            }
        }

        if (tokens[i].value != ")") {
            throw std::runtime_error("Expected ')' at the end of block");
        }
        i++;  // Move past ')'

        if (tokens[i].value != ";") {
            throw std::runtime_error("Expected ';' after block");
        }
        i++;  // Move past ';'

        return block;
    }

    void parseTokens(const std::vector<Token>& tokens, std::vector<Block>& blocks, std::vector<Parameter>& parameters) {
        size_t i = 0;

        while (i < tokens.size()) {
            if (tokens[i].type == KEYWORD && tokens[i].value == "parameter")
            {
                Parameter param = parseParameter(tokens, i);
                // printParameter(param);
                parameters.push_back(param);
            }
            else
            {
                Block block = parseBlock(tokens, i);
                // printBlock(block);
                blocks.push_back(block);
            }
        }
    }

    Parameter parseParameter(const std::vector<Token>& tokens, size_t& i) {
        Parameter param;

        if (tokens[i].type != KEYWORD || tokens[i].value != "parameter") {
            throw std::runtime_error("Expected 'parameter' keyword");
        }
        i++;  // Move past 'parameter'

        if (tokens[i].value != "<") {
            throw std::runtime_error("Expected '<' after 'parameter'");
        }
        i++;  // Move past '<'

        if (tokens[i].type != IDENTIFIER) {
            throw std::runtime_error("Expected type identifier after '<'");
        }
        param.type = tokens[i].value;
        i++;  // Move past type

        if (tokens[i].value != ">") {
            throw std::runtime_error("Expected '>' after type");
        }
        i++;  // Move past '>'

        if (tokens[i].value != "(") {
            throw std::runtime_error("Expected '(' after '>'");
        }
        i++;  // Move past '('

        if (tokens[i].type != STRING) {
            throw std::runtime_error("Expected string (parameter name) after '('");
        }
        param.name = tokens[i].value;
        i++;  // Move past parameter name

        if (tokens[i].value != ")") {
            throw std::runtime_error("Expected ')' after parameter name");
        }
        i++;  // Move past ')'

        if (tokens[i].value != "=") {
            throw std::runtime_error("Expected '=' after parameter definition");
        }
        i++;  // Move past '='

        if (tokens[i].type == KEYWORD && tokens[i].value == "optimize") {
            i++;  // Move past 'optimize'

            if (tokens[i].value != "<") {
                throw std::runtime_error("Expected '<' after 'optimize'");
            }
            i++;  // Move past '<'

            if (tokens[i].value == "[") {
                i++;  // Move past '['
                std::vector<std::string> values;
                while (i < tokens.size() && tokens[i].value != "]") {
                    if (tokens[i].type == NUMBER || tokens[i].type == IDENTIFIER) {
                        values.push_back(tokens[i].value);
                    }
                    else if (tokens[i].value != ",") {
                        throw std::runtime_error("Expected ',' or ']' in list of optimize values");
                    }
                    i++;
                }
                if (tokens[i].value != "]") {
                    throw std::runtime_error("Expected ']' to close list");
                }
                i++;  // Move past ']'
                i++;
                param.value = values;
            }
            else {
                std::string minValue = tokens[i].value;
                i++;  // Move past min value

                if (tokens[i].value != ",") {
                    throw std::runtime_error("Expected ',' between min and max values");
                }
                i++;  // Move past ','

                std::string maxValue = tokens[i].value;
                i++;  // Move past max value

                if (tokens[i].value != ">") {
                    throw std::runtime_error("Expected '>' after max value");
                }
                i++;  // Move past '>'
                param.value = std::make_pair(minValue, maxValue);
            }
        }
        else if (tokens[i].type == NUMBER || tokens[i].type == IDENTIFIER) {
            // Simple value assignment (e.g., parameter<int>("num")=5)
            param.value = tokens[i].value;
            i++;  // Move past value
        }
        else {
            throw std::runtime_error("Expected value or 'optimize' after '=' in parameter definition");
        }

        if (tokens[i].value != ";") {
            throw std::runtime_error("Expected ';' at the end of parameter definition");
        }
        i++;  // Move past ';'

        return param;
    }

    void printBlock(const Block& block)
    {
        std::cout << "Block: " << block.name << "\n";
        for (auto& kv : block.keyValues)
        {
            std::cout << "  " << kv.first << ": " << kv.second << "\n";
        }
    }

    void printParameter(const Parameter& param)
    {
        std::cout << "Parameter: " << param.name << "\n";
        std::cout << "  Type: " << param.type << "\n";
        std::visit([&](auto&& value) {
            using T = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<T, std::string>)
            {
                std::cout << "  Value: " << value << "\n";
            }
            else if constexpr (std::is_same_v<T, std::pair<std::string, std::string>>)
            {
                std::cout << "  Min Value: " << value.first << "\n";
                std::cout << "  Max Value: " << value.second << "\n";
            }
            else if constexpr (std::is_same_v<T, std::vector<std::string>>)
            {
                std::cout << "  Values: ";
                for (const auto& val : value)
                {
                    std::cout << val << " ";
                }
                std::cout << "\n";
            }
            }, param.value);
    }

    void addDeviceHostQualifiers(std::string& source)
    {
        std::regex functionRegex(R"(\b([a-zA-Z_]\w*)\s+([a-zA-Z_]\w*)\s*\([^)]*\)\s*\{)");

        std::string modifiedCode = std::regex_replace(source, functionRegex, "__device__ $&");

        source = modifiedCode;
    }

    void insertParameterDefinitions(std::string& source, const std::vector<Parameter>& parameters)
    {
        std::string insertion;

        for (size_t i = 0; i < parameters.size(); ++i)
        {
            insertion += "    int " + parameters[i].name + " = parameters[" + std::to_string(i) + "];";
        }

        size_t position = source.find("{");
        if (position != std::string::npos)
        {
            ++position;
            source.insert(position, "\n" + insertion);
        }
    }
}
