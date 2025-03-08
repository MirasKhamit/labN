#include <iostream>
#include <fstream>
#include <string>
#include <regex>

using namespace std;

void extractMetrics(const string& filename) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error opening file!" << endl;
        return;
    }

    string line;
    regex executionTimeRegex(R"(Execution time of Naive GEMV with array_size=\d+ and block_size=\d+: (\d+\.\d+))");
    regex dramThroughputRegex(R"(DRAM Throughput\s+%.*\s+(\d+\.\d+))");
    regex memoryThroughputRegex(R"(Memory Throughput\s+Gbyte/s\s+(\d+\.\d+))");
    regex l1TEXHitRateRegex(R"(L1/TEX Hit Rate\s+%.*\s+(\d+\.\d+))");
    regex achievedOccupancyRegex(R"(Achieved Occupancy\s+%.*\s+(\d+\.\d+))");
    regex smBusyRegex(R"(SM Busy\s+%.*\s+(\d+\.\d+))");
    regex theoreticalOccupancyRegex(R"(Theoretical Occupancy\s+%.*\s+(\d+\.\d+))");
    regex sharedMemoryConfigSizeRegex(R"(Shared Memory Configuration Size\s+Kbyte\s+(\d+\.\d+))");
    regex registersPerThreadRegex(R"(Registers Per Thread\s+register/thread\s+(\d+))");

    string executionTime;
    string dramThroughput;
    string memoryThroughput;
    string l1TEXHitRate;
    string achievedOccupancy;
    string smBusy;
    string theoreticalOccupancy;
    string sharedMemoryConfigSize;
    string registersPerThread;

    while (getline(file, line)) {
        smatch match;
        if (regex_search(line, match, executionTimeRegex)) {
            executionTime = match[1].str();
        }
        if (regex_search(line, match, dramThroughputRegex)) {
            dramThroughput = match[1].str();
        }
        if (regex_search(line, match, memoryThroughputRegex)) {
            memoryThroughput = match[1].str();
        }
        if (regex_search(line, match, l1TEXHitRateRegex)) {
            l1TEXHitRate = match[1].str();
        }
        if (regex_search(line, match, achievedOccupancyRegex)) {
            achievedOccupancy = match[1].str();
        }
        if (regex_search(line, match, smBusyRegex)) {
            smBusy = match[1].str();
        }
        if (regex_search(line, match, theoreticalOccupancyRegex)) {
            theoreticalOccupancy = match[1].str();
        }
        if (regex_search(line, match, sharedMemoryConfigSizeRegex)) {
            sharedMemoryConfigSize = match[1].str();
        }
        if (regex_search(line, match, registersPerThreadRegex)) {
            registersPerThread = match[1].str();
        }
    }

    file.close();

    // Print extracted values
    cout << "Execution Time (ms): " << executionTime << endl;
    cout << "DRAM Throughput (%): " << dramThroughput  << endl;
    cout << "Memory Throughput (GB/s): " << memoryThroughput  << endl;
    cout << "L1/TEX Hit Rate (%): " << l1TEXHitRate << endl;
    cout << "Achieved Occupancy (%): " << achievedOccupancy << endl;
    cout << "SM Busy (%): " << smBusy  << endl;
    cout << "Theoretical Occupancy (%): " << theoreticalOccupancy  << endl;
    cout << "Shared Memory Configuration Size (Kbyte): " << sharedMemoryConfigSize << endl;
    cout << "Registers Per Thread: " << registersPerThread << endl;
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        cerr << "Usage: " << argv[0] << " <filename>" << endl;
        return 1;
    }

    string filename = argv[1];
    extractMetrics(filename);

    return 0;
}