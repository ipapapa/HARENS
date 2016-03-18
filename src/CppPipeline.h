#pragma once
#include "IO.h"
#include "PcapReader.h"
#include "RabinHash.h"
#include "RedundancyEliminator_CPP.h"

class CppPipeline {
private:
	unsigned long long totalFileLen = 0;
	RedundancyEliminator_CPP re;
	//syncronize
	std::array<std::mutex, PAGABLE_BUFFER_NUM> bufferMutex;						//lock for buffer
	std::array<std::condition_variable, PAGABLE_BUFFER_NUM> bufferCond;
	std::array<std::mutex, RESULT_BUFFER_NUM> chunkingResultMutex;				//lock for chunkingResultBuffer
	std::array<std::condition_variable, RESULT_BUFFER_NUM> chunkingResultCond;
	std::array<bool, PAGABLE_BUFFER_NUM> bufferObsolete;						//states of buffer
	std::array<bool, RESULT_BUFFER_NUM> chunkingResultObsolete;					//states of chunkingResultBuffer
	bool readFileEnd = false;
	bool chunkingEnd = false;
	std::mutex readFileEndMutex, 
			   chunkingEndMutex;
	//shared data
	char overlap[WINDOW_SIZE - 1];
	char** buffer;
	FixedSizedCharArray charArrayBuffer;

	std::array<unsigned int, PAGABLE_BUFFER_NUM> bufferLen;
	std::array<std::deque<unsigned int>, RESULT_BUFFER_NUM> chunkingResultBuffer;
	//Result
	unsigned long long totalDuplicationSize = 0;
	//Time
	clock_t startReading, 
			startChunkPartitioning, 
			startChunkHashingAndMatching;
	float timeReading = 0, 
		  timeChunkPartitioning = 0, 
		  timeChunkHashingAndMatching = 0;

	void ReadFile();
	void Chunking();
	void Fingerprinting();

public:
	CppPipeline();
	~CppPipeline();

	int Execute();
	void Test(double &rate, double &time);
};