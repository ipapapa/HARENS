#pragma once
#include "IO.h"
#include "PcapReader.h"
#include "RabinHash.h"
#include "RedundancyEliminator_CPP.h"

/*
This class is the module of naive cpp implementation
*/
class NaiveCpp {
private:
	int count = 0;
	unsigned long long totalFileLen = 0;
	RedundancyEliminator_CPP re;

	//shared data
	bool readFirstTime = true;
	char overlap[WINDOW_SIZE - 1];
	char* buffer;
	FixedSizedCharArray charArrayBuffer;

	unsigned int bufferLen = 0;
	deque<unsigned int> chunkingResultBuffer;
	//Result
	unsigned long long totalDuplicationSize = 0;
	//Time
	clock_t startReading, 
			startChunkPartitioning, 
			startChunkHashingAndMatching;
	float timeReading = 0, 
		  timeChunkPartitioning = 0, 
		  timeChunkHashingAndMatching = 0, 
		  timeTotal = 0;

	bool ReadFile();
	void Chunking();
	void Fingerprinting();

public:
	NaiveCpp();
	~NaiveCpp();

	int Execute();
	void Test(double &rate, double &time);
};