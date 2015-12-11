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
	unsigned long long file_length = 0;
	RedundancyEliminator_CPP re;

	//shared data
	bool readFirstTime = true;
	char overlap[WINDOW_SIZE - 1];
	char* buffer;
	FixedSizedCharArray charArrayBuffer;

	unsigned int buffer_len = 0;
	deque<unsigned int> chunking_result;
	//Result
	unsigned long long total_duplication_size = 0;
	//Time
	clock_t start_read, 
			start_chunk, 
			start_fin;
	float tot_read = 0, 
		  tot_chunk = 0, 
		  tot_fin = 0, 
		  tot_time = 0;

public:
	NaiveCpp();
	~NaiveCpp();

	bool ReadFile();
	void Chunking();
	void Fingerprinting();

	int Execute();
	void Test(double &rate, double &time);
};