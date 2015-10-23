#pragma once
#include "IO.h"
#include "RabinHash.h"
#include "RedundancyEliminator_CPP.h"
#include "PcapReader.h"

/*
This class is the module of naive cpp implementation
*/
class NaiveCpp {
private:
	unsigned int file_length = 0;
	RedundancyEliminator_CPP re;
	ifstream ifs;
	PcapReader fileReader;
	unsigned int cur_file_pos = 0;

	//shared data
	bool readFirstTime = true;
	char overlap[WINDOW_SIZE - 1];
	char* buffer;
	FixedSizedCharArray charArrayBuffer;

	unsigned int buffer_len = 0;
	deque<unsigned int> chunking_result;
	//Result
	unsigned int total_duplication_size = 0;
	//Time
	clock_t start_read, start_chunk, start_fin;
	float tot_read = 0, tot_chunk = 0, tot_fin = 0, tot_time = 0;

public:
	NaiveCpp();
	~NaiveCpp();

	bool ReadFile();
	void Chunking();
	void Fingerprinting();

	int Execute();
};