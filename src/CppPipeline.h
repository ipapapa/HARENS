#pragma once
#include "IO.h"
#include "RabinHash.h"
#include "RedundancyEliminator_CPP.h"
#include "PcapReader.h"

class CppPipeline {
private:
	unsigned int file_length = 0;
	RedundancyEliminator_CPP re;
	ifstream ifs;
	//syncronize
	array<mutex, PAGABLE_BUFFER_NUM> buffer_mutex;						//lock for buffer
	array<condition_variable, PAGABLE_BUFFER_NUM> buffer_cond;
	array<mutex, RESULT_BUFFER_NUM> chunking_result_mutex;				//lock for chunking_result
	array<condition_variable, RESULT_BUFFER_NUM> chunking_result_cond;
	array<bool, PAGABLE_BUFFER_NUM> buffer_obsolete;					//states of buffer
	array<bool, RESULT_BUFFER_NUM> chunking_result_obsolete;			//states of chunking_result
	bool read_file_end = false;
	bool chunking_end = false;
	mutex read_file_end_mutex, chunking_end_mutex;
	//shared data
	char overlap[WINDOW_SIZE - 1];
	char** buffer;
	FixedSizedCharArray charArrayBuffer;

	array<unsigned int, PAGABLE_BUFFER_NUM> buffer_len;
	array<deque<unsigned int>, RESULT_BUFFER_NUM> chunking_result;
	//Result
	unsigned int total_duplication_size = 0;
	//Time
	clock_t start_read, start_chunk, start_fin;
	float tot_read = 0, tot_chunk = 0, tot_fin = 0;

public:
	CppPipeline();
	~CppPipeline();

	void ReadFile();
	void Chunking();
	void Fingerprinting();

	int Execute();
	void Test(double &rate, double &time);
};