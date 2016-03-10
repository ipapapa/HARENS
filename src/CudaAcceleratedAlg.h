#pragma once
#include "IO.h"
#include "PcapReader.h"
#include "RabinHash.h"
#include "RedundancyEliminator_CUDA.h"

class CudaAcceleratedAlg {
private:
	//file
	bool readFirstTime = true;
	unsigned long long file_length;
	FixedSizedCharArray charArrayBuffer;
	char overlap[WINDOW_SIZE - 1];
	//pagable buffer
	char* pagable_buffer;
	unsigned int pagable_buffer_len;
	//fixed buffer
	char* fixed_buffer;
	unsigned int fixed_buffer_len;
	//RedundancyEliminator_CUDA
	RedundancyEliminator_CUDA re;
	//chunking kernel asynchronize
	char* input_kernel;
	unsigned int* result_kernel;
	unsigned int* result_host;
	unsigned int result_host_len;
	//chunking result processing
	cudaStream_t stream;
	unsigned int* chunking_result;
	unsigned int chunking_result_len;
	//chunk matching 
	LRUStrHash<SHA_DIGEST_LENGTH> hash_pool;
	unsigned long long total_duplication_size = 0;
	//Time
	clock_t start, 
			end, 
			start_r, 
			end_r, 
			start_t, 
			end_t, 
			start_ck, 
			end_ck, 
			start_cp, 
			end_cp, 
			start_ch, 
			end_ch, 
			start_cm, 
			end_cm;
	double time_tot = 0, 
		   time_r = 0, 
		   time_t = 0, 
		   time_ck = 0, 
		   time_cp = 0, 
		   time_ch, 
		   time_cm;

	bool ReadFile();
	void ChunkingKernel();
	void ChunkingResultProc();
	void ChunkMatch();

public:
	CudaAcceleratedAlg();
	~CudaAcceleratedAlg();

	int Execute();
	void Test(double &rate, double &time);
};