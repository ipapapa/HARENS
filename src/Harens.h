#pragma once
#include "CircularQueuePool.h"
#include "IO.h"
#include "PcapReader.h"
#include "RabinHash.h"
#include "RedundancyEliminator_CUDA.h"

/*
* The algorithm of "Hardware Accelerated Redundancy Elimination in Network System"
*/
class Harens {
private:
	//map-reduce configurations
	int mapperNum;
	int reducerNum;	//Better be power of 2, it would make the module operation faster
	//determin if one thread is end
	bool readFileEnd = false;
	/*bool transfer_end = false;*/
	bool chunkingKernelEnd = false;
	bool chunkingResultProcEnd = false;
	bool chunkHashingEnd = false;
	mutex readFileEndMutex, 
		  chunkingKernelEndMutex, 
		  chunkingResultProcEndMutex, 
		  chunkHashingEndMutex;
	//file
	unsigned long long totalFileLen;
	FixedSizedCharArray charArrayBuffer;
	char overlap[WINDOW_SIZE - 1];
	//pagable buffer
	array<char*, PAGABLE_BUFFER_NUM> pagableBuffer;
	array<unsigned int, PAGABLE_BUFFER_NUM> pagableBufferLen;
	array<mutex, PAGABLE_BUFFER_NUM> pagableBufferMutex;
	array<condition_variable, PAGABLE_BUFFER_NUM> pagableBufferCond;
	array<bool, PAGABLE_BUFFER_NUM> pagableBufferObsolete;
	//fixed buffer
	array<char*, FIXED_BUFFER_NUM> fixedBuffer;
	array<unsigned int, FIXED_BUFFER_NUM> fixedBufferLen;
	/*array<mutex, FIXED_BUFFER_NUM> fixed_buffer_mutex;
	array<condition_variable, FIXED_BUFFER_NUM> fixed_buffer_cond;
	array<bool, FIXED_BUFFER_NUM> fixed_buffer_obsolete;*/
	//RedundancyEliminator_CUDA
	RedundancyEliminator_CUDA re;
	//chunking kernel asynchronize
	array<char*, FIXED_BUFFER_NUM> kernelInputBuffer;
	array<unsigned int*, FIXED_BUFFER_NUM> kernelResultBuffer;
	array<unsigned int*, FIXED_BUFFER_NUM> hostResultBuffer;
	array<unsigned int, FIXED_BUFFER_NUM> hostResultLen;
	array<mutex, FIXED_BUFFER_NUM> hostResultMutex;
	array<condition_variable, FIXED_BUFFER_NUM> hostResultCond;
	array<bool, FIXED_BUFFER_NUM> hostResultObsolete;
	array<bool, FIXED_BUFFER_NUM> hostResultExecuting;
	//chunking result processing
	array<cudaStream_t, RESULT_BUFFER_NUM> stream;
	array<unsigned int*, RESULT_BUFFER_NUM> chunkingResultBuffer;
	array<unsigned int, RESULT_BUFFER_NUM> chunkingResultLen;
	array<mutex, RESULT_BUFFER_NUM> chunkingResultMutex;
	array<condition_variable, RESULT_BUFFER_NUM> chunkingResultCond;
	array<bool, RESULT_BUFFER_NUM> chunkingResultObsolete;
	//chunk hashing
	thread *segmentThreads;
	CircularQueuePool chunkHashQueuePool;
	//chunk matching 
	thread *chunkMatchingThreads;
	LRUStrHash<SHA1_HASH_LENGTH> *circHashPool;
	unsigned long long *duplicationSize;
	unsigned long long totalDuplicationSize = 0;
	//Time
	clock_t start, 
			end, 
			startReading, 
			endReading, 
			startChunkingKernel, 
			endChunkingKernel, 
			startChunkPartitioning, 
			endChunkPartitioning, 
			startChunkHashing, 
			endChunkHashing, 
			startChunkMatching, 
			endChunkMatching;
	double timeTotal = 0, 
		   timeReading = 0, 
		   timeChunkingKernel = 0, 
		   timeChunkPartitioning = 0,
		   timeChunkHashing, 
		   timeChunkMatching;
	int count = 0;

	/*
	* Read data from a plain text or pcap file into memory.
	* Transfer is done in this step now.
	*/
	void ReadFile();

	/*
	* Transfer data from pagable buffer to pinned buffer.
	* Because memory transfer between GPU device memory and 
	* pinned buffer is way faster than between pagable buffer.
	*/
	//void Transfer();

	/*
	* Call the GPU kernel function to compute the Rabin hash value
	* of each sliding window
	*/
	void ChunkingKernel();
	
	/*
	* Mark the beginning of a window as a fingerprint based on the MODP rule.
	* The fingerprints divide the stream into chunks.
	*/
	void ChunkingResultProc();

	/*
	* Compute a non-collision hash (SHA-1) value for each chunk
	* Chunks are divided into segments and processed by function ChunkSegmentHashing.
	*/
	void ChunkHashing();

	/*
	* Match the chunks by their hash values
	*/
	void ChunkMatch(int hashPoolIdx);

	/*
	* Compute a non-collision hash (SHA-1) value for each chunk in the segment
	*/
	void ChunkSegmentHashing(int pagableBufferIdx, 
							 int chunkingResultIdx, 
							 int segmentNum);

public:
	Harens(int mapperNum, int reducerNum);
	~Harens();

	int Execute();
	void Test(double &rate, double &time);
};