#ifndef HARENS_RE_H
#define HARENS_RE_H
#include "CircularQueuePool.h"
#include "IO.h"
#include "PcapReader.h"
#include "RabinHash.h"
#include "RedundancyEliminator_CUDA.h"

/**
* \brief The algorithm of "Hardware Accelerated Redundancy Elimination in Network System"
* This is the kernel of re-module in project DRESS: 
* Distributed Redundancy Elimination System Simulation.
*/
class HarensRE {
private:
	// key functions
	RedundancyEliminator_CUDA re;
	// map-reduce configurations
	int mapperNum;
	int reducerNum;	
	// termination
	bool terminateSig = false;
	std::mutex terminateSigMutex;
	// threads
	std::thread tReadData;
	std::thread tChunkingKernel;
	std::thread tChunkingResultProc;
	std::thread tChunkHashing;
	std::thread *tChunkMatch;

	// request queue: sync between HandleGetRequest and ReadData
	std::queue<std::string> requestQueue;
	std::mutex requestQueueMutex;
	std::condition_variable newRequestCond;
	
	// result queue: sync between HandleGetRequest and ChunkHashing
	// consists of result, result length as uint8_t std::array, std::mutex, condition variable
	std::queue< std::tuple<std::vector< std::tuple<int, unsigned char*, int, char*> >*,
						   int*,
						   std::mutex*,
						   std::condition_variable*> > resultQueue;
	std::mutex resultQueueMutex;

	// package queue: sync between ReadData and ChunkingKernel
	// consists of pagable buffer indices for package data
	// -1 is used as deliminator between packages
	std::queue<int> packageQueue;
	std::mutex packageQueueMutex;
	std::condition_variable newPackageCond;

	// rabin result queue: sync between ChunkingKernel and ChunkingResultProc
	// consists of result-host indices for rabin hash
	// -1 is used as deliminator between requests
	std::queue<int> rabinQueue;
	std::mutex rabinQueueMutex;
	std::condition_variable newRabinCond;

	// chunk queue: sync between ChunkingResultProc and ChunkHashing
	// consists of chunking result buffer indices
	// -1 is used as deliminator between requests
	std::queue<int> chunkQueue;
	std::mutex chunkQueueMutex;
	std::condition_variable newChunksCond;

	// hash queue: sync between ChunkHashing and ChunkMatch
	// consists of response and condition variable
	std::queue< std::tuple<std::vector< std::tuple<int, unsigned char*, int, char*> >*,
						   int*,
						   std::mutex*,
						   std::condition_variable*> > hashQueue;
	std::mutex hashQueueMutex;
	std::condition_variable newHashCond;

	// pagable buffer
	std::array<char*, PAGABLE_BUFFER_NUM> pagableBuffer;
	std::array<unsigned int, PAGABLE_BUFFER_NUM> pagableBufferLen;
	std::array<std::mutex, PAGABLE_BUFFER_NUM> pagableBufferMutex;
	std::array<std::condition_variable, PAGABLE_BUFFER_NUM> pagableBufferCond;
	std::array<bool, PAGABLE_BUFFER_NUM> pagableBufferObsolete;

	// fixed buffer
	std::array<char*, FIXED_BUFFER_NUM> fixedBuffer;
	std::array<unsigned int, FIXED_BUFFER_NUM> fixedBufferLen;

	// chunking kernel asynchronize
	std::array<char*, FIXED_BUFFER_NUM> kernelInputBuffer;
	std::array<unsigned int*, FIXED_BUFFER_NUM> kernelResultBuffer;
	std::array<unsigned int*, FIXED_BUFFER_NUM> hostResultBuffer;
	std::array<unsigned int, FIXED_BUFFER_NUM> hostResultLen;
	std::array<std::mutex, FIXED_BUFFER_NUM> hostResultMutex;
	std::array<std::condition_variable, FIXED_BUFFER_NUM> hostResultCond;
	std::array<bool, FIXED_BUFFER_NUM> hostResultObsolete;
	std::array<bool, FIXED_BUFFER_NUM> hostResultExecuting;

	// chunking result processing
	std::array<cudaStream_t, RESULT_BUFFER_NUM> stream;
	std::array<unsigned int*, RESULT_BUFFER_NUM> chunkingResultBuffer;
	std::array<unsigned int, RESULT_BUFFER_NUM> chunkingResultLen;
	std::array<std::mutex, RESULT_BUFFER_NUM> chunkingResultMutex;
	std::array<std::condition_variable, RESULT_BUFFER_NUM> chunkingResultCond;
	std::array<bool, RESULT_BUFFER_NUM> chunkingResultObsolete;

	// chunk hashing
	std::thread *segmentThreads;

	// chunk matching 
	LRUStrHash<SHA1_HASH_LENGTH> *circHashPool;
	std::mutex *circHashPoolMutex;
	unsigned long long *duplicationSize;
	unsigned long long totalDuplicationSize = 0;
	unsigned long long totalFileLen = 0;
	
	// time
	clock_t startReading, 
			endReading, 
			startChunkingKernel, 
			endChunkingKernel, 
			startChunkPartitioning, 
			endChunkPartitioning, 
			startChunkHashing, 
			endChunkHashing, 
			startChunkMatching, 
			endChunkMatching;
	double timeReading = 0, 
		   timeChunkingKernel = 0, 
		   timeChunkPartitioning = 0,
		   timeChunkHashing = 0, 
		   timeChunkMatching = 0;

	/** 
	* \brief read data from file system based on given filenames.
	*/
	void ReadData();

	/**
	* \brief call the GPU kernel function to compute the Rabin hash value
	* of each sliding window
	*/
	void ChunkingKernel();
	
	/**
	* \brief mark the beginning of a window as a fingerprint based on the MODP rule.
	* The fingerprints divide the stream into chunks.
	*/
	void ChunkingResultProc();

	/**
	* \brief compute a non-collision hash (SHA-1) value for each chunk
	* Chunks are divided into segments and processed by function ChunkSegmentHashing.
	*/
	void ChunkHashing();

	/**
	* \brief match the chunks by their hash values
	*/
	void ChunkMatch();

	/**
	* \brief compute a non-collision hash (SHA-1) value for each chunk in the segment
	* \param pagableBufferIdx the index of pagable buffer that stores the chunks
	* \param chunkingResultIdx the index of the buffer that stores the chunking result
	* \param segmentNum the segment of chunking result that this std::thread is going to process
	* \param result a vector that stores the results, which are hash-value pairs
	* \param resultLenInUint8 the length of all the results if stored in a uint8_t std::array
	* \param resultMutex std::mutex of the result, used to prevent data race
	*/
	void ChunkSegmentHashing(int pagableBufferIdx, 
							 int chunkingResultIdx, 
							 int segmentNum,
							 std::vector< std::tuple<int, unsigned char*, int, char*> >* result,
							 int* resultLenInUint8,
							 std::mutex* resultMutex);

public:
	HarensRE(int mapperNum, int reducerNum);
	~HarensRE();

	/**
	* \brief fetching data for the GET request and do redundancy elimination process.
	* Simulating fetching data from server by reading files.
	* \param request the GET request (a file name stored data in server's file system)
	* \param result the hash-chunk pairs of the data. 
	* The two integers before hash value and data chunk are their lengths.
	* \param resultLenInUint8 the length of result if put it in a uint8_t std::array
	* MIND: result is a pointer, caller of this function should be responsible
	* to release the memory!
	*/
	void HandleGetRequest(std::string request,
						  std::vector< std::tuple<int, unsigned char*, int, char*> >* result,
						  int* resultLenInUint8);

	/**
	* \brief start the core of redundancy elimination module
	*/
	void Start();

	/**
	* \brief terminate the core of redundancy elimination module
	*/
	void Stop();
};

#endif /* HARENS_RE_H */