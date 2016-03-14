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
	mutex terminateSigMutex;
	// threads
	thread tReadData;
	thread tChunkingKernel;
	thread tChunkingResultProc;
	thread tChunkHashing;
	thread *tChunkMatch;
	// request queue: sync between HandleGetRequest and ReadData
	// consists of request, response, mutex, condition variable
	std::queue< std::tuple<std::string&, 
						   std::vector< std::tuple<int, unsigned char*, int, char*> >*&,
						   int&,
						   mutex&,
						   condition_variable&> > requestQueue;
	mutex requestQueueMutex;
	condition_variable newRequestCond;
	// package queue: sync between ReadData and ChunkingKernel
	// consists of pagable buffer indices for package data, response, mutex, condition variable
	std::queue< std::tuple<std::vector<int>,
						   std::vector< std::tuple<int, unsigned char*, int, char*> >*&,
						   int&,
						   mutex&,
						   condition_variable&> > packageQueue;
	mutex packageQueueMutex;
	condition_variable newPackageCond;
	// rabin result queue: sync between ChunkingKernel and ChunkingResultProc
	// consists of result-host indices for rabin hash, response, mutex, condition variable
	std::queue< std::tuple<std::vector<int>,
						   std::vector< std::tuple<int, unsigned char*, int, char*> >*&,
						   int&,
						   mutex&,
						   condition_variable&> > rabinQueue;
	mutex rabinQueueMutex;
	condition_variable newRabinCond;
	// chunk queue: sync between ChunkingResultProc and ChunkHashing
	// consists of chunking result buffer indices used, response, mutex, condition variable
	std::queue< std::tuple<std::vector<int>,
						   std::vector< std::tuple<int, unsigned char*, int, char*> >*&,
						   int&,
						   mutex&,
						   condition_variable&> > chunkQueue;
	mutex chunkQueueMutex;
	condition_variable newChunksCond;
	// hash queue: sync between ChunkHashing and ChunkMatch
	// consists of response and condition variable
	std::queue< std::tuple<std::vector< std::tuple<int, unsigned char*, int, char*> >*&,
						   int&,
						   condition_variable&> > hashQueue;
	mutex hashQueueMutex;
	condition_variable newHashCond;
	// pagable buffer
	array<char*, PAGABLE_BUFFER_NUM> pagableBuffer;
	array<unsigned int, PAGABLE_BUFFER_NUM> pagableBufferLen;
	array<mutex, PAGABLE_BUFFER_NUM> pagableBufferMutex;
	array<condition_variable, PAGABLE_BUFFER_NUM> pagableBufferCond;
	array<bool, PAGABLE_BUFFER_NUM> pagableBufferObsolete;
	// fixed buffer
	array<char*, FIXED_BUFFER_NUM> fixedBuffer;
	array<unsigned int, FIXED_BUFFER_NUM> fixedBufferLen;
	// chunking kernel asynchronize
	array<char*, FIXED_BUFFER_NUM> kernelInputBuffer;
	array<unsigned int*, FIXED_BUFFER_NUM> kernelResultBuffer;
	array<unsigned int*, FIXED_BUFFER_NUM> hostResultBuffer;
	array<unsigned int, FIXED_BUFFER_NUM> hostResultLen;
	array<mutex, FIXED_BUFFER_NUM> hostResultMutex;
	array<condition_variable, FIXED_BUFFER_NUM> hostResultCond;
	array<bool, FIXED_BUFFER_NUM> hostResultObsolete;
	array<bool, FIXED_BUFFER_NUM> hostResultExecuting;
	// chunking result processing
	array<cudaStream_t, RESULT_BUFFER_NUM> stream;
	array<unsigned int*, RESULT_BUFFER_NUM> chunkingResultBuffer;
	array<unsigned int, RESULT_BUFFER_NUM> chunkingResultLen;
	array<mutex, RESULT_BUFFER_NUM> chunkingResultMutex;
	array<condition_variable, RESULT_BUFFER_NUM> chunkingResultCond;
	array<bool, RESULT_BUFFER_NUM> chunkingResultObsolete;
	// chunk hashing
	thread *segmentThreads;
	// chunk matching 
	LRUStrHash<SHA1_HASH_LENGTH> *circHashPool;
	mutex *circHashPoolMutex;
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
	* \param segmentNum the segment of chunking result that this thread is going to process
	* \param result a vector that stores the results, which are hash-value pairs
	* \param resultLenInUint8 the length of all the results if stored in a uint8_t array
	* \param resultMutex mutex of the result, used to prevent data race
	*/
	void ChunkSegmentHashing(int pagableBufferIdx, 
							 int chunkingResultIdx, 
							 int segmentNum,
							 std::vector< std::tuple<int, unsigned char*, int, char*> >* result,
							 int& resultLenInUint8,
							 mutex& resultMutex);

public:
	HarensRE(int mapperNum, int reducerNum);
	~HarensRE();

	/**
	* \brief fetching data for the GET request and do redundancy elimination process.
	* Simulating fetching data from server by reading files.
	* \param request the GET request (a file name stored data in server's file system)
	* \param result the hash-chunk pairs of the data. 
	* The two integers before hash value and data chunk are their lengths.
	* \param resultLenInUint8 the length of result if put it in a uint8_t array
	* MIND: result is a pointer, caller of this function should be responsible
	* to release the memory!
	*/
	void HandleGetRequest(std::string request,
						  std::vector< std::tuple<int, unsigned char*, int, char*> >* result,
						  int& resultLenInUint8);

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