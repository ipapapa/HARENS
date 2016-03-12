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
	thread *chunk_match_threads;
	// request queue: sync between HandleGetRequest and ReadData
	// consists of request, response, mutex, condition variable
	std::queue< std::tuple<std::string&, 
						   std::vector< std::tuple<int, unsigned char*, int, char*> >*&,
						   mutex&,
						   condition_variable&> > requestQueue;
	mutex requestQueueMutex;
	condition_variable newRequestCond;
	// package queue: sync between ReadData and ChunkingKernel
	// consists of pagable buffer indices for package data, response, mutex, condition variable
	std::queue< std::tuple<std::vector<int>,
						   std::vector< std::tuple<int, unsigned char*, int, char*> >*&,
						   mutex&,
						   condition_variable&> > packageQueue;
	mutex packageQueueMutex;
	condition_variable newPackageCond;
	// rabin result queue: sync between ChunkingKernel and ChunkingResultProc
	// consists of result-host indices for rabin hash, response, mutex, condition variable
	std::queue< std::tuple<std::vector<int>,
						   std::vector< std::tuple<int, unsigned char*, int, char*> >*&,
						   mutex&,
						   condition_variable&> > rabinQueue;
	mutex rabinQueueMutex;
	condition_variable newRabinCond;
	// chunk queue: sync between ChunkingResultProc and ChunkHashing
	// consists of chunking result buffer indices used, response, mutex, condition variable
	std::queue< std::tuple<std::vector<int>,
						   std::vector< std::tuple<int, unsigned char*, int, char*> >*&,
						   mutex&,
						   condition_variable&> > chunkQueue;
	mutex chunkQueueMutex;
	condition_variable newChunksCond;
	// hash queue: sync between ChunkHashing and ChunkMatch
	// consists of response and condition variable
	std::queue< std::tuple<std::vector< std::tuple<int, unsigned char*, int, char*> >*&,
						   condition_variable&> > hashQueue;
	mutex hashQueueMutex;
	condition_variable newHashCond;
	// pagable buffer
	array<char*, PAGABLE_BUFFER_NUM> pagable_buffer;
	array<unsigned int, PAGABLE_BUFFER_NUM> pagable_buffer_len;
	array<mutex, PAGABLE_BUFFER_NUM> pagable_buffer_mutex;
	array<condition_variable, PAGABLE_BUFFER_NUM> pagable_buffer_cond;
	array<bool, PAGABLE_BUFFER_NUM> pagable_buffer_obsolete;
	// fixed buffer
	array<char*, FIXED_BUFFER_NUM> fixed_buffer;
	array<unsigned int, FIXED_BUFFER_NUM> fixed_buffer_len;
	// chunking kernel asynchronize
	array<char*, FIXED_BUFFER_NUM> input_kernel;
	array<unsigned int*, FIXED_BUFFER_NUM> result_kernel;
	array<unsigned int*, FIXED_BUFFER_NUM> result_host;
	array<unsigned int, FIXED_BUFFER_NUM> result_host_len;
	array<mutex, FIXED_BUFFER_NUM> result_host_mutex;
	array<condition_variable, FIXED_BUFFER_NUM> result_host_cond;
	array<bool, FIXED_BUFFER_NUM> result_host_obsolete;
	array<bool, FIXED_BUFFER_NUM> result_host_executing;
	// chunking result processing
	array<cudaStream_t, RESULT_BUFFER_NUM> stream;
	array<unsigned int*, RESULT_BUFFER_NUM> chunking_result;
	array<unsigned int, RESULT_BUFFER_NUM> chunking_result_len;
	array<mutex, RESULT_BUFFER_NUM> chunking_result_mutex;
	array<condition_variable, RESULT_BUFFER_NUM> chunking_result_cond;
	array<bool, RESULT_BUFFER_NUM> chunking_result_obsolete;
	// chunk hashing
	thread *segment_threads;
	// chunk matching 
	LRUStrHash<SHA_DIGEST_LENGTH> *circ_hash_pool;
	unsigned long long *duplication_size;
	unsigned long long total_duplication_size = 0;
	unsigned long long totalFileLen = 0;
	// time
	clock_t start_r, 
			end_r, 
			start_ck, 
			end_ck, 
			start_cp, 
			end_cp, 
			start_ch, 
			end_ch, 
			start_cm, 
			end_cm;
	double time_r = 0, 
		   time_ck = 0, 
		   time_cp = 0,
		   time_ch, 
		   time_cm;

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
	void ChunkMatch(int hashPoolIdx);

	/**
	* \brief compute a non-collision hash (SHA-1) value for each chunk in the segment
	*/
	void ChunkSegmentHashing(int pagableBufferIdx, 
							 int chunkingResultIdx, 
							 int segmentNum,
							 std::vector< std::tuple<int, unsigned char*, int, char*> >* result,
							 mutex& resultMutex);

public:
	HarensRE(int mapperNum, int reducerNum);
	~HarensRE();

	/**
	* \brief fetching data for the GET request and do redundancy elimination process.
	* Simulating fetching data from server by reading files.
	* \param the GET request (a file name stored data in server's file system)
	* \return the hash-chunk pairs of the data. 
	* The two integers before hash value and data chunk are their lengths.
	* MIND: return value is a pointer, caller of this function should be responsible
	* to release the memory!
	*/
	std::vector< std::tuple<int, unsigned char*, int, char*> >*
	HandleGetRequest(std::string request);

	/**
	* \brief start the core of redundancy elimination module
	*/
	void Start();

	/**
	* \brief terminate the core of redundancy elimination module
	*/
	void End();
};

#endif /* HARENS_RE_H */