#pragma once
#include <cuda_runtime_api.h>
#include <cuda.h>
#include "CircularQueuePool.h"
#include "RabinHash.h"
#include "LRUStrHash.h"
#include "LRUStrHashPool.h"
#include "Definition.h"
#include "EncryptionHashes.h"

/*
* The redundancy elimination module for cuda accelerated implementations
*/
class RedundancyEliminator_CUDA {
private:
	RabinHash hashFunc;
	LRUVirtualHash<SHA1_HASH_LENGTH>* circHash;

	/*
	* Add a new chunk into cache, if hash value queue is full, 
	* also delete the oldest chunk
	*/
	void addNewChunk(unsigned char* hashValue, 
					 char* chunk, 
					 unsigned int chunkSize, 
					 bool isDuplicate);

public:
	unsigned long long *kernelTA, *kernelTB, *kernelTC, *kernelTD;
	enum Type {MultiFingerprint, NonMultifingerprint};

	/*
	* Compute the hash value of each chunk.
	* The hash values are pushed into &chunkHashQ whenever it's computed,
	* and another thread would process the hash values simultaneously
	*/
	void ChunkHashingAsync(unsigned int* indices, 
						   int indicesNum, 
						   char* package,
						   CircularQueuePool &chunkHashQ);

	/*
	* Compute the hash value of each chunk.
	* both hash values and chunks are also stored in the result vector
	* and another thread would process the hash values simultaneously
	* MIND: caller is responsible in releasing the new chunk in result but not hash value
	*/
	void ChunkHashingAsync(unsigned int* indices, 
						   int indicesNum, 
						   char* package,
						   std::vector< std::tuple<int, unsigned char*, int, char*> >* result,
						   int* resultLenInUint8,
						   std::mutex* resultMutex);
	
	/*
	* Compute hash value for each chunk and find out the duplicate chunks.
	* Take a queue as input
	*/
	unsigned int fingerPrinting(std::deque<unsigned int> indexQ, 
								char* package);

	/*
	* Compute hash value for each chunk and find out the duplicate chunks.
	* Take a index array as input
	*/
	unsigned int fingerPrinting(unsigned int* idxArr, 
								unsigned int idxArrLen, 
								char* package);
	
	/*
	* Compute Rabin hash value of every single consecutive $WINDOW_SIZE$ bytes (see Definition.h)
	* Kernel function and data transfer between host and kernel would be processed simultaneously.
	*/
	void RabinHashAsync(char* inputKernel, 
						char* inputHost, 
						unsigned int inputLen, 
						unsigned int* resultKernel, 
						unsigned int* resultHost,
						cudaStream_t stream);

	RedundancyEliminator_CUDA() {}
	RedundancyEliminator_CUDA(Type type);
	void SetupRedundancyEliminator_CUDA(Type type);
	~RedundancyEliminator_CUDA();
};

//The cuda kernel functions
/*
* Put the computed resultPoint of each thread to subResult
*/
__device__ void SetResultElement(unsigned int* subResult,
								 const unsigned int idx,
								 const unsigned int resultPoint);

/*
* Get the memory address of the result for each block of threads
*/
__device__ unsigned int* GetSubResult(unsigned int* result, 
									  const unsigned int blockNum);

/*
* Get the input required by each block of threads
*/
__device__ const char* GetSubStr(const char *str, 
								 const unsigned int blockNum);

/*
* Transform four consecutive bytes into unsigned integer
*/
__device__ unsigned int GetUIntFromStr(const char* strs, 
									   const unsigned int idx);

/*
* Transform eight consecutive bytes into unsigned long long
*/
__device__ unsigned long long GetULongFromStr(const char* strs, 
											  const unsigned int idx);

/*
* Get the character required by each thread
*/
__device__ char GetChar(const char* subStr, 
						const unsigned int idx);

/*
* Compute Rabin hash value of every single consecutive $WINDOW_SIZE$ bytes (see Definition.h)
* Called by cuda kernel.
*/
__global__ void Hash(const unsigned long long *TA, 
					 const unsigned long long *TB, 
					 const unsigned long long *TC, 
					 const unsigned long long * TD,
					 const char *str, 
					 const unsigned int windowsNum, 
					 unsigned int *result);

/*
* Take the first $sizeof(unsigned int)$ bytes of hashValue as divident,
* and compute the result of divident modulo divisor.
*/
int mod(unsigned char* hashValue, 
		int divisor);