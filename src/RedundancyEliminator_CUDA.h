#pragma once
#include <cuda_runtime_api.h>
#include <cuda.h>
#include "CircularQueuePool.h"
#include "RabinHash.h"
#include "LRUStrHash.h"
#include "LRUStrHashPool.h"
#include "Definition.h"
#include "EncryptionHashes.h"

class RedundancyEliminator_CUDA {
private:
	RabinHash hashFunc;
	LRUVirtualHash<SHA_DIGEST_LENGTH>* circHash;

	//Add a new chunk into cache, if hash value queue is full also delete the oldest chunk
	void addNewChunk(unsigned char* hashValue, 
					 char* chunk, 
					 unsigned int chunkSize, 
					 bool isDuplicate);

public:
	unsigned long long *kernelTA, *kernelTB, *kernelTC, *kernelTD;
	enum Type {MultiFingerprint, NonMultifingerprint};

	//deque<unsigned int> chunking(char* kernelInput, unsigned int inputLen, unsigned long long *resultHost);
	//void ChunkHashing(unsigned int* indices, int indicesNum, char* package,
	//	char** chunkList, unsigned long long* chunkHashValueList, unsigned int* chunkLenList);
	//unsigned int ChunkMatching(deque<unsigned long long> &hashValues, deque<tuple<char*, unsigned int>> &chunks);
	/*deque<tuple<unsigned char*, unsigned int>> is for simulation, deque<unsigned char*> for real case*/

	void ChunkHashingAsync(unsigned int* indices, 
						   int indicesNum, 
						   char* package,
						   CircularQueuePool &chunkHashQ);
	unsigned int fingerPrinting(deque<unsigned int> indexQ, 
								char* package);
	unsigned int fingerPrinting(unsigned int* idxArr, 
								unsigned int idxArrLen, 
								char* package);
	void RabinHashAsync(char* inputKernel, 
						char* inputHost, 
						unsigned int inputLen, 
						unsigned int* resultKernel, 
						unsigned int* resultHost,
						cudaStream_t stream);

	//unsigned int eliminateRedundancy(char* package, 
	//								 unsigned int packageSize);
	RedundancyEliminator_CUDA() {}
	RedundancyEliminator_CUDA(Type type);
	void SetupRedundancyEliminator_CUDA(Type type);
	~RedundancyEliminator_CUDA();
};

__device__ void SetResultElement(unsigned int* subResult,
								 const unsigned int idx,
								 const unsigned int resultPoint);
__device__ unsigned int* GetSubResult(unsigned int* result, 
									  const unsigned int blockNum);
__device__ const char* GetSubStr(const char *str, 
								 const unsigned int blockNum);
__device__ unsigned int GetUIntFromStr(const char* strs, 
									   const unsigned int idx);
__device__ unsigned long long GetULongFromStr(const char* strs, 
											  const unsigned int idx);
__device__ char GetChar(const char* subStr, 
						const unsigned int idx);
__global__ void Hash(const unsigned long long *TA, 
					 const unsigned long long *TB, 
					 const unsigned long long *TC, 
					 const unsigned long long * TD,
					 const char *str, 
					 const unsigned int windowsNum, 
					 unsigned int *result);

int mod(unsigned char* hash, 
		int divisor);