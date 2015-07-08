#pragma once
#include <cuda_runtime_api.h>
#include <cuda.h>
#include "CircularPairQueue.h"
#include "RabinHash.h"
#include "CircularHash.h"
#include "CircularHashPool.h"
#include "Definition.h"

//The maximum block number in the server GPU
const int BLOCK_NUM = 4096;
//The maximum thread nubmer per block in the server GPU
const int THREAD_PER_BLOCK = 512;
//Mention! this is just the number of windows for each thread
//const int NUM_OF_WINDOWS_PER_THREAD = 4;
const unsigned int MAX_KERNEL_INPUT_LEN = BLOCK_NUM * THREAD_PER_BLOCK + WINDOW_SIZE - 1;

class RedundancyEliminator_CUDA {
private:
	RabinHash hashFunc;
	VirtualHash* circHash;

	//Add a new chunk into cache, if hash value queue is full also delete the oldest chunk
	void addNewChunk(unsigned long long hashValue, char* chunk, unsigned int chunkSize, bool isDuplicate);
	/*Take chunk and chunk size as input, hashValue as output*/
	inline unsigned long long computeChunkHash(char* chunk, unsigned int chunkSize);
public:
	unsigned long long *kernelTA, *kernelTB, *kernelTC, *kernelTD;
	enum Type {MultiFingerprint, NonMultifingerprint};

	//deque<unsigned int> chunking(char* kernelInput, unsigned int inputLen, unsigned long long *resultHost);
	void ChunkHashing(unsigned int* indices, int indicesNum, char* package,
		char** chunkList, unsigned long long* chunkHashValueList, unsigned int* chunkLenList);
	unsigned int ChunkMatching(deque<unsigned long long> &hashValues, deque<tuple<char*, unsigned int>> &chunks);
	/*deque<tuple<unsigned char*, unsigned int>> is for simulation, deque<unsigned char*> for real case*/
	void ChunkHashingAscyn(unsigned int* indices, int indicesNum, char* package, 
		unsigned long long* chunkHashValueList, unsigned int* chunkLenList, mutex &chunkMutex);
	void ChunkHashingAscynWithCircularQueue(unsigned int* indices, int indicesNum, char* package,
		CircularPairQueue<unsigned long long, unsigned int> &chunkHashQ);
	unsigned int fingerPrinting(deque<unsigned int> indexQ, char* package);
	void RabinHashAsync(char* inputKernel, char* inputHost, unsigned int inputLen, 
		unsigned long long* resultKernel, unsigned long long* resultHost, cudaStream_t stream);

	unsigned int eliminateRedundancy(char* package, unsigned int packageSize);
	RedundancyEliminator_CUDA() {}
	RedundancyEliminator_CUDA(Type type);
	void SetupRedundancyEliminator_CUDA(Type type);
	~RedundancyEliminator_CUDA();
};

__global__ void Hash(const unsigned long long *TA, const unsigned long long *TB, const unsigned long long *TC, const unsigned long long * TD,
	const char *str, const unsigned int windowsNum, unsigned long long *result/*, int *debug*/);
__device__ void SetResultElement(unsigned long long* subResult, const unsigned int idx, const unsigned long long resultPoint);
__device__ unsigned long long* GetSubResult(unsigned long long* result, const unsigned int blockNum);
__device__ const char* GetSubStr(const char *str, const unsigned int blockNum);
__device__ unsigned int GetUIntFromStr(const char* strs, const unsigned int idx);
__device__ unsigned long long GetULongFromStr(const char* strs, const unsigned int idx);
__device__ char GetChar(const char* subStr, const unsigned int idx);