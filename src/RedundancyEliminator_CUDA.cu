#include "RedundancyEliminator_CUDA.h"

#define CharArraySize(array) strlen(array)

//The cuda kernel functions
/*
* Put the computed resultPoint of each thread to subResult
*/
__device__ void SetResultElement(unsigned int* subResult, 
								 const unsigned int idx, 
								 const unsigned int resultPoint) {
	subResult[idx] = resultPoint;
}

/*
* Get the memory address of the result for each block of threads
*/
__device__ unsigned int* GetSubResult(unsigned int* result, 
									  const unsigned int blockNum) {
	return &(result[blockNum * THREAD_PER_BLOCK]);
}

/*
* Get the input required by each block of threads
*/
__device__ const char* GetSubStr(const char *str, 
								 const unsigned int blockNum) {
	return &(str[blockNum * THREAD_PER_BLOCK]);
}

/*
* Transform four consecutive bytes into unsigned integer
*/
__device__ unsigned int GetUIntFromStr(const char* strs, 
									   const unsigned int idx) {
	return (strs[idx] << 24) | (strs[idx + 1] << 16) | (strs[idx + 2] << 8) | (strs[idx + 3]);
}

/*
* Transform eight consecutive bytes into unsigned long long
*/
__device__ unsigned long long GetULongFromStr(const char* strs, 
											  const unsigned int idx) {
	//memcpy doesn't work in kernel, keeping it here for record
	/*
	unsigned long long result;
	memcpy((void*)&result, strs, BYTES_IN_ULONG);
	return result;
	*/
	
	unsigned long long result = strs[idx];
	for (int i = 1; i < 8; ++i)
		result = (result << 8) | strs[idx + i];
	return result;
	
}

/*
* Get the character required by each thread
*/
__device__ char GetChar(const char* subStr, 
						const unsigned int idx) {
	return subStr[idx];
}

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
					 unsigned int *result) {
	if (blockDim.x * blockIdx.x + threadIdx.x >= windowsNum)
		return;

	unsigned int blockNum = blockIdx.x;
	const char* subStr = GetSubStr(str, blockNum);
	unsigned int* subResult = GetSubResult(result, blockNum);

	__shared__ char s_str[THREAD_PER_BLOCK + 3];
	__shared__ char s_str_shift[THREAD_PER_BLOCK + 7];
	unsigned int threadNum = threadIdx.x;

	//This will cover all the elements
	s_str[threadNum] = subStr[threadNum];
	if (threadNum >= THREAD_PER_BLOCK - 3)
		s_str[threadNum + 3] = subStr[threadNum + 3];

	if(threadNum < 7)
		s_str_shift[threadNum] = subStr[threadNum + 4];
	s_str_shift[threadNum + 7] = subStr[threadNum + 11];
	// before starting the computation, make sure all shared memory are set
	__syncthreads();

	int h, i, j, k;

	h = s_str[threadNum];
	i = s_str[threadNum + 1];
	j = s_str[threadNum + 2];
	k = s_str[threadNum + 3];

	unsigned long long resultPoint = GetULongFromStr(s_str_shift, threadNum);
	resultPoint ^= TA[h] ^ TB[i] ^ TC[j] ^ TD[k];
	unsigned int resultMod = resultPoint & P_MINUS;
	SetResultElement(subResult, threadNum, resultMod);
	//debug
	/*debug[threadNum * 2] = windowStart;
	debug[threadNum * 2 + 1] = result[windowStart];*/
}

RedundancyEliminator_CUDA::RedundancyEliminator_CUDA(Type type) {
	if (type == NonMultifingerprint)
		circHash = new LRUStrHash<SHA_DIGEST_LENGTH>(MAX_CHUNK_NUM);
	else
		circHash = new LRUStrHashPool<SHA_DIGEST_LENGTH>(MAX_CHUNK_NUM);
	hashFunc = RabinHash();
	int tableSize = RabinHash::TABLE_ROW_NUM * BYTES_IN_ULONG;
	cudaMalloc((void**)&kernelTA, tableSize);
	cudaMalloc((void**)&kernelTB, tableSize);
	cudaMalloc((void**)&kernelTC, tableSize);
	cudaMalloc((void**)&kernelTD, tableSize);
	cudaMemcpy(kernelTA, hashFunc.GetTALONG(), tableSize, cudaMemcpyHostToDevice);
	cudaMemcpy(kernelTB, hashFunc.GetTBLONG(), tableSize, cudaMemcpyHostToDevice);
	cudaMemcpy(kernelTC, hashFunc.GetTCLONG(), tableSize, cudaMemcpyHostToDevice);
	cudaMemcpy(kernelTD, hashFunc.GetTDLONG(), tableSize, cudaMemcpyHostToDevice);
	//The real software need to generate a initial file named 0xFF here
	//Check LRU.cpp to see the reason
}

void RedundancyEliminator_CUDA::SetupRedundancyEliminator_CUDA(Type type) {
	if (type == NonMultifingerprint)
		circHash = new LRUStrHash<SHA_DIGEST_LENGTH>(MAX_CHUNK_NUM);
	else
		circHash = new LRUStrHashPool<SHA_DIGEST_LENGTH>(MAX_CHUNK_NUM);
	hashFunc = RabinHash();
	int tableSize = RabinHash::TABLE_ROW_NUM * BYTES_IN_ULONG;
	cudaMalloc((void**)&kernelTA, tableSize);
	cudaMalloc((void**)&kernelTB, tableSize);
	cudaMalloc((void**)&kernelTC, tableSize);
	cudaMalloc((void**)&kernelTD, tableSize);
	cudaMemcpy(kernelTA, hashFunc.GetTALONG(), tableSize, cudaMemcpyHostToDevice);
	cudaMemcpy(kernelTB, hashFunc.GetTBLONG(), tableSize, cudaMemcpyHostToDevice);
	cudaMemcpy(kernelTC, hashFunc.GetTCLONG(), tableSize, cudaMemcpyHostToDevice);
	cudaMemcpy(kernelTD, hashFunc.GetTDLONG(), tableSize, cudaMemcpyHostToDevice);
	//The real software need to generate a initial file named 0xFF here
	//Check LRU.cpp to see the reason
}

RedundancyEliminator_CUDA::~RedundancyEliminator_CUDA() {
	delete circHash;
	cudaFree(kernelTA);
	cudaFree(kernelTB);
	cudaFree(kernelTC);
	cudaFree(kernelTD);
	//The real software would delete all the generated files here
}

/*
* Add a new chunck into the file system, if the hash value queue is full, 
* also delete the oldest chunk.
*/
void RedundancyEliminator_CUDA::addNewChunk(unsigned char* hashValue, 
											char* chunk, 
											unsigned int chunkSize, 
											bool isDuplicate) {
	unsigned char* toBeDel = circHash->Add(hashValue, isDuplicate);
	//Remove chunk corresponding to toBeDel from storage
	/*fstream file(hashValue.c_str(), std::fstream::in | std::fstream::out);
	file << chunk;
	file.close();*/
}

/*
* Compute the hash value of each chunk.
* The hash values are pushed into &chunkHashQ whenever it's computed,
* and another thread would process the hash values simultaneously
*/
void RedundancyEliminator_CUDA::ChunkHashingAsync(unsigned int* indices, 
												  int indicesNum, 
												  char* package,
												  CircularQueuePool &chunkHashQ) 
{
	unsigned int prevIdx = 0;
	char* chunk;
	unsigned char *chunkHash = new unsigned char[SHA_DIGEST_LENGTH];
	unsigned int chunkLen;
	for (int i = 0; i < indicesNum; ++i) {
		if (prevIdx == 0) {
			prevIdx = indices[i];
			continue;
		}
		chunkLen = indices[i] - prevIdx;
		//if chunk is too small, combine it with the next chunk
		if (chunkLen < MIN_CHUNK_LEN)
			continue;

		chunk = &(package[prevIdx]);
		EncryptionHashes::computeSha1Hash(chunk, chunkLen, chunkHash);
		chunkHashQ.Push(chunkHash, chunkLen, (*mod));

		//Mind! never use sizeof(chunk) to check the chunk size
		prevIdx = indices[i];
	}
	//cout << "end\n";
}

/*
* Compute the hash value of each chunk.
* both hash values and chunks are also stored in the result vector
* and another thread would process the hash values simultaneously
* MIND: caller is responsible in releasing the new chunk in result but not hash value
*/
void RedundancyEliminator_CUDA::ChunkHashingAsync(unsigned int* indices, 
												  int indicesNum, 
												  char* package,
												  std::vector< std::tuple<int, unsigned char*, int, char*> >* result,
												  mutex& resultMutex) 
{
	unsigned int prevIdx = 0;
	char* chunk;
	unsigned char *chunkHash = new unsigned char[SHA_DIGEST_LENGTH];
	unsigned int chunkLen;
	for (int i = 0; i < indicesNum; ++i) {
		if (prevIdx == 0) {
			prevIdx = indices[i];
			continue;
		}
		chunkLen = indices[i] - prevIdx;
		//if chunk is too small, combine it with the next chunk
		if (chunkLen < MIN_CHUNK_LEN)
			continue;
		chunk = &(package[prevIdx]);
		EncryptionHashes::computeSha1Hash(chunk, chunkLen, chunkHash);

		//Make a new chunk because the old chunk would be released after chunk hashing
		char* newChunk = new char[chunkLen];
		memcpy(newChunk, &chunk, chunkLen);
		//Push result into vector
		resultMutex.lock();
		result->push_back(make_tuple(SHA_DIGEST_LENGTH,
									 chunkHash,
									 chunkLen,
									 newChunk));
		resultMutex.unlock();
		//Mind! never use sizeof(chunk) to check the chunk size
		prevIdx = indices[i];
	}
	//cout << "end\n";
}

/*
* Compute hash value for each chunk and find out the duplicate chunks.
* Take a queue as input
*/
unsigned int RedundancyEliminator_CUDA::fingerPrinting(deque<unsigned int> indexQ, 
													   char* package) 
{
	/*deque<unsigned char*> hashValues;
	deque<tuple<char*, unsigned int>> chunks;
	ChunkHashing(indexQ, package, hashValues, chunks);
	return ChunkMatching(hashValues, chunks);*/
	unsigned int duplicationSize = 0;
	unsigned int prevIdx = 0;
	char* chunk;
	unsigned int chunkLen;
	bool isDuplicate;
	for (deque<unsigned int>::const_iterator iter = indexQ.begin(); iter != indexQ.end(); ++iter) {
		if (prevIdx == 0) {
			prevIdx = *iter;
			continue;
		}
		chunkLen = *iter - prevIdx;
		chunk = &(package[prevIdx]);

		//Mind! never use sizeof(chunk) to check the chunk size
		unsigned char *chunkHash = new unsigned char[SHA_DIGEST_LENGTH];
		EncryptionHashes::computeSha1Hash(chunk, chunkLen, chunkHash);
		if (circHash->Find(chunkHash)) { //find duplications
			duplicationSize += chunkLen;
			isDuplicate = true;
		}
		else {
			isDuplicate = false;
		}
		addNewChunk(chunkHash, chunk, chunkLen, isDuplicate);
		prevIdx = *iter;
	}
	return duplicationSize;
}

/*
* Compute hash value for each chunk and find out the duplicate chunks.
* Take a index array as input
*/
unsigned int RedundancyEliminator_CUDA::fingerPrinting(unsigned int *idxArr, 
													   unsigned int idxArrLen, 
													   char* package) {
	unsigned int duplicationSize = 0;
	unsigned int prevIdx = 0;
	char* chunk;
	unsigned int chunkLen;
	bool isDuplicate;
	for (int i = 0; i < idxArrLen; ++i) {
		if (prevIdx == 0) {
			prevIdx = idxArr[i];
			continue;
		}
		chunkLen = idxArr[i] - prevIdx;
		//if chunk is too small, combine it with the next chunk
		if (chunkLen < MIN_CHUNK_LEN)
			continue;
		
		chunk = &(package[prevIdx]);

		//Mind! never use sizeof(chunk) to check the chunk size
		unsigned char* chunkHash = new unsigned char[SHA_DIGEST_LENGTH];
		EncryptionHashes::computeSha1Hash(chunk, chunkLen, chunkHash);
		if (circHash->Find(chunkHash)) { //find duplications
			duplicationSize += chunkLen;
			isDuplicate = true;
		}
		else {
			isDuplicate = false;
		}
		addNewChunk(chunkHash, chunk, chunkLen, isDuplicate);
		prevIdx = idxArr[i];
	}
	return duplicationSize;
}

/*
* Compute Rabin hash value of every single consecutive $WINDOW_SIZE$ bytes (see Definition.h)
* Kernel function and data transfer between host and kernel would be processed simultaneously.
*/
void RedundancyEliminator_CUDA::RabinHashAsync(char* inputKernel, 
											   char* inputHost, 
											   unsigned int inputLen, 
											   unsigned int* resultKernel, 
											   unsigned int* resultHost, 
											   cudaStream_t stream) {
	cudaMemcpyAsync(inputKernel, 
					inputHost,	
					inputLen, 
					cudaMemcpyHostToDevice, 
					stream);
	Hash<<<BLOCK_NUM, THREAD_PER_BLOCK, 0, stream>>> (kernelTA, 
													  kernelTB, 
													  kernelTC, 
													  kernelTD,
													  inputKernel, 
													  (inputLen - WINDOW_SIZE + 1), 
													  resultKernel);
	cudaMemcpyAsync(resultHost, 
					resultKernel,
					(inputLen - WINDOW_SIZE + 1) * BYTES_IN_UINT, 
					cudaMemcpyDeviceToHost, 
					stream);
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf(stderr, 
				"ERROR1: %s \n", 
				cudaGetErrorString(error));
	}
}

/*
* Take the first $sizeof(unsigned int)$ bytes of hashValue as divident,
* and compute the result of divident modulo divisor.
*/
int mod(unsigned char* hashValue, int divisor) {
	unsigned int hashValueInt;
	memcpy(&hashValueInt, hashValue, sizeof(unsigned int));
	return (int)(hashValueInt % divisor);
}