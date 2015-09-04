#include "RedundancyEliminator_CUDA.h"

#define CharArraySize(array) strlen(array)

__device__ void SetResultElement(unsigned long long* subResult, const unsigned int idx, const unsigned long long resultPoint) {
	subResult[idx] = resultPoint;
}

__device__ unsigned long long* GetSubResult(unsigned long long* result, const unsigned int blockNum) {
	return &(result[blockNum * THREAD_PER_BLOCK]);
}

__device__ const char* GetSubStr(const char *str, const unsigned int blockNum) {
	return &(str[blockNum * THREAD_PER_BLOCK]);
}

__device__ unsigned int GetUIntFromStr(const char* strs, const unsigned int idx) {
	return (strs[idx] << 24) | (strs[idx + 1] << 16) | (strs[idx + 2] << 8) | (strs[idx + 3]);
}

__device__ unsigned long long GetULongFromStr(const char* strs, const unsigned int idx) {
	/*unsigned long long result;
	memcpy((void*)&result, strs, BYTES_IN_ULONG);
	return result;
	*/
	
	unsigned long long result = strs[idx];
	for (int i = 1; i < 8; ++i)
	result = (result << 8) | strs[idx + i];
	return result;
	
}

__device__ char GetChar(const char* subStr, const unsigned int idx) {
	return subStr[idx];
}

__global__ void Hash(const unsigned long long *TA, const unsigned long long *TB, const unsigned long long *TC, const unsigned long long * TD,
	const char *str, const unsigned int windowsNum, unsigned long long *result/*, int *debug*/) {
	if (blockDim.x * blockIdx.x + threadIdx.x >= windowsNum)
		return;

	unsigned int blockNum = blockIdx.x;
	const char* subStr = GetSubStr(str, blockNum);
	unsigned long long* subResult = GetSubResult(result, blockNum);

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
	
	SetResultElement(subResult, threadNum, resultPoint);
	//debug
	/*debug[threadNum * 2] = windowStart;
	debug[threadNum * 2 + 1] = result[windowStart];*/
}

RedundancyEliminator_CUDA::RedundancyEliminator_CUDA(Type type) {
	if (type == NonMultifingerprint)
		circHash = new CircularHash(MAX_CHUNK_NUM);
	else
		circHash = new CircularHashPool(MAX_CHUNK_NUM);
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
	//Check Circular.cpp to see the reason
}

void RedundancyEliminator_CUDA::SetupRedundancyEliminator_CUDA(Type type) {
	if (type == NonMultifingerprint)
		circHash = new CircularHash(MAX_CHUNK_NUM);
	else
		circHash = new CircularHashPool(MAX_CHUNK_NUM);
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
	//Check Circular.cpp to see the reason
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
Add a new chunck into the file system, if the hash value queue is full, also delete the oldest chunk.
*/
void RedundancyEliminator_CUDA::addNewChunk(unsigned long long hashValue, char* chunk, unsigned int chunkSize, bool isDuplicate) {
	unsigned long long to_be_del = circHash->Add(hashValue, isDuplicate);
	/*fstream file(hashValue.c_str(), std::fstream::in | std::fstream::out);
	file << chunk;
	file.close();*/
}

void RedundancyEliminator_CUDA::ChunkHashing(unsigned int* indices, int indicesNum, char* package, 
	char** chunkList, unsigned long long* chunkHashValueList, unsigned int* chunkLenList) {
	unsigned int prevIdx = 0;
	for (int i = 0; i < indicesNum; ++i) {
		if (prevIdx == 0) {
			prevIdx = indices[i];
			continue;
		}
		chunkLenList[i - 1] = indices[i] - prevIdx;
		chunkList[i - 1] = &(package[prevIdx]);

		//Mind! never use sizeof(chunk) to check the chunk size
		chunkHashValueList[i - 1] = computeChunkHash(chunkList[i - 1], chunkLenList[i - 1]);
		prevIdx = indices[i];
	}
}

unsigned int RedundancyEliminator_CUDA::ChunkMatching(deque<unsigned long long> &hashValues, deque<tuple<char*, unsigned int>> &chunks) {
	unsigned int duplicationSize = 0;
	bool isDuplicate;
	deque<unsigned long long>::const_iterator hashValueIter = hashValues.begin();
	deque<tuple<char*, unsigned int>>::const_iterator chunksIter = chunks.begin();
	while (hashValueIter != hashValues.end()) {
		if (circHash->Find(*hashValueIter)) {
			duplicationSize += get<1>(*chunksIter);
			isDuplicate = true;
		}
		else {
			isDuplicate = false;
		}
		addNewChunk(*hashValueIter, get<0>(*chunksIter), get<1>(*chunksIter), isDuplicate);
		++hashValueIter;
		++chunksIter;
	}
	return duplicationSize;
}

void RedundancyEliminator_CUDA::ChunkHashingAscynWithCircularQueuePool(unsigned int* indices, int indicesNum, char* package,
	CircularQueuePool<tuple<unsigned long long, unsigned int>> &chunkHashQ) {
	//cout << "start\n";
	//unsigned int duplicationSize = 0;
	unsigned int prevIdx = 0;
	char* chunk;
	unsigned int chunkLen;
	unsigned long long chunkHashValue;
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
		chunkHashValue = computeChunkHash(chunk, chunkLen);
		chunkHashQ.Push(make_tuple(chunkHashValue, chunkLen), (*mod));

		//Mind! never use sizeof(chunk) to check the chunk size
		prevIdx = indices[i];
	}
	//cout << "end\n";
}

void RedundancyEliminator_CUDA::ChunkHashingAscyn(unsigned int* indices, int indicesNum, char* package, 
	unsigned long long* chunkHashValueList, unsigned int* chunkLenList, mutex &chunkMutex) {
	//unsigned int duplicationSize = 0;
	unsigned int prevIdx = 0;
	char* chunk;
	for (int i = 0; i < indicesNum; ++i) {
		if (prevIdx == 0) {
			prevIdx = indices[i];
			continue;
		}
		chunkLenList[i - 1] = indices[i] - prevIdx;
		chunk = &(package[prevIdx]);

		//Mind! never use sizeof(chunk) to check the chunk size
		chunkMutex.lock();
		chunkHashValueList[i - 1] = computeChunkHash(chunk, chunkLenList[i - 1]);
		chunkMutex.unlock();
		prevIdx = indices[i];
	}
}

unsigned int RedundancyEliminator_CUDA::fingerPrinting(deque<unsigned int> indexQ, char* package) {
	/*deque<unsigned char*> hashValues;
	deque<tuple<char*, unsigned int>> chunks;
	ChunkHashing(indexQ, package, hashValues, chunks);
	return ChunkMatching(hashValues, chunks);*/
	unsigned int duplicationSize = 0;
	unsigned int prevIdx = 0;
	char* chunk;
	unsigned long long chunkHash;
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
		chunkHash = computeChunkHash(chunk, chunkLen);
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

unsigned int RedundancyEliminator_CUDA::fingerPrinting(unsigned int *idxArr, unsigned int idxArrLen, char* package) {
	unsigned int duplicationSize = 0;
	unsigned int prevIdx = 0;
	char* chunk;
	unsigned long long chunkHash;
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
		chunkHash = computeChunkHash(chunk, chunkLen);
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

void RedundancyEliminator_CUDA::RabinHashAsync(char* inputKernel, char* inputHost, unsigned int inputLen, unsigned long long* resultKernel, unsigned long long* resultHost, cudaStream_t stream) {
	cudaMemcpyAsync(inputKernel, inputHost,	inputLen, cudaMemcpyHostToDevice, stream);
	Hash << <BLOCK_NUM, THREAD_PER_BLOCK, 0, stream>> > (kernelTA, kernelTB, kernelTC, kernelTD,
		inputKernel, (inputLen - WINDOW_SIZE + 1), resultKernel/*, debugDevice*/);
	cudaMemcpyAsync(resultHost, resultKernel,
		(inputLen - WINDOW_SIZE + 1) * BYTES_IN_ULONG, cudaMemcpyDeviceToHost, stream);
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf(stderr, "ERROR1: %s \n", cudaGetErrorString(error));
	}
}

//take a kernel global memory address and the size as input
unsigned int RedundancyEliminator_CUDA::eliminateRedundancy(char* package, unsigned int packageSize) {
	/*char *kernelInput;
	cudaMalloc((void**)&kernelInput, MAX_KERNEL_INPUT_LEN);*/
	unsigned int totalDuplicationSize = 0;
	deque<unsigned int> indexQ;
	char* packageInput[2];
	char* kernelInput[2];
	unsigned long long* resultHost[2];
	unsigned long long *kernelResult[2];
	clock_t start;
	clock_t end;
	double time = 0;

	const unsigned int MAX_WINDOW_NUM = MAX_KERNEL_INPUT_LEN - WINDOW_SIZE + 1;
	cudaMallocHost((void**)&packageInput[0], MAX_KERNEL_INPUT_LEN);
	cudaMallocHost((void**)&packageInput[1], MAX_KERNEL_INPUT_LEN);
	cudaMallocHost((void**)&resultHost[0], MAX_WINDOW_NUM * BYTES_IN_ULONG);
	cudaMallocHost((void**)&resultHost[1], MAX_WINDOW_NUM * BYTES_IN_ULONG);

	cudaMalloc((void**)&kernelInput[0], MAX_KERNEL_INPUT_LEN);
	cudaMalloc((void**)&kernelInput[1], MAX_KERNEL_INPUT_LEN);
	cudaMalloc((void**)&kernelResult[0], MAX_WINDOW_NUM * BYTES_IN_ULONG);
	cudaMalloc((void**)&kernelResult[1], MAX_WINDOW_NUM * BYTES_IN_ULONG);

	cudaStream_t* streams = new cudaStream_t[2];
	for (int i = 0; i < 2; ++i)
		cudaStreamCreate(&(streams[i]));

	int bufferIdx = 0;
	unsigned int curInputLen = MAX_KERNEL_INPUT_LEN, curWindowNum, curFilePos = 0;
	int iterator;
	for (iterator = 0; curInputLen == MAX_KERNEL_INPUT_LEN; ++iterator) {
		curInputLen = min(MAX_KERNEL_INPUT_LEN, packageSize - curFilePos);
		curWindowNum = curInputLen - WINDOW_SIZE + 1;
		memcpy(packageInput[bufferIdx], &(package[curFilePos]), curInputLen);

		start = clock();
		cudaStreamSynchronize(streams[bufferIdx]);
		
		//Because of unblock cuda process, deal with the 2 iteration eariler cuda output here
		if (iterator > 1) {
			for (unsigned int j = 0; j < MAX_WINDOW_NUM; ++j) {
				if ((resultHost[bufferIdx][j] & P_MINUS) == 0) { // marker found
					indexQ.push_back(j);
				}
			}
			end = clock();
			time += (end - start) * 1000 / CLOCKS_PER_SEC;
			totalDuplicationSize += fingerPrinting(indexQ, &(package[curFilePos - (MAX_WINDOW_NUM << 1)]));
			indexQ.clear();
		}

		RabinHashAsync(kernelInput[bufferIdx], packageInput[bufferIdx], curInputLen, kernelResult[bufferIdx], resultHost[bufferIdx], streams[bufferIdx]);

		bufferIdx ^= 1;
		curFilePos += curWindowNum;
	}

	if (iterator > 1) {
		start = clock();
		cudaDeviceSynchronize();
		for (unsigned int j = 0; j < MAX_WINDOW_NUM; ++j) {
			if ((resultHost[bufferIdx][j] & P_MINUS) == 0) { // marker found
				indexQ.push_back(j);
			}
		}
		end = clock();
		time += (end - start) * 1000 / CLOCKS_PER_SEC;
		totalDuplicationSize += fingerPrinting(indexQ, &(package[curFilePos - MAX_WINDOW_NUM - curWindowNum]));
		indexQ.clear();
	}

	start = clock();
	for (unsigned int j = 0; j < curWindowNum; ++j) {
		if ((resultHost[bufferIdx ^ 1][j] & P_MINUS) == 0) { // marker found
			indexQ.push_back(j);
		}
	}
	end = clock();
	time += (end - start) * 1000 / CLOCKS_PER_SEC;
	totalDuplicationSize += fingerPrinting(indexQ, &(package[curFilePos - curWindowNum]));

	printf("chunking time: %f ms\n", time);
	cudaFree(kernelResult[0]);
	cudaFreeHost(resultHost[0]);
	cudaFree(kernelInput[0]);
	cudaFreeHost(packageInput[0]);
	cudaFree(kernelResult[1]);
	cudaFreeHost(resultHost[1]);
	cudaFree(kernelInput[1]);
	cudaFreeHost(packageInput[1]);
	//cudaFree(kernelInput);
	return totalDuplicationSize;
}

/*
Compute the hash value of chunk, should use sha256 to avoid collision
*/
inline unsigned long long RedundancyEliminator_CUDA::computeChunkHash(char* chunk, unsigned int chunkSize) {
	return hashFunc.Hash(chunk, chunkSize);
	//SHA((unsigned char*)chunk, chunkSize, hashValue);
}

int mod(tuple<unsigned long long, unsigned int> tup, int divisor) {
	return get<0>(tup) % divisor;
}