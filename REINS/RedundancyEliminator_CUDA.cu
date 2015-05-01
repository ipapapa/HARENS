#include "RedundancyEliminator_CUDA.h"

#define CharArraySize(array) strlen(array)

__device__ void SetResultElement(ulong* subResult, uint idx, ulong resultPoint) {
	subResult[idx] = resultPoint;
}

__device__ ulong* GetSubResult(ulong* result, uint blockNum) {
	return &(result[blockNum * THREAD_PER_BLOCK]);
}

__device__ char* GetSubStr(char *str, uint blockNum) {
	return &(str[blockNum * THREAD_PER_BLOCK]);
}

__device__ uint GetUIntFromStr(char* strs, uint idx) {
	return (strs[idx] << 24) | (strs[idx + 1] << 16) | (strs[idx + 2] << 8) | (strs[idx + 3]);
}

__device__ ulong GetULongFromStr(char* strs, uint idx) {
	/*ulong result;
	memcpy((void*)&result, strs, BYTES_IN_ULONG);
	return result;
	*/
	
	ulong result = strs[idx];
	for (int i = 1; i < 8; ++i)
	result = (result << 8) | strs[idx + i];
	return result;
	
}

__device__ char GetChar(char* subStr, uint idx) {
	return subStr[idx];
}

__global__ void Hash(const ulong *TA, const ulong *TB, const ulong *TC, const ulong * TD,
	char *str, const uint windowsNum, ulong *result/*, int *debug*/) {
	if (blockDim.x * blockIdx.x + threadIdx.x >= windowsNum)
		return;

	uint blockNum = blockIdx.x;
	char* subStr = GetSubStr(str, blockNum);
	ulong* subResult = GetSubResult(result, blockNum);

	__shared__ char s_str[THREAD_PER_BLOCK + 3];
	__shared__ char s_str_shift[THREAD_PER_BLOCK + 7];
	uint threadNum = threadIdx.x;

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

	ulong resultPoint = GetULongFromStr(s_str_shift, threadNum);
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
void RedundancyEliminator_CUDA::addNewChunk(uchar* hashValue, char* chunk, uint chunkSize) {
	uchar* to_be_del = circHash->Add(hashValue);
	if (to_be_del != NULL)
		delete[] to_be_del;
	/*fstream file(hashValue.c_str(), std::fstream::in | std::fstream::out);
	file << chunk;
	file.close();*/
}

/*take a kernel global memory address and the size as input*/
//deque<uint> RedundancyEliminator_CUDA::chunking(char* input, uint inputLen, ulong *resultHost) {
//	deque<uint> indexQ = deque<uint>();
//	ulong *kernelResult;
//
//	const int streamNum = 3;
//	cudaStream_t streams[streamNum];
//	for (int i = 0; i < streamNum; ++i)
//		cudaStreamCreate(&(streams[i]));
//
//	uint numOfWindows = (inputLen - WINDOW_SIZE + 1);
//	uint numOfWindowsInLoop;
//	uint offset;
//	char* kernelInput;
//
//	//debug
//	/*int* debugHost = new int[BLOCK_NUM * THREAD_PER_BLOCK * sizeof(int) * 2];
//	int* debugDevice;
//	cudaMalloc((void**)&debugDevice, BLOCK_NUM * THREAD_PER_BLOCK * sizeof(int) * 2);*/
//
//	cudaMalloc((void**)&kernelInput, inputLen);
//	cudaMalloc((void**)&kernelResult, numOfWindows * BYTES_IN_ULONG);
//	for (int i = 0; i < streamNum; ++i) {
//		offset = numOfWindows / streamNum * i;
//		numOfWindowsInLoop = min(numOfWindows / streamNum, numOfWindows - offset);
//		uint resultSize = numOfWindowsInLoop * BYTES_IN_ULONG;
//
//		cudaMemcpyAsync((&(kernelInput[offset])), &(input[offset]), 
//			numOfWindowsInLoop + WINDOW_SIZE - 1, cudaMemcpyHostToDevice, streams[i]); 
//		
//		Hash <<<BLOCK_NUM, THREAD_PER_BLOCK, 0, streams[i]>>> (kernelTA, kernelTB, kernelTC, kernelTD,
//			&(kernelInput[offset]), numOfWindowsInLoop, &kernelResult[offset]/*, debugDevice*/);
//		cudaDeviceSynchronize();
//		// check for errors
//		/*cudaError_t error = cudaGetLastError();
//		if (error != cudaSuccess) {
//			fprintf(stderr, "ERROR1: %s \n", cudaGetErrorString(error));
//		}*/
//		cudaMemcpyAsync(&(resultHost[offset]), &kernelResult[offset], resultSize, cudaMemcpyDeviceToHost, streams[i]); 
//
//		//debug
//		/*cudaMemcpy(debugHost, debugDevice, BLOCK_NUM * THREAD_PER_BLOCK * sizeof(int) * 2, cudaMemcpyDeviceToHost);
//		if (i == 1) {
//			for (int j = 0; j < BLOCK_NUM * THREAD_PER_BLOCK; ++j)
//				cout << debugHost[2 * j] << '\t' << debugHost[2 * j + 1] << endl;
//		}*/
//
//		/*cout << offset << '\t' << resultHost[offset] << endl;
//		cout << offset + numOfWindowsInLoop - 1 << '\t' << resultHost[offset + numOfWindowsInLoop - 1] << endl;*/
//		for (uint j = offset; j < offset + numOfWindowsInLoop; ++j) {
//			if ((resultHost[j] & P_MINUS) == 0) { // marker found
//				indexQ.push_back(j);
//			}
//		}
//	}
//
//	//debug
//	/*delete[] debugHost;
//	cudaFree(debugDevice);*/
//
//	cudaFree(kernelInput);
//	cudaFree(kernelResult);
//	for (int i = 0; i < streamNum; ++i)
//		cudaStreamDestroy(streams[i]);
//
//	return indexQ;
//}

void RedundancyEliminator_CUDA::ChunkHashing(uint* indices, int indicesNum, char* package, 
	char** chunkList, uchar** chunkHashValueList, uint* chunkLenList) {
	uint prevIdx = 0;
	for (int i = 0; i < indicesNum; ++i) {
		if (prevIdx == 0) {
			prevIdx = indices[i];
			continue;
		}
		chunkLenList[i - 1] = indices[i] - prevIdx;
		chunkList[i - 1] = &(package[prevIdx]);

		//Mind! never use sizeof(chunk) to check the chunk size
		computeChunkHash(chunkList[i - 1], chunkLenList[i - 1], chunkHashValueList[i - 1]);
		prevIdx = indices[i];
	}
}

uint RedundancyEliminator_CUDA::ChunkMatching(deque<uchar*> &hashValues, deque<tuple<char*, uint>> &chunks) {
	uint duplicationSize = 0;
	deque<uchar*>::const_iterator hashValueIter = hashValues.begin();
	deque<tuple<char*, uint>>::const_iterator chunksIter = chunks.begin();
	while (hashValueIter != hashValues.end()) {
		if (circHash->Find(*hashValueIter)) {
			duplicationSize += get<1>(*chunksIter);
		}
		else {
			addNewChunk(*hashValueIter, get<0>(*chunksIter), get<1>(*chunksIter));
		}
		++hashValueIter;
		++chunksIter;
	}
	return duplicationSize;
}

void RedundancyEliminator_CUDA::ChunkHashingAscynWithCircularQueue(uint* indices, int indicesNum, char* package,
	CircularUcharArrayQueue &chunkHashValueQ, CircularUintQueue &chunkLenQ, mutex &chunkMutex) {
	//uint duplicationSize = 0;
	uint prevIdx = 0;
	char* chunk;
	uint chunkLen;
	uchar* chunkHashValue;
	for (int i = 0; i < indicesNum; ++i) {
		if (prevIdx == 0) {
			prevIdx = indices[i];
			continue;
		}
		chunk = &(package[prevIdx]);
		chunkLen = indices[i] - prevIdx;
		chunkLenQ.Push(chunkLen);
		chunkHashValue = chunkHashValueQ.LatentPush();

		//Mind! never use sizeof(chunk) to check the chunk size
		//chunkMutex.lock();
		computeChunkHash(chunk, chunkLen, chunkHashValue);
		//chunkMutex.unlock();
		prevIdx = indices[i];
	}
}

void RedundancyEliminator_CUDA::ChunkHashingAscyn(uint* indices, int indicesNum, char* package, 
	uchar* chunkHashValueList, uint* chunkLenList, mutex &chunkMutex) {
	//uint duplicationSize = 0;
	uint prevIdx = 0;
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
		computeChunkHash(chunk, chunkLenList[i - 1], &chunkHashValueList[(i - 1) * SHA_DIGEST_LENGTH]);
		chunkMutex.unlock();
		prevIdx = indices[i];
	}
}

uint RedundancyEliminator_CUDA::fingerPrinting(deque<uint> indexQ, char* package) {
	/*deque<uchar*> hashValues;
	deque<tuple<char*, uint>> chunks;
	ChunkHashing(indexQ, package, hashValues, chunks);
	return ChunkMatching(hashValues, chunks);*/
	uint duplicationSize = 0;
	uint prevIdx = 0;
	char* chunk;
	for (deque<uint>::const_iterator iter = indexQ.begin(); iter != indexQ.end(); ++iter) {
		if (prevIdx == 0) {
			prevIdx = *iter;
			continue;
		}
		uint chunkLen = *iter - prevIdx;
		chunk = &(package[prevIdx]);

		//Mind! never use sizeof(chunk) to check the chunk size
		uchar* chunkHash = new uchar[SHA_DIGEST_LENGTH];
		computeChunkHash(chunk, chunkLen, chunkHash);
		if (circHash->Find(chunkHash)) { //find duplications
			duplicationSize += chunkLen;
		}
		else {
			addNewChunk(chunkHash, chunk, chunkLen);
		}
		prevIdx = *iter;
	}
	//system("pause");
	return duplicationSize;
}

void RedundancyEliminator_CUDA::RabinHashAsync(char* inputKernel, char* inputHost, uint inputLen, ulong* resultKernel, ulong* resultHost, cudaStream_t stream) {
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
uint RedundancyEliminator_CUDA::eliminateRedundancy(char* package, uint packageSize) {
	/*char *kernelInput;
	cudaMalloc((void**)&kernelInput, MAX_KERNEL_INPUT_LEN);*/
	uint totalDuplicationSize = 0;
	deque<uint> indexQ;
	char* packageInput[2];
	char* kernelInput[2];
	ulong* resultHost[2];
	ulong *kernelResult[2];
	clock_t start;
	clock_t end;
	double time = 0;

	const uint MAX_WINDOW_NUM = MAX_KERNEL_INPUT_LEN - WINDOW_SIZE + 1;
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
	uint curInputLen = MAX_KERNEL_INPUT_LEN, curWindowNum, curFilePos = 0;
	start = clock();
	for (int iterator = 0; curInputLen == MAX_KERNEL_INPUT_LEN; ++iterator) {
		curInputLen = min(MAX_KERNEL_INPUT_LEN, packageSize - curFilePos);
		curWindowNum = curInputLen - WINDOW_SIZE + 1;
		memcpy(packageInput[bufferIdx], &(package[curFilePos]), curInputLen);

		cudaStreamSynchronize(streams[bufferIdx]);
		
		//Because of unblock cuda process, deal with the 2 iteration eariler cuda output here
		if (iterator > 1) {
			for (uint j = 0; j < MAX_WINDOW_NUM; ++j) {
				if ((resultHost[bufferIdx][j] & P_MINUS) == 0) { // marker found
					indexQ.push_back(j);
				}
			}
			totalDuplicationSize += fingerPrinting(indexQ, &(package[curFilePos - (MAX_WINDOW_NUM << 1)]));
			indexQ.clear();
		}

		RabinHashAsync(kernelInput[bufferIdx], packageInput[bufferIdx], curInputLen, kernelResult[bufferIdx], resultHost[bufferIdx], streams[bufferIdx]);

		bufferIdx ^= 1;
		curFilePos += curWindowNum;
	}

	cudaDeviceSynchronize();
	for (uint j = 0; j < MAX_WINDOW_NUM; ++j) {
		if ((resultHost[bufferIdx][j] & P_MINUS) == 0) { // marker found
			indexQ.push_back(j);
		}
	}
	totalDuplicationSize += fingerPrinting(indexQ, &(package[curFilePos - MAX_WINDOW_NUM - curWindowNum]));
	indexQ.clear();
	for (uint j = 0; j < curWindowNum; ++j) {
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
inline void RedundancyEliminator_CUDA::computeChunkHash(char* chunk, uint chunkSize, uchar* hashValue) {
	SHA((uchar*)chunk, chunkSize, hashValue);
}
