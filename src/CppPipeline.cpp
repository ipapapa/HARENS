#include "CppPipeline.h"
using namespace std;

CppPipeline::CppPipeline(): charArrayBuffer(MAX_BUFFER_LEN) {
	re.SetupRedundancyEliminator_CPP();

	buffer = new char*[PAGABLE_BUFFER_NUM];
	for (int i = 0; i < PAGABLE_BUFFER_NUM; ++i) {
		buffer[i] = new char[MAX_BUFFER_LEN];
		bufferLen[i] = 0;
		bufferObsolete[i] = true;
	}

	for (int i = 0; i < RESULT_BUFFER_NUM; ++i) {
		chunkingResultObsolete[i] = true;
	}
}

CppPipeline::~CppPipeline() {
	//delete everything that mallocated before
	for (int i = 0; i < PAGABLE_BUFFER_NUM; ++i)
		delete buffer[i];
	delete[] buffer;
}

int CppPipeline::Execute()
{
	IO::Print("\n================== Pipeline Version of C++ Implementation ===================\n");

	//Create threads 
	thread tReadFile(std::mem_fn(&CppPipeline::ReadFile), this);
	tReadFile.join();
	clock_t start = clock();
	thread tChunking(std::mem_fn(&CppPipeline::Chunking), this);
	thread tFingerprinting(std::mem_fn(&CppPipeline::Fingerprinting), this);

	tChunking.join();
	tFingerprinting.join();

	IO::Print("Found %s of redundency, "
		, IO::InterpretSize(totalDuplicationSize));
	IO::Print("which is %f %% of file\n"
		, (float)totalDuplicationSize / totalFileLen * 100);

	clock_t end = clock();
	IO::Print("Reading time: %f ms\n", timeReading);
	IO::Print("Chunking time: %f ms\n", timeChunkPartitioning);
	IO::Print("Fingerprinting time: %f ms\n", timeChunkHashingAndMatching);
	IO::Print("Total time: %f ms\n", ((float)end - start) * 1000 / CLOCKS_PER_SEC);
	IO::Print("=============================================================================\n");

	return 0;
}

void CppPipeline::Test(double &rate, double &time) {
	thread tReadFile(std::mem_fn(&CppPipeline::ReadFile), this);
	tReadFile.join();
	clock_t start = clock();
	thread tChunking(std::mem_fn(&CppPipeline::Chunking), this);
	thread tFingerprinting(std::mem_fn(&CppPipeline::Fingerprinting), this);

	tChunking.join();
	tFingerprinting.join();
	clock_t end = clock();

	rate = (float)totalDuplicationSize / totalFileLen * 100;
	time = ((float)end - start) * 1000 / CLOCKS_PER_SEC;
}

void CppPipeline::ReadFile() {
	int bufferIdx = 0;
	unsigned int curFilePos = 0;
	int curWindowNum;
	//Read the first part
	unique_lock<mutex> readFileInitLock(bufferMutex[bufferIdx]);
	startReading = clock();
	IO::fileReader->SetupReader(IO::input_file_name);
	IO::fileReader->ReadChunk(charArrayBuffer, MAX_BUFFER_LEN);
	bufferLen[bufferIdx] = charArrayBuffer.GetLen();
	memcpy(buffer[bufferIdx], charArrayBuffer.GetArr(), bufferLen[bufferIdx]);
	totalFileLen += bufferLen[bufferIdx];
		
	memcpy(overlap, &buffer[bufferIdx][bufferLen[bufferIdx] - WINDOW_SIZE + 1], WINDOW_SIZE - 1);	//copy the last window into overlap
	bufferObsolete[bufferIdx] = false;
	readFileInitLock.unlock();
	bufferCond[bufferIdx].notify_one();
	bufferIdx = (bufferIdx + 1) % PAGABLE_BUFFER_NUM;
	timeReading += ((float)clock() - startReading) * 1000 / CLOCKS_PER_SEC;
	//Read the rest
	while (true) {
		unique_lock<mutex> readFileIterLock(bufferMutex[bufferIdx]);
		while (bufferObsolete[bufferIdx] == false) {
			bufferCond[bufferIdx].wait(readFileIterLock);
		}
		startReading = clock();

		IO::fileReader->ReadChunk(charArrayBuffer, MAX_BUFFER_LEN - WINDOW_SIZE + 1);

		if (charArrayBuffer.GetLen() == 0) {
			readFileIterLock.unlock();
			bufferCond[bufferIdx].notify_all();
			break;	//Read nothing
		}
		memcpy(buffer[bufferIdx], overlap, WINDOW_SIZE - 1);	//copy the overlap into current part
		memcpy(&buffer[bufferIdx][WINDOW_SIZE - 1], charArrayBuffer.GetArr(), charArrayBuffer.GetLen());
		bufferLen[bufferIdx] = charArrayBuffer.GetLen() + WINDOW_SIZE - 1;
		totalFileLen += charArrayBuffer.GetLen();
		bufferObsolete[bufferIdx] = false;
		memcpy(overlap, &buffer[bufferIdx][charArrayBuffer.GetLen()], WINDOW_SIZE - 1);	//copy the last window into overlap
		readFileIterLock.unlock();
		bufferCond[bufferIdx].notify_one();
		bufferIdx = (bufferIdx + 1) % PAGABLE_BUFFER_NUM;
		timeReading += ((float)clock() - startReading) * 1000 / CLOCKS_PER_SEC;
	}
	IO::Print("File size: %s\n", IO::InterpretSize(totalFileLen));
	unique_lock<mutex> readFileEndLock(readFileEndMutex);
	readFileEnd = true;
	//In case the other threads stuck in waiting for condition variable
	bufferCond[bufferIdx].notify_all();
}

void CppPipeline::Chunking() {
	int bufferIdx = 0;
	int chunkingResultIdx = 0;
	while (true) {
		unique_lock<mutex> bufferLock(bufferMutex[bufferIdx]);
		if (bufferObsolete[bufferIdx]) {
			unique_lock<mutex> readFileEndLock(readFileEndMutex);
			if (readFileEnd){
				unique_lock<mutex> chunkingEndLock(chunkingEndMutex);
				chunkingEnd = true;
				return;
			}
			readFileEndLock.unlock();
			bufferCond[bufferIdx].wait(bufferLock);
		}

		startChunkPartitioning = clock();
		deque<unsigned int> currentChunkingResult = re.chunking(buffer[bufferIdx], bufferLen[bufferIdx]);
		timeChunkPartitioning += ((float)clock() - startChunkPartitioning) * 1000 / CLOCKS_PER_SEC;
		bufferLock.unlock();
		bufferCond[bufferIdx].notify_one();
		bufferIdx = (bufferIdx + 1) % PAGABLE_BUFFER_NUM;

		unique_lock<mutex> chunkingResultLock(chunkingResultMutex[chunkingResultIdx]);
		while (chunkingResultObsolete[chunkingResultIdx] == false) {
			chunkingResultCond[chunkingResultIdx].wait(chunkingResultLock);
		}
		chunkingResultBuffer[chunkingResultIdx] = currentChunkingResult;
		chunkingResultObsolete[chunkingResultIdx] = false;
		chunkingResultLock.unlock();
		chunkingResultCond[chunkingResultIdx].notify_one();
		chunkingResultIdx = (chunkingResultIdx + 1) % RESULT_BUFFER_NUM;

	}
}

void CppPipeline::Fingerprinting() {
	int bufferIdx = 0;
	int chunkingResultIdx = 0;
	//When the whole process starts, all chunking results are obsolete, that's the reason fingerprinting part need to check buffer state
	while (true) {
		//Get chunking result ready
		unique_lock<mutex> chunkingResultLock(chunkingResultMutex[chunkingResultIdx]);
		while (chunkingResultObsolete[chunkingResultIdx]) {
			unique_lock<mutex> chunkingEndLock(chunkingEndMutex);
			if (chunkingEnd)
				return;
			chunkingEndLock.unlock();
			chunkingResultCond[chunkingResultIdx].wait(chunkingResultLock);
		}

		//Buffer is already ready
		unique_lock<mutex> bufferLock(bufferMutex[bufferIdx]);

		startChunkHashingAndMatching = clock();

		totalDuplicationSize += re.fingerPrinting(chunkingResultBuffer[chunkingResultIdx], buffer[bufferIdx]);
		bufferObsolete[bufferIdx] = true;
		chunkingResultObsolete[chunkingResultIdx] = true;
		bufferLock.unlock();
		bufferCond[bufferIdx].notify_one();
		chunkingResultLock.unlock();
		chunkingResultCond[chunkingResultIdx].notify_one();

		bufferIdx = (bufferIdx + 1) % PAGABLE_BUFFER_NUM;
		chunkingResultIdx = (chunkingResultIdx + 1) % RESULT_BUFFER_NUM;
		timeChunkHashingAndMatching += ((float)clock() - startChunkHashingAndMatching) * 1000 / CLOCKS_PER_SEC;
	}
}