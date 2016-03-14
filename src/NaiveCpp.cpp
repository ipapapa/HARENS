#include "NaiveCpp.h"
using namespace std;

NaiveCpp::NaiveCpp() : charArrayBuffer(MAX_BUFFER_LEN) {
	re.SetupRedundancyEliminator_CPP();
	buffer = new char[MAX_BUFFER_LEN];
}

NaiveCpp::~NaiveCpp() {
	//delete everything that mallocated before
	delete[] buffer;
}

int	NaiveCpp::Execute()
{
	IO::Print("\n============================ C++ Implementation =============================\n");
	
	clock_t start, end;

	bool keepReading = true;
	do {
		keepReading = ReadFile();
		start = clock();
		Chunking();
		Fingerprinting();
		end = clock();
		timeTotal += ((float)end - start) * 1000 / CLOCKS_PER_SEC;
	} while (keepReading);

	IO::Print("Found %s of redundency, "
		, IO::InterpretSize(totalDuplicationSize));
	IO::Print("which is %f %% of file\n"
		, (float)totalDuplicationSize / totalFileLen * 100);

	IO::Print("Reading time: %f ms\n", timeReading);
	IO::Print("Chunking time: %f ms\n", timeChunkPartitioning);
	IO::Print("Fingerprinting time: %f ms\n", timeChunkHashingAndMatching);
	IO::Print("Total time: %f ms\n", timeTotal);
	IO::Print("=============================================================================\n");

	return 0;
}

void NaiveCpp::Test(double &rate, double &time) {
	clock_t start, end;

	bool keepReading = true;
	do {
		keepReading = ReadFile();
		start = clock();
		Chunking();
		Fingerprinting();
		end = clock();
		timeTotal += ((float)end - start) * 1000 / CLOCKS_PER_SEC;
	} while (keepReading);

	rate = (float)totalDuplicationSize / totalFileLen * 100;
	time = timeTotal;
}

bool NaiveCpp::ReadFile() {
	int curWindowNum;
	//Read the first part
	if (readFirstTime) {
		readFirstTime = false;
		startReading = clock();

		IO::fileReader->SetupReader(IO::input_file_name);
		IO::fileReader->ReadChunk(charArrayBuffer, MAX_BUFFER_LEN);
		bufferLen = charArrayBuffer.GetLen();
		memcpy(buffer, charArrayBuffer.GetArr(), bufferLen);
		totalFileLen += bufferLen;
		if (bufferLen != MAX_BUFFER_LEN)
			IO::Print("File size: %s\n", IO::InterpretSize(totalFileLen));
		return bufferLen == MAX_BUFFER_LEN;

		//copy the last window into overlap
		memcpy(overlap, &buffer[bufferLen - WINDOW_SIZE + 1], WINDOW_SIZE - 1);
		timeReading += ((float)clock() - startReading) * 1000 / CLOCKS_PER_SEC;
	}
	else { //Read the rest
		startReading = clock();
		IO::fileReader->ReadChunk(charArrayBuffer, MAX_BUFFER_LEN - WINDOW_SIZE + 1);
		memcpy(buffer, overlap, WINDOW_SIZE - 1);	//copy the overlap into current part
		memcpy(&buffer[WINDOW_SIZE - 1], charArrayBuffer.GetArr(), charArrayBuffer.GetLen());
		bufferLen = charArrayBuffer.GetLen() + WINDOW_SIZE - 1;
		totalFileLen += charArrayBuffer.GetLen();
		//copy the last window into overlap
		memcpy(overlap, &buffer[charArrayBuffer.GetLen()], WINDOW_SIZE - 1);	
		timeReading += ((float)clock() - startReading) * 1000 / CLOCKS_PER_SEC;
		if (bufferLen != MAX_BUFFER_LEN)
			IO::Print("File size: %s\n", IO::InterpretSize(totalFileLen));
		return bufferLen == MAX_BUFFER_LEN;
	}
}

void NaiveCpp::Chunking() {
	startChunkPartitioning = clock();
	deque<unsigned int> currentChunkingResult = re.chunking(buffer, bufferLen);
	timeChunkPartitioning += ((float)clock() - startChunkPartitioning) * 1000 / CLOCKS_PER_SEC;

	chunkingResultBuffer = currentChunkingResult;
}

void NaiveCpp::Fingerprinting() {
	/*When the whole process starts, all chunking results are obsolete, 
	* that's the reason fingerprinting part need to check buffer state*/
	startChunkHashingAndMatching = clock();

	totalDuplicationSize += re.fingerPrinting(chunkingResultBuffer, buffer);
	timeChunkHashingAndMatching += ((float)clock() - startChunkHashingAndMatching) * 1000 / CLOCKS_PER_SEC;
}
