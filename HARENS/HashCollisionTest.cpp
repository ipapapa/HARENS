#include "HashCollisionTest.h"
using namespace std;

HashCollisionTest::HashCollisionTest(bool isSha1Used, bool isCollisionCheck) 
	: charArrayBuffer(MAX_BUFFER_LEN) {
	this->isSha1Used = isSha1Used;
	this->isCollisionCheck = isCollisionCheck;
	re.SetupRedundancyEliminator_CPP_CollisionTest();
	buffer = new char[MAX_BUFFER_LEN];
}

HashCollisionTest::~HashCollisionTest() {
	//delete everything that mallocated before
	delete[] buffer;
}

int	HashCollisionTest::Execute()
{
	IO::Print("\n================= Collision Test Using C++ Implementation ==================\n");
	if (isSha1Used)
		IO::Print("SHA1 applied\n");
	else
		IO::Print("Rabin hash applied\n");

	bool keepReading = true;
	do {
		keepReading = ReadFile();
		Chunking();
		Fingerprinting();
	} while (keepReading);

	if (isCollisionCheck) {
		IO::Print("Found %s of false report with in %s of redundancy\
			, which is %f %% of all reported redundancy\n"
			, IO::InterpretSize(totalFalseReportSize)
			, IO::InterpretSize(totalDuplicationSize)
			, (float)totalFalseReportSize / totalDuplicationSize * 100);
	}

	IO::Print("Total chunk hashing and matching time: %f ms\n", totFin);
	IO::Print("=============================================================================\n");

	return 0;
}

bool HashCollisionTest::ReadFile() {
	int curWindowNum;
	//Read the first part
	if (readFirstTime) {
		readFirstTime = false;
		if (IO::FILE_FORMAT == PlainText) {
			ifs = ifstream(IO::input_file_name, ios::in | ios::binary | ios::ate);
			if (!ifs.is_open()) {
				printf("Can not open file %s\n", IO::input_file_name);
				return false;
			}

			fileLength = ifs.tellg();
			ifs.seekg(0, ifs.beg);
			IO::Print("File Length: %s \n", IO::InterpretSize(fileLength));
			bufferLen = min(MAX_BUFFER_LEN, fileLength - curFilePos);
			curWindowNum = bufferLen - WINDOW_SIZE + 1;
			ifs.read(buffer, bufferLen);
			curFilePos += curWindowNum;

			return bufferLen == MAX_BUFFER_LEN;
		}
		else if (IO::FILE_FORMAT == Pcap) {
			fileReader.SetupPcapHandle(IO::input_file_name);
			fileReader.ReadPcapFileChunk(charArrayBuffer, MAX_BUFFER_LEN);
			bufferLen = charArrayBuffer.GetLen();
			memcpy(buffer, charArrayBuffer.GetArr(), bufferLen);
			fileLength += bufferLen;

			return bufferLen == MAX_BUFFER_LEN;
		}
		else
			fprintf(stderr, "Unknown file format\n");

		memcpy(overlap, &buffer[bufferLen - WINDOW_SIZE + 1], WINDOW_SIZE - 1);	//copy the last window into overlap	
	}
	else { //Read the rest
		if (IO::FILE_FORMAT == PlainText) {
			bufferLen = min(MAX_BUFFER_LEN, fileLength - curFilePos + WINDOW_SIZE - 1);
			curWindowNum = bufferLen - WINDOW_SIZE + 1;
			memcpy(buffer, overlap, WINDOW_SIZE - 1);	//copy the overlap into current part
			ifs.read(&buffer[WINDOW_SIZE - 1], curWindowNum);
			memcpy(overlap, &buffer[curWindowNum], WINDOW_SIZE - 1);	//copy the last window into overlap
			curFilePos += curWindowNum;

			if (bufferLen != MAX_BUFFER_LEN)
				ifs.close();
			return bufferLen == MAX_BUFFER_LEN;
		}
		else if (IO::FILE_FORMAT == Pcap) {
			fileReader.ReadPcapFileChunk(charArrayBuffer, MAX_BUFFER_LEN - WINDOW_SIZE + 1);
			memcpy(buffer, overlap, WINDOW_SIZE - 1);	//copy the overlap into current part
			memcpy(&buffer[WINDOW_SIZE - 1], charArrayBuffer.GetArr(), charArrayBuffer.GetLen());
			bufferLen = charArrayBuffer.GetLen() + WINDOW_SIZE - 1;
			fileLength += charArrayBuffer.GetLen();
			memcpy(overlap, &buffer[charArrayBuffer.GetLen()], WINDOW_SIZE - 1);	//copy the last window into overlap
			
			if (bufferLen != MAX_BUFFER_LEN)
				IO::Print("File size: %s\n", IO::InterpretSize(fileLength));
			return bufferLen == MAX_BUFFER_LEN;
		}
		else
			fprintf(stderr, "Unknown file format\n");
	}
}

void HashCollisionTest::Chunking() {
	deque<unsigned int> currentChunkingResult = re.chunking(buffer, bufferLen);

	chunkingResult = currentChunkingResult;
}

void HashCollisionTest::Fingerprinting() {
	//When the whole process starts, all chunking results are obsolete, that's the reason fingerprinting part need to check buffer state
	start_fin = clock();
	
	if (isCollisionCheck) {
		tuple<int, int> result;
		if (isSha1Used)
			result = re.SHA1FingerPrintingWithCollisionCheck(chunkingResult, buffer);
		else
			result = re.RabinFingerPrintingWithCollisionCheck(chunkingResult, buffer);
		totalDuplicationSize += get<0>(result);
		totalFalseReportSize += get<1>(result);
	}
	else {
		if (isSha1Used)
			totalDuplicationSize += re.SHA1FingerPrinting(chunkingResult, buffer);
		else
			totalDuplicationSize += re.RabinFingerPrinting(chunkingResult, buffer);
	}

	totFin += ((float)clock() - start_fin) * 1000 / CLOCKS_PER_SEC;
}
