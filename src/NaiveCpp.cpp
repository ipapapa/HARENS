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
		tot_time += ((float)end - start) * 1000 / CLOCKS_PER_SEC;
	} while (keepReading);

	IO::Print("Found %s of redundency, which is %f %% of file\n", 
		IO::InterpretSize(total_duplication_size)
		, (float)total_duplication_size / file_length * 100);

	IO::Print("Reading time: %f ms\n", tot_read);
	IO::Print("Chunking time: %f ms\n", tot_chunk);
	IO::Print("Fingerprinting time: %f ms\n", tot_fin);
	IO::Print("Total time: %f ms\n", tot_time);
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
		tot_time += ((float)end - start) * 1000 / CLOCKS_PER_SEC;
	} while (keepReading);

	rate = (float)total_duplication_size / file_length * 100;
	time = tot_time;
}

bool NaiveCpp::ReadFile() {
	int curWindowNum;
	//Read the first part
	if (readFirstTime) {
		readFirstTime = false;
		start_read = clock();

		IO::fileReader->SetupReader(IO::input_file_name[0]);
		IO::fileReader->ReadChunk(charArrayBuffer, MAX_BUFFER_LEN);
		buffer_len = charArrayBuffer.GetLen();
		memcpy(buffer, charArrayBuffer.GetArr(), buffer_len);
		file_length += buffer_len;
		return buffer_len == MAX_BUFFER_LEN;

		memcpy(overlap, &buffer[buffer_len - WINDOW_SIZE + 1], WINDOW_SIZE - 1);	//copy the last window into overlap
		tot_read += ((float)clock() - start_read) * 1000 / CLOCKS_PER_SEC;
	}
	else { //Read the rest
		start_read = clock();
		IO::fileReader->ReadChunk(charArrayBuffer, MAX_BUFFER_LEN - WINDOW_SIZE + 1);
		memcpy(buffer, overlap, WINDOW_SIZE - 1);	//copy the overlap into current part
		memcpy(&buffer[WINDOW_SIZE - 1], charArrayBuffer.GetArr(), charArrayBuffer.GetLen());
		buffer_len = charArrayBuffer.GetLen() + WINDOW_SIZE - 1;
		file_length += charArrayBuffer.GetLen();
		memcpy(overlap, &buffer[charArrayBuffer.GetLen()], WINDOW_SIZE - 1);	//copy the last window into overlap
		tot_read += ((float)clock() - start_read) * 1000 / CLOCKS_PER_SEC;
		if (buffer_len != MAX_BUFFER_LEN)
			IO::Print("File size: %s\n", IO::InterpretSize(file_length));
		return buffer_len == MAX_BUFFER_LEN;
	}
}

void NaiveCpp::Chunking() {
	start_chunk = clock();
	deque<unsigned int> currentChunkingResult = re.chunking(buffer, buffer_len);
	tot_chunk += ((float)clock() - start_chunk) * 1000 / CLOCKS_PER_SEC;

	chunking_result = currentChunkingResult;
}

void NaiveCpp::Fingerprinting() {
	//When the whole process starts, all chunking results are obsolete, that's the reason fingerprinting part need to check buffer state
	start_fin = clock();

	total_duplication_size += re.fingerPrinting(chunking_result, buffer);
	tot_fin += ((float)clock() - start_fin) * 1000 / CLOCKS_PER_SEC;
}
