#include "HashCollisionTest.h"
using namespace std;

HashCollisionTest::HashCollisionTest() : charArrayBuffer(MAX_BUFFER_LEN) {
	re.SetupRedundancyEliminator_CPP();
	buffer = new char[MAX_BUFFER_LEN];
}

HashCollisionTest::~HashCollisionTest() {
	//delete everything that mallocated before
	delete[] buffer;
}

int	HashCollisionTest::Execute()
{
	clock_t start, end;

	IO::Print("\n============================ C++ Implementation =============================\n");

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

bool HashCollisionTest::ReadFile() {
	int curWindowNum;
	//Read the first part
	if (readFirstTime) {
		readFirstTime = false;
		start_read = clock();
		if (IO::FILE_FORMAT == PlainText) {
			ifs = ifstream(IO::input_file_name, ios::in | ios::binary | ios::ate);
			if (!ifs.is_open()) {
				printf("Can not open file %s\n", IO::input_file_name);
				return false;
			}

			file_length = ifs.tellg();
			ifs.seekg(0, ifs.beg);
			IO::Print("File Length: %s \n", IO::InterpretSize(file_length));
			buffer_len = min(MAX_BUFFER_LEN, file_length - cur_file_pos);
			curWindowNum = buffer_len - WINDOW_SIZE + 1;
			ifs.read(buffer, buffer_len);
			cur_file_pos += curWindowNum;

			return buffer_len == MAX_BUFFER_LEN;
		}
		else if (IO::FILE_FORMAT == Pcap) {
			fileReader.SetupPcapHandle(IO::input_file_name);
			fileReader.ReadPcapFileChunk(charArrayBuffer, MAX_BUFFER_LEN);
			buffer_len = charArrayBuffer.GetLen();
			memcpy(buffer, charArrayBuffer.GetArr(), buffer_len);
			file_length += buffer_len;

			return buffer_len == MAX_BUFFER_LEN;
		}
		else
			fprintf(stderr, "Unknown file format\n");

		memcpy(overlap, &buffer[buffer_len - WINDOW_SIZE + 1], WINDOW_SIZE - 1);	//copy the last window into overlap
		tot_read += ((float)clock() - start_read) * 1000 / CLOCKS_PER_SEC;
	}
	else { //Read the rest
		if (IO::FILE_FORMAT == PlainText) {
			start_read = clock();
			buffer_len = min(MAX_BUFFER_LEN, file_length - cur_file_pos + WINDOW_SIZE - 1);
			curWindowNum = buffer_len - WINDOW_SIZE + 1;
			memcpy(buffer, overlap, WINDOW_SIZE - 1);	//copy the overlap into current part
			ifs.read(&buffer[WINDOW_SIZE - 1], curWindowNum);
			memcpy(overlap, &buffer[curWindowNum], WINDOW_SIZE - 1);	//copy the last window into overlap
			cur_file_pos += curWindowNum;
			tot_read += ((float)clock() - start_read) * 1000 / CLOCKS_PER_SEC;

			if (buffer_len != MAX_BUFFER_LEN)
				ifs.close();
			return buffer_len == MAX_BUFFER_LEN;
		}
		else if (IO::FILE_FORMAT == Pcap) {
			start_read = clock();
			fileReader.ReadPcapFileChunk(charArrayBuffer, MAX_BUFFER_LEN - WINDOW_SIZE + 1);
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
		else
			fprintf(stderr, "Unknown file format\n");
	}
}

void HashCollisionTest::Chunking() {
	start_chunk = clock();
	deque<unsigned int> currentChunkingResult = re.chunking(buffer, buffer_len);
	tot_chunk += ((float)clock() - start_chunk) * 1000 / CLOCKS_PER_SEC;

	chunking_result = currentChunkingResult;
}

void HashCollisionTest::Fingerprinting() {
	//When the whole process starts, all chunking results are obsolete, that's the reason fingerprinting part need to check buffer state
	start_fin = clock();

	total_duplication_size += re.fingerPrinting(chunking_result, buffer);
	tot_fin += ((float)clock() - start_fin) * 1000 / CLOCKS_PER_SEC;
}
