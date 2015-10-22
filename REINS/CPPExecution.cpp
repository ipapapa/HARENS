#include "CPPExecution.h"
using namespace std;

namespace CPP_Namespace {

	const unsigned int MAX_BUFFER_LEN = 4090 * 512 + WINDOW_SIZE - 1;	//Just to keep it the same as cuda
	const unsigned int MAX_WINDOW_NUM = MAX_BUFFER_LEN - WINDOW_SIZE + 1;

	unsigned int file_length = 0;
	RedundancyEliminator_CPP re;
	ifstream ifs;
	PcapReader fileReader;
	unsigned int cur_file_pos = 0;

	//shared data
	bool readFirstTime = true;
	char overlap[WINDOW_SIZE - 1];
	char* buffer;
	FixedSizedCharArray charArrayBuffer(MAX_BUFFER_LEN);

	unsigned int buffer_len = 0;
	deque<unsigned int> chunking_result;
	//Result
	unsigned int total_duplication_size = 0;
	//Time
	clock_t start_read, start_chunk, start_fin;
	float tot_read = 0, tot_chunk = 0, tot_fin = 0, tot_time = 0;

	int CPPExecute()
	{
		clock_t start, end;

		IO::Print("\n============================ C++ Implementation =============================\n");

		re.SetupRedundancyEliminator_CPP();

		buffer = new char[MAX_BUFFER_LEN];

		bool keepReading = true;
		do {
			keepReading = ReadFile();
			start = clock();
			Chunking();
			Fingerprinting();
			end = clock();
			tot_time += ((float)end - start) * 1000 / CLOCKS_PER_SEC;
		} while (keepReading);

		IO::Print("Found %s of redundency, which is %f % of file\n", 
			IO::InterpretSize(total_duplication_size)
			, (float)total_duplication_size / file_length * 100);

		//delete everything that mallocated before
		delete[] buffer;

		IO::Print("Reading time: %f ms\n", tot_read);
		IO::Print("Chunking time: %f ms\n", tot_chunk);
		IO::Print("Fingerprinting time: %f ms\n", tot_fin);
		IO::Print("Total time: %f ms\n", tot_time);
		IO::Print("=============================================================================\n");

		return 0;
	}

	bool ReadFile() {
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

	void Chunking() {
		start_chunk = clock();
		deque<unsigned int> currentChunkingResult = re.chunking(buffer, buffer_len);
		tot_chunk += ((float)clock() - start_chunk) * 1000 / CLOCKS_PER_SEC;

		chunking_result = currentChunkingResult;
	}

	void Fingerprinting() {
		//When the whole process starts, all chunking results are obsolete, that's the reason fingerprinting part need to check buffer state
		start_fin = clock();

		total_duplication_size += re.fingerPrinting(chunking_result, buffer);
		tot_fin += ((float)clock() - start_fin) * 1000 / CLOCKS_PER_SEC;
	}

	/* old function
	void TestOfRabinHash(char* fileContent, int fileContentLen) {
	RabinHash rh;
	//cout << rh.Hash("kelu") << endl;
	int windowsize = 32;
	unsigned long long p = 32;
	//char currWindow[windowsize];

	set<unsigned long long> hashValueSet;
	map<int, unsigned long long> chunkMap;
	int totalBlocks = 0;
	int chunkNum = 0;
	char* chunk;
	const int CHUNK_SIZE = 32;
	for (int i = 0; i <= fileContentLen - windowsize; i++) {
	chunk = new char[CHUNK_SIZE];
	unsigned long long hashValWindow = rh.Hash(chunk, CHUNK_SIZE);
	hashValueSet.insert(hashValWindow);
	if (hashValWindow % p == 0) { // marker found
	int marker = i;
	chunkMap[marker] = hashValWindow;
	chunkNum++;
	}
	totalBlocks++;
	delete[] chunk;
	}

	cout << "Number of Windows :" << totalBlocks << endl;
	cout << "FingerPrint Map Size :" << hashValueSet.size() << endl;
	cout << "Number of chunks found :" << chunkNum << endl;
	}
	*/
}