#include "CPP_Main.h"
using namespace std;

namespace CPP_Namespace {

	const unsigned int MAX_BUFFER_LEN = 4090 * 512 + WINDOW_SIZE - 1;	//Just to keep it the same as cuda
	const unsigned int MAX_WINDOW_NUM = MAX_BUFFER_LEN - WINDOW_SIZE + 1;

	unsigned int file_length = 0;
	RedundancyEliminator_CPP re;
	ifstream ifs;
	char* fileName;
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
	float tot_read = 0, tot_chunk = 0, tot_fin = 0;

	int CPP_Main(int argc, char* argv[])
	{
		clock_t start = clock();
		cout << "\n============================ C++ Implementation =============================\n";
		if (argc != 2) {
			cout << "You used " << argc << " variables\n";
			cout << "Usage: " << argv[0] << " <filename>\n";
			system("pause");
			return -1;
		}

		fileName = argv[1];

		re.SetupRedundancyEliminator_CPP();

		buffer = new char[MAX_BUFFER_LEN];

		bool keepReading = true;
		do {
			keepReading = ReadFile();
			Chunking();
			Fingerprinting();
		} while (keepReading);

		cout << "Found " << total_duplication_size << " bytes of redundency, which is " << (float)total_duplication_size / file_length * 100 << " percent of file\n";

		//delete everything that mallocated before
		delete[] buffer;

		clock_t end = clock();
		cout << "Reading time: " << tot_read << " ms\n";
		cout << "Chunking time: " << tot_chunk << " ms\n";
		cout << "Fingerprinting time: " << tot_fin << " ms\n";
		cout << "Total time: " << ((float)end - start) * 1000 / CLOCKS_PER_SEC << " ms\n";
		cout << "=============================================================================\n";

		return 0;
	}

	bool ReadFile() {
		int curWindowNum;
		//Read the first part
		if (readFirstTime) {
			readFirstTime = false;
			start_read = clock();
			if (FILE_FORMAT == PlainText) {
				ifs = ifstream(fileName, ios::in | ios::binary | ios::ate);
				if (!ifs.is_open()) {
					cout << "Can not open file " << fileName << endl;
				}

				file_length = ifs.tellg();
				ifs.seekg(0, ifs.beg);
				cout << "File size: " << file_length / 1024 << " KB\n";
				buffer_len = min(MAX_BUFFER_LEN, file_length - cur_file_pos);
				curWindowNum = buffer_len - WINDOW_SIZE + 1;
				ifs.read(buffer, buffer_len);
				cur_file_pos += curWindowNum;

				return buffer_len == MAX_BUFFER_LEN;
			}
			else if (FILE_FORMAT == Pcap) {
				fileReader.SetupPcapHandle(fileName);
				fileReader.ReadPcapFileChunk(charArrayBuffer, MAX_BUFFER_LEN);
				buffer_len = charArrayBuffer.GetLen();
				memcpy(buffer, charArrayBuffer.GetArr(), buffer_len);
				file_length += buffer_len;

				return buffer_len == MAX_BUFFER_LEN;
			}
			else
				fprintf(stderr, "Unknown file format %s\n", FILE_FORMAT_TEXT[FILE_FORMAT]);

			memcpy(overlap, &buffer[buffer_len - WINDOW_SIZE + 1], WINDOW_SIZE - 1);	//copy the last window into overlap
			tot_read += ((float)clock() - start_read) * 1000 / CLOCKS_PER_SEC;
		}
		else { //Read the rest
			if (FILE_FORMAT == PlainText) {
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
			else if (FILE_FORMAT == Pcap) {
				start_read = clock();
				fileReader.ReadPcapFileChunk(charArrayBuffer, MAX_BUFFER_LEN - WINDOW_SIZE + 1);
				memcpy(buffer, overlap, WINDOW_SIZE - 1);	//copy the overlap into current part
				memcpy(&buffer[WINDOW_SIZE - 1], charArrayBuffer.GetArr(), charArrayBuffer.GetLen());
				buffer_len = charArrayBuffer.GetLen() + WINDOW_SIZE - 1;
				file_length += charArrayBuffer.GetLen();
				memcpy(overlap, &buffer[charArrayBuffer.GetLen()], WINDOW_SIZE - 1);	//copy the last window into overlap
				tot_read += ((float)clock() - start_read) * 1000 / CLOCKS_PER_SEC;

				if (buffer_len != MAX_BUFFER_LEN)
					cout << "File size: " << file_length / 1024 << " KB\n";
				return buffer_len == MAX_BUFFER_LEN;
			}
			else
				fprintf(stderr, "Unknown file format %s\n", FILE_FORMAT_TEXT[FILE_FORMAT]);
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