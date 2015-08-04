#include "CPP_Pipeline_Main.h"
using namespace std;

namespace CPP_Pipeline_Namespace {

	const int BUFFER_NUM = 2;
	const unsigned int MAX_BUFFER_LEN = 4090 * 512 + WINDOW_SIZE - 1;	//Just to keep it the same as cuda
	const unsigned int MAX_WINDOW_NUM = MAX_BUFFER_LEN - WINDOW_SIZE + 1;

	unsigned int file_length = 0;
	RedundancyEliminator_CPP re;
	ifstream ifs;
	char* fileName;
	//syncronize
	mutex buffer_mutex[2], chunking_result_mutex[2];		//lock for buffer and chunking_result
	bool buffer_obsolete[2] = { true, true }, chunking_result_obsolete[2] = { true, true };	//state of buffer and chunking_result
	bool no_more_input = false;
	bool no_more_chunking_result = false;
	//shared data
	char overlap[WINDOW_SIZE - 1];
	char** buffer = new char*[BUFFER_NUM];	//two buffers
	FixedSizedCharArray charArrayBuffer(MAX_BUFFER_LEN);

	unsigned int buffer_len[] = { 0, 0 };
	deque<unsigned int>* chunking_result = new deque<unsigned int>[2];
	//Result
	unsigned int total_duplication_size = 0;
	//Time
	clock_t start_read, start_chunk, start_fin;
	float tot_read = 0, tot_chunk = 0, tot_fin = 0;

	int CPP_Pipeline_Main(int argc, char* argv[])
	{
		clock_t start = clock();
		cout << "\n================== Pipeline Version of C++ Implementation ===================\n";
		if (argc != 2) {
			cout << "You used " << argc << " variables\n";
			cout << "Usage: " << argv[0] << " <filename>\n";
			system("pause");
			return -1;
		}

		fileName = argv[1];

		re.SetupRedundancyEliminator_CPP();

		for (int i = 0; i < BUFFER_NUM; ++i)
			buffer[i] = new char[MAX_BUFFER_LEN];

		//Create threads 
		thread tReadFile(ReadFile);
		thread tChunking(Chunking);
		thread tFingerprinting(Fingerprinting);

		tReadFile.join();
		tChunking.join();
		tFingerprinting.join();

		cout << "Found " << total_duplication_size << " bytes of redundency, which is " << (float)total_duplication_size / file_length * 100 << " percent of file\n";

		//delete everything that mallocated before
		for (int i = 0; i < BUFFER_NUM; ++i)
			delete[] buffer[i];
		delete[] buffer;

		clock_t end = clock();
		cout << "Reading time: " << tot_read << " ms\n";
		cout << "Chunking time: " << tot_chunk << " ms\n";
		cout << "Fingerprinting time: " << tot_fin << " ms\n";
		cout << "Total time: " << ((float)end - start) * 1000 / CLOCKS_PER_SEC << " ms\n";
		cout << "=============================================================================\n";

		return 0;
	}

	void ReadFile() {
		int bufferIdx = 0;
		unsigned int curFilePos = 0;
		int curWindowNum;
		PcapReader fileReader;
		//Read the first part
		buffer_mutex[bufferIdx].lock();
		start_read = clock();
		if (FILE_FORMAT == PlainText) {
			ifs = ifstream(fileName, ios::in | ios::binary | ios::ate);
			if (!ifs.is_open()) {
				cout << "Can not open file " << fileName << endl;
			}

			file_length = ifs.tellg();
			ifs.seekg(0, ifs.beg);
			cout << "File size: " << file_length / 1024 << " KB\n";
			buffer_len[bufferIdx] = min(MAX_BUFFER_LEN, file_length - curFilePos);
			curWindowNum = buffer_len[bufferIdx] - WINDOW_SIZE + 1;
			ifs.read(buffer[bufferIdx], buffer_len[bufferIdx]);
			curFilePos += curWindowNum;
		}
		else if (FILE_FORMAT == Pcap) {
			fileReader.SetupPcapHandle(fileName);
			fileReader.ReadPcapFileChunk(charArrayBuffer, MAX_BUFFER_LEN);
			buffer_len[bufferIdx] = charArrayBuffer.GetLen();
			memcpy(buffer[bufferIdx], charArrayBuffer.GetArr(), buffer_len[bufferIdx]);
			file_length += buffer_len[bufferIdx];
		}
		else
			fprintf(stderr, "Unknown file format %s\n", FILE_FORMAT_TEXT[FILE_FORMAT]);
		
		memcpy(overlap, &buffer[bufferIdx][buffer_len[bufferIdx] - WINDOW_SIZE + 1], WINDOW_SIZE - 1);	//copy the last window into overlap
		buffer_obsolete[bufferIdx] = false;
		buffer_mutex[bufferIdx].unlock();
		bufferIdx ^= 1;
		tot_read += ((float)clock() - start_read) * 1000 / CLOCKS_PER_SEC;
		//Read the rest
		if (FILE_FORMAT == PlainText) {
			while (curWindowNum == MAX_WINDOW_NUM) {
				buffer_mutex[bufferIdx].lock();
				while (!buffer_obsolete[bufferIdx]) {
					buffer_mutex[bufferIdx].unlock();
					this_thread::sleep_for(chrono::microseconds(500));
					buffer_mutex[bufferIdx].lock();
				}
				start_read = clock();
				buffer_len[bufferIdx] = min(MAX_BUFFER_LEN, file_length - curFilePos + WINDOW_SIZE - 1);
				curWindowNum = buffer_len[bufferIdx] - WINDOW_SIZE + 1;
				memcpy(buffer[bufferIdx], overlap, WINDOW_SIZE - 1);	//copy the overlap into current part
				ifs.read(&buffer[bufferIdx][WINDOW_SIZE - 1], curWindowNum);
				buffer_obsolete[bufferIdx] = false;
				memcpy(overlap, &buffer[bufferIdx][curWindowNum], WINDOW_SIZE - 1);	//copy the last window into overlap
				buffer_mutex[bufferIdx].unlock();
				bufferIdx ^= 1;
				curFilePos += curWindowNum;
				tot_read += ((float)clock() - start_read) * 1000 / CLOCKS_PER_SEC;
			}
			ifs.close();
		}
		else if (FILE_FORMAT == Pcap) {
			while (true) {
				buffer_mutex[bufferIdx].lock();
				while (!buffer_obsolete[bufferIdx]) {
					buffer_mutex[bufferIdx].unlock();
					this_thread::sleep_for(chrono::microseconds(500));
					buffer_mutex[bufferIdx].lock();
				}
				start_read = clock();
				fileReader.ReadPcapFileChunk(charArrayBuffer, MAX_BUFFER_LEN - WINDOW_SIZE + 1);
				
				if (charArrayBuffer.GetLen() == 0) {
					buffer_mutex[bufferIdx].unlock();
					break;	//Read nothing
				}
				memcpy(buffer[bufferIdx], overlap, WINDOW_SIZE - 1);	//copy the overlap into current part
				memcpy(&buffer[bufferIdx][WINDOW_SIZE - 1], charArrayBuffer.GetArr(), charArrayBuffer.GetLen());
				buffer_len[bufferIdx] = charArrayBuffer.GetLen() + WINDOW_SIZE - 1;
				file_length += charArrayBuffer.GetLen();
				buffer_obsolete[bufferIdx] = false;
				memcpy(overlap, &buffer[bufferIdx][charArrayBuffer.GetLen()], WINDOW_SIZE - 1);	//copy the last window into overlap
				buffer_mutex[bufferIdx].unlock();
				bufferIdx ^= 1;
				tot_read += ((float)clock() - start_read) * 1000 / CLOCKS_PER_SEC;
			}
			cout << "File size: " << file_length / 1024 << " KB\n";
		}
		else
			fprintf(stderr, "Unknown file format %s\n", FILE_FORMAT_TEXT[FILE_FORMAT]);
		no_more_input = true;
	}

	void Chunking() {
		int bufferIdx = 0;
		int chunkingResultIdx = 0;
		while (true) {
			buffer_mutex[bufferIdx].lock();
			if (buffer_obsolete[bufferIdx]) {
				buffer_mutex[bufferIdx].unlock();
				if (no_more_input)
					break;
				this_thread::sleep_for(chrono::microseconds(500));
				continue;
			}

			start_chunk = clock();
			deque<unsigned int> currentChunkingResult = re.chunking(buffer[bufferIdx], buffer_len[bufferIdx]);
			tot_chunk += ((float)clock() - start_chunk) * 1000 / CLOCKS_PER_SEC;
			buffer_mutex[bufferIdx].unlock();

			chunking_result_mutex[chunkingResultIdx].lock();
			while (!chunking_result_obsolete[chunkingResultIdx]) {
				chunking_result_mutex[chunkingResultIdx].unlock();
				this_thread::sleep_for(chrono::microseconds(500));
				chunking_result_mutex[chunkingResultIdx].lock();
			}
			chunking_result[chunkingResultIdx] = currentChunkingResult;
			chunking_result_obsolete[chunkingResultIdx] = false;
			chunking_result_mutex[chunkingResultIdx].unlock();
			chunkingResultIdx ^= 1;

			bufferIdx ^= 1;
		}
		no_more_chunking_result = true;
	}

	void Fingerprinting() {
		int bufferIdx = 0;
		int chunkingResultIdx = 0;
		//When the whole process starts, all chunking results are obsolete, that's the reason fingerprinting part need to check buffer state
		while (true) {
			lock<mutex, mutex>(buffer_mutex[bufferIdx], chunking_result_mutex[chunkingResultIdx]);
			lock_guard<mutex> lk1(buffer_mutex[bufferIdx], adopt_lock);
			lock_guard<mutex> lk2(chunking_result_mutex[chunkingResultIdx], adopt_lock);

			if (buffer_obsolete[bufferIdx] || chunking_result_obsolete[chunkingResultIdx]) {
				if (no_more_chunking_result)
					break;
				this_thread::sleep_for(chrono::microseconds(500));
				continue;
			}
			start_fin = clock();

			total_duplication_size += re.fingerPrinting(chunking_result[chunkingResultIdx], buffer[bufferIdx]);
			buffer_obsolete[bufferIdx] = true;
			chunking_result_obsolete[chunkingResultIdx] = true;
			bufferIdx ^= 1;
			chunkingResultIdx ^= 1;
			tot_fin += ((float)clock() - start_fin) * 1000 / CLOCKS_PER_SEC;
		}
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