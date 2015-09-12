#include "CPP_Pipeline_Main.h"
using namespace std;

namespace CPP_Pipeline_Namespace {

	const int BUFFER_NUM = 550;
	const int CHUNKING_RESULT_NUM = 2;
	const unsigned int MAX_BUFFER_LEN = 4096 * 512 + WINDOW_SIZE - 1;	//Just to keep it the same as cuda
	const unsigned int MAX_WINDOW_NUM = MAX_BUFFER_LEN - WINDOW_SIZE + 1;

	unsigned int file_length = 0;
	RedundancyEliminator_CPP re;
	ifstream ifs;
	char* fileName;
	//syncronize
	array<mutex, BUFFER_NUM> buffer_mutex;							//lock for buffer
	array<condition_variable, BUFFER_NUM> buffer_cond;
	array<mutex, CHUNKING_RESULT_NUM> chunking_result_mutex;		//lock for chunking_result
	array<condition_variable, CHUNKING_RESULT_NUM> chunking_result_cond;
	array<bool, BUFFER_NUM> buffer_obsolete;					//states of buffer
	array<bool, CHUNKING_RESULT_NUM> chunking_result_obsolete;	//states of chunking_result
	bool read_file_end = false;
	bool chunking_end = false;
	mutex read_file_end_mutex, chunking_end_mutex;
	//shared data
	char overlap[WINDOW_SIZE - 1];
	char** buffer = new char*[BUFFER_NUM];	//two buffers
	FixedSizedCharArray charArrayBuffer(MAX_BUFFER_LEN);

	array<unsigned int, BUFFER_NUM> buffer_len;
	array<deque<unsigned int>, CHUNKING_RESULT_NUM> chunking_result;
	//Result
	unsigned int total_duplication_size = 0;
	//Time
	clock_t start_read, start_chunk, start_fin;
	float tot_read = 0, tot_chunk = 0, tot_fin = 0;

	int CPP_Pipeline_Main(int argc, char* argv[])
	{
		cout << "\n================== Pipeline Version of C++ Implementation ===================\n";
		if (argc != 2) {
			cout << "You used " << argc << " variables\n";
			cout << "Usage: " << argv[0] << " <filename>\n";
			system("pause");
			return -1;
		}

		fileName = argv[1];

		re.SetupRedundancyEliminator_CPP();

		for (int i = 0; i < BUFFER_NUM; ++i) {
			buffer[i] = new char[MAX_BUFFER_LEN];
			buffer_len[i] = 0;
			buffer_obsolete[i] = true;
		}

		for (int i = 0; i < CHUNKING_RESULT_NUM; ++i) {
			chunking_result_obsolete[i] = true;
		}

		//Create threads 
		thread tReadFile(ReadFile);
		tReadFile.join();
		clock_t start = clock();
		thread tChunking(Chunking);
		thread tFingerprinting(Fingerprinting);

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
		unique_lock<mutex> readFileInitLock(buffer_mutex[bufferIdx]);
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
		readFileInitLock.unlock();
		buffer_cond[bufferIdx].notify_one();
		bufferIdx = (bufferIdx + 1) % BUFFER_NUM;
		tot_read += ((float)clock() - start_read) * 1000 / CLOCKS_PER_SEC;
		//Read the rest
		if (FILE_FORMAT == PlainText) {
			while (curWindowNum == MAX_WINDOW_NUM) {
				unique_lock<mutex> readFileIterLock(buffer_mutex[bufferIdx]);
				while (buffer_obsolete[bufferIdx] == false) {
					buffer_cond[bufferIdx].wait(readFileIterLock);
				}
				start_read = clock();
				buffer_len[bufferIdx] = min(MAX_BUFFER_LEN, file_length - curFilePos + WINDOW_SIZE - 1);
				curWindowNum = buffer_len[bufferIdx] - WINDOW_SIZE + 1;
				memcpy(buffer[bufferIdx], overlap, WINDOW_SIZE - 1);	//copy the overlap into current part
				ifs.read(&buffer[bufferIdx][WINDOW_SIZE - 1], curWindowNum);
				memcpy(overlap, &buffer[bufferIdx][curWindowNum], WINDOW_SIZE - 1);	//copy the last window into overlap
				buffer_obsolete[bufferIdx] = false;
				readFileIterLock.unlock();
				buffer_cond[bufferIdx].notify_one();
				bufferIdx = (bufferIdx + 1) % BUFFER_NUM;
				curFilePos += curWindowNum;
				tot_read += ((float)clock() - start_read) * 1000 / CLOCKS_PER_SEC;
			}
			ifs.close();
		}
		else if (FILE_FORMAT == Pcap) {
			while (true) {
				unique_lock<mutex> readFileIterLock(buffer_mutex[bufferIdx]);
				while (buffer_obsolete[bufferIdx] == false) {
					buffer_cond[bufferIdx].wait(readFileIterLock);
				}
				start_read = clock();
				fileReader.ReadPcapFileChunk(charArrayBuffer, MAX_BUFFER_LEN - WINDOW_SIZE + 1);
				
				if (charArrayBuffer.GetLen() == 0) {
					readFileIterLock.unlock();
					buffer_cond[bufferIdx].notify_all();
					break;	//Read nothing
				}
				memcpy(buffer[bufferIdx], overlap, WINDOW_SIZE - 1);	//copy the overlap into current part
				memcpy(&buffer[bufferIdx][WINDOW_SIZE - 1], charArrayBuffer.GetArr(), charArrayBuffer.GetLen());
				buffer_len[bufferIdx] = charArrayBuffer.GetLen() + WINDOW_SIZE - 1;
				file_length += charArrayBuffer.GetLen();
				buffer_obsolete[bufferIdx] = false;
				memcpy(overlap, &buffer[bufferIdx][charArrayBuffer.GetLen()], WINDOW_SIZE - 1);	//copy the last window into overlap
				readFileIterLock.unlock();
				buffer_cond[bufferIdx].notify_one();
				bufferIdx = (bufferIdx + 1) % BUFFER_NUM;
				tot_read += ((float)clock() - start_read) * 1000 / CLOCKS_PER_SEC;
			}
			cout << "File size: " << file_length / 1024 << " KB\n";
		}
		else
			fprintf(stderr, "Unknown file format %s\n", FILE_FORMAT_TEXT[FILE_FORMAT]);
		unique_lock<mutex> readFileEndLock(read_file_end_mutex);
		read_file_end = true;
		//In case the other threads stuck in waiting for condition variable
		buffer_cond[bufferIdx].notify_all();
	}

	void Chunking() {
		int bufferIdx = 0;
		int chunkingResultIdx = 0;
		while (true) {
			unique_lock<mutex> bufferLock(buffer_mutex[bufferIdx]);
			if (buffer_obsolete[bufferIdx]) {
				unique_lock<mutex> readFileEndLock(read_file_end_mutex);
				if (read_file_end){
					unique_lock<mutex> chunkingEndLock(chunking_end_mutex);
					chunking_end = true;
					return;
				}
				readFileEndLock.unlock();
				buffer_cond[bufferIdx].wait(bufferLock);
			}

			start_chunk = clock();
			deque<unsigned int> currentChunkingResult = re.chunking(buffer[bufferIdx], buffer_len[bufferIdx]);
			tot_chunk += ((float)clock() - start_chunk) * 1000 / CLOCKS_PER_SEC;
			bufferLock.unlock();
			buffer_cond[bufferIdx].notify_one();
			bufferIdx = (bufferIdx + 1) % BUFFER_NUM;

			unique_lock<mutex> chunkingResultLock(chunking_result_mutex[chunkingResultIdx]);
			while (chunking_result_obsolete[chunkingResultIdx] == false) {
				chunking_result_cond[chunkingResultIdx].wait(chunkingResultLock);
			}
			chunking_result[chunkingResultIdx] = currentChunkingResult;
			chunking_result_obsolete[chunkingResultIdx] = false;
			chunkingResultLock.unlock();
			chunking_result_cond[chunkingResultIdx].notify_one();
			chunkingResultIdx = (chunkingResultIdx + 1) % CHUNKING_RESULT_NUM;

		}
	}

	void Fingerprinting() {
		int bufferIdx = 0;
		int chunkingResultIdx = 0;
		//When the whole process starts, all chunking results are obsolete, that's the reason fingerprinting part need to check buffer state
		while (true) {
			//Get buffer ready
			unique_lock<mutex> bufferLock(buffer_mutex[bufferIdx]);
			while (buffer_obsolete[bufferIdx]) {
				unique_lock<mutex> chunkingEndLock(chunking_end_mutex);
				if (chunking_end)
					return;
				chunkingEndLock.unlock();
				buffer_cond[bufferIdx].wait(bufferLock);
			}
			//Get chunking result ready
			unique_lock<mutex> chunkingResultLock(chunking_result_mutex[chunkingResultIdx]);
			while (chunking_result_obsolete[chunkingResultIdx]) {
				unique_lock<mutex> chunkingEndLock(chunking_end_mutex);
				if (chunking_end)
					return;
				chunkingEndLock.unlock();
				chunking_result_cond[chunkingResultIdx].wait(chunkingResultLock);
			}

			start_fin = clock();

			total_duplication_size += re.fingerPrinting(chunking_result[chunkingResultIdx], buffer[bufferIdx]);
			buffer_obsolete[bufferIdx] = true;
			chunking_result_obsolete[chunkingResultIdx] = true;
			bufferLock.unlock();
			buffer_cond[bufferIdx].notify_one();
			chunkingResultLock.unlock();
			chunking_result_cond[chunkingResultIdx].notify_one();

			bufferIdx = (bufferIdx + 1) % BUFFER_NUM;
			chunkingResultIdx = (chunkingResultIdx + 1) % CHUNKING_RESULT_NUM;
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