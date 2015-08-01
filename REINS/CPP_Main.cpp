#include "CPP_Main.h"
using namespace std;

namespace CPP_Namespace {

	int CPP_Main(int argc, char* argv[])
	{
		clock_t start = clock();
		cout << "\n============================ C++ Implementation =============================\n";

		unsigned int length;
		char *packet; 
		//For pcap file
		string payload;

		if (argc != 2) {
			cout << "Usage: " << argv[0] << " <filename>\n";
			return -1;
		}

		clock_t start_read = clock();

		if (FILE_FORMAT == PlainText) {
			ifstream ifs(argv[1], ios::in | ios::binary | ios::ate);
			if (!ifs.is_open()) {
				fprintf(stderr, "Can not open file %s\n", argv[1]);
				return -1;
			}

			length = ifs.tellg();
			ifs.seekg(0, ifs.beg);

			packet = new char[length];
			ifs.read(packet, length);
			ifs.close();
		}
		else if (FILE_FORMAT == Pcap) {
			PcapReader fileReader;
			payload = fileReader.ReadPcapFile(argv[1]);
			packet = &payload[0];
			length = payload.length();
		}
		else {
			fprintf(stderr, "Unknown file format %s\n", FILE_FORMAT_TEXT[FILE_FORMAT]);
			return -1;
		}
		
		cout << "Reading time: " << ((float)clock() - start_read) * 1000 / CLOCKS_PER_SEC << " ms\n";

		cout << "File size: " << length / 1024 << " KB\n";

		CPP_TestOfRabinFingerprint((char*)packet, length);
		
		if (FILE_FORMAT == PlainText)
			delete[] packet;
		
		cout << "Total time: " << ((float)clock() - start) * 1000 / CLOCKS_PER_SEC << " ms\n";
		cout << "=============================================================================\n";
		//system("pause");
		return 0;
	}

	void CPP_TestOfRabinFingerprint(char* fileContent, unsigned int fileContentLen) {
		RedundancyEliminator_CPP re;
		re.SetupRedundancyEliminator_CPP();
		unsigned int duplicationSize = re.eliminateRedundancy(fileContent, fileContentLen);
		cout << "Found " << duplicationSize << " bytes of redundency, which is " << (float)duplicationSize / fileContentLen * 100 << " percent of file\n";
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