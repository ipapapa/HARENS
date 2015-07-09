#include "CPP_Main.h"
using namespace std;

namespace CPP_Namespace {

	int CPP_Main(int argc, char* argv[])
	{
		clock_t start = clock();
		cout << "\n============================ C++ Implementation =============================\n";

		unsigned int length;
		//char *buffer;

		if (argc != 2) {
			cout << "Usage: " << argv[0] << " <filename>\n";
			return -1;
		}
		/*ifstream ifs(argv[1], ios::in | ios::binary | ios::ate);
		if (!ifs.is_open()) {
			cout << "Can not open file " << argv[1] << endl;
			return -1;
		}

		length = ifs.tellg();
		ifs.seekg(0, ifs.beg);

		buffer = new char[length];
		ifs.read(buffer, length);
		ifs.close();*/
		clock_t start_read = clock();
		string payload = PcapReader::ReadPcapFile(argv[1]);
		char* packet = &payload[0];
		length = payload.length();
		
		cout << "Reading time: " << ((float)clock() - start_read) * 1000 / CLOCKS_PER_SEC << " ms\n";

		cout << "File size: " << length / 1024 << " KB\n";

		CPP_TestOfRabinFingerprint((char*)packet, length);
		//delete[] buffer;
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