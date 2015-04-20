#include "CPP_Main.h"
using namespace std;

namespace CPP_Namespace {

	const bool USE_SOCKET = true;

	int CPP_Main(int argc, char* argv[])
	{
		clock_t start = clock();
		cout << "\n============================ C++ Implementation =============================\n";

		uint length;
		char *buffer;

		if (argc != 2) {
			cout << "Usage: " << argv[0] << " <filename>\n";
			return -1;
		}
		ifstream ifs(argv[1], ios::in | ios::binary | ios::ate);
		if (!ifs.is_open()) {
			cout << "Can not open file " << argv[1] << endl;
			return -1;
		}

		length = ifs.tellg();
		ifs.seekg(0, ifs.beg);

		clock_t start_read = clock();
		buffer = new char[length];
		ifs.read(buffer, length);
		ifs.close();
		cout << "Reading time: " << ((float)clock() - start_read) / CLOCKS_PER_SEC << " s\n";

		cout << "File size: " << length / 1024 << " KB\n";

		CPP_TestOfRabinFingerprint(buffer, length);
		delete[] buffer;
		cout << "Total time: " << ((float)clock() - start) / CLOCKS_PER_SEC << " s\n";
		cout << "=============================================================================\n";
		//system("pause");
		return 0;
	}

	void CPP_TestOfRabinFingerprint(char* fileContent, uint fileContentLen) {
		RedundancyEliminator_CPP re;
		re.SetupRedundancyEliminator_CPP();
		uint duplicationSize = re.eliminateRedundancy(fileContent, fileContentLen);
		cout << "Found " << duplicationSize << " bytes of redundency, which is " << (float)duplicationSize / fileContentLen * 100 << " percent of file\n";
	}

	/* old function
	void TestOfRabinHash(char* fileContent, int fileContentLen) {
	RabinHash rh;
	//cout << rh.Hash("kelu") << endl;
	int windowsize = 32;
	ulong p = 32;
	//char currWindow[windowsize];

	set<ulong> hashValueSet;
	map<int, ulong> chunkMap;
	int totalBlocks = 0;
	int chunkNum = 0;
	char* chunk;
	const int CHUNK_SIZE = 32;
	for (int i = 0; i <= fileContentLen - windowsize; i++) {
	chunk = new char[CHUNK_SIZE];
	ulong hashValWindow = rh.Hash(chunk, CHUNK_SIZE);
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