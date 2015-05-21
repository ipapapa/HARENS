#include "CUDA_Main.h"
using namespace std;

namespace CUDA_Namespace {

	int CUDA_Main(int argc, char* argv[])
	{
		clock_t start = clock();
		cout << "\n============================ CUDA Implementation ============================\n";
		if (argc != 2) {
			cout << "Usage: " << argv[0] << " <filename>\n";
			return -1;
		}
		ifstream ifs(argv[1], ios::in | ios::binary | ios::ate);
		if (!ifs.is_open()) {
			cout << "Can not open file " << argv[1] << endl;
			return -1;
		}

		uint length = ifs.tellg();
		ifs.seekg(0, ifs.beg);

		clock_t start_read = clock();
		char *buffer = new char[length];
		ifs.read(buffer, length);
		ifs.close();
		cout << "Reading time: " << ((float)clock() - start_read) * 1000 / CLOCKS_PER_SEC << " ms\n";

		// string fileContent((istreambuf_iterator<char>(ifs)), (istreambuf_iterator<char>()));	
		cout << "File size: " << length / 1024 << " KB\n";
		CUDA_TestOfRabinFingerprint(buffer, length);
		delete[] buffer;
		clock_t end = clock();
		cout << "Total time: " << ((float)end - start) * 1000 / CLOCKS_PER_SEC << " ms\n";
		cout << "=============================================================================\n";
		return 0;
	}

	void CUDA_TestOfRabinFingerprint(char* fileContent, uint fileContentLen) {
		RedundancyEliminator_CUDA re(RedundancyEliminator_CUDA::NonMultifingerprint);
		uint duplicationSize = re.eliminateRedundancy(fileContent, fileContentLen);

		cout << "Found " << duplicationSize << " bytes of redundency, which is " << (float)duplicationSize / fileContentLen * 100 << " percent of file\n";
	}
}