#include "NaiveCpp.h"
#include "CppPipeline.h"
#include "CudaAcceleratedAlg.h"
#include "Harens.h"
#include "HashCollisionTest.h"

enum Method { CPP_Imp, CPP_Pipeline, CUDA_Imp, CUDA_Pipeline, CUDA_COMPARE, ALL };
Method method = CUDA_Pipeline;

void PrintUsage();
void CompareHash(bool isCollisionCheck);
void Execute(Method method, int mapperNum, int reducerNum);

int main(int argc, char* argv[]) {
	int argNum = 1;
	int mapperNum = 11;
	int reducerNum = 212;
	bool compareSHA1AndRabin = false;
	bool isCollisionCheck;
	while (argNum < argc) {
		try {
			string arg = argv[argNum];
			if (arg == "-h") {
				PrintUsage();
				return 0;
			}
			else if (arg == "-c") {
				compareSHA1AndRabin = true;
				string test = argv[++argNum];
				if (test == "speed" || test == "SPEED") {
					isCollisionCheck = false;
				}
				else if (test == "collision" || test == "COLLISION") {
					isCollisionCheck = true;
				}
				else {
					printf("Error: unknown test: %s\n", test);
					return 1;
				}
			}
			else if (arg == "-a") {
				string alg = argv[++argNum];
				if (alg == "cpp" || alg == "CPP") {
					method = CPP_Imp;
				}
				else if (alg == "cpp_pipe" || alg == "CPP_PIPE") {
					method = CPP_Pipeline;
				}
				else if (alg == "cuda" || alg == "CUDA") {
					method = CUDA_Imp;
				}
				else if (alg == "cuda_pipe" || alg == "CUDA_PIPE") {
					method = CUDA_Pipeline;
				}
				else {
					printf("Error: unknown algorithm: %s\n", alg);
					return 1;
				}
			}
			else if (arg == "-m") {
				++argNum;
				mapperNum = std::stoi(argv[argNum], nullptr, 10);
			}
			else if (arg == "-r") {
				++argNum;
				reducerNum = std::stoi(argv[argNum], nullptr, 10);
			}
			else if (arg == "-f") {
				while (argNum + 1 < argc && argv[argNum + 1][0] != '-') {
					++argNum;
					size_t size = 0;
					while (argv[argNum][size] != '\0') {
						++size;
					}
					++size;
					char* file_name = new char[size];
					memcpy(file_name, argv[argNum], size);
					IO::input_file_name.push_back(file_name);
				}
			}
			else if (arg == "-i") {
				string format = argv[++argNum];
				if (format == "pcap" || format == "PCAP") {
					IO::SetFileFormat(FileFormat::Pcap);
				}
				else if (format == "plain" || format == "PLAIN") {
					IO::SetFileFormat(FileFormat::PlainText);
				}
				else {
					printf("Error: unknown format: %s\n", format);
					return 1;
				}
			}
			else if (arg == "-o") {
				string outputChannel = argv[++argNum];
				if (outputChannel != "console" && outputChannel != "CONSOLE") {
					IO::output_file_name = argv[argNum];
				}
			}
			else {
				throw exception();
			}
			++argNum;
		}
		catch (exception e) {
			PrintUsage();
			return 1;
		}
	}
	if (IO::input_file_name.empty()) {
		PrintUsage();
		return 1;
	}
	if (compareSHA1AndRabin) {
		CompareHash(isCollisionCheck);
	}
	else {
		Execute(method, mapperNum, reducerNum);
	}
	return 0;
}

void PrintUsage() {
	printf("Usage:\n\
-c:\tcompare the performance of using Rabin hash and SHA1 as chunk hash\n\
\tfunction\n\
\tspeed:\t\trepot the speed of two methods\n\
\tcollision:\treport hash collision of two methods\n\
-a:\tthe algorithm to use, choose from \n\
\tcpp:\t\t\tnaive C++ implementation\n\
\tcpp_pipe:\t\tC++ multi-threaded pipeline implementation\n\
\tcuda:\t\t\tCUDA accelerated algorithm\n\
\tcuda_pipe (default):\tCUDA accelerated algorithm with multi-theaded\n\
\t\t\t\tpipeline and single-machine MapReduce\n\
-m:\tmapper number (parameter for cuda_pipe)\n\
\te.g. -m 8 (default)\n\
-r:\treducer number (parameter for cuda_pipe)\n\
\te.g. -r 256 (default)\n\
-f:\tinput file name (required)\n\
-i:\tinput format, choose from plain (default) and pcap\n\
-o:\toutput to console/file\n\
\te.g. -o console (default) or -o \"file name\"\n\
-h:\thelp\n\
e.g.\tREINS.exe -f inputfile.txt -a cuda_pipe -m 8 -r 256\n\
\tREINS.exe -f inputfile.txt -o outputfile.txt -c speed\n");
}

void CompareHash(bool isCollisionCheck) {
	//use rabin hash
	HashCollisionTest(0, isCollisionCheck).Execute();
	//use sha1 hash
	HashCollisionTest(1, isCollisionCheck).Execute();
	//use md5 hash
	HashCollisionTest(2, isCollisionCheck).Execute();
}

void Execute(Method method, int mapperNum, int reducerNum) {
	switch (method) {
	case CPP_Imp:
		NaiveCpp().Execute();
		break;
	case CPP_Pipeline:
		CppPipeline().Execute();
		break;
	case CUDA_Imp:
		CudaAcceleratedAlg().Execute();
		break;
	case CUDA_Pipeline:
		Harens(mapperNum, reducerNum).Execute();
		break;
	case CUDA_COMPARE:
		CudaAcceleratedAlg().Execute();
		Harens(mapperNum, reducerNum).Execute();
		break;
	default:
		NaiveCpp().Execute();
		CppPipeline().Execute();
		CudaAcceleratedAlg().Execute();
		Harens(mapperNum, reducerNum).Execute();
	}
}