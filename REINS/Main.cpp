#include "CPPExecution.h"
#include "CPPPipelineExecution.h"
#include "CUDAExecution.h"
#include "CUDAPipelineExecution.h"

enum Method { CPP_Imp, CPP_Pipeline, CUDA_Imp, CUDA_Pipeline, CUDA_COMPARE, ALL };
Method method = ALL;

void PrintUsage();
void Execute(Method method);

int main(int argc, char* argv[]) {
	int argNum = 1;
	while (argNum < argc) {
		try {
			string arg = argv[argNum];
			if (arg == "-h") {
				PrintUsage();
				return 0;
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
			else if (arg == "-f") {
				++argNum;
				size_t size = 0;
				while (argv[argNum][size] != '\0') {
					++size;
				}
				++size;
				IO::input_file_name = new char[size];
				memcpy(IO::input_file_name, argv[argNum], size);
			}
			else if (arg == "-i") {
				string format = argv[++argNum];
				if (format == "pcap" || format == "PCAP") {
					IO::FILE_FORMAT = FileFormat::Pcap;
				}
				else if (format == "plain" || format == "PLAIN") {
					IO::FILE_FORMAT = FileFormat::PlainText;
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
	if (IO::input_file_name == nullptr) {
		PrintUsage();
		return 1;
	}
	Execute(method);
	return 0;
}

void PrintUsage() {
	printf("Usage:\n\
-a:\tthe algorithm to use, choose from \n\
\tcpp:\t\t\tnaive C++ implementation\n\
\tcpp_pipe:\t\tC++ multi-threaded pipeline implementation\n\
\tcuda:\t\t\tCUDA accelerated algorithm\n\
\tcuda_pipe (default):\tCUDA accelerated algorithm \n\
\t\t\t\twith multi-theaded pipeline\n\
-f:\tinput file name (required)\n\
-i:\tinput format, choose from plain (default) and pcap\n\
-o:\toutput to console/file\n\
\te.g. -o console (default) or -o \"file name\"\n\
-h:\thelp\n\
	");
}

void Execute(Method method) {
	switch (method) {
	case CPP_Imp:
		CPP_Namespace::CPPExecute();
		break;
	case CPP_Pipeline:
		CPP_Pipeline_Namespace::CPPPipelineExecute();
		break;
	case CUDA_Imp:
		CUDA_Namespace::CUDAExecute();
		break;
	case CUDA_Pipeline:
		CUDA_Pipeline_Namespace::CUDAPipelineExecute();
		break;
	case CUDA_COMPARE:
		CUDA_Namespace::CUDAExecute();
		CUDA_Pipeline_Namespace::CUDAPipelineExecute();
		break;
	default:
		CPP_Namespace::CPPExecute();
		CPP_Pipeline_Namespace::CPPPipelineExecute();
		CUDA_Namespace::CUDAExecute();
		CUDA_Pipeline_Namespace::CUDAPipelineExecute();
	}
}