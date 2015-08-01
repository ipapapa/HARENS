#include "CPP_Main.h"
#include "CPP_Pipeline_Main.h"
#include "CUDA_Main.h"
#include "CUDA_Pipeline_Main.h"
#include "CUDA_Pipeline_PushMod_Main.h"

enum Method { CPP_Imp, CPP_Pipeline, CUDA_Imp, CUDA_Pipeline, CUDA_Pipeline_PushMod, CUDA_COMPARE, ALL };
Method method = CPP_Imp;

int main(int argc, char* argv[]) {
	switch (method) {
	case CPP_Imp:
		CPP_Namespace::CPP_Main(argc, argv);
		break;
	case CPP_Pipeline:
		CPP_Pipeline_Namespace::CPP_Pipeline_Main(argc, argv);
		break;
	case CUDA_Imp:
		//for (int i = 0; i < 5; ++i)
			CUDA_Namespace::CUDA_Main(argc, argv);
		break;
	case CUDA_Pipeline:
		CUDA_Pipeline_Namespace::CUDA_Pipeline_Main(argc, argv);
		break;
	case CUDA_Pipeline_PushMod: 
		CUDA_Pipeline_PushMod_Namespace::CUDA_Pipeline_PushMod_Main(argc, argv);
		break;
	case CUDA_COMPARE:
		CUDA_Namespace::CUDA_Main(argc, argv);
		CUDA_Pipeline_Namespace::CUDA_Pipeline_Main(argc, argv);
		break;
	default:
		CPP_Namespace::CPP_Main(argc, argv);
		CPP_Pipeline_Namespace::CPP_Pipeline_Main(argc, argv);
		CUDA_Namespace::CUDA_Main(argc, argv);
		CUDA_Pipeline_Namespace::CUDA_Pipeline_Main(argc, argv);
		//CUDA_Pipeline_PushMod_Namespace::CUDA_Pipeline_PushMod_Main(argc, argv);
	}
	return 0;
}