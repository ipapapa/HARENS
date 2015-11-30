#include <stdio.h>
#include "NaiveCpp.h"
#include "CppPipeline.h"
#include "CudaAcceleratedAlg.h"
#include "Harens.h"
#include "HashCollisionTest.h"
using namespace std;

int main() {
	IO::input_file_name = "SampleInput.txt";
	double rate, time;
	NaiveCpp().Test(rate, time);
	if (rate < 0.379982 || rate > 0.379983) {
		fprintf(stderr, "In NaiveCpp, expected redundancy rate %f, got %f\n", 0.379982, rate);
		exit(EXIT_FAILURE);
	}
	if (time > 150) {
		fprintf(stderr, "In NaiveCpp, expected run time %f, took %f\n", 150, time);
		exit(EXIT_FAILURE);
	}
	CppPipeline().Test(rate, time);
	if (rate < 0.379982 || rate > 0.379983) {
		fprintf(stderr, "In CppPipeline, expected redundancy rate %f, got %f\n", 0.379982, rate);
		exit(EXIT_FAILURE);
	}
	if (time > 150) {
		fprintf(stderr, "In CppPipeline, expected run time %f, took %f\n", 150, time);
		exit(EXIT_FAILURE);
	}
	CudaAcceleratedAlg().Test(rate, time);
	if (rate < 0.379982 || rate > 0.379983) {
		fprintf(stderr, "In CudaAcceleratedAlg, expected redundancy rate %f, got %f\n", 0.379982, rate);
		exit(EXIT_FAILURE);
	}
	if (time > 150) {
		fprintf(stderr, "In CudaAcceleratedAlg, expected run time %f, took %f\n", 150, time);
		exit(EXIT_FAILURE);
	}
	Harens().Test(rate, time);
	if (rate < 0.379982 || rate > 0.379983) {
		fprintf(stderr, "In Harens, expected redundancy rate %f, got %f\n", 0.379982, rate);
		exit(EXIT_FAILURE);
	}
	if (time > 150) {
		fprintf(stderr, "In Harens, expected run time %f, took %f\n", 150, time);
		exit(EXIT_FAILURE);
	}
	exit(EXIT_SUCCESS);
}