CXX = /usr/bin/g++-5
CXXFLAGS=-I../src
VPATH=../src

.PHONY: test
test:
	$(CXX) runtest.cpp -o runtest.o
	$(CXX) runtest.o NaiveCpp.o CppPipeline.o CudaAcceleratedAlg.o Harens.o HashCollisionTest.o \
	RedundancyEliminator_CPP_CollisionTest.o RedundancyEliminator_CPP.o RedundancyEliminator_CUDA.o \
	RabinHash.o IO.o PcapReader.o FixedSizedCharArray.o -o runtest
	runtest

.PHONY: clean
clean:
	rm -rf *.o runtest