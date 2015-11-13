#This is a makefile for HARENS
#Please make sure you have installed nvcc in your computer and added it in system path

CXX = /usr/bin/g++-5
NVCC = /usr/local/cuda/bin/nvcc
cudaflags = -gencode arch=compute_20,code=sm_20
cudaflags += -gencode arch=compute_35,code=sm_35
cflags = -c -std=c++11 -O2 
iflags += -I/usr/local/cuda-7.5/include -I/usr/include
build = -std=c++11 -O2 -I/usr/local/cuda-7.5/include -I/usr/include
lflags = -L/usr/local/cuda-7.5/lib64
lflags += -lcudart -lpcap -lcrypto
objects := $(patsubst %.cpp,%.o,$(wildcard *.cpp)) $(patsubst %.cu,%.o,$(wildcard *.cu))

#HARENS: $(objects)
#	$(cp) -M $(cflags) $(iflags) $(lflags) $(wildcard *.cpp) -o $(objects)
#	$(cp) -o HARENS $(objects)

all: run

#Tell make to make one .out file for each .cpp file found in the current directory
run: $(objects)
	$(NVCC) Main.o NaiveCpp.o CppPipeline.o CudaAcceleratedAlg.o Harens.o HashCollisionTest.o \
	RedundancyEliminator_CPP_CollisionTest.o RedundancyEliminator_CPP.o RedundancyEliminator_CUDA.o \
	RabinHash.o IO.o PcapReader.o FixedSizedCharArray.o \
	$(iflags) $(lflags) -o $@

#Rule how to create arbitary .out files. 
#First state what is needed for them e.g. additional headers, .cpp files in an include folder...
#Then the command to create the .out file, probably you want to add further options to the g++ call.
#%.o: %.cpp %.h
Main.o: Main.cpp NaiveCpp.o CppPipeline.o CudaAcceleratedAlg.o Harens.o HashCollisionTest.o
	$(CXX) $(cflags) $(iflags) $(lflags) $< -o $@

NaiveCpp.o: NaiveCpp.cpp IO.o RabinHash.o RedundancyEliminator_CPP.o PcapReader.o
	$(CXX) $(cflags) $(iflags) $(lflags) $< -o $@

CppPipeline.o: CppPipeline.cpp IO.o RabinHash.o RedundancyEliminator_CPP.o PcapReader.o
	$(CXX) $(cflags) $(iflags) $(lflags) $< -o $@

CudaAcceleratedAlg.o: CudaAcceleratedAlg.cpp IO.o RabinHash.o PcapReader.o RedundancyEliminator_CUDA.o
	$(CXX) $(cflags) $(iflags) $(lflags) $< -o $@

Harens.o: Harens.cpp IO.o RabinHash.o PcapReader.o RedundancyEliminator_CUDA.o
	$(CXX) $(cflags) $(iflags) $(lflags) $< -o $@

HashCollisionTest.o: HashCollisionTest.cpp IO.o RabinHash.o RedundancyEliminator_CPP_CollisionTest.o PcapReader.o
	$(CXX) $(cflags) $(iflags) $(lflags) $< -o $@

RedundancyEliminator_CPP_CollisionTest.o: RedundancyEliminator_CPP_CollisionTest.cpp RedundancyEliminator_CPP.o
	$(CXX) $(cflags) $(iflags) $(lflags) $< -o $@

RedundancyEliminator_CPP.o: RedundancyEliminator_CPP.cpp RabinHash.o
	$(CXX) $(cflags) $(iflags) $(lflags) $< -o $@

RedundancyEliminator_CUDA.o: RedundancyEliminator_CUDA.cu RabinHash.o
	$(NVCC) $(cudaflags) $(cflags) $(iflags) $(lflags) $< -o $@

RabinHash.o: RabinHash.cpp
	$(CXX) $(cflags) $(iflags) $(lflags) $< -o $@

IO.o: IO.cpp
	$(CXX) $(cflags) $(iflags) $(lflags) $< -o $@

PcapReader.o: PcapReader.cpp FixedSizedCharArray.o
	$(CXX) $(cflags) $(iflags) $(lflags) $< -o $@

FixedSizedCharArray.o: FixedSizedCharArray.cpp
	$(CXX) $(cflags) $(iflags) $(lflags) $< -o $@

%.o: %.h 
$(objects): Definition.h LRUQueue.h LRUQueuePool.h SelfMantainedLRUQueue.h LRUVirtualHash.h \
LRUStrHash.h LRUStrHashPool.h makefile
RedundancyEliminator_CPP_CollisionTest.o: LRUHash.h 

.PHONY: intall
install:
	sudo add-apt-repository ppa:ubuntu-toolchain-r/test
	sudo apt-get update
	sudo apt-get install build-essential
	sudo apt-get install gcc-4.9 g++-4.9 cpp-4.9
	sudo apt-get install gcc-5 g++-5 cpp-5
	sudo apt-get install cuda
	sudo apt-get upgrade cuda
	sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-4.9 100
	sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-4.9 100
	sudo update-alternatives --install /usr/bin/cpp cpp-bin /usr/bin/cpp-4.9 100
	sudo apt-get install libpcap-dev
	sudo apt-get install openssl
	export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-7.5/lib64

.PHONY: clean
clean:
	rm -rf *.o run