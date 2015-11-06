#This is a makefile for HARENS
#Please make sure you have installed nvcc in your computer and added it in system path

cp = /usr/local/cuda/bin/nvcc
cudaflags = -gencode arch=compute_20,code=sm_20
cudaflags += -gencode arch=compute_35,code=sm_35
cflags = -std=c++11 -O2 
cflags += -I/usr/local/cuda-7.5/include/ -I/usr/include/
lflags = -L/usr/local/cuda-7.5/lib64/
lflags += -lcudart -lpcap -lcrypto

#HARENS: $(objects)
#	$(cp) -M $(cflags) $(lflags) $(wildcard *.cpp) -o $(objects)
#	$(cp) -o HARENS $(objects)

#Tell make to make one .out file for each .cpp file found in the current directory
run: $(patsubst %.cpp,%.o,$(wildcard *.cpp))
	$(cp) $< -o $@

#Rule how to create arbitary .out files. 
#First state what is needed for them e.g. additional headers, .cpp files in an include folder...
#Then the command to create the .out file, probably you want to add further options to the g++ call.
%.o: %.cpp %.h
	$(cp) $(cudaflags) $(cflags) $(lflags) $< -o $@

.PHONY: intall
install:
	sudo apt-get install build-essential
	sudo apt-get install gcc-5 g++-5 cpp-5
	sudo apt-get install cuda
	sudo apt-get upgrade cuda
	sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-5 50
	sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-5 50
	sudo update-alternatives --install /usr/bin/cpp cpp-bin /usr/bin/cpp-5 50
	sudo apt-get install libpcap
	sudo apt-get install openssl

.PHONY: clean
clean:
	rm -rf *.o run