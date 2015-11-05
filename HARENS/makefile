#This is a makefile for HARENS
#Please make sure you have installed nvcc in your computer and added it in system path

cp = g++-5
cflags = -std=c++11 -O2 
cflags += -I/usr/local/cuda-7.5/include -I./WpdPack/Include -I./openssl_build_x64/include 
lflags = -L/usr/local/cuda-7.5/lib64 -L./WpdPack/Lib -L./openssl_build_x64/lib 
lflags += -llibcudart -llibpacket.a -llibwpcap.a -llibeay64.lib -lssleay64.lib 

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
	$(cp) $(cflags) $(lflags) $< -o $@

.PHONY: intall
install:
	sudo apt-get install build-essential
	sudo apt-get install gcc-5 g++-5 cpp-5

.PHONY: clean
clean:
	rm *.o run