#This is a makefile for HARENS
#Please make sure you have installed nvcc in your computer and added it in system path

cp = nvcc
cflags = -std=c++11 -O2 
cflags += -I./WpdPack/Include -I./openssl_build_x64/include
lflags = -L./WpdPack/Lib -L./openssl_build_x64/lib
lflags += -llibpacket.a -llibwpcap.a -llibeay64.lib -lssleay64.lib
objects := $(patsubst %.cpp ,%.o, $(wildcard *.cpp))

HARENS: $(objects)
	$(cp) -M $(cflags) $(lflags) $(wildcard *.cpp) > $(objects)
	$(cp) -o HARENS $(objects)

.PHONY: clean
clean:
	rm *.o temp