#pragma once
#include "VritualReader.h"

/*
* A reader of plain file.
* This class is created to make the program perform consistent 
* when dealing with plain text file and pcap file.
*/
class PlainFileReader: public VirtualReader {
private:
	std::ifstream ifs;
	int fileLen;
	int curFilePos;
	char* buffer;

public:
	PlainFileReader() {
		buffer = new char[MAX_BUFFER_LEN];
	}

	~PlainFileReader() {
		delete[] buffer;
	}

	/*
	* Open pcap file and set a handle
	*/
	void SetupReader(char* fileName) override;

	/*
	* Read the whole pcap file into memory by packets until it reaches the limit.
	*/
	void ReadChunk(FixedSizedCharArray &charArray, unsigned int readLen) override;

	/*
	* Read the whole pcap file into memory by packets
	*/
	char* ReadAll(char* fileName) override;
};