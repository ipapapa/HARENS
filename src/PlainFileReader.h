#pragma once
#include "VirtualReader.h"

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

protected:
	/*
	* Set up a file stream for plain file.
	*/
	void SetupFile(char* filename) override;

public:
	PlainFileReader();

	~PlainFileReader();

	/*
	* Set up the reader for a file List, and set up the first file
	* Make sure the filenameList is not empty.
	*/
	void SetupReader(std::vector<char*> filenameList) override;

	/*
	* Read the whole file into memory until it reaches the limit.
	*/
	void ReadChunk(FixedSizedCharArray &charArray, unsigned int readLen) override;

	/*
	* Read the whole file into memory
	*/
	char* ReadAll(char* fileName) override;
};