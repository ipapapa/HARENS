#pragma once
#include "FixedSizedCharArray.h"

/*
* A base calss for plain text reader and pcap reader
*/
class VirtualReader {
protected:
	std::vector<char*> filenameList;
	int fileNum;
	int fileIdx;

	/*
	* Set up a file for the reader.
	* set a handle for pcap file, a file stream for plain file.
	*/
	virtual void SetupFile(char* filename) = 0;

public:
	/*
	* Set up the reader for a file List, and set up the first file
	*/
	virtual void SetupReader(std::vector<char*> filenameList) = 0;

	/*
	* Read the whole file into memory by packets until it reaches the limit.
	*/
	virtual void ReadChunk(FixedSizedCharArray &charArray, unsigned int readLen) = 0;

	/*
	* Read the whole file into memory, stored in string
	*/
	virtual char* ReadAll(char* filename) = 0;
};