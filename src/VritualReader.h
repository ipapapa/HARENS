#pragma once
#include "FixedSizedCharArray.h"

/*
* A base calss for plain text reader and pcap reader
*/
class VirtualReader {
public:
	/*
	* Set up the reader for a file - a handle for pcap file, a file stream for plain file.
	*/
	virtual void SetupReader(char* filename) = 0;

	/*
	* Read the whole file into memory by packets until it reaches the limit.
	*/
	virtual void ReadChunk(FixedSizedCharArray &charArray, unsigned int readLen) = 0;

	/*
	* Read the whole file into memory, stored in string
	*/
	virtual char* ReadAll(char* filename) = 0;
};