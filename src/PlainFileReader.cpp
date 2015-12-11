#include "PlainFileReader.h"

/*
* Open pcap file and set a handle
*/
void PlainFileReader::SetupReader(char* fileName) {
	ifs = std::ifstream(fileName, std::ios::in | std::ios::binary | std::ios::ate);
	if (!ifs.is_open()) {
		fprintf(stderr, "Cannot open file %s\n", fileName);
		exit(EXIT_FAILURE);
	}
	fileLen = ifs.tellg();
	ifs.seekg(0, ifs.beg);
	curFilePos = 0;
}

/*
* Read the whole pcap file into memory by packets until it reaches the limit.
*/
void PlainFileReader::ReadChunk(FixedSizedCharArray &charArray, unsigned int readLen) {
	//Fit the read length if there's not this much content
	if (fileLen - curFilePos < readLen) {
		readLen = fileLen - curFilePos;
	}
	curFilePos += readLen;
	charArray.ClearArr(readLen);

	ifs.read(buffer, readLen);
	charArray.Append(buffer, readLen, readLen);
}

/*
* Read the whole pcap file into memory by packets
*/
char* PlainFileReader::ReadAll(char* fileName) {
	char* fileContent = new char[fileLen];
	ifs.read(fileContent, fileLen);
	return fileContent;
}