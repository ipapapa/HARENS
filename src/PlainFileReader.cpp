#include "PlainFileReader.h"

/*
* Set up a file stream for plain file.
*/
void PlainFileReader::SetupFile(char* filename) {
	ifs = std::ifstream(filename, std::ios::in | std::ios::binary | std::ios::ate);
	if (!ifs.is_open()) {
		fprintf(stderr, "Cannot open file %s\n", filename);
		exit(EXIT_FAILURE);
	}
	fileLen = ifs.tellg();
	ifs.seekg(0, ifs.beg);
	curFilePos = 0;
}

/*
* Set up the reader for a file List, and set up the first file
* Make sure the filenameList is not empty.
*/
void PlainFileReader::SetupReader(std::vector<char*> _filenameList) {
	filenameList = _filenameList;
	fileNum = filenameList.size();
	fileIdx = 0;
	SetupFile(filenameList[fileIdx]);
}

/*
* Read the whole file into memory until it reaches the limit.
*/
void PlainFileReader::ReadChunk(FixedSizedCharArray &charArray, unsigned int readLen) {
	//Clear char array
	charArray.ClearArr(readLen);
	unsigned int lenLimit = readLen;
	//Fit the read length if there's not this much content
	while (fileLen - curFilePos < readLen) {
		if (fileIdx == fileNum - 1) { //No more files
			readLen = fileLen - curFilePos;
			break;
		}
		else {						  //Read current file and move on the the next one
			//Read current file
			ifs.read(buffer, fileLen - curFilePos);
			charArray.Append(buffer, fileLen - curFilePos, fileLen - curFilePos);
			readLen -= fileLen - curFilePos;
			//Set up for next file
			SetupFile(filenameList[++fileIdx]);
		}
	}
	curFilePos += readLen;
	
	ifs.read(buffer, readLen);
	charArray.Append(buffer, readLen, lenLimit);
}

/*
* Read the whole file into memory
*/
char* PlainFileReader::ReadAll(char* filename) {
	ifs = std::ifstream(filename, std::ios::in | std::ios::binary | std::ios::ate);
	if (!ifs.is_open()) {
		fprintf(stderr, "Cannot open file %s\n", filename);
		exit(EXIT_FAILURE);
	}
	fileLen = ifs.tellg();
	ifs.seekg(0, ifs.beg);
	curFilePos = 0;
	char* fileContent = new char[fileLen];
	ifs.read(fileContent, fileLen);
	return fileContent;
}