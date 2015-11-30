#pragma once
#include "Definition.h"

/*
* A fixed sized buffer for strings
*/
class FixedSizedCharArray
{
private:
	char* arr;
	unsigned int arrSize;			//size of char array
	unsigned int contentSize;		//size of char array content

	//Store the extra len that cannot fit in arr
	char* buffer;
	const unsigned int MIN_BUFFER_SIZE = 1024;
	unsigned int bufferSize;		//size of the buffer array
	unsigned int bufferContentSize;	//size of the buffer content

	/*
	* Return true when buffer is empty, otherwise return false. 
	* Make sure arrLenLimit is no larger than arrSize
	*/
	bool CpyBufferToArr(unsigned int arrLenLimit);

public:
	FixedSizedCharArray(unsigned int size);
	~FixedSizedCharArray();

	/*
	* Return true when buffer is empty, otherwise return false.
	* Make sure arrLenLimit is no larger than arrSize
	*/
	bool Append(char* other, int otherSize, unsigned int arrLenLimit);

	char* GetArr();
	unsigned int GetLen();

	/*
	* Return true when buffer is empty, otherwise return false.
	* Make sure arrLenLimit is no larger than arrSize
	*/
	bool ClearArr(unsigned int arrLenLimit);
};

