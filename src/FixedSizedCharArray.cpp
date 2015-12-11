#include "FixedSizedCharArray.h"

FixedSizedCharArray::FixedSizedCharArray(unsigned int size)
{
	arrSize = size;
	arr = new char[arrSize];
	contentSize = 0;

	bufferSize = MIN_BUFFER_SIZE;
	buffer = new char[bufferSize];
	bufferContentSize = 0;
}

FixedSizedCharArray::~FixedSizedCharArray()
{
	delete[] arr;
	delete[] buffer;
}

/*
* Return true when buffer is empty, otherwise return false.
* Make sure arrLenLimit is no larger than arrSize
*/
bool FixedSizedCharArray::Append(char* other, unsigned int otherSize, unsigned int arrLenLimit) {
	if (bufferContentSize != 0) {
		if (!CpyBufferToArr(arrLenLimit))
			return false;
	}

	if (other == nullptr)
		return true;

	if (contentSize + otherSize <= arrLenLimit) {
		memcpy(&arr[contentSize], other, otherSize);
		contentSize += otherSize;
		return true;
	}
	else {
		memcpy(&arr[contentSize], other, arrLenLimit - contentSize);
		otherSize -= arrLenLimit - contentSize;
		if (bufferContentSize + otherSize <= bufferSize) {
			memcpy(&buffer[bufferContentSize], &other[arrLenLimit - contentSize], otherSize);
		}
		else {
			char* tmp = buffer;
			bufferSize = bufferContentSize + otherSize;
			buffer = new char[bufferSize];
			memcpy(buffer, tmp, bufferContentSize);
			delete[] tmp;
			
			memcpy(&buffer[bufferContentSize], &other[arrLenLimit - contentSize], otherSize);
		}
		bufferContentSize += otherSize;
		contentSize = arrLenLimit;
		return false;
	}
}

char* FixedSizedCharArray::GetArr() {
	return arr;
}

unsigned int FixedSizedCharArray::GetLen() {
	return contentSize;
}

/*
* Return true when buffer is empty, otherwise return false.
* Make sure arrLenLimit is no larger than arrSize
*/
bool FixedSizedCharArray::ClearArr(unsigned int arrLenLimit) {
	contentSize = 0;
	//The program only enters this block when called by pcap reader
	if (bufferContentSize != 0) {
		return CpyBufferToArr(arrLenLimit);
	}
	return true;
}

/*
* Return true when buffer is empty, otherwise return false.
* Make sure arrLenLimit is no larger than arrSize
*/
bool FixedSizedCharArray::CpyBufferToArr(unsigned int arrLenLimit) {
	if (contentSize + bufferContentSize <= arrLenLimit) {	//no need to leave a byte for end of string character in arr
		memcpy(&arr[contentSize], buffer, bufferContentSize);
		contentSize += bufferContentSize;
		bufferContentSize = 0;
		return true;
	}
	else {
		memcpy(&arr[contentSize], buffer, arrLenLimit - contentSize);
		bufferContentSize -= (arrLenLimit - contentSize);	//The size of content left in buffer
		memcpy(buffer, &buffer[arrLenLimit - contentSize], bufferContentSize);
		contentSize = arrLenLimit;
		return false;
	}
}