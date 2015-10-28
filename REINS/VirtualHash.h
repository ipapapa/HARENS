#pragma once
#include "Definition.h"

class VirtualHash
{
protected:
	typedef std::unordered_map<unsigned char*, unsigned int, CharArrayHashFunc, CharArrayEqualTo> charPtMap;
	unsigned int size;

public:
	VirtualHash() {}
	VirtualHash(unsigned int _size);
	void SetupVirtualHash(unsigned int _size);
	~VirtualHash();

	virtual unsigned char* Add(unsigned char* hashValue, const bool isDuplicated) = 0;

	virtual bool Find(unsigned char* hashValue) = 0;

	virtual bool FindAndAdd(unsigned char* hashValue, unsigned char* toBeDel) = 0;
};