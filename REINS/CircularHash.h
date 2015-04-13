#pragma once
#include <openssl/sha.h>
#include "VirtualHash.h"
#include "Definition.h"

class CircularHash : public VirtualHash {
private:
	CircularQueue circularQueue;
	charPtMap map;

public:
	CircularHash(uint _size);
	~CircularHash();
	uchar* Add(uchar* hashValue);
	bool Find(uchar* hashValue);
};

