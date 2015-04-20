#pragma once
#include <openssl/sha.h>
#include "VirtualHash.h"
#include "Definition.h"

class CircularHash : public VirtualHash {
private:
	CircularQueue circularQueue;
	charPtMap map;

public:
	CircularHash() {}
	CircularHash(uint _size);
	void SetupCircularHash(uint _size);
	~CircularHash();
	uchar* Add(uchar* hashValue);
	bool Find(uchar* hashValue);
};

