#pragma once
#include <openssl/sha.h>
#include "VirtualHash.h"
#include "CircularQueue.h"
#include "Definition.h"

class CircularHash : public VirtualHash {
private:
	CircularQueue<uchar*> circularQueue;
	charPtMap map;

public:
	CircularHash() {}
	CircularHash(uint _size);
	void SetupCircularHash(uint _size);
	~CircularHash();
	uchar* Add(uchar* hashValue);
	bool Find(uchar* hashValue);
};

