#pragma once
#include "Definition.h"

class SelfMantainedCircularQueue
{
public:
	unsigned char** queue;
	int front, rear;	//rear point to the last used entry, there's an empty entry after rear
	unsigned int size;

	SelfMantainedCircularQueue() {}

	SelfMantainedCircularQueue(int _size) {
		SetupCircularQueue(_size);
	}

	void SetupCircularQueue(int _size) {
		size = _size;
		queue = new unsigned char*[size];
		for (int i = 0; i < size; ++i)
			queue[i] = nullptr;
		front = 0;
		rear = size - 1;
	}

	unsigned char* Add(unsigned char* hashValue) {
		unsigned char* to_be_del = nullptr;
		if ((rear + 2) % size == front) {
			to_be_del = queue[front];
			front = (front + 1) % size;
		}
		rear = (rear + 1) % size;
		queue[rear] = hashValue;
		return to_be_del;
	}

	~SelfMantainedCircularQueue() {
		/*while ((rear + 1) % size == front) {
			delete[] queue[front];
			front = (front + 1) % size;
		}
		delete[] queue;*/
	}
};

