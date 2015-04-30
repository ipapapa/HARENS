#pragma once
#include "Definition.h"

class SelfMantainedCircularQueue
{
public:
	uchar** queue;
	int front, rear;	//rear point to the last used entry, there's an empty entry after rear
	uint size;

	SelfMantainedCircularQueue() {}

	SelfMantainedCircularQueue(int _size) {
		size = _size;
		queue = new uchar*[size];
		front = 0;
		rear = size - 1;
	}

	void SetupCircularQueue(int _size) {
		size = _size;
		queue = new uchar*[size];
		front = 0;
		rear = size - 1;
	}

	uchar* Add(uchar* hashValue) {
		uchar* to_be_del = (uchar*)0;
		if ((rear + 2) % size == front) {
			to_be_del = queue[front];
			front = (front + 1) % size;
		}
		rear = (rear + 1) % size;
		queue[rear] = hashValue;
		return to_be_del;
	}

	~SelfMantainedCircularQueue() {
		/*while (front != rear) {
			delete[] queue[front];
			front = (front + 1) % size;
		}*/
		//delete[] queue;
	}
};

