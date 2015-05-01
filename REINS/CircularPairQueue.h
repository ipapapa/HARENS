#pragma once
#include "Definition.h"
using namespace std;

/*
We have 2 threads accessing an object of this class simultaneously
one for push, the other for pop
*/
template <class S, class T>
class CircularPairQueue
{
public:
	S* firstQ;
	T* secondQ;
	int front, rear;	//rear point to the last used entry, there's an empty entry after rear
	uint size;
	mutex contentMutex;
	condition_variable contentCond;

	CircularPairQueue() {
		size = TEST_MAX_KERNEL_INPUT_LEN;
		firstQ = new S[size];
		secondQ = new T[size];
		front = 0;
		rear = size - 1;
		//The full situation is front == (rear + 2) % size
	}

	CircularPairQueue(int _size) {
		size = _size;
		firstQ = new S[size];
		secondQ = new T[size];
		front = 0;
		rear = size - 1;
		//The full situation is front == (rear + 2) % size
	}

	/*CircularUintQueue& operator=(CircularUintQueue obj) {
	this->queue = obj.queue;
	this->front = obj.front;
	this->rear = obj.rear;
	this->size = obj.size;
	return *this;
	}*/

	void Push(S firstHashValue, T secondHashValue) {
		//Make sure that the queue is not full
		unique_lock<mutex> contentLock(contentMutex);
		if ((rear + 2) % size == front) {
			contentCond.wait(contentLock);
		}
		contentLock.unlock();

		rear = (rear + 1) % size;
		firstQ[rear] = firstHashValue;
		secondQ[rear] = secondHashValue;
		//notify pop that one entry is added into queue
		contentCond.notify_one();
	}

	void Pop(S& firstHashValue, T& secondHashValue) {
		//Make sure that the queue is not empty
		unique_lock<mutex> contentLock(contentMutex);
		if ((rear + 1) % size == front) {
			contentCond.wait(contentLock);
		}
		contentLock.unlock();

		firstHashValue = firstQ[front];
		secondHashValue = secondQ[front];
		front = (front + 1) % size;
		contentCond.notify_one();
	}

	bool IsEmpty() {
		bool isEmpty;
		unique_lock<mutex> contentLock(contentMutex);
		isEmpty = ((rear + 1) % size == front);
		contentLock.unlock();
		contentCond.notify_one();
		return isEmpty;
	}

	~CircularPairQueue() {
		delete[] firstQ;
		delete[] secondQ;
	}
};

