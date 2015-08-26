#pragma once
#include "Definition.h"
using namespace std;

/*
We have 2 threads accessing an object of this class simultaneously
one for push, the other for pop
*/
template <class T>
class CircularQueuePool
{
public:
	T** queuePool;
	int poolSize;
	int *front, *rear;	//rear point to the last used entry, there's an empty entry after rear
	unsigned int *queueSize;
	mutex *contentMutex;
	condition_variable *contentCond;

	CircularQueuePool(int _poolSize) {
		poolSize = _poolSize;
		queuePool = new T*[poolSize];
		front = new int[poolSize];
		rear = new int[poolSize];
		queueSize = new unsigned int[poolSize];
		contentMutex = new mutex[poolSize];
		contentCond = new condition_variable[poolSize];

		for (int i = 0; i < poolSize; ++i) {
			queueSize[i] = TEST_MAX_KERNEL_INPUT_LEN;
			queuePool[i] = new T[queueSize[i]];
			front[i] = 0;
			rear[i] = queueSize[i] - 1;
		}
		//The full situation is front == (rear + 2) % size
	}

	CircularQueuePool(int _poolSize, int _size) {
		poolSize = _poolSize;
		queuePool = new T*[poolSize];
		front = new int[poolSize];
		rear = new int[poolSize];
		queueSize = new unsigned int[poolSize];
		contentMutex = new mutex[poolSize];
		contentCond = new condition_variable[poolSize];

		for (int i = 0; i < poolSize; ++i) {
			queueSize[i] = _size;
			queuePool[i] = new T[queueSize[i]];
			front[i] = 0;
			rear[i] = queueSize[i] - 1;
		}
		//The full situation is front == (rear + 2) % size
	}

	void Push(T hashValue, int (*mod)(T, int)) {
		//Make sure that the queue is not full
		int poolAnchor = mod(hashValue, poolSize);
		unique_lock<mutex> contentLock(contentMutex[poolAnchor]);
		while ((rear[poolAnchor] + 2) % queueSize[poolAnchor] == front[poolAnchor]) {
			contentCond[poolAnchor].wait(contentLock);
		}
		rear[poolAnchor] = (rear[poolAnchor] + 1) % queueSize[poolAnchor];
		queuePool[poolAnchor][rear[poolAnchor]] = hashValue;
		contentLock.unlock();

		//notify pop that one entry is added into queue
		contentCond[poolAnchor].notify_one();
	}

	T Pop(int poolAnchor) {
		//Make sure that the queue is not empty
		unique_lock<mutex> contentLock(contentMutex[poolAnchor]);
		while ((rear[poolAnchor] + 1) % queueSize[poolAnchor] == front[poolAnchor]) {
			contentCond[poolAnchor].wait(contentLock);
		}
		contentLock.unlock();

		T ret = queuePool[poolAnchor][front[poolAnchor]];
		front[poolAnchor] = (front[poolAnchor] + 1) % queueSize[poolAnchor];
		contentCond[poolAnchor].notify_one();
		return ret;
	}

	bool IsEmpty(int poolAnchor) {
		bool isEmpty;
		unique_lock<mutex> contentLock(contentMutex[poolAnchor]);
		isEmpty = ((rear[poolAnchor] + 1) % queueSize[poolAnchor] == front[poolAnchor]);
		contentLock.unlock();
		contentCond[poolAnchor].notify_one();
		return isEmpty;
	}

	~CircularQueuePool() {
		for (int i = 0; i < poolSize; ++i) {
			delete[] queuePool[i];
		}
		delete[] queuePool;
		delete[] front;
		delete[] rear;
		delete[] queueSize;
		delete[] contentMutex;
		delete[] contentCond;
	}
};

