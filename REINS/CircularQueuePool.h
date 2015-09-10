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
	const unsigned int queueSize;
	mutex *frontMutex, *rearMutex; 
	//Size control
	unsigned int *curQueueSize;
	mutex *curQueueSizeMutex;
	condition_variable *emptyCond, *fullCond;

	CircularQueuePool(int _poolSize) : queueSize(TEST_MAX_KERNEL_INPUT_LEN) {
		Initiate(_poolSize);
	}

	CircularQueuePool(int _poolSize, int _size) : queueSize(_size) {
		Initiate(_poolSize);
	}

	void Initiate(int _poolSize) {
		poolSize = _poolSize;
		queuePool = new T*[poolSize];
		front = new int[poolSize];
		rear = new int[poolSize];
		frontMutex = new mutex[poolSize];
		rearMutex = new mutex[poolSize];

		curQueueSize = new unsigned int[poolSize];
		curQueueSizeMutex = new mutex[poolSize];
		emptyCond = new condition_variable[poolSize];
		fullCond = new condition_variable[poolSize];

		for (int i = 0; i < poolSize; ++i) {
			queuePool[i] = new T[queueSize];
			front[i] = 0;
			rear[i] = queueSize - 1;
			curQueueSize[i] = 0;
		}
		//The full situation is front == (rear + 2) % size
	}

	void Push(T hashValue, int (*mod)(T, int)) {
		//Make sure that the queue is not full
		int poolAnchor = mod(hashValue, poolSize);
		unique_lock<mutex> sizeLock(curQueueSizeMutex[poolAnchor]);
		while (curQueueSize[poolAnchor] >= queueSize) {
			fullCond[poolAnchor].wait(sizeLock);
		}
		//sizeLock.unlock();

		//rearMutex[poolAnchor].lock();
		rear[poolAnchor] = (rear[poolAnchor] + 1) % queueSize;
		queuePool[poolAnchor][rear[poolAnchor]] = hashValue;
		//rearMutex[poolAnchor].unlock();

		//sizeLock.lock();
		++curQueueSize[poolAnchor];
		sizeLock.unlock();

		//notify pop that one entry is added into queue
		emptyCond[poolAnchor].notify_one();
	}

	T Pop(int poolAnchor) {
		//Make sure that the queue is not empty
		unique_lock<mutex> sizeLock(curQueueSizeMutex[poolAnchor]);
		while (curQueueSize[poolAnchor] <= 0) {
			emptyCond[poolAnchor].wait(sizeLock);
		}
		//sizeLock.unlock();

		//frontMutex[poolAnchor].lock();
		T ret = queuePool[poolAnchor][front[poolAnchor]];
		front[poolAnchor] = (front[poolAnchor] + 1) % queueSize;
		//frontMutex[poolAnchor].unlock();

		//sizeLock.lock();
		--curQueueSize[poolAnchor];
		sizeLock.unlock();

		fullCond[poolAnchor].notify_one();
		return ret;
	}

	bool IsEmpty(int poolAnchor) {
		unique_lock<mutex> sizeLock(curQueueSizeMutex[poolAnchor]);
		bool isEmpty = (curQueueSize[poolAnchor] <= 0);
		sizeLock.unlock();
		return isEmpty;
	}

	~CircularQueuePool() {
		for (int i = 0; i < poolSize; ++i) {
			delete[] queuePool[i];
		}
		delete[] queuePool;
		delete[] front;
		delete[] rear;
		delete[] frontMutex;
		delete[] rearMutex;
		delete[] curQueueSize;
		delete[] curQueueSizeMutex;
		delete[] emptyCond;
		delete[] fullCond;
	}
};

