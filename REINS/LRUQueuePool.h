#pragma once
#include "Definition.h"
using namespace std;

/*
We have 2 threads accessing an object of this class simultaneously
one for push, the other for pop
*/
class LRUQueuePool
{
public:
	unsigned char*** chunkHashQueuePool;
	unsigned int** chunkLenQueuePool;
	int poolSize;
	int *front, *rear;	//rear point to the last used entry, there's an empty entry after rear
	const unsigned int queueSize;
	//Size control
	unsigned int *curQueueSize;
	mutex *curQueueSizeMutex;
	condition_variable *emptyCond, *fullCond;

	LRUQueuePool(int _poolSize) : queueSize(TEST_MAX_KERNEL_INPUT_LEN) {
		Initiate(_poolSize);
	}

	LRUQueuePool(int _poolSize, int _size) : queueSize(_size) {
		Initiate(_poolSize);
	}

	void Initiate(int _poolSize) {
		poolSize = _poolSize;
		chunkHashQueuePool = new unsigned char**[poolSize];
		chunkLenQueuePool = new unsigned int*[poolSize];
		front = new int[poolSize];
		rear = new int[poolSize];

		curQueueSize = new unsigned int[poolSize];
		curQueueSizeMutex = new mutex[poolSize];
		emptyCond = new condition_variable[poolSize];
		fullCond = new condition_variable[poolSize];

		for (int i = 0; i < poolSize; ++i) {
			chunkHashQueuePool[i] = new unsigned char*[queueSize];
			for (int j = 0; j < queueSize; ++j) {
				chunkHashQueuePool[i][j] = new unsigned char[SHA_DIGEST_LENGTH];
			}
			chunkLenQueuePool[i] = new unsigned int[queueSize];
			front[i] = 0;
			rear[i] = queueSize - 1;
			curQueueSize[i] = 0;
		}
		//The full situation is front == (rear + 2) % size
	}

	void Push(unsigned char* hashValue, unsigned int chunkLen, int(*mod)(unsigned char*, int)) {
		//Make sure that the queue is not full
		int poolAnchor = mod(hashValue, poolSize);
		unique_lock<mutex> sizeLock(curQueueSizeMutex[poolAnchor]);
		while (curQueueSize[poolAnchor] >= queueSize) {
			fullCond[poolAnchor].wait(sizeLock);
		}
		//sizeLock.unlock();

		//rearMutex[poolAnchor].lock();
		rear[poolAnchor] = (rear[poolAnchor] + 1) % queueSize;
		memcpy(chunkHashQueuePool[poolAnchor][rear[poolAnchor]], hashValue, SHA_DIGEST_LENGTH);
		chunkLenQueuePool[poolAnchor][rear[poolAnchor]] = chunkLen;
		//rearMutex[poolAnchor].unlock();

		//sizeLock.lock();
		++curQueueSize[poolAnchor];
		sizeLock.unlock();

		//notify pop that one entry is added into queue
		emptyCond[poolAnchor].notify_one();
	}

	tuple<unsigned char*, unsigned int> Pop(int poolAnchor) {
		//Make sure that the queue is not empty
		unique_lock<mutex> sizeLock(curQueueSizeMutex[poolAnchor]);
		while (curQueueSize[poolAnchor] <= 0) {
			emptyCond[poolAnchor].wait(sizeLock);
		}
		//sizeLock.unlock();

		//frontMutex[poolAnchor].lock();
		tuple<unsigned char*, unsigned int> ret = make_tuple(
			chunkHashQueuePool[poolAnchor][front[poolAnchor]],
			chunkLenQueuePool[poolAnchor][front[poolAnchor]]);
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

	~LRUQueuePool() {
		for (int i = 0; i < poolSize; ++i) {
			for (int j = 0; j < queueSize; ++j)
				delete[] chunkHashQueuePool[i][j];
			delete[] chunkHashQueuePool[i];
			delete[] chunkLenQueuePool[i];
		}
		delete[] chunkHashQueuePool;
		delete[] chunkLenQueuePool;
		delete[] front;
		delete[] rear;
		delete[] curQueueSize;
		delete[] curQueueSizeMutex;
		delete[] emptyCond;
		delete[] fullCond;
	}
};

