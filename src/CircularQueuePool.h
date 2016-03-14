#pragma once
#include "Definition.h"
using namespace std;

/*
* We have 2 threads accessing an object of this class simultaneously
* one for push, the other for pop
*/
class CircularQueuePool
{
public:
	//Pool of data
	unsigned char*** chunkHashQueuePool;
	unsigned int** chunkLenQueuePool;
	int poolSize;					//Number of queues in the pool
	int *front, *rear;				/*Rear point to the last used entry, 
									there's an empty entry after rear*/
	const unsigned int queueSize;	//Number of entries in a queue
	//Access control - prevent reading/writing when empty/full
	unsigned int *curQueueSize;
	mutex *curQueueSizeMutex;
	condition_variable *emptyCond, *fullCond;

	/*
	* Initiaze the queue pool.
	* 80000 is a number that is big enough to keep it non-blocking.
	*/
	CircularQueuePool(int _poolSize) : queueSize(800000) {
		Initiate(_poolSize);
	}

	CircularQueuePool(int _poolSize, int _size) : queueSize(_size) {
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
				//Allocate memory for the queue pool
				chunkHashQueuePool[i][j] = new unsigned char[SHA1_HASH_LENGTH];
			}
			chunkLenQueuePool[i] = new unsigned int[queueSize];
			front[i] = 0;
			rear[i] = queueSize - 1;
			curQueueSize[i] = 0;
		}
		//The full situation is front == (rear + 2) % size
	}

	/*
	* Push a hash value into the hash pool.
	* Pause and wait for notification if the corresponding queue is full.
	*/
	void Push(unsigned char* hashValue, unsigned int chunkLen, int(*mod)(unsigned char*, int)) {
		//Make sure that the queue is not full
		int poolAnchor = mod(hashValue, poolSize);
		unique_lock<mutex> sizeLock(curQueueSizeMutex[poolAnchor]);
		while (curQueueSize[poolAnchor] >= queueSize) {
			fullCond[poolAnchor].wait(sizeLock);
		}
		//Do push
		rear[poolAnchor] = (rear[poolAnchor] + 1) % queueSize;
		memcpy(chunkHashQueuePool[poolAnchor][rear[poolAnchor]], hashValue, SHA1_HASH_LENGTH);
		chunkLenQueuePool[poolAnchor][rear[poolAnchor]] = chunkLen;

		++curQueueSize[poolAnchor];
		sizeLock.unlock();

		//notify pop that one entry is added into queue
		emptyCond[poolAnchor].notify_one();
	}

	/*
	* Pop the oldest hash value out of a queue the hash pool.
	* The queue is defined by its index in the pool.
	* Pause and wait for notification if the corresponding queue is empty.
	*/
	tuple<unsigned char*, unsigned int> Pop(int poolAnchor) {
		//Make sure that the queue is not empty
		unique_lock<mutex> sizeLock(curQueueSizeMutex[poolAnchor]);
		while (curQueueSize[poolAnchor] <= 0) {
			emptyCond[poolAnchor].wait(sizeLock);
		}
		//Do pop
		tuple<unsigned char*, unsigned int> ret = make_tuple(
			chunkHashQueuePool[poolAnchor][front[poolAnchor]],
			chunkLenQueuePool[poolAnchor][front[poolAnchor]]);
		front[poolAnchor] = (front[poolAnchor] + 1) % queueSize;

		--curQueueSize[poolAnchor];
		sizeLock.unlock();

		fullCond[poolAnchor].notify_one();
		return ret;
	}

	/*
	* Check if a queue defined by its index in the pool is empty.
	*/
	bool IsEmpty(int poolAnchor) {
		unique_lock<mutex> sizeLock(curQueueSizeMutex[poolAnchor]);
		bool isEmpty = (curQueueSize[poolAnchor] <= 0);
		sizeLock.unlock();
		return isEmpty;
	}

	~CircularQueuePool() {
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
