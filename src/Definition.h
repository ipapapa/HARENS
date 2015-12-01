#pragma once
#include <time.h>
#include <stdlib.h>
#include <set>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <array>
#include <list>
#include <deque>
#include <cstring>
#include <string>
#include <time.h>
#include <algorithm>
#include <mutex>
#include <thread>
#include <chrono>
#include <deque>
#include <condition_variable>
#include <openssl/sha.h>
#include <openssl/md5.h>

//The number of bytes in C++ built-in types
const unsigned int BYTES_IN_INT = sizeof(int);
const unsigned int BYTES_IN_UINT = sizeof(unsigned int);
const unsigned int BYTES_IN_ULONG = sizeof(unsigned long long);
//Length of sliding window in objects/packets chunking step (in bytes)
const int WINDOW_SIZE = 12;
//while P = 2^k, fingerprint % P means fingerprint & P_MINUS (P - 1). We set P = 32 here
const int P_MINUS = 0x1F;	
const int MIN_CHUNK_LEN = 32;
//The maximum chunk number that can store in cache
const unsigned int MAX_CHUNK_NUM = 8388608;		//2^23

//The maximum block number in the server GPU
const int BLOCK_NUM = 4096;
//The maximum thread nubmer per block in the server GPU
const int THREAD_PER_BLOCK = 512;
//Mention! this is just the number of windows for each thread
//const int NUM_OF_WINDOWS_PER_THREAD = 4;
const unsigned int MAX_KERNEL_INPUT_LEN = BLOCK_NUM * THREAD_PER_BLOCK + WINDOW_SIZE - 1;

//The number of pagable buffer needed to read all the data into memory
const int PAGABLE_BUFFER_NUM = 1000;
//The number of fixed buffer needed to transfer data between pagable buffer to kernel memory (for CUDA)
const int FIXED_BUFFER_NUM = 3;
//Size of buffer
const unsigned int MAX_BUFFER_LEN = MAX_KERNEL_INPUT_LEN;
//number of windows in a buffer
const unsigned int MAX_WINDOW_NUM = MAX_BUFFER_LEN - WINDOW_SIZE + 1;
//number of buffer to store chunking results
const int RESULT_BUFFER_NUM = 3;