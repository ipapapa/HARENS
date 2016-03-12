#pragma once
#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <cstdarg> 
#include <cstring>
#include <deque>
#include <fstream>
#include <functional>
#include <iostream>
#include <list>
#include <map>
#include <mutex>
#include <openssl/md5.h>
#include <openssl/sha.h>
#include <queue>
#include <set>
#include <sstream>
#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <thread>
#include <time.h>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

extern "C" {
#include <pcap.h>
}
#if defined(WIN32) || defined(WIN64)
#include <winsock2.h>
#else	//UN*X system
#include <netinet/ip.h>
#include <arpa/inet.h>
#endif

//The number of bytes in C++ built-in types
const unsigned int BYTES_IN_INT = sizeof(int);
const unsigned int BYTES_IN_UINT = sizeof(unsigned int);
const unsigned int BYTES_IN_ULONG = sizeof(unsigned long long);
//Length of sliding window in objects/packets chunking step (in bytes)
const int WINDOW_SIZE = 12;
//while P = 2^k, fingerprint % P means fingerprint & P_MINUS (P - 1). We set P = 32 here
const int P_MINUS = 0x1F;	
const int MIN_CHUNK_LEN = 32;
/*The maximum chunk number that can store in cache
Make it 2^27, so that the cache would be about 4 GB*/
const unsigned long long MAX_CHUNK_NUM = 134217728;		//2^27

//The maximum block number in the server GPU
const int BLOCK_NUM = 4096;
//The maximum thread nubmer per block in the server GPU
const int THREAD_PER_BLOCK = 512;
//Mention! this is just the number of windows for each thread
//const int NUM_OF_WINDOWS_PER_THREAD = 4;
const unsigned int MAX_KERNEL_INPUT_LEN = BLOCK_NUM * THREAD_PER_BLOCK + WINDOW_SIZE - 1;

//The number of pagable buffer needed to read all the data into memory
const int PAGABLE_BUFFER_NUM = 5000;
//The number of fixed buffer needed to transfer data between pagable buffer to kernel memory (for CUDA)
const int FIXED_BUFFER_NUM = 3;
//Size of buffer
const unsigned int MAX_BUFFER_LEN = MAX_KERNEL_INPUT_LEN;
//number of windows in a buffer
const unsigned int MAX_WINDOW_NUM = MAX_BUFFER_LEN - WINDOW_SIZE + 1;
//number of buffer to store chunking results
const int RESULT_BUFFER_NUM = 3;