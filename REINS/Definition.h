#pragma once
#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <set>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <array>
#include <list>
#include <deque>
#include <string>
#include <time.h>
#include <algorithm>
#include <mutex>
#include <thread>
#include <chrono>
#include <deque>
#include <condition_variable>
#include <openssl/sha.h>

static class ExtraOrdinaryLargeFileException : public std::exception {
	virtual const char* what() const throw() {
		return "File should not be larger than 1 PB";
	}
} extraOrdinaryLargeFileException;

const std::string METRICS[] {"Bytes", "KB", "MB", "GB", "TB"};
static std::string InterpretSize(int file_len) {
	for (int i = 0; i < 5; ++i) {
		if (file_len < 1000) {
			return std::to_string(file_len) + " " + METRICS[i];
		}
		file_len /= 1000;
	}
	throw extraOrdinaryLargeFileException;
}

const unsigned int BYTES_IN_INT = sizeof(int);
const unsigned int BYTES_IN_UINT = sizeof(unsigned int);
const unsigned int BYTES_IN_ULONG = sizeof(unsigned long long);
const unsigned int MAX_CHUNK_NUM = 8388608;		//2^23
const int WINDOW_SIZE = 12;
const int TEST_MAX_KERNEL_INPUT_LEN = 4096 * 128;	//For testing
//while P = 2^k, fingerprint % P means fingerprint & P_MINUS (P - 1). We set P = 32 here
const int P_MINUS = 0x1F;	
const int MIN_CHUNK_LEN = 32;
enum FileFormat {PlainText, Pcap, UnkownTest};
const std::string FILE_FORMAT_TEXT[] {"PlainText", "Pcap", "UnkownTest"};
const FileFormat FILE_FORMAT = PlainText;