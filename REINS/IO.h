#pragma once
#include <stdio.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <cstdarg> 

enum FileFormat { PlainText, Pcap, UnkownTest };

static class ExtraOrdinaryLargeFileException : public std::exception {
	virtual const char* what() const throw() {
		return "File should not be larger than 1 PB";
	}
} extraOrdinaryLargeFileException;

static class IO {
public:
	static FileFormat FILE_FORMAT;
	static const std::string METRICS[];
	static char* input_file_name;
	static char* output_file_name;

	static void Print(const char* format, ...);

	static std::string InterpretSize(int file_len);
};

