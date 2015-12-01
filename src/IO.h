#pragma once
#include <stdio.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <cstdarg> 

/*
* Files with plaintext format can be read in directly,
* while files with pcap format can only be read in via pcap reader.
*/
enum FileFormat { PlainText, Pcap, UnkownTest };

static class ExtraOrdinaryLargeFileException : public std::exception {
	virtual const char* what() const throw() {
		return "File should not be larger than 1 PB";
	}
} extraOrdinaryLargeFileException;

/*
* IO-related parameters and functions
*/
class IO {
public:
	static FileFormat FILE_FORMAT;
	static const std::string METRICS[];
	static char* input_file_name;
	static char* output_file_name;

	/*
	* Print contents into console or file according to the setting of output_file_name.
	* Accept parameters in the form of printf("format", ...)
	*/
	static void Print(const char* format, ...);

	/*
	* Interpret size in bytes into the metrics easy to read
	*/
	static std::string InterpretSize(int file_len);
};

