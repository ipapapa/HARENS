#include "IO.h"

FileFormat IO::fileFormat = PlainText;
VirtualReader* IO::fileReader = new PlainFileReader();
const std::string IO::METRICS[]{ "Bytes", "KB", "MB", "GB", "TB" };
std::vector<char*> IO::input_file_name;
int IO::fileIdx = 0;
char* IO::output_file_name = nullptr;

/*
* Print contents into console or file according to the setting of output_file_name.
* Accept parameters in the form of printf("format", ...)
*/
void IO::Print(const char* format, ...) {
	va_list args;
	va_start(args, format);
	if (output_file_name == nullptr)
		vprintf(format, args);
	else {
		FILE* outputFile = fopen(output_file_name, "a");
		vfprintf(outputFile, format, args);
		fclose(outputFile);
	}
	va_end(args);
}

/*
* Interpret size in bytes into the metrics easy to read
*/
const char* IO::InterpretSize(unsigned long long fileLen) {
	for (int i = 0; i < 5; ++i) {
		if (fileLen < 1000) {
			return (std::to_string(fileLen) + " " + METRICS[i]).c_str();
		}
		fileLen /= 1000;
	}
	throw extraOrdinaryLargeFileException;
}