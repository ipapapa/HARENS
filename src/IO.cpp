#include "IO.h"

FileFormat IO::FILE_FORMAT = PlainText;
const std::string IO::METRICS[]{ "Bytes", "KB", "MB", "GB", "TB" };
char* IO::input_file_name = nullptr;
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
std::string IO::InterpretSize(int file_len) {
	for (int i = 0; i < 5; ++i) {
		if (file_len < 1000) {
			return std::to_string(file_len) + " " + METRICS[i];
		}
		file_len /= 1000;
	}
	throw extraOrdinaryLargeFileException;
}