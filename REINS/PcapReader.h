#pragma once
/* Ethernet addresses are 6 bytes */
#define ETHER_ADDR_LEN	6

#include "Definition.h"
extern "C" {
#include <pcap.h>
#include <inaddr.h>
}
#if defined(WIN32) || defined(WIN64)
#include <winsock2.h>
#else	//UN*X system
#include <netinet/ip.h>
#include <arpa/inet.h>
#endif
//defines for the packet type code in an ETHERNET header
#define ETHER_TYPE_IP (0x0800)
#define ETHER_TYPE_8021Q (0x8100)

class PcapReader {
private:
	static char* Deframe(const unsigned char* packet);

public:
	static std::string PcapReader::ReadPcapFile(char* fileName);
};