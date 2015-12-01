#pragma once
/* Ethernet addresses are 6 bytes */
#define ETHER_ADDR_LEN	6

#include <utility>
#include "Definition.h"
#include "FixedSizedCharArray.h"
extern "C" {
#include <pcap.h>
//#include <inaddr.h>
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
#define WLAN_HEADER_LEN (24)	//No Address 4 in 802.11 header in our track
#define MASK_00000011 (0x03)
#define MASK_00001111 (0x0F)
#define MASK_00001100 (0x0C)
#define MASK_00110000 (0x30)
#define MASK_11110000 (0xF0)
#define WLAN_DATA (true)
#define SNAP_EXTENSION_USED (0xAA)
const std::string EMPTY_STR = "";
#define EMPTY (std::make_pair((char*)(&EMPTY_STR[0]), 0))

class PcapReader {
private:
	pcap_t* handle;

	/*
	* Remove the frame of data to get the load-offs
	*/
	std::pair<char*, int> Deframe(const unsigned char* packet, int frameLen);
	
	/*
	* Return true when remaining length > 0, otherwise return false
	*/
	bool proceed(unsigned char* &ptr, int &remainingLen, int proceedLen);

public:
	/*
	* Read the whole pcap file into memory by packets
	*/
	std::string ReadPcapFile(char* fileName);

	/*
	* Open pcap file and set a handle
	*/
	void SetupPcapHandle(char* fileName);
	
	/*
	* Read the whole pcap file into memory by packets until it reaches the limit.
	* Returns true when it reads something
	*/
	void ReadPcapFileChunk(FixedSizedCharArray &charArray, unsigned int readLen);
};