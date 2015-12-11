#pragma once
#include "VritualReader.h"

// Ethernet addresses are 6 bytes
#define ETHER_ADDR_LEN	6
//defines for the packet type code in an ETHERNET header
#define ETHER_TYPE_IP (0x0800)
#define ETHER_TYPE_8021Q (0x8100)
#define WLAN_HEADER_LEN (24)	//No Address 4 in 802.11 header in our track
//The masks of bytes
#define MASK_00000011 (0x03)
#define MASK_00001111 (0x0F)
#define MASK_00001100 (0x0C)
#define MASK_00110000 (0x30)
#define MASK_11110000 (0xF0)
#define WLAN_DATA (true)
#define SNAP_EXTENSION_USED (0xAA)
const std::string EMPTY_STR = "";
#define EMPTY (std::make_pair((char*)(&EMPTY_STR[0]), 0))

/*
* A reader of pcap file.
* A pcap file is a capture file saved in the format that libpcap and WinPcap use 
* can be read by applications that understand that format, such as tcpdump, Wireshark,
* CA NetMaster, or Microsoft Network Monitor 3.x. (wikipedia: pcap)
*/
class PcapReader: public VirtualReader {
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

protected:
	/*
	* Set up a pcap handle for pcap file.
	*/
	void SetupFile(char* filename) override;

public:
	/*
	* Set up the reader for a file List, and set up the first file.
	* Make sure the filenameList is not empty.
	*/
	void SetupReader(std::vector<char*> filenameList) override;
	
	/*
	* Read the whole pcap file into memory by packets until it reaches the limit.
	*/
	void ReadChunk(FixedSizedCharArray &charArray, unsigned int readLen) override;

	/*
	* Read the whole pcap file into memory by packets
	*/
	char* ReadAll(char* fileName) override;
};