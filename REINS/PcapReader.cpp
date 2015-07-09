#include "PcapReader.h"

char* PcapReader::Deframe(const unsigned char* packet) {
	int ethernetOffset;

	unsigned int sizeIp;
	unsigned int sizeTcp;

	//ethernet packet
	unsigned char* pkgPtr = (unsigned char*)packet;
	int ether_type = ((int)(pkgPtr[12]) << 8) | (int)pkgPtr[13];
	if (ether_type == ETHER_TYPE_IP)
		ethernetOffset = 14;
	else if ((((int)(pkgPtr[16]) << 8) | (int)pkgPtr[17]) == ETHER_TYPE_8021Q) {
		ethernetOffset = 18;
	}
	else {
		fprintf(stderr, "Unknown ethernet type, %04X, skipping...\n", ether_type);
		return "";
	}
	pkgPtr += ethernetOffset;

	//ip packet
	sizeIp = (pkgPtr[0] & 0xF) * 4;	//The 5th to 8th bits of the first byte is IHL, the word(4 bytes) number in IP header.
	if (sizeIp < 20) {
		printf("  * Invalid IP header length: %u bytes\n", sizeIp);
		return "";
	}
	pkgPtr += sizeIp;

	//tcp packet
	sizeTcp = (pkgPtr[12] & 0xF0) >> 2;	//The first 4 bits of 12th byte in TCP header is the word(4 bytes) number in TCP header.
	if (sizeTcp < 20) {
		printf("    * Invalid TCP header length: %u bytes\n", sizeTcp);
		return "";
	}
	pkgPtr += sizeTcp;

	return (char *)pkgPtr;
}

std::string PcapReader::ReadPcapFile(char* fileName) {
	//temporary packet buffers 
	struct pcap_pkthdr *header; // The header that pcap gives us 
	const u_char *packet; // The actual packet 
	std::string fileContent;
	int res;

	//open the pcap file 
	pcap_t *handle;
	char errbuf[PCAP_ERRBUF_SIZE]; //not sure what to do with this, oh well 
	handle = pcap_open_offline(fileName, errbuf);   //call pcap library function 

	if (handle == NULL) {
		fprintf(stderr, "Couldn't open pcap file %s: %s\n", fileName, errbuf);
		throw;
	}

	//begin processing the packets in this particular file, one at a time 
	while (res = pcap_next_ex(handle, &header, &packet) >= 0) {
		if (res == 0)	//the timeout set with pcap_open_live() has elapsed
			continue;

		fileContent.append(Deframe(packet));

	} //end internal loop for reading packets (all in one file) 

	pcap_close(handle);  //close the pcap file 

	return fileContent;
}