#include "PcapReader.h"

//void PcapReader::Deframe(unsigned char* useless, const struct pcap_pkthdr* pkthdr, const unsigned char* packet) {
//#define SIZE_ETHERNET 14
//	const struct sniff_ethernet *ethernet; /* The ethernet header */
//	const struct sniff_ip *ip; /* The IP header */
//	const struct sniff_tcp *tcp; /* The TCP header */
//
//	unsigned int size_ip;
//	unsigned int size_tcp;
//
//	ethernet = (struct sniff_ethernet*)(packet);
//	ip = (struct sniff_ip*)(packet + SIZE_ETHERNET);
//	size_ip = IP_HL(ip) * 4;
//	if (size_ip < 20) {
//		printf("   * Invalid IP header length: %u bytes\n", size_ip);
//		return;
//	}
//	tcp = (struct sniff_tcp*)(packet + SIZE_ETHERNET + size_ip);
//	size_tcp = TH_OFF(tcp) * 4;
//	if (size_tcp < 20) {
//		printf("   * Invalid TCP header length: %u bytes\n", size_tcp);
//		return;
//	}
//	payload = (unsigned char *)(packet + SIZE_ETHERNET + size_ip + size_tcp);
//}

void PcapReader::ReadPcapFile(char* fileName, char* &payload, unsigned int &payloadLen) {
#define SIZE_ETHERNET 14
	const struct sniff_ethernet *ethernet; /* The ethernet header */
	const struct sniff_ip *ip; /* The IP header */
	const struct sniff_tcp *tcp; /* The TCP header */

	unsigned int size_ip;
	unsigned int size_tcp;

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

	//----------------- 
	//begin processing the packets in this particular file, one at a time 
	while (res = pcap_next_ex(handle, &header, &packet) >= 0) {
		if (res == 0)	//the timeout set with pcap_open_live() has elapsed
			continue;

		ethernet = (struct sniff_ethernet*)(packet);
		ip = (struct sniff_ip*)(packet + SIZE_ETHERNET);
		size_ip = IP_HL(ip) * 4;
		if (size_ip < 20) {
			printf("   * Invalid IP header length: %u bytes\n", size_ip);
			continue;
		}
		tcp = (struct sniff_tcp*)(packet + SIZE_ETHERNET + size_ip);
		size_tcp = TH_OFF(tcp) * 4;
		if (size_tcp < 20) {
			printf("   * Invalid TCP header length: %u bytes\n", size_tcp);
			continue;
		} 
		fileContent.append((char *)(packet + SIZE_ETHERNET + size_ip + size_tcp));

	} //end internal loop for reading packets (all in one file) 

	pcap_close(handle);  //close the pcap file 

	payload = &fileContent[0];
	payloadLen = fileContent.length();
}