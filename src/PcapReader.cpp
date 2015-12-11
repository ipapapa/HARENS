#include "PcapReader.h"

/*
* Remove the frame of data to get the load-offs
*/
std::pair<char*, int> PcapReader::Deframe(const unsigned char* packet, int frameLen) {
	unsigned int sizeIp;
	unsigned int sizeTcpUdp;

	//ethernet packet
	unsigned char* pkgPtr = (unsigned char*)packet;
	int etherType;
	
	if(WLAN_DATA) {		//If this is a 802.11 frame header
		//The 3rd and 4th bits in the first byte of 802.11 header is type
		int wlanType = ((int)pkgPtr[0] & MASK_00001100) >> 2;

		if(wlanType == 2)	{	//When the two bits in type == 10, it's data frame 
			if (!proceed(pkgPtr, frameLen, WLAN_HEADER_LEN))	//Skip 802.11 frame header
				return EMPTY;
			unsigned char dsap = pkgPtr[0];
			int lowerTwoBitsInControlField = (int)(pkgPtr[2] & MASK_00000011);
			if (lowerTwoBitsInControlField == 3) {//lower two bits are 1 and 1
				if (!proceed(pkgPtr, frameLen, 3))	//control field 1 byte
					return EMPTY;
			}
			else {
				if (!proceed(pkgPtr, frameLen, 4))	//control field 2 byte
					return EMPTY;
			}

			if (dsap == SNAP_EXTENSION_USED) {	//There's SNAP extension
				if ((((int)pkgPtr[0]) | ((int)pkgPtr[1]) | (int)pkgPtr[2]) == 0) {
					etherType = ((int)pkgPtr[3] << 8) | ((int)pkgPtr[4]);
					if (etherType == ETHER_TYPE_IP) {
						if (!proceed(pkgPtr, frameLen, 5))
							return EMPTY;
					}
					else {
						//fprintf(stderr, "Unknown ethernet type in 802.2 header, %04X, skipping...\n", etherType);
						return EMPTY;
					}
				}
				else {	//oui != 0
					//fprintf(stderr, "Unknown oui number %04X\n", (((int)pkgPtr[0] << 16) | ((int)pkgPtr[1] << 8) | (int)pkgPtr[2]));
					return EMPTY;
				}
			}
			else {	//no SNAP extension != 0
				//fprintf(stderr, "No SNAP extension. DSAP = %04X\n", dsap);
				return EMPTY;
			}
		}
		else {
			//fprintf(stderr, "WLAN frame type %u, not data frame...\n", wlanType);
			return EMPTY;
		}
	}
	else {				//If this is a 802.3 frame header
		//Ether type is the 13th and 14th byte in ethernet header when it's not 802.1q
		etherType = ((int)pkgPtr[12] << 8) | (int)pkgPtr[13];
		if (etherType == ETHER_TYPE_IP) {
			if (!proceed(pkgPtr, frameLen, 14))
				return EMPTY;
		}
		else if ((((int)pkgPtr[16] << 8) | (int)pkgPtr[17]) == ETHER_TYPE_8021Q) {
			if (!proceed(pkgPtr, frameLen, 18))
				return EMPTY;
		}
		else {
			//fprintf(stderr, "Unknown ethernet type, %04X, skipping...\n", etherType);
			return EMPTY;
		}
	}

	//ip packet
	sizeIp = ((int)pkgPtr[0] & MASK_00001111) * 4;	//The 5th to 8th bits of the first byte is IHL, the word(4 bytes) number in IP header.
	if (sizeIp < 20) {
		fprintf(stderr, "  * Invalid IP header length: %u bytes\n", sizeIp);
		return EMPTY;
	}
	int protocol = (int)pkgPtr[9];
	if (!proceed(pkgPtr, frameLen, sizeIp))
		return EMPTY;

	//tcp/udp packet
	if(protocol == 0x6) {	//It's tcp
		sizeTcpUdp = (pkgPtr[12] & MASK_11110000) >> 2;	//The first 4 bits of 12th byte in TCP header is the word(4 bytes) number in TCP header.
		if (sizeTcpUdp < 20) {
			fprintf(stderr, "    * Invalid TCP header length: %u bytes\n", sizeTcpUdp);
			return EMPTY;
		}
	}
	else if(protocol == 0x11) {	//It's udp
		sizeTcpUdp = ((int)(pkgPtr[4]) << 8) | (int)pkgPtr[5];
		if(sizeTcpUdp < 8){
			fprintf(stderr, "	* Invalid Udp header length: %u bytes\n", sizeTcpUdp);
			return EMPTY;
		}
	}
	else {
		//fprintf(stderr, "	* Unknown protocol number %u, neither TCP or UDP protocol\n", protocol);
		return EMPTY;
	}
	if (!proceed(pkgPtr, frameLen, sizeTcpUdp))
		return EMPTY;
	return std::make_pair((char *)pkgPtr, frameLen);
}

/*
* Return true when remaining length > 0, otherwise return false
*/
bool PcapReader::proceed(unsigned char* &ptr, int &remainingLen, int proceedLen) {
	ptr += proceedLen;
	remainingLen -= proceedLen;
	if (remainingLen > 0) {
		return true;
	}
	else {
		return false;
	}
}

/*
* Set up a pcap handle for pcap file.
*/
void PcapReader::SetupFile(char* filename) {
	char errbuf[PCAP_ERRBUF_SIZE]; //not sure what to do with this, oh well 
	handle = pcap_open_offline(filename, errbuf);   //call pcap library function 

	if (handle == NULL) {
		fprintf(stderr, "Couldn't open pcap file %s: %s\n", filename, errbuf);
		throw;
	}
}

/*
* Open pcap file and set a handle.
* Make sure the filenameList is not empty.
*/
void PcapReader::SetupReader(std::vector<char*> _filenameList) {
	filenameList = _filenameList;
	fileNum = filenameList.size();
	fileIdx = 0;
	//open the first pcap file 
	SetupFile(filenameList[fileIdx]);
}

/*
* Read the whole pcap file into memory by packets until it reaches the limit.
*/
void PcapReader::ReadChunk(FixedSizedCharArray &charArray, unsigned int readLenLimit) {
	//temporary packet buffers 
	struct pcap_pkthdr *header; // The header that pcap gives us 
	const u_char *packet; // The actual packet 
	std::string fileContent;
	int res;
	//Clear char array and put data (no more than readLenLimit) from arr buffer into arr 
	charArray.ClearArr(readLenLimit);

	//begin processing the packets in this particular file, one at a time 
	while (true) {
		res = pcap_next_ex(handle, &header, &packet);
		
		//Read to the end of file
		if (res < 0) {
			if (fileIdx < fileNum - 1) {
				//Move on to the next file
				SetupFile(filenameList[++fileIdx]);
				continue;
			}
			else {		//No more files
				break;
			}
		}

		if (res == 0)	//the timeout set with pcap_open_live() has elapsed
			continue;

		std::pair<char*, int> str_len_pair = Deframe(packet, header->caplen);
		if (str_len_pair.second > 0) {
			if (!charArray.Append(str_len_pair.first, str_len_pair.second, readLenLimit))
				break;
		}

	} //end internal loop for reading packets (all in one file) 
}

/*
* Read the whole pcap file into memory by packets
*/
char* PcapReader::ReadAll(char* fileName) {
	//temporary packet buffers 
	struct pcap_pkthdr *header; // The header that pcap gives us 
	const u_char *packet; // The actual packet 
	std::string fileContent;
	int res;

	//open the pcap file 
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

		std::pair<char*, int> str_len_pair = Deframe(packet, header->caplen);
		if (str_len_pair.second > 0)
			fileContent.append(str_len_pair.first);

	} //end internal loop for reading packets (all in one file) 

	pcap_close(handle);  //close the pcap file 

	return &fileContent[0];
}