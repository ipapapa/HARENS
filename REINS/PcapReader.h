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
#define LITTLE_ENDIAN 0x41424344UL 
#define BIG_ENDIAN    0x44434241UL
#define PDP_ENDIAN    0x42414443UL
#define ENDIAN_ORDER  ('ABCD')
struct ip
{
#if ENDIAN_ORDER == LITTLE_ENDIAN
	unsigned int ip_hl : 4;		/* header length */
	unsigned int ip_v : 4;		/* version */
#elif ENDIAN_ORDER == BIG_ENDIAN
	unsigned int ip_v : 4;		/* version */
	unsigned int ip_hl : 4;		/* header length */
#else
	#pragma message ("This machine is PDP. Terminate the program now!")
#endif
	u_int8_t ip_tos;			/* type of service */
	u_short ip_len;			/* total length */
	u_short ip_id;			/* identification */
	u_short ip_off;			/* fragment offset field */
#define	IP_RF 0x8000			/* reserved fragment flag */
#define	IP_DF 0x4000			/* dont fragment flag */
#define	IP_MF 0x2000			/* more fragments flag */
#define	IP_OFFMASK 0x1fff		/* mask for fragmenting bits */
	u_int8_t ip_ttl;			/* time to live */
	u_int8_t ip_p;			/* protocol */
	u_short ip_sum;			/* checksum */
	struct in_addr ip_src, ip_dst;	/* source and dest address */
};
#else	//UN*X system
#include <netinet/ip.h>
#include <arpa/inet.h>
#endif
//defines for the packet type code in an ETHERNET header
#define ETHER_TYPE_IP (0x0800)
#define ETHER_TYPE_8021Q (0x8100)

class PcapReader {
private:
	//static unsigned char* payload; /* Packet payload */

	/* Ethernet header */
	struct sniff_ethernet {
		unsigned char ether_dhost[ETHER_ADDR_LEN]; /* Destination host address */
		unsigned char ether_shost[ETHER_ADDR_LEN]; /* Source host address */
		unsigned short ether_type; /* IP? ARP? RARP? etc */
	};

	/* IP header */
	struct sniff_ip {
		unsigned char ip_vhl;		/* version << 4 | header length >> 2 */
		unsigned char ip_tos;		/* type of service */
		unsigned short ip_len;		/* total length */
		unsigned short ip_id;		/* identification */
		unsigned short ip_off;		/* fragment offset field */
		#define IP_RF 0x8000		/* reserved fragment flag */
		#define IP_DF 0x4000		/* dont fragment flag */
		#define IP_MF 0x2000		/* more fragments flag */
		#define IP_OFFMASK 0x1fff	/* mask for fragmenting bits */
		unsigned char ip_ttl;		/* time to live */
		unsigned char ip_p;		/* protocol */
		unsigned short ip_sum;		/* checksum */
		struct in_addr ip_src, ip_dst; /* source and dest address */
	};
	#define IP_HL(ip)		(((ip)->ip_vhl) & 0x0f)
	#define IP_V(ip)		(((ip)->ip_vhl) >> 4)

	/* TCP header */
	typedef unsigned int tcp_seq;

	struct sniff_tcp {
		unsigned short th_sport;	/* source port */
		unsigned short th_dport;	/* destination port */
		tcp_seq th_seq;		/* sequence number */
		tcp_seq th_ack;		/* acknowledgement number */
		unsigned char th_offx2;	/* data offset, rsvd */
		#define TH_OFF(th)	(((th)->th_offx2 & 0xf0) >> 4)
				unsigned char th_flags;
		#define TH_FIN 0x01
		#define TH_SYN 0x02
		#define TH_RST 0x04
		#define TH_PUSH 0x08
		#define TH_ACK 0x10
		#define TH_URG 0x20
		#define TH_ECE 0x40
		#define TH_CWR 0x80
		#define TH_FLAGS (TH_FIN|TH_SYN|TH_RST|TH_ACK|TH_URG|TH_ECE|TH_CWR)
		unsigned short th_win;		/* window */
		unsigned short th_sum;		/* checksum */
		unsigned short th_urp;		/* urgent pointer */
	};

public:
	//static void Deframe(unsigned char* useless, const struct pcap_pkthdr* pkthdr, const unsigned char* packet);
	static void PcapReader::ReadPcapFile(char* fileName, char* &payload, unsigned int &payloadLen);
};