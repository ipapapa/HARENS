#pragma once
#include <unordered_map>
#include <vector>
#include <stack>

class LinkedTrie
{
	struct LinkedTrieNode {
		//As in Trie
		char val;
		LinkedTrieNode* parent;
		std::vector<LinkedTrieNode*> children;
		
		//As in list
		LinkedTrieNode* prev;
		LinkedTrieNode* next;

		LinkedTrieNode() {
			val = 0;
			parent = nullptr;
			prev = nullptr;
			next = nullptr;
		}

		LinkedTrieNode(char _val, LinkedTrieNode* _parent) {
			val = _val;
			parent = _parent;
			prev = nullptr;
			next = nullptr;
		}

		void Clean() {
			if (!children.empty()) {
				for (auto& child : children) {
					child->Clean();
				}
			}
			delete this;
		}
	} *root;

	size_t MAX_SIZE = 1024 * 1024 * 1024;
	size_t size;
	LinkedTrieNode* oldestInput;
	LinkedTrieNode* latestInput;

public:
	LinkedTrie();
	~LinkedTrie();

	/*return true if str already exists, otherwise return false*/
	bool FindAndPut(char* str, int len);
};

