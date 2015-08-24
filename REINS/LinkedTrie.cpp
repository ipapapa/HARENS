#include "LinkedTrie.h"


LinkedTrie::LinkedTrie()
{
	root = new LinkedTrieNode();
	oldestInput = nullptr;
	latestInput = nullptr;
	size = 0;
}


LinkedTrie::~LinkedTrie()
{
	root->Clean();
}

/*return true if str already exists, otherwise return false*/
bool LinkedTrie::FindAndPut(char* str, int len) {
	LinkedTrieNode* curNode = root;
	//Add string to Trie
	for (int i = 0; i < len; ++i) {
		LinkedTrieNode* nodeFound = nullptr;
		for (int j = 0; j < curNode->children.size(); ++j) {
			if (curNode->children[j]->val == str[i])
				nodeFound = curNode->children[j];
		}
		if (nodeFound == nullptr) {	//Node not found
			LinkedTrieNode* child = new LinkedTrieNode(str[i], curNode);
			curNode->children.push_back(child);
			curNode = child;
			size += 1;
		}
		else {										//Node found
			curNode = nodeFound;
		}
	}

	if (curNode->prev == nullptr && curNode->next == nullptr) {	//If curNode is newly added
		//Add string into LinkedList
		if (oldestInput == nullptr) {	//LinkedList is empty
			oldestInput = latestInput = curNode;
		}
		else {							//LinkedList is not empty
			latestInput->next = curNode;
			curNode->prev = latestInput;
			latestInput = curNode;
		}

		if (size > MAX_SIZE) {
			//Remove some string
			LinkedTrieNode* nodePtr = oldestInput;
			while (size > MAX_SIZE && nodePtr != nullptr) {
				if (!nodePtr->children.empty()) {	//Cannot remove non-leaf nodes
					nodePtr = nodePtr->next;
					continue;
				}
				
				LinkedTrieNode* toBeDel = nodePtr;
				//Remove it from LinkedList
				if (nodePtr == oldestInput)
					oldestInput = nodePtr->next;
				else
					nodePtr->prev->next = nodePtr->next;
				if (nodePtr == latestInput)
					latestInput = nodePtr->prev;
				else
					nodePtr->next->prev = nodePtr->prev;
				//Remove it from Trie
				do {
					char toBeDelVal = toBeDel->val;
					toBeDel = toBeDel->parent;
					delete toBeDel->children[toBeDelVal];
					//toBeDel->children.erase(toBeDelVal);
					--size;
				} while (toBeDel->prev != nullptr || toBeDel->next != nullptr || toBeDel == root);
			}
		}
		return false;
	}
	else {														//If curNode already exists in LinkedList
		if (curNode == latestInput)
			return true;
		
		if (curNode == oldestInput) {
			oldestInput = curNode->next;
		}
		else {
			curNode->prev->next = curNode->next;
		}
		curNode->next = nullptr;
		curNode->prev = latestInput;
		latestInput->next = curNode;
		latestInput = curNode;
	}
}