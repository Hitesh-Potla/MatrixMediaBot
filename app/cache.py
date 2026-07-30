from collections import OrderedDict
from typing import Optional, Dict, Any

class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False
        self.data: Optional[Dict[str, Any]] = None


class PrefixCache:
    """
    A prefix matching cache using a Trie with LRU eviction.
    It caches generated responses for specific queries and serves them if a new query
    has a significant cached prefix.
    """
    def __init__(self, max_size: int = 1000, min_match_ratio: float = 0.8, min_match_length: int = 15):
        self.root = TrieNode()
        self.max_size = max_size
        self.min_match_ratio = min_match_ratio
        self.min_match_length = min_match_length
        self.size = 0
        self.lru = OrderedDict()

    def insert(self, query: str, answer: str, sources: list):
        query = query.lower().strip()
        if not query:
            return

        if query in self.lru:
            self.lru.move_to_end(query)
            # Update data
            node = self._get_node(query)
            if node:
                node.data = {"answer": answer, "sources": sources}
            return

        if self.size >= self.max_size:
            oldest_key, _ = self.lru.popitem(last=False)
            self._delete(oldest_key)
            self.size -= 1

        node = self.root
        for char in query:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        
        node.is_end_of_word = True
        node.data = {"answer": answer, "sources": sources}
        self.lru[query] = True
        self.size += 1

    def _get_node(self, query: str) -> Optional[TrieNode]:
        node = self.root
        for char in query:
            if char not in node.children:
                return None
            node = node.children[char]
        return node if node.is_end_of_word else None

    def _delete(self, query: str):
        node = self._get_node(query)
        if node:
            node.is_end_of_word = False
            node.data = None

    def search_prefix(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Searches for a cached query that is a prefix of the given query.
        Returns the cached data if a valid prefix match is found.
        """
        query = query.lower().strip()
        node = self.root
        last_match = None
        matched_prefix = ""
        last_match_prefix = ""

        for char in query:
            if char in node.children:
                node = node.children[char]
                matched_prefix += char
                if node.is_end_of_word:
                    last_match = node.data
                    last_match_prefix = matched_prefix
            else:
                break

        if last_match:
            # Check if the prefix match is substantial enough
            # It must be at least `min_match_length` characters OR it's an exact match
            is_exact = len(last_match_prefix) == len(query)
            is_long_enough = len(last_match_prefix) >= self.min_match_length
            is_high_ratio = len(last_match_prefix) >= len(query) * self.min_match_ratio

            if is_exact or (is_long_enough and is_high_ratio):
                self.lru.move_to_end(last_match_prefix)
                return last_match

        return None
