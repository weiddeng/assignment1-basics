import json
import regex as re
from typing import Iterator, Iterable
import heapq
from constants import PAT_IN_CHUNK

# Note: Trie-based greedy tokenization is not exactly correct for BPE; merge order can make a difference.

class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []

        self._merge_priority = {merge: i for i, merge in enumerate(self.merges)}
        self.bytes_to_id = {v: k for k, v in self.vocab.items()}
        for special_token in self.special_tokens:
            special_token_bytes = special_token.encode('utf-8')
            if special_token_bytes not in self.bytes_to_id:
                next_id = len(self.vocab)
                self.vocab[next_id] = special_token_bytes
                self.bytes_to_id[special_token_bytes] = next_id

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        with open(vocab_filepath, 'r') as f:
            # json.load(f) swallows a JSON file and converts it into a Python dictionary
            vocab_raw = json.load(f)

        vocab = {}
        # "13": "\r"
        for k, v in vocab_raw.items():
            vocab[int(k)] = v.encode('utf-8')

        merges = []
        with open(merges_filepath, 'r') as f:
            # [Token A]<SEPARATOR>[Token B]\n
            # <SEPARATOR> is a space
            # TODO: Actually the correct way is to use something special just like <SEPARATOR> for separation,
            # as [Token B] may contain space. But let's pretend it works - it does!
            for line in f:
                str_1, str_2 = line.rstrip().rsplit(' ', 1)
                merges.append((str_1.encode('utf-8'), str_2.encode('utf-8')))

        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        result_ids = []

        if self.special_tokens:
            # Sort special tokens by length (longest first) to handle overlapping tokens correctly!
            sorted_tokens = sorted(self.special_tokens, key=len, reverse=True)
            pat = "|".join(re.escape(t) for t in sorted_tokens)
            # Use split with capturing group to keep separators
            parts = re.split(f"({pat})", text)

            for part in parts:
                if part in self.special_tokens:
                    special_bytes = part.encode('utf-8')
                    if special_bytes in self.bytes_to_id:
                        result_ids.append(self.bytes_to_id[special_bytes])
                elif part:
                    result_ids.extend(self._tokenize_chunk(part))
        else:
            result_ids.extend(self._tokenize_chunk(text))

        return result_ids

    def _encode_lazy(self, text: str) -> Iterator[int]:
        if self.special_tokens:
            # Sort special tokens by length (longest first) to handle overlapping tokens correctly!
            sorted_tokens = sorted(self.special_tokens, key=len, reverse=True)
            pat = "|".join(re.escape(t) for t in sorted_tokens)
            # Use split with capturing group to keep separators
            parts = re.split(f"({pat})", text)

            for part in parts:
                if part in self.special_tokens:
                    special_bytes = part.encode('utf-8')
                    if special_bytes in self.bytes_to_id:
                        yield self.bytes_to_id[special_bytes]
                elif part:
                    yield from self._tokenize_chunk_lazy(part)
        else:
            yield from self._tokenize_chunk_lazy(text)

    def _tokenize_chunk(self, text_chunk: str) -> list[int]:
        result_ids = []
        # Pre-tokenize using GPT-2 pattern
        for match in re.finditer(PAT_IN_CHUNK, text_chunk):
            word = match.group().encode('utf-8')
            tokens = self._apply_bpe_to_word(word)
            result_ids.extend(tokens)
        return result_ids

    def _tokenize_chunk_lazy(self, text_chunk: str) -> Iterator[int]:
        # Pre-tokenize using GPT-2 pattern
        for match in re.finditer(PAT_IN_CHUNK, text_chunk):
            word = match.group().encode('utf-8')
            tokens = self._apply_bpe_to_word(word)
            yield from tokens

    def _apply_bpe_to_word(self, word: bytes) -> list[int]:
        """
        Apply BPE merges to a single word using a Doubly Linked List and a Priority Queue.
        """
        # Define Node for Doubly Linked List
        class _Node:
            __slots__ = ('data', 'prev', 'next', 'is_active')  # Saves memory
            def __init__(self, data: bytes):
                self.data = data
                self.prev: '_Node' | None = None
                self.next: '_Node' | None = None
                self.is_active = True  # To handle stale pairs in the priority queue

        # Create sentinel head/tail nodes to simplify boundary conditions
        head = _Node(b'')
        tail = _Node(b'')
        head.next = tail
        tail.prev = head

        current_node = head
        for b in word:
            new_node = _Node(bytes([b]))
            current_node.next = new_node
            new_node.prev = current_node
            current_node = new_node
        current_node.next = tail
        tail.prev = current_node

        # Populate Priority Queue with potential merges
        pq = []
        fifo_tiebreaker = 0
        current_node = head.next
        while current_node and current_node.next != tail:
            pair = (current_node.data, current_node.next.data)
            if pair in self._merge_priority:
                priority = self._merge_priority[pair]
                # current_node is a reference to the *first* node of the pair
                heapq.heappush(pq, (priority, fifo_tiebreaker, pair, current_node))
                fifo_tiebreaker += 1
            current_node = current_node.next

        while pq:
            priority, _, original_pair, left_node = heapq.heappop(pq)

            right_node = left_node.next

            # A merge is stale if either of its nodes has already been merged,
            # OR if the data has changed (nodes were merged and recreated)!
            if (not left_node.is_active or not right_node or not right_node.is_active or
                left_node.data != original_pair[0] or right_node.data != original_pair[1]):
                continue

            # Perform merge
            left_node.is_active = False
            right_node.is_active = False

            merged_node = _Node(left_node.data + right_node.data)

            # Rewire the linked list to insert the new node and remove the old ones
            prev_node = left_node.prev
            next_node = right_node.next

            prev_node.next = merged_node
            merged_node.prev = prev_node
            merged_node.next = next_node
            next_node.prev = merged_node

            # Add new potential merges to the priority queue
            if merged_node.prev != head:
                new_pair_left = (merged_node.prev.data, merged_node.data)
                if new_pair_left in self._merge_priority:
                    new_priority = self._merge_priority[new_pair_left]
                    heapq.heappush(pq, (new_priority, fifo_tiebreaker, new_pair_left, merged_node.prev))
                    fifo_tiebreaker += 1

            if merged_node.next != tail:
                new_pair_right = (merged_node.data, merged_node.next.data)
                if new_pair_right in self._merge_priority:
                    new_priority = self._merge_priority[new_pair_right]
                    heapq.heappush(pq, (new_priority, fifo_tiebreaker, new_pair_right, merged_node))
                    fifo_tiebreaker += 1

        # Traverse the final list and convert byte tokens to IDs
        token_ids = []
        current_node = head.next
        while current_node != tail:
            token_bytes = current_node.data
            if token_bytes in self.bytes_to_id:
                token_ids.append(self.bytes_to_id[token_bytes])
            else:
                # Fallback for single bytes
                for b in token_bytes:
                    token_ids.append(self.bytes_to_id[bytes([b])])
            current_node = current_node.next

        return token_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self._encode_lazy(text)

    def decode(self, ids: list[int]) -> str:
        return b''.join([self.vocab.get(id, b'') for id in ids]).decode('utf-8', errors='replace')
