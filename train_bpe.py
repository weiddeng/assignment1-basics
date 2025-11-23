# vocab_size & 80-20 distribution

# Took 4h7m to run for owt_train, on Lambda workstation.
# Do NOT use scalene to profile. Would give rise to bug.

import regex as re
import json
from multiprocessing import Pool
from collections import Counter
from cs336_basics.pretokenization_example import find_chunk_boundaries
from constants import PAT_IN_CHUNK
NUM_PROCESSES = 4

def find_boundaries(input_path):
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, NUM_PROCESSES, b"<|endoftext|>")  # give rise to NUM_PROCESSES many chunks, or fewer in corner cases
        # print(boundaries)
        return boundaries

def process_big_chunk(input_path, pat_between_chunks, start, end):
    word_counts = Counter()
    with open(input_path, "rb") as f:
        f.seek(start)
        # Here ignore and replace are equivalent for our goal. Just think about it.
        big_chunk = f.read(end - start).decode("utf-8", errors="ignore")  # unicode string
        chunks = re.split(pat_between_chunks, big_chunk)
        for chunk in chunks:
            # Note the GPT-2 pattern is exhaustive. And match.group() may have a leading space
            for match in re.finditer(PAT_IN_CHUNK, chunk):
                word_counts[match.group().encode('utf-8')] += 1  # convert back to bytes
        return word_counts

def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    # vocab: dict[int, bytes] The tokenizer vocabulary, a mapping from int (token ID) to bytes (token bytes).
    # merges: list[tuple[bytes, bytes]] A list of BPE merges produced from training.

    # remove special tokens before BPE vocabulary construction
    PAT_BETWEEN_CHUNKS = "|".join(re.escape(t) for t in special_tokens)

    # list[int]
    boundaries = find_boundaries(input_path)

    with Pool(NUM_PROCESSES) as pool:
        word_counts_chunks = pool.starmap(process_big_chunk, [(input_path, PAT_BETWEEN_CHUNKS, s, e) for s, e in zip(boundaries[:-1], boundaries[1:])])

    total_word_counts = Counter()
    for word_count in word_counts_chunks:
        total_word_counts.update(word_count)
    # print(total_word_counts.most_common(50))

    # Build vocab bottom-up
    vocab = {i: bytes([i]) for i in range(256)}
    for i, token in enumerate(special_tokens):
        vocab[256+i] = token.encode("utf-8")

    merges = []

    # Stateful, updated after each merge
    word_splits = {}
    inverted_index = {}
    bp_counts = Counter()

    for word, freq in total_word_counts.items():
        word_split = [bytes([b]) for b in word]
        word_splits[word] = word_split

        for bp in zip(word_split[:-1], word_split[1:]):
            bp_counts[bp] += freq

            if bp in inverted_index:
                inverted_index[bp].add(word)
            else:
                inverted_index[bp] = set([word])

    # Now merge
    for index in range(256 + len(special_tokens), vocab_size):
        merge_candidates, highest_count = [], -1
        for bp, count in bp_counts.items():
            if count > highest_count:
                merge_candidates = [bp]
                highest_count = count
            elif count == highest_count:
                merge_candidates.append(bp)

        # --- THE FIX ---
        if not merge_candidates:
            print("No more pairs to merge. Stopping BPE training early.")
            break
        # --- END OF FIX ---

        bp_to_merge = max(merge_candidates)
        merges.append(bp_to_merge)
        bytes_merged = bp_to_merge[0] + bp_to_merge[1]
        vocab[index] = bytes_merged

        bp_counts_decre = Counter()
        # Now get ready for the next merge
        # NOTE: inverted_index[bp] is a Python set and iteration is non-deterministic, which may have consequences
        # for word in list(inverted_index[bp_to_merge]):  👈 OLD CODE ｜ NEW CODE 👇 didn't add noticeable latency
        for word in sorted(list(inverted_index[bp_to_merge])):
            freq = total_word_counts[word]
            # TODO: Use double linked list instead of list for split, much better state management
            split = word_splits[word]
            new_split = []

            i = 0
            while i < len(split):
                # Check if we can merge at position i (need at least 2 elements remaining)
                if i < len(split) - 1 and (split[i], split[i+1]) == bp_to_merge:
                    if i > 0:
                        bp_left = (split[i-1], split[i])
                        bp_counts_decre[bp_left] += freq
                        # TODO: Trying to optimize, but these 2 blocks would fail tests/test_train_bpe.py
                        # if bp_left in inverted_index and word in inverted_index[bp_left]:
                        #     inverted_index[bp_left].remove(word)
                        #     if not inverted_index[bp_left]:
                        #         del inverted_index[bp_left]
                    if i < len(split) - 2:
                        bp_right = (split[i+1], split[i+2])
                        bp_counts_decre[bp_right] += freq
                        # if bp_right in inverted_index and word in inverted_index[bp_right]:
                        #     inverted_index[bp_right].remove(word)
                        #     if not inverted_index[bp_right]:
                        #         del inverted_index[bp_right]

                    new_split.append(bytes_merged)
                    i += 2
                else:
                    new_split.append(split[i])
                    i += 1

            word_splits[word] = new_split

            for bp in zip(new_split[:-1], new_split[1:]):
                if bp[0] == bytes_merged or bp[1] == bytes_merged:
                    if bp in inverted_index:
                        inverted_index[bp].add(word)
                    else:
                        inverted_index[bp] = set([word])

                    if bp in bp_counts:
                        bp_counts[bp] += freq
                    else:
                        bp_counts[bp] = freq

        # Update states
        del inverted_index[bp_to_merge]
        del bp_counts[bp_to_merge]

        bp_counts -= bp_counts_decre


    return vocab, merges



if __name__ == "__main__":
    vocab, merges = train_bpe(
        "./data/owt_train.txt",
        vocab_size=32000,
        special_tokens=["<|endoftext|>"]
    )

    merges_file_path = "merges_owt_train.txt"
    with open(merges_file_path, "w", encoding="utf-8") as f:
        for p1, p2 in merges:
            # Decode bytes into strings for writing to a text file
            # The 'replace' handler handles bytes that aren't valid utf-8
            p1_str = p1.decode('utf-8', 'replace')
            p2_str = p2.decode('utf-8', 'replace')
            f.write(f"{p1_str} {p2_str}\n")
    
    print(f"Merges saved to {merges_file_path}")

    vocab_file_path = "vocab_owt_train.json"
    json_vocab = {
        # JSON keys must be strings, so convert the int token ID
        str(idx): token.decode('utf-8', 'replace')
        for idx, token in vocab.items()
    }

    with open(vocab_file_path, "w", encoding="utf-8") as f:
        # Use json.dump() to write the dictionary to the file
        # indent=2 makes the file nicely formatted and readable
        json.dump(json_vocab, f, ensure_ascii=False, indent=2)

    print(f"Vocabulary saved to {vocab_file_path}")