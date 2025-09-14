import os
import cProfile
import numpy as np
from tokenizer import Tokenizer


def format_bytes(size_bytes: int) -> str:
    """Converts a size in bytes to a human-readable format (KB, MB, GB)."""
    if size_bytes < 1024:
        return f"{size_bytes} bytes"
    elif size_bytes < 1024**2:
        return f"{size_bytes/1024:.2f} KB"
    elif size_bytes < 1024**3:
        return f"{size_bytes/1024**2:.2f} MB"
    else:
        return f"{size_bytes/1024**3:.2f} GB"

filepaths = [
    "/Users/weideng/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt",
    "/Users/weideng/assignment1-basics/data/TinyStoriesV2-GPT4-valid.txt",
    "/Users/weideng/assignment1-basics/data/owt_train.txt",
    "/Users/weideng/assignment1-basics/data/owt_valid.txt"
]

print("--- File Sizes ---")
for path in filepaths:
    try:
        size_in_bytes = os.path.getsize(path)
        readable_size = format_bytes(size_in_bytes)
        filename = os.path.basename(path)
        print(f"{filename:<40} {readable_size}")
    except FileNotFoundError:
        print(f"ERROR: File not found at {path}")
print("-" * 20)


def tokenize_file_stream(tokenizer: Tokenizer, filepath: str, savepath: str):
    """
    Tokenizes a large file by streaming and writing to a raw binary file,
    avoiding high memory usage.
    """
    print(f"\nStarting tokenization of '{os.path.basename(filepath)}'...")
    
    with open(savepath, 'wb') as f:
        # Create an iterator that yields tokens from the source text file
        token_iterator = tokenizer.encode_iterable(open(filepath, 'r', encoding='utf-8'))
        
        buffer = []
        # Process and write tokens in chunks for efficiency
        chunk_size = 1_000_000
        
        for i, token in enumerate(token_iterator):
            buffer.append(token)
            
            # This is the core streaming logic. Instead of holding all tokens in RAM,
            # we write them to disk periodically in manageable chunks.
            if len(buffer) >= chunk_size:
                # Convert the buffer to a numpy array and write its raw bytes
                np.array(buffer, dtype=np.uint16).tofile(f)
                buffer.clear() # Free up memory by clearing the buffer
                print(f"Processed and wrote {i+1:,} tokens...", end='\r')

        # After the loop, write any remaining tokens left in the buffer
        if buffer:
            np.array(buffer, dtype=np.uint16).tofile(f)
            print(f"Processed and wrote a total of {i+1:,} tokens.")

    print(f"\nSuccessfully saved tokens to '{savepath}'")
    print(f"Load with: np.memmap('{savepath}', dtype=np.uint16, mode='r')")


tok_tinystories = Tokenizer.from_files(
    vocab_filepath="/Users/weideng/assignment1-basics/vocab.json",
    merges_filepath="/Users/weideng/assignment1-basics/merges.txt",
    special_tokens=["<|endoftext|>"]
)

tok_owt = Tokenizer.from_files(
    vocab_filepath="/Users/weideng/assignment1-basics/vocab_owt_train.json",
    merges_filepath="/Users/weideng/assignment1-basics/merges_owt_train.txt",
    special_tokens=["<|endoftext|>"]
)


# We now call the streaming function and save to a .bin file to indicate it's raw binary data.
# cProfile.run('tokenize_file_stream(tok_tinystories, "/Users/weideng/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt", "/Users/weideng/assignment1-basics/data-output/TinyStoriesV2-GPT4-train-tokens.bin")', 'TinyStoriesV2-GPT4-train-tokens.prof')

# cProfile.run('tokenize_file_stream(tok_tinystories, "/Users/weideng/assignment1-basics/data/TinyStoriesV2-GPT4-valid.txt", "/Users/weideng/assignment1-basics/data-output/TinyStoriesV2-GPT4-valid-tokens.bin")', 'TinyStoriesV2-GPT4-valid-tokens.prof')

# cProfile.run('tokenize_file_stream(tok_owt, "/Users/weideng/assignment1-basics/data/owt_train.txt", "/Users/weideng/assignment1-basics/data-output/owt_train.bin")', 'owt_train.prof')

cProfile.run('tokenize_file_stream(tok_owt, "/Users/weideng/assignment1-basics/data/owt_valid.txt", "/Users/weideng/assignment1-basics/data-output/owt_valid.bin")', 'owt_valid.prof')


# --- How to load and use the created .bin file ---
#
# import numpy as np
#
# # Use np.memmap to open the file as a NumPy array without loading it all into RAM
# train_tokens = np.memmap(
#     "/Users/weideng/assignment1-basics/data-output/TinyStoriesV2-GPT4-train-tokens.bin",
#     dtype=np.uint16,
#     mode='r' # 'r' for read-only
# )
#
# print(f"\nSuccessfully loaded memory-mapped array.")
# print(f"Shape: {train_tokens.shape}")
# print(f"First 10 tokens: {train_tokens[:10]}")
#