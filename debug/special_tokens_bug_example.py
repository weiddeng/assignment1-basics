"""
Example illustrating why special tokens need to be sorted by length.

This bug occurs when you have overlapping special tokens like:
- "<|endoftext|>"
- "<|endoftext|><|endoftext|>"

The regex engine needs to match the longest possible token first.
"""

import re

print("=" * 70)
print("THE OVERLAPPING SPECIAL TOKENS BUG")
print("=" * 70)

# Example special tokens
special_tokens = ["<|endoftext|>", "<|endoftext|><|endoftext|>"]
text = "Hello <|endoftext|><|endoftext|> world <|endoftext|>"

print(f"\nSpecial tokens: {special_tokens}")
print(f"Input text: '{text}'")
print("\nDesired behavior:")
print("  - '<|endoftext|><|endoftext|>' should be treated as ONE token")
print("  - Single '<|endoftext|>' should be treated as ONE token")

print("\n" + "=" * 70)
print("BUGGY BEHAVIOR (without sorting)")
print("=" * 70)

# Original approach - tokens in original order
pat_buggy = "|".join(re.escape(t) for t in special_tokens)
print(f"\nRegex pattern (original order): {pat_buggy}")
print("Which becomes: '<\\|endoftext\\|>|<\\|endoftext\\|><\\|endoftext\\|>'")

# Split with capturing group to keep separators
parts_buggy = re.split(f"({pat_buggy})", text)
print(f"\nSplit result: {parts_buggy}")

# Filter out empty strings
parts_buggy = [p for p in parts_buggy if p]
print(f"After filtering: {parts_buggy}")

print("\nPROBLEM: The regex matches '<|endoftext|>' FIRST because it appears")
print("first in the pattern. So '<|endoftext|><|endoftext|>' gets matched as")
print("TWO separate '<|endoftext|>' tokens instead of one double token!")

# Count how many times each special token appears
single_count = parts_buggy.count("<|endoftext|>")
double_count = parts_buggy.count("<|endoftext|><|endoftext|>")
print(f"\nToken counts:")
print(f"  '<|endoftext|>': {single_count} (WRONG - should be 1)")
print(f"  '<|endoftext|><|endoftext|>': {double_count} (WRONG - should be 1)")

print("\n" + "=" * 70)
print("FIXED BEHAVIOR (with sorting by length)")
print("=" * 70)

# Fixed approach - sort by length (longest first)
sorted_tokens = sorted(special_tokens, key=len, reverse=True)
print(f"\nSorted tokens (longest first): {sorted_tokens}")

pat_fixed = "|".join(re.escape(t) for t in sorted_tokens)
print(f"\nRegex pattern (sorted): {pat_fixed}")
print("Which becomes: '<\\|endoftext\\|><\\|endoftext\\|>|<\\|endoftext\\|>'")
print("                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
print("                 Longer pattern comes FIRST")

# Split with the fixed pattern
parts_fixed = re.split(f"({pat_fixed})", text)
parts_fixed = [p for p in parts_fixed if p]
print(f"\nSplit result: {parts_fixed}")

# Count tokens
single_count = parts_fixed.count("<|endoftext|>")
double_count = parts_fixed.count("<|endoftext|><|endoftext|>")
print(f"\nToken counts:")
print(f"  '<|endoftext|>': {single_count} (CORRECT!)")
print(f"  '<|endoftext|><|endoftext|>': {double_count} (CORRECT!)")

print("\n" + "=" * 70)
print("WHY THIS MATTERS")
print("=" * 70)

print("""
When regex alternatives are separated by '|', the regex engine tries
them from left to right and stops at the FIRST match.

Without sorting:
  Pattern: '<|endoftext|>|<|endoftext|><|endoftext|>'
  Text: '<|endoftext|><|endoftext|>'
  Result: Matches '<|endoftext|>' immediately, leaving another '<|endoftext|>'

With sorting (longest first):
  Pattern: '<|endoftext|><|endoftext|>|<|endoftext|>'
  Text: '<|endoftext|><|endoftext|>'
  Result: Matches the full '<|endoftext|><|endoftext|>' as intended

This is crucial for tokenizers where different special tokens might
share prefixes or be concatenations of each other.
""")

print("=" * 70)
print("REAL-WORLD EXAMPLE")
print("=" * 70)

print("""
Imagine a tokenizer with these special tokens:
- '<|im_start|>'
- '<|im_end|>'
- '<|im_start|>system'
- '<|im_start|>user'
- '<|im_start|>assistant'

Without sorting by length, '<|im_start|>system' would always be
tokenized as ['<|im_start|>', 'system'] instead of a single token.

With sorting, the longer tokens are checked first, ensuring correct
tokenization.
""")