# GPT-2 style pre-tokenization pattern
# Matches:
# - Contractions: 's, 'd, 'm, 't, 'll, 've, 're
# - Letters (with optional leading space)
# - Numbers (with optional leading space)
# - Non-alphanumeric characters (with optional leading space)
# - Whitespace (not followed by non-whitespace)
PAT_IN_CHUNK = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""