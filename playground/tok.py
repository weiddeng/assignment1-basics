from tokenizer import Tokenizer
from sampled_documents import tinystories_docs, owt_docs

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

for doc in tinystories_docs:
    # print(tok_tinystories.encode(doc))
    print(len(doc.encode('utf-8')))
    # computing the compression ratio
    print(len(doc.encode('utf-8')) * 1. / len(tok_tinystories.encode(doc)))
    print('\n')

print ("~"*20)

for doc in owt_docs:
    # print(tok_owt.encode(doc))
    print(len(doc.encode('utf-8')))
    print(len(doc.encode('utf-8')) * 1. / len(tok_owt.encode(doc)))
    print('\n')
