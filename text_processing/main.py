from processing import Process
if __name__ == "__main__":
    text = "London was the old capital of United Kingdom which has the most population in the world and I'll be go there someday with Hung's girlfriend."
    process = Process()

    while_space = process.whiles_space_tokenizer(text)
    word_punct = process.word_punct_tokenizer(text)
    tree_bank = process.tree_bank_word_tokennizer(text)

    print('================================================================')
    print(f"Whitespace Tokenizer: {while_space}")
    print(f"Word Punct Tokenizer: {word_punct}")
    print(f"Treebank Word Tokenizer: {tree_bank}")
    print('================================================================')

    stem_normal = process.stem_normalize(while_space)
    lemmatization_normal = process.lemmazation_normalize(while_space)
    spacy_normalize = process.spacy_normalize(while_space)

    print(f"Stemmed Words: {stem_normal}")
    print(f"Lemmatized Words (WordNet): {lemmatization_normal}")
    print(f"Lemmatized Words (spaCy): {spacy_normalize}")
