from nltk.tokenize import  WhitespaceTokenizer, TreebankWordTokenizer, WordPunctTokenizer
from nltk.stem import PorterStemmer,WordNetLemmatizer
import nltk
import spacy

nltk.download('punkt')
nltk.download('wordnet')

class Process():
    def __init__(self, ):
        pass
    def whiles_space_tokenizer(self, text) -> list:
        """
        This method performs tokenization (splitting a sentence into words) based on whitespace.
        It divides the input sentence into a list of words separated by spaces.

        Args:
            text (str): The string of text to be tokenized.

        Returns:
            list: A list of tokens (words) obtained by splitting the input string.
                Each element in the list is a word from the original text, separated by whitespace.
        """
        return WhitespaceTokenizer().tokenize(text)
    def word_punct_tokenizer(self, text) -> list:
        """
        This method tokenizes text by splitting words based on whitespace and punctuation marks.
        It creates tokens for each word and punctuation symbol found in the input string.

        Args:
            text (str): The input string to be tokenized.

        Returns:
            list: A list of tokens obtained by splitting the input string based on spaces and punctuation.
                Each element in the list is a word or punctuation mark from the original text.
        """
        return WordPunctTokenizer().tokenize(text)
    def tree_bank_word_tokennizer(self, text):
        """
        Tokenizes the input text using the Treebank Word Tokenizer, which splits
        text based on linguistic rules that separate contractions and punctuation.

        Args:
            text (str): The input string to be tokenized.

        Returns:
            list: A list of tokens, where each token is a word, contraction, or punctuation
                  based on Treebank tokenization rules.
        """
        return  TreebankWordTokenizer().tokenize(text)
        
    def stem_normalize(self, words)-> list:
        """Stemming reduces words to their root form, which can help in various
        natural language processing tasks by normalizing the text.

        Args:
            list_text (list): A list of words (strings) to be stemmed.

        Returns:
            list: A list of stemmed words, where each word has been reduced to its root form
                  using the Porter Stemmer.
        """
        return [PorterStemmer().stem(word) for word in words]
    def lemmazation_normalize(self,words)->list:
        
            """Lemmatizes a list of words.

            Args:
                words (list): A list of words (strings) to be lemmatized.

            Returns:
                list: A list of lemmatized words.
            """
            return  [WordNetLemmatizer().lemmatize(word) for word in words]
        
    def spacy_normalize(self, words: list) -> list:
        """spacy a list of words.

        Args:
            words (list): A list of words (strings) to be lemmatized.

        Returns:
            list: A list of lemmatized words.
        """
        # Process the words through spaCy
        doc = spacy.load("en_core_web_sm")(" ".join(words))
        return [token.lemma_ for token in doc]

