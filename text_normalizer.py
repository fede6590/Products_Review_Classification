import re
from tokenize import TokenInfo
import nltk
import spacy
import unicodedata
import unidecode

from bs4 import BeautifulSoup
from contractions import CONTRACTION_MAP
from nltk.tokenize.toktok import ToktokTokenizer

nltk.download('stopwords')

tokenizer = ToktokTokenizer()
stopword_list = nltk.corpus.stopwords.words('english')
nlp = spacy.load('en_core_web_sm')


def remove_html_tags(text):
    # text = re.sub('<[^<]+?>', '', text)
    text = BeautifulSoup(text, 'html.parser')
    text = text.get_text()
    return text


def stem_text(text):
    ps = nltk.porter.PorterStemmer()
    tok_text = tokenizer.tokenize(text)
    for tok in tok_text:
        tok_text[tok_text.index(tok)] = ps.stem(tok)
    text = ' '.join(tok_text)
    return text


def lemmatize_text(text):
    text = nlp(text)
    text = ' '.join([tok.lemma_ if tok.lemma_ != '-PRON-' else tok.text for tok in text])
    return text


def expand_contractions(text, contraction_mapping=CONTRACTION_MAP):
    for key, value in contraction_mapping.items():
        text = text.replace(key, value)
    return text


def remove_accented_chars(text):
    text = unidecode.unidecode(text)
    return text


def remove_special_chars(text, remove_digits=False):
    to_remove = r'[^a-zA-Z0-9\s]' if not remove_digits else r'[^a-zA-Z\s]'
    text = re.sub(to_remove, '', text)
    return text


def remove_stopwords(text, is_lower_case=False, stopwords=stopword_list):
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopwords]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stopwords]
    text = ' '.join(filtered_tokens) 
    return text


def remove_extra_new_lines(text):
    text = re.sub(r'\s', ' ', text)
    return text


def remove_extra_whitespace(text):
    text = re.sub(' +', ' ', text)
    return text
    

def normalize_corpus(
    corpus,
    html_stripping=True,
    contraction_expansion=True,
    accented_char_removal=True,
    text_lower_case=True,
    text_stemming=False,
    text_lemmatization=False,
    special_char_removal=True,
    remove_digits=True,
    stopword_removal=True,
    stopwords=stopword_list
):
    
    normalized_corpus = []

    # Normalize each doc in the corpus
    for doc in corpus:
        # Remove HTML
        if html_stripping:
            doc = remove_html_tags(doc)
            
        # Remove extra newlines
        doc = remove_extra_new_lines(doc)
        
        # Remove accented chars
        if accented_char_removal:
            doc = remove_accented_chars(doc)
            
        # Expand contractions    
        if contraction_expansion:
            doc = expand_contractions(doc)
            
        # Lemmatize text
        if text_lemmatization:
            doc = lemmatize_text(doc)
            
        # Stemming text
        if text_stemming and not text_lemmatization:
            doc = stem_text(doc)
            
        # Remove special chars and\or digits    
        if special_char_removal:
            doc = remove_special_chars(
                doc,
                remove_digits=remove_digits
            )  

        # Remove extra whitespace
        doc = remove_extra_whitespace(doc)

         # Lowercase the text    
        if text_lower_case:
            doc = doc.lower()

        # Remove stopwords
        if stopword_removal:
            doc = remove_stopwords(
                doc,
                is_lower_case=text_lower_case,
                stopwords=stopwords
            )

        # Remove extra whitespace
        doc = remove_extra_whitespace(doc)
        doc = doc.strip()
            
        normalized_corpus.append(doc)
        
    return normalized_corpus
