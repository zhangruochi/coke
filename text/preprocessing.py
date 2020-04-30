import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import spacy


# nlp = spacy.load("en_core_web_sm")
# lemmatizer = Lemmatizer(nlp.vocab.lookups)


# nltk.download('stopwords')
# nltk.download('punkt')
# stop_words = set(stopwords.words('english'))


def string_clean(text_string,lowerd = False, punctuations_removed = False, numbers_removed = False, stopwords = None, lemmatizer = None):
    """ clean the string
    
    Parameters
    ----------
    text_string (str) : input string which should be cleaned
    lowerd (Boolean): input string whether should be lowered
    punctuations_removed (Boolean): punctuations whether should be removed
    numbers_removed (Boolean): numbers whether should be removed
    stopwords: stopwords which should be removed
    lemmatizer: (Obj): a obj to lemmatize word
        
    Returns
    -------
    new string
    """
    if text_string and isinstance(text_string,str): 
        
        ## to lowercase
        if lowerd:
            text_string = text_string.lower()

        ## remove numbers
        if numbers_removed:
            text_string = re.sub(r'\d+', '', text_string)

        ## Remove punctuation
        if punctuations_removed:
            text_string = text_string.translate(str.maketrans({punc: " " for punc in string.punctuation}))
        
        
        ## Stop words removal
        if stopwords:
            tokens = word_tokenize(text_string)
            if lemmatizer:
                text_string = " ".join([lemmatizer.lookup(word) for word in tokens if not word in stopwords])
            else:
                text_string = " ".join([word for word in tokens if not word in stop_words])

        ## Remove white space
        text_string = re.sub(r'\s+', ' ', text_string).strip()
    
    return text_string

    
