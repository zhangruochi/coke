import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('english'))


def string_clean(text_string):
    if text_string and isinstance(text_string,str): 
        ## to lowercase
        text_string = text_string.lower()
#         ## remove numbers
#         result = re.sub(r'\d+', '', input_str)
        ## Remove punctuation
        text_string = text_string.translate(str.maketrans({punc: " " for punc in string.punctuation}))
        ## Remove white space
        text_string = re.sub(r'\s+', ' ', text_string).strip()
        ## Stop words removal
        tokens = word_tokenize(text_string)
        text_string = " ".join([word for word in tokens if not word in stop_words])
        return text_string
    else:
        return text_string
