from nltk.corpus import stopwords
from nltk import word_tokenize, sent_tokenize
from nltk.stem import SnowballStemmer
import string
import re

#tokenizer http://sentiment.christopherpotts.net/code-data/happyfuntokenizing.py
emoticon_string = r"""
  (?:
    [<>]?
    [:;=8]                     # eyes
    [\-o\*\']?                 # optional nose
    [\)\]\(\[dDpP/\:\}\{@\|\\] # mouth      
    |
    [\)\]\(\[dDpP/\:\}\{@\|\\] # mouth
    [\-o\*\']?                 # optional nose
    [:;=8]                     # eyes
    [<>]?
  )"""

# The components of the tokenizer:
regex_strings = (
  # Phone numbers:
  r"""
  (?:
    (?:            # (international)
      \+?[01]
      [\-\s.]*
    )?            
    (?:            # (area code)
      [\(]?
      \d{3}
      [\-\s.\)]*
    )?    
    \d{3}          # exchange
    [\-\s.]*   
    \d{4}          # base
  )"""
  ,
  # Emoticons:
  emoticon_string
  ,    
  # HTML tags:
   r"""<[^>]+>"""
  ,
  # Twitter username:
  r"""(?:@[\w_]+)"""
  ,
  # Twitter hashtags:
  r"""(?:\#+[\w_]+[\w\'_\-]*[\w_]+)"""
  ,
  # Remaining word types:
  r"""
  (?:[a-z][a-z'\-_]+[a-z])       # Words with apostrophes or dashes.
  |
  (?:[+\-]?\d+[,/.:-]\d+[+\-]?)  # Numbers, including fractions, decimals.
  |
  (?:[\w_]+)                     # Words without apostrophes or dashes.
  |
  (?:\.(?:\s*\.){1,})            # Ellipsis dots. 
  |
  (?:\S)                         # Everything else that isn't whitespace.
  """
  )

url_regex = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
word_re = re.compile(r"""(%s)""" % "|".join(regex_strings), re.VERBOSE | re.I | re.UNICODE)

def process_text(text, stem=True, remove_links=True, punctuation_exception='#@'):
  puntuacion = string.punctuation + '…–«»“”¡¿´¨‘'
  #Remove exceptions
  exceptions_table = str.maketrans('','',punctuation_exception)
  filtered_punctiation = puntuacion.translate(exceptions_table)
  # Remove links
  if remove_links:
      text = re.sub(url_regex, '', text)
  #Remove punctuation
  remove_punct_map = dict.fromkeys(map(ord, filtered_punctiation))
  text = text.translate(remove_punct_map)
  remove_digits_map = dict.fromkeys(map(ord, string.digits))
  text = text.translate(remove_digits_map)
  #Remove extra spaces
  text = ' '.join(text.split())
  # obtain tokens 
  tokens = word_re.findall(text)

  if stem:
      stemmer = SnowballStemmer("spanish")
      tokens = [stemmer.stem(t) for t in tokens]
      #tokens = text.apply(stemmer.stem)

  return tokens

def stop_words(language='spanish'):
  return stopwords.words(language)