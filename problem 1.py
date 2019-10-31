import requests
from bs4 import BeautifulSoup
from copy import deepcopy

import nltk
from nltk.util import trigrams
import operator

URL = "https://en.wikipedia.org/wiki/Google"
r = requests.get(URL)

soup = BeautifulSoup(r.content, 'html.parser')
text = soup.find_all('p')  # get all p tags (paragraphs with wiki text in them)
textAll = ""
for i in range(len(text)):  # add the text only from all the p tags in to one string
    textAll += text[i].get_text()

refs = True
while refs: # remove all reference numbers (ex: [71][72]) for easier manipulation and readability
    start = textAll.find('[')
    if start == -1:
        refs = False
        continue
    else:
        finish = textAll.find(']')
        sub = textAll[start:finish + 1]
        textAll = textAll.replace(sub, '')

# write to input file
file = open("input.txt", 'w+')
file.write(textAll)
file.close()
file = open("input.txt", 'r')


# Tokenization
pconts = file.read()
conts = deepcopy(pconts)
wtokens = nltk.word_tokenize(conts)

token_ans = open("0_tokens.txt", 'w+')
token_ans.write(str(wtokens))
token_ans.close()



# POS
pos_ans = open("1_pos.txt", 'w+')
pos_ans.write(str(nltk.pos_tag(wtokens)))
pos_ans.close()



# Stemming
pStemmer = nltk.PorterStemmer()
stems = ""

for i in wtokens:
    stems += pStemmer.stem(i) + " "

stems_ans = open("2_stems.txt", 'w+')
stems_ans.write(stems)
stems_ans.close()




# Lemmatization

# https://www.machinelearningplus.com/nlp/lemmatization-examples-python/
# Couldn't figure out how to pass the POS tag to lemmatize() myself, I take no credit for this function
def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": nltk.corpus.wordnet.ADJ,
                "N": nltk.corpus.wordnet.NOUN,
                "V": nltk.corpus.wordnet.VERB,
                "R": nltk.corpus.wordnet.ADV}

    return tag_dict.get(tag, nltk.corpus.wordnet.NOUN)


lemat = nltk.WordNetLemmatizer()

lems = ""

for i in wtokens:
    lems += lemat.lemmatize(i, get_wordnet_pos(i)) + " "

lem_ans = open("3_lems.txt", 'w+')
lem_ans.write(lems)
lem_ans.close()



# Trigrams
bad_chars = ['(', ')', ',', '?', '!', '=', ':', ';', '.', '\'', '\"']   # remove common punctuation so they aren't
                                                                        # included in trigrams as words
for i in bad_chars:
    conts = conts.replace(i, ' ')

tris = list(trigrams(conts.split()))
tri_ans = open("4_tris.txt", 'w+')
tri_ans.write(str(tris))
tri_ans.close()



# NER
conts = deepcopy(pconts)

ner_ans = open('5_ner.txt', 'w+')
ner_ans.write(str(nltk.ne_chunk(nltk.pos_tag(nltk.wordpunct_tokenize(conts)))))
ner_ans.close()

file.close()
