#!/usr/bin/env python
# coding: utf-8

# # SOAL UAS

# 1.	Buatlah tutorial  analisa k-mean clustering clustering data abstrak ekonomi-manajemen pada  pta.trunojoyo.ac.id
# 
# 2.	Lakukan analisa topik modelling menggunakan metode Latent Semantic Analysis pada data abstrak  ekonomi-manajemen di pta.trunojoyo.ac.id 
# 
# Ket: 
# 1.	Dikerjakan jupyter notebook  dan didokumentasikan menggunakan Jupyterbook
# 
# 2.	Hasil di hosting di github ( link url repository github anda  di upload di schoology )
# 
# 3.	tutorial no 1:
# 
# a.	Penjelasan tentang crawling dan proses melakukan crawling pada pta.trunojoyo.ac.id data abtrak ekonomi-manajemen
# 
# b.	Pre Proses data dan  Tf-idf dari data abtrak ekonomi-manajemen pta.trunojoyo.ac.id 
# 
# c.	Analisa  k-mean clustering pada abtrak ekonomi-manajemen pta.trunojoyo.ac.id
# 

# ## Crawling data

# Pada tugas sebelumnya, saya melakukan crawling data dari website https://pta.trunojoyo.ac.id/.
# 
# Cara crawling:
# 
# 1. Menambahkan link website
# 
# 2. Membaca formula dari kontennya
# 
# 3. Jalankan code di bawah menggunakan terminal pada Visual Studio Code dengan perintah "scrapy runspider namafile.py"
# 
# 4. Kemudian saya mengekstrak data hasil crawling menjadi .csv. dengan perintah "scrapy runspider namafile.py -o namafile.csv"

# In[1]:


import scrapy


class CrawlingPTA(scrapy.Spider):
    name = 'crawlingpta'
    start_urls = ['https://pta.trunojoyo.ac.id/c_search/byprod/7',
                'https://pta.trunojoyo.ac.id/c_search/byprod/7/2',
                'https://pta.trunojoyo.ac.id/c_search/byprod/7/3',
                'https://pta.trunojoyo.ac.id/c_search/byprod/7/4',
                'https://pta.trunojoyo.ac.id/c_search/byprod/7/5',
                'https://pta.trunojoyo.ac.id/c_search/byprod/7/6',
                'https://pta.trunojoyo.ac.id/c_search/byprod/7/7',
                'https://pta.trunojoyo.ac.id/c_search/byprod/7/8',
                'https://pta.trunojoyo.ac.id/c_search/byprod/7/9',
                'https://pta.trunojoyo.ac.id/c_search/byprod/7/10',
                'https://pta.trunojoyo.ac.id/c_search/byprod/7/11']

    def parse(self,response):
        for item in response.css('a.gray.button'):
            try:
                page_link = item.css('a.gray.button').attrib['href']
                if page_link is not None:
                    yield response.follow(page_link, self.parse_page)
            except:
                yield 'not found'
    
    def parse_page(self, response):
        for item in response.css('ul.items.list_style'):
            try:
                yield {
                    'abstract' : item.xpath('//*[@id="content_journal"]/ul/li/div[4]/div[2]/p/text()').get(),
                }
            except:
                yield {
                    'abstract' : 'not found'
                }


# ## Preprocessing data

# Setelah melakukan crawling data lakukan proses preprocessing

# ### Install packages / module

# In[2]:


import pandas as pd
import numpy as np
import string
import re #regrex libray
import nltk
import swifter
import Sastrawi
import nltk
nltk.download('punkt')
nltk.download('stopwords')

from nltk.tokenize import word_tokenize
from nltk.tokenize.punkt import PunktSentenceTokenizer
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.probability import FreqDist
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer


# ### Load dataset

# In[3]:


import pandas as pd

MY_DATA = pd.read_csv("dataset.csv")

MY_DATA.head(54)


# ### Case folding

# In[4]:


MY_DATA['abstrak'] = MY_DATA['abstrak'].str.lower()

print('Case Folding Result:\n')
print(MY_DATA['abstrak'].head(54))
print('\n\n\n')


# ### Tokenizing

# In[5]:


import string 
import re #regex library


# import word_tokenize & FreqDist from NLTK
from nltk.tokenize import word_tokenize 
from nltk.probability import FreqDist

# ------ Tokenizing ---------

def remove_tweet_special(text):
    # remove tab, new line, ans back slice
    text = text.replace('\\t'," ").replace('\\n'," ").replace('\\u'," ").replace('\\',"")
    # remove non ASCII (emoticon, chinese word, .etc)
    text = text.encode('ascii', 'replace').decode('ascii')
    # remove mention, link, hashtag
    text = ' '.join(re.sub("([@#][A-Za-z0-9]+)|(\w+:\/\/\S+)"," ", text).split())
    # remove incomplete URL
    return text.replace("http://", " ").replace("https://", " ")
                
MY_DATA['abstrak'] = MY_DATA['abstrak'].apply(remove_tweet_special)

#remove number
def remove_number(text):
    return  re.sub(r"\d+", "", text)

MY_DATA['abstrak'] = MY_DATA['abstrak'].apply(remove_number)

#remove punctuation
def remove_punctuation(text):
    return text.translate(str.maketrans("","",string.punctuation))

MY_DATA['abstrak'] = MY_DATA['abstrak'].apply(remove_punctuation)

#remove whitespace leading & trailing
def remove_whitespace_LT(text):
    return text.strip()

MY_DATA['abstrak'] = MY_DATA['abstrak'].apply(remove_whitespace_LT)

#remove multiple whitespace into single whitespace
def remove_whitespace_multiple(text):
    return re.sub('\s+',' ',text)

MY_DATA['abstrak'] = MY_DATA['abstrak'].apply(remove_whitespace_multiple)

# remove single char
def remove_singl_char(text):
    return re.sub(r"\b[a-zA-Z]\b", "", text)

MY_DATA['abstrak'] = MY_DATA['abstrak'].apply(remove_singl_char)

# NLTK word rokenize 
def word_tokenize_wrapper(text):
    return word_tokenize(text)

MY_DATA['abstrak_tokens'] = MY_DATA['abstrak'].apply(word_tokenize_wrapper)

print('Tokenizing Result : \n') 
print(MY_DATA['abstrak_tokens'].head(54))
print('\n\n\n')


# ### Menghitung Frekuensi Distribusi Token

# In[6]:


# NLTK calc frequency distribution
def freqDist_wrapper(text):
    return FreqDist(text)

MY_DATA['abstrak_tokens_fdist'] = MY_DATA['abstrak_tokens'].apply(freqDist_wrapper)

print('Frequency Tokens : \n') 
print(MY_DATA['abstrak_tokens_fdist'].head(54).apply(lambda x : x.most_common()))


# ## Filtering (Stopword Removal)

# In[7]:


from nltk.corpus import stopwords

# ----------------------- get stopword from NLTK stopword -------------------------------
# get stopword indonesia
list_stopwords = stopwords.words('indonesian')


# ---------------------------- manualy add stopword  ------------------------------------
# append additional stopword
list_stopwords.extend(["yg", "dg", "rt", "dgn", "ny", "d", 'klo', 
                       'kalo', 'amp', 'biar', 'bikin', 'bilang', 
                       'gak', 'ga', 'krn', 'nya', 'nih', 'sih', 
                       'si', 'tau', 'tdk', 'tuh', 'utk', 'ya', 
                       'jd', 'jgn', 'sdh', 'aja', 'n', 't', 
                       'nyg', 'hehe', 'pen', 'u', 'nan', 'loh', 'rt',
                       '&amp', 'yah'])

# ----------------------- add stopword from txt file ------------------------------------
# read txt stopword using pandas
txt_stopword = pd.read_csv("dataset.csv", names= ["stopwords"], header = None)

# convert stopword string to list & append additional stopword
list_stopwords.extend(txt_stopword["stopwords"][0].split(' '))

# ---------------------------------------------------------------------------------------

# convert list to dictionary
list_stopwords = set(list_stopwords)


#remove stopword pada list token
def stopwords_removal(words):
    return [word for word in words if word not in list_stopwords]

MY_DATA['abstrak_tokens_WSW'] = MY_DATA['abstrak_tokens'].apply(stopwords_removal) 


print(MY_DATA['abstrak_tokens_WSW'].head(54))


# ### Normalization

# In[8]:


normalizad_word = pd.read_csv("dataset.csv")

normalizad_word_dict = {}

for index, row in normalizad_word.iterrows():
    if row[0] not in normalizad_word_dict:
        normalizad_word_dict[row[0]] = row[1] 

def normalized_term(document):
    return [normalizad_word_dict[term] if term in normalizad_word_dict else term for term in document]

MY_DATA['abstrak_normalized'] = MY_DATA['abstrak_tokens_WSW'].apply(normalized_term)

MY_DATA['abstrak_normalized'].head(54)


# ### Stemmer

# In[9]:


# import Sastrawi package
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import swifter
import pandas as pd 
import numpy as np


# create stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# stemmed
def stemmed_wrapper(term):
    return stemmer.stem(term)

term_dict = {}

for document in MY_DATA['abstrak_normalized']:
    for term in document:
        if term not in term_dict:
            term_dict[term] = ' '
            
print(len(term_dict))
print("------------------------")

for term in term_dict:
    term_dict[term] = stemmed_wrapper(term)
    print(term,":" ,term_dict[term])
    
print(term_dict)
print("------------------------")


# apply stemmed term to dataframe
def get_stemmed_term(document):
    return [term_dict[term] for term in document]

MY_DATA['abstrak_tokens_stemmed'] = MY_DATA['abstrak_normalized'].swifter.apply(get_stemmed_term)
print(MY_DATA['abstrak_tokens_stemmed'])


# ## Save prepocessing data

# In[9]:


MY_DATA["abstrak_tokens_stemmed"].to_csv("Text_Preprocessing_PTA.csv")


# ## Prepare data

# In[10]:


import pandas as pd 
import numpy as np

MY_DATA = pd.read_csv("Text_Preprocessing_PTA.csv", usecols=["abstrak_tokens_stemmed"])
MY_DATA.columns = ["abstrak"]
MY_DATA.head(54)


# ## Term frekuensi

# In[11]:


from sklearn.feature_extraction.text import CountVectorizer

a=len(document)
document = MY_DATA['abstrak']

# Create a Vectorizer Object
vectorizer = CountVectorizer()

vectorizer.fit(document)

# Printing the identified Unique words along with their indices
print("Vocabulary: ", vectorizer.vocabulary_)

# Encode the Document
vector = vectorizer.transform(document)

# Summarizing the Encoded Texts
print("Encoded Document is:")
print(vector.toarray())


# In[12]:


a = vectorizer.get_feature_names()


# ## TF-ID

# In[13]:


tfidf = TfidfTransformer(use_idf=True, norm='l2', smooth_idf=True)
tf = tfidf.fit_transform(vectorizer.fit_transform(document)).toarray()


# In[14]:


dfb = pd.DataFrame(data=tf,index=list(range(1, len(tf[:,1])+1, )),columns=[a])
dfb


# In[15]:


dfb.to_excel('./TF-IDF.xlsx')


# ## Finding Optimal Clusters

# Pengelompokan adalah operasi tanpa pengawasan, dan KMeans mengharuskan kami menentukan jumlah klaster. Salah satu pendekatan sederhana adalah memplot SSE untuk berbagai ukuran cluster. Kami mencari "siku" di mana SSE mulai turun. MiniBatchKMeans memperkenalkan beberapa kebisingan jadi saya menaikkan ukuran batch dan init lebih tinggi. Sayangnya implementasi Kmeans reguler terlalu lambat. Anda akan melihat keadaan acak yang berbeda akan menghasilkan grafik yang berbeda. Disini saya memilih 14 cluster.

# In[52]:


import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.cluster import MiniBatchKMeans
def find_optimal_clusters(MY_DATA, max_k):
    iters = range(2, max_k+1, 2)
    
    sse = []
    for k in iters:
        sse.append(MiniBatchKMeans(n_clusters=k, init_size=100, batch_size=100, random_state=20).fit(MY_DATA).inertia_)
        print('Fit {} clusters'.format(k))
        
    f, ax = plt.subplots(1, 1)
    ax.plot(iters, sse, marker='o')
    ax.set_xlabel('Cluster Centers')
    ax.set_xticks(iters)
    ax.set_xticklabels(iters)
    ax.set_ylabel('SSE')
    ax.set_title('SSE by Cluster Center Plot')
    
find_optimal_clusters(tf, 20)

