#!/usr/bin/env python
# coding: utf-8

# ## Topic Modelling pada Web Mining

# Langkah-langkah:
# 
# 
# 1.   Crawling data
# 2.   Preprocessing data (Cleansing data dan stopword)
# 3.   Modelling LSA
# 

# ### 1. Pengertian Web Mining

# **Web Content Mining** merupakan suatu proses otomatis untuk menemukan
# informasi yang berguna dari dokumen atau data. Pada prinsipnya teknik ini mengekstraksi kata kunci yang terkandung pada dokumen. Teknik Web Content Mining berfokus pada penggalian informasi dari isi (content) yang ada pada sebuah web, seperti teks, gambar, audio, video, metadata dan hyperlink. Ada dua strategi yang umum digunakan:
# 
# 1.   Langsung melakukan mining terhadap data
# 2.   Melakukan pencarian serta mengimprove hasil pencarian seperti layaknya search engine (Srivastava et al, 2000).
# 
# 
# Web Content Mining juga dapat disebut sebagai Web Text Mining yang dalam implementasi biasanya penggalian data terletak pada daerah teks. Teknik ini juga lebih banyak berhubungan dengan disiplin information retrieval (IR), tetapi dengan inovasi yang terus menerus dan seiring dengan perkembangan dunia teknologi informasi, secara dramatis akan meningkatkan akurasi dalam analisis dan mampu menghemat biaya (Hermanto dan Sartono, _). 

# ### 2. Pengertian Data Crawling

# **Data Crawling** merupakan proses pengumpulan data sekunder yang mampu memilah informasi secara cepat dan otomatis tanpa ada reduplikasi data. Proses ini memiliki kaitan dengan data scraping. Begitu semua data telah terkumpul melalui crawling, data scraping akan mengeksekusinya.
# 
# Dataset yang digunakan adalah hasil crawling data pada **Portal Tugas Akhir Universitas Trunojoyo Madura**. Dapat diakses melalui link https://pta.trunojoyo.ac.id/.
# 
# Berikut adalah kode program untuk proses crawling data yang menghasilkan sejumlah 93 data. Dengan rincian terdapat 7 kolom yaitu kolom Judul, Penulis, Pembimbing 1, Pembimbing 2, abstrak, Kata Kunci dan Label. Untuk pelabelan dilakukan secara manual.
# 
# Untuk video tutorialnya dapat diakses pada link Youtube berikut https://youtu.be/uDeoJtYMbEU.

# ### 3. Preprocessing Data

# **Data preprocessing** merupakan proses yang mengubah data mentah ke dalam bentuk yang lebih mudah dipahami. Proses ini penting dilakukan karena data mentah sering kali tidak memiliki format yang teratur. Selain itu, data mining juga tidak dapat memproses data mentah, sehingga proses ini sangat penting dilakukan untuk mempermudah proses berikutnya, yakni analisis data.
# 
# Langkah-langkah preprocessing data:
# 
# 1.   Cleansing data
# 2.   Stopword
# 
# 

# #### 1. Import Module

# In[1]:


# data visualisation and manipulation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
import nltk
nltk.download('stopwords')
#configure
# sets matplotlib to inline and displays graphs below the corressponding cell.
get_ipython().run_line_magic('matplotlib', 'inline')
style.use('fivethirtyeight')
sns.set(style='whitegrid',color_codes=True)

#import nltk
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize,sent_tokenize

#preprocessing
from nltk.corpus import stopwords  #stopwords
from nltk import word_tokenize,sent_tokenize # tokenizing
from nltk.stem import PorterStemmer,LancasterStemmer  # using the Porter Stemmer and Lancaster Stemmer and others
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer  # lammatizer from WordNet

# for named entity recognition (NER)
from nltk import ne_chunk

# vectorizers for creating the document-term-matrix (DTM)
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer

#stop-words
stop_words=set(nltk.corpus.stopwords.words('indonesian'))


# #### 2. Memuat Dataset

# In[2]:


df=pd.read_csv("dataset.csv")


# In[3]:


df.head(54)


# Hanya kolom **'abstrak'** yang akan diproses.

# In[4]:


# drop the publish date.
df.drop(['Dokumen'],axis=1,inplace=True)


# In[5]:


df.head(54)


# #### 3. Cleaning Data dan Preprocessing Data

# Proses ini merupakan preprocessing data dengan menggunakan lemmatizer dan bisa juga menggunakan stemmer. Melakukan stopword pada kata yang telah digunakan bersama dengan kata-kata dengan panjang lebih pendek dari 3 karakter untuk mengurangi beberapa kata yang menyimpang.

# In[6]:


def clean_text(headline):
  le=WordNetLemmatizer()
  word_tokens=word_tokenize(headline)
  tokens=[le.lemmatize(w) for w in word_tokens if w not in stop_words and len(w)>3]
  cleaned_text=" ".join(tokens)
  return cleaned_text


# In[7]:


# time taking
import nltk
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('omw-1.4')
df['abstrak_cleaned_text']=df['abstrak'].apply(clean_text)


# In[8]:


df.head(54)


# Dapat melihat perbedaan setelah penghapusan stopwords dan beberapa kata yang lebih pendek. Juga kata-kata telah dilemmatisasi seperti dalam 'yang', 'dengan', 'adalah', dll.

# Kemudian gunakan kolom yang belum diproses.

# In[9]:


df.drop(['abstrak'],axis=1,inplace=True)


# In[10]:


df.head(54)


# Kita juga bisa melihat headline berita tertentu.

# In[11]:


df['abstrak_cleaned_text'][0]


# #### 4. Mengekstraksi Fitur dan Membuat Document Term Matrix (DTM)
# 
# Dalam DTM nilainya adalah nilai TFidf. Tentukan beberapa parameter dari vectorizer Tfidf. Beberapa poin penting:
# 
# 1.   LSA umumnya diimplementasikan dengan nilai Tfidf di mana-mana dan bukan dengan Count Vectorizer.
# 2.   Max_features bergantung pada daya komputasi Anda dan juga pada eval. metrik (skor koherensi adalah metrik untuk model topik). Coba nilai yang memberikan evaluasi terbaik. metrik dan tidak membatasi kekuatan pemrosesan.
# 3.   Nilai default untuk min_df & max_df bekerja dengan baik.
# 4.   Dapat mencoba nilai yang berbeda untuk ngram_range.

# In[12]:


vect =TfidfVectorizer(stop_words=stop_words,max_features=1000) # to play with. min_df,max_df,max_features etc...


# In[13]:


vect_text=vect.fit_transform(df['abstrak_cleaned_text'])


# Sekarang kita dapat melihat kata-kata yang paling sering dan langka di headline berita berdasarkan skor idf. Semakin kecil nilainya; lebih umum adalah kata dalam berita utama.

# In[14]:


print(vect_text.shape)
print(vect_text)


# In[15]:


idf=vect.idf_


# In[16]:


dd=dict(zip(vect.get_feature_names(), idf))
l=sorted(dd, key=(dd).get)
# print(l)
print(l[0],l[-1])
print(dd['penelitian'])
print(dd['yamaha'])  # police is most common and forecast is least common among the news headlines.


# Oleh karena itu kita dapat melihat bahwa berdasarkan **nilai idf** , **'hasil'** adalah kata **paling sering** sedangkan **'wisata'** **paling jarang** muncul di antara abstrak.

# ### 4. Topic Modelling

# Suatu metode pada  analisa penambangan teks untuk melakukan penemuan data-data teks yang tersembunyi dan menemukan hubungan antara teks yang satu dengan lainnya dari suatu corpus.
# * Mengelompokkan data teks yang didasarkan pada topik tertentu.
# * Ide-tema-pokok bahasan utama dalam suatu kumpulan teks.
# 
# Diimplementasikan untuk :
# * Mengetahui topik bahasan di media sosial
# * Mengetahui topik penelitian
# * Mengetahui topik yang dibahas dalam review produk, dll.
# 
# Tugas pokok topik modelling :
# * Menemukan topik dalam kumpulan teks (Corpus) 
# * Mengetahui proporsi topik dalam suatu dokumen
# 
# **Topic** mengandung  distribusi dari beberapa kosakata (term) yang bersifat tetap.
# 
# **Dokumen** mengandung proporsi topik-topik tertentu sesuai kata-kata yang terkandung di dalamnya.

# #### 1. Latent Semantic Analysis (LSA)

# Pendekatan pertama yang saya gunakan adalah LSA. LSA pada dasarnya adalah dekomposisi nilai tunggal. 
# 
# SVD menguraikan DTM asli menjadi tiga matriks S=U.(sigma).(V.T). Di sini matriks U menunjukkan matriks topik-dokumen sedangkan (V) adalah matriks istilah-topik.
# 
# Setiap baris dari matriks U(matriks term dokumen) adalah representasi vektor dari dokumen yang bersangkutan. Panjang vektor ini adalah jumlah topik yang diinginkan. Representasi vektor untuk suku-suku dalam data kami dapat ditemukan dalam matriks V (matriks istilah-topik).
# 
# Jadi, SVD memberi kita vektor untuk setiap dokumen dan istilah dalam data kita. Panjang setiap vektor adalah k. Kemudian kita dapat menggunakan vektor-vektor ini untuk menemukan kata dan dokumen serupa menggunakan metode kesamaan kosinus.
# 
# Kita dapat menggunakan fungsi truncatedSVD untuk mengimplementasikan LSA. Parameter n_components adalah jumlah topik yang ingin kita ekstrak.
# Model tersebut kemudian di fit dan ditransformasikan pada hasil yang diberikan oleh vectorizer.
# 
# Terakhir perhatikan bahwa LSA dan LSI (I untuk pengindeksan) adalah sama dan yang terakhir hanya terkadang digunakan dalam konteks pencarian informasi.

# Rumus Singular Value Decomposition (SVD) :

# $\begin{equation}
# A_{m n}=U_{m m} x S_{m n} x V_{n n}^{T}
# \end{equation}$

# Rumus Cosine Similarity :

# $\begin{equation}
# \cos \alpha=\frac{A \cdot B}{|A||B|}=\frac{\sum_{i=1}^{n} A_{i} X B_{i}}{\sqrt{\sum_{i=1}^{n}\left(A_{i}\right)^{2}} X \sqrt{\sum_{i=1}^{n}\left(B_{i}\right)^{2}}}
# \end{equation}$

# In[17]:


from sklearn.decomposition import TruncatedSVD
lsa_model = TruncatedSVD(n_components=10, algorithm='randomized', n_iter=10, random_state=42)

lsa_top=lsa_model.fit_transform(vect_text)


# In[18]:


print(lsa_top)
print(lsa_top.shape)  # (no_of_doc*no_of_topics)


# In[19]:


l=lsa_top[0]
print("Document 0 :")
for i,topic in enumerate(l):
  print("Topic ",i," : ",topic*100)
  


# Mirip dengan dokumen lain kita bisa melakukan ini. Namun perhatikan bahwa nilai tidak menambah 1 seperti di LSA itu bukan kemungkinan topik dalam dokumen.

# In[20]:


print(lsa_model.components_.shape) # (no_of_topics*no_of_words)
print(lsa_model.components_)


# Proses untuk mendapatkan daftar kata-kata penting untuk masing-masing dari 10 topik seperti yang ditunjukkan. Untuk kesederhanaan di sini saya telah menunjukkan 10 kata untuk setiap topik.

# In[21]:


# most important words for each topic
vocab = vect.get_feature_names()

for i, comp in enumerate(lsa_model.components_):
    vocab_comp = zip(vocab, comp)
    sorted_words = sorted(vocab_comp, key= lambda x:x[1], reverse=True)[:10]
    print("Topic "+str(i)+": ")
    for t in sorted_words:
        print(t[0],end=" ")
    print("\n")
         


# #### 2. Latent Dirichlet Allocation (LDA)  

# **Latent Dirichlet Allocation** merupakan model generatif yaitu model yang menghasilkan distribusi output dan input didasarkan pada variabel-variabel latent.
# 
# LDA adalah teknik yang paling populer. Topik kemudian menghasilkan kata berdasarkan distribusi probabilitasnya. Mengingat kumpulan data dokumen, LDA mundur dan mencoba mencari tahu topik apa yang akan membuat dokumen tersebut di tempat pertama.
# 
# Untuk memahami matematika sepertinya diperlukan pengetahuan tentang distribusi Dirichlet (distribusi distribusi) yang cukup rumit dan ditinggalkan untuk saat ini.

# Rumus LDA :

# $\begin{equation}
# p(D \mid a, \beta)=\prod_{d=1}^{M} \int p\left(\theta_{\mathrm{d}} \mid \alpha\right)\left(\prod_{n=1}^{N_{\mathrm{d}}} \sum_{Z_{\mathrm{dn}}} p\left(Z_{\mathrm{dn}} \mid \theta_{\mathrm{d}}\right) p\left(W_{\mathrm{dn}} \mid Z_{\mathrm{dn}}, \beta\right)\right) d \theta_{\mathrm{d}}
# \end{equation}$

# In[22]:


from sklearn.decomposition import LatentDirichletAllocation
lda_model=LatentDirichletAllocation(n_components=10,learning_method='online',random_state=42,max_iter=1) 
# n_components is the number of topics


# In[23]:


lda_top=lda_model.fit_transform(vect_text)


# In[24]:


print(lda_top.shape)  # (no_of_doc,no_of_topics)
print(lda_top)


# In[25]:


sum=0
for i in lda_top[0]:
  sum=sum+i
print(sum)  


# Perhatikan bahwa nilai pada baris tertentu ditambahkan ke 1. Ini karena setiap nilai menunjukkan % kontribusi topik terkait dalam dokumen.

# In[26]:


# composition of doc 0 for eg
print("Document 0: ")
for i,topic in enumerate(lda_top[0]):
  print("Topic ",i,": ",topic*100,"%")


# Seperti yang kita lihat, Topik 7 & 8 dominan ada di dokumen 0.

# In[27]:


print(lda_model.components_)
print(lda_model.components_.shape)  # (no_of_topics*no_of_words)


# Kata-kata paling penting untuk sebuah topik. (katakan 10 kali ini)

# In[28]:


# most important words for each topic
vocab = vect.get_feature_names()

for i, comp in enumerate(lda_model.components_):
    vocab_comp = zip(vocab, comp)
    sorted_words = sorted(vocab_comp, key= lambda x:x[1], reverse=True)[:10]
    print("Topic "+str(i)+": ")
    for t in sorted_words:
        print(t[0],end=" ")
    print("\n")


# Untuk memvisualisasikan kata-kata dalam suatu topik dengan lebih baik, kita dapat melihat kata cloud. Untuk setiap topik, 50 kata teratas diplot.

# In[29]:


get_ipython().system(' pip install wordcloud')


# In[30]:


from wordcloud import WordCloud
# Generate a word cloud image for given topic

def draw_word_cloud(index):
  imp_words_topic=""
  comp=lda_model.components_[index]
  vocab_comp = zip(vocab, comp)
  sorted_words = sorted(vocab_comp, key= lambda x:x[1], reverse=True)[:50]
  for word in sorted_words:
    imp_words_topic=imp_words_topic+" "+word[0]

  wordcloud = WordCloud(width=600, height=400).generate(imp_words_topic)
  plt.figure( figsize=(5,5))
  plt.imshow(wordcloud)
  plt.axis("off")
  plt.tight_layout()
  plt.show()


# In[31]:


# topic 0
draw_word_cloud(0)


# In[32]:


# topic 1
draw_word_cloud(1)  # ...

