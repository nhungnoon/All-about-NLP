
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import nltk
import os


# ### Data Source
# ### https://www.kaggle.com/sanglequang/van-hoc-viet-nam

# In[2]:


# list current directory os.listdir(DIR): os.getcwd()
DIR = "con-hoang_HBC/"
valid_file = ['.txt']
doc_sum = []
for text in os.listdir(DIR):
    ext = os.path.splitext(text)[1]
    if ext.lower() in valid_file:
        text_ = open(os.path.join(DIR,text)).read()
        doc_sum.append(text_)


# In[65]:


doc_sum[0][0:32]


# ### Tokenize words 

# In[4]:


#split words in each list
base_tokens = [i.split() for i in doc_sum]
#make list of list to become list
list_tokens = [item for sublist in base_tokens for item in sublist]


# In[66]:


list_tokens[0:10]


# ### Word Frequency

# In[6]:


freq_uni_base_corpus = nltk.FreqDist(list_tokens)
print(freq_uni_base_corpus.most_common(20))


# ### Calculate the probability of the word appear in the corpus

# In[7]:


smoothed_dist_uni_base = nltk.LaplaceProbDist(freq_uni_base_corpus)


# In[8]:


smoothed_dist_uni_base.logprob('gà rán')


# In[9]:


smoothed_dist_uni_base.logprob('thuộc')


# ### Bi-gram Word Frequency

# In[10]:


from nltk.util import bigrams
#frequency of bigrams for base corpus
base_bigrams = nltk.bigrams(list_tokens)
freq_bi_base_corpus = nltk.FreqDist(base_bigrams)
print(freq_bi_base_corpus.most_common(20))


# ### Topic Modelling using Gensim  

# In[67]:


# Topic Modelling using Gensim 
import gensim
from gensim import corpora
# Create dictionary based on the list of tokens
dictionary = corpora.Dictionary(d.split() for d in list_tokens)


# In[16]:


# create word_matrix to input into the model
list_2 = [d.split() for d in list_tokens]
word_matrix = [dictionary.doc2bow(doc) for doc in list_2]


# In[17]:


# Create the model
Lda = gensim.models.ldamodel.LdaModel
ldamodel = Lda(word_matrix, num_topics=5, id2word = dictionary, passes=50)
#passes equal epochs - how many times the models keeps training


# In[18]:


# Show topic
topics = ldamodel.show_topics(num_topics=3, num_words=10, formatted=False)
topics[0][1] # show the word and the possibility 


# ### Miscellanous - Splitting lists into dataframe 

# In[68]:


text_ = open(os.path.join(DIR,'part0020.txt')).read()


# In[69]:


sentence = text_.split("\n")


# In[71]:


sentence[0]


# In[72]:


df = pd.DataFrame(np.array(sentence).reshape(40,2), columns = list("ab"))
df.head(2)

