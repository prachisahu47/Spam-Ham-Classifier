#!/usr/bin/env python
# coding: utf-8

# In[13]:


features = []

with open('final_features.txt', 'r') as filehandle:
    for line in filehandle:
        currentPlace = line[:-1]
        features.append(currentPlace)


# In[14]:


import numpy as np
predictors_ham =np.array([])
file1=open('naiveBayesFeature0.txt', 'r')
predictors_ham=np.loadtxt(file1,delimiter='\n')


# In[15]:


predictors_spam =np.array([])
file2=open('naiveBayesFeature1.txt', 'r')
predictors_spam=np.loadtxt(file2,delimiter='\n')


# In[16]:



file3=open('naiveBayesPspam.txt', 'r')
p_spam=np.loadtxt(file3,delimiter='\n')


# In[17]:


num_cols=len(features)


# In[18]:


import os
import numpy as np
import pandas as pd
from nltk import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from io import StringIO 


# In[19]:


DIR = 'test'
length_dir = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])


# In[20]:


import string
from nltk.corpus import stopwords

def process_text(text):
    no_punc = [char for char in text if char not in string.punctuation]
    no_punc = ''.join(no_punc)
    return ' '.join([word for word in no_punc.split() if word.lower() not in stopwords.words('english')])


# In[21]:


predicted_label=np.array([])
name = "email"
ext = ".txt"
for i in range(length_dir):
    f_name = DIR+"/"+name+str(i)+ext
    f = open(f_name, "r")
    f_data = f.read()
    f_data=process_text(f_data)
    #print(abc)
    tokens = word_tokenize(f_data)
    arr = np.array(tokens)
    testEmail=np.zeros(num_cols)
    
    for j in arr:
        try:
            indx=features.index(j)
            testEmail[indx]=1
        except:{}
    p_label_0=np.power(predictors_ham[0],testEmail)
    p0=np.prod(p_label_0)
    p0=p0*(1-p_spam)
    p_label_1=np.power(predictors_spam[1],testEmail)
    p1=np.prod(p_label_1)
    p1=p1*p_spam
    if p0>p1:
        predicted_label=np.append(predicted_label,0)
    else:
        predicted_label=np.append(predicted_label,1)
        


# In[22]:


predicted_label=predicted_label.astype(int)
print(predicted_label)

