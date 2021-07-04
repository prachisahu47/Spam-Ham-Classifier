#!/usr/bin/env python
# coding: utf-8

# In[127]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nltk import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
# import nltk
# nltk.download('stopwords')

df = pd.read_csv('trainingData.csv')
df.head()
df.shape


#Labels of Train data

y_train = np.array([])
y_train = df['label_num'].to_numpy()


#Function to Process the text data and 1. Remove Punctuation 2.Stop Words 3.Stemming
#import string
#from nltk.corpus import stopwords
#def process_text(text):
#     no_punc = [char for char in text if char not in string.punctuation]
#     no_punc = ''.join(no_punc)
#     return ' '.join([word for word in no_punc.split() if word.lower() not in stopwords.words('english')])

#df['text']=df['text'].apply(process_text)


def count_words(text):
    words = word_tokenize(text)
    return len(words)
df['count']=df['text'].apply(count_words)


#Convert mails to feature vectors
vectorizer= CountVectorizer()
message_bow = vectorizer.fit_transform(df['text'])

#Extract training data in x_train


# print(vectorizer.get_feature_names())
# print(message_bow)
x_train = np.array([[]])
x_train = message_bow.toarray()
# print(training_data)

#n(rows) and n(cols)
num_rows, num_cols = x_train.shape

# for i in range(num_rows):
#     print(np.sum(training_data[i]))
#     print(np.sum(c[i]))

#Convert >1 to 1
c = (x_train >= 1).astype(int)
x_train=c
# for i in range(num_rows):
#     print(np.sum(c[i]))

# for i in range(num_rows):
#     for j in range(num_cols):
#         if(training_data[i][j]>1):
#             training_data[i][j] = 1
#     print(np.sum(training_data[i]))








# In[128]:


#Information gain of Target attribute i.e. y_train
total_label = y_train.size
print(total_label)

P_tar = np.sum(y_train)
N_tar = total_label - P_tar


IG = np.negative((P_tar/total_label)*np.log2(P_tar/total_label) + (N_tar/total_label)*np.log2(N_tar/total_label))

print(IG)


# In[129]:


import math
#Calculate Entropy, Gain of each features
n_col = int(x_train.size/total_label)
print(n_col)

gain = np.array([])
top_features = np.array([])
 
#table will be like [[NN,NY],[YN,YY]] where rows are features and cols are labels(Y=1,N=0)
for col in range(n_col):
    table = np.array([[0,0],[0,0]])
    #prob = 0
    for row in range(total_label):
        #prob = prob + x_train[row][col]
        table[x_train[row][col]][y_train[row]] = table[x_train[row][col]][y_train[row]] + 1
#     print(table)
    #print(prob)
    total_label1 = table[0][0] + table[0][1]
    total_label2 = table[1][0] + table[1][1]
#     print(total_label1)
#     print(total_label2)
    t1 = (table[0][0]/total_label1)*np.log2(table[0][0]/total_label1)
    t2 = (table[0][1]/total_label1)*np.log2(table[0][1]/total_label1)
    t3 = (table[1][0]/total_label2)*np.log2(table[1][0]/total_label2)
    t4 = (table[1][1]/total_label2)*np.log2(table[1][1]/total_label2)
    if(math.isnan(t1)):
        t1 = 0
    if(math.isnan(t2)):
        t2 = 0
    if(math.isnan(t3)):
        t3 = 0
    if(math.isnan(t4)):
        t4 = 0
    entropy = np.negative(t1 + t2)*((total_label1)/total_label) + np.negative(t3 + t4)*(total_label2/total_label)
#     print(entropy)
    gain = np.append(gain,(IG-entropy))
       


# In[132]:


#sorted index of features in descending order og information gain
top_features = gain.argsort()[-n_col:][::-1]


# In[133]:


with open('topFeatures.txt', 'w') as filehandle:
    for item in top_features:
        filehandle.write('%s\n' % item)


# In[6]:


with open('AllFeatures.txt', 'w') as filehandle:
    for item in vectorizer.get_feature_names():
        filehandle.write('%s\n' % item)


# In[7]:


file1=open('topFeatures.txt','r')
import numpy as np
topFeatures=np.loadtxt(file1,delimiter='\n')


# In[134]:


required_features=500
FeaturesIndex=topFeatures[0:required_features]


# In[135]:


FeaturesIndex=np.sort(FeaturesIndex)


# In[136]:


FeaturesIndex=FeaturesIndex.astype(int)


# In[75]:


features = []

with open('AllFeatures.txt', 'r') as filehandle:
    for line in filehandle:
        currentPlace = line[:-1]
        features.append(currentPlace)


# In[137]:


final_features=list( features[i] for i in FeaturesIndex )


# In[145]:


with open('final_features.txt', 'w') as filehandle:
    for item in final_features:
        filehandle.write('%s\n' % item)


# In[138]:


final_data=x_train[:,FeaturesIndex]


# In[139]:


num_cols=len(FeaturesIndex)


# In[140]:


#Naive Bayes model training
label=np.ones(2)
feature_estimator=np.ones((2,num_cols)) #bernouli featueres estimators
for i in range(num_rows):
    if y_train[i]==0 :
        index=0
    else:
        index=1
    label[index]+=1
    feature_estimator[index]=np.add(feature_estimator[index],final_data[i])

feature_estimator[0]=np.divide(feature_estimator[0],label[0])
feature_estimator[1]=np.divide(feature_estimator[1],label[1])


# In[141]:


with open('naiveBayesFeature0.txt', 'w') as filehandle:
    for item in feature_estimator[0]:
        filehandle.write('%s\n' % item)


# In[142]:


with open('naiveBayesFeature1.txt', 'w') as filehandle:
    for item in feature_estimator[1]:
        filehandle.write('%s\n' % item)


# In[143]:



p_spam=0
p_spam=np.sum(y_train)
p_spam=p_spam/num_rows
print(p_spam)


# In[144]:


with open('naiveBayesPspam.txt', 'w') as filehandle:
        filehandle.write('%s\n' % p_spam)

