# -*- coding: utf-8 -*-

import sys, csv, re, html
import pickle, math
import pandas as pd
import numpy as np 
#import Word2Vec
from string import punctuation

from sklearn.naive_bayes import MultinomialNB

#import re
import nltk
nltk.download('stopwords')
nltk.download('words')
nltk.download('maxent_ne_chunker')
nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

from nltk.tokenize import word_tokenize,sent_tokenize
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from gensim.scripts.glove2word2vec import glove2word2vec  
#from glove import Glove
from gensim.models import Word2Vec
#nltk.download('punkt') 
train_file = "El-reg-En-full.csv"
dev_file = "2018-EI-reg-En-dev-full.csv"
test_file = "2018-EI-reg-En-test-full.csv"

def glovetoword2vec_conversion(glove_input_file, word2vec_output_file):
    glove2word2vec(glove_input_file, word2vec_output_file)

def strip_punctuation(s):
    return ''.join(c for c in s if c not in punctuation)

def stopwordsremoval_stemming(sentence):
    stopwords_removed = [word for word in sentence.split(' ') if word not in stopwords.words('english')]
    return stopwords_removed

def clean_str(string): 
    string = html.unescape(string)
    string = string.replace("\\n", " ") 
    string = re.sub(r"@[A-Za-z0-9_s(),!?\'\`]+", "", string)
    string = re.sub(r"\*", " ", string)
    string = re.sub(r"\'s", " ", string)
    string = re.sub(r"\'m", " \'m", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", "", string)
    string = re.sub(r"'", "", string) #Remove single code
    string = re.sub(r"!", " !", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)   
    string = re.sub(r"http.?://[^\s]+[\s]?", "", string) #remove url
    string = re.sub(r"\?", " ?", string)
    string = re.sub(r"\s{2,}", " ", string)
    return stopwordsremoval_stemming(string.strip().lower())

def preprocessing(train):
    processed_corpus = {} #All dataset, 
    train_labels = []  # list
    df=pd.read_csv(train, encoding='latin-1')
    
    i = 1
    id = df["ID"]
    train_sentences=df["Tweet"]
    train_labels=df["AffectDimension"]
    intensity_scores=df["IntensityScore"]
    #print(intensity_scores)
    
    #for (index, item) in train_labels.iteritems():
    #    train_labels.append(item)
    for (index, row), (index2,item2) in zip(train_sentences.iteritems(), id.iteritems()):
        sentence = sent_tokenize(row) # sentence tokenize, list of sentences
        words = word_tokenize(row)
        processed_tweet = []
        
        sen1 = clean_str(row)
        processed_tweet = sen1
        processed_corpus[item2] = processed_tweet
        i = i+1
     
    #for key, val in train_labels.items():
    #    print (key, val, "\n")
    return processed_corpus, train_labels

def bigram_list1(input_list_single):
    bigram_list = [] 
    for i in range(len(input_list_single)-1):
        bigram_list.append((input_list_single[i], input_list_single[i+1])) 
    return bigram_list

def find_bigrams(input_list):
    bigram_list = []
    processed_corpus = {}
    
    for key, val in input_list.items():
        bigram_list = bigram_list1(val)
        processed_corpus[key] = bigram_list
    #print(processed_corpus)
    return processed_corpus

def trigram_list1(input_list_single):
    trygram_list = [] 
    for i in range(len(input_list_single)-2):
        trygram_list.append((input_list_single[i], input_list_single[i+1], input_list_single[i+2]))
    return trygram_list


def find_trygrams(input_list):
    trygram_list = []
    processed_corpus = {}
    
    for key, val in input_list.items():
        trygram_list = trigram_list1(val)
        processed_corpus[key] = trygram_list
    #print(processed_corpus)
    return processed_corpus


def find_entropy(input_list): 
    processed_corpus = {}
    for key, val in input_list.items(): 
        entropy_of_row = 0.0
        for i in range(len(val)):
            prob = [ float(val[i].count(c)) / len(val[i]) for c in dict.fromkeys(list(val[i])) ]
            entropy = sum([ -p * math.log(p) / math.log(2.0) for p in prob ])
            entropy_of_row = entropy_of_row + entropy
        processed_corpus[key] = entropy_of_row
    return processed_corpus



def stemmer(input_list):
    s = WordNetLemmatizer()
    bigram_list = []
    for i in range(len(input_list)):
        bigram_list.append((s.lemmatize(input_list[i],'v')).lower()) #To remove ing too('v' for detecting verb convert right formate)
    return bigram_list

def stemming(process_list):
    processed_corpus = {} #All dataset,
    processed_corpus_stemmer = {}
    
    for index, row in process_list.items():        #print(index,row)
        sen1 = stemmer(row)
        processed_corpus_stemmer[index] = sen1
    #print("======After Stemming=======")
    #for key, val in processed_corpus_stemmer.items():
    #    print (key, val, "\n")   
    return processed_corpus_stemmer


def word_frequency(allwords, processed_data):
    allwords_Length = len(allwords)
    friq_list = 0
    counter = 0
    for i in range(len(processed_data)):
        for j in range(allwords_Length):
            if processed_data[i] == allwords[j]:
                counter = counter + 1
    rowlen = len(processed_data)
    if rowlen == 0:
        rowlen = 1
    friq_list = counter / rowlen 
    return friq_list

def pre_word_frequency(processed_data):
    allwords = []
    processed_corpus_stemmer = {}
    for key, val in processed_data.items():
        allwords = allwords + val
        
    for index, row in processed_data.items():
        sen1 = word_frequency(allwords, row)
        processed_corpus_stemmer[index] = sen1
    return processed_corpus_stemmer


def pre_word_frequency_dev(processed_data,new_processed_file_dev):
    allwords = []
    processed_corpus_stemmer = {}
    for key, val in processed_data.items():
        allwords = allwords + val
        
    for index, row in new_processed_file_dev.items():        
        #print(index,len(row))
        sen1 = word_frequency(allwords, row)
        processed_corpus_stemmer[index] = sen1
    return processed_corpus_stemmer

def pos_tag_word(processed_list):
    friq_list = []
    friq_list.append(nltk.pos_tag(processed_list))
    return friq_list

def pre_pos_tag_word(processed_data1):
    processed_corpus_pos = {}
    ##print("======Word POS analysis Start=======")
    for index, row in processed_data1.items():
        sen1 = pos_tag_word(row)
        processed_corpus_pos[index] = sen1
    #for key, val in processed_corpus_pos.items():
    #    print (key, val, "\n")
    
    #print("======Word POS Analysis End=======")
    return processed_corpus_pos

def tf_idf_word(allwords, single_list):
    friq_list = 0
    allword_length = len(allwords)
    counter = 0
    for i in range(len(single_list)):
        counter = 0
        for j in range(allword_length):
            if single_list[i] == allwords[j]:
                counter = counter + 1
        tf = (counter/allword_length)
        idf = (allword_length/(1+counter))
        friq_list = friq_list + (tf*idf)
    
    rowlen = len(single_list)
    if rowlen == 0:
        rowlen = 1
    return friq_list/rowlen

def pre_tf_idf_word(processed_data):
    processed_corpus_tfIDF = {}
    #print("======TF-IDF analysis Start=======")
    allwords = []
    #print("================================================")
    for key, val in processed_data.items():
        allwords = allwords + val 
    #print(allwords)
    for index, row in processed_data.items():
        sen1 = tf_idf_word(allwords, row)
        processed_corpus_tfIDF[index] = sen1
    #for key, val in processed_corpus_tfIDF.items():
    #    print (key, val, "\n")
    
    #print("======Word POS Analysis End=======")
    return processed_corpus_tfIDF


def pre_tf_idf_word_dev(processed_data,new_processed_file_dev):
    processed_corpus_tfIDF = {} 
    allwords = [] 
    for key, val in processed_data.items():
        allwords = allwords + val 
        
    for index, row in new_processed_file_dev.items():
        sen1 = tf_idf_word(allwords, row)
        processed_corpus_tfIDF[index] = sen1 
    return processed_corpus_tfIDF

def find_emotions_anger(new_processed_file):
    file = open("anger-synonym.txt", "r") 
    df = file.read()
    words = nltk.word_tokenize(df)
    sen1 = stemmer(words)
    counter = 0
    for i in range(len(new_processed_file)):
        for j in range(len(sen1)):
            if new_processed_file[i] == sen1[j]:
                counter = counter + 1
    #print("Anger",counter)
    return counter

def find_emotions_joy(new_processed_file):
    file = open("joy-synonym.txt", "r") 
    df = file.read()
    words = nltk.word_tokenize(df)
    sen1 = stemmer(words)
    counter = 0
    for i in range(len(new_processed_file)):
        for j in range(len(sen1)):
            if new_processed_file[i] == sen1[j]:
                counter = counter + 1
    #print("Joy",counter)
    return counter


def find_emotions_sadness(new_processed_file):
    file = open("sadness-synonym.txt", "r") 
    df = file.read()
    words = nltk.word_tokenize(df)
    sen1 = stemmer(words)
    counter = 0
    for i in range(len(new_processed_file)):
        for j in range(len(sen1)):
            if new_processed_file[i] == sen1[j]:
                counter = counter + 1
    #print("sadness",counter)
    return counter

def find_emotions_fear(new_processed_file):
    file = open("fear-synonym.txt", "r") 
    df = file.read()
    words = nltk.word_tokenize(df)
    sen1 = stemmer(words)
    counter = 0
    for i in range(len(new_processed_file)):
        for j in range(len(sen1)):
            if new_processed_file[i] == sen1[j]:
                counter = counter + 1
    #print("fear",counter)
    return counter
    
def pre_find_emotions_anger(processed_data1):
    processed_corpus_anger = {}
    processed_corpus_joy = {}
    processed_corpus_sadness = {}
    processed_corpus_fear = {}
    
    ##print("=========== Count Anger Words ===========")
    for index, row in processed_data1.items():
        sen_anger = find_emotions_anger(row)
        sen_joy = find_emotions_joy(row)
        sen_sadness = find_emotions_sadness(row)
        sen_fear = find_emotions_fear(row)
        
        processed_corpus_anger[index] = sen_anger
        processed_corpus_joy[index] = sen_joy
        processed_corpus_sadness[index] = sen_sadness
        processed_corpus_fear[index] = sen_fear
        
    #for key, val in processed_corpus_anger.items():
    #    print (key, val, "\n")
    
    #print("======Word POS Analysis End=======")
    return processed_corpus_anger, processed_corpus_joy, processed_corpus_sadness, processed_corpus_fear

def find_hasTag_anger(new_processed_file):
    file1 = open("hashTag-anger.txt", "r", encoding="latin-1") 
    file2 = open("hashTag-joy.txt", "r", encoding="latin-1") 
    file3 = open("hashTag-sadness.txt", "r", encoding="latin-1") 
    file4 = open("hashTag-fear.txt", "r", encoding="latin-1") 
    
    df1 = file1.read()
    df2 = file2.read()
    df3 = file3.read()
    df4 = file4.read()
    #print()
    words1 = nltk.word_tokenize(df1)
    words2 = nltk.word_tokenize(df2)
    words3 = nltk.word_tokenize(df3)
    words4 = nltk.word_tokenize(df4)
    counter_anger = 0
    counter_joy = 0
    counter_sadness = 0
    counter_fear = 0
    #print(words1)
    output1 = []
    output2 = []
    output3 = []
    output4 = []
    
    for j in range(len(words1)):
        output1.append(('#'+words1[j]).lower())
    for j in range(len(words2)):
        output2.append(('#'+words2[j]).lower())
    for j in range(len(words3)):
        output3.append(('#'+words3[j]).lower())
    for j in range(len(words4)):
        output4.append(('#'+words4[j]).lower())
    
    for i in range(len(new_processed_file)):
        for j in range(len(output1)):
            if new_processed_file[i] == output1[j]:
                counter_anger = counter_anger + 1
        for j in range(len(output2)):
            if new_processed_file[i] == output2[j]:
                counter_joy = counter_joy + 1
        for j in range(len(output3)):
            if new_processed_file[i] == output3[j]:
                counter_sadness = counter_sadness + 1
        for j in range(len(output4)):
            if new_processed_file[i] == output4[j]:
                counter_fear = counter_fear + 1
        
    #print("Anger",counter_anger)
    return counter_anger, counter_joy, counter_sadness, counter_fear
    
def hasTag_Count(processed_data):
    hasTag_Count_anger = {}
    hasTag_Count_joy = {}
    hasTag_Count_sadness = {}
    hasTag_Count_fear = {}
    
    #print("=========== Count Anger Words ===========")
    
    for index, row in processed_data.items():
        Tag_anger, Tag_joy, Tag_sadness, Tag_fear = find_hasTag_anger(row)
        #sen_joy = find_emotions_joy(row) , Tag_joy, Tag_sadness, Tag_fear
        #sen_sadness = find_emotions_sadness(row)
        #sen_fear = find_emotions_fear(row)
        
        hasTag_Count_anger[index] = Tag_anger
        hasTag_Count_joy[index] = Tag_joy
        hasTag_Count_sadness[index] = Tag_sadness
        hasTag_Count_fear[index] = Tag_fear
        
    #for key, val in hasTag_Count_anger.items():
    #    print (key, val, "\n")
    
    #print("======Word POS Analysis End=======")
    return hasTag_Count_anger, hasTag_Count_joy, hasTag_Count_sadness, hasTag_Count_fear
    
def unique_unigram(processed_data):
    processed_corpus_tfIDF = {}
    #print("======TF-IDF analysis Start=======")
    allwords = []
    bigram_list = []
    seen = []
    count_all = 0
    count_unique = 0
    #print("================================================")
    for key, val in processed_data.items():
        allwords = allwords + val 
    #print(allwords)
    for i in range(len(allwords)):
        count_all = count_all + 1
        #print(allwords[i])
        if allwords[i] not in seen:
            count_unique = count_unique + 1
            seen.append(allwords[i])
   # print(seen)
    #print("Total WOrds: ",count_all,"\n","Unique words: ", count_unique)
    for key, val in processed_data.items():
        bigram_list = []
        for i in range(len(seen)): #loop goes 0 to 95
            counter = 0
            for j in range(len(val)):
                if seen[i] == val[j]:
                    counter = counter + 1
            bigram_list.append(counter) #To remove in
        processed_corpus_tfIDF[key] = bigram_list
    #print("sadness",counter)
    #for key, val in processed_corpus_tfIDF.items():
    #    print(key,val) 
    return processed_corpus_tfIDF, seen

  
def unique_unigram_dev(processed_data,new_processed_file_dev):
    processed_corpus_tfIDF = {} 
    allwords = []
    bigram_list = []
    seen = []
    count_all = 0
    count_unique = 0
    #print("================================================")
    for key, val in processed_data.items():
        allwords = allwords + val  
    for i in range(len(allwords)):
        count_all = count_all + 1 
        if allwords[i] not in seen:
            count_unique = count_unique + 1
            seen.append(allwords[i]) 
            
    for key, val in new_processed_file_dev.items():
        bigram_list = []
        for i in range(len(seen)): #loop goes 0 to 95
            counter = 0
            for j in range(len(val)):
                if seen[i] == val[j]:
                    counter = counter + 1
            bigram_list.append(counter) #To remove in
        processed_corpus_tfIDF[key] = bigram_list 
    return processed_corpus_tfIDF, seen


def remove2(x):
    return x[1:-1]

def remove(x):
    return x.replace("'", "")

def main():
    c = {}
    processed_file, train_label = preprocessing(train_file)
    new_processed_file = stemming(processed_file)
    
    #unique_unigram_feature, unique_words =unique_unigram(new_processed_file)
    
    entropy_value = find_entropy(new_processed_file)
    word_frequency = pre_word_frequency(new_processed_file)
    pos_features = pre_pos_tag_word(new_processed_file)
    tf_idf_feature = pre_tf_idf_word(new_processed_file)
    emo_anger, emo_joy, emo_sadness, emo_fear = pre_find_emotions_anger(new_processed_file)
    hasTag_Count_anger, hasTag_Count_joy, hasTag_Count_sadness, hasTag_Count_fear = hasTag_Count(new_processed_file)
    
    #training
    
    download_dir = "Final-train-dataset-all-without-Unig.csv" #where you want the file to be downloaded to 
    csv = open(download_dir, "w") 
    feature= "label"
    for row in range(0,11):
        feature= str(feature) + "," + "name" + str(row)
    row = feature + "\n"
    csv.write(row)
    i=0
    processed_corpus_tfIDF ={} 
    for key, val in processed_file.items():
        bigram_list = []
        bigram_list = str(train_label[i]) + ',' + str(entropy_value[key])+','+ str(word_frequency[key]) +',' \
        + str(tf_idf_feature[key]) +',' + str(emo_anger[key]) +',' + str(emo_joy[key]) +',' + str(emo_sadness[key]) \
        +','+ str(emo_fear[key])+',' + str(hasTag_Count_anger[key])+',' + str(hasTag_Count_joy[key])+',' \
        + str(hasTag_Count_sadness[key])+',' + str(hasTag_Count_fear[key]) + '\n'
        csv.write(bigram_list)
        print(key, bigram_list)
        i += 1
        processed_corpus_tfIDF[key] = bigram_list
    
    
    #===================Dev file processing=============================#
    processed_file_dev, dev_label = preprocessing(dev_file)
    new_processed_file_dev = stemming(processed_file_dev)
    ##unique_unigram_feature_dev, unique_words =unique_unigram_dev(new_processed_file,new_processed_file_dev)
    #print(unique_words)
    
    entropy_value_dev = find_entropy(new_processed_file_dev)
    word_frequency_dev = pre_word_frequency_dev(new_processed_file,new_processed_file_dev)
    tf_idf_feature_dev = pre_tf_idf_word_dev(new_processed_file,new_processed_file_dev)
    emo_anger_dev, emo_joy_dev, emo_sadness_dev, emo_fear_dev = pre_find_emotions_anger(new_processed_file_dev)
    hasTag_C_anger_dev, hasTag_C_joy_dev, hasTag_C_sadness_dev, hasTag_C_fear_dev = hasTag_Count(new_processed_file_dev)
    
    #validation
    download_dir = "Final-dev-dataset-all-without-uni.csv" #where you want the file to be downloaded to 
    csv = open(download_dir, "w") 
    feature= "label"
    for row in range(0,11):
        feature= str(feature) + "," + "name" + str(row)
    row = feature + "\n"
    csv.write(row)
    i=0
    processed_corpus_tfIDF ={} 
    for key, val in new_processed_file_dev.items():
        bigram_list = []
        bigram_list = str(dev_label[i]) +','+ str(entropy_value_dev[key])+','+ str(word_frequency_dev[key]) +',' \
        + str(tf_idf_feature_dev[key]) +',' + str(emo_anger_dev[key]) +',' + str(emo_joy_dev[key]) +',' \
        + str(emo_sadness_dev[key]) \
        +','+ str(emo_fear_dev[key])+',' + str(hasTag_C_anger_dev[key])+',' + str(hasTag_C_joy_dev[key])+',' \
        + str(hasTag_C_sadness_dev[key])+',' + str(hasTag_C_fear_dev[key]) + '\n'
        csv.write(bigram_list)
        #print(key, bigram_list)
        i += 1
        processed_corpus_tfIDF[key] = bigram_list
    
    print ("Dev processed_file done!") #3
    
    #===================Test file processing=============================#
    processed_file_test, test_label = preprocessing(test_file) 
    new_processed_file_test = stemming(processed_file_test)
    #unique_unigram_feature_test, unique_words =unique_unigram_dev(new_processed_file,new_processed_file_test)
    #print("HHHHHHHHHHHHHHHHHHH:",unique_unigram_feature_test)
    
    entropy_value_test = find_entropy(new_processed_file_test)
    word_frequency_test = pre_word_frequency_dev(new_processed_file,new_processed_file_test)
    tf_idf_feature_test = pre_tf_idf_word_dev(new_processed_file,new_processed_file_test)
    emo_anger_test, emo_joy_test, emo_sadness_test, emo_fear_test = pre_find_emotions_anger(new_processed_file_test)
    hasTag_C_anger_test, hasTag_C_joy_test, hasTag_C_sadness_test, hasTag_C_fear_test = hasTag_Count(new_processed_file_test)
    
    #testing
    
    download_dir = "Final-test-dataset-all-without-Unig.csv" #where you want the file to be downloaded to 
    csv = open(download_dir, "w") 
    feature= "label"
    for row in range(0,11):
        feature= str(feature) + "," + "name" + str(row)
    row = feature + "\n"
    csv.write(row)
    i=0
    processed_corpus_tfIDF ={} 
    for key, val in processed_file_test.items():
        bigram_list = []
        bigram_list = str(test_label[i]) + ',' + str(entropy_value_test[key])+','+ str(word_frequency_test[key]) +',' \
        + str(tf_idf_feature_test[key]) +',' + str(emo_anger_test[key]) +',' + str(emo_joy_test[key]) +',' \
        + str(emo_sadness_test[key]) \
        +','+ str(emo_fear_test[key])+',' + str(hasTag_C_anger_test[key])+',' + str(hasTag_C_joy_test[key])+',' \
        + str(hasTag_C_sadness_test[key])+',' + str(hasTag_C_fear_test[key]) + '\n'
        csv.write(bigram_list)
        print(key, bigram_list)
        i += 1
        processed_corpus_tfIDF[key] = bigram_list
    print ("Test processed_file done!") #3
    
    print ("Code End")
if __name__=="__main__":
    main()