# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 03:39:42 2020

@author: bushra
"""

 # -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 14:28:29 2019

@author: bushra
"""
import time

import codecs
codecs.register_error("strict", codecs.ignore_errors)

import random
import validators
from urllib.parse import urlunparse,urlparse
import re
import pandas as pd
#from rfc3986 import urlparse
from rfc3986 import is_valid_uri
import itertools
import tldextract

import homoglyphs as hg,os
import sys 
import numpy as np
from nltk import FreqDist
from nltk.corpus import gutenberg
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import re
import random
import csv
from nltk.corpus import brown
from collections import Counter
import urllib.request
from difflib import SequenceMatcher
import homoglyphs as hg,os
from urllib.request import Request, urlopen
import time
start = time.time()
homoglyphs =hg.Homoglyphs(languages={'en'},
            strategy=hg.STRATEGY_LOAD,
            ascii_strategy=hg.STRATEGY_REMOVE
        )
import random
import validators
import re
import pandas as pd
from rfc3986 import urlparse
from rfc3986 import is_valid_uri
import itertools
import sys 
from nltk import FreqDist
from nltk.corpus import gutenberg
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import re
import random
import csv
from nltk.corpus import brown
from collections import Counter
import urllib.request
import spacy
import pandas as pd 
import codecs,re
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec


import pkg_resources
from symspellpy import SymSpell, Verbosity

sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
dictionary_path = pkg_resources.resource_filename(
    "symspellpy", "frequency_dictionary_en_82_765.txt")
bigram_path = pkg_resources.resource_filename(
    "symspellpy", "frequency_bigramdictionary_en_243_342.txt")
# term_index is the column of the term and count_index is the
# column of the term frequency
sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
sym_spell.load_bigram_dictionary(bigram_path, term_index=0, count_index=2)

def get_related(seed,n=10):
  if(seed in nlp.vocab):
      word=nlp.vocab[seed]
      
      filtered_words = [w for w in word.vocab if w.is_lower == word.is_lower and w.prob >= -15]
      similarity = sorted(filtered_words, key=lambda w: word.similarity(w), reverse=True)
      return True,similarity[:n]
  else:
      words=[]
      max_edit_distance_lookup = 2
      suggestion_verbosity = n # TOP, CLOSEST, ALL
      suggestions = sym_spell.lookup(seed, suggestion_verbosity,
                                        max_edit_distance_lookup)
      
      for p in suggestions:
            words.append(p.term.lower())
      return False, words
  
        
#words = set(brown.words())
pathmalfilespatterns='E:/PrimaryStudy1/Code/outputs/ngrams/'   
# Import libraries
#Creating URL embeddings
from ekphrasis.classes.segmenter import Segmenter
import re
seg_eng = Segmenter(corpus="twitter",max_split_length=40) 

import pickle ,os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import time
from collections import Counter
from urllib.parse import urlparse
import tldextract
from gensim.models import Doc2Vec
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


import collections
import os
import random
import numpy as np
from tqdm import tqdm
import sys, email
import pandas as pd 
import math
from tqdm import tqdm
import string
import collections
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.model_selection import cross_val_score, cross_validate, StratifiedKFold
import lightgbm as lgb
from lightgbm import LGBMClassifier


from scipy.sparse import hstack
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 11:42:35 2019

@author: bushra
"""
import numpy as np
import matplotlib.pyplot as plt

from sklearn import ensemble
from sklearn import datasets
import pandas as pd
import csv
from numpy import average
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import cross_val_predict
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.datasets import make_classification
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import f1_score
from sklearn.metrics import coverage_error
from sklearn.svm import LinearSVC
from sklearn.metrics import average_precision_score
# Load CSV (using python)
from sklearn.metrics import precision_recall_curve
from sklearn.multiclass import OneVsRestClassifier
from lightgbm import LGBMClassifier
from sklearn.utils.fixes import signature
from sklearn.metrics import label_ranking_average_precision_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import cm

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
def lgb_f1_score(y_hat, data):
    y_true = data.get_label()
    y_hat = np.round(y_hat) # scikits f1 doesn't like probabilities
    return 'f1', f1_score(y_true, y_hat), True

from sklearn import metrics
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from numpy import sort
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
import itertools,re
from sklearn.model_selection import StratifiedKFold
# Load datasets
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn import feature_extraction

 # -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 14:28:29 2019

@author: bushra
"""
import random
import validators
from urllib.parse import urlunparse,urlparse
import re
import pandas as pd
#from rfc3986 import urlparse
from rfc3986 import is_valid_uri
import itertools
import tldextract
from gensim.models.doc2vec import TaggedDocument
import homoglyphs as hg,os
import sys 
import numpy as np
from nltk import FreqDist
from nltk.corpus import gutenberg
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import re
import random
import csv
from nltk.corpus import brown
from collections import Counter
import urllib.request
from difflib import SequenceMatcher
import homoglyphs as hg,os
from urllib.request import Request, urlopen
import time
start = time.time()
homoglyphs =hg.Homoglyphs(languages={'en'},
            strategy=hg.STRATEGY_LOAD,
            ascii_strategy=hg.STRATEGY_REMOVE
        )
import random
import validators
import re
import pandas as pd
from rfc3986 import urlparse
from rfc3986 import is_valid_uri
import itertools
import sys 
from nltk import FreqDist
from nltk.corpus import gutenberg
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import re
import random
import csv
from nltk.corpus import brown
from collections import Counter
import urllib.request
import spacy
import pandas as pd 
import codecs,re
from gensim.test.utils import datapath, get_tmpfile
#from gensim.models import KeyedVectors
#from gensim.scripts.glove2word2vec import glove2word2vec
#
#codecs.register_error("strict", codecs.ignore_errors)
#import re
#path="E:\\PrimaryStudy1\\Code\\part2\\wordembeddingDataset\\glove.840B.300d\\"
#word2vec_output_file = path+'glove.840B.300d.txt.word2vec'
##glove2word2vec(path+'glove.840B.300d.txt', word2vec_output_file)
#model = KeyedVectors.load_word2vec_format(word2vec_output_file)
#print(model.wv.most_similar(positive='secure'))
#wordsindex2word=model.wv.index2word
      
#words = set(brown.words())
def findIP(List):
        M=List[0]
        ip = re.findall( r'[0-9-]+(?:\.[0-9]+){3}', M )
        if ip !=[]:
            List.remove(M)
        else:
            ip=''
        return str(ip)

def findProtocol(List):
   
    for M in List:
        if(':' in M):
            protocol=M.split(':').pop(0)
            List.remove(M)
            return protocol
        else:
            return ''
        
def findDomains(List):
     domain=''
     sld=''
     tld=''
     port=''
     for M in List:
         if('.' in M):
             splitlist=M.split('.')
             if(M.count('.')>1):
                 tld=splitlist.pop()
                 ld=tld.split(':')
                 tld=ld.pop(0)
                 if(ld!=[]):
                     port=ld.pop()
                 sld=splitlist.pop()
                 for i,l in enumerate(splitlist):
                    domain+=l
                    if(i!=len(splitlist)-1):
                        domain+='.'
                 List.remove(M)
                 return tld,sld,domain,port
             else:
                tld=splitlist.pop()
                ld=tld.split(':')
                tld=ld.pop(0)
                if(ld!=[]):
                     port=ld.pop()
                domain=splitlist.pop()
                List.remove(M)
                return tld,sld,domain,port
     return tld,sld,domain,port
def get_words(path):
    """load stop words """
    
    with open(path, 'r', encoding="utf-8") as f:
        stopwords = f.read().splitlines()
        return stopwords    
def urltokenizer(url):
    tld=''
    sld=''
    domain=''
    port=''
    path=''
    exe=''
    protocol=''
    urlparsed = urlparse(url)
    Major=url.split('/')  #[http:,www.google.com.sg, webhp?hl=zh-CN] 
    ip= (findIP(Major))
    tem=urlparsed.scheme
    if(tem is not None):
        if('.' in tem):
            Lst=tem.split('/')
            [tld,sld,domain,port]=(findDomains(Lst))
            
        else:
            protocol=urlparsed.scheme
    
    userinfo=urlparsed.userinfo
    host=urlparsed.host
    pathurl=urlparsed.path
    if(pathurl is not None and host is None):
        Lst=pathurl.split('/')
        if('.' in pathurl):
            [tld,sld,domain,port]=(findDomains(Lst))
            [path,exe]=(findPathQueryFragment(Lst))
        else:
            path=pathurl
    elif(host is not None and pathurl is None):
        Lst=host.split('/')
        ip= (findIP(Lst))
        if('.' in host):
            [tld,sld,domain,port]=(findDomains(Lst))
            [path,exe]=(findPathQueryFragment(Lst))
        else:
            path+=host
    elif(host is not None and pathurl is not None):
        Lst=host.split('/')
        ip= (findIP(Lst))
        if('.' in host):
            [tld,sld,domain,port]=(findDomains(Lst))
        if('.' in pathurl):
            Lst1=pathurl.split('/')
            if(Lst!=[]):
                for q in Lst:
                    Lst1.append(Lst.remove(q))
            [path,exe]=(findPathQueryFragment(Lst1))
        if('.' not in host):
            path+=host
        if('.' not in pathurl):
            path+=pathurl
        
            
    
    parameter=urlparsed.query
    fragment=urlparsed.fragment
    [sld,domain,tld]=tldextract.extract(url)
    if(host==None):
           host=sld+'.'+domain+'.'+tld
    if(protocol=='' and protocol==None):
        protocol='http'
    if(path=='/'):
        path=''
    if(domain!=''):
        domain=''.join(converter(tokenparts(domain.split('.'))))
    if(domain=='' and ip!=''):
        domain=ip
    if(path!=None and path!='' and path!='/'):
    
        path='/'.join(converter(tokenparts(re.findall(r"[\w']+", path))))
    else:
        path=''
    if(sld!='None' and sld!='' and sld!=None):
    
        sld='.'.join(converter(tokenparts(sld.split('.'))))
    else:
        sld='www'
    if(parameter!=None):
        parameter=''.join(converter(tokenparts(re.findall(r"[\w']+", parameter))))
    else:
        parameter=''
    if(fragment!=None):
        fragment=''.join(converter(tokenparts(re.findall(r"[\w']+", fragment))))
    else:
        fragment=''
    if(exe==''):
        exe=''
    return [ip,port,protocol,tld,(sld),userinfo,host,domain,(path),(parameter),(fragment),exe]
          
       
                
def findPathQueryFragment(List):
    path=''
    query=''
    fragment=''
    exe=''
    remaining=''
    last='p'
    previousch=''
   #eas?camp=1932-1;cre=mu&grpid=1738&tag_id=618&nums=FGApbjFAAA

    for i,M in enumerate(List):
        remaining=''
        if('.' in M and i==len(List)-1):  
            newList=M.split('.') 
            exe+=newList.pop()
            for s,q in enumerate(newList):
                if(q is not None):
                    path+='/'+newList.pop(s)
           
                  
        else:
            if(i<len(List)-1):
                path+=M+'/'
            else:
                path+=M
        
                
      
           
         
    return path,exe   
def convert(ls,op): 
      
    # Converting integer list to string list 
    s = [str(i) for i in ls] 
      
    # Join list items using join() 
    res = ("op".join(s)) 
      
    return(res) 

# Extract NLP features

# config 1: Bag-of-word without tf-idf
# config 2: Bag-of-word with tf-idf
# config 3: N-gram without tf-idf
# config 4: N-gram with tf-idf
_digits = re.compile('\d')
def contains_digits(d):
     return bool(_digits.search(d))
def tokenparts(words):
    finalwords=[]
    for w in words:
       w=re.sub('\W+',' ',w)
       w=re.sub('\d+',' ',w)
       w=re.sub('_',' ',w)
       w=re.sub(';',' ',w)
       
       if(w!='' and w!='None' ):
#           words.append(w.split('_'))
#           words.remove(w)
        
           subwords=(seg_eng.segment(w))
           
           finalwords.append(subwords.split(' '))
    return (finalwords)


def converter(listoflist):
    lst=[]
   
    for ls in listoflist:
        
        if(type(ls) is list):
            for e in ls:
                if(e !=''):
                    lst.append((e))
        else:
           lst.append(ls)
    return lst  

def get_keywords(path):
    """load stop words """
    
    with open(path, 'r', encoding="utf-8") as f:
        stopwords = f.read().splitlines()
      
        return stopwords
nlp = spacy.load('en_core_web_lg')
import requests

# This is needed to limit the frequeny 
# by which we are going to hit the API 
# endpoints. Only certain number of 
# requests can be made in a mintue
import time

# This is needed to convert API 
# responses into JSON objects
import json

# Godaddy developer key and secret
api_key = "insert your API Key"
secret_key = "insert your secret Key"

# API key and secret are sent in the header
headers = {"Authorization" : "sso-key {}:{}".format(api_key, secret_key)}

# Domain availability and appraisal end points
url = "https://api.godaddy.com/v1/domains/available"
appraisal = "https://api.godaddy.com/v1/appraisal/{}"

# If a domain name is available 
# decide whether to appraise or not
do_appraise = True

# Number of domains to check in each call. 
# For example, we can not check more than 500 
# domain names in one call so we need to split 
# the list of domain names into chunks
chunk_size = 500

# Filter domain names by length
max_length = 30

# Filter domain names by price range
min_price = 0
max_price = 5000

# If appraisal is enabled, only include 
# domain names with min appraisal price
min_appr_price = 0

# When a domain is appraised, Godaddy API 
# returns similar domains sold. This is a 
# nice feature to take a look at sold domains. 
# To filter similar sold domains we can do that 
# by setting the min sale price and the min 
# year the domain was sold
min_sale_price = 0
min_sale_year = 2000

# Domain name structure: 
# prefix + keyword + suffix + extension
# You can manually insert few values into
# these lists and start the search or read
# from files as demonstrated below


# This list holds all generated domains
# It is the list we are going to check
#all_domains = getdomain()
# This list holds similar domains sold
# This is retrieved from Godaddy appraisal API
similar_domains = []
# This holds available domains found that match
# the search criteria
found_domains = {}


# Generate domains

# This function splits all domains into chunks
# of a given size
def chunks(array, size):
   for i in range(0, len(array), size):
      yield array[i:i + size]
# Split the original array into subarrays
#Split the original array into subarrays


def check_availability(curl):
   [sld,domain,tld]=tldextract.extract(curl)
   dm= "{}{}{}.{}".format('', domain,'', tld)
   if(dm in available):
       return True, available[dm]
   else:
       domain_chunks = list(chunks(dm, chunk_size))
       
       for domains in domain_chunks:
            availability_res=requests.post(url, json=domain_chunks, headers=headers)
            #print(availability_res.text)
            try:
                for domain in json.loads(availability_res.text)["domains"]:
                       
                      if domain["available"]:
                         price = float(domain["price"])/1000000
                         found_domains[domain["domain"]]=price
                         print('avai',domain) 
                         available[dm]=price
                         return True, price
                      elif("errors" in json.loads(availability_res.text) ):
                              available[dm]=price
                              return True,0
                      else:
                           return False,-1
            except:
                 return False,-1
def concatenationAdversary(seed): # concatenate the malicious pattern with legal pattern 
            concatoptionsub=[]
            concatoptiondn=[]
            #print(seed)
            [Flag, Words]=get_related(seed,20)
            if(Flag==True):
                for i,similar in enumerate(Words):
                     if(i>5): 
                       relatedword=similar.lower_
                       if(relatedword!=seed.lower()):
                           concatoptionsub.append(seed+'.'+relatedword)  #concatenate real domain as subdomain
                           concatoptiondn.append(relatedword+'-'+seed)  #concatenate real domain in domain name
                           concatoptiondn.append(seed+'-'+relatedword) #concatenate real domain as path element
            else:
               for i,similar in enumerate(Words):
                   
                       relatedword=similar.lower()
                       if(relatedword!=seed.lower()):
                           concatoptionsub.append(seed+'.'+relatedword)  #concatenate real domain as subdomain
                           concatoptiondn.append(relatedword+'-'+seed)  #concatenate real domain in domain name
                      
                 
            
            return concatoptiondn,concatoptionsub 
def RepAdversary(seed,id): # concatenate the malicious pattern with legal pattern 
            concatoptions=[]
            if(id=='d'):
                
                concatoptions.append(seed+'-'+seed)
            else:
                concatoptions.append(seed+'/'+seed)
                concatoptions.append(seed+'_'+seed)
                
            return concatoptions
def swapAdversary(part,id): # concatenate the malicious pattern with legal pattern
            swap=[]
            if(len(part.split('-'))>1):
                 parts=part.split('-')
                 for i in range(len(parts)-2):
                       if(id=='d'):
                           p=parts[i+1]+parts[i]
                       else:
                           p=parts[i+1]+'/'+parts[i]
                       for j in range(i+2,len(parts)-1):
                            if(id=='d'):
                                p+=parts[j]
                            else:
                                p=p+'/'+parts[j]
                       swap.append(p)
            return swap
       
            
def get_new_domains(seed):
        concatoptions=[]
        print(seed)
        [Flag, seeds]=get_related(seed,15)
        if(Flag==True):
            for i,similar in enumerate(seeds):
                   if(i!=len(seeds)-1 and i>4):
                       relatedword=str(similar.lower_)+'-'+str(seeds[i+1].lower_)
                       if(relatedword!=seed.lower()):
                           concatoptions.append(relatedword)
                       relatedword=seeds[i+1].lower_+'-'+similar.lower_
                       if(relatedword!=seed.lower()):
                           concatoptions.append(relatedword)
        else:
            for i,similar in enumerate(seeds):
                if(i!=len(seeds)-1):
                       #relatedword=str(similar.lower())+'-'+str(seeds[i+1].lower())
                       relatedword=(similar.lower())+str(seed.lower())
                       if(relatedword!=seed.lower()):
                           concatoptions.append(relatedword)
                       relatedword=seed+(similar.lower())
                       if(relatedword!=seed.lower()):
                           concatoptions.append(relatedword)
            
        return concatoptions       

#def SubstituteAdversary(part,seed): # concatenate the malicious pattern with legal pattern 
#       
#            #tld
#
##def substitutionAdversary(): # replace the legal pattern with malicious pattern
##    
##
def TldManipulation():
    path='E:/PrimaryStudy1/ngrams/tld_.txt' 
    return get_keywords(path)
def extensionManipulation():
    path='E:/PrimaryStudy1/ngrams/exe_.txt' 
    return get_keywords(path)
def protocolManipulation():
    path='E:/PrimaryStudy1/ngrams/protocol_.txt' 
    return get_keywords(path)
    
def generatehom(domain):
        array = np.zeros((100,len(domain)),'U1')  
        for i,each in enumerate(domain):
           hum=hg.Homoglyphs()
           listofoptions=(hum.get_combinations(each))
        
           for j,l in enumerate(listofoptions):
               
                        array[j][i]=l
                        if(i>0and j>0):
                            for k in range(i-1,-1,-1):
                                if(array[j][k]==''):
                                        ind=random.randint(0,j-1)
                                        array[j][k]=array[ind][k]
                                    
                    
           l=len(listofoptions)   
           if(j>1):
               while(array[l][i-1]!=''):
                      ind=random.randint(1,j-1)
                      array[l][i]=array[ind][i]
                      l=l+1
        domains=[]    
        
        
#       
        for row in array:
               newdm=''.join(row)
#               
#               if(newdm!='' and newdm!=domain):
#                  domains.append(newdm)
               #print(newdm)
               newdm=homoglyphs.to_ascii(newdm)
               
               for dm in newdm:
                   if(dm!='' and dm!=domain and dm not in domains):
                       domains.append(dm)
                     
        if(domain in domains):
            domains.remove(domain)
        return domains    
from urllib.request import Request, urlopen   
import requests  
def validate_domain(domain):
		try:
			domain_idna = domain.encode('idna').decode()
		except UnicodeError:
			# '.tla'.encode('idna') raises UnicodeError: label empty or too long
			# This can be obtained when __omission takes a one-letter domain.
			return False
		if len(domain) == len(domain_idna) and domain != domain_idna:
			return False
		allowed = re.compile('(?=^.{4,253}$)(^((?!-)[a-zA-Z0-9-]{1,63}(?<!-)\.)+[a-zA-Z]{2,63}\.?$)', re.IGNORECASE)
		return allowed.match(domain_idna)       
def reconstruct(List): #protocol,ip,userinfo,domain,sld,tld,port,path,parameter,exe,fragment
    [protocol,ip,userinfo,domain,sld,tld,port,path,parameter,exe,fragment]=List
    CraftedUrl=''
    if(exe!=''):
        p=list(path)
        if(p[len(path)-1]=='/'):
            p[len(path)-1]='.'
            path=''.join(p)
            path=path+'.'+exe
        else:
            path+='.'+exe
    #hostname=protocol+'://'+'www.'+domain+'.'+tld
    CraftedUrl=urlunparse((protocol, (sld+'.'+domain+'.'+tld), path,'',parameter, fragment))
        
    return CraftedUrl

def removeduplicates(crafted):
  return list(dict.fromkeys(crafted))
import textdistance
def write(url,curl,d,method,writer):
           
            if(curl not in craftedurl and (is_valid_uri(curl))==True and validators.url(curl)==True and curl!='' and curl not in alreadyregister):
                available,price=check_availability(curl)   
                if(available):
                       
                       [ip,port,protocol,tld,(sld),userinfo,host,domain,(path),(parameter),(fragment),exe]=urltokenizer(url)
                       if(textdistance.levenshtein(url,curl)<int(len(url)*0.20) or domain in curl):
                           craftedurl.append(curl)
                           
                           print('This URL is available',curl,'@ price',price)
                           writer.writerow([url,curl,d,method,str(price)])   
                       else:
                            alreadyregister.append(curl)
                else:
                    alreadyregister.append(curl)
            else:
                    alreadyregister.append(curl)
qwerty = {
		'1': '2q', '2': '3wq1', '3': '4ew2', '4': '5re3', '5': '6tr4', '6': '7yt5', '7': '8uy6', '8': '9iu7', '9': '0oi8', '0': 'po9',
		'q': '12wa', 'w': '3esaq2', 'e': '4rdsw3', 'r': '5tfde4', 't': '6ygfr5', 'y': '7uhgt6', 'u': '8ijhy7', 'i': '9okju8', 'o': '0plki9', 'p': 'lo0',
		'a': 'qwsz', 's': 'edxzaw', 'd': 'rfcxse', 'f': 'tgvcdr', 'g': 'yhbvft', 'h': 'ujnbgy', 'j': 'ikmnhu', 'k': 'olmji', 'l': 'kop',
		'z': 'asx', 'x': 'zsdc', 'c': 'xdfv', 'v': 'cfgb', 'b': 'vghn', 'n': 'bhjm', 'm': 'njk'
		}
qwertz = {
		'1': '2q', '2': '3wq1', '3': '4ew2', '4': '5re3', '5': '6tr4', '6': '7zt5', '7': '8uz6', '8': '9iu7', '9': '0oi8', '0': 'po9',
		'q': '12wa', 'w': '3esaq2', 'e': '4rdsw3', 'r': '5tfde4', 't': '6zgfr5', 'z': '7uhgt6', 'u': '8ijhz7', 'i': '9okju8', 'o': '0plki9', 'p': 'lo0',
		'a': 'qwsy', 's': 'edxyaw', 'd': 'rfcxse', 'f': 'tgvcdr', 'g': 'zhbvft', 'h': 'ujnbgz', 'j': 'ikmnhu', 'k': 'olmji', 'l': 'kop',
		'y': 'asx', 'x': 'ysdc', 'c': 'xdfv', 'v': 'cfgb', 'b': 'vghn', 'n': 'bhjm', 'm': 'njk'
		}
azerty = {
		'1': '2a', '2': '3za1', '3': '4ez2', '4': '5re3', '5': '6tr4', '6': '7yt5', '7': '8uy6', '8': '9iu7', '9': '0oi8', '0': 'po9',
		'a': '2zq1', 'z': '3esqa2', 'e': '4rdsz3', 'r': '5tfde4', 't': '6ygfr5', 'y': '7uhgt6', 'u': '8ijhy7', 'i': '9okju8', 'o': '0plki9', 'p': 'lo0m',
		'q': 'zswa', 's': 'edxwqz', 'd': 'rfcxse', 'f': 'tgvcdr', 'g': 'yhbvft', 'h': 'ujnbgy', 'j': 'iknhu', 'k': 'olji', 'l': 'kopm', 'm': 'lp',
		'w': 'sxq', 'x': 'wsdc', 'c': 'xdfv', 'v': 'cfgb', 'b': 'vghn', 'n': 'bhj'
		}
keyboards = [ qwerty, qwertz, azerty ]
def bitsquatting(domain):
		result = []
		masks = [1, 2, 4, 8, 16, 32, 64, 128]
		for i in range(0, len(domain)):
			c = domain[i]
			for j in range(0, len(masks)):
				b = chr(ord(c) ^ masks[j])
				o = ord(b)
				if (o >= 48 and o <= 57) or (o >= 97 and o <= 122) or o == 45:
					result.append(domain[:i] + b + domain[i+1:])

		return result
def hyphenation(domain):
		result = []

		for i in range(1, len(domain)):
			result.append(domain[:i] + '-' + domain[i:])

		return result
def insertion(domain):
		result = []

		for i in range(1, len(domain)-1):
			for keys in keyboards:
				if domain[i] in keys:
					for c in keys[domain[i]]:
						result.append(domain[:i] + c + domain[i] + domain[i+1:])
						result.append(domain[:i] + domain[i] + c + domain[i+1:])

		return list(set(result))
def omission(domain):
		result = []

		for i in range(0, len(domain)):
			result.append(domain[:i] + domain[i+1:])

		n = re.sub(r'(.)\1+', r'\1', domain)

		if n not in result and n != domain:
			result.append(n)

		return list(set(result))
def repetition(domain):
		result = []
        
		for i in range(0, len(domain)):
			if domain[i].isalpha():
				result.append(domain[:i] + domain[i] + domain[i] + domain[i+1:])

		return list(set(result))
def subdomain(domain):
		result = []

		for i in range(1, len(domain)):
			if domain[i] not in ['-', '.'] and domain[i-1] not in ['-', '.']:
				result.append(domain[:i] + '.' + domain[i:])

		return result

def vowel_swap(domain):
		vowels = 'aeiou'
		result = []

		for i in range(0, len(domain)):
			for vowel in vowels:
				if domain[i] in vowels:
					result.append(domain[:i] + vowel + domain[i+1:])

		return (result)

def addition(domain):
		result = []

		for i in range(97, 123):
			result.append(domain + chr(i))

		return result
def transposition(domain):
    result = []
    for i in range(0, len(domain)-1):
        if domain[i+1] != domain[i]:
            result.append(domain[:i] + domain[i+1] + domain[i] + domain[i+2:])
    return result    
def vowel_swap_cons(domain):
		vowels = 'bcdfghjklmnpqrstvwxyz'
		result = []

		for i in range(0, len(domain)):
			for vowel in vowels:
				if domain[i] in vowels:
					result.append(domain[:i] + vowel + domain[i+1:])

		return (result)
def char_Adversary(part):
    Char_Adversaries=dict()
    Char_Adversaries['homoglyphs']=generatehom(part)
    Char_Adversaries['transpose']=transposition(part)
    Char_Adversaries['add']=addition(part)
    Char_Adversaries['swap']=vowel_swap(part)
    Char_Adversaries['sub']=subdomain(part)
    Char_Adversaries['repi']=repetition(part)
    Char_Adversaries['om']=omission(part)
    Char_Adversaries['insert']=insertion(part)
    Char_Adversaries['hyp']=hyphenation(part)
    Char_Adversaries['bitsq']=bitsquatting(part)
    return Char_Adversaries
def word_Adversary(part,id):    
    Word_Adversaries=dict()
    Word_Adversaries['wconcatdn'],Word_Adversaries['wconcatsub']=concatenationAdversary(part)
    Word_Adversaries['wrepi']=RepAdversary(part,id)
    Word_Adversaries['wswap']=swapAdversary(part,id)
    return Word_Adversaries
def domainAdversary(seedurl,flag):
   
    [ip,port,protocol,tld,(sld),userinfo,host,domain,(path),(parameter),(fragment),exe]=urltokenizer(seedurl)
    domainAdversaries=dict()
    domainAdversaries['Char_Adversaries']=char_Adversary(domain)
    domainAdversaries['Word_Adversaries']=word_Adversary(domain,'d')
    if(flag==0):
        filename='F:/Evasion Attack/Dataset/Adversaries/NewDomainAdversaryUpdate.csv'
        dmad=open(filename, 'w' ,encoding='utf-8',newline='')
        DM= csv.writer(dmad)
        DM.writerow(['seedurl','craftedurl','adversarytype','adversarymethod','price'])
    else:
        filename='F:/Evasion Attack/Dataset/Adversaries/NewDomainAdversaryUpdate.csv'
        dmad=open(filename, 'a' ,encoding='utf-8',newline='')
        DM= csv.writer(dmad)
        #DM.writerow(['seedurl','craftedurl','adversarytype','adversarymethod'])

        
    for d in domainAdversaries.keys():
         dcharadversaries=domainAdversaries[d]
         for method in dcharadversaries.keys():
             adversarialexamples=dcharadversaries[method]
             for a in adversarialexamples:
                
                 craftedurl=(reconstruct([protocol,ip,userinfo,a,sld,tld,port,path,parameter,exe,fragment]))
                 write(seedurl,craftedurl,d,method,DM)
    dmad.close()
def concatenationPath(domain,path): # concatenate the malicious pattern with legal pattern 
            concatoptionsA=[]
            concatoptionB=[]
            if(path==''):
                concatoptionsA.append('/'+domain)
            else:
                for p in path.split('/'):
                    newpath=re.sub(p,domain+'/'+p,path)
                    if(newpath!=''):
                        concatoptionsA.append(newpath)
                    for e in extensionManipulation():
                        concatoptionB.append(newpath+'.'+e)
                
            return concatoptionsA,concatoptionB   
         
def PathAdversary(seedurl,flag):
    [ip,port,protocol,tld,(sld),userinfo,host,domain,(path),(parameter),(fragment),exe]=urltokenizer(seedurl)
    method='Path_Adversary'
    if(flag==0):
        filename='F:/Evasion Attack/Dataset/Adversaries/NewPathAdversaryUpdate.csv'
        dmad=open(filename, 'w' ,encoding='utf-8',newline='')
        DM= csv.writer(dmad)
        DM.writerow(['seedurl','craftedurl','adversarytype','adversarymethod','price'])
    else:
        filename='F:/Evasion Attack/Dataset/Adversaries/NewPathAdversaryUpdate.csv'
        dmad=open(filename, 'a' ,encoding='utf-8',newline='')
        DM= csv.writer(dmad)
        #DM.writerow(['seedurl','craftedurl','adversarytype','adversarymethod'])
          
    for d in get_new_domains(domain):
        
        A,B=concatenationPath(domain,path)
        
        for c in A :
                 craftedurl=(reconstruct([protocol,ip,userinfo,d,sld,tld,port,c,parameter,exe,fragment]))
                 write(seedurl,craftedurl,'path_dm',method,DM)
        for b in B:
                 craftedurl=(reconstruct([protocol,ip,userinfo,d,sld,tld,port,b,parameter,exe,fragment]))
                 write(seedurl,craftedurl,'path_exe',method,DM)
            
    dmad.close()
def TLDAdversary(seedurl,flag):
    if(flag==0):
        filename='F:/Evasion Attack/Dataset/Adversaries/NewTLDAdversaryUpdate.csv'
        tldmad=open(filename, 'w' ,encoding='utf-8',newline='')
        TLD= csv.writer(tldmad)
        TLD.writerow(['seedurl','craftedurl','adversarytype','adversarymethod,price'])
    else:
        filename='F:/Evasion Attack/Dataset/Adversaries/NewTLDAdversaryUpdate.csv'
        tldmad=open(filename, 'a' ,encoding='utf-8',newline='')
        TLD= csv.writer(tldmad)
        #TLD.writerow(['seedurl','craftedurl','adversarytype','adversarymethod'])
    [ip,port,protocol,tld,(sld),userinfo,host,domain,(path),(parameter),(fragment),exe]=urltokenizer(seedurl)
    for ctld in TldManipulation():
        method='TLD_Adversary' 
        curl=reconstruct([protocol,ip,userinfo,domain,sld,ctld,port,path,parameter,exe,fragment])
        write(seedurl,curl,method,'tld_mal',TLD)  
def generateAdversarialExamples(seedurl,flag):
      domainAdversary(seedurl,flag)
      PathAdversary(seedurl,flag)
      TLDAdversary(seedurl,flag)
      
craftedurl=[]
alreadyregister=[]
available=dict()
seedDataset='F:/Evasion Attack/Dataset/Legitimate/crawledurlslegal.csv'
Dataset = pd.read_csv(seedDataset,encoding="utf-8") 
flag=0
#from sklearn.utils import shuffle
#Dataset = shuffle(Dataset)
seedurls=list(Dataset['url'])
print(seedurls[0])
hostname=[]
#pathemd='E:/Primary Studies/Deep Learning Based Evasion/crawl-300d-2M-subword/crawl-300d-2M-subword.vec'
#model = KeyedVectors.load_word2vec_format(pathemd)
total_len=len(seedurls)
for i in range(0,total_len,4):
    s=seedurls[i]
    print('processing', i)
    if(i==0):
        generateAdversarialExamples(s,0)
    else:     
        generateAdversarialExamples(s,1)