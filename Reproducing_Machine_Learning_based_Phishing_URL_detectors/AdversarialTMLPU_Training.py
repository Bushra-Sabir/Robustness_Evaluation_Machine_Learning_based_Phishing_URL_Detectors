# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 11:15:27 2020

@author: bushra
"""

# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
Created on January 2021

@author: bushra Sabir
"""

#SET DATAPATHs and FEATUREPATHs
OriginalDatapath="path of original dataset"
Datapath="Path in which adversarial dataset is stored"
Featurepath="Path to store Adversarial training features "

# Import libraries
from sklearn.model_selection import cross_val_score, cross_validate, StratifiedKFold
from sklearn import metrics
import validators
from rfc3986 import is_valid_uri
import re,tldextract
import pandas as pd
from rfc3986 import urlparse
from nltk import FreqDist
from nltk.corpus import gutenberg
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import csv
from nltk.corpus import brown
from collections import Counter
import urllib.request
from difflib import SequenceMatcher
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn import ensemble
from sklearn import datasets
from numpy import average
from sklearn.model_selection import train_test_split
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
from sklearn.metrics import precision_recall_curve
from sklearn.multiclass import OneVsRestClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import label_ranking_average_precision_score
import matplotlib as mpl
from matplotlib import cm
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from Feature_extraction import feature_extract
from lib.functions import *
import itertools
import os
import time
from scipy.sparse import hstack
start = time.time()
import codecs
codecs.register_error("strict", codecs.ignore_errors)
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score,balanced_accuracy_score
from sklearn import feature_extraction
from ekphrasis.classes.segmenter import Segmenter
from sklearn.preprocessing import OrdinalEncoder
from hyperopt import hp, tpe, fmin, Trials, STATUS_OK,early_stop
from multiprocessing import  Pool
import multiprocessing


ord_enc = OrdinalEncoder()

All_Known_TLD = ['com', 'at', 'uk', 'pl', 'be', 'biz', 'co', 'jp', 'co_jp', 'cz', 'de', 'eu', 'fr', 'info', 'it', 'ru', 'lv', 'me', 'name', 'net', 'nz', 'org', 'us']

# List of Suspicious Words Present in URL
Suspicious_Words=['secure','account','update','banking','login','click','confirm','password','verify','signin','ebayisapi','lucky','bonus','congratulation']

# List of Suspicious Top Level Domains in URLs
Suspicious_TLD=['zip','cricket','link','work','party','gq','kim','country','science','tk','exe','bin']


def attributes():
    """Output file attributes."""
    lexical = [
        'qtd_ponto_url', 'qtd_hifen_url', 'qtd_underline_url',
        'qtd_barra_url', 'qtd_interrogacao_url', 'qtd_igual_url',
        'qtd_arroba_url', 'qtd_comercial_url', 'qtd_exclamacao_url',
        'qtd_espaco_url', 'qtd_til_url', 'qtd_virgula_url',
        'qtd_mais_url', 'qtd_asterisco_url', 'qtd_hashtag_url',
        'qtd_cifrao_url', 'qtd_porcento_url', 'qtd_tld_url',
        'comprimento_url', 'qtd_ponto_dominio', 'qtd_hifen_dominio',
        'qtd_underline_dominio', 'qtd_barra_dominio', 'qtd_interrogacao_dominio',
        'qtd_igual_dominio', 'qtd_arroba_dominio', 'qtd_comercial_dominio',
        'qtd_exclamacao_dominio', 'qtd_espaco_dominio', 'qtd_til_dominio',
        'qtd_virgula_dominio', 'qtd_mais_dominio', 'qtd_asterisco_dominio',
        'qtd_hashtag_dominio', 'qtd_cifrao_dominio', 'qtd_porcento_dominio',
        'qtd_vogais_dominio', 'comprimento_dominio', 'formato_ip_dominio',
        'server_client_dominio', 'qtd_ponto_diretorio', 'qtd_hifen_diretorio',
        'qtd_underline_diretorio', 'qtd_barra_diretorio', 'qtd_interrogacao_diretorio',
        'qtd_igual_diretorio', 'qtd_arroba_diretorio', 'qtd_comercial_diretorio',
        'qtd_exclamacao_diretorio', 'qtd_espaco_diretorio', 'qtd_til_diretorio',
        'qtd_virgula_diretorio', 'qtd_mais_diretorio', 'qtd_asterisco_diretorio',
        'qtd_hashtag_diretorio', 'qtd_cifrao_diretorio', 'qtd_porcento_diretorio',
        'comprimento_diretorio', 'qtd_ponto_arquivo', 'qtd_hifen_arquivo',
        'qtd_underline_arquivo', 'qtd_barra_arquivo', 'qtd_interrogacao_arquivo',
        'qtd_igual_arquivo', 'qtd_arroba_arquivo', 'qtd_comercial_arquivo',
        'qtd_exclamacao_arquivo', 'qtd_espaco_arquivo', 'qtd_til_arquivo',
        'qtd_virgula_arquivo', 'qtd_mais_arquivo', 'qtd_asterisco_arquivo',
        'qtd_hashtag_arquivo', 'qtd_cifrao_arquivo', 'qtd_porcento_arquivo',
        'comprimento_arquivo', 'qtd_ponto_parametros', 'qtd_hifen_parametros',
        'qtd_underline_parametros', 'qtd_barra_parametros', 'qtd_interrogacao_parametros',
        'qtd_igual_parametros', 'qtd_arroba_parametros', 'qtd_comercial_parametros',
        'qtd_exclamacao_parametros', 'qtd_espaco_parametros', 'qtd_til_parametros',
        'qtd_virgula_parametros', 'qtd_mais_parametros', 'qtd_asterisco_parametros',
        'qtd_hashtag_parametros', 'qtd_cifrao_parametros', 'qtd_porcento_parametros',
        'comprimento_parametros', 'presenca_tld_argumentos', 'qtd_parametros',
        'email_na_url', 'extensao_arquivo'
    ]

#    blacklist = ['url_presente_em_blacklists', 'presenca_ip_blacklists', 'dominio_presente_em_blacklists']
#
#    host = ['dominio_presente_em_rbl', 'tempo_resposta', 'possui_spf', 'localizacao_geografica_ip',
#            'numero_as_ip', 'ptr_ip', 'tempo_ativacao_dominio', 'tempo_expiracao_dominio',
#            'qtd_ip_resolvido', 'qtd_nameservers', 'qtd_servidores_mx', 'valor_ttl_associado']
#
#    others = ['certificado_tls_ssl', 'qtd_redirecionamentos', 'url_indexada_no_google', 'dominio_indexado_no_google', 'url_encurtada']

    list_attributes = []
    list_attributes.extend(lexical)

#    list_attributes.extend(['phishing'])

    return list_attributes

def externallexicalattributes():
    """Output file attributes."""
    lexical = [
        'qtd_ponto_url', 'qtd_hifen_url', 'qtd_underline_url',
        'qtd_barra_url', 'qtd_interrogacao_url', 'qtd_igual_url',
        'qtd_arroba_url', 'qtd_comercial_url', 'qtd_exclamacao_url',
        'qtd_espaco_url', 'qtd_til_url', 'qtd_virgula_url',
        'qtd_mais_url', 'qtd_asterisco_url', 'qtd_hashtag_url',
        'qtd_cifrao_url', 'qtd_porcento_url', 'qtd_tld_url',
        'comprimento_url', 'qtd_ponto_dominio', 'qtd_hifen_dominio',
        'qtd_underline_dominio', 'qtd_barra_dominio', 'qtd_interrogacao_dominio',
        'qtd_igual_dominio', 'qtd_arroba_dominio', 'qtd_comercial_dominio',
        'qtd_exclamacao_dominio', 'qtd_espaco_dominio', 'qtd_til_dominio',
        'qtd_virgula_dominio', 'qtd_mais_dominio', 'qtd_asterisco_dominio',
        'qtd_hashtag_dominio', 'qtd_cifrao_dominio', 'qtd_porcento_dominio',
        'qtd_vogais_dominio', 'comprimento_dominio', 'formato_ip_dominio',
        'server_client_dominio', 'qtd_ponto_diretorio', 'qtd_hifen_diretorio',
        'qtd_underline_diretorio', 'qtd_barra_diretorio', 'qtd_interrogacao_diretorio',
        'qtd_igual_diretorio', 'qtd_arroba_diretorio', 'qtd_comercial_diretorio',
        'qtd_exclamacao_diretorio', 'qtd_espaco_diretorio', 'qtd_til_diretorio',
        'qtd_virgula_diretorio', 'qtd_mais_diretorio', 'qtd_asterisco_diretorio',
        'qtd_hashtag_diretorio', 'qtd_cifrao_diretorio', 'qtd_porcento_diretorio',
        'comprimento_diretorio', 'qtd_ponto_arquivo', 'qtd_hifen_arquivo',
        'qtd_underline_arquivo', 'qtd_barra_arquivo', 'qtd_interrogacao_arquivo',
        'qtd_igual_arquivo', 'qtd_arroba_arquivo', 'qtd_comercial_arquivo',
        'qtd_exclamacao_arquivo', 'qtd_espaco_arquivo', 'qtd_til_arquivo',
        'qtd_virgula_arquivo', 'qtd_mais_arquivo', 'qtd_asterisco_arquivo',
        'qtd_hashtag_arquivo', 'qtd_cifrao_arquivo', 'qtd_porcento_arquivo',
        'comprimento_arquivo', 'qtd_ponto_parametros', 'qtd_hifen_parametros',
        'qtd_underline_parametros', 'qtd_barra_parametros', 'qtd_interrogacao_parametros',
        'qtd_igual_parametros', 'qtd_arroba_parametros', 'qtd_comercial_parametros',
        'qtd_exclamacao_parametros', 'qtd_espaco_parametros', 'qtd_til_parametros',
        'qtd_virgula_parametros', 'qtd_mais_parametros', 'qtd_asterisco_parametros',
        'qtd_hashtag_parametros', 'qtd_cifrao_parametros', 'qtd_porcento_parametros',
        'comprimento_parametros', 'presenca_tld_argumentos', 'qtd_parametros',
        'email_na_url', 'extensao_arquivo'
    ]

    #blacklist = ['url_presente_em_blacklists', 'presenca_ip_blacklists', 'dominio_presente_em_blacklists']

    host = ['rbl', 'time_domain', 'spf', 'country', 'asn', 'activation_time',
                     'expiration_time', 'count_ip', 'count_ns', 'count_mx','ttl']

    others = ['certificado_tls_ssl', 'qtd_redirecionamentos', 'url_indexada_no_google', 'dominio_indexado_no_google', 'url_encurtada']

    list_attributes = []
    list_attributes.extend(lexical)
    #list_attributes.extend(blacklist)
    list_attributes.extend(host)
    list_attributes.extend(others)
    list_attributes.extend(['phishing'])

    return list_attributes
def extractFeatures(url):
                      
            dict_url = start_url(url)

            """LEXICAL"""
            # URL
            dot_url = str(count(dict_url['url'], '.'))
            hyphe_url = str(count(dict_url['url'], '-'))
            underline_url = str(count(dict_url['url'], '_'))
            bar_url = str(count(dict_url['url'], '/'))
            question_url = str(count(dict_url['url'], '100'))
            equal_url = str(count(dict_url['url'], '='))
            arroba_url = str(count(dict_url['url'], '@'))
            ampersand_url = str(count(dict_url['url'], '&'))
            exclamation_url = str(count(dict_url['url'], '!'))
            blank_url = str(count(dict_url['url'], ' '))
            til_url = str(count(dict_url['url'], '~'))
            comma_url = str(count(dict_url['url'], ','))
            plus_url = str(count(dict_url['url'], '+'))
            asterisk_url = str(count(dict_url['url'], '*'))
            hashtag_url = str(count(dict_url['url'], '#'))
            money_sign_url = str(count(dict_url['url'], '$'))
            percentage_url = str(count(dict_url['url'], '%'))
            len_url = str(length(dict_url['url']))
            email_exist = str(valid_email(dict_url['url']))
            count_tld_url = str(count_tld(dict_url['url']))
            # DOMAIN
            dot_host = str(count(dict_url['host'], '.'))
            hyphe_host = str(count(dict_url['host'], '-'))
            underline_host = str(count(dict_url['host'], '_'))
            bar_host = str(count(dict_url['host'], '/'))
            question_host = str(count(dict_url['host'], '100'))
            equal_host = str(count(dict_url['host'], '='))
            arroba_host = str(count(dict_url['host'], '@'))
            ampersand_host = str(count(dict_url['host'], '&'))
            exclamation_host = str(count(dict_url['host'], '!'))
            blank_host = str(count(dict_url['host'], ' '))
            til_host = str(count(dict_url['host'], '~'))
            comma_host = str(count(dict_url['host'], ','))
            plus_host = str(count(dict_url['host'], '+'))
            asterisk_host = str(count(dict_url['host'], '*'))
            hashtag_host = str(count(dict_url['host'], '#'))
            money_sign_host = str(count(dict_url['host'], '$'))
            percentage_host = str(count(dict_url['host'], '%'))
            vowels_host = str(count_vowels(dict_url['host']))
            len_host = str(length(dict_url['host']))
            ip_exist = str(valid_ip(dict_url['host']))
            server_client = str(check_word_server_client(dict_url['host']))
            # DIRECTORY
            if dict_url['path']:
                dot_path = str(count(dict_url['path'], '.'))
                hyphe_path = str(count(dict_url['path'], '-'))
                underline_path = str(count(dict_url['path'], '_'))
                bar_path = str(count(dict_url['path'], '/'))
                question_path = str(count(dict_url['path'], '100'))
                equal_path = str(count(dict_url['path'], '='))
                arroba_path = str(count(dict_url['path'], '@'))
                ampersand_path = str(count(dict_url['path'], '&'))
                exclamation_path = str(count(dict_url['path'], '!'))
                blank_path = str(count(dict_url['path'], ' '))
                til_path = str(count(dict_url['path'], '~'))
                comma_path = str(count(dict_url['path'], ','))
                plus_path = str(count(dict_url['path'], '+'))
                asterisk_path = str(count(dict_url['path'], '*'))
                hashtag_path = str(count(dict_url['path'], '#'))
                money_sign_path = str(count(dict_url['path'], '$'))
                percentage_path = str(count(dict_url['path'], '%'))
                len_path = str(length(dict_url['path']))
            else:
                dot_path = '100'
                hyphe_path = '100'
                underline_path = '100'
                bar_path = '100'
                question_path = '100'
                equal_path = '100'
                arroba_path = '100'
                ampersand_path = '100'
                exclamation_path = '100'
                blank_path = '100'
                til_path = '100'
                comma_path = '100'
                plus_path = '100'
                asterisk_path = '100'
                hashtag_path = '100'
                money_sign_path = '100'
                percentage_path = '100'
                len_path = '100'
            # FILE
            if dict_url['path']:
                dot_file = str(count(posixpath.basename(dict_url['path']), '.'))
                hyphe_file = str(count(posixpath.basename(dict_url['path']), '-'))
                underline_file = str(
                    count(posixpath.basename(dict_url['path']), '_'))
                bar_file = str(count(posixpath.basename(dict_url['path']), '/'))
                question_file = str(
                    count(posixpath.basename(dict_url['path']), '100'))
                equal_file = str(count(posixpath.basename(dict_url['path']), '='))
                arroba_file = str(count(posixpath.basename(dict_url['path']), '@'))
                ampersand_file = str(
                    count(posixpath.basename(dict_url['path']), '&'))
                exclamation_file = str(
                    count(posixpath.basename(dict_url['path']), '!'))
                blank_file = str(count(posixpath.basename(dict_url['path']), ' '))
                til_file = str(count(posixpath.basename(dict_url['path']), '~'))
                comma_file = str(count(posixpath.basename(dict_url['path']), ','))
                plus_file = str(count(posixpath.basename(dict_url['path']), '+'))
                asterisk_file = str(
                    count(posixpath.basename(dict_url['path']), '*'))
                hashtag_file = str(
                    count(posixpath.basename(dict_url['path']), '#'))
                money_sign_file = str(
                    count(posixpath.basename(dict_url['path']), '$'))
                percentage_file = str(
                    count(posixpath.basename(dict_url['path']), '%'))
                len_file = str(length(posixpath.basename(dict_url['path'])))
                extension = str(extract_extension(
                    posixpath.basename(dict_url['path'])))
            else:
                dot_file = '100'
                hyphe_file = '100'
                underline_file = '100'
                bar_file = '100'
                question_file = '100'
                equal_file = '100'
                arroba_file = '100'
                ampersand_file = '100'
                exclamation_file = '100'
                blank_file = '100'
                til_file = '100'
                comma_file = '100'
                plus_file = '100'
                asterisk_file = '100'
                hashtag_file = '100'
                money_sign_file = '100'
                percentage_file = '100'
                len_file = '100'
                extension = '100'
            # PARAMETERS
            if dict_url['query']:
                dot_params = str(count(dict_url['query'], '.'))
                hyphe_params = str(count(dict_url['query'], '-'))
                underline_params = str(count(dict_url['query'], '_'))
                bar_params = str(count(dict_url['query'], '/'))
                question_params = str(count(dict_url['query'], '100'))
                equal_params = str(count(dict_url['query'], '='))
                arroba_params = str(count(dict_url['query'], '@'))
                ampersand_params = str(count(dict_url['query'], '&'))
                exclamation_params = str(count(dict_url['query'], '!'))
                blank_params = str(count(dict_url['query'], ' '))
                til_params = str(count(dict_url['query'], '~'))
                comma_params = str(count(dict_url['query'], ','))
                plus_params = str(count(dict_url['query'], '+'))
                asterisk_params = str(count(dict_url['query'], '*'))
                hashtag_params = str(count(dict_url['query'], '#'))
                money_sign_params = str(count(dict_url['query'], '$'))
                percentage_params = str(count(dict_url['query'], '%'))
                len_params = str(length(dict_url['query']))
                tld_params = str(check_tld(dict_url['query']))
                number_params = str(count_params(dict_url['query']))
            else:
                dot_params = '100'
                hyphe_params = '100'
                underline_params = '100'
                bar_params = '100'
                question_params = '100'
                equal_params = '100'
                arroba_params = '100'
                ampersand_params = '100'
                exclamation_params = '100'
                blank_params = '100'
                til_params = '100'
                comma_params = '100'
                plus_params = '100'
                asterisk_params = '100'
                hashtag_params = '100'
                money_sign_params = '100'
                percentage_params = '100'
                len_params = '100'
                tld_params = '100'
                number_params = '100'


            _lexical = [
                dot_url, hyphe_url, underline_url, bar_url, question_url,
                equal_url, arroba_url, ampersand_url, exclamation_url,
                blank_url, til_url, comma_url, plus_url, asterisk_url, hashtag_url,
                money_sign_url, percentage_url, count_tld_url, len_url, dot_host,
                hyphe_host, underline_host, bar_host, question_host, equal_host,
                arroba_host, ampersand_host, exclamation_host, blank_host, til_host,
                comma_host, plus_host, asterisk_host, hashtag_host, money_sign_host,
                percentage_host, vowels_host, len_host, ip_exist, server_client,
                dot_path, hyphe_path, underline_path, bar_path, question_path,
                equal_path, arroba_path, ampersand_path, exclamation_path,
                blank_path, til_path, comma_path, plus_path, asterisk_path,
                hashtag_path, money_sign_path, percentage_path, len_path, dot_file,
                hyphe_file, underline_file, bar_file, question_file, equal_file,
                arroba_file, ampersand_file, exclamation_file, blank_file,
                til_file, comma_file, plus_file, asterisk_file, hashtag_file,
                money_sign_file, percentage_file, len_file, dot_params,
                hyphe_params, underline_params, bar_params, question_params,
                equal_params, arroba_params, ampersand_params, exclamation_params,
                blank_params, til_params, comma_params, plus_params, asterisk_params,
                hashtag_params, money_sign_params, percentage_params, len_params,
                tld_params, number_params, email_exist, extension
            ]



            result = []
            result.extend(_lexical)
            return result  

def get_words(path):
    """load stop words """
    
    with open(path, 'r', encoding="utf-8") as f:
        stopwords = f.read().splitlines()
        return stopwords    




# Calculate the total number delimeters in a URL
def Total_delims(str):

    delim = ['-', '_', '?', '=', '&']
    count = 0
    for i in str:
        for j in delim:
            if i == j:
                count += 1
    return count

# Binary feature for hostname tokens
def is_known_tld(url):

    tld = tldextract.extract(url).suffix
    if tld in All_Known_TLD:
        return 0
    else:
        return 1

# Binary feature for path tokens
def is_known_path(url):

    path = urlparse(url.lower()).path
    if(path!='' and path!=None):
        for i in Suspicious_Words:
            
            if i in path:
                return 1
            else:
                continue
    return 0
    
def exe_in_url(url):
    url=url.lower()
    if url.find('.exe')!=-1:
        return 1
    return 0
def consonants_ratio_urllength(url):
    vowels = 0
    consonants = 0

    for i in url :
        if(i == 'a' or i == 'e' or i == 'i' or i == 'o' or i == 'u'
           or i == 'A' or i == 'E' or i == 'I' or i == 'O' or i == 'U'):
                    vowels = vowels + 1
        else:
                    consonants = consonants + 1
    return consonants/(len(url))
    
def consonants_ratio_vowels(url):
    vowels = 0
    consonants = 0

    for i in url :
        if(i == 'a' or i == 'e' or i == 'i' or i == 'o' or i == 'u'
           or i == 'A' or i == 'E' or i == 'I' or i == 'O' or i == 'U'):
                    vowels = vowels + 1
        else:
                    consonants = consonants + 1
    if(vowels==0): 
        return 0 
    else :            
        return consonants/vowels
def countcharacters(url, character):
    count = url.count(character)
    return count
def numberofdigits(url):
    dict_url = start_url(url)
    digit=0
    for u in url:
        if(u.isnumeric()):
            digit+=1
    
    if((len(dict_url['host']))!=0):
        return digit/(len(dict_url['host']))
    else:
        return 0


def listbasedfeatures(url):
    dic=dict({'Total_delimitors':'','Known_TLD':'','Suspicious_word_in_path':'','Exe_in_url':'','Consonants_ratio_vowels':'','Numberofdigits':'','consonants_ratio_urllength':''})
    dic['Total_delimitors']=Total_delims(url)
    dic['Known_TLD']=is_known_tld(url)
    dic['Suspicious_word_in_path']=is_known_path(url)
    dic['Exe_in_url']= exe_in_url(url)
    dic['Consonants_ratio_vowels']=consonants_ratio_vowels(url)
    dic['Numberofdigits']=numberofdigits(url)
    dic['consonants_ratio_urllength']=consonants_ratio_urllength(url)
    return dic    
def Alphabetscount(url):
    L=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']   
    values=[]
    for l in L:
        values.append(countcharacters(url.lower(), l.lower()))
    return L,values
def shortningservice(url):
    shortening_services = r"bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|is\.gd|cli\.gs|" \
                      r"yfrog\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac|su\.pr|twurl\.nl|snipurl\.com|" \
                      r"short\.to|BudURL\.com|ping\.fm|post\.ly|Just\.as|bkite\.com|snipr\.com|fic\.kr|loopt\.us|" \
                      r"doiop\.com|short\.ie|kl\.am|wp\.me|rubyurl\.com|om\.ly|to\.ly|bit\.do|t\.co|lnkd\.in|db\.tt|" \
                      r"qr\.ae|adf\.ly|goo\.gl|bitly\.com|cur\.lv|tinyurl\.com|ow\.ly|bit\.ly|ity\.im|q\.gs|is\.gd|" \
                      r"po\.st|bc\.vc|twitthis\.com|u\.to|j\.mp|buzurl\.com|cutt\.us|u\.bb|yourls\.org|x\.co|" \
                      r"prettylinkpro\.com|scrnch\.me|filoops\.info|vzturl\.com|qr\.net|1url\.com|tweez\.me|v\.gd|" \
                      r"tr\.im|link\.zip\.net"
    match = re.search(shortening_services, url.lower())
    return -1 if match else 1

def extractBasicLexicalExternalFeatures(url):

                      
            
            dict_url = start_url(url)

            """LEXICAL"""
            # URL
            dot_url = str(count(dict_url['url'], '.'))
            hyphe_url = str(count(dict_url['url'], '-'))
            underline_url = str(count(dict_url['url'], '_'))
            bar_url = str(count(dict_url['url'], '/'))
            question_url = str(count(dict_url['url'], '?'))
            equal_url = str(count(dict_url['url'], '='))
            arroba_url = str(count(dict_url['url'], '@'))
            ampersand_url = str(count(dict_url['url'], '&'))
            exclamation_url = str(count(dict_url['url'], '!'))
            blank_url = str(count(dict_url['url'], ' '))
            til_url = str(count(dict_url['url'], '~'))
            comma_url = str(count(dict_url['url'], ','))
            plus_url = str(count(dict_url['url'], '+'))
            asterisk_url = str(count(dict_url['url'], '*'))
            hashtag_url = str(count(dict_url['url'], '#'))
            money_sign_url = str(count(dict_url['url'], '$'))
            percentage_url = str(count(dict_url['url'], '%'))
            len_url = str(length(dict_url['url']))
            email_exist = str(valid_email(dict_url['url']))
            count_tld_url = str(count_tld(dict_url['url']))
            # DOMAIN
            dot_host = str(count(dict_url['host'], '.'))
            hyphe_host = str(count(dict_url['host'], '-'))
            underline_host = str(count(dict_url['host'], '_'))
            bar_host = str(count(dict_url['host'], '/'))
            question_host = str(count(dict_url['host'], '?'))
            equal_host = str(count(dict_url['host'], '='))
            arroba_host = str(count(dict_url['host'], '@'))
            ampersand_host = str(count(dict_url['host'], '&'))
            exclamation_host = str(count(dict_url['host'], '!'))
            blank_host = str(count(dict_url['host'], ' '))
            til_host = str(count(dict_url['host'], '~'))
            comma_host = str(count(dict_url['host'], ','))
            plus_host = str(count(dict_url['host'], '+'))
            asterisk_host = str(count(dict_url['host'], '*'))
            hashtag_host = str(count(dict_url['host'], '#'))
            money_sign_host = str(count(dict_url['host'], '$'))
            percentage_host = str(count(dict_url['host'], '%'))
            vowels_host = str(count_vowels(dict_url['host']))
            len_host = str(length(dict_url['host']))
            ip_exist = str(valid_ip(dict_url['host']))
            server_client = str(check_word_server_client(dict_url['host']))
            # DIRECTORY
            if dict_url['path']:
                dot_path = str(count(dict_url['path'], '.'))
                hyphe_path = str(count(dict_url['path'], '-'))
                underline_path = str(count(dict_url['path'], '_'))
                bar_path = str(count(dict_url['path'], '/'))
                question_path = str(count(dict_url['path'], '?'))
                equal_path = str(count(dict_url['path'], '='))
                arroba_path = str(count(dict_url['path'], '@'))
                ampersand_path = str(count(dict_url['path'], '&'))
                exclamation_path = str(count(dict_url['path'], '!'))
                blank_path = str(count(dict_url['path'], ' '))
                til_path = str(count(dict_url['path'], '~'))
                comma_path = str(count(dict_url['path'], ','))
                plus_path = str(count(dict_url['path'], '+'))
                asterisk_path = str(count(dict_url['path'], '*'))
                hashtag_path = str(count(dict_url['path'], '#'))
                money_sign_path = str(count(dict_url['path'], '$'))
                percentage_path = str(count(dict_url['path'], '%'))
                len_path = str(length(dict_url['path']))
            else:
                dot_path = 0
                hyphe_path = 0
                underline_path = 0
                bar_path = 0
                question_path = 0
                equal_path = 0
                arroba_path = 0
                ampersand_path = 0
                exclamation_path = 0
                blank_path = 0
                til_path = 0
                comma_path = 0
                plus_path = 0
                asterisk_path = 0
                hashtag_path = 0
                money_sign_path = 0
                percentage_path = 0
                len_path = 0
            # FILE
            if dict_url['path']:
                dot_file = str(count(posixpath.basename(dict_url['path']), '.'))
                hyphe_file = str(count(posixpath.basename(dict_url['path']), '-'))
                underline_file = str(
                    count(posixpath.basename(dict_url['path']), '_'))
                bar_file = str(count(posixpath.basename(dict_url['path']), '/'))
                question_file = str(
                    count(posixpath.basename(dict_url['path']), '?'))
                equal_file = str(count(posixpath.basename(dict_url['path']), '='))
                arroba_file = str(count(posixpath.basename(dict_url['path']), '@'))
                ampersand_file = str(
                    count(posixpath.basename(dict_url['path']), '&'))
                exclamation_file = str(
                    count(posixpath.basename(dict_url['path']), '!'))
                blank_file = str(count(posixpath.basename(dict_url['path']), ' '))
                til_file = str(count(posixpath.basename(dict_url['path']), '~'))
                comma_file = str(count(posixpath.basename(dict_url['path']), ','))
                plus_file = str(count(posixpath.basename(dict_url['path']), '+'))
                asterisk_file = str(
                    count(posixpath.basename(dict_url['path']), '*'))
                hashtag_file = str(
                    count(posixpath.basename(dict_url['path']), '#'))
                money_sign_file = str(
                    count(posixpath.basename(dict_url['path']), '$'))
                percentage_file = str(
                    count(posixpath.basename(dict_url['path']), '%'))
                len_file = str(length(posixpath.basename(dict_url['path'])))
                extension = str(extract_extension(
                    posixpath.basename(dict_url['path'])))
            else:
                dot_file = 0
                hyphe_file = 0
                underline_file = 0
                bar_file = 0
                question_file = 0
                equal_file = 0
                arroba_file = 0
                ampersand_file = 0
                exclamation_file = 0
                blank_file = 0
                til_file = 0
                comma_file = 0
                plus_file = 0
                asterisk_file = 0
                hashtag_file = 0
                money_sign_file = 0
                percentage_file = 0
                len_file = 0
                extension = 0
            # PARAMETERS
            if dict_url['query']:
                dot_params = str(count(dict_url['query'], '.'))
                hyphe_params = str(count(dict_url['query'], '-'))
                underline_params = str(count(dict_url['query'], '_'))
                bar_params = str(count(dict_url['query'], '/'))
                question_params = str(count(dict_url['query'], '?'))
                equal_params = str(count(dict_url['query'], '='))
                arroba_params = str(count(dict_url['query'], '@'))
                ampersand_params = str(count(dict_url['query'], '&'))
                exclamation_params = str(count(dict_url['query'], '!'))
                blank_params = str(count(dict_url['query'], ' '))
                til_params = str(count(dict_url['query'], '~'))
                comma_params = str(count(dict_url['query'], ','))
                plus_params = str(count(dict_url['query'], '+'))
                asterisk_params = str(count(dict_url['query'], '*'))
                hashtag_params = str(count(dict_url['query'], '#'))
                money_sign_params = str(count(dict_url['query'], '$'))
                percentage_params = str(count(dict_url['query'], '%'))
                len_params = str(length(dict_url['query']))
                tld_params = str(check_tld(dict_url['query']))
                number_params = str(count_params(dict_url['query']))
            else:
                dot_params = 0
                hyphe_params = 0
                underline_params = 0
                bar_params = 0
                question_params = 0
                equal_params = 0
                arroba_params = 0
                ampersand_params = 0
                exclamation_params = 0
                blank_params = 0
                til_params = 0
                comma_params = 0
                plus_params = 0
                asterisk_params = 0
                hashtag_params = 0
                money_sign_params = 0
                percentage_params = 0
                len_params = 0
                tld_params = 0
                number_params = 0

            """BLACKLIST"""
            #blacklist_url = str(check_blacklists(dict_url['protocol'] + '://' + dict_url['url']))
            #blacklist_ip = str(check_blacklists_ip(dict_url))
            #blacklist_domain = str(check_blacklists(dict_url['protocol'] + '://' + dict_url['host']))

            """HOST"""
            
            with timeout(60):
                try:
                    spf = str(valid_spf(dict_url['host']))
                except: 
                    spf=-1
            with timeout(60):
                try:
                    rbl = str(check_rbl(dict_url['host']))
                except: 
                    rbl=-1
            with timeout(60):
                try:
                    time_domain = str(check_time_response(dict_url['protocol'] + '://' + dict_url['host']))
                    
                except: 
                    time_domain=-1
                    
            with timeout(60):
                try:
                    asn = str(get_asn_number(dict_url)) 
                except: 
                    asn=-1
            with timeout(60):
                try:
                    country = str(get_country(dict_url))
                except: 
                    country=-1  
            with timeout(60):
                try:
                    ptr = str(get_ptr(dict_url))
                except: 
                    ptr=-1         
            activation_time = str(time_activation_domain(url))
            expiration_time = str(expiration_date_register(url))
                      
            with timeout(60):
                try:
                    count_ip = str(count_ips(dict_url))
                except: 
                    count_ip=-1 
            with timeout(60):
                try:
                    count_ns = str(count_name_servers(dict_url))
                except: 
                    count_ns=-1         
            with timeout(60):
                try:
                    count_mx = str(count_mx_servers(dict_url))
                except: 
                    count_mx=-1         
            with timeout(60):
                try:
                    ttl = str(extract_ttl(dict_url))
                except: 
                    ttl=-1            
            _host = [rbl, time_domain, spf, country, asn, activation_time,
                     expiration_time, count_ip, count_ns, count_mx, ttl]        
                    
        
            """OTHERS"""
            with timeout(60):
                try:
                    ssl = str(check_ssl('https://' + dict_url['url']))
                except: 
                    ssl=-1       
                    
            with timeout(60):
                try:
                    count_redirect = str(count_redirects(
                        dict_url['protocol'] + '://' + dict_url['url']))
                except: 
                    count_redirect=-1
            with timeout(60):
                try:
                    google_url = str(google_search(dict_url['url']))
                except: 
                    google_url=-1
            with timeout(60):
                try:
                    google_domain = str(google_search(dict_url['host']))
                except: 
                    google_domain=-1
            with timeout(60):
                try:
                    shortener = str(check_shortener(dict_url))
                except: 
                    shortener=-1        
                    
                    
                    

            _others = [ssl, count_redirect, google_url, google_domain, shortener]

                
           
            _lexical = [
                dot_url, hyphe_url, underline_url, bar_url, question_url,
                equal_url, arroba_url, ampersand_url, exclamation_url,
                blank_url, til_url, comma_url, plus_url, asterisk_url, hashtag_url,
                money_sign_url, percentage_url, count_tld_url, len_url, dot_host,
                hyphe_host, underline_host, bar_host, question_host, equal_host,
                arroba_host, ampersand_host, exclamation_host, blank_host, til_host,
                comma_host, plus_host, asterisk_host, hashtag_host, money_sign_host,
                percentage_host, vowels_host, len_host, ip_exist, server_client,
                dot_path, hyphe_path, underline_path, bar_path, question_path,
                equal_path, arroba_path, ampersand_path, exclamation_path,
                blank_path, til_path, comma_path, plus_path, asterisk_path,
                hashtag_path, money_sign_path, percentage_path, len_path, dot_file,
                hyphe_file, underline_file, bar_file, question_file, equal_file,
                arroba_file, ampersand_file, exclamation_file, blank_file,
                til_file, comma_file, plus_file, asterisk_file, hashtag_file,
                money_sign_file, percentage_file, len_file, dot_params,
                hyphe_params, underline_params, bar_params, question_params,
                equal_params, arroba_params, ampersand_params, exclamation_params,
                blank_params, til_params, comma_params, plus_params, asterisk_params,
                hashtag_params, money_sign_params, percentage_params, len_params,
                tld_params, number_params, email_exist, extension
            ]

            #_blacklist = [blacklist_url, blacklist_ip, blacklist_domain]
#['dominio_presente_em_rbl', 'tempo_resposta', 'possui_spf', 'localizacao_geografica_ip',
#             'tempo_ativacao_dominio', 'tempo_expiracao_dominio',
#            'qtd_ip_resolvido', 'qtd_nameservers', 'qtd_servidores_mx', 'valor_ttl_associado']
            
            result = []
            result.extend(_lexical)
            #result.extend(_blacklist)
            result.extend(_host)
            result.extend(_others)
            result.extend([''])

            return result    

def extractBasicLexicalFeatures(url):

                      
            dict_url = start_url(url)

            """LEXICAL"""
            # URL
            dot_url = str(count(dict_url['url'], '.'))
            hyphe_url = str(count(dict_url['url'], '-'))
            underline_url = str(count(dict_url['url'], '_'))
            bar_url = str(count(dict_url['url'], '/'))
            question_url = str(count(dict_url['url'], '100'))
            equal_url = str(count(dict_url['url'], '='))
            arroba_url = str(count(dict_url['url'], '@'))
            ampersand_url = str(count(dict_url['url'], '&'))
            exclamation_url = str(count(dict_url['url'], '!'))
            blank_url = str(count(dict_url['url'], ' '))
            til_url = str(count(dict_url['url'], '~'))
            comma_url = str(count(dict_url['url'], ','))
            plus_url = str(count(dict_url['url'], '+'))
            asterisk_url = str(count(dict_url['url'], '*'))
            hashtag_url = str(count(dict_url['url'], '#'))
            money_sign_url = str(count(dict_url['url'], '$'))
            percentage_url = str(count(dict_url['url'], '%'))
            len_url = str(length(dict_url['url']))
            email_exist = str(valid_email(dict_url['url']))
            count_tld_url = str(count_tld(dict_url['url']))
            # DOMAIN
            dot_host = str(count(dict_url['host'], '.'))
            hyphe_host = str(count(dict_url['host'], '-'))
            underline_host = str(count(dict_url['host'], '_'))
            bar_host = str(count(dict_url['host'], '/'))
            question_host = str(count(dict_url['host'], '100'))
            equal_host = str(count(dict_url['host'], '='))
            arroba_host = str(count(dict_url['host'], '@'))
            ampersand_host = str(count(dict_url['host'], '&'))
            exclamation_host = str(count(dict_url['host'], '!'))
            blank_host = str(count(dict_url['host'], ' '))
            til_host = str(count(dict_url['host'], '~'))
            comma_host = str(count(dict_url['host'], ','))
            plus_host = str(count(dict_url['host'], '+'))
            asterisk_host = str(count(dict_url['host'], '*'))
            hashtag_host = str(count(dict_url['host'], '#'))
            money_sign_host = str(count(dict_url['host'], '$'))
            percentage_host = str(count(dict_url['host'], '%'))
            vowels_host = str(count_vowels(dict_url['host']))
            len_host = str(length(dict_url['host']))
            ip_exist = str(valid_ip(dict_url['host']))
            server_client = str(check_word_server_client(dict_url['host']))
            # DIRECTORY
            if dict_url['path']:
                dot_path = str(count(dict_url['path'], '.'))
                hyphe_path = str(count(dict_url['path'], '-'))
                underline_path = str(count(dict_url['path'], '_'))
                bar_path = str(count(dict_url['path'], '/'))
                question_path = str(count(dict_url['path'], '100'))
                equal_path = str(count(dict_url['path'], '='))
                arroba_path = str(count(dict_url['path'], '@'))
                ampersand_path = str(count(dict_url['path'], '&'))
                exclamation_path = str(count(dict_url['path'], '!'))
                blank_path = str(count(dict_url['path'], ' '))
                til_path = str(count(dict_url['path'], '~'))
                comma_path = str(count(dict_url['path'], ','))
                plus_path = str(count(dict_url['path'], '+'))
                asterisk_path = str(count(dict_url['path'], '*'))
                hashtag_path = str(count(dict_url['path'], '#'))
                money_sign_path = str(count(dict_url['path'], '$'))
                percentage_path = str(count(dict_url['path'], '%'))
                len_path = str(length(dict_url['path']))
            else:
                dot_path = '100'
                hyphe_path = '100'
                underline_path = '100'
                bar_path = '100'
                question_path = '100'
                equal_path = '100'
                arroba_path = '100'
                ampersand_path = '100'
                exclamation_path = '100'
                blank_path = '100'
                til_path = '100'
                comma_path = '100'
                plus_path = '100'
                asterisk_path = '100'
                hashtag_path = '100'
                money_sign_path = '100'
                percentage_path = '100'
                len_path = '100'
            # FILE
            if dict_url['path']:
                dot_file = str(count(posixpath.basename(dict_url['path']), '.'))
                hyphe_file = str(count(posixpath.basename(dict_url['path']), '-'))
                underline_file = str(
                    count(posixpath.basename(dict_url['path']), '_'))
                bar_file = str(count(posixpath.basename(dict_url['path']), '/'))
                question_file = str(
                    count(posixpath.basename(dict_url['path']), '100'))
                equal_file = str(count(posixpath.basename(dict_url['path']), '='))
                arroba_file = str(count(posixpath.basename(dict_url['path']), '@'))
                ampersand_file = str(
                    count(posixpath.basename(dict_url['path']), '&'))
                exclamation_file = str(
                    count(posixpath.basename(dict_url['path']), '!'))
                blank_file = str(count(posixpath.basename(dict_url['path']), ' '))
                til_file = str(count(posixpath.basename(dict_url['path']), '~'))
                comma_file = str(count(posixpath.basename(dict_url['path']), ','))
                plus_file = str(count(posixpath.basename(dict_url['path']), '+'))
                asterisk_file = str(
                    count(posixpath.basename(dict_url['path']), '*'))
                hashtag_file = str(
                    count(posixpath.basename(dict_url['path']), '#'))
                money_sign_file = str(
                    count(posixpath.basename(dict_url['path']), '$'))
                percentage_file = str(
                    count(posixpath.basename(dict_url['path']), '%'))
                len_file = str(length(posixpath.basename(dict_url['path'])))
                extension = str(extract_extension(
                    posixpath.basename(dict_url['path'])))
            else:
                dot_file = '100'
                hyphe_file = '100'
                underline_file = '100'
                bar_file = '100'
                question_file = '100'
                equal_file = '100'
                arroba_file = '100'
                ampersand_file = '100'
                exclamation_file = '100'
                blank_file = '100'
                til_file = '100'
                comma_file = '100'
                plus_file = '100'
                asterisk_file = '100'
                hashtag_file = '100'
                money_sign_file = '100'
                percentage_file = '100'
                len_file = '100'
                extension = '100'
            # PARAMETERS
            if dict_url['query']:
                dot_params = str(count(dict_url['query'], '.'))
                hyphe_params = str(count(dict_url['query'], '-'))
                underline_params = str(count(dict_url['query'], '_'))
                bar_params = str(count(dict_url['query'], '/'))
                question_params = str(count(dict_url['query'], '100'))
                equal_params = str(count(dict_url['query'], '='))
                arroba_params = str(count(dict_url['query'], '@'))
                ampersand_params = str(count(dict_url['query'], '&'))
                exclamation_params = str(count(dict_url['query'], '!'))
                blank_params = str(count(dict_url['query'], ' '))
                til_params = str(count(dict_url['query'], '~'))
                comma_params = str(count(dict_url['query'], ','))
                plus_params = str(count(dict_url['query'], '+'))
                asterisk_params = str(count(dict_url['query'], '*'))
                hashtag_params = str(count(dict_url['query'], '#'))
                money_sign_params = str(count(dict_url['query'], '$'))
                percentage_params = str(count(dict_url['query'], '%'))
                len_params = str(length(dict_url['query']))
                tld_params = str(check_tld(dict_url['query']))
                number_params = str(count_params(dict_url['query']))
            else:
                dot_params = '100'
                hyphe_params = '100'
                underline_params = '100'
                bar_params = '100'
                question_params = '100'
                equal_params = '100'
                arroba_params = '100'
                ampersand_params = '100'
                exclamation_params = '100'
                blank_params = '100'
                til_params = '100'
                comma_params = '100'
                plus_params = '100'
                asterisk_params = '100'
                hashtag_params = '100'
                money_sign_params = '100'
                percentage_params = '100'
                len_params = '100'
                tld_params = '100'
                number_params = '100'

#            """BLACKLIST"""
#            blacklist_url = str(check_blacklists(dict_url['protocol'] + '://' + dict_url['url']))
#            blacklist_ip = str(check_blacklists_ip(dict_url))
#            blacklist_domain = str(check_blacklists(dict_url['protocol'] + '://' + dict_url['host']))
#
#            """HOST"""
#            spf = str(valid_spf(dict_url['host']))
#            rbl = str(check_rbl(dict_url['host']))
#            time_domain = str(check_time_response(dict_url['protocol'] + '://' + dict_url['host']))
#            asn = str(get_asn_number(dict_url))
#            country = str(get_country(dict_url))
#            ptr = str(get_ptr(dict_url))
#            activation_time = str(time_activation_domain(dict_url))
#            expiration_time = str(expiration_date_register(dict_url))
#            count_ip = str(count_ips(dict_url))
#            count_ns = str(count_name_servers(dict_url))
#            count_mx = str(count_mx_servers(dict_url))
#            ttl = str(extract_ttl(dict_url))
#
#            """OTHERS"""
#            ssl = str(check_ssl('https://' + dict_url['url']))
#            count_redirect = str(count_redirects(
#                dict_url['protocol'] + '://' + dict_url['url']))
#            google_url = str(google_search(dict_url['url']))
#            google_domain = str(google_search(dict_url['host']))
#            shortener = str(check_shortener(dict_url))

            _lexical = [
                dot_url, hyphe_url, underline_url, bar_url, question_url,
                equal_url, arroba_url, ampersand_url, exclamation_url,
                blank_url, til_url, comma_url, plus_url, asterisk_url, hashtag_url,
                money_sign_url, percentage_url, count_tld_url, len_url, dot_host,
                hyphe_host, underline_host, bar_host, question_host, equal_host,
                arroba_host, ampersand_host, exclamation_host, blank_host, til_host,
                comma_host, plus_host, asterisk_host, hashtag_host, money_sign_host,
                percentage_host, vowels_host, len_host, ip_exist, server_client,
                dot_path, hyphe_path, underline_path, bar_path, question_path,
                equal_path, arroba_path, ampersand_path, exclamation_path,
                blank_path, til_path, comma_path, plus_path, asterisk_path,
                hashtag_path, money_sign_path, percentage_path, len_path, dot_file,
                hyphe_file, underline_file, bar_file, question_file, equal_file,
                arroba_file, ampersand_file, exclamation_file, blank_file,
                til_file, comma_file, plus_file, asterisk_file, hashtag_file,
                money_sign_file, percentage_file, len_file, dot_params,
                hyphe_params, underline_params, bar_params, question_params,
                equal_params, arroba_params, ampersand_params, exclamation_params,
                blank_params, til_params, comma_params, plus_params, asterisk_params,
                hashtag_params, money_sign_params, percentage_params, len_params,
                tld_params, number_params, email_exist, extension
            ]

#            _blacklist = [blacklist_url, blacklist_ip, blacklist_domain]
#
#            _host = [rbl, time_domain, spf, country, asn, ptr, activation_time,
#                     expiration_time, count_ip, count_ns, count_mx, ttl]

#            _others = [ssl, count_redirect, google_url, google_domain, shortener]

            result = []
            result.extend(_lexical)
#            result.extend(_blacklist)
#            result.extend(_host)
#            result.extend(_others)
#            result.extend([''])

            return result    

import math
 
def hist(source):
    hist = {}; l = 0;
    for e in source:
        l += 1
        if e not in hist:
            hist[e] = 0
        hist[e] += 1
    return (l,hist)
 
def entropy(hist,l):
    elist = []
    for v in hist.values():
        c = v / l
        elist.append(-c * math.log(c ,2))
    return sum(elist)
         
def AllLexicalFeatures(i,writer,url,label):
    Features1=extractBasicLexicalExternalFeatures(url)
    Features2=listbasedfeatures(url)
    [keys,Feature3]=Alphabetscount(url)
    Feature4=shortningservice(url)
    (l,h) = hist(url)
    Feature5=entropy(h, l)
    if(i==0):
        AttributeNames=[['url'],attributes(),list(Features2.keys()),list(keys),['shortnighservice'],['entropy'],['label']]
        writer.writerow(list(itertools.chain(*AttributeNames)))
    FeatureList=[[url],Features1,list(Features2.values()),Feature3,[Feature4],[Feature5],[label]]   
    FeaturesFinal=list(itertools.chain(*FeatureList))
    writer.writerow(FeaturesFinal)
import posixpath  

def LexicalFeatureVectorGenerator(urllist,labellist):
    if (not os.path.exists(Featurepath+'\\Basic_Lexical_Results\\TrainFeatures.csv')):
        filetowrite=open( Featurepath+'\\Basic_Lexical_Results\\TrainFeatures.csv', 'w' ,encoding='utf-8',newline='') 
        writer1 = csv.writer(filetowrite)
        for i,url in enumerate(urllist):
            url=url[0]
            if(is_valid_uri(url)==True and validators.url(url)==True):
                AllLexicalFeatures(i,writer1,''.join(url),labellist[i])        

def AllExternalLexicalFeatures(df):
    labels=list(df['label'])
    urls=list(df['url'])
    print('total URLs', len(urls))
    for i,url in enumerate(urls):
        url=url[0]
        #print(url)
        filetowrite=open(Featurepath+'Basic_Lexical_External_Results/'+'TrainFeatures_multiprocess.csv', 'a+' ,encoding='utf-8',newline='') 
        writer = csv.writer(filetowrite)
        start_time = time.time()
        try:
            Features1 = extractBasicLexicalFeatures(url)
            Features2=listbasedfeatures(url)
            [keys,Feature3]=Alphabetscount(url)
            Feature4=shortningservice(url)
            (l,h) = hist(url)
            Feature5=entropy(h, l)
                
            FeatureList=[[url],Features1,list(Features2.values()),Feature3,[Feature4],[Feature5],[labels[i]]]    
            FeaturesFinal=list(itertools.chain(*FeatureList))
            writer.writerow(FeaturesFinal)
            filetowrite.close()
            print("Done generating features --- %s seconds ---" % (time.time() - start_time))
        except:
          print('skipping Error')
    #print('AllLexicalFeatures')


#from func_timeout import func_timeout, FunctionTimedOut


class NoDaemonProcess(multiprocessing.Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)

# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
class MyPool(multiprocessing.pool.Pool):
    Process = NoDaemonProcess
    
    
def parallelize_dataframe(df, func, n_cores=multiprocessing.cpu_count()):
    print("Number of CPU ",multiprocessing.cpu_count())
    df_split = np.array_split(df, n_cores)
    pool = MyPool(n_cores)
    pool.map(func, df_split)
    pool.close()
    pool.join()
    
# Parallel Processing
def ExternalLexicalFeatureVectorGenerator(urllist,labellist):
           
        filetowrite=open(Featurepath+'Basic_Lexical_External_Results/TrainFeatures_multiprocess1.csv', 'w+' ,encoding='utf-8',newline='') 
        writer = csv.writer(filetowrite)
        zerourl=urllist[0]
        Features2=listbasedfeatures(zerourl[0])
        [keys,Feature3]=Alphabetscount(zerourl[0])
        AttributeNames=[['url'],externallexicalattributes(),list(Features2.keys()),list(keys),['shortnighservice'],['entropy'],['label']]
        writer.writerow(list(itertools.chain(*AttributeNames)))
        filetowrite.close()
        d={'url':urllist,'label':labellist}
        url_data=pd.DataFrame(d, columns=['url','label'])
        parallelize_dataframe(url_data, AllExternalLexicalFeatures)

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
            if(protocol==None):
                protocol='http'
                
    
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
    if(path=='/'):
        path=''
    if(domain=='' and ip!=''):
        domain=ip
    if(path==None or path=='/'):
         path=''
    if(sld==None or sld==''):
         sld='www'
    if(parameter==None):
        parameter=''
    if(fragment==None):
          fragment=''
    if(host==None):
           host=sld+'.'+domain+'.'+tld
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
       
                


seg_eng = Segmenter(corpus="twitter") 
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
#          words.append(w.split('_'))
#          words.remove(w)
           try:
            	subwords=(seg_eng.segment(w))
           except:
                subwords=w
           
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

def obtainwordlist(url):
    url=re.sub('\W',' ',url)
    url=re.sub('\d',' ',url)
    words=url.split(' ')    
    finalwords=tokenparts(words)
    final=converter(finalwords)
    
    return final   
def extract_features(start_n_gram, end_n_gram,flag=''):
        if(flag=='c'):
           return feature_extraction.text.CountVectorizer(ngram_range=(start_n_gram, end_n_gram),min_df=0.001,analyzer='char',tokenizer=obtainwordlist)
        elif(flag=='p'):
            return feature_extraction.text.CountVectorizer(analyzer='word',min_df=0.001,ngram_range=(start_n_gram, end_n_gram),tokenizer=obtainwordlist)

        else:
            return feature_extraction.text.CountVectorizer(analyzer='word', ngram_range=(start_n_gram, end_n_gram),min_df=0.001,tokenizer=obtainwordlist)
# Build the Classifiers


models = {
            
            'rf' : RandomForestClassifier,
            'dt':DecisionTreeClassifier,
            'xgb':XGBClassifier,
            'lgbm':LGBMClassifier,
            'lr' : LogisticRegression,
            'knn' : KNeighborsClassifier,
            'svm' : LinearSVC
            
        }
    


def evaluate(clf, X_train, X_test, y_train, y_test):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    FPR=0
    FNR=0
    Auc=0
    cnmat = confusion_matrix(np.asarray(y_test),y_pred).ravel().tolist()
    if(len(cnmat)>2):
            TN=cnmat[0]
            FP=cnmat[1]
            FN=cnmat[2]
            TP=cnmat[3]
            totalpos=(FN+TP)
            totalneg=(FP+TN)
        
            if(totalpos!=0):
                FNR=(FN/totalpos)
           
            if(totalneg!=0):    
                FPR=(FP/totalneg)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred, pos_label=1)
    Auc=metrics.auc(fpr, tpr)           
          
    return y_pred,accuracy_score(y_test, y_pred), f1_score(y_test, y_pred, average='macro'),\
           f1_score(y_test, y_pred, average='weighted'),recall_score(y_test, y_pred, average='macro'), precision_score(y_test, y_pred, average='macro'),FPR,FNR ,Auc
#train(urllisttr,X_train_transformed,Y,clf,clf_name,config)
def search_space(model):
    model = model.lower()
    space = {}
    if model == 'knn':
                space = {
                         'n_neighbors': hp.choice('n_neighbors', range(1,30)),
                                         'scale': hp.choice('scale', [0, 1]),
                         'normalize': hp.choice('normalize', [0, 1]),
                        }
                
    elif model == 'svm':
                 space = {
                             'C': hp.uniform('C', 0, 3),
                             'kernel': hp.choice('kernel', ['linear']),
                             #'gamma': hp.uniform('gamma', 0.01, 10),
                             'scale': hp.choice('scale', [0, 1]),
                         }
    elif model == 'rf':
        space = {
                   'max_depth': hp.choice('max_depth', range(190,200)),
                   #'max_features': hp.choice('max_features', range(1,3)),
                   'n_estimators': hp.choice('n_estimators', range(100,200)),
                   'criterion': hp.choice('criterion', ["entropy"]),
                }
    elif model == 'lr':
       space = {
              'warm_start' : hp.choice('warm_start', [False]),
              'fit_intercept' : hp.choice('fit_intercept', [True]),
              'tol' : hp.uniform('tol', 0.00007, 0.0001),
               'C' : hp.uniform('C', 2, 3),
               'solver' : hp.choice('solver', ['lbfgs']),
               'max_iter' : hp.choice('max_iter', range(150,300)),
               'scale': hp.choice('scale', [0, 1]),
               'normalize': hp.choice('normalize', [0, 1]),
               'multi_class' : 'auto',
               'class_weight' : 'balanced'
               }
    elif model == 'dt':
         space = {
                'max_depth': hp.choice('max_depth', range(195,250)),
                #'max_features': hp.choice('max_features', range(1,10)),
                'criterion': hp.choice('criterion', ["entropy"]),
                'scale': hp.choice('scale', [0, 1]),
                'normalize': hp.choice('normalize', [0, 1])
            }
    elif model=='lgbm':
        #LIGHTGBM PARAMETERS
        space = {
          'learning_rate': hp.choice('learning_rate',np.arange(0.05,0.31,0.05)),
          'max_depth': hp.choice('max_depth',np.arange(10,50,1,dtype=int)),
          #'min_child_weight':hp.choice('min_child_weight',np.arange(1,8,1,dtype=int)),
          #'colsample_bytree':hp.choice('colsample_bytree',np.arange(0.3,0.8,0.1)),
          #'subsample':hp.uniform('subsample',0.8,1),
          'num_leaves' : hp.quniform('num_leaves', 120, 300, 1),
          'n_estimators':200
          }

      
    elif model =='xgb':
        
         tree_method = [{'tree_method' : 'exact'},
               {'tree_method' : 'approx'},
               {'tree_method' : 'hist',
                #'max_bin': hp.quniform('max_bin', 2**3, 2**7, 1),
                'grow_policy' : {'grow_policy': {'grow_policy':'depthwise'},
                                'grow_policy' : {'grow_policy':'lossguide',
                                                 'max_leaves': hp.quniform('max_leaves', 100, 200, 1)}}}]
         
         
         space ={
                'tree_method' : hp.choice('tree_method', tree_method),
                'max_depth': hp.quniform('max_depth', 1, 100, 1),
                #'max_leaves':hp.quniform('max_leaves', 100, 200, 1),
               }
        

    space['model'] = model
    return space

from sklearn.preprocessing import scale, normalize
from sklearn.metrics import make_scorer,balanced_accuracy_score
from sklearn.metrics import matthews_corrcoef
def FPR(clf, X, y):
     y_pred = clf.predict(X)
     cm= confusion_matrix(y, y_pred) 
     tn= cm[0, 0]
     fp=cm[0, 1]
     fn=cm[1, 0]
     tp= cm[1, 1]
     totalpos=(fn+tp)
     totalneg=(fp+tn)
             
     return fp/totalneg
 

def get_acc_status(clf,X_,y):
    custom_scorer = {'accuracy': make_scorer(accuracy_score),
                 'balanced_accuracy': make_scorer(balanced_accuracy_score),
                 'precision': make_scorer(precision_score, average='macro'),
                 'recall': make_scorer(recall_score, average='macro'),
                 'f1': make_scorer(f1_score, average='macro'),
                 'auc':'roc_auc',
                 'Matthew': make_scorer(matthews_corrcoef),
                 'FPR':FPR
                 }
    kf = StratifiedKFold(n_splits=10,random_state=42,shuffle=True)
    score = cross_validate(clf, X_, y,cv=kf,scoring=custom_scorer,n_jobs=-1)
    clf.fit(X_,y)
    
   
    print('score is ####################',score)
    return {'loss':1-score['test_Matthew'].mean(), 'status': STATUS_OK, 'Trained_Model': clf,'scoreMetrics': score}

from sklearn.preprocessing import scale, normalize
from sklearn.metrics import make_scorer,balanced_accuracy_score
def scale_normalize(params,X_):
    if 'normalize' in params:
        if params['normalize'] == 1:
            X_ = normalize(X_)
        del params['normalize']
    if 'scale' in params:
        if params['scale'] == 1:
            X_ = scale(X_,with_mean=False)
        del params['scale']
        
    return X_
def obj_fnc(params): 
    model = params.get('model').lower()
    
    if model == 'lgbm' :
        integer_params = ['num_leaves']
        for param in integer_params:
               params[param] = int(params[param])   
    if model == 'xgb' :
        integer_params = ['max_depth']
        for param in integer_params:
               params[param] = int(params[param])
        if params['tree_method']['tree_method'] == 'hist':
                #max_bin = params['tree_method'].get('max_bin')
                #params['max_bin'] = int(max_bin)
                if params['tree_method']['grow_policy']['grow_policy']['grow_policy'] == 'depthwise':
                    grow_policy = params['tree_method'].get('grow_policy').get('grow_policy').get('grow_policy')
                    params['grow_policy'] = grow_policy
                    params['tree_method'] = 'hist'
                else:
                    max_leaves = params['tree_method']['grow_policy']['grow_policy'].get('max_leaves')
                    params['grow_policy'] = 'lossguide'
                    params['max_leaves'] = int(max_leaves)
                    params['tree_method'] = 'hist'
        else:
                params['tree_method'] = params['tree_method'].get('tree_method')
    
   
    X_ = scale_normalize(params,X[:]) 
    del params['model']
    clf = models[model](**params)
    return(get_acc_status(clf,X_,y))
   
X=[]
y=[]

def getBestScoreMetricfromTrials(trials):
    valid_trial_list = [trial for trial in trials
                            if STATUS_OK == trial['result']['status']]
    losses = [ float(trial['result']['loss']) for trial in valid_trial_list]
    index_having_minumum_loss = np.argmin(losses)
    best_trial_obj = valid_trial_list[index_having_minumum_loss]
    return best_trial_obj['result']['scoreMetrics']

def getBestModelfromTrials(trials):
    valid_trial_list = [trial for trial in trials
                            if STATUS_OK == trial['result']['status']]
    losses = [ float(trial['result']['loss']) for trial in valid_trial_list]
    index_having_minumum_loss = np.argmin(losses)
    best_trial_obj = valid_trial_list[index_having_minumum_loss]
    return best_trial_obj['result']['Trained_Model']
def train(urllist,examples,labels,clf,clf_name,config,overall):
        hypopt_trials = Trials()
        global X
        global y
        X=examples
        y=labels
        best_params = fmin(obj_fnc, search_space(clf_name), algo=tpe.suggest, max_evals=500, trials= hypopt_trials,early_stop_fn=early_stop.no_progress_loss(50))
        print(hypopt_trials.best_trial['result']['loss'])
        bestmodel = getBestModelfromTrials(hypopt_trials)
        scores= getBestScoreMetricfromTrials(hypopt_trials)
        overall.writerow([clf_name,best_params,(1-hypopt_trials.best_trial['result']['loss']),scores['test_accuracy'].mean(),scores['test_balanced_accuracy'].mean(),
                 scores['test_precision'].mean(),scores['test_recall'].mean(),scores['test_f1'].mean(),scores['test_auc'].mean(),scores['test_FPR'].mean(),(1-scores['test_recall'].mean())])
        return bestmodel 
#    classification(urllisttr,X_train_transformed,Y,config)

def classification(urllisttr,X_train_transformed,Y,config):
    
    clf=models
    #FoldDetails=open(Datapath+config+'//Results//K_foldResults.csv', 'w+' ,encoding='utf-8',newline='')
    #details = csv.writer(FoldDetails)  
    #details.writerow(['Classifier','kfold','Accuracy','FscoreMacro','FscoreWeighted','recall_macro','preci_macro','FPR','FNR','AUC'])
    
    for clf_name, clf in clf.items():
        Overall=open(Datapath+config+'//Results//ValidationSummary'+clf_name+'.csv','w+',encoding='utf-8',newline='')
        overall=csv.writer(Overall)
        overall.writerow(['Classifier','best_parameters','MCC','accuracy','bal_accu','precision','recall','f1','auc','fpr','fnr'])  
        bestmodel=train(urllisttr,X_train_transformed,Y,clf,clf_name,config,overall)
        print(clf_name,'   finished')
        Overall.close()
        pkl_filename = Datapath+config+'//Model//'+clf_name+'models.pkl'
        with open(pkl_filename, 'wb') as file:
                pickle.dump(bestmodel, file)
        file.close()
 #urlbased("CharngramsUrl_Results/",X_train,Y,start_n_gram,end_n_gram,'c')    
def lexicalexternalbased(config,X_train,Y):
       if(not os.path.isfile(Featurepath+'Basic_Lexical_External_Results/TrainFeatures_multiprocess.csv')):
           print('Creating Dataset')
           ExternalLexicalFeatureVectorGenerator(X_train,Y)  
       Featureset = pd.read_csv(Featurepath+'Basic_Lexical_External_Results/TrainFeatures_multiprocess.csv',encoding="utf-8") 
       Featureset=Featureset.fillna(-1)
       Featureset=Featureset.dropna()
       Featureset["country"] = ord_enc.fit_transform(Featureset[["country"]])
       label=Featureset['label']
       X_train=Featureset['url']
       Features=Featureset.drop(['url','label'], axis=1)
       Features=Features.replace('?',-1)
       Features=Features.replace('TRUE',1)
       Features=Features.replace('FALSE',0)
       classification(X_train,Features,label,config) 
def lexicalbased(config,X_train,Y):
       LexicalFeatureVectorGenerator(X_train,Y)  
       Featureset = pd.read_csv(Featurepath+'TrainFeatures.csv',encoding="utf-8") 
       label=Featureset['label']
       Features=Featureset.drop(['url','label'], axis=1)
       classification(X_train,Features,label,config) 
       
def urlbased(config,X_train,Y,st,ed,orignalflag=''):
    urllisttr=converter(X_train.tolist())
    pkl_filename =Featurepath+(config)+'v.pkl'
    if(orignalflag=='c'):
                vectorizer = extract_features(st, ed,orignalflag)
    else:
                vectorizer = extract_features( st, ed)
    X_train_transformed=vectorizer.fit_transform(urllisttr)
    voca=vectorizer.get_feature_names()
    file1 = open(Datapath+config+'voc.txt',"w")
    for v in voca:
        file1.write(v+'\n') 
    file1.write(str(len(voca)))
    file1.close()
    with open(pkl_filename, 'wb') as file:
                pickle.dump(vectorizer, file)
    file.close()
    X_train_transformed = X_train_transformed.astype(np.float64)
    
    classification(urllisttr,X_train_transformed,Y,config) 
   
def save(hostname,domainname,ext,tldomain,pathname):
    path=Featurepath+'//parts.pkl'
    with open(path, 'wb') as f:
            pickle.dump([hostname,domainname,ext,tldomain,pathname], f)
def get_parts(X_train):
    hostname=[]
    domainname=[]
    ext=[]
    tldomain=[]
    pathname=[]

    for url in X_train:
            token=urltokenizer(url)
           
            [ip,port,protocol,tld,(sld),userinfo,host,domain,(path),(parameter),(fragment),exe]=token
              
            hostname.append(host)
       
            domainname.append(domain)
        
            pathname.append(path)
        
            tldomain.append(tld)
       
            ext.append(exe)
    save(hostname,domainname,ext,tldomain,pathname)
    return hostname,domainname,ext,tldomain,pathname
def get_transformed_features(X_train,st,ed,org=''):
    if(org=='c'):
        vectorizer1 = extract_features(st, ed,org)
        vectorizer2 = extract_features(st, ed,org)
        vectorizer3 = extract_features(st, ed,org)
        vectorizer4 = extract_features(st, ed,org)
        vectorizer5 = extract_features(st, ed,org)
    else:
        vectorizer1 = extract_features(start_n_gram=st, end_n_gram=ed)
        vectorizer2 = extract_features(start_n_gram=st, end_n_gram=ed)
        vectorizer3 = extract_features(start_n_gram=st, end_n_gram=ed)
        vectorizer4 = extract_features(start_n_gram=st, end_n_gram=ed)
        vectorizer5 = extract_features(start_n_gram=st, end_n_gram=ed)
   
    path=Featurepath+'parts.pkl'
 
    if(os.path.isfile(path)):
            [hostname,domainname,ext,tldomain,pathname]= pickle.load(open(path, "rb" ))
    else:
	    [hostname,domainname,ext,tldomain,pathname]=get_parts(X_train)
    
    F1= (vectorizer1.fit_transform(hostname))
    F2= (vectorizer2.fit_transform(domainname))
    F3=(vectorizer3.fit_transform(ext))
    F4=(vectorizer4.fit_transform(tldomain))
    F5=(vectorizer5.fit_transform(pathname))
    
    FeatureVector= hstack([F1,F2,F3,F4,F5]).toarray()
    return FeatureVector,vectorizer1,vectorizer2,vectorizer3,vectorizer4,vectorizer5
#def fit_transformed_features(X_test,v1,v2,v3,v4,v5):
#    [hostname,domainname,ext,tldomain,pathname]=get_parts(X_test)
#    F11= v1.transform(hostname)
#    F22= v2.transform(domainname)
#    F33=v3.transform(ext)
#    F44=v4.transform(tldomain)
#    F55=v5.transform(pathname)
#    
#    FeatureVector= hstack([F11,F22,F33,F44,F55]).toarray()
#    return FeatureVector
    
def partbased(config,X_train,Y,st,ed,orignalflag=''):
    urllisttr=converter(X_train.tolist())
  
    [X_train_transformed,v1,v2,v3,v4,v5]=get_transformed_features(urllisttr,st,ed,orignalflag)
    pkl_filename1= Featurepath+(config)+'v1.pkl'
    pkl_filename2= Featurepath+(config)+'v2.pkl'
    pkl_filename3 =Featurepath+(config)+'v3.pkl'
    pkl_filename4 =Featurepath+(config)+'v4.pkl'
    pkl_filename5 = Featurepath+(config)+'v5.pkl'
    with open(pkl_filename1, 'wb') as file:
                pickle.dump(v1, file)
                with open(pkl_filename2, 'wb') as file2:
                    pickle.dump(v2, file2)
                with open(pkl_filename3, 'wb') as file3:
                    pickle.dump(v3, file3)
                with open(pkl_filename4, 'wb') as file4:
                    pickle.dump(v4, file4)
                with open(pkl_filename5, 'wb') as file5:
                    pickle.dump(v5, file5)
    file.close()
    file2.close()
    file3.close()
    file4.close()
    file5.close()
                
    filef = open(Datapath+config+'voca.txt',"w")
    voca1=v1.get_feature_names()
    for v in voca1:
        filef.write(v+'\n') 
    voca2=v2.get_feature_names()
    for v in voca2:
        filef.write(v+'\n') 
    voca3=v3.get_feature_names()
    for v in voca3:
        filef.write(v+'\n') 
    voca4=v4.get_feature_names()
    for v in voca4:
        filef.write(v+'\n')
    voca5=v5.get_feature_names()
    for v in voca5:
        filef.write(v+'\n') 
    filef.write(str((len(voca1)+len(voca2)+len(voca3)+len(voca4)+len(voca5))))
    filef.close()
    #X_train_transformed = X_train_transformed.astype(np.float64)
    
    classification(urllisttr,X_train_transformed,Y,config)

def AdvtrainingData():

    data_dir_path = OriginalDatapath
    Legitimate = (pd.read_csv(data_dir_path+'Legitimate/Leg_Training.csv',encoding="utf-8") )
    Malicious = pd.read_csv(data_dir_path+'Phishing/Phish_Training.csv',encoding="utf-8") 
    mal=len(Malicious)
    Leg=Legitimate.loc[0:(len(Malicious)-1),:]   
    A1 = pd.read_csv(Datapath+'DomainAdversary.csv',encoding="utf-8") 
    A2 = pd.read_csv(Datapath+'TLDAdversary.csv',encoding="utf-8") 
    A3 = pd.read_csv(Datapath+'PathAdversary.csv',encoding="utf-8") 
    AV1=A1.loc[0:10000,:]
    AV2=A2.loc[0:10000,:]
    AV3=A3.loc[0:10000,:]
    Mali=pd.concat([AV1, AV2, AV3])
    Mali=Mali['craftedurl']
    mali=len(Mali)
    frame1=(Leg['url'],Malicious['url'],Mali)
    url=np.concatenate(frame1).tolist()
    leg=len(Leg)
    print(leg)
    print(mal+mali)
    label0=np.zeros((leg))
    label1=np.ones((mal))
    label2=np.ones(mali)
    label=(np.concatenate((label0,label1,label2)))
    label=pd.Series(label.tolist())
    d={'url':url,'label':label}
    url_data=pd.DataFrame(d, columns=['url','label'])
    return url_data['url'],url_data['label']

#Basic Lexical Features

def selectConfiguration():
    #Orignal Data
   
    [X_train,Y]=AdvtrainingData()
   
    
    for i in range(0,6):
            config = i
            print("Current config:", config)
            
            if config==0:
                 print("Basic_Lexical_Results")
                 lexicalbased('Basic_Lexical_Results//',X_train,Y)  
            
            elif config==1:
                print("BoWUrl_Results")
                start_n_gram = 1
                end_n_gram = 1
                urlbased("BoWUrl_Results//",X_train,Y,start_n_gram,end_n_gram)
                   
            elif config == 3:
                print("CharngramsUrl_Results//")
                start_n_gram = 2
                end_n_gram = 8
                urlbased("CharngramsUrl_Results//",X_train,Y,start_n_gram,end_n_gram,'c')
          
            elif config == 2:
                print("BigramUrl_Results")
                start_n_gram = 1
                end_n_gram = 2
                urlbased("BigramUrl_Results//",X_train,Y,start_n_gram,end_n_gram)
                
            elif config == 4:
                print("Bag-of-word of url parts")
                start_n_gram = 1
                end_n_gram = 1
                partbased("BoWUrlParts_Results//",X_train,Y,start_n_gram,end_n_gram,'p')
            if config==5:
                 print("Basic_Lexical_External_Results")
                 lexicalexternalbased('Basic_Lexical_External_Results//',X_train,Y)  
                  
#            elif config == 5:
#                #print("BigramUrlParts_Results//")
#                start_n_gram = 1
#                end_n_gram = 2
#                partbased("BigramUrlParts_Results//",X_train,Y,start_n_gram,end_n_gram)
#            
#            elif config == 6:
#                #print("CharngramsParts_Results")
#                start_n_gram = 2
#                end_n_gram = 4
#                partbased("CharngramsParts_Results//",X_train,Y,start_n_gram,end_n_gram,'c')
#            
           
selectConfiguration()
