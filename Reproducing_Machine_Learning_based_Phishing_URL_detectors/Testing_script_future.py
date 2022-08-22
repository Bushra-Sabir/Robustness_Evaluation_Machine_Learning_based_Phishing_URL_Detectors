"""
Created on Wed Jun 23 10:38:08 2021

@author: bushra
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 11:58:11 2021

@author: bushra
"""

from sklearn import metrics
Datapath="Path to Data, Models, and Results"
Featurepath="Path to features"
import validators
from rfc3986 import is_valid_uri
import random
import validators
import re,tldextract
import pandas as pd
from rfc3986 import urlparse
from rfc3986 import is_valid_uri
#from Levenshtein import distance
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
from difflib import SequenceMatcher
import pickle
import numpy as np
import matplotlib.pyplot as plt
import sklearn.ensemble._forest
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
#from sklearn.utils.fixes import signature
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
#words = set(brown.words())
pathmalfilespatterns='F:/PrimaryStudy1/Code/outputs/ngrams/' 
import pandas as pd
import csv
from Feature_extraction import feature_extract
from lib.functions import *
import itertools



# Import libraries
import pickle ,os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import time
from collections import Counter

from Feature_extraction import feature_extract
from lib.functions import *

import tldextract
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score,matthews_corrcoef

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
from sklearn.metrics import precision_recall_curve,roc_auc_score
from sklearn.multiclass import OneVsRestClassifier
from lightgbm import LGBMClassifier
#from sklearn.utils.fixes import signature
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
 # -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 14:28:29 2019

@author: bushra
"""
import random
import validators
from urllib.parse import urlunparse
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
import codecs,re
codecs.register_error("strict", codecs.ignore_errors)

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

from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec


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
from ekphrasis.classes.segmenter import Segmenter
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
        return 0.0 
    else :            
        return consonants/vowels
def countcharacters(url, character):
    count = url.count(character)
    return count
def numberofdigits(url):
    urlparsed = urlparse(url)
    digit=0
    for u in url:
        if(u.isnumeric()):
            digit+=1
    if(urlparsed.host==None):
        [sld,domain,tld]=tldextract.extract(url)
        host=domain+'.'+tld
    else:
        host=urlparsed.host
    if(len(host)!=0):
        return digit/len(host)
    else:
        return 0.0


def listbasedfeatures(url):
    dic=dict({'Total_delimitors':'','Known_TLD':'','Suspicious_word_in_path':'','Exe_in_url':'','Consonants_ratio_vowels':'','Numberofdigits':'0','consonants_ratio_urllength':'0'})
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
import signal
from contextlib import contextmanager


@contextmanager
def timeout(time):
    # Register a function to raise a TimeoutError on the signal.
    signal.signal(signal.SIGALRM, raise_timeout)
    # Schedule the signal to be sent after ``time``.
    signal.alarm(time)

    try:
        yield
    except TimeoutError:
        pass
    finally:
        # Unregister the signal so it won't be triggered
        # if the timeout is not reached.
        signal.signal(signal.SIGALRM, signal.SIG_IGN)


def raise_timeout(signum, frame):
    raise TimeoutError
    

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
import time 
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

def LexicalFeatureVectorGenerator(filename,urllist,labellist):
    if (not os.path.exists(filename)):
        filetowrite=open(filename, 'w' ,encoding='utf-8',newline='') 
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
def ExternalLexicalFeatureVectorGenerator(filename,urllist,labellist):
           
        filetowrite=open(filename, 'w+' ,encoding='utf-8',newline='') 
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
"""
Created on Thu Sep  5 12:52:44 2019

@author: bushra
"""


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
#           words.append(w.split('_'))
#           words.remove(w)
           try:
               subwords=(seg_eng.segment(w))
           except:
               subwords=w
           finalwords.append(subwords.split(' '))
    return (finalwords)

def obtainwordlist(url):
    words=re.findall(r'\w+\b', url)
    
    finalwords=tokenparts(words)
    final=converter(finalwords)
    
    return final   
#def extract_features(start_n_gram, end_n_gram,flag=''):
#        if(flag=='c'):
#           return feature_extraction.text.CountVectorizer(ngram_range=(start_n_gram, end_n_gram),min_df=0.001,analyzer='char')
#        
#        else:
#            return feature_extraction.text.CountVectorizer(analyzer='word', ngram_range=(start_n_gram, end_n_gram),tokenizer=obtainwordlist)
## Build the Classifiers
def extract_features(start_n_gram, end_n_gram,flag=''):
        if(flag=='c'):
           return feature_extraction.text.CountVectorizer(ngram_range=(start_n_gram, end_n_gram),min_df=0.001,analyzer='char')
        #elif(flag=='p'):
         #  return feature_extraction.text.CountVectorizer(analyzer='word',ngram_range=(start_n_gram, end_n_gram),min_df=0.001,tokenizer=obtainwordlist)
        
        else:
            return feature_extraction.text.CountVectorizer(analyzer='word', ngram_range=(start_n_gram, end_n_gram),tokenizer=obtainwordlist)
# Build the Classifiers

def build_classifiers():
   
    clfs = {
            
            'RF': RandomForestClassifier(n_estimators=100,max_depth=None, max_leaf_nodes=None, random_state=42,
                                         n_jobs=-1),
            'DT':DecisionTreeClassifier(random_state=42),
            
            'SVM': OneVsRestClassifier(LinearSVC(random_state=42, C=0.1), n_jobs=-1),
            'LR': LogisticRegression(C=0.1, multi_class='ovr', n_jobs=-1, solver='lbfgs', max_iter=3000,
                                     random_state=42),
                   
            'XGB': XGBClassifier(max_depth=0, max_leaves=100, grow_policy='lossguide',
                                 n_jobs=-1, random_state=42, tree_method='hist'),
            'LGBM': LGBMClassifier(n_estimator=100,num_leaves=100, max_depth=-1, n_jobs=-1, random_state=42),
            'KNN': KNeighborsClassifier()
           
            }

    return clfs
  

def evaluate(clf,  X_test,  y_test):
    
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
    
    Auc=roc_auc_score(y_test, y_pred)   
    MCC= matthews_corrcoef (y_test,y_pred) 
          
    return y_pred,accuracy_score(y_test, y_pred), f1_score(y_test, y_pred, average='macro'),\
           f1_score(y_test, y_pred, average='weighted'),recall_score(y_test, y_pred, average='macro'), precision_score(y_test, y_pred, average='macro'),FPR,FNR ,Auc,MCC
#train(urllisttr,X_train_transformed,Y,clf,clf_name,config)
def train(urllist,X,y,clf,clf_name,config,detail,overall):
        accs = []
        f_mac = []
        f_weighted = []
        recall_macro=[]
        preci_macro=[]
        FPR=[]
        FNR=[]
        MCC=[]
        AUC=[]
        kf = StratifiedKFold(n_splits=10,random_state=42,shuffle=True)
        count=1
       
        for train_index, test_index in kf.split(X, y):
           # print("TRAIN:", train_index, "TEST:", test_index)
            if(config=='Basic_Lexical_External_Results'):
                    
                  X_train, X_test = X.loc[train_index], X.loc[test_index]
                  y_train, y_test = y.loc[train_index], y.loc[test_index]
            else:
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
            clf.fit(X_train, y_train)
            results = evaluate(clf, X_test, y_test)
            print(results)
            accs.append(results[1])
            f_mac.append(results[2])
            f_weighted.append(results[3])
            recall_macro.append(results[4])
            preci_macro.append(results[5])
            FPR.append(results[6])
            FNR.append(results[7])
            AUC.append(results[8])
            MCC.append(results[9])
            #['Classifier','kfold','Accuracy','FscoreMacro','FscoreWeighted','recall_macro','preci_macro','FPR','FNR']
            detail.writerow([clf_name,count,results[1],results[2],results[3],results[4],results[5],results[6],results[7],results[8],results[9]])
            count+=1
        row=[]
        row.append(clf_name)
        row.append(np.mean(np.asarray(accs)))
        row.append(np.mean( np.asarray(f_mac)))
        row.append(np.mean(np.asarray(f_weighted)))
        row.append(np.mean(np.asarray(recall_macro)))
        row.append(np.mean(np.asarray(preci_macro)))
        row.append(np.mean(np.asarray(FPR)))
        row.append(np.mean(np.asarray(FNR)))
        row.append(np.mean(np.asarray(AUC)))
        row.append(np.mean(np.asarray(MCC)))
        overall.writerow(row)
        return clf  
#    classification(urllisttr,X_train_transformed,Y,config)

def classification(urllisttr,X_train_transformed,Y,config):
    modelslist=[]
    clf=build_classifiers()
    FoldDetails=open(Datapath+config+'//Results//K_foldResults.csv', 'w' ,encoding='utf-8',newline='')
    details = csv.writer(FoldDetails)  
    details.writerow(['Classifier','kfold','Accuracy','FscoreMacro','FscoreWeighted','recall_macro','preci_macro','FPR','FNR','AUC'])
    Overall=open(Datapath+config+'//Results//ValidationSummary.csv','w',encoding='utf-8',newline='')
    overall = csv.writer(Overall)  
    overall.writerow(['Classifier','Accuracy','FscoreMacro','FscoreWeighted','recall_macro','preci_macro','FPR','FNR','AUC'])
    for clf_name, clf in clf.items():
         
        train(urllisttr,X_train_transformed,Y,clf,clf_name,config,details,overall)
        modelslist.append(clf)
        
    pkl_filename = Datapath+config+'//Model//models.pkl'
    with open(pkl_filename, 'wb') as file:
                pickle.dump(modelslist, file)
    Overall.close()
    FoldDetails.close()
def write_details(X_train,y_pred,y_actual,clf_name,config,atype,typee,method):
    #miss classified examples
    method=list(method)
    typee=list(typee)
    X_train=list(X_train)
    y_pred=list(y_pred)
    y_actual=list(y_actual)
    filetowrite=open(Datapath+atype+'//'+config+'//Results//Details//'+clf_name+'Original_Models_AdversarialTesting_Details.csv','w',encoding='utf-8',newline='')
    writer = csv.writer(filetowrite)  
    writer.writerow(['urls','type','method','classification'])
    for j,example in enumerate(y_actual):
                if(y_pred[j]!=y_actual[j]):
                    writer.writerow([X_train[j],typee[j],method[j],str(y_actual[j])+
                                     ' found other wise'])
                else:
                    writer.writerow([X_train[j],typee[j],method[j],'correct'])
    filetowrite.close()               
#test(atype,urllistte,X_test_transformed,Y_test,config,typee,method)                 
def test(atype,urllistte,X_test_transformed,Y_test,config,typee,method):
    Overall=open(Datapath+atype+'//'+config+'//Results//Original_Models_AdversarialTesting_Summary.csv','a+',encoding='utf-8',newline='')
    overall = csv.writer(Overall)  
    overall.writerow(['Classifier','Accuracy','FscoreMacro','FscoreWeighted','recall_macro','preci_macro','FPR','FNR','AUC','MCC'])
       
    clf_name=['rf','dt','lr','svm','knn','lgbm','xgb']
    for clf in clf_name:
        if(clf=='lgbm'):
            X_test_transformed=X_test_transformed._get_numeric_data()
            Y_test=np.ravel(Y_test, order='C')
        pkl_filename = Datapath+config+'//Model//'+clf+'models.pkl'
        if(os.path.isfile(pkl_filename)):
                  model= pickle.load(open(pkl_filename, "rb" ))
                  print(clf)
                  results=evaluate(model,X_test_transformed,Y_test)
                  write_details(urllistte,results[0],Y_test,clf,config,atype,typee,method)
                  overall.writerow([clf,results[1],results[2],results[3],results[4],results[5],results[6],results[7],results[8],results[9]])
    Overall.close()
   

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
 #urlbased("CharngramsUrl_Results/",X_train,Y,start_n_gram,end_n_gram,'c') 
# lexicalbased(atype,'Basic_Lexical_Results/',X_test,Y_test,typee,method)
from sklearn.preprocessing import OrdinalEncoder

ord_enc = OrdinalEncoder()

def lexicalbased(atype,config,X_test,Y,typee,method):
       flag='test'
       filename=Featurepath+'Basic_Lexical_Results//'+flag+atype+'Features.csv'
       LexicalFeatureVectorGenerator(filename,X_test,Y,filename)  
       Featureset = pd.read_csv(filename,encoding="utf-8") 
       label=Featureset['label']
       Features=Featureset.drop(['url','label'], axis=1)
       Features=Features.fillna('0')
       test(atype,X_test,Features,label,config,typee,method) 
  
def lexicalexternalbased(atype,config,X_test,Y,typee,method):
       flag='test'
       global filename
       filename=Featurepath+'Basic_Lexical_External_Results/'+flag+atype+'Features.csv'
       ExternalLexicalFeatureVectorGenerator(filename,X_test,Y)  
       Featureset = pd.read_csv(filename,encoding="utf-8") 
       Featureset=Featureset.fillna(-1)
       Featureset=Featureset.dropna()
       Featureset["country"] = ord_enc.fit_transform(Featureset[["country"]])
       label=Featureset['label']
       X_train=Featureset['url']
       Features=Featureset.drop(['url','label'], axis=1)
       Features=Features.replace('?',-1)
       Features=Features.replace('TRUE',1)
       Features=Features.replace('True',1)
       Features=Features.replace('true',1)
       Features=Features.replace('FALSE',0)
       Features=Features.replace('False',0)
       Features=Features.replace('false',0)
       print("shape of features: ", np.shape(Features))
       test(atype,X_test,Features,label,config,typee,method) 
def get_oov(vectorizer,X,config,atype,st,ed,orignalflag=''):
    trainingvoc=set(vectorizer.vocabulary_.keys()) 
    vectorizer1=extract_features(st, ed,orignalflag)
    if(X!=[] and set(X)!={''}):
        X_train_transformed=vectorizer1.fit_transform((X))
        testingvoc=set(vectorizer1.vocabulary_.keys())
        diffference=trainingvoc-testingvoc
        voca=vectorizer1.get_feature_names()
        count_list = X_train_transformed.toarray().sum(axis=0)   
        dictionary=dict(zip(voca,count_list))
        
        file1 = open('/hpcfs/users/a1735399/Final/Publish_Code/Code/'+atype+'//'+config+'testvoc.txt',"w")
        for v in dictionary:
             file1.write(v+' '+str(dictionary[v])+'\n') 
        file1.write(str(len(voca)))
        file1.close()
        file2 = open('/hpcfs/users/a1735399/Final/Publish_Code/Code/'+atype+'//'+config+'diffvoc.txt',"w")
        for v in diffference:
            file2.write(str(v)+'\n') 
        file2.write(str(len(diffference)))
        file2.close()
    
#urlbased(atype,"BigramUrl_Results/",X_test,Y_test,typee,method,start_n_gram,end_n_gram)
                
def urlbased(atype,config,X_test,Y_test,typee,method,st,end,originalflag=''):
    urllistte=converter(X_test.tolist())
    pkl_filename =Featurepath+(config)+'v.pkl'
    if(os.path.isfile(pkl_filename)):
           vectorizer= pickle.load(open(pkl_filename, "rb" ))
           get_oov(vectorizer,urllistte,config,atype,st,end,originalflag)
           X_test_transformed=vectorizer.transform(urllistte)
           X_test_transformed = X_test_transformed.astype(np.float64)
           test(atype,urllistte,X_test_transformed,Y_test,config,typee,method) 
   

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
    [hostname,domainname,ext,tldomain,pathname]=get_parts(X_train)
    
    F1= (vectorizer1.fit_transform(hostname))
    F2= (vectorizer2.fit_transform(domainname))
    F3=(vectorizer3.fit_transform(ext))
    F4=(vectorizer4.fit_transform(tldomain))
    F5=(vectorizer5.fit_transform(pathname))
    
    FeatureVector= hstack([F1,F2,F3,F4,F5]).toarray()
    return FeatureVector,vectorizer1,vectorizer2,vectorizer3,vectorizer4,vectorizer5

def fit_transformed_features(X_test,v1,v2,v3,v4,v5,config,atype,st,ed,orignalflag=''):
    [hostname,domainname,ext,tldomain,pathname]=get_parts(X_test)
    # get_oov(vectorizer,urllistte,config,atype,st,end,originalflag)
    get_oov(v1,hostname,config,atype,st,ed,orignalflag)
    get_oov(v2,domainname,config,atype,st,ed,orignalflag)
    get_oov(v3,ext,config,atype,st,ed,orignalflag)
    get_oov(v4,tldomain,config,atype,st,ed,orignalflag)
    get_oov(v5,pathname,config,atype,st,ed,orignalflag)
    F11= v1.transform(hostname)
    F22= v2.transform(domainname)
    F33=v3.transform(ext)
    F44=v4.transform(tldomain)
    F55=v5.transform(pathname)
    FeatureVector= hstack([F11,F22,F33,F44,F55]).toarray()
    return FeatureVector
  #partbased(atype,"BigramUrlParts_Results/",X_test,Y_test,typee,method,start_n_gram,end_n_gram)
         
def partbased(atype,config,X_test,Y_test,typee,method,st,ed,orignalflag=''):
    urllistte=converter(X_test.tolist())
    pkl_filename1= Featurepath+(config)+'v1.pkl'
    pkl_filename2= Featurepath+(config)+'v2.pkl'
    pkl_filename3 =Featurepath+(config)+'v3.pkl'
    pkl_filename4 =Featurepath+(config)+'v4.pkl'
    pkl_filename5 = Featurepath+(config)+'v5.pkl'
    if(os.path.isfile(pkl_filename1)):
           v1= pickle.load(open(pkl_filename1, "rb" ))
    if(os.path.isfile(pkl_filename2)):
           v2= pickle.load(open(pkl_filename2, "rb" ))
    if(os.path.isfile(pkl_filename3)):
            v3= pickle.load(open(pkl_filename3, "rb" ))
    if(os.path.isfile(pkl_filename4)):    
            v4= pickle.load(open(pkl_filename4, "rb" ))
    if(os.path.isfile(pkl_filename5)): 
            v5= pickle.load(open(pkl_filename5, "rb" ))
  
    X_test_transformed= fit_transformed_features(X_test,v1,v2,v3,v4,v5,config,atype,st,ed,orignalflag)
    Y_test=np.ones(len(X_test_transformed))
    test(atype,urllistte,X_test_transformed,Y_test,config,typee,method)
from sklearn.utils import shuffle


def testFutureData(atype):
    
    if(atype=='All_Adversary'):
        data_dir_path = 'path to adversarial dataset'+'/Group_Future_Targets//Test_'
        A1 = pd.read_csv(data_dir_path+'DomainAdversary.csv',encoding="utf-8") 
        A2 = pd.read_csv(data_dir_path+'TLDAdversary.csv',encoding="utf-8") 
        A3 = pd.read_csv(data_dir_path+'PathAdversary.csv',encoding="utf-8") 
        #A4=pd.read_csv(data_dir_path+'dmswap.csv',encoding="utf-8") 

        
        A3=A3.loc[0:len(A1),:]
        #AV4=A4
        Mali=pd.concat([A1, A2, A3]).reset_index() 
        
         
    else:
        data_dir_path = 'path to adversarial dataset'+'Group_Future_Targets//Test_'+atype
        Malicious = pd.read_csv(data_dir_path+'.csv',encoding="utf-8")
        if(atype=='PathAdversary'):
             Malicious = shuffle(Malicious)
             Malicious=Malicious.reset_index() 
             Mali=Malicious.loc[0:50000,:]
        else:
            Mali=Malicious
    print(Mali.columns)
    DeceptiveURLs=Mali['craftedurl']
    BenignURLs=Mali['seedurl'].drop_duplicates()    
    labeld=np.ones(len(DeceptiveURLs))
    labelb=np.zeros(len(BenignURLs))
    print("benign",labelb)
    frame1=([DeceptiveURLs,BenignURLs])
    url=np.concatenate(frame1).tolist()
    label=(np.concatenate((labeld,labelb)))
    label=pd.Series(label.tolist())
    d={'url':url,'label':label,'type':Mali['adversarytype'],'method':Mali['adversarymethod']}
    url_data=pd.DataFrame(d, columns=['url','label','type','method'])
    print(len(url_data))
    return url_data['url'],url_data['label'],url_data['type'],url_data['method']


#Basic Lexical Features
import posixpath
    
  for j in range(0,4):
        if(j==1):
          atype='DomainAdversary'          
        if j==2:
          atype='TLDAdversary'
        if j==3:
         atype='PathAdversary'
        if j==0:
          atype='All_Adversary'
        print(atype)
       
        [X_test,Y_test,typee,method]=testFutureData(atype)
        for i in range(0,6):
            config = i
            print("Current config:", config)
            if config==0:
                 print("Basic_Lexical_Results")
                 lexicalbased(atype,'Basic_Lexical_Results//',X_test,Y_test,typee,method)

            elif config==1:
                print("BoWUrl_Results")
                start_n_gram = 1
                end_n_gram = 1
                urlbased(atype,"BoWUrl_Results//",X_test,Y_test,typee,method,start_n_gram,end_n_gram)
                   
            elif config == 2:
                print("BigramUrl_Results")
                start_n_gram = 1
                end_n_gram = 2
                urlbased(atype,"BigramUrl_Results//",X_test,Y_test,typee,method,start_n_gram,end_n_gram)
           elif config == 3:
                print("CharngramsUrl_Results/")
                start_n_gram = 2
                end_n_gram = 8
                urlbased(atype,"CharngramsUrl_Results//",X_test,Y_test,typee,method,start_n_gram,end_n_gram,'c')
            elif config == 4:
                print("Bag-of-word of url parts")
                start_n_gram = 1
                end_n_gram = 1
                partbased(atype,"BoWUrlParts_Results//",X_test,Y_test,typee,method,start_n_gram,end_n_gram)
            elif config==5:
                 print("Basic_Lexical_External_Results")
                 lexicalexternalbased(atype,'Basic_Lexical_External_Results//',X_test,Y_test,typee,method)

selectConfiguration()