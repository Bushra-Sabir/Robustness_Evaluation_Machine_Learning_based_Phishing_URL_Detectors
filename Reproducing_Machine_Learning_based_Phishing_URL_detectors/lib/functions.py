# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 12:07:31 2021

@author: bushra
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 14:54:19 2021

@author: bushra
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 12:41:37 2021

@author: bushra
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 11:10:31 2021

@author: bushra
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 12:29:38 2021

@author: bushra
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 15:09:33 2021

@author: bushra
"""

from urllib import parse
from dns import resolver, reversename
from bs4 import BeautifulSoup
from rblwatch import RBLSearch
from .spf import get_spf_record, check_spf
from .blacklists import google_safebrowsing, phishtank
import re
import ipaddress
import requests
import geoip2.database
import whois
PATH = '/hpcfs/users/a1735399/Final/Publish_Code/Code/Model_development/adv/lib/files/'
import codecs,re
import tldextract
from datetime import date
codecs.register_error("strict", codecs.ignore_errors)
#import tldextract


def start_url(url):
    """Split URL into: protocol, host, path, params, query and fragment."""
    try: 
        if not parse.urlparse(url.strip()).scheme:
            url = 'http://' + url
        protocol, host, path, params, query, fragment = parse.urlparse(url.strip())
    
        result = {
            'url': host + path + params + query + fragment,
            'protocol': protocol,
            'host': host,
            'path': path,
            'params': params,
            'query': query,
            'fragment': fragment
        }
        
    except:
        try:
            [sld,domain,tld]=tldextract.extract(url)
            host=domain+'.'+tld
        except:
            host=url
        result={'url': url,
                'protocol':'http',
                'host':host,
                'path': '',
                'params': '',
                'query': '',
                'fragment': ''
                }

    return result

def count(text, character):
    """Return the amount of certain character in the text."""
    return text.count(character)


def count_vowels(text):
    """Return the number of vowels."""
    vowels = ['a', 'e', 'i', 'o', 'u']
    count = 0
    for i in vowels:
        count += text.lower().count(i)
    return count


def length(text):
    """Return the length of a string."""
    return len(text)


def valid_ip(host):
    """Return if the domain has a valid IP format (IPv4 or IPv6)."""
    try:
        ipaddress.ip_address(host)
        return 1
    except Exception:
        return 0


def valid_email(text):
    """Return if there is an email in the text."""
    if re.findall(r'[\w\.-]+@[\w\.-]+', text):
        return 1
    else:
        return 0


def check_shortener(url):
    """Check if the domain is a shortener."""
    file = open(PATH + 'shorteners.txt', 'r')
    for line in file:
        with_www = "www." + line.strip()
        if line.strip() == url['host'].lower() or with_www == url['host'].lower():
            file.close()
            return 1
    file.close()
    return 0


def check_tld(text):
    """Check for presence of Top-Level Domains (TLD)."""
    file = open(PATH + 'tlds.txt', 'r')
    pattern = re.compile("[a-zA-Z0-9.]")
    for line in file:
        i = (text.lower().strip()).find(line.strip())
        while i > -1:
            if ((i + len(line) - 1) >= len(text)) or not pattern.match(text[i + len(line) - 1]):
                file.close()
                return 1
            i = text.find(line.strip(), i + 1)
    file.close()
    return 0


def count_tld(text):
    """Return amount of Top-Level Domains (TLD) present in the URL."""
    file = open(PATH + 'tlds.txt', 'r')
    count = 0
    pattern = re.compile("[a-zA-Z0-9.]")
    for line in file:
        i = (text.lower().strip()).find(line.strip())
        while i > -1:
            if ((i + len(line) - 1) >= len(text)) or not pattern.match(text[i + len(line) - 1]):
                count += 1
            i = text.find(line.strip(), i + 1)
    file.close()
    return count


def count_params(text):
    """Return number of parameters."""
    return len(parse.parse_qs(text))


def check_word_server_client(text):
    """Return whether the "server" or "client" keywords exist in the domain."""
    if "server" in text.lower() or "client" in text.lower():
        return 1
    return 0


def count_ips(url):
    """Return the number of resolved IPs (IPv4)."""
    if valid_ip(url['host']):
        return 1

    try:
        answers = resolver.query(url['host'], 'A')
        return len(answers)
    except Exception:
        return -1


def count_name_servers(url):
    """Return number of NameServers (NS) resolved."""
    count = 0
    if count_ips(url):
        try:
            answers = resolver.query(url['host'], 'NS')
            return len(answers)
        except (resolver.NoAnswer, resolver.NXDOMAIN):
            split_host = url['host'].split('.')
            while len(split_host) > 0:
                split_host.pop(0)
                supposed_domain = '.'.join(split_host)
                try:
                    answers = resolver.query(supposed_domain, 'NS')
                    count = len(answers)
                    break
                except Exception:
                    count = 0
        except Exception:
            count = 0
    return count


def count_mx_servers(url):
    """Return Number of Resolved MX Servers."""
    count = 0
    if count_ips(url):
        try:
            answers = resolver.query(url['host'], 'MX')
            return len(answers)
        except (resolver.NoAnswer, resolver.NXDOMAIN):
            split_host = url['host'].split('.')
            while len(split_host) > 0:
                split_host.pop(0)
                supposed_domain = '.'.join(split_host)
                try:
                    answers = resolver.query(supposed_domain, 'MX')
                    count = len(answers)
                    break
                except Exception:
                    count = 0
        except Exception:
            count = 0
    return count


def extract_ttl(url):
    """Return Time-to-live (TTL) value associated with hostname."""
    try:
        ttl = resolver.query(url['host']).rrset.ttl
        return ttl
    except Exception:
        return -1
from datetime import datetime
def d(s):
  [month, day, year] = map(int, s.split('/'))
  return date(year, month, day)
def days(start, end):
  return abs(d(end) - d(start)).days


def time_activation_domain(url):
    """Return time (in days) of domain activation."""
    try:
            [sld,domain,tld]=tldextract.extract(url)
            host=domain+'.'+tld
    except:
            host=url
        
    try:
        result_whois = whois.whois(host)
        if not result_whois:
           # print('not here',result_whois)
            return -1
        try:
            if(len(result_whois['creation_date'])>1):
              creation_date = result_whois['creation_date'][0].date()
              d1 = creation_date.strftime("%m/%d/%Y")#datetime.strptime(formated_date, "%Y-%m-%d")
              d2 = datetime.today().strftime("%m/%d/%Y")
              print(d1)
              print(d2)
              return days(d1,d2)
        except:
            print('here')
            creation_date = result_whois['creation_date'].date()
            d1 = creation_date.strftime("%m/%d/%Y")#datetime.strptime(formated_date, "%Y-%m-%d")
            d2 = datetime.today().strftime("%m/%d/%Y")
            print(d1)
            print(d2)
            return days(d1,d2)
        #formated_date = " ".join(creation_date.split()[:1])
        
    
    except Exception:
        print('not here exception')
        return -1


def expiration_date_register(url):
    """Retorna time (in days) for register expiration."""
    try:
            [sld,domain,tld]=tldextract.extract(url)
            host=domain+'.'+tld
    except:
            host=url
        
    try:
        result_whois = whois.whois(host)
        if not result_whois:
            #print('not here',result_whois)
            return -1
        try:
            if (len(result_whois['expiration_date'])>1):
              expiration_date = result_whois['expiration_date'][0].date()
              #print('len is more than 1',expiration_date,type(expiration_date))
              d1 = expiration_date.strftime("%m/%d/%Y")#datetime.strptime(formated_date, "%Y-%m-%d")
              d2 = datetime.today().strftime("%m/%d/%Y")
              print(d1)
              print(d2)
              return days(d1,d2)
            
        except:
            expiration_date = result_whois['expiration_date'].date()
            #print('len is less than 1', expiration_date)
            d1 = expiration_date.strftime("%m/%d/%Y")#datetime.strptime(formated_date, "%Y-%m-%d")
            d2 = datetime.today().strftime("%m/%d/%Y")
            print(d1)
            print(d2)
            return days(d1,d2)
    except Exception:
        print('not here exception')
        return -1

#

def extract_extension(text):
    """Return file extension name."""
    file = open(PATH + 'extensions.txt', 'r')
    pattern = re.compile("[a-zA-Z0-9.]")
    for extension in file:
        i = (text.lower().strip()).find(extension.strip())
        while i > -1:
            if ((i + len(extension) - 1) >= len(text)) or not pattern.match(text[i + len(extension) - 1]):
                file.close()
                return extensions(extension.rstrip().split('.')[-1])
            i = text.find(extension.strip(), i + 1)
    file.close()
    return '100'


def check_ssl(url):
    """Check if the ssl certificate is valid."""
    try:
        requests.get(url, verify=True, timeout=3)
        return 1
    except Exception:
        return 0

def extensions(exe):
    exes=-1
    extension=['php','txt','js','html','htm','cgi','aspx','exe','jpg','png','dll','com','gif','zip','jar','bin','pdf','c','lua','rar','css','asp','xhtml',
'jsp',
'h',
'doc',
'pl',
'cfm',
'swf',
'sys',
'py',
'docx',
'xml',
'svg',
'm',
'torrent',
'dat',
'dds',
'rss',
'tmp',
'cpp'
]
    for e in exe:
       if(e in extension):
           exes=extension.index(e)
       else:
           exes=100 #not given
    return exes
def count_redirects(url):
    """Return the number of redirects in a URL."""
    try:
        response = requests.get(url, timeout=3)
        if response.history:
            return len(response.history)
        else:
            return 0
    except Exception:
        return 100


def get_asn_number(url):
    """Return the ANS number associated with the IP."""
    try:
        with geoip2.database.Reader(PATH + 'GeoLite2-ASN.mmdb') as reader:
            if valid_ip(url['host']):
                ip = url['host']
            else:
                ip = resolver.query(url['host'], 'A')
                ip = ip[0].to_text()

            if ip:
                response = reader.asn(ip)
                return response.autonomous_system_number
            else:
                return -1
    except Exception:
        return -1


def get_country(url):
    """Return the country associated with IP."""
    try:
        if valid_ip(url['host']):
            ip = url['host']
        else:
            ip = resolver.query(url['host'], 'A')
            ip = ip[0].to_text()

        if ip:
            reader = geoip2.database.Reader(PATH + 'GeoLite2-Country.mmdb')
            response = reader.country(ip)
            return response.country.iso_code
        else:
            return -1
    except Exception:
        return -1


def get_ptr(url):
    """Return PTR associated with IP."""
    try:
        if valid_ip(url['host']):
            ip = url['host']
        else:
            ip = resolver.query(url['host'], 'A')
            ip = ip[0].to_text()

        if ip:
            r = reversename.from_address(ip)
            result = resolver.query(r, 'PTR')[0].to_text()
            return result
        else:
            return -1
    except Exception:
        return -1


def google_search(url):
    """Check if the url is indexed in google."""
    user_agent = 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/48.0.2564.116 Safari/537.36'
    headers = {'User-Agent': user_agent}

    query = {'q': 'info:' + url}
    google = "https://www.google.com/search?" + parse.urlencode(query)
    try:
        data = requests.get(google, headers=headers)
    except Exception:
        return 100
    data.encoding = 'ISO-8859-1'
    soup = BeautifulSoup(str(data.content), "html.parser")
    try:
        (soup.find(id="rso").find(
            "div").find("div").find("h3").find("a"))['href']
        return 1
    except AttributeError:
        return 0


def valid_spf(domain):
    """Check if within the registered domain has SPF and if it is valid."""
    spf = get_spf_record(domain)
    if spf is not None:
        return check_spf(spf, domain)
    return 0


def check_blacklists(url):
    """Check if the URL or Domain is malicious through Google Safebrowsing, Phishtank, and WOT."""
    if (google_safebrowsing(url) or phishtank(url)):
        return 1
    return 0


def check_blacklists_ip(url):
    """Check if the IP is malicious through Google Safebrowsing, Phishtank and WOT."""
    try:
        
            ip = resolver.query(url)
            ip = ip[0].to_text()

            if ip:
                if (google_safebrowsing(ip) or phishtank(ip)):
                    return True
                return False
            else:
                return -1
    except Exception:
        return -1


def check_rbl(domain):
    """Check domain presence on RBL (Real-time Blackhole List)."""
    searcher = RBLSearch(domain)
    try:
        listed = searcher.listed
    except Exception:
        return 0
    for key in listed:
        if key == 'SEARCH_HOST':
            pass
        elif listed[key]['LISTED']:
            return 1
    return 0


def check_time_response(domain):
    """Return the response time in seconds."""
    try:
        latency = requests.get(domain, headers={'Cache-Control': 'no-cache'}).elapsed.total_seconds()
        return latency
    except Exception:
        return 0


def read_file(archive):
    """Read the file with the URLs."""
    with open(archive, 'r') as f:
        urls = ([line.rstrip() for line in f])
        return urls