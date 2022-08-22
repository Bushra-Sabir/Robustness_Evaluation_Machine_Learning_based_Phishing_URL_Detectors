import configparser
import requests
import json

config = configparser.ConfigParser()
config.read('config.ini')

GoogleAPI_Key='AIzaSyDOrEtG4YUrx21wFhmK_D3YgkVxBlf5G0Y'
def google_safebrowsing(url):
    client_id = "Bushra"
    version = "v4"
    api_key = GoogleAPI_Key
    platform_types = ['ANY_PLATFORM']
    threat_types = ['THREAT_TYPE_UNSPECIFIED',
                    'MALWARE', 'SOCIAL_ENGINEERING',
                    'UNWANTED_SOFTWARE', 'POTENTIALLY_HARMFUL_APPLICATION']
    threat_entry_types = ['URL']
    api_url = 'https://safebrowsing.googleapis.com/v4/threatMatches:find?key=%s' % (api_key)
    threat_entries = [{'url': url}]
    payload = {
        'client': {
            'clientId': client_id,
            'clientVersion': version
        },
        'threatInfo': {
            'threatTypes': threat_types,
            'platformTypes': platform_types,
            'threatEntryTypes': threat_entry_types,
            'threatEntries': threat_entries
        }
    }
    headers = {'content-type': 'application/json'}
    try:
        response = requests.post(api_url, headers=headers, json=payload).json().get('matches', None)
        if response is not None:
            return True
        else:
            return False
    except Exception:
        return '?'


def phishtank(url):
    with open('/hpcfs/users/a1735399/Final/Publish_Code/Code/Model_development/lib/files/database_phishtank.json') as db:
        data = json.load(db)
    for d in data:
        if (url == d['url']):
            return True
    return False


