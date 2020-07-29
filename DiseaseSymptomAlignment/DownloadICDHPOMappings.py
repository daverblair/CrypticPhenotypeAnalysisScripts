#!/usr/bin/env python
# coding: utf-8

"""
This script downloads and parses the mappings between the HPO and ICD10/9 from the BioPortal.
Note, these mappings were produced with lexical matching tool LOOM.
"""



import urllib.request, urllib.error, urllib.parse
import json
import os
from pprint import pprint

#must provide your own access key
API_KEY = ""

def get_json(url):
    opener = urllib.request.build_opener()
    opener.addheaders = [('Authorization', 'apikey token=' + API_KEY)]
    return json.loads(opener.open(url).read())

# Get the available resources
current_URL = "http://data.bioontology.org/mappings?ontologies=HP,ICD10CM"
f=open('RawDataFiles/HP_ICD10_BioOnt.txt','w')
f.write('HP Term\tICD10 Term\n')
while True:
    currentMapping = get_json(current_URL)
    #first obtain and parse the HPO to ICD10

    for mapping in currentMapping['collection']:
        ICD_ID = mapping['classes'][0]['@id'].split('/')[-1]
        hp_ID = mapping['classes'][1]['@id'].split('/')[-1]
        hp_ID = hp_ID.replace('_',':')
        f.write(hp_ID+'\t'+ICD_ID+'\n')


    current_URL = currentMapping['links']['nextPage']
    if current_URL==None:
        break
    else:
        print('Completed downloading/parsing page %d, now downloading/parsing page %d. Total page number: %d' %(currentMapping['page'],currentMapping['nextPage'],currentMapping['pageCount']))
f.close()



current_URL = "http://data.bioontology.org/mappings?ontologies=HP,ICD9CM"
f=open('RawDataFiles/HP_ICD9_BioOnt.txt','w')
f.write('HP Term\tICD9 Term\n')
while True:
    currentMapping = get_json(current_URL)

    for mapping in currentMapping['collection']:
        ICD_ID = mapping['classes'][0]['@id'].split('/')[-1]
        hp_ID = mapping['classes'][1]['@id'].split('/')[-1]
        hp_ID = hp_ID.replace('_',':')
        f.write(hp_ID+'\t'+ICD_ID+'\n')


    current_URL = currentMapping['links']['nextPage']
    if current_URL==None:
        break
    else:
        print('Completed downloading/parsing page %d, now downloading/parsing page %d. Total page number: %d' %(currentMapping['page'],currentMapping['nextPage'],currentMapping['pageCount']))
f.close()
