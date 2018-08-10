
import spacy
import textacy
import pandas as pd


nlp = spacy.load('en')
import re
import json
from pprint import pprint
stop_words = ['OBSOLETE', 'def', '"']

# ------------------- #

def match_id_pattern(text):
    pattern = '^id: GO:[0-9]*$'
    m = re.search(pattern, text)
    if m is not None:
        pattern = 'id: GO:[0-9]'
        return re.sub(pattern, '', text)
    else:
        return None

def match_synonym_pattern(text):
    pattern = '^synonym:*'
    m = re.search(pattern, text)
    if m is not None:
        pattern = '^synonym:'
        return re.sub(pattern, '', text)
    else:
        return None

def match_def_pattern(text):
    pattern = '^def:*'
    m = re.search(pattern, text)
    if m is not None:
        pattern = '^def:'
        return re.sub(pattern, '', text)
    else:
        return None

def match_function_pattern(text):
    pattern = '^namespace:*'
    m = re.search(pattern, text)
    if m is not None:
        pattern = '^namespace:'
        return re.sub(pattern,'',text)
    else:
        return None



def process_txt(text):
    global stop_words
    for s in stop_words:
        try:
            val_txt = text.replace(s, '')
        except:
            pass
    text = tokenize_text(text)
    return text

def tokenize_text(txt):

    txt = textacy.preprocess.remove_punct(txt, marks=';,:[]()-+.=<>')
    txt = textacy.preprocess.replace_urls(txt, replace_with=' ')
    txt = textacy.preprocess.replace_numbers(txt, replace_with = ' ')
    txt = textacy.preprocess.replace_currency_symbols(txt, replace_with=None)

    res = []
    txt = textacy.preprocess.normalize_whitespace(txt)
    doc = textacy.Doc(txt, lang='en')
    for s in textacy.extract.words(doc, exclude_pos=None, min_freq=1):
        if len(str(s)) > 2 :
            res.append(str(s).lower())
    return res

def aux_write(df, id, text_data, type):
    text_data = process_txt(text_data)
    df = df.append ({
        'id' : id,
        'txt' : text_data,
        'type' : type
    },ignore_index=True)
    return df

def read_data():

    cur_key = None
    cur_val = ''
    cur_type = None
    df = pd.DataFrame(columns=['id','txt','type'])
    count = 0
    with open('go.obo', 'r') as f:
        l = f.readline()

        while l:
            l = l.strip('\n')
            txt = l.strip()
            key = match_id_pattern(txt)

            # write the previous record
            if key is not None:
                if cur_key is not None:
                    df =  aux_write(df,cur_key,cur_val,cur_type)
                    cur_val = ''
                count += 1

                cur_key = key

            syn_txt = match_synonym_pattern(txt)
            if syn_txt is not None :
                cur_val += syn_txt
            def_txt = match_def_pattern(txt)
            if def_txt is not None:
                cur_val += def_txt
            type_txt = match_function_pattern(txt)
            if type_txt is not None:
                cur_type = type_txt

            l = f.readline()

    df.to_pickle('obo_data.pkl')
    print(df.head(100))

read_data()

# temp_json = 'temp_data.json'
# with open(temp_json, 'w') as file:
#     file.write(json.dumps(res_dict))
# with open(temp_json) as tmp_file:
#     data_dict = json.loads(tmp_file.read())



#
# def process_key(key_txt):
#     res = key_txt.split(':')[-1]
#     return res
#



#
# def main_1():
#     data_dict_1 = {}
#     for k, v in data_dict.items():
#         k = process_key(k)
#         v = process_val(v)
#         data_dict_1[k] = v
#
#     temp_json_1 = 'temp_data_1.json'
#     with open(temp_json_1, 'w') as file:
#         file.write(json.dumps(data_dict_1))
#
#
#
#
#
# def main_2():
#     temp_json_1 = 'temp_data_1.json'
#     with open(temp_json_1) as tmp_file:
#         data_dict_2 = json.loads(tmp_file.read())
#     for k, v in data_dict_2.items():
#         v = tokenize_text(v)
#         data_dict_2[k] = v
#     temp_json_2 = 'temp_data_2.json'
#     with open(temp_json_2, 'w') as file:
#         file.write(json.dumps(data_dict_2))
#
# # -----------#
#
# main_1()
# main_2()