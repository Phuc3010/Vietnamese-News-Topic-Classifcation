import torch
import re
import numpy as np
from underthesea import word_tokenize
import os
from utils import dict_map
import time
import string
from tqdm import tqdm

def create_label(folder_path):
    news_folder = os.listdir(folder_path)
    label_list = list()
    for folder_name in news_folder:
        label_list.append(folder_name)
    label_list = sorted(list(label_list))
    label_list = {news:idx for idx, news in enumerate(label_list)}
    return label_list

def remove_noise_sent(sent):
    if len(sent.split(" ")) < 5:
        return True
    if len(sent) < 10:
        return True
    return False

def text_preprocessing(text, stop_words):
    '''
    Preprocessing text:
    - Tone normalization
    - Remove stop words
    - Remove punctuation
    - Word Tokenize
    '''
    text = tone_normalization(text, dict_map)
    text = re.sub(f'[{string.punctuation}\d\n]', '', text)
    text = text.split(" ")
    text = [word for word in text if word not in stop_words and len(word)>1]
    return text

def get_stopwords(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        stop_words = f.read().split('\n')
    return stop_words

def tone_normalization(text, dict_map):
    for i, j in dict_map.items():
        text = text.replace(i, j)
    return text 

def preprocessing_pipeline(dataset, word_to_index: dict):
    tokenized_dataset = []
    stop_words = get_stopwords("/kaggle/input/stopwords-vietnamese/stopwords.txt")
    for idx in tqdm(range(len(dataset))):
        text = dataset[idx][0]
        text = text_preprocessing(text, stop_words)
        if len(text) <= 0:
            continue
        text_ids = [word_to_index[word] if word in word_to_index else 1 for word in text]
        tokenized_dataset.append([text_ids, dataset[idx][1]])
    return tokenized_dataset

def create_dataset(folder_path, label_list):
    news_folds = os.listdir(folder_path)
    dataset = list()
    for news_folder in news_folds:
        files = os.listdir(folder_path+"/"+news_folder)
        label = label_list[news_folder]
        for news_txt in tqdm(files):
            data_path = folder_path+"/"+news_folder+"/"+news_txt
            with open(data_path, "rb") as reader:
                contents = reader.read()
            contents = contents.decode("utf-16").strip()
            contents = contents.split("\r\n")        
            for line in contents:
                if len(line.split(" ")) < 20 or len(line) < 5:
                    continue
                else:
                    dataset.append([line, label])
    return dataset