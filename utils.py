import re
import string
from underthesea import word_tokenize
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import torch

dict_map = {
    "òa": "oà",
    "Òa": "Oà",
    "ÒA": "OÀ",
    "óa": "oá",
    "Óa": "Oá",
    "ÓA": "OÁ",
    "ỏa": "oả",
    "Ỏa": "Oả",
    "ỎA": "OẢ",
    "õa": "oã",
    "Õa": "Oã",
    "ÕA": "OÃ",
    "ọa": "oạ",
    "Ọa": "Oạ",
    "ỌA": "OẠ",
    "òe": "oè",
    "Òe": "Oè",
    "ÒE": "OÈ",
    "óe": "oé",
    "Óe": "Oé",
    "ÓE": "OÉ",
    "ỏe": "oẻ",
    "Ỏe": "Oẻ",
    "ỎE": "OẺ",
    "õe": "oẽ",
    "Õe": "Oẽ",
    "ÕE": "OẼ",
    "ọe": "oẹ",
    "Ọe": "Oẹ",
    "ỌE": "OẸ",
    "ùy": "uỳ",
    "Ùy": "Uỳ",
    "ÙY": "UỲ",
    "úy": "uý",
    "Úy": "Uý",
    "ÚY": "UÝ",
    "ủy": "uỷ",
    "Ủy": "Uỷ",
    "ỦY": "UỶ",
    "ũy": "uỹ",
    "Ũy": "Uỹ",
    "ŨY": "UỸ",
    "ụy": "uỵ",
    "Ụy": "Uỵ",
    "ỤY": "UỴ",
    }

def save_models(state, state_path):
    torch.save(state, state_path)

def load_model(state_path):
    state = torch.load(state_path)
    return state

def get_stopwords(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        stop_words = f.read().split('\n')
    return stop_words

def load_word_embedding(file_emb):
    reader = open(file_emb, encoding='utf-8', newline='\n', errors='ignore')
    vocab, word_emb = [], []
    for line in reader:
        words = line.strip().split(' ')
        vocab.append(words[0])
        word_emb.append(list(map, words[1:]))
    return vocab, np.array(word_emb)

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params
    
def collate_fn(batch):
    '''
    Create Collate function for batch processing to handle Dynamic Sequence length
    '''
    inputs, labels = zip(*batch)
    seq_len = [len(input) for input in inputs]
    inputs = pad_sequence(inputs, batch_first=True, padding_value=0)
    return inputs, seq_len, labels


