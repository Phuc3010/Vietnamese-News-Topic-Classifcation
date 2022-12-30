import torch
import numpy as np
import torch.nn.functional as F
from torch import nn, optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence, pad_packed_sequence
import numpy as n

class AttentionLSTM(nn.Module):
    def __init__(self, pretrained_embedding, input_size, hidden_size, n_layers, dropout, n_classes, bidirectional=True) -> None:
        super(AttentionLSTM, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(pretrained_embedding)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=n_layers, \
        batch_first=True, bidirectional=bidirectional)
        self.attention = nn.Linear(hidden_size*2, 1)
        self.classifier = nn.Linear(hidden_size*2, n_classes)
        self.softmax = nn.Softmax(dim=-1)
    
    @staticmethod
    def make_mask(seq_len, max_seq_len):
        idx = torch.arange(max_seq_len).to(seq_len).repeat(seq_len.size(0), 1)
        mask = torch.gt(seq_len.unsqueeze(-1), idx).to(seq_len)
        return mask
    
    @staticmethod
    def mask_attn(attention, mask=None):
        if mask is not None:
            attention = F.softmax(attention)
        else:
            mask = ((1-mask)*-np.inf).to(attention)
            for idx in range(attention.dim()-mask.dim()):
                mask = mask.unsqueeze(1)
            attention = F.softmax(attention+mask, dim=-1)
        return attention

    def forward(self, inputs, inputs_len):
        inputs_embed = self.embedding(inputs)
        inputs_packed = pack_padded_sequence(inputs_embed, inputs_len, batch_first=True,\
        enforce_sorted=False)
        output_packed, (hidden, cells)= self.lstm(inputs_packed)
        max_seq_len = torch.max(inputs_len)
        mask = self.make_mask(inputs_len, max_seq_len)
        # output_padded shape: BxSxD
        output_padded = pad_packed_sequence(output_packed, batch_first=True)[0]
        attn_weights = self.attention(output_padded)
        attn_weights = self.mask_attn(attn_weights, mask)
        output = (attn_weights*output_padded).sum(dim=1)
        logits = self.classifier(output)
        return logits