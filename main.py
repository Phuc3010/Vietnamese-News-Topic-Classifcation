import os
from torch import nn
import pickle
import numpy as np
from model import AttentionLSTM
import torch
import argparse
from evaluate import evaluate
from train import train
from utils import load_word_embedding, count_parameters, collate_fn
from preprocessing import create_dataset, create_label, preprocessing_pipeline
from dataset import NewsDataset

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    # Dasaset
    parser.add_argument("--train_folder_path", type=str, required=True, default="/kaggle/input/vntc-data/Train_Full/Train_Full")
    parser.add_argument("--test_folder_path", type=str, required=True, default="/kaggle/input/vntc-data/Test_Full/Test_Full")
    parser.add_argument("output_dir", type=str, required=True, default="./model.pth")
    parser.add_argument("--do_eval", type=bool, default=False)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--word_embedding_path", type=str, default="/kaggle/input/word2vec-embedding/word_embedding.npy")
    parser.add_argument("--vocab_to_idx_path", type=str, default="/kaggle/input/word-to-index-dict/word_to_index (1).pkl")
    # Model parameters
    parser.add_argument("--state", default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--word_embedding_size", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--dropout_rate", type=float, default=0.5)
    args = parser.parse_args()

    word_embedding = np.load(args.word_embedding_path)
    word_embedding = np.pad(word_embedding, ((2,0),(0,0)), mode='constant', constant_values =0.0)
    word_embedding = torch.tensor(word_embedding).clone().detach()
    word_to_idx = pickle.load(open(args.vocab_to_idx_path, "rb"))

    label_list = create_label(args.train_folder_path)
    n_classes = len(label_list)
    train_dataset = create_dataset(args.train_folder_path, label_list)
    test_dataset = create_dataset(args.test_folder_path, label_list)
    train_dataset = preprocessing_pipeline(train_dataset, word_to_idx)
    test_dataset = preprocessing_pipeline(test_dataset, word_to_idx)
    train_dataset = NewsDataset(train_dataset)
    test_dataset = NewsDataset(test_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset, collate_fn=collate_fn,batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_dataset, collate_fn=collate_fn,batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=2)

    model = AttentionLSTM(pretrained_embedding=word_embedding, input_size=args.word_embedding_size,
                          hidden_size=args.hidden_size, n_layers=args.n_layers, dropout=args.dropout, n_classes=n_classes)
    criterion = nn.CrossEntropyLoss()
    if args.state is not None:
        state = torch.load(args.state)
        model.load_state_dict(state['model_state_dict'])
        
    if args.do_eval==False:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        if args.state is not None:
            optimizer.load_state_dict(state['optimizer_state_dict']) 
        saved_epochs = 0 if args.state is None else state['epoch']
        total_epochs = args.epochs - saved_epochs
        print(f">> Model summary: {count_parameters(model)}")
        train(model=model, data_loader=train_loader, optimizer=optimizer, criterion=criterion,
             total_epochs=args.epochs, save_path=args.output_dir, device=args.device)
    else:
        for p in model.parameters():
            p.requires_grad_(False)
        accuracy, total_loss = evaluate(model, test_loader, criterion, device=args.device)
        print(f"Accuracy of test dataset: {100*accuracy:.3f}")