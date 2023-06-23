import pandas as pd
import numpy as np
import torch

from torch.utils.data import Dataset
from typing import Dict, Tuple


PATH = 'data/filtered_data.csv'
TRUE = 1
FALSE = 0

class PurchaseHistory(Dataset):
    """
    Dataset creation and preprocessing
    """
    def __init__(self, mode="Train", max_len=70, data_dir=PATH, neg_sample_size=100):
        self.mode = mode
        self.max_len = max_len
        self.data_dir = data_dir
        self.neg_sample_size = neg_sample_size
        self.user_seq, self.item_seq, self.user2idx, self.item2idx, self.item_size = self._preprocess()
        self.negative_sample = self._popular_sampler(self.item_seq)
        
        self.PAD = 0
        self.MASK = len(self.item_seq) + 1

    def _preprocess(self) -> Tuple[pd.DataFrame, pd.Series, Dict[int, int], Dict[int, int], int]:
        """
        Load and preprocess the data
        """
        df = pd.read_csv(self.data_dir, index_col='Unnamed: 0')
        df.drop('count_', axis=1, inplace=True)
        df.order_ts = pd.to_datetime(df['order_ts'])
        df.sort_values(['user_id', 'order_ts'], inplace=True)
        
        user2idx = {v: k for k, v in enumerate(df['user_id'].unique())}
        item2idx = {v: k + 1 for k, v in enumerate(df['item_id'].unique())}
        item_size = len(item2idx)
        
        df['user_id'] = df['user_id'].map(user2idx)
        df['item_id'] = df['item_id'].map(item2idx)
        
        user_seq = df.groupby(by="user_id")
        user_seq = user_seq.apply(lambda user: list(user["item_id"]))
        
        user_seq = user_seq[user_seq.agg(len) > 1]
        if self.mode == 'Train':
            user_seq = user_seq.reset_index(drop=True)
            pass
        else:
            user_seq = user_seq[user_seq.agg(len) > 4]
            user_seq = user_seq.reset_index(drop=True)
        
        return user_seq, df.groupby(by="item_id").size(), user2idx, item2idx, item_size

    def _popular_sampler(self, item_seq: pd.Series) -> pd.Index:
        """
        Sample popular items
        """
        popular_items = item_seq.index.tolist()
        item_probs = item_seq / item_seq.sum()
        sampled_items = np.random.choice(popular_items, size=len(item_seq), replace=False, p=item_probs)
        negative_sample = pd.Index(sampled_items)
        return negative_sample

    def _eval_dataset(self, tokens: list, labels: list) -> Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor]:
        """
        Create validation/test dataset
        - Leave-one-out evaluation
        """
        candidates = []
        candidates.append(tokens[-1])

        sample_count = 0
        for item in self.negative_sample:
            if sample_count == self.neg_sample_size:
                break
            if item not in set(tokens):
                candidates.append(item)
                sample_count += 1
        
        tokens = tokens[:-1] + [self.MASK]
        tokens = tokens[-self.max_len:] 
        
        pad_len = self.max_len - len(tokens)
        tokens = [self.PAD] * pad_len + tokens
        
        labels = [TRUE] + [FALSE] * self.neg_sample_size

        return torch.LongTensor(tokens), torch.LongTensor(candidates), torch.LongTensor(labels)

    def __len__(self):
        return len(self.user_seq)

    def __getitem__(self, index):

        seq = self.user_seq[index]
        tokens = []
        labels = []

        if self.mode == "Train":
            if len(seq) <= 4:
                tokens = seq[:-1] + [self.MASK]
                labels = [self.PAD] * (len(seq) - 1) + [seq[-1]]
                
                tokens = tokens[-self.max_len:]
                labels = labels[-self.max_len:]
                pad_len = self.max_len - len(tokens)
                tokens = [self.PAD] * pad_len + tokens
                labels = [self.PAD] * pad_len + labels
                return torch.LongTensor(tokens), torch.LongTensor(labels)
            
            tokens = seq[:-3] + [self.MASK]
            labels = [self.PAD] * (len(seq) - 3) + [seq[-3]]
            
            tokens = tokens[-self.max_len:]
            labels = labels[-self.max_len:]
            pad_len = self.max_len - len(tokens)
            
            tokens = [self.PAD] * pad_len + tokens
            labels = [self.PAD] * pad_len + labels

            return torch.LongTensor(tokens), torch.LongTensor(labels)

        elif self.mode == "Valid":
            tokens = seq[:-1]
            return self._eval_dataset(tokens, labels)

        elif self.mode == "Test":
            tokens = seq[:]
            return self._eval_dataset(tokens, labels)


class PredictionDataset(Dataset):
    """
    Dataset for prediction with user IDs and mappings for users and items
    """
    def __init__(self, data_dir=PATH, max_len=70):
        self.data_dir = data_dir
        self.max_len = max_len
        self.user_seq, self.item_seq, self.user2idx, self.item2idx, self.item_size = self._preprocess()
        self.reverse_user2idx = {v: k for k, v in self.user2idx.items()}
        self.reverse_item2idx = {v: k for k, v in self.item2idx.items()}

        self.PAD = 0
        self.MASK = len(self.item2idx) + 1

    def _preprocess(self) -> Tuple[list, list, Dict[int, int], Dict[int, int]]:
        """
        Load the data and extract user IDs, sequences, and mappings
        """
        df = pd.read_csv(self.data_dir, index_col='Unnamed: 0')
        df.drop('count_', axis=1, inplace=True)
        df.order_ts = pd.to_datetime(df['order_ts'])
        df.sort_values(['user_id', 'order_ts'], inplace=True)

        user2idx = {v: k for k, v in enumerate(df['user_id'].unique())}
        item2idx = {v: k + 1 for k, v in enumerate(df['item_id'].unique())}
        item_size = len(item2idx)

        df['user_id'] = df['user_id'].map(user2idx)
        df['item_id'] = df['item_id'].map(item2idx)
        
        user_seq = df.groupby(by="user_id")
        user_seq = user_seq.apply(lambda user: list(user["item_id"]))

        return user_seq, df.groupby(by="item_id").size(), user2idx, item2idx, item_size

    def __len__(self):
        return len(self.user_seq)

    def __getitem__(self, index):
        sequence = self.user_seq[index]

        tokens = sequence[:]
        tokens = tokens[-self.max_len:]
        pad_len = self.max_len - len(tokens)
        tokens = [self.PAD] * pad_len + tokens

        return torch.LongTensor(tokens), index
    
    def map_user_id(self, idx: int) -> int:
        return self.reverse_user2idx[idx]

    def map_item_id(self, idx: int) -> int:
        return self.reverse_item2idx[idx]



    