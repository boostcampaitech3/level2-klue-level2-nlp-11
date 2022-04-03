import pickle
import os
import pandas as pd
from ast import literal_eval
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, Trainer
from sklearn.model_selection import StratifiedShuffleSplit
from collections import defaultdict
import random
import warnings
warnings.filterwarnings('ignore')
os.environ["TOKENIZERS_PARALLELISM"] = "true"

class RE_Dataset(torch.utils.data.Dataset):
    def __init__(self, pair_dataset, labels):
        self.pair_dataset = pair_dataset
        self.labels = labels

    def __getitem__(self, idx):
        item = {key:val for key,val in self.pair_dataset[idx].items()}
        item['labels'] = torch.tensor(self.labels[idx])

        for tar in [65, 36]:
            mask_temp = torch.zeros_like(item['input_ids'])
            start_idx = list(item['input_ids'].squeeze()).index(tar)
            end_idx = list(item['input_ids'].squeeze()).index(tar, start_idx+ 1)
            mask_temp[:, start_idx:end_idx+1] = 1

            if tar == 65:
                item['sub_mask'] = mask_temp
            else: item['obj_mask'] = mask_temp
            
        return item

    def __len__(self):
        return len(self.labels)

class BucketTrainer(Trainer):
    def get_train_dataloader(self) -> DataLoader:

        return DataLoader(self.train_dataset, batch_sampler=self.train_sampler, collate_fn=collate_fn)

    def get_eval_dataloader(self, eval_dataset) -> DataLoader:
        if eval_dataset is not None:
            return DataLoader(eval_dataset, batch_sampler=self.valid_sampler, collate_fn=collate_fn)
        else:
            return DataLoader(self.eval_dataset, batch_sampler=self.valid_sampler, collate_fn=collate_fn)

def load_data(dataset_dir):
    dataset = pd.read_csv(dataset_dir,
                converters={'subject_entity':literal_eval, 'object_entity': literal_eval})

    return dataset

def tokenized_dataset(dataset, tokenizer):
    sub_df = dataset['subject_entity'].apply(pd.Series).add_prefix('sub_')
    obj_df = dataset['object_entity'].apply(pd.Series).add_prefix('obj_')
    dataset = pd.concat([dataset, sub_df], axis=1)
    dataset = pd.concat([dataset, obj_df], axis=1)

    tokens = []
    for row in dataset.itertuples():
        temp = [i for i in row.sentence]
        if row.sub_start_idx > row.obj_start_idx:
            temp[row.sub_start_idx:row.sub_end_idx+1] = [f'^#{row.sub_type}#{row.sub_word}^']
            temp[row.obj_start_idx:row.obj_end_idx+1] = [f'@+{row.obj_type}+{row.obj_word}@']
        else:
            temp[row.obj_start_idx:row.obj_end_idx+1] = [f'@+{row.obj_type}+{row.obj_word}@']
            temp[row.sub_start_idx:row.sub_end_idx+1] = [f'^#{row.sub_type}#{row.sub_word}^']

        tokenized_sentences = tokenizer(
                ''.join(temp),
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=256,
                add_special_tokens=True
                ) 
        tokens.append(tokenized_sentences)

    return tokens

def make_sampler(data, batch_size=64, max_pad_len=20):
    sentence_length = [sen['input_ids'].shape[1] for sen in data]
    bucket_dict = defaultdict(list)

    for index, src_length in enumerate(sentence_length):
        bucket_dict[(src_length // max_pad_len)].append(index)

    batch_dict = defaultdict(list)

    for key, bucket in bucket_dict.items():
        for start in range(0, len(bucket), batch_size):
            batch_dict[key].append(bucket[start:start+batch_size])

    surplus = []
    sampler = []
    for batch_set in batch_dict.values():
        for batch in batch_set:
            if len(batch) == batch_size:
                sampler.append(batch)
            else:
                surplus.extend(batch)
    sampler.extend([surplus[start:start+batch_size] for start in range(0, len(surplus), batch_size)])
    random.shuffle(sampler)
    return sampler

def collate_fn(batch_samples):
    max_len = max([i['input_ids'].shape[1] for i in batch_samples])
    batch = defaultdict(list)
    for data in batch_samples:
        pad_len = max_len - data['input_ids'].shape[1]
        for key, val in data.items():
            if key != 'labels':
                if key == 'input_ids':
                    input_id = torch.cat((val, torch.ones(1,pad_len)), dim=1).type(torch.long)
                    batch[key].append(input_id)
                    
                elif key != 'token_type_ids':
                    batch[key].append(torch.cat((val, torch.zeros(1,pad_len)), dim=1).type(torch.long))
            else:
                batch[key].append(val)
                
    batch['input_ids'] = torch.stack(batch['input_ids']).squeeze(1)
    batch['attention_mask'] = torch.stack(batch['attention_mask']).squeeze(1)
    batch['sub_mask'] = torch.stack(batch['sub_mask']).squeeze(1)
    batch['obj_mask'] = torch.stack(batch['obj_mask']).squeeze(1)
    batch['labels'] = torch.stack(batch['labels'])
    return batch

def split_data(dataset):
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
    for train_index, dev_index in split.split(dataset, dataset["label"]):
        train_dataset = dataset.loc[train_index]
        dev_dataset = dataset.loc[dev_index]
    
    return train_dataset,dev_dataset
