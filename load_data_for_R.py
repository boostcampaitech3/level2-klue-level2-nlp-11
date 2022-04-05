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
        item = self.pair_dataset[idx]
        item['labels'] = torch.tensor(self.labels[idx])
            
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
    dataset = pd.read_csv(dataset_dir)
    if 'train' in dataset_dir:
        dataset = clean_dataset(dataset)
    dataset['subject_entity'] = dataset['subject_entity'].map(literal_eval)
    dataset['object_entity'] = dataset['object_entity'].map(literal_eval)
    
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
            #f'^#{row.sub_type}#{row.sub_word}^'+' '+ f'@+{row.obj_type}+{row.obj_word}@',
                ''.join(temp),
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=256,
                add_special_tokens=True
                ) 
        tokens.append(tokenized_sentences)

    return tokens

def make_entity_mask(tokens):
    for token in tokens:
        for tar, name in zip([65, 36], ['sub_mask', 'obj_mask']):
            mask_temp = torch.zeros_like(token['input_ids'])
            sentence = list(token['input_ids'].squeeze())
            start_idx = sentence.index(tar)
            end_idx = sentence.index(tar, start_idx+ 1)
            mask_temp[:, start_idx:end_idx+1] = 1 # start ^,@ 부터 end ^,@ 까지 1 masking
            
            # mask_temp[:, [start_idx,end_idx]] = 1 # ^,@ 위치에만 1 masking

            # start_idx2 = sentence.index(tar, end_idx+1)
            # end_idx2 = sentence.index(tar, start_idx2+ 1)
            # mask_temp[:, start_idx2:end_idx2+1] = 1

            token[name] = mask_temp
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

def split_data(dataset, num_splits):
    if num_splits == 1:
        test_size = 0.1
    else: test_size = 0.2
    split = StratifiedShuffleSplit(n_splits=num_splits, test_size=test_size, random_state=42)
    for train_index, dev_index in split.split(dataset, dataset["label"]):
        train_dataset = dataset.loc[train_index]
        dev_dataset = dataset.loc[dev_index]
    
        yield train_dataset, dev_dataset

def clean_dataset(dataset):
    # mislabeling 수정
    dataset = dataset.drop_duplicates(['sentence','subject_entity','object_entity','label'])
    dataset.loc[dataset['id'] == 32107, 'subject_entity'] = "{'word': '이용빈', 'start_idx': 0, 'end_idx': 2, 'type': 'PER'}"
    dataset.loc[dataset['id'] == 1435, 'object_entity'] = "{'word': '조오섭', 'start_idx': 0, 'end_idx': 2, 'type': 'PER'}"
    dataset.loc[dataset['id'] == 9269, 'object_entity'] = "{'word': '김성진', 'start_idx': 21, 'end_idx': 23, 'type': 'PER'}"
    dataset.loc[dataset['id'] == 30870, 'object_entity'] = "{'word': '김성진', 'start_idx': 21, 'end_idx': 23, 'type': 'PER'}"
    dataset.loc[dataset['id'] == 1334, 'subject_entity'] = "{'word': '김성진', 'start_idx': 21, 'end_idx': 23, 'type': 'PER'}"
    dataset.loc[dataset['id'] == 30530, 'subject_entity'] = "{'word': '김성진', 'start_idx': 21, 'end_idx': 23, 'type': 'PER'}"
    dataset.loc[dataset['id'] == 8477, 'object_entity'] = "{'word': '김성진', 'start_idx': 21, 'end_idx': 23, 'type': 'PER'}"

    # index오류 수정
    dataset.loc[dataset['id'] == 13780, 'object_entity'] = "{'word': '시동', 'start_idx': 4, 'end_idx': 5, 'type': 'POH'}"
    dataset.loc[dataset['id'] == 15584, 'object_entity'] = "{'word': '시동', 'start_idx': 4, 'end_idx': 5, 'type': 'POH'}"
    dataset.loc[dataset['id'] == 630, 'object_entity'] = "{'word': '시동', 'start_idx': 44, 'end_idx': 45, 'type': 'POH'}"
    dataset.loc[dataset['id'] == 25109, 'object_entity'] = "{'word': '은교', 'start_idx': 4, 'end_idx': 5, 'type': 'POH'}"
    dataset.loc[dataset['id'] == 25756, 'object_entity'] = "{'word': '스승의 은혜', 'start_idx': 13, 'end_idx': 18, 'type': 'POH'}"

    drop_ids = [18458, 6749, 8364, 11511, 25094, 277, 19074] # 19074:스승의 은혜
    dataset = dataset[dataset['id'].map(lambda x: x not in drop_ids)] # mislabeling drop
    dataset = dataset.reset_index(drop=True)

    return dataset

def delete_outlier(dataset):
    dataset['duplicated'] = dataset.duplicated('sentence', keep=False)
    dataset = dataset.sort_values(['sentence','label'])
    prev_sen, flag, tt = None, False, []

    for row in dataset.itertuples():
        if row.duplicated == True and row.label != 'no_relation':
            flag = True 
            tt.append(False)
            prev_sen = row.sentence
            continue

        if row.sentence == prev_sen and flag == True:
            if row.label == 'no_relation':
                tt.append(True)
                prev_sen = row.sentence
                continue
                
        flag = False
        tt.append(False)
        prev_sen = row.sentence

    dataset['condition'] = tt
    dataset = dataset[dataset['condition'] != True].reset_index()

    return dataset
