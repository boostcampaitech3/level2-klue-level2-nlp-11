import pickle as pickle
import os
from collections import defaultdict
import pandas as pd
from ast import literal_eval
import torch
from torch.utils.data import DataLoader
from transformers import Trainer
from sklearn.model_selection import StratifiedShuffleSplit
import re
import random
import hanja
import warnings
warnings.filterwarnings('ignore')
os.environ["TOKENIZERS_PARALLELISM"] = "true"

class RE_Dataset(torch.utils.data.Dataset):
    """ Dataset 구성을 위한 class."""
    def __init__(self, pair_dataset, labels):
        self.pair_dataset = pair_dataset
        self.labels = labels

    def __getitem__(self, idx):
        item = {key:val for key,val in self.pair_dataset[idx].items()}
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

def preprocessing_dataset(dataset):
    """ 처음 불러온 csv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다."""
    #기존 데이터셋 entity열에 저장된 word, start_idx, end_idx, sub_type 값들을 각 열에 풀어서 연장
    dataset['subject_entity'] = dataset['subject_entity'].map(literal_eval)
    dataset['object_entity'] = dataset['object_entity'].map(literal_eval)
    sub_df = dataset['subject_entity'].apply(pd.Series).add_prefix('sub_')
    obj_df = dataset['object_entity'].apply(pd.Series).add_prefix('obj_')
    dataset = pd.concat([dataset, sub_df], axis=1)
    dataset = pd.concat([dataset, obj_df], axis=1)
    
    # sentence = dataset['sentence'].values
    # subject_entity = dataset['sub_word'].values
    # object_entity = dataset['obj_word'].values
    
    # pattern_list = [re.compile(r'(\([가-힣\w\s]+\))\1|\"\"'), re.compile(r'[一-龥]'), re.compile(r'\([\d]{1,2}\)|\(\)')]
    # replace_list = [halfLenStr, hanjaToHangeul, '']
    # target_col_list = [[sentence], [sentence, subject_entity, object_entity], [sentence]]
    
    # for pat, repl, target_col in zip(pattern_list, replace_list, target_col_list):
    #     for tgt in target_col:
    #         for i in range(len(dataset)):
    #             if pat.search(tgt[i]):
    #                 tgt[i] = pat.sub(repl, tgt[i])
    
    # dataset['sentence'] = sentence
    # dataset['sub_word'] = subject_entity
    # dataset['obj_word'] = object_entity
    
    return dataset

def load_data(dataset_dir):
    """ csv 파일을 경로에 맡게 불러 옵니다. """
    pd_dataset = pd.read_csv(dataset_dir)
    if 'train' in dataset_dir:
        pd_dataset = clean_dataset(pd_dataset)
    dataset = preprocessing_dataset(pd_dataset)
    return dataset

def split_data(dataset):
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
    for train_index, dev_index in split.split(dataset, dataset["label"]):
        train_dataset = dataset.loc[train_index]
        dev_dataset = dataset.loc[dev_index]
    
    return train_dataset,dev_dataset

def tokenized_dataset(dataset, tokenizer, input_type):
    """ tokenizer에 따라 sentence를 tokenizing 합니다."""
    tokens = []
    sentences = list(dataset['sentence'])

    for idx, row in enumerate(dataset.itertuples()):
        if input_type == 'double': # input이 두개인 경우
            temp = row.sub_word + '[SEP]' + row.obj_word
            tokenized_sentences = tokenizer(
                temp,
                sentences[idx],
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=256,
                add_special_tokens=True,
                )
                
        elif input_type == 'single': # input이 한개인 경우
            temp = [i for i in row.sentence]
            if row.sub_start_idx > row.obj_start_idx:
                temp[row.sub_start_idx:row.sub_end_idx+1] = [f'#^{row.sub_type}^{row.sub_word}#']
                temp[row.obj_start_idx:row.obj_end_idx+1] = [f'@+{row.obj_type}+{row.obj_word}@']
            else:
                temp[row.obj_start_idx:row.obj_end_idx+1] = [f'@+{row.obj_type}+{row.obj_word}@']
                temp[row.sub_start_idx:row.sub_end_idx+1] = [f'#^{row.sub_type}^{row.sub_word}#']

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

    batch_sampler = [bucket[start:start+batch_size] for bucket in bucket_dict.values() for start in range(0, len(bucket), batch_size)]
    random.shuffle(batch_sampler)

    return batch_sampler

def collate_fn(batch_samples):
    max_len = max([i['input_ids'].shape[1] for i in batch_samples])
    batch = defaultdict(list)
    for data in batch_samples:
        pad_len = max_len - data['input_ids'].shape[1]
        for key, val in data.items():
            if key != 'labels':
                if key == 'input_ids':
                    batch[key].append(torch.cat((val, torch.ones(1,pad_len)), dim=1).type(torch.long))
                else:
                    batch[key].append(torch.cat((val, torch.zeros(1,pad_len)), dim=1).type(torch.long))
            else:
                batch[key].append(val)
                
    batch['input_ids'] = torch.stack(batch['input_ids']).squeeze(1)
    batch['token_type_ids'] = torch.stack(batch['token_type_ids']).squeeze(1)
    batch['attention_mask'] = torch.stack(batch['attention_mask']).squeeze(1)
    batch['labels'] = torch.stack(batch['labels'])
    return batch

def halfLenStr(matchobj):
    string = matchobj[0]
    return string[:len(string)//2]

def hanjaToHangeul(matchobj):
    return hanja.translate(matchobj[0], 'substitution')

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
