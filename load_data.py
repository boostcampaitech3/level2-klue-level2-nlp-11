import pickle as pickle
import os
import pandas as pd
from ast import literal_eval
import torch
from sklearn.model_selection import StratifiedShuffleSplit
import re
import hanja

class RE_Dataset(torch.utils.data.Dataset):
    """ Dataset 구성을 위한 class."""
    def __init__(self, pair_dataset, labels):
        self.pair_dataset = pair_dataset
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.pair_dataset.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def preprocessing_dataset(dataset):
    """ 처음 불러온 csv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다."""
    #기존 데이터셋 entity열에 저장된 word, start_idx, end_idx, sub_type 값들을 각 열에 풀어서 연장
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
    pd_dataset = pd.read_csv(dataset_dir, 
                converters={'subject_entity':literal_eval, 'object_entity':literal_eval})
    dataset = preprocessing_dataset(pd_dataset)
    return dataset

def split_data(dataset):
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
    for train_index, dev_index in split.split(dataset, dataset["label"]):
        train_dataset = dataset.loc[train_index]
        dev_dataset = dataset.loc[dev_index]
    
    return train_dataset,dev_dataset

def tokenized_dataset(dataset, tokenizer):
    """ tokenizer에 따라 sentence를 tokenizing 합니다."""
    concat_entity = []
    for row in dataset.itertuples():
        temp = [i for i in row.sentence]
        if row.sub_start_idx > row.obj_start_idx:
            temp[row.sub_start_idx:row.sub_end_idx+1] = [f'#^{row.sub_type}^{row.sub_word}#']
            temp[row.obj_start_idx:row.obj_end_idx+1] = [f'@+{row.obj_type}+{row.obj_word}@']
        else:
            temp[row.obj_start_idx:row.obj_end_idx+1] = [f'@+{row.obj_type}+{row.obj_word}@']
            temp[row.sub_start_idx:row.sub_end_idx+1] = [f'#^{row.sub_type}^{row.sub_word}#']
        concat_entity.append(''.join(temp))
        
    tokenized_sentences = tokenizer(
        concat_entity,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256,
        add_special_tokens=True,
        )
    return tokenized_sentences
  
def halfLenStr(matchobj):
    string = matchobj[0]
    return string[:len(string)//2]

def hanjaToHangeul(matchobj):
    return hanja.translate(matchobj[0], 'substitution')
