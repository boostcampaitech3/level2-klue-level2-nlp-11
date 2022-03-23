import pickle as pickle
import os
import pandas as pd
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
  subject_entity = []
  object_entity = []
  sentence = []
  new_se=[]
  new_oe=[]
  for i,j in zip(dataset['subject_entity'], dataset['object_entity']):
    i = i[1:-1].split(',')[0].split(':')[1]
    j = j[1:-1].split(',')[0].split(':')[1]

    subject_entity.append(i)
    object_entity.append(j)
  
  # pat = re.compile(r'(\([가-힣\w\s]+\))\1')
  # for s in dataset['sentence']:
  #   if pat.search(s):
  #     s = pat.sub(repl, s)
  #   sentence.append(s)
  
  pat = re.compile(r'[一-龥]')
  for i,j,k in zip(dataset['sentence'], subject_entity, object_entity):
    if pat.search(i):
      i = pat.sub(hanjaToHangel, i)
    sentence.append(i)
    
    if pat.search(j):
      j = pat.sub(hanjaToHangel, j)
    new_se.append(j)
    
    if pat.search(k):
      k = pat.sub(hanjaToHangel, k)
    new_oe.append(k)
  
  out_dataset = pd.DataFrame({'id':dataset['id'], 'sentence':sentence,'subject_entity':new_se,'object_entity':new_oe,'label':dataset['label'],})
  return out_dataset

def load_data(dataset_dir):
  """ csv 파일을 경로에 맡게 불러 옵니다. """
  pd_dataset = pd.read_csv(dataset_dir)
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
  for e01, e02 in zip(dataset['subject_entity'], dataset['object_entity']):
    temp = ''
    temp = e01 + '[SEP]' + e02
    concat_entity.append(temp)
  tokenized_sentences = tokenizer(
      concat_entity,
      list(dataset['sentence']),
      return_tensors="pt",
      padding=True,
      truncation=True,
      max_length=256,
      add_special_tokens=True,
      )
  return tokenized_sentences

def repl(matchobj):
    string = matchobj[0]
    return string[:len(string)//2]

def hanjaToHangel(matchobj):
    return hanja.translate(matchobj[0], 'substitution')