import pickle
import os
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer


class RE_Dataset_for_R(torch.utils.data.Dataset):
    pass


def preprocessing_dataset_for_R(dataset):
    # 데이터셋 데이터프레임을 가져와서 문장의 엔티티의 앞, 뒤에 스페셜 토큰을 넣어서 반환
    # subject_entity와 object_entity의 위치 정보를 가져와서
    # 위치 정보에 맞춰서 토큰 삽입
    # dataset['']
    sentence = []
    for sub, ob, sent in zip(dataset['subject_entity'], dataset['object_entity'], dataset['sentence']):
        if len(sub[1:-1].split(',')) > 4:
            start_1 = int(sub[1:-1].split(',')[2].split(':')[1])
            end_1 = int(sub[1:-1].split(',')[3].split(':')[1])
        else:
            start_1 = int(sub[1:-1].split(',')[1].split(':')[1])
            end_1 = int(sub[1:-1].split(',')[2].split(':')[1])
        if len(ob[1:-1].split(',')) > 4:
            print(ob)
            start_2 = int(ob[1:-1].split(',')[2].split(':')[1])
            end_2 = int(ob[1:-1].split(',')[3].split(':')[1])
        else:
            start_2 = int(ob[1:-1].split(',')[1].split(':')[1])
            end_2 = int(ob[1:-1].split(',')[2].split(':')[1])

        if start_1 > start_2:
            sent = sent[:start_1] + '[SUBS]' \
            + sent[start_1:end_1] + '[SUBE]' \
            + sent[end_1:start_2] + '[OBJS]' \
            + sent[start_2:end_2] + '[OBJE]' \
            + sent[end_2:]
        else:
            sent = sent[:start_2] + '[OBJS]' \
            + sent[start_2:end_2] + '[OBJE]' \
            + sent[end_2:start_1] + '[SUBS]' \
            + sent[start_1:end_1] + '[SUBE]' \
            + sent[end_1:]

        sentence.append(sent)

    out_dataset = pd.DataFrame({'sentence':sentence})

    return out_dataset
    


def load_data_for_R(dataset_dir):
    pd_dataset = pd.read_csv(dataset_dir)
    dataset = preprocessing_dataset_for_R(pd_dataset)

    return dataset


temp = "{'word': '비틀즈', 'start_idx': 24, 'end_idx': 26, 'type': 'ORG'}"
print(int(temp[1:-1].split(',')[1].split(':')[1]))
print(int(temp[1:-1].split(',')[2].split(':')[1]))

tokenizer = AutoTokenizer.from_pretrained('./vocab')
print(tokenizer.tokenize('[SUBS]이순신[SUBE]은 조선 중기의 [OBJS]무신[OBJE]이다.'))

# dataset = load_data_for_R('../dataset/train/train.csv')
# dataset.head()

print(eval(temp))
print(type(eval(temp)))
print(eval(temp)['word'])