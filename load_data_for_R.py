import pickle
import os
import pandas as pd
from ast import literal_eval
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from sklearn.model_selection import StratifiedShuffleSplit
import re
import hanja


class RE_Dataset_for_R(torch.utils.data.Dataset):
    def __init__(self, tokenized_dataset, labels, train=True):
        self.tokenized_dataset = tokenized_dataset
        self.labels = labels
        self.train = train
    
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.tokenized_dataset.items()}
        if self.train:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        else:
            item['labels'] = torch.tensor(0, dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)


def preprocessing_dataset_for_R(dataset):
    # 데이터셋 데이터프레임을 가져와서 문장의 엔티티의 앞, 뒤에 스페셜 토큰을 넣어서 반환
    # subject_entity와 object_entity의 위치 정보를 가져와서
    # 위치 정보에 맞춰서 토큰 삽입
    # dataset['']
    sub_df = dataset['subject_entity'].apply(pd.Series).add_prefix('sub_')
    obj_df = dataset['object_entity'].apply(pd.Series).add_prefix('obj_')
    dataset = pd.concat([dataset, sub_df], axis=1)
    dataset = pd.concat([dataset, obj_df], axis=1)


    # sentence = dataset['sentence'].values
    # subject_entity = dataset['sub_word'].values
    # object_entity = dataset['obj_word'].values

    # need_change = ["subject_entity", "object_entity", "sub_start_idx", "sub_end_idx", "obj_start_idx", "obj_end_idx"]


    # pattern_list = [re.compile(r'(\([가-힣\w\s]+\))\1|\"\"'), re.compile(r'[一-龥]'), re.compile(r'\([\d]{1,2}\)|\(\)')]
    # replace_list = [halfLenStr, hanjaToHangeul, '']
    # target_col_list = [[sentence], [sentence, subject_entity, object_entity], [sentence]]
    # tmp = []
    # for pat, repl, target_col in zip(pattern_list, replace_list, target_col_list):
    #     for tgt in target_col:
    #         for i in range(len(dataset)):
    #             if pat.search(tgt[i]):
    #                 tmp.append(i)
    #                 tgt[i] = pat.sub(repl, tgt[i])
                        
            
    # dataset['sentence'] = sentence
    # dataset['sub_word'] = subject_entity
    # dataset['obj_word'] = object_entity
    # tmp = list(set(tmp))
    # for i in tmp: 
    #     sen = str(dataset.loc[[i],['sentence']].values[0][0])
    #     sub = str(dataset.loc[[i], ['sub_word']].values[0][0])
    #     obj = str(dataset.loc[[i], ['obj_word']].values[0][0])
    #     subs = sen.find(sub)
    #     sube = subs + len(sub) - 1
    #     objs = sen.find(str(obj))
    #     obje = objs + len(obj) - 1
    #     subentity = str(dataset.loc[[i],["subject_entity"]].values).replace(str(dataset.loc[[i],["sub_start_idx"]].values[0][0]), str(subs))
    #     subentity = subentity.replace(str(dataset.loc[[i],["sub_end_idx"]].values[0][0]), str(sube))
    #     objentity = str(dataset.loc[[i],["object_entity"]].values).replace(str(dataset.loc[[i],["obj_start_idx"]].values[0][0]), str(objs))
    #     objentity = objentity.replace(str(dataset.loc[[i],["obj_end_idx"]].values[0][0]), str(obje))
    #     need_change_val = [subentity, objentity, subs, sube, objs, obje]
    #     for ii in range(len(need_change)):
    #         dataset.loc[[i],[need_change[ii]]] = need_change_val[ii]


    sentence = []
    labels = []
    for sent, start_1, end_1, start_2, end_2, label in zip(dataset['sentence'], 
                                                        dataset['sub_start_idx'], 
                                                        dataset['sub_end_idx'], 
                                                        dataset['obj_start_idx'], 
                                                        dataset['obj_end_idx'],
                                                        dataset['label']):
        if start_1 < start_2:
            sent = '[CLS]' + sent[:start_1] + '[SUBS]' \
            + sent[start_1:end_1+1] + '[SUBE]' \
            + sent[end_1+1:start_2] + '[OBJS]' \
            + sent[start_2:end_2+1] + '[OBJE]' \
            + sent[end_2+1:] + '[SEP]'
        else:
            sent = '[CLS]' + sent[:start_2] + '[OBJS]' \
            + sent[start_2:end_2+1] + '[OBJE]' \
            + sent[end_2+1:start_1] + '[SUBS]' \
            + sent[start_1:end_1+1] + '[SUBE]' \
            + sent[end_1+1:] + '[SEP]'

        sentence.append(sent)
        labels.append(label)

    out_dataset = pd.DataFrame({'sentence':sentence, 'label':labels})

    return out_dataset
    

def load_data_for_R(dataset_dir):
    pd_dataset = pd.read_csv(dataset_dir,
                converters={'subject_entity':literal_eval, 'object_entity': literal_eval})
    dataset = preprocessing_dataset_for_R(pd_dataset)

    return dataset


def split_data(dataset):
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
    for train_index, dev_index in split.split(dataset, dataset["label"]):
        train_dataset = dataset.loc[train_index]
        dev_dataset = dataset.loc[dev_index]
    
    return train_dataset,dev_dataset


def convert_sentence_to_features(dataset, tokenizer, max_len):
    max_seq_len = max_len
    mask_padding_with_zero = True
    pad_token = tokenizer.pad_token_id

    all_input_ids = []
    all_attention_mask = []
    all_sub_mask = []
    all_obj_mask = []
    all_label = []
    m_len = 0
    for sent, label in zip(dataset['sentence'], dataset['label']):
        token = tokenizer.tokenize(sent)

        m_len = max(m_len, len(token))
        subs_p = token.index('[SUBS]')
        sube_p = token.index('[SUBE]')
        objs_p = token.index('[OBJS]')
        obje_p = token.index('[OBJE]')

        token[subs_p] = '@'
        token[sube_p] = '@'
        token[objs_p] = '^'
        token[obje_p] = '^'

        special_tokens_count = 1

        if len(token) < max_seq_len - special_tokens_count:
            input_ids = tokenizer.convert_tokens_to_ids(token)
            attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            padding_length = max_seq_len - len(input_ids)
            input_ids = input_ids + ([pad_token] * padding_length)
            attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)

            sub_mask = [0] * len(attention_mask)
            obj_mask = [0] * len(attention_mask)

            sub_mask[subs_p] = 1
            sub_mask[sube_p] = 1
            obj_mask[objs_p] = 1
            obj_mask[obje_p] = 1
       

            assert len(input_ids) == max_seq_len, "Error with input length {} vs {}".format(len(input_ids), max_seq_len)
            assert len(attention_mask) == max_seq_len, "Error with attention mask length {} vs {}".format(
                len(attention_mask), max_seq_len
            )

            all_input_ids.append(input_ids)
            all_attention_mask.append(attention_mask)
            all_sub_mask.append(sub_mask)
            all_obj_mask.append(obj_mask)
            all_label.append(label)

    all_features = {
        'input_ids': torch.tensor(all_input_ids),
        'attention_mask': torch.tensor(all_attention_mask),
        'sub_mask': torch.tensor(all_sub_mask),
        'obj_mask': torch.tensor(all_obj_mask)
    }

    return all_features, all_label


def halfLenStr(matchobj):
    string = matchobj[0]
    return string[:len(string)//2]

def hanjaToHangeul(matchobj):
    return hanja.translate(matchobj[0], 'substitution')


# tokenizer = AutoTokenizer.from_pretrained('./vocab_robertaLarge')
# dataset = load_data_for_R('../dataset/train/train.csv')
# print(dataset[:41])
# features, labels = convert_sentence_to_features(dataset, tokenizer, 256)
# print({key: torch.tensor(val[0]) for key, val in features.items()})
# print(features)
# print(labels)

# dataset = pd.read_csv('../dataset/train/train.csv')

# count = 0
# for i in range(len(dataset)):
#     if count >= 5250:
#         break
#     if dataset.loc[i]['label'] == 'no_relation':
#         dataset = dataset.drop(i)
#         count += 1

# dataset.to_csv('train_temp.csv', index=False)