import pickle
import os
import pandas as pd
from ast import literal_eval
import torch
from tqdm import tqdm
from transformers import AutoTokenizer


class RE_Dataset_for_R(torch.utils.data.Dataset):
    def __init__(self, tokenized_dataset, labels):
        self.tokenized_dataset = tokenized_dataset
        self.labels = labels
    
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.tokenized_dataset.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)


def preprocessing_dataset_for_R(dataset):
    # 데이터셋 데이터프레임을 가져와서 문장의 엔티티의 앞, 뒤에 스페셜 토큰을 넣어서 반환
    # subject_entity와 object_entity의 위치 정보를 가져와서
    # 위치 정보에 맞춰서 토큰 삽입
    # dataset['']
    label_list = ['no_relation', 'org:top_members/employees', 'org:members',
       'org:product', 'per:title', 'org:alternate_names',
       'per:employee_of', 'org:place_of_headquarters', 'per:product',
       'org:number_of_employees/members', 'per:children',
       'per:place_of_residence', 'per:alternate_names',
       'per:other_family', 'per:colleagues', 'per:origin', 'per:siblings',
       'per:spouse', 'org:founded', 'org:political/religious_affiliation',
       'org:member_of', 'per:parents', 'org:dissolved',
       'per:schools_attended', 'per:date_of_death', 'per:date_of_birth',
       'per:place_of_birth', 'per:place_of_death', 'org:founded_by',
       'per:religion']

    sub_df = dataset['subject_entity'].apply(pd.Series).add_prefix('sub_')
    obj_df = dataset['object_entity'].apply(pd.Series).add_prefix('obj_')
    dataset = pd.concat([dataset, sub_df], axis=1)
    dataset = pd.concat([dataset, obj_df], axis=1)

    sentence = []
    labels = []
    for sent, start_1, end_1, start_2, end_2, label in zip(dataset['sentence'], 
                                                        dataset['sub_start_idx'], 
                                                        dataset['sub_end_idx'], 
                                                        dataset['obj_start_idx'], 
                                                        dataset['obj_end_idx'],
                                                        dataset['label']):
        if start_1 < start_2:
            sent = '<s>' + sent[:start_1] + '[SUBS]' \
            + sent[start_1:end_1+1] + '[SUBE]' \
            + sent[end_1+1:start_2] + '[OBJS]' \
            + sent[start_2:end_2+1] + '[OBJE]' \
            + sent[end_2+1:] + '</s>'
        else:
            sent = '<s>' + sent[:start_2] + '[OBJS]' \
            + sent[start_2:end_2+1] + '[OBJE]' \
            + sent[end_2+1:start_1] + '[SUBS]' \
            + sent[start_1:end_1+1] + '[SUBE]' \
            + sent[end_1+1:] + '</s>'

        sentence.append(sent)
        labels.append(label_list.index(label))

    out_dataset = pd.DataFrame({'sentence':sentence, 'label':labels})

    return out_dataset
    

def load_data_for_R(dataset_dir):
    pd_dataset = pd.read_csv(dataset_dir,
                converters={'subject_entity':literal_eval, 'object_entity': literal_eval})
    dataset = preprocessing_dataset_for_R(pd_dataset)

    return dataset


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
        print(token)
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

            for i in range(subs_p+1, sube_p):
                sub_mask[i] = 1
            for i in range(objs_p+1, obje_p):
                obj_mask[i] = 1

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

    return RE_Dataset_for_R(all_features, all_label)
