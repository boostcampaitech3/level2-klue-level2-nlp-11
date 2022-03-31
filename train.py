import pickle as pickle
import os
import pandas as pd
import torch
import sklearn
import numpy as np
import random
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments, RobertaConfig, RobertaTokenizer, RobertaForSequenceClassification, BertTokenizer, set_seed
from load_data import *
import wandb
import yaml
import shutil

def klue_re_micro_f1(preds, labels):
    """KLUE-RE micro f1 (except no_relation)"""
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
    no_relation_label_idx = label_list.index("no_relation")
    label_indices = list(range(len(label_list)))
    label_indices.remove(no_relation_label_idx)
    return sklearn.metrics.f1_score(labels, preds, average="micro", labels=label_indices) * 100.0

def klue_re_auprc(probs, labels):
    """KLUE-RE AUPRC (with no_relation)"""
    labels = np.eye(30)[labels]

    score = np.zeros((30,))
    for c in range(30):
        targets_c = labels.take([c], axis=1).ravel()
        preds_c = probs.take([c], axis=1).ravel()
        precision, recall, _ = sklearn.metrics.precision_recall_curve(targets_c, preds_c)
        score[c] = sklearn.metrics.auc(recall, precision)
    return np.average(score) * 100.0

def compute_metrics(pred):
    """ validation을 위한 metrics function """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    probs = pred.predictions

    # calculate accuracy using sklearn's function
    f1 = klue_re_micro_f1(preds, labels)
    auprc = klue_re_auprc(probs, labels)
    acc = accuracy_score(labels, preds) # 리더보드 평가에는 포함되지 않습니다.

    return {
        'micro f1 score': f1,
        'auprc' : auprc,
        'accuracy': acc,
    }

def label_to_num(label):
    num_label = []
    with open('dict_label_to_num.pkl', 'rb') as f:
        dict_label_to_num = pickle.load(f)
    for v in label:
        num_label.append(dict_label_to_num[v])
    
    return num_label

def train():
    config = yaml.load(open("./config.yaml", "r"), Loader=yaml.FullLoader) # load config
    #wandb.init(project='klue',entity='klue') # wandb init
    set_seed(42) # set random seed
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    
    # load model and tokenizer
    MODEL_NAME = config["model"]["name"]
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # load dataset
    dataset = load_data(config["dir"]["train"])
    if config["debugging"] == True:
        dataset = dataset.loc[:1000]

    train_dataset, dev_dataset= split_data(dataset)

    train_label = label_to_num(train_dataset['label'].values)
    dev_label = label_to_num(dev_dataset['label'].values)

    # tokenizing dataset
    tokenized_train = tokenized_dataset(train_dataset, tokenizer, config['input_type'])
    tokenized_dev = tokenized_dataset(dev_dataset, tokenizer, config['input_type'])

    # make batch_indices
    train_sampler = make_sampler(tokenized_train, batch_size=config['TA']['batch_size'], max_pad_len=20)
    valid_sampler = make_sampler(tokenized_dev, batch_size=config['TA']['batch_size'], max_pad_len=100)

    # make dataset for pytorch.
    RE_train_dataset = RE_Dataset(tokenized_train, train_label)
    RE_dev_dataset = RE_Dataset(tokenized_dev, dev_label)
    
    # setting model hyperparameter
    model_config =  AutoConfig.from_pretrained(MODEL_NAME)
    model_config.num_labels = 30

    model =  AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=model_config)
    #print(model.config)
    model = model.to(device)

    Args = config['TA']
    training_args = TrainingArguments(
      output_dir=Args["output_dir"],          # output directory
      save_total_limit=5,              # number of total save model.
      save_steps=500,                 # model saving step.
      num_train_epochs=int(Args["epoch"]),              # total number of training epochs
      learning_rate=float(Args["LR"]),               # learning_rate
      per_device_train_batch_size=int(Args["batch_size"]),  # batch size per device during training
      per_device_eval_batch_size=int(Args["batch_size"]),   # batch size for evaluation
      warmup_steps=500,                # number of warmup steps for learning rate scheduler
      weight_decay=0.01,               # strength of weight decay
      logging_dir=Args["log_dir"],            # directory for storing logs
      logging_steps=500,              # log saving step.
      evaluation_strategy='epoch',
      save_strategy='epoch', # evaluation strategy to adopt during training
                                  # `no`: No evaluation during training.
                                  # `steps`: Evaluate every `eval_steps`.
                                  # `epoch`: Evaluate every end of epoch.
      load_best_model_at_end = True, 
      report_to='wandb'
    )
    
    trainer = BucketTrainer(
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=RE_train_dataset,  # training dataset
        eval_dataset=RE_dev_dataset,
        compute_metrics = compute_metrics      
    )
    trainer.train_sampler = train_sampler
    trainer.valid_sampler = valid_sampler

    # train model
    trainer.train()
    model.save_pretrained(config["dir"]["best_dir"])
    shutil.copy('config.yaml', config["dir"]["best_dir"])

def main():
    train()

if __name__ == '__main__':
    main()
