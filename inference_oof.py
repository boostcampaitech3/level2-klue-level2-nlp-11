from transformers import AutoTokenizer, AutoConfig
import torch
from torch.utils.data import DataLoader
from load_data_for_R import *
from model_for_R import *
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F

def inference(model, tokenized_sent, device):
	dataloader = DataLoader(tokenized_sent, batch_size=32, shuffle=False, collate_fn=collate_fn)
	model.eval()
	output_pred = []
	output_prob = []
	for i, data in enumerate(tqdm(dataloader)):
		with torch.no_grad():
			outputs = model(
			input_ids = data['input_ids'].to(device),
			attention_mask = data['attention_mask'].to(device),
			sub_mask = data['sub_mask'].to(device),
			obj_mask = data['obj_mask'].to(device),
			labels = None
			)

			logits = outputs[0]
			prob = F.softmax(logits, dim=-1).detach().cpu().numpy()
			output_prob.append(prob)

	return np.concatenate(output_prob, axis=0)

def num_to_label(label):
	"""
	숫자로 되어 있던 class를 원본 문자열 라벨로 변환 합니다.
	"""
	origin_label = []
	with open('dict_num_to_label.pkl', 'rb') as f:
		dict_num_to_label = pickle.load(f)
	for v in label:
		origin_label.append(dict_num_to_label[v])
	return origin_label


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    MODEL_NAME = 'klue/roberta-large'
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model_config =  AutoConfig.from_pretrained(MODEL_NAME)
    model = R_BigBird(model_config, 0.1)

    dataset = load_data('../dataset/test/test_data.csv')
    tokenized_test = tokenized_dataset(dataset, tokenizer)
    tokenized_test = make_entity_mask(tokenized_test)
    test_label = list(map(int,dataset['label'].values))
    RE_dataset_test = RE_Dataset(tokenized_test, test_label)

    probs = []
    for fold in range(1,6):
        model.load_state_dict(torch.load(f'./best_model/{fold}_best_model/pytorch_model.bin'))
        model.to(device) 
        output_prob = inference(model, RE_dataset_test, device)
        probs.append(output_prob)
    
    prob = sum(probs)/5
    pred = np.argmax(prob, axis=-1).tolist()
    prob = prob.tolist()
    original_label = num_to_label(pred)
    
    test = pd.read_csv('../dataset/test/test_data.csv')
    test_id = test['id'].to_list()
    output = pd.DataFrame({'id':test_id, 'pred_label':original_label, 'probs':prob})

    output.to_csv('./prediction/submission.csv', index=False)

    print('Finish!!!!!!!!!')

if __name__ == '__main__':
    main()
