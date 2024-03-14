import pandas
import numpy as np
import os

# use bert to get the CLS token embedding
import torch
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
from datetime import datetime

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
model.eval()

def get_cls_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[0][0].numpy()


def data_to_embedding(data, save_path, train=True, start_idx=0):
    # get the CLS token embedding for each text
    cls_embeddings = []
    data = data
    for idx in tqdm(range(start_idx, len(data))):
        text = data['text'].values[idx]
        try:
            cls_embeddings.append(get_cls_embedding(text))
        except:
            cls_embeddings.append(np.zeros(768))
            print(f'Error at index {idx}')

        if idx % (len(data) // 10) == 0 or idx == len(data) - 1:
            print(f'{datetime.now()} - {idx} / {len(data)}')

            cls_embeddings_arr = np.array(cls_embeddings)

            # save the embeddings
            save_name = 'train' if train else 'test'
            if os.path.exists(f'{save_path}/{save_name}_embeddings_{idx}.npy'):
                raise ValueError(f'File {save_path}/{save_name}_embeddings_{idx}.npy already exists')
            
            np.save(f'{save_path}/{save_name}_embeddings_{idx}.npy', cls_embeddings_arr)
            np.save(f'{save_path}/{save_name}_ids_{idx}.npy', data['id'].values)
            if train:
                np.save(f'{save_path}/{save_name}_labels_{idx}.npy', data['label'].values)


save_path = './raw_data/raw_embeddings/'
if not os.path.exists(save_path):
    os.makedirs(save_path)

train_data = pandas.read_csv('train.csv')
test_data = pandas.read_csv('test.csv')

# remove nan if text or label is nan
train_data = train_data.dropna(subset=['text', 'label'])
test_data = test_data.dropna(subset=['text'])

data_to_embedding(train_data, save_path, train=True, start_idx=16609)
data_to_embedding(test_data, save_path, train=False)