import pandas
import numpy as np
import os

# use bert to get the CLS token embedding
import torch
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
from datetime import datetime

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased').to(device)
model.eval()

def get_cls_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].detach().cpu()


def data_to_embedding(data, save_path, batch_size=16, train=True, start_idx=0):
    # get the CLS token embedding for each text
    cls_embeddings = []
    data = data
    # batch the data

    for idx in tqdm(range(start_idx, len(data), batch_size)):
        batch = data[idx:idx+batch_size]
        texts = batch['text'].values.tolist()
        try:
            cls_embeddings.append(get_cls_embedding(texts))
        except:
            cls_embeddings.append(torch.zeros((batch_size, 768)))
            print(f'Error at index {idx}')

    cls_embeddings_arr = torch.cat(cls_embeddings, dim=0).numpy()
    
    # save the embeddings
    save_name = 'train' if train else 'test'
    if os.path.exists(f'{save_path}/{save_name}_embeddings.npy'):
        raise ValueError(f'File {save_path}/{save_name}_embeddings.npy already exists')
    
    np.save(f'{save_path}/{save_name}_embeddings.npy', cls_embeddings_arr)
    np.save(f'{save_path}/{save_name}_ids.npy', data['id'].values)
    if train:
        np.save(f'{save_path}/{save_name}_labels.npy', data['label'].values)

    assert len(cls_embeddings_arr) == len(data), f'Length of embeddings {len(cls_embeddings_arr)} does not match length of data {len(data)}'

    #     if idx % (100 * batch_size) == 0:
    #         print(f'{datetime.now()} - {idx} / {len(data)}')

    #         cls_embeddings_arr = torch.cat(cls_embeddings, dim=0).numpy()

    #         # save the embeddings
    #         save_name = 'train' if train else 'test'
    #         if os.path.exists(f'{save_path}/{save_name}_embeddings_{start_idx}_{idx}.npy'):
    #             raise ValueError(f'File {save_path}/{save_name}_embeddings_{start_idx}_{idx}.npy already exists')
            
    #         np.save(f'{save_path}/{save_name}_embeddings_{start_idx}_{idx}.npy', cls_embeddings_arr)
    #         np.save(f'{save_path}/{save_name}_ids_{start_idx}_{idx}.npy', data['id'].values)
    #         if train:
    #             np.save(f'{save_path}/{save_name}_labels_{start_idx}_{idx}.npy', data['label'].values)

    #         cls_embeddings = []
    #         start_idx = idx + 1
    
    # cls_embeddings_arr = torch.cat(cls_embeddings, dim=0).numpy()
    
    # # save the embeddings
    # save_name = 'train' if train else 'test'
    # if os.path.exists(f'{save_path}/{save_name}_embeddings_{start_idx}_{idx}.npy'):
    #     raise ValueError(f'File {save_path}/{save_name}_embeddings_{start_idx}_{idx}.npy already exists')
    
    # np.save(f'{save_path}/{save_name}_embeddings_{start_idx}_{idx}.npy', cls_embeddings_arr)
    # np.save(f'{save_path}/{save_name}_ids_{start_idx}_{idx}.npy', data['id'].values)
    # if train:
    #     np.save(f'{save_path}/{save_name}_labels_{start_idx}_{idx}.npy', data['label'].values)
    
    print(f'{datetime.now()} - {len(data)} / {len(data)}')

def prep_data(save_path, start_idx=0):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    train_data = pandas.read_csv('train.csv')
    test_data = pandas.read_csv('test.csv')

    # remove nan if text or label is nan
    train_data = train_data.dropna(subset=['text', 'label'])
    test_data = test_data.dropna(subset=['text'])

    # data_to_embedding(train_data, save_path, train=True)
    data_to_embedding(test_data, save_path, train=False)

def cat_npys(save_path, batch_size=16, train=True):
    save_name = 'train' if train else 'test'
    embeddings = []
    file_name_format = '{save_name}_embeddings_{start}_{end}.npy'

    for file in os.listdir(save_path):
        if file.startswith(f'{save_name}_embeddings'):
            # sort by start index
            start, end = file.split('.')[0].split('_')[-2:]
            start, end = int(start), int(end)
            embeddings.append((start, end, file))
    embeddings = sorted(embeddings, key=lambda x: x[0])

    embeddings_npy = []
    for start, end, file in embeddings:
        if os.path.exists(f'{save_path}/{file}'):
            load_file = np.load(f'{save_path}/{file}')
            print(f'Loaded {file} with shape {load_file.shape}')
            embeddings_npy.append(load_file)
        else:
            raise ValueError(f'File {save_path}/{file} does not exist')

    embeddings = np.concatenate(embeddings_npy, axis=0)
    np.save(f'{save_path}/{save_name}_embeddings.npy', embeddings)

if __name__ == '__main__':
    prep_data('./raw_data/raw_embeddings3/')
    # cat_npys('./raw_data/raw_embeddings2/', train=True)
    # cat_npys('./raw_data/raw_embeddings2/', train=False)