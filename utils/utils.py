import numpy as np
import json
import jieba
import torch

from tqdm import tqdm

def remove_product_means(source_vec, out_dir, product):
    # load products
    with open(product, 'r', encoding='utf-8') as f:
        products = json.load(f)

    # tokenize products
    product_tokens = []
    for product in products:
        product_tokens += [token for token in jieba.cut(product)]
    product_tokens = set(product_tokens)

    # load tencent embedding
    word_embedding = {}
    with open(source_vec, 'r') as f:
        for line in tqdm(f.readlines()):
            line = line.split(' ')
            word, embedding = line[0], line[1:-1]
            word_embedding[word] = np.array([float(embed) for embed in embedding])

    # build word2idx and idx2word
    word2idx = {'[PAD]': 0, '[UNK]': 1}
    embedding_vec = [np.zeros(len(embedding)), np.zeros(len(embedding))]
    for word, embedding in word_embedding.items():
        word2idx[word] = len(word2idx)
        embedding_vec.append(embedding)
    idx2word = {idx: word for word, idx in word2idx.items()}
    embedding_vec = np.array(embedding_vec)

    # remove product mean
    all_product_token_embedding = []
    for token in product_tokens:
        idx = word2idx.get(token)
        if idx:
            all_product_token_embedding.append(embedding_vec[idx])
    mean_embedding = np.mean(np.array(all_product_token_embedding), axis=0)

    for token in product_tokens:
        idx = word2idx.get(token)
        if idx:
            embedding_vec[idx] -= mean_embedding

    # save files
    torch.save(word2idx, out_dir + 'word2idx.pt')
    torch.save(idx2word, out_dir + 'idx2word.pt')
    torch.save(embedding_vec, out_dir + 'embedding.pt')

remove_product_means('data/embedding/ctb.50d.vec', 'data/FundamentalChemical/',
                     'data/FundamentalChemical/products.json')