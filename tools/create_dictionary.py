"""
This code is from Hengyuan Hu's repository.
https://github.com/hengyuan-hu/bottom-up-attention-vqa
"""
from __future__ import print_function
import os
import sys
import json
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset_RAD import Dictionary

def create_dictionary(dataroot):
    dictionary = Dictionary()
    questions = []
    files = [
        'trainset.json',
    ]
    for path in files:
        question_path = os.path.join(dataroot, path)
        qs = json.load(open(question_path))
        for q in qs:
            dictionary.tokenize(q['question'], True)
    return dictionary

def create_glove_embedding_init(idx2word, glove_file):
    word2emb = {}
    with open(glove_file, 'r') as f:
        entries = f.readlines()
    emb_dim = len(entries[0].split(' ')) - 1
    print('embedding dim is %d' % emb_dim)
    weights = np.zeros((len(idx2word), emb_dim), dtype=np.float32)

    for entry in entries:
        vals = entry.split(' ')
        word = vals[0]
        vals = list(map(float, vals[1:]))
        word2emb[word] = np.array(vals)
    for idx, word in enumerate(idx2word):
        if word not in word2emb:
            continue
        weights[idx] = word2emb[word]
    return weights, word2emb


if __name__ == '__main__':
    RAD_dir = 'data_RAD'
    d = create_dictionary(RAD_dir)
    d.dump_to_file(RAD_dir + '/dictionary.pkl')

    d = Dictionary.load_from_file(RAD_dir + '/dictionary.pkl')
    emb_dim = 300
    glove_file = RAD_dir + '/glove/glove.6B.%dd.txt' % emb_dim
    weights, word2emb = create_glove_embedding_init(d.idx2word, glove_file)
    np.save(RAD_dir + '/glove6b_init_%dd.npy' % emb_dim, weights)
