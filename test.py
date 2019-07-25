"""
This code is modified based on Jin-Hwa Kim's repository (Bilinear Attention Networks - https://github.com/jnhwkim/ban-vqa) by Xuan B. Nguyen
"""
import argparse
import torch
from torch.utils.data import DataLoader
import dataset_RAD
import base_model
import utils
import pandas as pd
import os
import json
answer_types = ['CLOSED', 'OPEN', 'ALL']
quesntion_types = ['COUNT', 'COLOR', 'ORGAN', 'PRES', 'PLANE', 'MODALITY', 'POS', 'ABN', 'SIZE', 'OTHER', 'ATTRIB']
def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, 1)[1].data  # argmax
    one_hots = torch.zeros(*labels.size()).to(logits.device)
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ensemble', type=bool, default=False,
                        help='ensemble flag. If True, generate a logit file which is used in the ensemble part')
    # MODIFIABLE MEVF HYPER-PARAMETERS--------------------------------------------------------------------------------
    # Model loading/saving
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--input', type=str, default='saved_models/SAN_MEVF',
                        help='input file directory for loading a model')
    parser.add_argument('--output', type=str, default='results',
                        help='output file directory for saving VQA answer prediction file')
    # Utilities
    parser.add_argument('--epoch', type=int, default=19,
                        help='the best epoch')

    # Gradient accumulation
    parser.add_argument('--batch_size', type=int, default=1,
                        help='batch size')

    # Choices of Attention models
    parser.add_argument('--model', type=str, default='SAN', choices=['BAN', 'SAN'],
                        help='the model we use')

    # Choices of RNN models
    parser.add_argument('--rnn', type=str, default='LSTM', choices=['LSTM', 'GRU'],
                        help='the RNN we use')

    # BAN - Bilinear Attention Networks
    parser.add_argument('--gamma', type=int, default=2,
                        help='glimpse in Bilinear Attention Networks')
    parser.add_argument('--use_counter', action='store_true', default=False,
                        help='use counter module')

    # SAN - Stacked Attention Networks
    parser.add_argument('--num_stacks', default=2, type=int,
                        help='num of stacks in Stack Attention Networks')

    # Utilities - gpu
    parser.add_argument('--gpu', type=int, default=0,
                        help='specify index of GPU using for training, to use CPU: -1')

    # Question embedding
    parser.add_argument('--op', type=str, default='c',
                        help='concatenated 600-D word embedding')

    # Joint representation C dimension
    parser.add_argument('--num_hid', type=int, default=1024,
                        help='dim of joint semantic features')

    # Activation function + dropout for classification module
    parser.add_argument('--activation', type=str, default='relu', choices=['relu'],
                        help='the activation to use for final classifier')
    parser.add_argument('--dropout', default=0.5, type=float, metavar='dropout',
                        help='dropout of rate of final classifier')

    # Train with RAD
    parser.add_argument('--use_RAD', action='store_true', default=False,
                        help='Using TDIUC dataset to train')
    parser.add_argument('--RAD_dir', type=str,
                        help='RAD dir')

    # Optimization hyper-parameters
    parser.add_argument('--eps_cnn', default=1e-5, type=float, metavar='eps_cnn',
                        help='eps - batch norm for cnn')
    parser.add_argument('--momentum_cnn', default=0.05, type=float, metavar='momentum_cnn',
                        help='momentum - batch norm for cnn')

    # input visual feature dimension
    parser.add_argument('--feat_dim', default=64, type=int,
                        help='visual feature dim')

    # Auto-encoder component hyper-parameters
    parser.add_argument('--autoencoder', action='store_true', default=False,
                        help='End to end model?')
    parser.add_argument('--ae_model_path', type=str, default='pretrained_ae.pth',
                        help='the maml_model_path we use')

    # MAML component hyper-parameters
    parser.add_argument('--maml', action='store_true', default=False,
                        help='End to end model?')
    parser.add_argument('--maml_model_path', type=str, default='pretrained_maml.weights',
                        help='the maml_model_path we use')

    # Return args
    args = parser.parse_args()
    return args
# Load questions
def get_question(q, dataloader):
    q = q.squeeze(0)
    str = []
    dictionary = dataloader.dataset.dictionary
    for i in range(q.size(0)):
        str.append(dictionary.idx2word[q[i]] if q[i] < len(dictionary.idx2word) else '_')
    return ' '.join(str)

# Load answers
def get_answer(p, dataloader):
    _m, idx = p.max(1)
    return dataloader.dataset.label2ans[idx.item()]

# Logit computation (for train, test or evaluate)
def get_result(model, dataloader, device, args):
    keys = ['count', 'real', 'true', 'real_percent', 'score', 'score_percent']
    question_types_result = dict((i, dict((j, dict((k, 0.0) for k in keys)) for j in quesntion_types)) for i in answer_types)
    result = dict((i, dict((j, 0.0) for j in keys)) for i in answer_types)
    with torch.no_grad():
        for v, q, a, ans_type, q_types, p_type in iter(dataloader):
            if p_type[0] != "freeform":
                continue
            if args.maml:
                v[0] = v[0].reshape(v[0].shape[0], 84, 84).unsqueeze(1)
            if args.autoencoder:
                v[1] = v[1].reshape(v[1].shape[0], 128, 128).unsqueeze(1)
            v[0] = v[0].to(device)
            v[1] = v[1].to(device)
            q = q.to(device)
            a = a.to(device)
            # inference and get logit
            if args.autoencoder:
                features, _ = model(v, q)
            else:
                features = model(v, q)
            preds = model.classifier(features)
            final_preds = preds
            batch_score = compute_score_with_logits(final_preds, a.data).sum()

            # Compute accuracy for each type answer
            result[ans_type[0]]['count'] += 1.0
            result[ans_type[0]]['true'] += float(batch_score)
            result[ans_type[0]]['real'] += float(a.sum())

            result['ALL']['count'] += 1.0
            result['ALL']['true'] += float(batch_score)
            result['ALL']['real'] += float(a.sum())

            q_types = q_types[0].split(", ")
            for i in q_types:
                question_types_result[ans_type[0]][i]['count'] += 1.0
                question_types_result[ans_type[0]][i]['true'] += float(batch_score)
                question_types_result[ans_type[0]][i]['real'] += float(a.sum())

                question_types_result['ALL'][i]['count'] += 1.0
                question_types_result['ALL'][i]['true'] += float(batch_score)
                question_types_result['ALL'][i]['real'] += float(a.sum())
        for i in answer_types:
            result[i]['score'] = result[i]['true']/result[i]['count']
            result[i]['score_percent'] = round(result[i]['score']*100,1)
            for j in quesntion_types:
                if question_types_result[i][j]['count'] != 0.0:
                    question_types_result[i][j]['score'] = question_types_result[i][j]['true'] / question_types_result[i][j]['count']
                    question_types_result[i][j]['score_percent'] = round(question_types_result[i][j]['score']*100, 1)
                if question_types_result[i][j]['real'] != 0.0:
                    question_types_result[i][j]['real_percent'] = round(question_types_result[i][j]['real']/question_types_result[i][j]['count']*100.0, 1)
    return result, question_types_result

# Test phase
if __name__ == '__main__':
    args = parse_args()
    print(args)
    torch.backends.cudnn.benchmark = True
    args.device = torch.device("cuda:" + str(args.gpu) if args.gpu >= 0 else "cpu")

    # Check if evaluating on TDIUC dataset or VQA dataset
    if args.use_RAD:
        dictionary = dataset_RAD.Dictionary.load_from_file(os.path.join(args.RAD_dir , 'dictionary.pkl'))
        eval_dset = dataset_RAD.VQAFeatureDataset(args.split, args, dictionary)

    batch_size = args.batch_size

    constructor = 'build_%s' % args.model
    model = getattr(base_model, constructor)(eval_dset, args)
    print(model)
    eval_loader = DataLoader(eval_dset, batch_size, shuffle=False, num_workers=0, pin_memory=True, collate_fn=utils.trim_collate)

    def save_questiontype_results(outfile_path, quesntion_types_result):
        for i in quesntion_types_result:
            pd.DataFrame(quesntion_types_result[i]).transpose().to_csv(outfile_path + '/question_type_' + i + '.csv')
    # Testing process
    def process(args, model, eval_loader):
        model_path = args.input + '/model_epoch%s.pth' % args.epoch
        print('loading %s' % model_path)
        model_data = torch.load(model_path)

        # Comment because do not use multi gpu
        # model = nn.DataParallel(model)
        model = model.to(args.device)
        model.load_state_dict(model_data.get('model_state', model_data))

        model.train(False)
        if not os.path.exists(args.output):
            os.makedirs(args.output)
        if args.use_RAD:
            result, quesntion_types_result = get_result(model, eval_loader, args.device, args)
            outfile_path = args.output + '/' + args.input.split('/')[1]
            outfile = outfile_path + '/results.json'
            if not os.path.exists(os.path.dirname(outfile)):
                os.makedirs(os.path.dirname(outfile))
            print(result)
            print(quesntion_types_result)
            json.dump(result, open(outfile, 'w'))
            save_questiontype_results(outfile_path, quesntion_types_result)
        return
    process(args, model, eval_loader)
