import sys
import os
from bilinear import MAPEP
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from preprocess import data_loader, data_loader_feature
from configuration import config as cf
from util import util_metric
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
import pickle
import random


def save_model(model_dict, best_acc, save_dir, save_prefix):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    filename = 'ACC[{:.4f}], {}.pt'.format(best_acc, save_prefix)
    save_path_pt = os.path.join(save_dir, filename)
    print('save_path_pt',save_path_pt)
    torch.save(model_dict, save_path_pt, _use_new_zipfile_serialization=False)
    print('Save Model Over: {}, ACC: {:.4f}\n'.format(save_prefix, best_acc))


def load_data(config):
    residue2idx = pickle.load(open('../data/residue2idx.pkl', 'rb'))
    config.vocab_size = len(residue2idx)
    config.token2index = residue2idx
    train_iter_orgin, test_iter = data_loader.load_data(config)
    train_iter_fc, test_iter_fc = data_loader_feature.load_data_fc(config)
    return train_iter_orgin, test_iter, train_iter_fc, test_iter_fc


def set_seed():
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def get_loss(logits, label, criterion):
    loss = criterion(logits.view(-1, config.num_class), label.view(-1))
    loss = (loss.float()).mean()
    # flooding method
    loss = (loss - 0.06).abs() + 0.06
    return loss


def periodic_test(test_iter, model, criterion, config):
    print('#' * 60 + 'Periodic Test' + '#' * 60)
    test_metric, test_loss, test_repres_list, test_label_list, \
        test_roc_data, test_prc_data = eval_func(test_iter, model, criterion, config)

    print('test current performance')
    print('[ACC,\tPrecision,\tSensitivity,\tSpecificity,\tF1,\tAUC,\tMCC]')
    print(test_metric.numpy())
    print('#' * 60 + 'Over' + '#' * 60)
    return test_metric, test_loss, test_repres_list, test_label_list


def train_func(train_iter, test_iter, model, optimizer, criterion, config):
    steps = 0
    best_acc = 0
    best_performance = 0

    other_train_iter = config.other_train_iter
    other_train_iter_list = list(other_train_iter)

    for epoch in range(1, config.epoch + 1):
        repres_list = []
        label_list = []
        print('current epoch num: ', epoch)
        for i, batch in enumerate(train_iter):
            input, label = batch
            input_o = other_train_iter_list[i]
            logits, output = model(input, input_o)
            repres_list.extend(output.cpu().detach().numpy())
            label_list.extend(label.cpu().detach().numpy())
            loss = get_loss(logits, label, criterion)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            steps += 1

        test_metric, test_loss, test_repres_list, test_label_list = periodic_test(test_iter,
                                                                                  model,
                                                                                  criterion,
                                                                                  config)
        test_acc = test_metric[0]
        if test_acc > best_acc:
            best_acc = test_acc
            best_performance = test_metric
            if config.save_best and best_acc > config.threshold:
                save_model(model.state_dict(), best_acc, config.result_folder, config.learn_name)
                torch.save(model, 'best_model.pt')

        test_label_list = [x + 2 for x in test_label_list]
        repres_list.extend(test_repres_list)
        label_list.extend(test_label_list)

    return best_performance


def eval_func(data_iter, model, criterion, config):
    device = torch.device("cuda" if config.cuda else "cpu")
    label_pred = torch.empty([0], device=device)
    label_real = torch.empty([0], device=device)
    pred_prob = torch.empty([0], device=device)

    other_test_iter_list = list(config.other_test_iter)
    iter_size, corrects, avg_loss = 0, 0, 0
    repres_list = []
    label_list = []

    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(data_iter):
            input, label = batch
            input_o = other_test_iter_list[i]
            logits, output = model(input, input_o)
            repres_list.extend(output.cpu().detach().numpy())
            label_list.extend(label.cpu().detach().numpy())

            loss = criterion(logits.view(-1, config.num_class), label.view(-1))
            loss = (loss.float()).mean()
            avg_loss += loss

            pred_prob_all = F.softmax(logits, dim=1)
            pred_prob_positive = pred_prob_all[:, 1]
            pred_prob_sort = torch.max(pred_prob_all, 1)
            pred_class = pred_prob_sort[1]
            corrects += (pred_class == label).sum()

            iter_size += label.shape[0]

            label_pred = torch.cat([label_pred, pred_class.float()])
            label_real = torch.cat([label_real, label.float()])
            pred_prob = torch.cat([pred_prob, pred_prob_positive])

    metric, roc_data, prc_data = util_metric.caculate_metric(label_pred, label_real, pred_prob)
    avg_loss /= iter_size
    return metric, avg_loss, repres_list, label_list, roc_data, prc_data


def train_test(train_iter, test_iter, config):
    print('=' * 50, 'train-test', '=' * 50)
    print('len(train_iter)', len(train_iter))
    print('len(test_iter)', len(test_iter))
    model = MAPEP(config)
    if config.cuda:
        model.cuda()

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=config.lr, weight_decay=config.reg)
    criterion = nn.CrossEntropyLoss()

    best_performance = train_func(train_iter, test_iter, model, optimizer, criterion, config)
    last_test_metric, last_test_loss, last_test_repres_list, last_test_label_list, \
        last_test_roc_data, last_test_prc_data = eval_func(test_iter, model, criterion, config)
    print('[ACC,\tPrecision,\tSensitivity,\tSpecificity,\tF1,\tAUC,\tMCC]')
    print(last_test_metric.numpy())
    print('*' * 60 + 'The Last Test Over' + '*' * 60)

    return model, best_performance, last_test_metric


def select_dataset():
    # ACP dataset
    # path_train_data = '../data/ACP_dataset/tsv/ACP-Mixed-80-train.tsv'
    # path_test_data = '../data/ACP_dataset/tsv/ACP-Mixed-80-test.tsv'
    # path_train_data = '../data/ACP_dataset/tsv/LEE_Dataset.tsv'
    # path_test_data = '../data/ACP_dataset/tsv/Independent dataset.tsv'
    path_train_data = '../data/ACP_dataset/tsv/ACP2_main_train.tsv'
    path_test_data = '../data/ACP_dataset/tsv/ACP2_main_test.tsv'
    # path_train_data = '../data/ACP_dataset/tsv/ACP2_alternate_train.tsv'
    # path_test_data = '../data/ACP_dataset/tsv/ACP2_alternate_test.tsv'
    return path_train_data, path_test_data


def load_config():
    path_train_data, path_test_data = select_dataset()
    config = cf.get_train_config()
    set_seed()

    '''initialize result folder'''
    result_folder = '../result/' + config.learn_name
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    '''Save all variables in configuration'''
    config.path_train_data = path_train_data
    config.path_test_data = path_test_data
    config.result_folder = result_folder
    return config


if __name__ == '__main__':
    '''load configuration'''
    config = load_config()

    # set device
    torch.cuda.set_device(config.device)

    '''load data'''
    train_iter, test_iter, fc_train_iter, fc_test_iter = load_data(config)
    config.other_train_iter = fc_train_iter
    config.other_test_iter = fc_test_iter

    '''train procedure'''

    model, best_performance, last_test_metric = train_test(train_iter, test_iter, config)

    print('best_performance')
    print('\t[ACC,\tPrecision,\tSensitivity,\tSpecificity,\tF1,\tAUC,\tMCC]')
    print('\t{}'.format(best_performance))

    '''save train result'''
    # save the model if specific conditions are met

    best_acc = best_performance[0]
    last_test_acc = last_test_metric[0]
    if last_test_acc > best_acc:
        best_acc = last_test_acc
        best_performance = last_test_metric
        if config.save_best and best_acc >= config.threshold:
            save_model(model.state_dict(), best_acc, config.result_folder, config.learn_name)



