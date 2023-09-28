import torch
import pickle
import torch.utils.data as Data
from preprocess.data_loader_feature import get_feature
from configuration import config as cf
from model.bilinear import MAPEP
import torch.nn.functional as F



token2index = pickle.load(open('./data/residue2idx.pkl', 'rb'))

def transform_token2index(sequences):
    for i, seq in enumerate(sequences):
        sequences[i] = list(seq)

    token_list = list()
    max_len = 0
    for seq in sequences:
        seq_id = [token2index[residue] for residue in seq]
        token_list.append(seq_id)
        if len(seq) > max_len:
            max_len = len(seq)

    return token_list, max_len

def make_data_with_unified_length(token_list, max_len):
    max_len = 52 + 2  # add [CLS] and [SEP]
    data = []
    for i in range(len(token_list)):
        token_list[i] = [token2index['[CLS]']] + token_list[i] + [token2index['[SEP]']]
        n_pad = max_len - len(token_list[i])
        token_list[i].extend([0] * n_pad)
        data.append(token_list[i])
    return data

class MyDataSet(Data.Dataset):
    def __init__(self, input_ids):
        self.input_ids = input_ids

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx]


def construct_dataset(data):
    batch_size = 1
    input_ids = data
    input_ids = torch.LongTensor(input_ids)

    data_loader = Data.DataLoader(MyDataSet(input_ids), batch_size=batch_size, drop_last=False)
    return data_loader

def construct_dataset_fc(data):
    batch_size = 1
    feature = data
    feature = torch.FloatTensor(feature)
    data_loader = Data.DataLoader(MyDataSet(feature), batch_size=batch_size, drop_last=False)
    return data_loader

def load_config():
    config = cf.get_train_config()
    config.max_len = 54
    return config


def process(seq):
    token_list, max_len = transform_token2index([seq])
    data_sample = make_data_with_unified_length(token_list, max_len)
    data_loader_sample = construct_dataset(data_sample)

    data_sample_fc = get_feature(seq, {'morgan', 'aaindex'})
    data_loader_sample_fc = construct_dataset_fc(data_sample_fc)

    input = list(data_loader_sample)[0]
    input_o = list(data_loader_sample_fc)[0]
    return input, input_o


if __name__ == '__main__':
    # Example peptide sequence
    peptide_sequence = 'FAKKLLAKALKL'
    input, input_o = process(peptide_sequence)
    config = load_config()
    config.vocab_size = len(token2index)
    model = MAPEP(config)
    state_dict = torch.load('ME-PEP_model.pt')
    model.load_state_dict(state_dict)
    device = torch.device('cuda')
    model = model.to(device)
    input = input.to(device)
    input_o = input_o.to(device)
    model.eval()
    with torch.no_grad():
        output, _ = model(input, input_o)
        pred_prob_all = F.softmax(output, dim=1)
        pred_prob_positive = pred_prob_all[:, 1]
        print('*' * 10 + ' the classification result ' + '*' * 10)
        if pred_prob_positive > 0.5:
            print('This seq is a anticancer peptide')
        else:
            print('This seq is not a anticancer peptide')