# ---encoding:utf-8---
import torch
import torch.utils.data as Data
import torch.optim as optim
import pickle
from propy.AAComposition import CalculateAAComposition
from propy.PyPro import GetProDes
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs
from tqdm import tqdm
from util import util_file
import torch


class AAindex:
    def __init__(self, length_sequence):
        import pandas as pd
        self.length_sequence = length_sequence
        self.obj = pd.read_csv('./AAindex_12.csv')  # AAidx数据文件与该文件放在一起
        self.pro_name_list = self.obj['AccNo'].tolist()

    def get_feature_name(self):
        feature_name_list = []
        for i in range(1, self.length_sequence + 1):
            for element in self.pro_name_list:
                featureName = 'AAindex_pos%s_%s' % (i, element)
                feature_name_list.append(featureName)
        return feature_name_list

    def main(self, sequence):
        AA_list_sort = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W',
                        'Y']
        AAindex_dict = {}
        for ele in AA_list_sort:
            AAindex_dict[ele] = self.obj[ele].tolist()
        AAindex_dict['X'] = [0] * 12
        feature = []
        for item in sequence:
            if item not in AAindex_dict.keys():
                feature.extend([0] * 12)
            else:
                feature.extend(AAindex_dict[item])
        return feature


class PWAA:  # 该种特征要取上下游等长的情况才行
    def __init__(self):
        self.AA_list = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W',
                        'Y']

    def get_feature_name(self):
        feature_name_list = []
        for element in self.AA_list:
            featureName = 'PWAA_feature_%s' % element
            feature_name_list.append(featureName)
        return feature_name_list

    def main(self, sequence):
        length_up_down = (len(sequence) - 1) / 2
        feature = []
        for aa_char in self.AA_list:
            sum_inter = 0
            if aa_char not in sequence:
                feature.append(0)
            else:
                for sequence_index, sequence_char in enumerate(sequence):
                    if sequence_char == aa_char:
                        j = sequence_index - length_up_down  # 这里10到时要改成上下游的那个L
                        sum_inter = sum_inter + (j + abs(j) / length_up_down)
                c = (1 / (length_up_down * (length_up_down + 1))) * sum_inter
                feature.append(c)
        return feature


# one_hot
def one_hot_encoding(pep):
    amino_acid_list = ['A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y',
                       'V']
    one_hot_encoding = []
    for res in pep:
        one_hot = [0] * 20
        one_hot[amino_acid_list.index(res)] = 1
        one_hot_encoding += one_hot
    one_hot_encoding = one_hot_encoding + [2] * (2000 - len(one_hot_encoding))
    #     one_hot_encoding=one_hot_encoding+[2]*(2000-len(one_hot_encoding))
    return one_hot_encoding


def one_hot_encoding_1000(pep):
    amino_acid_list = ['A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y',
                       'V']
    one_hot_encoding = []
    for res in pep:
        one_hot = [0] * 20
        one_hot[amino_acid_list.index(res)] = 1
        one_hot_encoding += one_hot
    one_hot_encoding = one_hot_encoding + [2] * (1000 - len(one_hot_encoding))
    #     one_hot_encoding=one_hot_encoding+[2]*(2000-len(one_hot_encoding))
    return one_hot_encoding


def get_aaindex_feature(pep):
    if len(pep) > 100:
        pep = pep[:100]
    feature_aaindex = AAindex(100).main(pep)
    feature_aaindex = feature_aaindex + ((100 - len(pep)) * 12) * [0]
    return feature_aaindex

# one_hot_padding
def one_hot_padding(pep):
    amino_acid_list = ['A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y',
                       'V']
    one_hot_encoding = []
    for res in pep:
        one_hot = [0] * 20
        one_hot[amino_acid_list.index(res)] = 1
        one_hot_encoding += one_hot
    one_hot_encoding_left = one_hot_encoding + [2] * (1000 - len(one_hot_encoding))
    if len(pep) % 2 == 0:
        one_hot_encoding_center = [2] * 20 * int((50 - len(pep)) / 2) + one_hot_encoding + [2] * 20 * int(
            (50 - len(pep)) / 2)
    else:
        one_hot_encoding_center = [2] * 20 * int((50 - len(pep) + 1) / 2) + one_hot_encoding + [2] * 20 * int(
            (50 - len(pep) - 1) / 2)
    one_hot_encoding_right = [2] * (1000 - len(one_hot_encoding)) + one_hot_encoding
    one_hot_mix = one_hot_encoding_left + one_hot_encoding_center + one_hot_encoding_right
    return one_hot_mix


def get_morgan_feature(pep):
    mol = Chem.MolFromSequence(pep)
    if mol is None:
        return [0]*512
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=512, useFeatures=False)
    arr = np.zeros((0,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return list(fp)


def get_feature(data, feature_category={'one_hot', 'aac', 'ctd', 'dp'}):
    feature_list = []
    for i in tqdm(range(len(data))):
        pep_feature = []
        # one_hot
        if 'one_hot' in feature_category:
            pep_feature = pep_feature + one_hot_encoding(data[i])
        if 'one_hot_1' in feature_category:
            pep_feature = pep_feature + one_hot_encoding_1000(data[i])
        if 'one_hot_padding' in feature_category:
            pep_feature = pep_feature + one_hot_padding(data[i])
        # aac
        if 'aac' in feature_category:
            pep_feature = pep_feature + list(CalculateAAComposition(data[i]).values())
        if 'pwaa' in feature_category:
            pep_feature = pep_feature + PWAA().main(data[i])
        # dp
        if 'dp' in feature_category:
            pep_feature = pep_feature + list(GetProDes(data[i]).GetDPComp().values())
        # ctd
        if 'ctd' in feature_category:
            pep_feature = pep_feature + list(GetProDes(data[i]).GetCTD().values())
        if 'paac' in feature_category:
            pep_feature = pep_feature + list(GetProDes(data[i]).GetPAAC(lamda=1).values())
        if 'morgan' in feature_category:
            pep_feature = pep_feature + get_morgan_feature(data[i])
        if 'aaindex' in feature_category:
            pep_feature = pep_feature + get_aaindex_feature(data[i])
        feature_list.append(pep_feature)
    return feature_list



# 构造迭代器
def construct_dataset(data, config):
    cuda = config.cuda
    batch_size = config.batch_size
    feature = data
    if cuda:
        feature = torch.cuda.FloatTensor(feature)
    else:
        feature = torch.FloatTensor(feature)

    data_loader = Data.DataLoader(MyDataSet(feature), batch_size=batch_size, shuffle=True,
                                  drop_last=False)
    return data_loader


class MyDataSet(Data.Dataset):
    def __init__(self, feature):
        self.feature = feature

    def __len__(self):
        return len(self.feature)

    def __getitem__(self, idx):
        return self.feature[idx]


def load_data_fc(config):
    path_data_train = config.path_train_data
    path_data_test = config.path_test_data

    # data augmentation
    # sequences_train, labels_train = data_augmentation.augmentation(path_data_train, config, append = False)

    sequences_train, labels_train = util_file.load_tsv_format_data(path_data_train)
    sequences_test, labels_test = util_file.load_tsv_format_data(path_data_test)

    data_train = get_feature(sequences_train, {'morgan', 'aaindex'})
    data_test = get_feature(sequences_test, {'morgan', 'aaindex'})

    data_loader_train = construct_dataset(data_train, config)
    data_loader_test = construct_dataset(data_test, config)

    return data_loader_train, data_loader_test
