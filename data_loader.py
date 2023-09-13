import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from data_utils import encode_sequence, encode_categorical, generate_dict, split_train_test


class DatasetBCRSORT(Dataset):
    def __init__(self, data_in):
        # encoding sequence
        data_in.loc[:, 'cdr3_aa'] = data_in.apply(lambda row: encode_sequence(row['cdr3_aa']), axis=1)

        # encoding categorical variables
        encoding_dict = generate_dict()
        categorical_variable = ['V_gene', 'J_gene', 'isotype']
        if 'label' in data_in.columns.values.tolist():
            categorical_variable += ['label']

        for variable in categorical_variable:
            data_in.loc[:, variable] = data_in.apply(lambda row: encode_categorical(row[variable], encoding_dict[variable]), axis=1)

        if 'label' in data_in.columns.values.tolist():
            self.label = torch.nn.functional.one_hot(torch.from_numpy(data_in['label'].values).type(torch.LongTensor), num_classes=3)
            self.data = data_in.drop(['label'], axis=1, inplace=False)
        else:
            self.label = None
            self.data = data_in

        self.dataset_size = data_in.shape[0]

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        data = self.data.iloc[idx].values.tolist()
        seq = data[0]
        features = data[1:]
        data = torch.from_numpy(np.concatenate([seq, features]))

        if self.label is None:
            label = None
        else:
            label = self.label[idx]

        return data, label


def load_dataset(file_in, mode, test_ratio=None):
    if isinstance(file_in, str):
        data_in = pd.read_csv(file_in)
    elif isinstance(file_in, list):
        data_in = pd.concat((pd.read_csv(f) for f in file_in))

    if mode == 'predict':
        col_requirement = ['cdr3_aa', 'V_gene', 'J_gene', 'isotype']
    elif mode == 'train':
        col_requirement = ['cdr3_aa', 'V_gene', 'J_gene', 'isotype', 'label']
    else:
        raise Exception("Invalid mode to load dataset")

    col_input = data_in.columns.values
    for col in col_requirement:
        if col not in col_input:
            raise Exception("No %s feature in the input data" % col)

    data_in = data_in.loc[:, col_requirement]
    data_in.drop_duplicates(subset=col_requirement, inplace=True)

    if test_ratio is None:
        return DatasetBCRSORT(data_in=data_in)
    else:
        data_train, data_val, data_test = split_train_test(data_in, test_ratio, seed=12345)
        TrainDatasetBCR = DatasetBCRSORT(data_in=data_train)
        ValDatasetBCR = DatasetBCRSORT(data_in=data_val)
        TestDatasetBCR = DatasetBCRSORT(data_in=data_test)

        return TrainDatasetBCR, ValDatasetBCR, TestDatasetBCR
