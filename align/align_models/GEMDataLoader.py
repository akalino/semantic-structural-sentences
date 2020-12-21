import torch
import pandas as pd


class GEMDataLoader(object):

    def __init__(self, _train_path, _valid_path, _test_path, _batch_sz):
        self.train_path = _train_path
        self.valid_path = _valid_path
        self.test_path = _test_path
        self.batch_size = _batch_sz
        self.train_x, self.train_y, \
        self.valid_x, self.valid_y, \
        self.test_x, self.test_y, self.all_triples = self.process()

    def process(self):
        train_df = pd.read_csv(self.train_path)
        train_df = train_df[train_df['rel_idx'] != -1]
        train_df_x = train_df[['head_idx', 'rel_idx', 'tai_idx']]
        train_df_y = train_df[['sent_id']]
        train_x = torch.from_numpy(train_df_x.values)
        train_y = torch.from_numpy(train_df_y.values)
        valid_df = pd.read_csv(self.valid_path)
        valid_df = valid_df[valid_df['rel_idx'] != -1]
        valid_df_x = valid_df[['head_idx', 'rel_idx', 'tai_idx']]
        valid_df_y = valid_df[['sent_id']]
        valid_x = torch.from_numpy(valid_df_x.values)
        valid_y = torch.from_numpy(valid_df_y.values)
        test_df = pd.read_csv(self.test_path)
        test_df = test_df[test_df['rel_idx'] != -1]
        test_df_x = test_df[['head_idx', 'rel_idx', 'tai_idx']]
        test_df_y = test_df[['sent_id']]
        test_x = torch.from_numpy(test_df_x.values)
        test_y = torch.from_numpy(test_df_y.values)
        all_triples = torch.cat([train_x, valid_x, test_x])
        return train_x, train_y, valid_x, valid_y, test_x, test_y, all_triples

    def get_validation(self):
        return self.valid_x, self.valid_y

    def get_validation_batches(self):
        valid_batch_x = []
        valid_batch_y = []
        permutation = torch.randperm(self.valid_x.size()[0])
        if self.batch_size == -1:
            valid_batch_x.append(self.valid_x)
            valid_batch_y.append(self.valid_y)
        else:
            for i in range(0, self.valid_x.size()[0], self.batch_size):
                indices = permutation[i:i + self.batch_size]
                valid_batch_x.append(self.valid_x[indices])
                valid_batch_y.append(self.valid_y[indices])
        return valid_batch_x, valid_batch_y

    def get_test_batches(self):
        test_batch_x = []
        test_batch_y = []
        permutation = torch.randperm(self.test_x.size()[0])
        if self.batch_size == -1:
            test_batch_x.append(self.test_x)
            test_batch_y.append(self.test_y)
        else:
            for i in range(0, self.test_x.size()[0], self.batch_size):
                indices = permutation[i:i + self.batch_size]
                test_batch_x.append(self.test_x[indices])
                test_batch_y.append(self.test_y[indices])
        return test_batch_x, test_batch_y

    def create_all_batches(self):
        permutation = torch.randperm(self.train_x.size()[0])
        batch_x = []
        batch_y = []
        if self.batch_size == -1:
            batch_x.append(self.train_x)
            batch_y.append(self.train_y)
        else:
            for i in range(0, self.train_x.size()[0], self.batch_size):
                indices = permutation[i:i + self.batch_size]
                batch_x.append(self.train_x[indices])
                batch_y.append(self.train_y[indices])
        return batch_x, batch_y
