import pickle
from torch.utils.data import Dataset
from PIL import Image

class GetDataset(Dataset):

    def __init__(self, dataset, transform = None, disjoint_user=False):
        self.train_file_name = 'train_index.pkl'
        self.test_file_name = 'test_index.pkl'
        if disjoint_user:
            self.train_file_name = 'disjoint_user_' + self.train_file_name
            self.test_file_name = 'disjoint_user_' + self.test_file_name
    
        if dataset == "train":
            with open(self.train_file_name, 'rb') as train_index_file:
                self.pairs = pickle.load(train_index_file)
        else:
            with open(self.train_file_name, 'rb') as test_index_file:
                self.pairs = pickle.load(test_index_file)
        self.transform = transform

    def __getitem__(self, index):
        item = self.pairs[index]

        X = Image.open(item[0])
        Y = Image.open(item[1])

        if self.transform is not None:
            X = self.transform(X)
            Y = self.transform(Y)

        return [X, Y, item[2]]

    def __len__(self):
        return len(self.pairs)


class CLIPDataset(Dataset):

    def __init__(self, dataset, transform = None):
        self.train_file_name = 'clip_disjoint_user_train_index.pkl'
        self.test_file_name = 'disjoint_user_test_index.pkl'
    
        if dataset == "train":
            with open(self.train_file_name, 'rb') as train_index_file:
                self.pairs = pickle.load(train_index_file)
        else:
            with open(self.train_file_name, 'rb') as test_index_file:
                self.pairs = pickle.load(test_index_file)
        self.transform = transform

    def __getitem__(self, index):
        item = self.pairs[index]

        X = Image.open(item[0])
        Y = Image.open(item[1])

        if self.transform is not None:
            X = self.transform(X)
            Y = self.transform(Y)

        return [X, Y]

    def __len__(self):
        return len(self.pairs)
