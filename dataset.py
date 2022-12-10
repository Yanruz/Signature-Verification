import pickle
from torch.utils.data import Dataset
from PIL import Image

class GetDataset(Dataset):

    def __init__(self, dataset, transform = None):
        if dataset == "train":
            with open('train_index.pkl', 'rb') as train_index_file:
                self.pairs = pickle.load(train_index_file)
        else:
            with open('test_index.pkl', 'rb') as test_index_file:
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

