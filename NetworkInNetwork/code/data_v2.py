import os
import torch
# import cPickle as pickle
from six.moves import cPickle as pickle
import numpy as np
import torchvision.transforms as transforms
import platform


class dataset():

    def __init__(self, root=None, train=True):
        self.root = root
        self.train = train
        self.transform = transforms.ToTensor()
        if self.train:
            self.train_data, self.train_labels = self.load_CIFAR10(root, train)
            self.train_data = torch.from_numpy(
                self.train_data.astype('float32')).permute(0, 3, 1, 2)
            self.train_labels = torch.tensor(
                self.train_labels, dtype=torch.long)
        else:
            self.test_data, self.test_labels = self.load_CIFAR10(root, train)
            self.test_data = torch.from_numpy(
                self.test_data.astype('float32')).permute(0, 3, 1, 2)
            self.test_labels = torch.tensor(self.test_labels, dtype=torch.long)

    def load_CIFAR10(self, ROOT, TRAIN):
        """ load all of cifar """
        if TRAIN:
            xs = []
            ys = []
            for b in range(1, 6):
                f = os.path.join(ROOT, 'data_batch_%d' % (b,))
                X, Y = self.load_CIFAR_batch(f)
                xs.append(X)
                ys.append(Y)
            Xtr = np.concatenate(xs)
            Ytr = np.concatenate(ys)
            del X, Y
            # Ytr_onehot = np.zeros((Ytr.shape[0], 10))
            # Ytr_onehot[np.arange(Ytr.shape[0]), Ytr] = 1
            # del Ytr
            # return Xtr, Ytr_onehot
            return Xtr, Ytr
        else:
            Xte, Yte = self.load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
            # Yte_onehot = np.zeros((Yte.shape[0], 10))
            # Yte_onehot[np.arange(Yte.shape[0]), Yte] = 1
            # del Yte
            # return Xte, Yte_onehot
            return Xte, Yte

    def load_pickle(self, f):
        version = platform.python_version_tuple()

        if version[0] == '2':
            return pickle.load(f)
        elif version[0] == '3':
            return pickle.load(f, encoding='latin1')
        raise ValueError("invalid python version: {}".format(version))

    def load_CIFAR_batch(self, filename):
        """ load single batch of cifar """

        with open(filename, 'rb') as f:
            datadict = self.load_pickle(f)
            X = datadict['data']
            Y = datadict['labels']
            X = X.reshape(10000, 3, 32, 32).transpose(
                0, 2, 3, 1).astype("float")
            Y = np.array(Y)
            return X, Y

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def __getitem__(self, index):
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        return img, target
