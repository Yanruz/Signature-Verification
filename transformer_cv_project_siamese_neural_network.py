# -*- coding: utf-8 -*-
"""Copy of CV_Project_Siamese_Neural_Network.ipynb

## Import and Install all the necessary packages
"""

import torchvision
from torch.optim import RMSprop, Adam
from torch.utils.data import DataLoader
import numpy as np
import torch.nn as nn
from torch.nn import Linear, Conv2d, MaxPool2d, LocalResponseNorm, Dropout
from torch.nn.functional import relu
from torch.nn import Module
import torch.nn.functional as F
import torch

from PIL import Image
from PIL.ImageOps import invert
import numpy as np
from torch import Tensor

from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import pickle
from random import randrange

from torch.optim import RMSprop, Adam
from torch.utils.data import DataLoader
from torch import save
from torch import load
import matplotlib.pyplot as plt
from torchmetrics.classification import BinaryAccuracy
import torchvision.transforms as transforms




"""### Additional Utility Functions """

def invert_image_path(path):
    image_file = Image.open(path)  # open colour image
    image_file = image_file.convert('L')
    # image_array = np.array(invert(image_file))
    # for i in range(image_array.shape[0]):
    #     for j in range(image_array.shape[1]):
    #         if image_array[i][j] > 50:
    #             image_array[i][j] = 255
    #         else:
    #             image_array[i][j] = 0
    # image_array = image_array / 255.0
    # return Tensor(image_array).view(1, 224, 224)
    return image_file

def imshow(img,text=None):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic',fontweight='bold',bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()    

def show_plot(iteration,loss):
    plt.plot(iteration,loss)
    plt.show()





"""## Load Dataset :

Datasets can be downloaded from this Link: https://cedar.buffalo.edu/NIJ/data/
"""

# # Commented out IPython magic to ensure Python compatibility.
# !ls
# from google.colab import drive
# drive.mount('/content/drive')

# # %cd  /content/drive/'My Drive'/signatures/
# !ls

"""### Preprocessing and Loading Dataset"""

base_path_org = 'signatures/full_org/original_%d_%d.png'
base_path_forg = 'signatures/full_forg/forgeries_%d_%d.png'


# def fix_pair(x, y):
#     if x == y:
#         return fix_pair(x, randrange(1, 24))
#     else:
#         return x, y


# data = []
# n_samples_of_each_class = 13500

# for _ in range(n_samples_of_each_class):
#     anchor_person = randrange(1, 55)
#     anchor_sign = randrange(1, 24)
#     pos_sign = randrange(1, 24)
#     anchor_sign, pos_sign = fix_pair(anchor_sign, pos_sign)
#     neg_sign = randrange(1, 24)
#     positive = [base_path_org % (anchor_person, anchor_sign), base_path_org % (anchor_person, pos_sign), 1]
#     negative = [base_path_org % (anchor_person, anchor_sign), base_path_forg % (anchor_person, neg_sign), 0]
#     data.append(positive)
#     data.append(negative)

# train, test = train_test_split(data, test_size=0.15)
# with open('train_index.pkl', 'wb') as train_index_file:
#     pickle.dump(train, train_index_file)

# with open('test_index.pkl', 'wb') as test_index_file:
#     pickle.dump(test, test_index_file)

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
        X = invert_image_path(item[0])
        Y = invert_image_path(item[1])

        if self.transform is not None:
            X = self.transform(X)
            Y = self.transform(Y)

        return [X, Y, item[2]]

    def __len__(self):
        return len(self.pairs)





"""##Transformer model"""

from transformers import ViTConfig, ViTModel

class TransformerNet(Module):
    def __init__(self):
        super(TransformerNet, self).__init__()

        configuration = ViTConfig(num_channels = 1)
        # Initializing a model (with random weights) from the vit-base-patch16-224 style configuration
        self.vit = ViTModel(configuration)

         # Set up model architecture
        # self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224')
        self.fc_1 = nn.Linear(768, 256)
        self.fc_out = nn.Linear(256, 128)
        self.dropout = nn.Dropout(0.5)
    
    def forward_once(self,input):
        # Type of output `BaseModelOutputWithPooling`
        vit_outputs = self.vit(input)
        
        # Shape of pooler_output: (batch_size, hidden_size)
        pooler_output = vit_outputs.pooler_output
        
        # Pass through the linear layout to predict the class
        # Shape of output: (batch_size, classes_)
        outputs = torch.relu(self.fc_1(pooler_output))
        outputs = self.fc_out(self.dropout(outputs))
        
        return outputs

    def forward(self, input1, input2):
        # forward pass of input 1
        output1 = self.forward_once(input1)
        # forward pass of input 2
        output2 = self.forward_once(input2)
        return output1, output2


"""## Define Loss Function"""

class ContrastiveLoss(torch.nn.Module):

    def __init__(self, margin=2):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive






"""## Train the Model"""

# Train and save model

# Model, Criterion, Optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TransformerNet().cuda()
criterion = ContrastiveLoss()
optimizer = Adam(model.parameters())

# Train dataset
train_dataset = GetDataset("train", transform=transforms.Compose([transforms.Resize((224,224)),
                                                                      transforms.ToTensor()
                                                                      ]))
train_loader = DataLoader(train_dataset, batch_size=48, shuffle=True)
    
# Define train function
def train():
	for epoch in range(1, 11):
		total_loss = 0
		for batch_index, data in enumerate(train_loader):
			A = data[0].cuda()
			B = data[1].cuda()
			optimizer.zero_grad()
			label = data[2].float().cuda()

			f_A, f_B = model(A, B)

			loss = criterion(f_A, f_B, label)
			total_loss += loss.item()

			print('Epoch {}, batch {}, loss={}'.format(epoch, batch_index, loss.item()))
			loss.backward()
			optimizer.step()
		print('Average epoch loss={}'.format(total_loss / (len(train_dataset) // 16)))

train()

# Save model
torch.save(model.state_dict(), "./transformer_model.pt")





"""## Test the Model"""

# Load the saved model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TransformerNet().cuda()
model.load_state_dict(torch.load("./transformer_model.pt"))
model.eval()

# Load test dataset
test_dataset = GetDataset("test", transform=transforms.Compose([transforms.Resize((224,224)),
                                                                      transforms.ToTensor()
                                                                      ]))
test_loader = DataLoader(test_dataset,num_workers=6,batch_size=1,shuffle=True)

# Distance of 20 outputs
for i, data in enumerate(test_loader,0):
    A = data[0]
    B = data[1]
    label = data[2].float()
    output1,output2 = model(A.cuda(), B.cuda())
    eucledian_distance = F.pairwise_distance(output1, output2)
    if label==torch.FloatTensor([[0]]):
        label="Orginial"
    else:
        label="Forged"
    concatenated = torch.cat((A,B),0)
    imshow(torchvision.utils.make_grid(concatenated),'Dissimilarity: {:.2f} Label: {}'.format(eucledian_distance.item(),label))
    if i == 20:
        break






"""## Compute Accuracy

"""

# compute accuracy
avg_accuracy = 0
avg_dist = 0
n_batch = 0

def compute_accuracy_roc(predictions, labels):
    pos = np.sum(labels == 1)
    neg = np.sum(labels == 0)

    max_dist = np.max(predictions)
    min_dist = np.min(predictions)
    step = 0.001

    max_accuracy = 0
    optimal_dist = 0

    for dist in np.arange(min_dist, max_dist + step, step):
        idx1 = predictions.ravel() <= dist
        idx2 = predictions.ravel() > dist

        true_pos = float(np.sum(labels[idx1] == 1)) / pos
        true_neg = float(np.sum(labels[idx2] == 0)) / neg

        accuracy = 0.5 * (true_pos + true_neg)

        if accuracy > max_accuracy:
            max_accuracy = accuracy
            optimal_dist = dist

    return max_accuracy, optimal_dist


# Load test dataset
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=True)


# test
def test():
    global avg_accuracy, avg_dist, n_batch
    for batch_index, data in enumerate(test_loader):
        A = data[0]
        B = data[1]
        labels = data[2].long()

        f_a, f_b = model.forward(A.cuda(), B.cuda())
        dist = F.pairwise_distance(f_a, f_b)

        accuracy, dist = compute_accuracy_roc(dist.detach().cpu().numpy(), labels.detach().numpy())
        print('Max accuracy for batch {} = {} at d = {}'.format(batch_index, accuracy, dist))
        avg_accuracy += accuracy
        avg_dist += dist
        n_batch += 1

print('CEDAR1:')
test()
print('Average accuracy across all batches={} at d={}'.format(avg_accuracy / n_batch, avg_dist / n_batch))
