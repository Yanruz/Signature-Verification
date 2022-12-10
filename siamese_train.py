# -*- coding: utf-8 -*-
"""Copy of CV_Project_Siamese_Neural_Network.ipynb

## Import and Install all the necessary packages
"""
import torch
import torchvision
from torch.optim import RMSprop, Adam
from torch.utils.data import DataLoader
from torch.optim import RMSprop, Adam
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import argparse
import os

from models import ContrastiveLoss
from dataset import GetDataset
from utils import *

"""
Datasets can be downloaded from this Link: https://cedar.buffalo.edu/NIJ/data/
"""
parser = argparse.ArgumentParser(description='arguments')
parser.add_argument('--model_name', type=str, help='name of the model to use: SiameseConvNet, TransformerNet')
parser.add_argument('--epochs', type=int, default=20, help='number of epochs')
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--lr', type=float, default=0.001, help='learing rate')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight_decay')

def main(args):
    #args
    epochs = args.epochs
    batch_size=args.batch_size
    lr = args.lr
    weight_decay = args.weight_decay
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Preprocessing and Loading Dataset
    if not (os.path.exists('test_index.pkl') and os.path.exists('train_index.pkl')):
        dataset_gen()

    # Model, Criterion, Optimizer
    model = get_model(args.model_name).to(device)
    criterion = ContrastiveLoss()
    optimizer = Adam(model.parameters())

    # Train Dataset
    transform = get_transform(args.model_name)
    print(transform)
    train_dataset = GetDataset("train", transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
    # Test Dataset
    test_dataset = GetDataset("test", transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Define train function
    train(epochs, train_loader, model, optimizer, criterion, test_loader)

    # Save model
    torch.save(model.state_dict(), "./{}_model.pt".format(args.model_name))





    """## Visualize the Model"""
    # Load the saved model
    model = get_model(args.model_name).to(device)
    model.load_state_dict(torch.load("./{}_model.pt".format(args.model_name)))
    model.eval()

    # Load test dataset
    test_dataset = GetDataset("test", transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1)

    # Distance of 20 outputs
    for i, data in enumerate(test_loader,0):
        A = data[0]
        B = data[1]
        label = data[2].float()
        output1,output2 = model(A.cuda(), B.cuda())
        eucledian_distance = torch.nn.functional.pairwise_distance(output1, output2)
        if label==torch.FloatTensor([[0]]):
            label="Orginial"
        else:
            label="Forged"
        concatenated = torch.cat((A,B),0)
        imshow(torchvision.utils.make_grid(concatenated),'Dissimilarity: {:.2f} Label: {}'.format(eucledian_distance.item(),label))
        if i == 20:
            break

    # # # Load test dataset
    # test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

    # # test
    # avg_accuracy, avg_dist, n_batch = test(test_loader, model)
    # print('CEDAR1:')
    # print('Average accuracy across all batches={} at d={}'.format(avg_accuracy / n_batch, avg_dist / n_batch))

if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    main(args)