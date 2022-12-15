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

from models import ContrastiveLoss, CLIPModel, BCEWithLogitsLoss
from dataset import GetDataset, CLIPDataset
from utils import *

"""
Datasets can be downloaded from this Link: https://cedar.buffalo.edu/NIJ/data/
"""
parser = argparse.ArgumentParser(description='arguments')
parser.add_argument('--model_name', type=str, default='SiameseConvNet', help='name of the model to use: SiameseConvNet, TransformerNet, vit_base')
parser.add_argument('--epochs', type=int, default=20, help='number of epochs')
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--lr', type=float, default=0.001, help='learing rate')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight_decay')
parser.add_argument('--disjoint_user', type=bool, default=True)
parser.add_argument('--clip', type=bool, default=False)
parser.add_argument('--loss', type=str, default='contrastive')
parser.add_argument('--fs', type=bool, default=False, help='enable few shot training')


def main(args):
    #args
    epochs = args.epochs
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Preprocessing and Loading Dataset
    if args.clip:
        print("Clip Training")
        dataset_gen(args.disjoint_user, args.clip)
    elif not (os.path.exists('test_index.pkl') and os.path.exists('train_index.pkl')) and args.disjoint_user == False:
        dataset_gen()
    elif not (os.path.exists('disjoint_user_test_index.pkl') and os.path.exists('disjoint_user_train_index.pkl'))and args.disjoint_user: 
        dataset_gen(args.disjoint_user)

    # Train Dataset
    transform, image_size = get_transform(args.model_name)
    train_dataset = GetDataset("train", transform=transform, disjoint_user=args.disjoint_user)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        
    # Test Dataset
    test_dataset = GetDataset("test", transform=transform, disjoint_user=args.disjoint_user)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)


    # Model, Criterion, Optimizer
    model = get_model(args.model_name).to(device)
    if args.loss == 'contrastive':
        criterion = ContrastiveLoss()
    elif args.loss == 'BCE':
        criterion = BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr = args.lr, weight_decay=args.weight_decay)

    if args.clip:
        clip_model = CLIPModel(model).to(device)
        criterion = CLIPLoss()
        train_dataset = CLIPDataset("train", transform)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # Define train function
    if args.clip:
        model = train(epochs, train_loader, clip_model, optimizer, criterion, test_loader, args.clip) 
    else:
        model = train(epochs, train_loader, model, optimizer, criterion, test_loader, few_shot=args.fs)

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
    print(args, flush=False)
    main(args)