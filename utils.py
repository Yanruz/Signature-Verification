import os
import matplotlib.pyplot as plt
import numpy as np
from torch.nn.functional import pairwise_distance
import torch
from random import randrange
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from models import *
import pickle

def dataset_gen(disjoint_user=False, clip=False):
    base_path_org = 'signatures/full_org/original_%d_%d.png'
    base_path_forg = 'signatures/full_forg/forgeries_%d_%d.png'
    assert os.path.exists('signatures')
    data = []
    
    def gen_pairs(n_samples_of_each_class=13500, user_low=1, user_high=55, clip=False):
        data = []
        for _ in range(n_samples_of_each_class):
            anchor_person = randrange(user_low, user_high+1)
            anchor_sign, pos_sign = randrange(1, 25), randrange(1, 25)
            anchor_sign, pos_sign = fix_pair(anchor_sign, pos_sign)
            if clip:
                neg1_sign, neg2_sign = randrange(1, 25), randrange(1, 25)
                neg1_sign, neg2_sign = fix_pair(neg1_sign, neg2_sign)   
                positive = (base_path_org % (anchor_person, anchor_sign), base_path_org % (anchor_person, pos_sign))
                negative = (base_path_forg % (anchor_person, neg1_sign), base_path_forg % (anchor_person, neg2_sign))         
            else:
                neg_sign = randrange(1, 25)
                positive = (base_path_org % (anchor_person, anchor_sign), base_path_org % (anchor_person, pos_sign), 1)
                negative = (base_path_org % (anchor_person, anchor_sign), base_path_forg % (anchor_person, neg_sign), 0)
            data.append(positive)
            data.append(negative)
        return list(set(data))
    
    train_file_name = 'train_index.pkl'
    test_file_name = 'test_index.pkl'
    
    if disjoint_user:
        train_file_name = 'disjoint_user_' + train_file_name
        test_file_name = 'disjoint_user_' + test_file_name
        train = gen_pairs(n_samples_of_each_class=int(13500*0.65), user_low=1, user_high=35, clip=clip)
        test = gen_pairs(n_samples_of_each_class=int(13500*0.35), user_low=36, user_high=55)
        if clip: train_file_name = 'clip_' + train_file_name
    else:
        data = gen_pairs()
        train, test = train_test_split(data, test_size=0.35)


    with open(train_file_name, 'wb') as train_index_file:
        pickle.dump(train, train_index_file)

    with open(test_file_name, 'wb') as test_index_file:
        pickle.dump(test, test_index_file)
    
def fix_pair(x, y):
    if x == y:
        return fix_pair(x, randrange(1, 24))
    else:
        return x, y

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


def get_model(model_name):
    if model_name == 'SiameseConvNet':
        return SiameseConvNet()
    elif model_name == 'TransformerNet':
        return TransformerNet()
    elif 'vit' in model_name:
        return vit_base()
    elif 'resnet50_pretrained' == model_name:
        return resnet_50(True)
    elif 'resnet50' == model_name:
        return resnet50(False)
    elif 'resnet' == model_name:
        return resnet_18(False)
    elif 'resnet_pretrained' == model_name:
        return resnet_18(True)

def get_transform(model_name):
    num_output_channels = 1
    if model_name == 'SiameseConvNet':
        image_size = (220, 155)
    elif model_name == 'TransformerNet':
        image_size = (224, 224)
    elif 'vit' in model_name:
        image_size = (224, 224)
        num_output_channels = 3
    elif 'resnet' in model_name:
        image_size = (224, 224)
        num_output_channels = 3

    transform =  transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize(image_size),
                    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    transforms.Grayscale(num_output_channels=num_output_channels),
                    transforms.ToTensor()
                    ])
                    # transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))
                    # transforms.RandomRotation(degrees=(0, 90))
    return transform, image_size

def train(epochs, train_loader, model, optimizer, criterion, test_loader, clip=False, few_shot=False):
    for epoch in range(1, epochs+1):
        total_loss = 0
        for batch_index, data in enumerate(train_loader):
            optimizer.zero_grad()

            A, B = data[0].cuda(), data[1].cuda()
            if clip:
                logits, label = model(A, B)
                label = label.cuda()
                loss = criterion(logits, label)
                total_loss += loss.item()
            else:
                label = data[2].float().cuda()
                f_A, f_B = model(A, B)
                loss = criterion(f_A, f_B, label)
                total_loss += loss.item()

            # if batch_index%100 == 0:
            #     print('Epoch {}, batch {}, loss={}'.format(epoch, batch_index, loss.item()) ,flush=False)
            loss.backward()
            optimizer.step()
            if few_shot:
                break
        print('Average epoch loss={}'.format(total_loss / len(train_loader)),flush=False)
        if test_loader:
            accuracy, dist = test(test_loader, model, clip)
            print('accuracy across all batches={} at d={}'.format(accuracy, dist),flush=False)
    return model



def test(test_loader, model, clip=False):
    preds = []
    targets = []
    for batch_index, data in enumerate(test_loader):
        A = data[0]
        B = data[1]
        labels = data[2].long()

        if clip: 
            f_a, f_b = model.forward_features(A.cuda(), B.cuda())
            a = torch.diagonal(f_a @ f_a.T)
            b = torch.diagonal(f_a @ f_b.T)
            dist = pairwise_distance(a, b)
            # dist = torch.diagonal(logits)
        else:
            f_a, f_b = model.forward(A.cuda(), B.cuda())
            dist = pairwise_distance(f_a, f_b)
        preds.append(dist.detach().cpu().numpy())
        targets.append(labels.detach().numpy())
    avg_accuracy, avg_dist = compute_accuracy_roc(np.concatenate(preds, axis=0), np.concatenate(targets, axis=0))
    return avg_accuracy, avg_dist