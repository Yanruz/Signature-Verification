import os
import matplotlib.pyplot as plt
import numpy as np
from torch.nn.functional import pairwise_distance
from random import randrange
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from models import *
import pickle

def dataset_gen():
    base_path_org = 'signatures/full_org/original_%d_%d.png'
    base_path_forg = 'signatures/full_forg/forgeries_%d_%d.png'
    assert os.path.exists('signatures')
    data = []
    n_samples_of_each_class = 13500

    for _ in range(n_samples_of_each_class):
        anchor_person = randrange(1, 55)
        anchor_sign = randrange(1, 24)
        pos_sign = randrange(1, 24)
        anchor_sign, pos_sign = fix_pair(anchor_sign, pos_sign)
        neg_sign = randrange(1, 24)
        positive = [base_path_org % (anchor_person, anchor_sign), base_path_org % (anchor_person, pos_sign), 1]
        negative = [base_path_org % (anchor_person, anchor_sign), base_path_forg % (anchor_person, neg_sign), 0]
        data.append(positive)
        data.append(negative)
    train, test = train_test_split(data, test_size=0.15)

    with open('train_index.pkl', 'wb') as train_index_file:
        pickle.dump(train, train_index_file)

    with open('test_index.pkl', 'wb') as test_index_file:
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

def get_transform(model_name):
	if model_name == 'SiameseConvNet':
		image_size = (220, 155)
	elif model_name == 'TransformerNet':
		image_size = (224, 224)
	transform =  transforms.Compose([transforms.Resize(image_size),
                    transforms.Grayscale(num_output_channels=1),
                    transforms.ToTensor()])
	return transform

def train(epochs, train_loader, model, optimizer, criterion, test_loader=None):
	for epoch in range(1, epochs+1):
		total_loss = 0
		for batch_index, data in enumerate(train_loader):
			A = data[0].cuda()
			B = data[1].cuda()
			optimizer.zero_grad()
			label = data[2].float().cuda()

			f_A, f_B = model(A, B)

			loss = criterion(f_A, f_B, label)
			total_loss += loss.item()

			if batch_index%10 == 0:
				print('Epoch {}, batch {}, loss={}'.format(epoch, batch_index, loss.item()) ,flush=False)
			loss.backward()
			optimizer.step()
		print('Average epoch loss={}'.format(total_loss / len(train_loader)),flush=False)
		if test_loader:
			avg_accuracy, avg_dist, n_batch = test(test_loader, model)
			print('Average accuracy across all batches={} at d={}'.format(avg_accuracy / n_batch, avg_dist / n_batch),flush=False)



def test(test_loader, model):
    avg_accuracy, avg_dist, n_batch = 0,0,0
    for batch_index, data in enumerate(test_loader):
        A = data[0]
        B = data[1]
        labels = data[2].long()

        f_a, f_b = model.forward(A.cuda(), B.cuda())
        dist = pairwise_distance(f_a, f_b)

        accuracy, dist = compute_accuracy_roc(dist.detach().cpu().numpy(), labels.detach().numpy())
        if batch_index%10 == 50:
            print('Max accuracy for batch {} = {} at d = {}'.format(batch_index, accuracy, dist),flush=False)
        avg_accuracy += accuracy
        avg_dist += dist
        n_batch += 1
    return avg_accuracy, avg_dist, n_batch