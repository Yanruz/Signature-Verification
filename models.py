from transformers import ViTConfig, ViTModel
from transformers import ViTFeatureExtractor, ViTForImageClassification
import torch.nn as nn 
import torch.nn.functional as F
import torch
from torchvision.models import resnet18, resnet50

class resnet_18(nn.Module):
    def __init__(self, weights=False):
        super(resnet_18, self).__init__()
        self.model = resnet18(weights)
        self.model.fc = nn.Linear(512, 128)
    def forward(self, input1, input2):
        output1, output2 = self.model(input1), self.model(input2)
        return output1, output2

class resnet_50(nn.Module):
    def __init__(self, weights=False):
        super(resnet_50, self).__init__()
        self.model = resnet50()
        self.model.fc = nn.Linear(2048, 512)
    def forward(self, input1, input2):
        output1, output2 = self.model(input1), self.model(input2)
        return output1, output2

class vit_base(nn.Module):
    def __init__(self):
        super(vit_base, self).__init__()
        self.vit = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
        self.vit.classifier = nn.Linear(768, 128)
    def forward(self, input1, input2):
        output1, output2 = self.vit(input1).logits, self.vit(input2).logits
        return output1, output2
        
class TransformerNet(nn.Module):
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

class SiameseConvNet(nn.Module):
    def __init__(self):
        super(SiameseConvNet, self).__init__()
        
        # Setting up the Sequential of CNN Layers
        self.cnn = nn.Sequential(
            
            nn.Conv2d(1, 48, kernel_size=(11, 11), stride=1),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(48, alpha=1e-4, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2),
            
            nn.Conv2d(48, 128, kernel_size=(5, 5), stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(128, alpha=1e-4, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2),
            nn.Dropout(p=0.3),

            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 96, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2),
            nn.Dropout(p=0.3),

            nn.Flatten(1,-1),
            
            nn.Linear(25 * 17 * 96, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            
            nn.Linear(1024, 128),
            nn.ReLU(inplace=True)
        )

    def forward(self, input1, input2):
        output1 = self.cnn(input1)
        output2 = self.cnn(input2)
        return output1, output2

"""## Define Loss Function"""

class ContrastiveLoss(nn.Module):

    def __init__(self, margin=2):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean(label * torch.pow(euclidean_distance, 2) +
                                      (1-label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive

class BCEWithLogitsLoss(nn.Module):

    def __init__(self, margin=0):
        super(BCEWithLogitsLoss, self).__init__()
        self.loss = nn.BCEWithLogitsLoss()
        self.shift = 1
        self.margin = margin
        

    def forward(self, output1, output2, label):
        euclidean_distance = -1*F.pairwise_distance(output1, output2) + self.shift + self.margin
        loss = self.loss(euclidean_distance, label)
        return loss


class CLIPModel(nn.Module):
    def __init__(self, model, temperature=1):
        super().__init__()
        self.image_encoder = model
        self.temperature = temperature

    def forward(self, input1, input2):
        # Getting Image Features
        img1_feat, img2_feat = self.image_encoder(input1, input2)
        logits = (img1_feat @ img2_feat.T) / self.temperature
        targets = F.softmax(
            (img1_feat @ img1_feat.T + img2_feat @ img2_feat.T) / 2 * self.temperature, dim=-1
        )
        # targets = torch.arange(len(input1))
        return logits, targets
    def forward_features(self, input1, input2):
        img1_feat, img2_feat = self.image_encoder(input1, input2)
        return img1_feat, img2_feat
        
def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()

class CLIPLoss(nn.Module):

    def __init__(self):
        super(CLIPLoss, self).__init__()
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, logits, label):
        img1_loss = cross_entropy(logits, label, reduction='none')
        img2_loss = cross_entropy(logits.T, label.T, reduction='none')
        loss =  (img1_loss + img1_loss) / 2.0
        return loss.mean()
        # return self.loss(logits, label)