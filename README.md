# Signature-Verification

### Requirements
* Python 3.8 (or newer)
* PyTorch install 1.11.0 (older versions may work too)
* torchvision
* Other dependencies: transformer, numpy


### Model Training
To train a siamese network run:
```
python siamese_train.py  \
  --model_name SiameseConvNet --loss BCE
```
Parameters:
--model_name: SiameseConvNet, TransformerNet， resnet, resnet_pretrained, vit_pretrained
--loss: BCE, contrastive
