import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import torchvision
from torchvision.utils import save_image
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader

import os
import time
import argparse
from datetime import datetime

""" Custimiza Lib """
import dataloaders
from config import ROOT, RESULT_PATH

print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)
iter_count=0
val_iter_count=0
# === VGG ====
cfgs = {
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}

class VGG(nn.Module):
    def __init__(self, features, num_classes=1000):
        super(VGG, self).__init__()
        self.features = features
        # avgpool 是為了強制將我們的 feature 轉換成 512x1x1
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.features(x)
        # x = self.avgpool(x)
        return x

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

def _vgg(arch, cfg, batch_norm, pretrained, progress, **kwargs):
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    return model

def vgg19(pretrained=False, progress=True, **kwargs):
    return _vgg('vgg19', 'E', False, pretrained, progress, **kwargs)
# =========
def initialize_model(model_name, num_classes, feature_extract=True, image_size=64, use_pretrained=True):
    model_ft = None

    if model_name == "resnet":
        """ Resnet18"""
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = image_size
    elif model_name == "vgg":
        pre_model = models.vgg19(pretrained=True)
        pre_model = pre_model.state_dict()

        model_ft = vgg19()
        model_ft_dict = model_ft.state_dict()

        pre_model = {k: v for k, v in pre_model.items() if k in model_ft_dict}
        model_ft_dict.update(pre_model) 
        model_ft.load_state_dict(pre_model)
        
        """ VGG19"""
        # model_ft = models.vgg19(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        
        for param in model_ft.features[32].parameters():
            param.requires_grad = True
        for param in model_ft.features[34].parameters():
            param.requires_grad = True

        input_size = image_size
    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def train_model(model, dataloaders, criterion, optimizer, writer, num_epochs=25, is_inception=False):
    global iter_count
    global val_iter_count

    since = time.time()

    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase


        # running_loss = 0.0
        # Iterate over data.
        phase='train'
        for patch1, patch2 in dataloaders[phase]:
            model.train()

            patch1 = patch1.to(device)
            patch2 = patch2.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(phase == 'train'):
                # Get model outputs and calculate loss
                # Special case for inception because in training it has an auxiliary output. In train
                #   mode we calculate the loss by summing the final output and the auxiliary output
                #   but in testing we only consider the final output.

                feature1 = model(patch1)
                feature2 = model(patch2)
                loss = criterion(feature1, feature2)
                loss = loss.mean()

                if phase == 'train':
                    # backward + optimize only if in training phase
                    loss.backward()
                    optimizer.step()

                # statistics
                writer.add_scalar(phase, loss.item(), iter_count)
                # running_loss += loss.item() * patch1.size(0)
                iter_count += 1

            if iter_count % 200 == 0:
                for patch1, patch2, in dataloaders['val']:
                    model.eval()

                    patch1 = patch1.to(device)
                    patch2 = patch2.to(device)

                    feature1 = model(patch1)
                    feature2 = model(patch2)
                    loss = criterion(feature1, feature2)
                    loss = loss.mean()

                    writer.add_scalar('val', loss.item(), val_iter_count)
                    val_iter_count += 1
                
                if not os.path.isdir(f"//mnt/train-data1/fine-tune-models/{ args.data }"):
                    print(f"create {args.data}")
                    os.makedirs(f"//mnt/train-data1/fine-tune-models/{ args.data }")

                torch.save(model.state_dict(), f"//mnt/train-data1/fine-tune-models/{ args.data }/{ iter_count }.ckpt")



        # epoch_loss = running_loss / len(dataloaders[phase].dataset)

        # print('{} Loss: {:.4f}'.format(phase, epoch_loss))
        # writer.add_scalar(phase + '_epoch', epoch_loss, epoch)

        # if not os.path.isdir(f"//mnt/train-data1/fine-tune-models/{ args.data }"):
        #     print(f"create {args.data}")
        #     os.makedirs(f"//mnt/train-data1/fine-tune-models/{ args.data }")

        # torch.save(model.state_dict(), f"//mnt/train-data1/fine-tune-models/{ args.data }/{ epoch }.ckpt")

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    return model


""" set parameters """
parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='bottle')
args = parser.parse_args()

# ================================================= Initialize model =======================================================
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device", device)

# Initialize the model for this run
model_ft, input_size = initialize_model('vgg', 15, True, use_pretrained=True)
# Print the model we just instantiated
# print(model_ft)

model_ft = model_ft.to(device)
print("Params to learn:")
params_to_update = []
for name,param in model_ft.named_parameters():
    if param.requires_grad == True:
        params_to_update.append(param)
        print("\t",name)

# ================================================= DataLoader =============================================
print("Initializing Datasets and Dataloaders...")
# Create training and validation datasets
train_dataset = dataloaders.MvtecLoaderForFineTune( f"/mnt/train-data1/corn/bottle_patch/train", 'ft-train' )
train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)

validation_dataset = dataloaders.MvtecLoaderForFineTune( f"/mnt/train-data1/corn/bottle_patch/val", 'ft-val' )
validation_loader = DataLoader(validation_dataset, batch_size=1024, shuffle=False)

# ================================================== Train Model =======================================
writer = SummaryWriter(log_dir="{}/fine_tune_{}_{}".format(RESULT_PATH, args.data, datetime.now()))
# Observe that all parameters are being optimized
optimizer_ft = optim.Adam(params_to_update, lr=0.00001)
# Setup the loss fxn
criterion = nn.L1Loss()
# criterion = nn.CosineSimilarity(dim=1, eps=1e-6)
# Train and evaluate
model_ft = train_model(model_ft, {
    "train": train_loader,
    "val": validation_loader
}, criterion, optimizer_ft, writer, num_epochs=100)