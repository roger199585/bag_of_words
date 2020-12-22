""" STD Library """
import os
import sys
import argparse
import numpy as np
from PIL import Image
from datetime import datetime

""" Pytorch Library """
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter

""" Customize Library """
import networks.autoencoder as autoencoder
from config import ROOT, RESULT_PATH

def image_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

class MVTecLoader(torch.utils.data.Dataset):
    def __init__(self, category, patch_size=64, _type='train', errorType=None):
        if errorType == None:
            self.root = f"{ ROOT }/dataset/{ category }/{ _type }_resize/good"
        else:
            self.root = f"{ ROOT }/dataset/{ category }/{ _type }_resize/{ errorType }"
        
        self.samples = [x for x in os.listdir(self.root) if x.endswith('.png') ]
        
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomCrop(patch_size),                                                                            
            torchvision.transforms.ToTensor(),
        ])

    def __getitem__(self, index):
        path = os.path.join(self.root, self.samples[index])
        
        img = image_loader(path)
        
        imgs = []
        for i in range(4):
            patch = self.transform(img)  # turn the image to a tensor
            imgs.append(patch)
        
        return torch.stack(imgs)

    def __len__(self):
        return len(self.samples)

if __name__ == "__main__":
    """ Set parameters """ 
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='bottle')
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--lr', type=int, default=1e-3)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--patch_size', type=int, default=64)
    parser.add_argument('--image_size', type=int, default=1024)
    args = parser.parse_args()

    

    train_loader = torch.utils.data.DataLoader(
        MVTecLoader(args.data),
        batch_size = args.batch,
        shuffle = True,
        num_workers = 16, 
        pin_memory = True
    )
    test_loader = torch.utils.data.DataLoader(
        MVTecLoader(args.data, _type='test', errorType='all'),
        batch_size = args.batch,
        shuffle = True,
        num_workers = 16, 
        pin_memory = True
    )

    writer = SummaryWriter(log_dir=f"{RESULT_PATH}/image_test_{datetime.now()}")

    model = autoencoder.autoencoder(3)
    model = model.cuda()
    L1_loss = torch.nn.L1Loss()

    optimizer_AE = torch.optim.Adam(
        model.parameters(), 
        lr=args.lr
    )

    # for idx, imgs in enumerate(train_loader):
    #     print(imgs.reshape(imgs.shape[0] * imgs.shape[1], imgs.shape[2], args.patch_size, args.patch_size).shape)

    for epoch in range(args.epochs):
        model.train()
        for idx, imgs in enumerate(train_loader):
            imgs = imgs.cuda()
            imgs = imgs.reshape(imgs.shape[0] * imgs.shape[1], imgs.shape[2], args.patch_size, args.patch_size)
            reconstruct_imgs, _ = model(imgs)
            ae_loss = L1_loss(reconstruct_imgs, imgs)

            optimizer_AE.zero_grad()
            ae_loss.backward()
            optimizer_AE.step()
        
        model.eval()
        for idx, test_imgs in enumerate(test_loader):
            test_imgs = test_imgs.cuda()
            test_imgs = test_imgs.reshape(test_imgs.shape[0] * test_imgs.shape[1], test_imgs.shape[2], args.patch_size, args.patch_size)
            test_reconstruct_imgs, _ = model(test_imgs)
            test_ae_loss = L1_loss(test_reconstruct_imgs, test_imgs)

        writer.add_images("Reconstruct Patch/train", reconstruct_imgs, epoch)    
        writer.add_images("Origin Patch/train", imgs, epoch)
        writer.add_scalar('Loss/train', ae_loss, epoch)

        writer.add_images("Reconstruct Patch/test", test_reconstruct_imgs, epoch)    
        writer.add_images("Origin Patch/test", test_imgs, epoch)
        writer.add_scalar('Loss/test', test_ae_loss, epoch)

        print(f"epoch [{epoch}/{args.epochs}] loss/train: {ae_loss.item()} loss/test: {test_ae_loss.item()}")