import os
from tqdm import tqdm
from PIL import Image

class ImageConverter():
    def __init__(
        self,
        ROOT='/root/AFS/bag_of_words/dataset', 
        TYPES=['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper'],
        SIZE=1024
    ):
        self.ROOT = ROOT
        self.TYPES = TYPES
        self.SIZE = (SIZE, SIZE)
        
    def resizeImage(self, category, whichSet):
        dataPath = '{}/{}/{}/'.format(self.ROOT, category, whichSet)
        resizeDataPath = '{}/{}/{}_resize/'.format(self.ROOT, category, whichSet)

        subFolders = os.listdir(dataPath)

        if whichSet == 'train':
            for name in subFolders:
                currentRoot = dataPath + name
                images = os.listdir(currentRoot)

                if not os.path.isdir(resizeDataPath + name):
                    os.makedirs(resizeDataPath + name)

                for imageName in tqdm(images):
                    if imageName.endswith('.png'):
                        im = Image.open(currentRoot + '/' + imageName)
                        im = im.resize(self.SIZE)
                        im.save(resizeDataPath + name + '/' +imageName)
                    else:
                        print(imageName, "is not a picture")
        else:
            if not os.path.isdir('{}/{}/{}_resize/all'.format(self.ROOT, category, whichSet)):
                os.makedirs('{}/{}/{}_resize/all'.format(self.ROOT, category, whichSet))
            
            allTestingSetImageIndex = 0
            for name in subFolders:
                currentRoot = dataPath + name
                images = os.listdir(currentRoot)
                images = sorted(images, key=lambda a: a[:-4])
                
                if not os.path.isdir(resizeDataPath + name):
                    os.makedirs(resizeDataPath + name)
                    
                if name == 'good':
                    for imageName in tqdm(images):
                        if imageName.endswith('.png'):
                            im = Image.open(currentRoot + '/' + imageName)
                            im = im.resize(self.SIZE)
                            im.save(resizeDataPath + name + '/' +imageName)
                        else:
                            print(imageName, "is not a picture")
                else:
                    # testing image                     
                    for imageName in tqdm(images):
                        if imageName.endswith('.png'):
                            im = Image.open(currentRoot + '/' + imageName)
                            im = im.resize(self.SIZE)
                            im.save(resizeDataPath + name + '/' +imageName)
                            im.save(resizeDataPath + 'all/{}.png'.format(str(allTestingSetImageIndex).zfill(3)))
                            
                            allTestingSetImageIndex += 1
                        else:
                            print(imageName, "is not a picture")
                    
                
                
    
    def start(self):
        print("Start converting...\n")
        for _type in self.TYPES:
            print(_type + ':')

            print('Training set')
            self.resizeImage(_type, 'train')

            print('Testing set')
            self.resizeImage(_type, 'test')

            print('Ground truth')
            self.resizeImage(_type, 'ground_truth')
    
    
