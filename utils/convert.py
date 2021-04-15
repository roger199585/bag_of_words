import os
import sys
from tqdm import tqdm
from PIL import Image, ImageFilter

class ImageConverter():
    def __init__(
        self,
        ROOT='/home/dinosaur/bag_of_words', 
        TYPES=['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper'],
        SIZE=1024,
        quantization='',
        blur=''
    ):
        self.ROOT = ROOT + '/dataset'
        self.TYPES = TYPES
        self.SIZE = (SIZE, SIZE)
        self.quantization = True if quantization == "True" else False
        self.blur = True if blur == "True" else False
         
    def resizeImage(self, category, whichSet, quantize=False, blur=False):
        dataPath = '{}/{}/{}/'.format(self.ROOT, category, whichSet)
        resizeDataPath = '{}/{}/{}_resize/'.format(self.ROOT, category, whichSet)

        subFolders = os.listdir(dataPath)

        if whichSet == 'train':
            for name in subFolders:
                currentRoot = dataPath + name
                images = os.listdir(currentRoot)
                import glob

                images = glob.glob(os.path.join(currentRoot, '*.png'))
                if not os.path.isdir(resizeDataPath + name):
                    os.makedirs(resizeDataPath + name)

                for imageName in tqdm(images):
                    if imageName.endswith('.png'):
                        imageName = imageName.split('/')[-1]
                        im = Image.open(currentRoot + '/' + imageName)
                        im = im.resize(self.SIZE)
                        im = im.quantize(colors=256) if quantize else im
                        im = im.filter(ImageFilter.GaussianBlur(radius=5)) if blur else im
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
                            im = im.quantize(colors=256) if quantize else im
                            im = im.filter(ImageFilter.GaussianBlur(radius=5)) if blur else im
                            im.save(resizeDataPath + name + '/' +imageName)
                        else:
                            print(imageName, "is not a picture")
                else:
                    # testing image                     
                    for imageName in tqdm(images):
                        if imageName.endswith('.png'):
                            im = Image.open(currentRoot + '/' + imageName)
                            im = im.resize(self.SIZE)
                            im = im.quantize(colors=256) if quantize else im
                            im = im.filter(ImageFilter.GaussianBlur(radius=5)) if blur else im
                            im.save(resizeDataPath + name + '/' +imageName)
                            im.save(resizeDataPath + 'all/{}.png'.format(str(allTestingSetImageIndex).zfill(3)))
                            
                            allTestingSetImageIndex += 1
                        else:
                            print(imageName, "is not a picture")
    
    def start(self):
        print("Start converting...\n")
        for _type in self.TYPES:
            print(_type + ':')

            print('Training set', self.blur)
            self.resizeImage(_type, 'train', quantize=self.quantization, blur=self.blur)

            print('Testing set', self.blur)
            self.resizeImage(_type, 'test', quantize=self.quantization, blur=self.blur)

            print('Ground truth')
            self.resizeImage(_type, 'ground_truth', quantize=False, blur=False)
    
    

# import os
# import sys
# from tqdm import tqdm
# from PIL import Image, ImageFilter

# train_image_count = 0
# val_image_count = 0
# test_image_count = 0
# trainResizeDataPath = '/home/corn/Project/bag_of_words/dataset/fine-tune/train'
# valResizeDataPath = '/home/corn/Project/bag_of_words/dataset/fine-tune/val'
# testResizeDataPath = '/home/corn/Project/bag_of_words/dataset/fine-tune/test'

# for _type in ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper']:
#     dataPath = '/home/corn/Project/bag_of_words/dataset/{}/train/'.format(_type)
#     subFolders = os.listdir(dataPath)

#     for name in subFolders:
#         currentRoot = dataPath + name
#         images = os.listdir(currentRoot)
#         import glob

#         images = glob.glob(os.path.join(currentRoot, '*.png'))

#         for imageName in tqdm(images):
#             if imageName.endswith('.png'):
#                 imageName = imageName.split('/')[-1]
#                 im = Image.open(currentRoot + '/' + imageName)
#                 im = im.resize((224, 224))
#                 im.save(trainResizeDataPath + '/' + _type + '-' + str(train_image_count) + '.png' )
#                 train_image_count += 1
#             else:
#                 print(imageName, "is not a picture")
    
#     dataPath = '/home/corn/Project/bag_of_words/dataset/{}/test/'.format(_type)
#     subFolders = os.listdir(dataPath)
#     for name in subFolders:
#         currentRoot = dataPath + name
#         images = os.listdir(currentRoot)
#         images = sorted(images, key=lambda a: a[:-4])
            
#         if name == 'good':
#             for imageName in tqdm(images):
#                 if imageName.endswith('.png'):
#                     im = Image.open(currentRoot + '/' + imageName)
#                     im = im.resize((224, 224))
#                     im.save(valResizeDataPath + '/' + _type + '-' + str(val_image_count) + '.png' )
#                     im.save(testResizeDataPath + '/' + _type + '-' + str(test_image_count) + '.png' )
#                     val_image_count += 1
#                     test_image_count += 1
#                 else:
#                     print(imageName, "is not a picture")
#         else:
#             # testing image                     
#             for imageName in tqdm(images):
#                 if imageName.endswith('.png'):
#                     im = Image.open(currentRoot + '/' + imageName)
#                     im = im.resize((224, 224))
#                     im.save(testResizeDataPath + '/' + _type + '-' + str(test_image_count) + '.png')
#                     test_image_count += 1
#                 else:
#                     print(imageName, "is not a picture")
