from PIL import Image
import os

# TYPES = ["pill", "capsule", "carpet", "grid", "hazelnut", "screw", "tile", "toothbrush", "zipper"]
ROOT = 'dataset/'
TYPES=["wood"]
SIZE = (1024, 1024)


# create training set
print("Start converting...\n")
for _type in TYPES:
    print(_type)
    trainDataPath = 'DB/' + _type + '/test/color/'
    # trainDataPath = 'DB/'+ _type + '/test/broken_large/'
    trainResizeDataPath = ROOT + _type + '/test/color_resize/'
    trainImages = os.listdir(trainDataPath)
    
    for imageName in trainImages:
        if imageName.endswith('.png'):
            im = Image.open(trainDataPath + imageName)
            im = im.resize(SIZE, Image.BILINEAR)
            if not os.path.isdir(trainResizeDataPath):
                os.makedirs(trainResizeDataPath)
            im.save(trainResizeDataPath+imageName)
        else:
            print(imageName, "is not a picture")
