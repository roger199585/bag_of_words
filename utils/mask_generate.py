import numpy
import matplotlib.pyplot as plt
from PIL import Image
import os

from config import ROOT

image_size = (1024, 1024)
# mask_size = (64, 64)
mask_size = (128, 128)

bg = numpy.uint8(255*numpy.ones(image_size))
m = numpy.uint8(numpy.zeros(mask_size))

x = 0
y = 0
count = 0
while x <= 1024 and y <= 1024:
    background = Image.fromarray(bg)
    mask = Image.fromarray(m)

    background.paste(mask, (x, y))
    background.save(ROOT + '/dataset/big_mask128/mask' + str(count) +'.png')
    
    x += 128
    if x == 1024:
        x = 0
        y += 128
    count += 1