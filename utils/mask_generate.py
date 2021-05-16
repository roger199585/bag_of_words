import numpy
import matplotlib.pyplot as plt
from PIL import Image
import os

from config import ROOT

image_size = (256, 256)
# mask_size = (64, 64)
mask_size = (16, 16)

bg = numpy.uint8(255*numpy.ones(image_size))
m = numpy.uint8(numpy.zeros(mask_size))

x = 0
y = 0
count = 0
while x <= 256 and y <= 256:
    background = Image.fromarray(bg)
    mask = Image.fromarray(m)

    background.paste(mask, (x, y))
    background.save(ROOT + '/dataset/ssl_mask/mask' + str(count) +'.png')
    
    x += 16
    if x == 256:
        x = 0
        y += 16
    count += 1