import argparse
import sys
from utils.convert import ImageConverter

"""
使用範例
python preprocess.py --type bottle --type capsule ...
python preprocess.py --type ALL --size 512
"""

""" set parameters """
parser = argparse.ArgumentParser()
parser.add_argument('-t','--types', action='append', help='You can input <ALL>, or specific category <bottle><cable><capsule><carpet><grid><hazelnut><leather><metal_nut><pill><screw><tile><toothbrush><transistor><wood><zipper>', required=True)
parser.add_argument('-s', '--size', type=int, default=1024)
parser.add_argument('-r', '--root', type=str, default='/home/dinosaur/bag_of_words')
parser.add_argument('-q', '--quantization', type=str, default='False')
parser.add_argument('-b', '--blur', type=str, default='False')
args = parser.parse_args()

if args.types == ['ALL']:
    c = ImageConverter(SIZE=args.size, ROOT=args.root)
else:
    c = ImageConverter(TYPES=args.types, SIZE=args.size, ROOT=args.root, quantization=args.quantization, blur=args.blur)
    
c.start()
