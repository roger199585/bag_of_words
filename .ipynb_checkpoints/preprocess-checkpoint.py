import argparse
import sys
from utils.convert import ImageConverter

"""
使用範例
python prprocess.py --type bottle --type capsule ...
python prprocess.py --type ALL --size 512
"""

""" set parameters """
parser = argparse.ArgumentParser()
parser.add_argument('-t','--types', action='append', help='You can input <ALL>, or specific category <bottle><cable><capsule><carpet><grid><hazelnut><leather><metal_nut><pill><screw><tile><toothbrush><transistor><wood><zipper>', required=True)
parser.add_argument('-s', '--size', type=int, default=1024)
parser.add_argument('-r', '--root', type=str, default='/root/AFS/bag_of_words/dataset')
args = parser.parse_args()

if args.types == ['ALL']:
    c = ImageConverter(SIZE=args.size, ROOT=args.root)
else:
    c = ImageConverter(TYPES=args.types, SIZE=args.size, ROOT=args.root)
    
c.start()