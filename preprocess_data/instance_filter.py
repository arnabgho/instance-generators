import os
import sys
from PIL import Image
import random
import numpy as np
from shutil import copyfile

wd=os.environ['CITYSCAPES_DATASET']

ins_d=os.environ['SEGMENTATION_MAP_CITYSCAPES_DATASET']

wd_fine=os.path.join( wd, 'gtFine')
wd_imgs=os.path.join( wd, 'leftImg8bit')

if not os.path.exists(ins_d ):
    os.makedirs( ins_d  )

for (dirpath,dirnames,filenames) in os.walk(wd_fine):
    for filename in filenames:
        if 'gtFine_color' in filename:
            copyfile(os.path.join( dirpath , filename  ) , os.path.join( ins_d , filename )   )

