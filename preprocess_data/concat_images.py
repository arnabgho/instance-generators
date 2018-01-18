import os
import sys
from PIL import Image
import random
import numpy as np
wd=os.environ['CITYSCAPES_DATASET']

wd_instance= os.path.join( wd, 'gen_instances'  )
wd_imgs    = os.path.join( wd, 'gen_images' )
wd_translation = os.path.join(wd, 'inst2img')

if not os.path.exists(wd_translation):
    os.makedirs(wd_translation)

for (dirpath,dirnames,filenames) in os.walk( wd_imgs):
    for filename in filenames:
        img_file=os.path.join(dirpath , filename )
        inst_file=img_file.replace('gen_images','gen_instances')

        images = map(Image.open, [ inst_file , img_file  ])
        widths, heights = zip(*(i.size for i in images))
        total_width = sum(widths)
        max_height = max(heights)

        new_im = Image.new('RGB', (total_width, max_height))

        x_offset = 0
        for im in images:
            new_im.paste(im, (x_offset,0))
            x_offset += im.size[0]

        new_im.save( os.path.join( wd_translation , filename  ) )
