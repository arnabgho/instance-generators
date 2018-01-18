import os
import sys
from PIL import Image
import random
import numpy as np
wd=os.environ['CITYSCAPES_DATASET']

ins_d=os.environ['INSTANCE_CITYSCAPES_DATASET']

wd_fine=os.path.join( wd, 'gtFine')
wd_imgs=os.path.join( wd, 'leftImg8bit')


def image_creator(selected,np_instance_map,np_img):
    new_inst=np.zeros_like(np_img)
    new_img=np.zeros_like( np_img)
    for i in range(new_inst.shape[0]):
        for j in range(new_inst.shape[1]):
            if np_instance_map[i][j] in selected:
                new_inst[i][j]=[255,255,255]
                new_img[i][j]=np_img[i][j]
    return new_inst,new_img


def processor(filename,img,instance_map,instance_dir,image_dir):
    np_instance_map=np.array(instance_map)
    np_img= np.array(img)

    ids=np.unique(np_instance_map)
    condition = np.divide(ids, 1000)==26
    car_ids=np.extract(condition, ids)
    num_samples=car_ids.shape[0]
    for i in range(num_samples):
        k = random.randint(1,num_samples)
        selected=random.sample( car_ids  , k  )
        new_inst,new_img=image_creator(selected,np_instance_map,np_img)
        img_new_inst=Image.fromarray(new_inst,'RGB')
        img_new_img=Image.fromarray(new_img,'RGB')
        img_new_inst.save( os.path.join( instance_dir,str(i) + '_' + filename  ))
        img_new_img.save( os.path.join( image_dir , str(i) + '_' +  filename   ))


for (dirpath,dirnames,filenames) in os.walk(wd_fine):
    for filename in filenames:
        if 'gtFine_instanceIds' in filename:
            instance_map=Image.open(os.path.join(dirpath,filename))
            img_filename=filename.replace('gtFine_instanceIds','leftImg8bit')
            img_dirpath=dirpath.replace('gtFine','leftImg8bit')
            img=Image.open(os.path.join(img_dirpath,img_filename))

            instance_dir=dirpath.replace('gtFine','gen_instances')
            image_dir = dirpath.replace('gtFine','gen_images')
            if not os.path.exists(instance_dir ):
                os.makedirs( instance_dir  )

            if not os.path.exists(image_dir ):
                os.makedirs( image_dir)

            processor(filename,img,instance_map,instance_dir,image_dir)

