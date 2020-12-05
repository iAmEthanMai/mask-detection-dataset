import os
import cv2
from xml.etree.ElementTree import ElementTree
import re
import random
import numpy as np
from os.path import join

# convert voc annotation format to darknet format
def xml_to_darknet(path):
    root = ElementTree().parse(path)

    img_path = root.find('filename').text.replace('png','txt')
    img_size = root.find('size')
    width = int(img_size.find('width').text)
    height = int(img_size.find('height').text)
    
    with open('../data/labels/' + img_path, 'w') as f:
        lines = []

        
        for node in root.findall('object'):
            object_ = dict(class_=None, x=None, y=None, width=None, height=None)

            # class
            class_name =  node.find('name').text

            if(class_name == 'without_mask'):
                object_['class_'] = '0'
            elif(class_name == 'with_mask'):
                object_['class_'] = '1'
            else:
                object_['class_'] = '2'

            
            
            # bounding box
            bnd_box = node.find("bndbox")
            x_min = float(bnd_box[0].text)
            y_min = float(bnd_box[1].text)
            x_max = float(bnd_box[2].text)
            y_max = float(bnd_box[3].text)


            dw = float(1/width)
            dh = float(1/height)

            w = float(x_max - x_min)
            h = float(y_max - y_min)

            x = float((x_min + x_max)/2 -1)
            y = float((y_min + y_max)/2 -1)

            w = float(w * dw)
            h = float(h * dh)
            x = float(x * dw)
            y = float(y * dh)
            
            object_['x'] = str(x)
            object_['y'] = str(y)
            object_['width'] = str(w)
            object_['height'] = str(h)

            line = object_['class_'] + ' ' + object_['x'] + ' ' + object_['y'] + ' ' + object_['width'] + ' ' + object_['height']
            
            lines.append(line)
            lines.append('\n')

        for line in lines[:-1]:
            f.write(line)   
    f.close






def process_data():

    # get the paths of all the images available
    img_paths = []
        
    for dirname, _, filenames in os.walk('../data/images'):
        for filename in filenames:
            # img_paths.append(os.path.join('../mask-detection-dataset/data/images', filename)) # local machine
            img_paths.append(os.path.join('/content/mask-detection-dataset/data/images', filename)) # google colab
            # img_paths.append(os.path.join('./mask-detection-dataset/data/images', filename)) # kaggle
            
    # shuffle data
    random.shuffle(img_paths)

    # split
    train, validate, test = np.split(img_paths, [int(len(img_paths)*0.8), int(len(img_paths)*0.9)])
    
    # training images set = 80% of all images
    # validating images set = 10% of all images
    # testing images set = 10% of all images

    # write train.txt
    with open('../data/train.txt', 'w') as f:
        lines = list('\n'.join(train))
        f.writelines(lines)
    f.close

    # write validate.txt
    with open('../data/validate.txt', 'w') as f:
        lines = list('\n'.join(validate))
        f.writelines(lines)
    f.close

    # write test.txt
    with open('../data/test.txt', 'w') as f:
        lines = list('\n'.join(test))
        f.writelines(lines)
    f.close

    # process annotations
    for dirname, _, filenames in os.walk('../data/xml'):
        for filename in filenames:
            annotation_path = (os.path.join(dirname, filename))
            xml_to_darknet(annotation_path)
    
    


        

 
process_data()
