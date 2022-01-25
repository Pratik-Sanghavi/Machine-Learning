import numpy as np
import random
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2 as cv
from PIL import Image

def flipping(img, axis = "horizontal"):
    """
    Flip image about the axis chosen
    Arguments: Image and axis of rotation (since this is a 180 degree rotation about the chosen axis)
    Returns: Flipped image as per chosen axis
    """
    if axis == "horizontal":
        return cv.flip(img, 1)
    elif axis == "vertical":
        return cv.flip(img, 0)
    elif axis == "both":
        return cv.flip(img, -1)
    
def random_cropping(img, scale = 0.9):
    """
    Random crop of the image based on the scale chosen
    Arguments: Image and scale based on which cropping dimensions are chosen
    Returns: Cropped image as per chosen axis
    """
    height, width = int(img.shape[0] * scale), int(img.shape[1] * scale)
    x = random.randint(0, img.shape[1] - width)
    y = random.randint(0, img.shape[0] - height)
    cropped = img[y:y+height, x:x+height]
    return cv.resize(cropped, (img.shape[1], img.shape[0]), interpolation=cv.INTER_AREA);

def random_rotation(img, rotation_point = None):
    """
    Random rotation of the image based on the rotation point chosen
    Arguments: Image and rotation point about which image will be rotated
    Returns: Rotated image as per chosen rotation point and randomly generated angle
    """
    (height, width, depth) = img.shape
    
    if rotation_point is None:
        rotation_point = (width//2, height//2)
    
    angle = random.randint(5, 355)
    
    rotMat = cv.getRotationMatrix2D(rotation_point, angle, 1.0)
    dimensions = (width, height)
    return cv.warpAffine(img, rotMat,dimensions)

def load_images(path):
    from PIL import Image
    import glob
    from numpy import asarray
    image_list = []
    for filename in glob.glob(path + '/*.png'):
        im=Image.open(filename)
        im = asarray(im)
        resized_im = cv.resize(im, (400,400),interpolation=cv.INTER_CUBIC)
        image_list.append(resized_im)
    return image_list

def data_augmentation(arr):
    maxm = 654
    axes = ["horizontal","vertical","both"]
    res = []
    res.extend(arr)
    for i in tqdm(range(len(arr))):
        img = arr[i]
        if len(res) >= maxm:
            break
        flipping_choice = random.randint(0,2)
        res.append(flipping(img,axes[flipping_choice]))
        
        if len(res) >= maxm:
            break

        res.append(random_cropping(img))
        
        if len(res) >= maxm:
            break
        
        res.append(random_rotation(img))

    return res

def get_dirs():
    dirs = [i[0] for i in os.walk('./Data/train')]
    dirs = dirs[1:]
    dirs_to_use = []

    for path in dirs:
        path = path.replace('\\','/')
        dirs_to_use.append(path)

    return dirs_to_use

def write_images(image_arr, path, category):
    for i in tqdm(range(len(image_arr))):
        img = image_arr[i]
        img = Image.fromarray(img)
        img_name = category + str(i) + '.png'
        
        img.save(os.path.join(path, img_name))
        
    return

dirs_to_use = get_dirs()

categories = [path.replace('./Data/train/', '') for path in dirs_to_use]

new_dirs = ['./Data/train_updated/'+ category for category in categories]

for i in range(len(dirs_to_use)):
    temp = load_images(dirs_to_use[i])
    category = categories[i]
    path = new_dirs[i]
    print(f"Starting augmentation for {category}")
    temp = data_augmentation(temp)
    print(f"Finished augmentation. Now proceeding to write for {category}")
    write_images(temp, path, category)
    print(f"Finished writing for {category}")