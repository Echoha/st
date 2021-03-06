import os
import numpy as np
import scipy.misc as scm
from glob import glob

import cv2
def save_img(out_path, img):
    img = np.clip(img, 0, 255).astype(np.uint8)
    scm.imsave(out_path, img)
def get_img(src, img_size=False):
	img = cv2.imread(src)[:,:,(2,1,0)]
	if not (len(img.shape) == 3 and img.shape[2] == 3):
		img = np.dstack((img, img, img))
	if img_size is not False:
		img_target_size = (img_size[0],img_size[1])
		img = cv2.resize(img,img_target_size,interpolation = cv2.INTER_CUBIC)
	return img



def get_content_image(img_path, size):
    suffix_list = ['.jpg', '.JPG', '.jpeg', '.JPEG']
    suffix = os.path.splitext(img_path)
    if suffix[1] in suffix_list:
        img = scm.imread(img_path, mode='RGB')
        h, w, c = np.shape(img)
        if h!=size or w!=size:
            img = np.zeros([265, 256, 3], np.uint8)
        img = img[:h, :w, ::-1]  # rgb to bgr
        return img
        
def get_image(img_path, size=None):
    suffix_list = ['.jpg', '.JPG', '.jpeg', '.JPEG']
    suffix = os.path.splitext(img_path)
    if suffix[1] in suffix_list:
      img = scm.imread(img_path, mode='RGB')
      h, w, c = np.shape(img)
      img = img[:h, :w, ::-1]  # rgb to bgr
      if size:
        img = scm.imresize(img, (size, size))
      return img
    else:
      return np.zeros([265, 256, 3], np.uint8)


def inverse_image(img):
    img[img > 255] = 255
    img[img < 0] = 0
    img = img[:, :, ::-1]  # bgr to rgb
    return img


def make_project_dir(project_dir):
    if not os.path.exists(project_dir):
        os.makedirs(project_dir)
        os.makedirs(os.path.join(project_dir,'models'))
        os.makedirs(os.path.join(project_dir,'train_result'))
        os.makedirs(os.path.join(project_dir,'test_result'))


def data_loader(dataset):
    print('images Load ....')
    data_path = dataset

    if os.path.exists(data_path + '.npy'):
        data = np.load(data_path + '.npy')
    else:
        data = []
        data = glob(os.path.join(data_path, "*.*"))
        np.save(data_path + '.npy', data)
    print('images Load Done')

    return data
