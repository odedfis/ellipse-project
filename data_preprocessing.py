from keras.preprocessing import image
from utils import *

IMAGE_SZ = 50


def load_images(file_names):
    imgs = np.empty((len(file_names), IMAGE_SZ, IMAGE_SZ, 3))
    for i, img_file in tqdm(enumerate(file_names)):
        imgs[i, :, :, :] = image.img_to_array(
            image.load_img(img_file)) / 255.0
    return imgs



