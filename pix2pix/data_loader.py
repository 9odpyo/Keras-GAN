import scipy
from glob import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt

image_path = '/data01/ML_DATA/grayscale/*'

class DataLoader():
    def __init__(self, dataset_name, img_res=(128, 128)):
        self.dataset_name = dataset_name
        self.img_res = img_res

    def load_data(self, batch_size=1, is_testing=False):
        path = glob(image_path)

        batch_images = np.random.choice(path, size=batch_size)

        imgs_A = []
        imgs_B = []
        for img_path in batch_images:
            bgr_image = cv2.imread(img_path, cv2.IMREAD_COLOR)
            img_A = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
            gray_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img_B = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)

            img_A = scipy.misc.imresize(img_A, self.img_res)
            img_B = scipy.misc.imresize(img_B, self.img_res)

            # If training => do random flip
            if not is_testing and np.random.random() < 0.5:
                img_A = np.fliplr(img_A)
                img_B = np.fliplr(img_B)

            imgs_A.append(img_A)
            imgs_B.append(img_B)

        imgs_A = np.array(imgs_A)/127.5 - 1.
        imgs_B = np.array(imgs_B)/127.5 - 1.

        return imgs_A, imgs_B

    def load_batch(self, batch_size=1, is_testing=False):
        path = glob(image_path)

        self.n_batches = int(len(path) / batch_size)

        for i in range(self.n_batches-1):
            batch = path[i*batch_size:(i+1)*batch_size]
            imgs_A, imgs_B = [], []
            for img in batch:
                bgr_image = cv2.imread(img, cv2.IMREAD_COLOR)
                img_A = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
                gray_img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
                img_B = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)

                img_A = scipy.misc.imresize(img_A, self.img_res)
                img_B = scipy.misc.imresize(img_B, self.img_res)

                if not is_testing and np.random.random() > 0.5:
                        img_A = np.fliplr(img_A)
                        img_B = np.fliplr(img_B)

                imgs_A.append(img_A)
                imgs_B.append(img_B)

            imgs_A = np.array(imgs_A)/127.5 - 1.
            imgs_B = np.array(imgs_B)/127.5 - 1.

            yield imgs_A, imgs_B
