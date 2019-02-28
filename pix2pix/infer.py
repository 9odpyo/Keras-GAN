import numpy as np
import scipy
import matplotlib.pyplot as plt
import cv2
from builder import build_generator


def inference():
    prev_weight = 'saved_model/model_248_acc99.853515625.weights'
    image_path = 'target.png'
    model = build_generator()

    model.load_weights(prev_weight)

    gray_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    gray_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
    gray_img = scipy.misc.imresize(gray_img, (256, 256))
    gray_img = np.fliplr(gray_img)
    gray_img = np.array([gray_img]) / 127.5 - 1.

    fake_img = model.predict(gray_img)
    fake_img = 0.5 * fake_img + 0.5
    plt.imshow(fake_img[0])
    plt.show()


inference()
