import numpy as np
import cv2 as cv
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import math

def random_noise(img):
    noise = np.random.uniform(.95, 1.05, (50, 50))
    new_img = img * noise
    new_img = new_img / np.max(new_img)
    return new_img
            

ds_path = './train'
tds = []
for item in tqdm(os.listdir(ds_path)):
    path = f'{ds_path}/{item}'
    img = cv.imread(path, 0)
    img = cv.resize(img, (50, 50)) / 255
    noisy = random_noise(img)
    tds.append([noisy, img])

tds = np.array(tds).astype(np.float32)
np.random.shuffle(tds)
print(tds.shape)
np.save('ds.npy', tds)


