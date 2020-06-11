#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras 
import os 
import shutil
from keras.preprocessing import image


img_dir = r'C:\Users\myung\AI\images'
fnames = ['imgDog{}.jpg'.format(i) for i in range(10)]

img_name = 'imgDog6.jpg' #for x in range(10) .format(i)
img_path = os.path.join(img_dir, img_name)

img = image.load_img(img_path, target_size=(250, 250))
img_tensor = image.img_to_array(img)

img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor.shape
(1, 250, 250, 3)
 
 # scaling into [0, 1]
img_tensor /= 255

plt.rcParams['figure.figsize'] = (10, 10)
plt.imshow(img_tensor[0])
plt.show()
print(img_tensor[0])

