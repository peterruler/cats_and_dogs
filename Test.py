import pandas as pd
import matplotlib.pyplot as plt
import os
from PIL import Image
import random

# Load train and test data
test_dir = '/Users/peterstroessler/Documents/Projects/cats_and_dogs/catdog/test'
# Load the CSV file using pandas
data_frame = pd.read_csv('result.csv')

id_list = []
class_ = {0: 'cat', 1: 'dog'}

fig, axes = plt.subplots(2, 5, figsize=(20, 12), facecolor='w')

for ax in axes.ravel():
    
    i = random.choice(data_frame['id'].values)
    label = data_frame.loc[data_frame['id'] == i, 'label'].values[0]
    if label > 0.5:
        label = 1
    else:
        label = 0
        
    img_path = os.path.join(test_dir, '{}.jpg'.format(i))
    img = Image.open(img_path)
    
    ax.set_title(class_[label])
    ax.imshow(img)
plt.axis('off')
plt.show()