# Apple Silicon M1 install Pytorch

- Explanation: https://www.youtube.com/watch?v=VEDy-c5Sk8Y
- Commands: https://github.com/jeffheaton/t81_558_deep_learning/blob/pytorch/install/pytorch-install-aug-2022.ipynb
- Source: https://github.com/jeffheaton/t81_558_deep_learning/tree/pytorch/install

# Uninstall anaconda (if you have a previous installation, otherwise please skip this paragraph)

- https://docs.anaconda.com/free/anaconda/install/uninstall/
- `conda install anaconda-clean`
- `anaconda-clean --yes`
``` 
sudo rm -rf anaconda3
sudo rm -rf ~/anaconda3
sudo rm -rf ~/opt/anaconda3

```

- `Ctrl+Shift+p` enter `code` add to PATH
```
source  ~/.bash_profile
# in a VSCode terminal, type in:
sudo code  ~/.bash_profile  
```

# Install miniconda for ARM64 M1 Mac

- https://docs.conda.io/en/latest/miniconda.html

- To Check:
```
> python
> install platform
> platform.platform()
# outputs:
'macOS-13.4.1-arm64-arm-64bit'
```

# Activate pytorch when torch isn't found
 
 - `conda activate torch`

# Development environment

- IDE, VS Code download: https://code.visualstudio.com/
- Check Python `python --version` and `which python`
- I had a previous  `brew install python` which worked well
- Use visualstudio code (free) in combination with coderunner plugin, press play to run actual python script

# Pytorch Cats & Dogs Tutorial (German)

Youtube deutsch Pytorch Cats & Dogs CV:

- #13 https://www.youtube.com/watch?v=32lHVbT09h8
- #14 https://www.youtube.com/watch?v=cNwMpWt6IHk
- #15 https://www.youtube.com/watch?v=Zj5QkjmYmBI
- #16 https://www.youtube.com/watch?v=dDO7ihzkoC4
- #17 https://www.youtube.com/watch?v=RkcRqphggLY
- #18 https://www.youtube.com/watch?v=ulCylfDJRKo
- #19 https://www.youtube.com/watch?v=kabjuJWLvus
- #20 https://www.youtube.com/watch?v=26esNjWkEHA
- #21 https://www.youtube.com/watch?v=PjlFR-PdwXk
- #22 https://www.youtube.com/watch?v=P08d-1RnczM
- #23 https://www.youtube.com/watch?v=rJUZfEJZas8

# Unzip dataset and create working directory

- trainingdata used in this exercise, is from: 

https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition
in a folder named `catdog/train` and `catdog/test`
(to kaggle.com, login with google)

# Download the cats and dogs dataset (additional)

Additional traningdata not used in this exercise (login with google):
````
wget --no-check-certificate \
    https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip \
    -O cats_and_dogs_filtered.zip
````



https://www.kaggle.com/code/adinishad/pytorch-cats-and-dogs-classification
````
unzip cats_and_dogs_filtered.zip
mkdir -p cats_and_dogs/train
mv cats_and_dogs_filtered/train/cats/* cats_and_dogs/train
mv cats_and_dogs_filtered/train/dogs/* cats_and_dogs/train
`````

# Jupyter Notebook

- https://www.youtube.com/watch?v=2WL-XTl2QYI

- `conda install -y jupyter` install jupyter notebook
- `which jupyter` outputs something like `Users/peterstroessler/miniconda3/bin/jupyter` (add to your PATH)
- to run call something like (with your username) `/Users/peterstroessler/miniconda3/bin/jupyter notebook`


# OpenCV

- `# import cv2` (cv2 is obsolete)


# Second implementation of cats and dogs (working)

- https://github.com/vashiegaran/Pytorch-CNN-with-cats-and-dogs-/blob/main/CatvsDog.ipynb

```
conda activate torch
# new gernerate csv evaluation
python CatDog2.py 
```

# Load train and test data replace with your path
```
train_dir = '/Users/peterstroessler/Documents/Projects/cats_and_dogs/catdog/train'
test_dir = '/Users/peterstroessler/Documents/Projects/cats_and_dogs/catdog/test'
import glob

train_list = glob.glob(os.path.join(train_dir,'*.jpg'))
test_list = glob.glob(os.path.join(test_dir, '*.jpg'))
```

# Display sample image source in plot

```
#original
random_idx = np.random.randint(1,1000,size=10)

fig = plt.figure()
i=1
for idx in random_idx:
    ax = fig.add_subplot(2,5,i)
    img = Image.open(train_list[idx])
    plt.imshow(img)
    i+=1

plt.axis('off')
plt.show()
```

# Demo / Run Test.py
- Test / read of bulk (already evaluated) csv file!
- In a console run: `python Test.py`