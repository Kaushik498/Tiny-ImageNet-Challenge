import os 
import numpy as np  
from PIL import Image 

# file opening
dataPath = r'/media/kirito/New Volume1/data/tiny-imagenet-200/train/n01443537/images'

#traverses every file available in the directory
img = next(os.walk(dataPath))[2]

for imgs in img : 
	# reads each image 
	img1 = Image.open(dataPath+'/'+imgs)
	img1.show()
	break 
