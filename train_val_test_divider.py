import shutil
from tqdm import tqdm 
import os 
import random

LOCATION = 'D:\Github Projects\Bangla-Handwriting\Datasets\Bangla Handwriting Dataset - Augmented'
DESTINATION = 'D:\Github Projects\Bangla-Handwriting\Datasets\Bangla Handwriting Dataset - Augmented - Splitted'

# tqdm is only used for interactive loading 
# loading the training data 
for letterfolder in os.listdir(LOCATION): 
    
    letter_path = os.path.join(LOCATION, letterfolder) 
    for qualityfolder in os.listdir(letter_path):
        quality_path = os.path.join(letter_path, qualityfolder) 

        for img in tqdm(os.listdir(quality_path)):   
            path = os.path.join(quality_path, img)  
            split = "Train"
            n = random.random()
            if n<0.8:
                split = "Train"
            elif n>=0.8 and n<0.9:
                split = "Validation"
            else: 
                split = "Test"
                
            target = DESTINATION+'/'+split+'/'+letterfolder+'/'+qualityfolder+'/'

            os.makedirs(os.path.dirname(target), exist_ok=True)
            
            target_file = os.path.join(target,img)

            shutil.copy(path, target)
