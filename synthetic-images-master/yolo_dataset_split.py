
import os
import shutil
import random

train_percentage=0.8
#val and test percentage will be equal of remaining percentage
val_range= train_percentage + (1 - train_percentage)/2

current_dataset_path='D:\Handwriting_Ext_Files\handwriting_yolo_source'
yolo_dataset_path='D:\Handwriting_Ext_Files\handwriting_yolo_dataset'

image_folder_path = os.path.join(current_dataset_path,'images')
label_folder_path = os.path.join(current_dataset_path,'labels') 


img_destination=''
label_destination=''
for file in os.listdir(image_folder_path):
    img_source = os.path.join(image_folder_path,file)
    
    file_name = file[:-4]
    label_source = os.path.join(label_folder_path,file_name+'.txt')
    
    
    random_number = random.random()
    if random_number <= train_percentage:
        img_destination = os.path.join(yolo_dataset_path,'train','images')
        label_destination = os.path.join(yolo_dataset_path,'train','labels')
    elif random_number > train_percentage and random_number <= val_range:
        img_destination = os.path.join(yolo_dataset_path,'valid','images')
        label_destination = os.path.join(yolo_dataset_path,'valid','labels')
    else:
        img_destination = os.path.join(yolo_dataset_path,'test','images')
        label_destination = os.path.join(yolo_dataset_path,'test','labels')
    
    if not os.path.exists(img_destination):
        os.makedirs(img_destination)                 # only if it does not yet exist
    shutil.move(img_source, img_destination) # add source dir to filename

    if not os.path.exists(label_destination):
        os.makedirs(label_destination)                 # only if it does not yet exist
    shutil.move(label_source, label_destination) # add source dir to filename