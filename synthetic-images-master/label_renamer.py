import os

#rename letter folders so there is total image count for all letters


count = 0

synthesized_path= 'D:\Github Projects\Bangla-Handwriting\synthetic-images-master\labels'

for letter_folder in os.listdir(synthesized_path):
    for file in os.listdir(os.path.join(synthesized_path,letter_folder)):
        file_path = os.path.join(synthesized_path,letter_folder,file)
        
        new_name=str(count)+'-syn.txt'
        new_path= os.path.join(synthesized_path,letter_folder,new_name)
        os.rename(file_path,new_path)
        count = count + 1
        