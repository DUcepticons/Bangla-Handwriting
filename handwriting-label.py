import os
import csv

path="D:/Bangla Handwriting Project/Bangla Handwriting Dataset Processed/Labelling/g/0.9" #Change path here

files = []
# r=root, d=directories, f = files
for r, d, f in os.walk(path):
    for file in f:
        if '.jpg' in file:
            files.append(os.path.join(r, file))

for f in files:
    print(f)
    filename= f[:-3]+"csv"
    
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([0, 0, 0, 0, 0, 0, 0.9, 0, 0, 0, 0]) #Change value and letter here
        
    print(filename)