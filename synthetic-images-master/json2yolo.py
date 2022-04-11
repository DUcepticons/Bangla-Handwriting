# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 20:12:18 2020

@author: akash

Creaet a folder named 'labels' in same directory
txt files will be saved there
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 20:12:18 2020

@author: akash

Creaet a folder named 'labels' in same directory
txt files will be saved there
"""
import os

anns_dir= "annotations"

img_width= 1280
img_height=960

letter_to_num={'a':'0',
               'b':'1',
               'c':'2',
               'd':'3',
               'e':'4',
               'f':'5',
               'g':'6',
               'h':'7',
               'i':'8',
               'j':'9',
               'k':'10',
               }
for letter_folder in os.listdir(anns_dir):
    path = anns_dir+'/'+letter_folder
    with open(path+'/annotations.json') as f:
      #data = json.load(f)
      data = f.read()
      data = list(eval(data))
    
    print('Image count:', len(data))
    
    
    for item in data:
        fileName = item["path"].split('\\')[-1].replace('.png','')
        annotations = item["annotations"]
        for annotation in annotations:
            coordinates = annotation["coordinates"]
            height = coordinates["height"]
            width = coordinates["width"]
            x_center = coordinates["x"]
            y_center = coordinates["y"]
            label = annotation["label"]
        
            f = open("labels/"+letter_folder+'/'+fileName+".txt", "a")#"w" to overwrite
            
            
            #===============class x_center y_center width height
            f.write(letter_to_num[letter_folder] +' '+ str(x_center/img_width)+" "+str(y_center/img_height)+" "+str(width/img_width)+" "+str(height/img_height)+'\n')
            f.close()
            
    print("\nComplete!")

