# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 16:28:25 2020

@author: User
"""


import Augmentor
import os
import shutil


def move_file():
    source = "C:\\Users\\User\\Desktop\\Riad's File\\Generated Dataset\\Riad\\k\\0.6\\output\\"
    #dest1 = "C:\\Users\\User\\Desktop\\Riad's File\\Generated Dataset\\Riad\\k\\0.6"
    #dest2 = "C:\\Users\\User\\Desktop\\Riad's File\\Generated Dataset\\b"
    #dest3 = "C:\\Users\\User\\Desktop\\Riad's File\\Generated Dataset\\c"
    files = os.listdir(source)
    for f in files:
        shutil.move(source+f, dest1)
        '''if f[0]=="a":
            shutil.move(source+f, dest1)
        elif f[0]=="b":
            shutil.move(source+f, dest2)
        elif f[0]=="c":
            shutil.move(source+f, dest3)'''
    os.rmdir("C:\\Users\\User\\Desktop\\Riad's File\\Generated Dataset\\Riad\\k\\0.6\\output")

# Function to rename multiple files 
s="C:\\Users\\User\\Desktop\\Riad's File\\Generated Dataset\\Riad\\k\\0.9\\output\\"
def rename_all():
   i = 1
   j=1
   k=1
   count=1501
   path=s
   for filename in os.listdir(path):
       
      my_dest ="k" + "-1"+"-"+str(count)+".jpg"
      if i==56:
          j+=1
      if i<=10:
          my_dest ="a" + "-"+str(j)+"-"+str(count)+".jpg"
          count+=1
      elif i<=20:
          if count==11: 
              count=1
          my_dest ="b" + "-"+str(j)+"-"+str(count) + ".jpg"
          count+=1
      elif i<=30:
          if count==11: 
              count=1
          my_dest ="c" + "-"+str(j)+"-"+str(count) + ".jpg"
          count+=1
      count+=1
      my_source =path + filename
      my_dest =path + my_dest
      # rename() function will
      # rename all the files
      os.rename(my_source, my_dest)
      i += 1
      
  
p = Augmentor.Pipeline("C:\\Users\\User\\Desktop\\Riad's File\\Generated Dataset\\Riad\\k\\0.9")
#p.ground_truth("/path/to/ground_truth_images")
# Add operations to the pipeline as normal:
p.rotate(probability=1, max_left_rotation=10, max_right_rotation=10)
p.random_brightness(probability=1,min_factor=0.8,max_factor=1.3)
p.random_contrast(probability=1,min_factor=0.5,max_factor=1.5)
#p.skew_left_right(probability=1,magnitude=0.4)
#p.random_distortion(probability=1, grid_width=10, grid_height=10, magnitude=10)
#p.skew(probability=0.8)
p.sample(500)
rename_all()
move_file()
