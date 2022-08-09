import os

dir = "D:\Github Projects\Bangla-Handwriting\Datasets\Landmark detection\\k-original\\"

for count, filename in enumerate(os.listdir(dir)):
        dst ='k-'+str(count+1) + ".jpg"
        #dst ="flipped-"+ filename[:-8] + "good.mp4"
        src =dir+ filename
        dst =dir+ dst
          
        # rename() function will
        # rename all the files
        os.rename(src, dst)

