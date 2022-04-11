import os
import subprocess
import shutil

input_dir = 'D:\Test'
output_dir = 'D:\Segmentation_Dataset'

if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
    os.mkdir(output_dir)
else:
    os.mkdir(output_dir)

for dirs in os.listdir(input_dir):
    sub_dirs = os.path.join(input_dir,dirs)

    if os.path.exists(os.path.join(output_dir,dirs)):
            shutil.rmtree(os.path.join(output_dir,dirs))
            os.mkdir(os.path.join(output_dir,dirs))
    else:
        os.mkdir(os.path.join(output_dir,dirs))

    for sub_dir in os.listdir(sub_dirs):
        files = os.path.join(sub_dirs, sub_dir)
        outputdir = os.path.join(output_dir,dirs,sub_dir)

        if os.path.exists(outputdir):
            shutil.rmtree(outputdir)
            os.mkdir(outputdir)
        else:
            os.mkdir(outputdir)

        i = files + "\\"
        o = outputdir + "\\"
        print(i,o)
        command = "python create.py -obj " + i + " -o " + o
        subprocess.call(command, shell=True)
        

            

        
