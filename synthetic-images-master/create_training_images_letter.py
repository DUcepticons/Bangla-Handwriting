import os
import subprocess
import shutil

input_dir = 'D:\Handwriting_Ext_Files\Bangla_Handwriting_Dataset_Augmented_16K'
output_dir = 'D:\Handwriting_Ext_Files\Synthesized_Dataset_Yolo'

if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
    os.mkdir(output_dir)
else:
    os.mkdir(output_dir)

for dirs in os.listdir(input_dir):
    files = os.path.join(input_dir,dirs)

    if os.path.exists(os.path.join(output_dir,dirs)):
            shutil.rmtree(os.path.join(output_dir,dirs))
            os.mkdir(os.path.join(output_dir,dirs))
    else:
        os.mkdir(os.path.join(output_dir,dirs))


    outputdir = os.path.join(output_dir,dirs)

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
        

            

        
