import os
path='cityscapes/leftImg8bit_trainvaltest_foggy/leftImg8bit_foggy'
path='cityscapes/leftImg8bit_trainvaltest\leftImg8bit'
count=0
for root,dirs,files in os.walk(path):
    count+=len(files)
print(count)