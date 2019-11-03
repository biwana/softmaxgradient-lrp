import sys
import shutil
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import os

def crop_resize_img(imgfile, source_dir, target_dir):
    img = Image.open(os.path.join(source_dir, imgfile))
    width, height = img.size

    left = 0
    upper = 0
    right = width
    lower = height

    # center crop
    if width > height:
        left = (width - height) // 2
        right = left + height
    else:
        upper = (height - width) // 2
        lower = upper + width

    img = img.crop((left, upper, right, lower))
    
    img = img.resize((224, 224))
    width, height = img.size
    
    if width != 224 or height != 224:
        print("bad resize")

#     # resize to 256 x 256
#     img = img.resize((256, 256))
    
#     # final crop
#     new_width = 224
#     new_height = 224
    
#     width, height = img.size

#     left = width // 2 - new_width // 2
#     upper = height // 2 - new_height // 2
#     right = width // 2 + new_width // 2
#     lower = height // 2 + new_height // 2
    
#     img = img.crop((left, upper, right, lower))
    
    img.save(os.path.join(target_dir, imgfile))
    return
    
def main():
    source_dir = './data/coco/val2017'
    target_dir = './data/coco/val2017-center-crop-224x224'
    if not os.path.isdir(target_dir):
        os.mkdir(target_dir)
        
    img_path_list = os.listdir(source_dir)
    for image_path in tqdm(img_path_list):
        if image_path.endswith('.jpg'):
#             print(image_path)
            crop_resize_img(image_path, source_dir, target_dir)

if __name__ == "__main__":
    main()