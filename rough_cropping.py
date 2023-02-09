import os
import shutil
import argparse

# Get Facial Landmarks
import cv2
import random
from mtcnn import MTCNN

parser = argparse.ArgumentParser()
parser.add_argument('--in_root', type=str, default="", help='process folder')
# parser.add_argument('--out_root', type=str, default="", help='output folder')
args = parser.parse_args()
indir = args.in_root
# out_dir = args.out_root

detector = MTCNN()

imgs = sorted([x for x in os.listdir(indir) if x.endswith(".jpg") or x.endswith(".png")])
out_detection = os.path.join(indir, "detections")
if os.path.exists(out_detection):
    shutil.rmtree(out_detection)
os.makedirs(out_detection)
for img in imgs:
    src = os.path.join(indir, img)
    print(src)
    if img.endswith(".jpg"):
        dst = os.path.join(out_detection, img.replace(".jpg", ".txt"))
    if img.endswith(".png"):
        dst = os.path.join(out_detection, img.replace(".png", ".txt"))

    if not os.path.exists(dst):
        image = cv2.cvtColor(cv2.imread(src), cv2.COLOR_BGR2RGB)
        # import pdb; pdb.set_trace()
        result = detector.detect_faces(image)

        if len(result)>0:
            index = 0
            if len(result)>1: # if multiple faces, take the biggest face
                size = -100000
                for r in range(len(result)):
                    size_ = result[r]["box"][2] + result[r]["box"][3]
                    if size < size_:
                        size = size_
                        index = r

            bounding_box = result[index]['box']
            keypoints = result[index]['keypoints']
            if result[index]["confidence"] > 0.9:

                if img.endswith(".jpg"):
                    dst = os.path.join(out_detection, img.replace(".jpg", ".txt"))
                if img.endswith(".png"):
                    dst = os.path.join(out_detection, img.replace(".png", ".txt"))

                outLand = open(dst, "w")
                outLand.write(str(float(keypoints['left_eye'][0])) + " " + str(float(keypoints['left_eye'][1])) + "\n")
                outLand.write(str(float(keypoints['right_eye'][0])) + " " + str(float(keypoints['right_eye'][1])) + "\n")
                outLand.write(str(float(keypoints['nose'][0])) + " " +      str(float(keypoints['nose'][1])) + "\n")
                outLand.write(str(float(keypoints['mouth_left'][0])) + " " + str(float(keypoints['mouth_left'][1])) + "\n")
                outLand.write(str(float(keypoints['mouth_right'][0])) + " " + str(float(keypoints['mouth_right'][1])) + "\n")
                outLand.close()
                print(result)   
                
                
# Crop and Align Image

from PIL import Image
import numpy as np
import sys
sys.path.append('Deep3DFaceRecon_pytorch')
from Deep3DFaceRecon_pytorch.util.preprocess import align_img
from Deep3DFaceRecon_pytorch.util.load_mats import load_lm3d

lm_dir = os.path.join(indir, "detections")
img_files = sorted([x for x in os.listdir(indir) if x.lower().endswith(".png") or x.lower().endswith(".jpg")])
lm_files = sorted([x for x in os.listdir(lm_dir) if x.endswith(".txt")])

lm3d_std = load_lm3d("Deep3DFaceRecon_pytorch/BFM/") 

out_dir = os.path.join(indir, 'rough')
os.makedirs(out_dir, exist_ok=True)

if not os.path.exists(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    
for img_file, lm_file in zip(img_files, lm_files):

    img_path = os.path.join(indir, img_file)
    lm_path = os.path.join(lm_dir, lm_file)
    im = Image.open(img_path).convert('RGB')
    _,H = im.size
    lm = np.loadtxt(lm_path).astype(np.float32)
    lm = lm.reshape([-1, 2])
    lm[:, -1] = H - 1 - lm[:, -1]
    
    target_size = 1024.
    rescale_factor = 300
    center_crop_size = 700
    output_size = 512

    _, im_high, _, _ = align_img(im, lm, lm3d_std, target_size=target_size, rescale_factor=rescale_factor)
    
    left = int(im_high.size[0]/2 - center_crop_size/2)
    upper = int(im_high.size[1]/2 - center_crop_size/2)
    right = left + center_crop_size
    lower = upper + center_crop_size

    im_cropped = im_high.crop((left, upper, right,lower))
    im_cropped = im_cropped.resize((output_size, output_size), resample=Image.LANCZOS)
    out_path = os.path.join(out_dir, img_file.split(".")[0] + ".png")
    im_cropped.save(out_path)