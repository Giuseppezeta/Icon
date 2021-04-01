import cv2
import os
from numpy import array, int32


def pre_process(path, outpath):
  frame_skip = 20
  i=0
  for name in os.listdir(path):
    n=0
    for name2 in os.listdir(os.path.join(path,name)):
      for image in os.listdir(os.path.join(path,name,name2)):
        if n > frame_skip:
          n=0
          print(os.path.join(path,name,name2,image))
          img = cv2.imread(os.path.join(path,name,name2,image))
          #cv2.imshow("a",img)
          #cv2.waitKey(0)
          file_path = os.path.join(outpath, "frame_" + str(i))
          new_path = file_path + '.jpg'
          cv2.imwrite(new_path, img)
          i = i + 1
        else:
          n = n + 1

path = r'\Users\Giuseppe\Desktop\CPT\train'
path2 = r'\Users\Giuseppe\Desktop\CPT\test'
out = r'\Users\Giuseppe\Desktop\IC\train'
out2 = r'\Users\Giuseppe\Desktop\IC\test'
pre_process(path, out)
pre_process(path2, out2)