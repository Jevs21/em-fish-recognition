import sys
import os
import shutil
import sys
import os
import argparse
import json
import math
import numpy as np
import cv2 as cv

DATA_PATH = os.path.join("training_data_neg")

def main():
    imgs = [os.path.join(DATA_PATH, v) for v in os.listdir(DATA_PATH) if ".jpg" in v]

    for i in range(len(imgs)):
        print(f"{imgs[i]} {i}/{len(imgs)}")
        if not keepImage(imgs[i]):
            os.remove(imgs[i])
        


            
def keepImage(p):
    image = cv.imread(p)
    cv.imshow("cur", image)
    key = cv.waitKey(0)
    if key == ord('y'):
        cv.destroyAllWindows()
        return True
    else:
        cv.destroyAllWindows()
        return False

    


if __name__ == '__main__':
    sys.exit(main())
