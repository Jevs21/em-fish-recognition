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

DATA_PATH = os.path.join("..", "..", '..', "Downloads", "train_videos")
OUT_PATH  = os.path.join("..", "..", '..', "Downloads", "training_videos_sub")

def main():
    # emptyDirectory(OUT_PATH)
    base = [v.split(".")[0] for v in os.listdir(DATA_PATH) if ".mp4" in v]
    vids = [os.path.join(DATA_PATH, v) for v in os.listdir(DATA_PATH) if ".mp4" in v]
    meta = [os.path.join(DATA_PATH, v) for v in os.listdir(DATA_PATH) if ".json" in v]

    print(len(vids))

    for i in range(len(vids)):
        v_path = vids[i]
        m_path = meta[i]
        o_path = os.path.join(OUT_PATH, base[i] + ".mp4")
        if os.path.exists(o_path):
            print(f"{i} already exists.")
            continue
        else:
            print(f"{i}/{len(vids)}")

        ret = showFirstFrame(v_path)
        if ret:
            print("keep")
            shutil.copy2(v_path, OUT_PATH)
            shutil.copy2(m_path, OUT_PATH)
        else:
            print("No")

            


def emptyDirectory(d):
    for item in os.listdir(d):
        os.remove(os.path.join(d, item))


def showFirstFrame(v_path):
    cap = cv.VideoCapture(v_path)
    ret, frame = cap.read()

    if not ret:
        print("Cant recieve frame (video complete)")
        return False

    #Display the resulting frame
    cv.imshow(f"Frame", frame)

    #Set waitKey 
    key = cv.waitKey(0)
    if key == ord('y'):
        cv.destroyAllWindows()
        return True
    else:
        cv.destroyAllWindows()
        return False
    
    


if __name__ == '__main__':
    sys.exit(main())
