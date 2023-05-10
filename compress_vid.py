import sys
import os
import argparse
import json
import math
import numpy as np
import cv2 as cv

DATA_PATH = os.path.abspath(os.path.join("..", "..", '..', "Downloads", "train_videos"))

def main():
    base = [v.split(".")[0] for v in os.listdir(DATA_PATH) if ".mp4" in v]
    vids = [os.path.join(DATA_PATH, v) for v in os.listdir(DATA_PATH) if ".mp4" in v]
    meta = [os.path.join(DATA_PATH, v) for v in os.listdir(DATA_PATH) if ".json" in v]

    for i in range(len(vids)):
        print(f"Analyzing {base[i]}")

        frames = compressVid(vids[i])
        f_num  = len(frames)
        h, w, c = frames[0].shape

        
        for x in range(w):
            for y in range(h):

                stack = []
                
                for f in frames:
                    print(f[y, x])
                    break
                break
            break


        break
    



def compressVid(p):
    cap = cv.VideoCapture(p)

    frames = []

    f_num = 0
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            print("Cant recieve frame (stream end?)")
            break

        # cv.imshow('frame', frame)
        
        frames.append(frame)
        # gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        f_num += 1

        # if cv.waitKey(0) == ord('q'):
        #     break
    
    cap.release()
    cv.destroyAllWindows()

    return frames

if __name__ == '__main__':
    sys.exit(main())