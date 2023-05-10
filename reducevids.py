import sys
import os
import argparse
import json
import math
import numpy as np
import cv2 as cv
from tensorflow import device

DATA_PATH = os.path.abspath(os.path.join("..", "..", '..', "Downloads", "training_videos_sub"))
OUT_PATH = os.path.join("reducedvids")

def main():
    # emptyDirectory(OUT_PATH)
    base = [v.split(".")[0] for v in os.listdir(DATA_PATH) if ".mp4" in v]
    vids = [os.path.join(DATA_PATH, v) for v in os.listdir(DATA_PATH) if ".mp4" in v]
    meta = [os.path.join(DATA_PATH, v) for v in os.listdir(DATA_PATH) if ".json" in v]

    n = 0

    sizes = []
    for j in range(len(vids)):
        print(f"Analyzing {base[j]} ({j}/{len(vids)})")
        cur_meta = parseMeta(meta[j])

        out_p = compressVid(vids[j], cur_meta)

        orig_size = os.path.getsize(vids[j])
        out_size  = os.path.getsize(out_p)

        orig_rt   = getRuntime(vids[j])
        out_rt    = getRuntime(out_p)

        res = float(orig_size) / float(out_size)
        rres = float(orig_rt) / float(out_rt)

        sizes.append((orig_size, out_size, res, orig_rt, out_rt, rres))

        # if j > 2: break
    

    for s in sizes:
        print(s)
    
    print(sum([s[2] for s in sizes]) / len(sizes))
    print(f"{sum([s[3] for s in sizes]) / len(sizes)} -> {sum([s[4] for s in sizes]) / len(sizes)} ({sum([s[5] for s in sizes]) / len(sizes)})")

       



def emptyDirectory(d):
    for item in os.listdir(d):
        os.remove(os.path.join(d, item))


    
def compressVid(v_path, meta):
    o_path = os.path.join(OUT_PATH, os.path.basename(v_path))
    cap = cv.VideoCapture(v_path)
    return o_path

    images = []

    f_num = 0
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            print("Cant recieve frame (video complete)")
            break
        
        for d in meta['detections']:
            if int(d['frame']) == f_num:
                images.append(frame)
                break

        f_num += 1

    
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    out = cv.VideoWriter(o_path, fourcc, 2, (1280, 720))
    
    for i in range(len(images)):
        out.write(images[i])
    out.release()

    return o_path


def getRuntime(v_path):
    cap = cv.VideoCapture(v_path)
    fps = cap.get(cv.CAP_PROP_FPS)

    f_num = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Cant recieve frame (video complete)")
            break
        f_num += 1
    
    return round(float(f_num) / float(fps))



def parseMeta(p):
    content = {}
    with open(p, "r") as fp:
        content = json.loads(fp.read())
        fp.close()
    return content


def printMeta(m):
    print(json.dumps(m, sort_keys=True, indent=4))



if __name__ == '__main__':
    sys.exit(main())
