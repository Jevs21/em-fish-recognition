import sys
import os
import argparse
import json
import math
import numpy as np
import cv2 as cv

DATA_PATH = os.path.abspath(os.path.join("..", "..", '..', "Downloads", "training_videos_sub"))

def main():
    base = [v.split(".")[0] for v in os.listdir(DATA_PATH) if ".mp4" in v]
    vids = [os.path.join(DATA_PATH, v) for v in os.listdir(DATA_PATH) if ".mp4" in v]
    meta = [os.path.join(DATA_PATH, v) for v in os.listdir(DATA_PATH) if ".json" in v]

    unique_sizes = set()
    for i in range(len(vids)):
        print(f"Analyzing {base[i]}")

        cur_meta = parseMeta(meta[i])
        images, sizes = analyzeVideo(vids[i], cur_meta)
        for s in sizes:
            unique_sizes.add(s)
    
        # saveSummaryImage(base[i], images, sizes)
        break
    
    print("OUTPUT SIZES:")
    for s in unique_sizes:
        print(s)


def parseMeta(p):
    content = {}
    with open(p, "r") as fp:
        content = json.loads(fp.read())
        fp.close()
    return content


def analyzeVideo(v_path, meta):
    cap = cv.VideoCapture(v_path)

    extracted_images = []
    extracted_sizes  = []

    f_num = 0
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            print("Cant recieve frame (stream end?)")
            break
        
        is_det = False
        for d in meta['detections']:
            # print(d)
            if int(d['frame'])== f_num:
                # print("Yee")
                x2 = int(d['x'])
                y2 = int(d['y'])
                x1 = int(d['w'])
                y1 = int(d['h'])

                cv.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                is_det = True
                break

        cv.imshow('frame', frame)

        f_num += 1
        wait_val = 20 if not is_det else 0

        if cv.waitKey(wait_val) == ord('q'):
            break
    
    cv.destroyAllWindows()

    return (extracted_images, extracted_sizes)

if __name__ == '__main__':
    sys.exit(main())