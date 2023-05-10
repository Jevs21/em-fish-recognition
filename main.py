import sys
import os
import argparse
import json
import math
import numpy as np
import cv2 as cv
import tensorflow as tf

DATA_PATH = os.path.abspath(os.path.join("..", "..", '..', "Downloads", "train_videos"))

def main():
    base = [v.split(".")[0] for v in os.listdir(DATA_PATH) if ".mp4" in v]
    vids = [os.path.join(DATA_PATH, v) for v in os.listdir(DATA_PATH) if ".mp4" in v]
    meta = [os.path.join(DATA_PATH, v) for v in os.listdir(DATA_PATH) if ".json" in v]

    for i in range(len(vids)):
        print(f"Analyzing {base[i]}")

        cur_meta = parseMeta(meta[i])
        # printMeta(cur_meta)
        images, sizes = analyzeVideo(vids[i], cur_meta)
        saveSummaryImage(base[i], images, sizes)
        break


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
        
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        detection = list(filter(lambda box: int(box['frame']) == f_num, meta['detections']))

        for d in detection:
            x2 = int(d['x'])
            y2 = int(d['y'])
            x1 = int(d['w'])
            y1 = int(d['h'])
            sub = gray[y1:y2, x1:x2]
            
            extracted_images.append(sub)
            extracted_sizes.append((x2-x1, y2-y1))

            cv.rectangle(gray, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # cv.imshow('frame', gray)

        f_num += 1

        # if cv.waitKey(1) == ord('q'):
        #     break
    
    cv.destroyAllWindows()

    return (extracted_images, extracted_sizes)


def saveSummaryImage(name, imgs, sizes):
    amt = len(imgs)
    imgs_y = math.ceil(math.sqrt(amt))
    imgs_x = min(10, imgs_y)

    print(f"{amt} images")
    print(imgs_x)
    print(imgs_y)
    max_x   = 0
    total_y = 0

    for s in sizes:
        max_x = max(max_x, s[0])
        total_y += s[1]
    
    res = np.zeros((total_y, max_x, 3), np.uint8)

    cur_y = 0
    for i in range(len(imgs)):
        for y in range(sizes[i][1]):
            for x in range(sizes[i][0]):
                res[cur_y][x] = imgs[i][y][x]
            cur_y += 1

    cv.imshow('summary', res)
    if cv.waitKey(0) == ord('q'):
        cv.destroyAllWindows()
    
    

    # print(f"{max_x}x{total_y}")



def parseMeta(p):
    content = {}
    with open(p, "r") as fp:
        content = json.loads(fp.read())
        fp.close()
    return content


def printMeta(m):
    print(json.dumps(m, sort_keys=True, indent=4))


def parseArgs():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input-data", type=str, required=False,
        help="Path to input data directory")
    args = vars(ap.parse_args())
    
    if args['input_data']:
        DATA_PATH = args['input_data']
    

    return args


if __name__ == '__main__':
    args = parseArgs()

    sys.exit(main())
