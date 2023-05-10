import sys
import os
import argparse
import json
import math
import numpy as np
import cv2 as cv

DATA_PATH = os.path.abspath(os.path.join("..", "..", '..', "Downloads", "train_videos"))
OUT_PATH  = os.path.join("training_images")

def main():
    emptyDirectory(OUT_PATH)
    base = [v.split(".")[0] for v in os.listdir(DATA_PATH) if ".mp4" in v]
    vids = [os.path.join(DATA_PATH, v) for v in os.listdir(DATA_PATH) if ".mp4" in v]
    meta = [os.path.join(DATA_PATH, v) for v in os.listdir(DATA_PATH) if ".json" in v]

    n = 0
    annot_fn = os.path.join(OUT_PATH, "annotations.csv")
    fp = open(annot_fn, "a")
    for i in range(len(vids)):
        print(f"Analyzing {base[i]} ({i}/{len(vids)})")

        cur_meta = parseMeta(meta[i])
        # printMeta(cur_meta)
        images, sizes = analyzeVideo(vids[i], cur_meta)

        for i in range(len(images)):
            if sizes[i][4].upper() != "FLAT":
                continue
            
            cv.imwrite(os.path.join(OUT_PATH, f"img{n}.jpg"), images[i])
            fp.write(f"img{n}.jpg,{sizes[i][0]},{sizes[i][1]},{sizes[i][2]},{sizes[i][3]},{sizes[i][4].upper()}\n")
            n += 1

    fp.close()
        # break


def emptyDirectory(d):
    for item in os.listdir(d):
        os.remove(os.path.join(d, item))

def appendAnnotation(n, data):
    annot_fn = os.path.join(OUT_PATH, "annotations.csv")
    with open(annot_fn, "a") as fp:
        fp.write(f"img{n}.jpg,{data[0]},{data[1]},{data[2]},{data[3]},{data[4].upper()}\n")
        fp.close()


def showImage(i_path, x1, y1, x2, y2):
    image = cv.imread(i_path)
    cv.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

    cv.imshow('frame', image)
    if cv.waitKey(0) == ord('q'):
        cv.destroyAllWindows()
    
    return
    

def analyzeVideo(v_path, meta):
    cap = cv.VideoCapture(v_path)

    extracted_images = []
    extracted_sizes  = []

    f_num = 0
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            print("Cant recieve frame (video complete)")
            break
        
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        detection = list(filter(lambda box: int(box['frame']) == f_num, meta['detections']))

        for d in detection:
            x2 = int(d['x'])
            y2 = int(d['y'])
            x1 = int(d['w'])
            y1 = int(d['h'])
            # sub = gray[y1:y2, x1:x2]
            
            extracted_images.append(gray)
            extracted_sizes.append([x1, y1, x2, y2, d['species']])


        f_num += 1

    
    cv.destroyAllWindows()

    return (extracted_images, extracted_sizes)


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
    # with open("training_images/annotations.csv") as fp:
    #     for line in fp.readlines():
    #         p = line.rstrip().split(",")
            
    #         if int(p[1]) > int(p[3]):
    #             print(f"x issue {p[1]} > {p[3]} for {p[0]}")
    #             showImage(os.path.join(OUT_PATH, p[0]), int(p[1]), int(p[2]), int(p[3]), int(p[4]) )

    #         if int(p[2]) > int(p[4]):
    #             print(f"y issue {p[2]} > {p[4]} for {p[0]}")