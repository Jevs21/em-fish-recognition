import sys
import os
import argparse
import json
import math
import numpy as np
import cv2 as cv
from tensorflow import device

DATA_PATH = os.path.abspath(os.path.join("..", "..", '..', "Downloads", "train_videos"))
POS_OUT_PATH  = os.path.join("tdata_full_pos_final")
NEG_OUT_PATH  = os.path.join("tdata_full_neg_final")

def main():
    # emptyDirectory(POS_OUT_PATH)
    # emptyDirectory(NEG_OUT_PATH)
    base = [v.split(".")[0] for v in os.listdir(DATA_PATH) if ".mp4" in v]
    vids = [os.path.join(DATA_PATH, v) for v in os.listdir(DATA_PATH) if ".mp4" in v]
    meta = [os.path.join(DATA_PATH, v) for v in os.listdir(DATA_PATH) if ".json" in v]

    POS_N = 0
    NEG_N = 0
    n = 0
    pfp = open(os.path.join(POS_OUT_PATH, "annotations.csv"), "a")
    nfp = open(os.path.join(NEG_OUT_PATH, "annotations.csv"), "a")
    for j in range(len(vids)):
        print(f"Analyzing {base[j]} ({j}/{len(vids)}) [n={n}]")

        # if POS_N > 1000 and NEG_N > 1000:
        #     print("Got enough")
        #     break

        cur_meta = parseMeta(meta[j])
        # printMeta(cur_meta)

        # POSITIVE
        images, sizes = analyzeVideo(vids[j], cur_meta)
        pos_len = len(images)

        for i in range(len(images)):
            if sizes[i][4].upper() != "FLAT" and sizes[i][4].upper() != "NONE":
                continue
            
            if sizes[i][4].upper() != "NONE":
                if NEG_N < 1000:
                    cv.imwrite(os.path.join(POS_OUT_PATH, f"img{n}.jpg"), cv.resize(images[i], (224, 224)))
                    pfp.write(f"img{n}.jpg,{sizes[i][0]},{sizes[i][1]},{sizes[i][2]},{sizes[i][3]},{sizes[i][4].upper()}\n")
                    n += 1
                    # NEG_N += 1
            elif sizes[i][4].upper() != "FLAT":
                if POS_N < 1000:
                    cv.imwrite(os.path.join(NEG_OUT_PATH, f"img{n}.jpg"), cv.resize(images[i], (224, 224)))
                    nfp.write(f"img{n}.jpg,{sizes[i][0]},{sizes[i][1]},{sizes[i][2]},{sizes[i][3]},{sizes[i][4].upper()}\n")
                    n += 1
                    # POS_N += 1
        

        # NEGATIVE
        images, sizes = analyzeVideoNeg(vids[j], cur_meta)
        neg_count = 0
        for i in range(len(images)):

            if sizes[i][4].upper() != "FLAT" and sizes[i][4].upper() != "NONE":
                continue
            
            if sizes[i][4].upper() != "NONE":
                if NEG_N < 1000:
                    cv.imwrite(os.path.join(POS_OUT_PATH, f"img{n}.jpg"), cv.resize(images[i], (224, 224)))
                    pfp.write(f"img{n}.jpg,{sizes[i][0]},{sizes[i][1]},{sizes[i][2]},{sizes[i][3]},{sizes[i][4].upper()}\n")
                    n += 1
                    # NEG_N += 1
            elif sizes[i][4].upper() != "FLAT":
                if POS_N < 1000:
                    cv.imwrite(os.path.join(NEG_OUT_PATH, f"img{n}.jpg"), cv.resize(images[i], (224, 224)))
                    nfp.write(f"img{n}.jpg,{sizes[i][0]},{sizes[i][1]},{sizes[i][2]},{sizes[i][3]},{sizes[i][4].upper()}\n")
                    n += 1
                    # POS_N += 1
            
            if neg_count > (3 * pos_len):
                break

            neg_count += 1
            

    pfp.close()
    nfp.close()


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
    
def analyzeVideoNeg(v_path, meta):
    cap = cv.VideoCapture(v_path)

    extracted_images = []
    extracted_sizes  = []

    # print(meta)
    detected_frames = [int(d['frame']) for d in meta['detections']]

    f_num = 0
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            print("Cant recieve frame (video complete)")
            break
        
        is_blank = True
        for n in detected_frames:
            if abs(f_num - n) < 10:
                is_blank = False
                f_num += 1
                break
        
        if not is_blank:
            continue


        x2 = 0
        y2 = 0
        x1 = 0
        y1 = 0
        # sub = gray[y1:y2, x1:x2]
        
        extracted_images.append(frame)
        extracted_sizes.append([x1, y1, x2, y2, "NONE"])
                
        
        # if len(detection) == 0:
        #     extracted_images.append(frame)
        #     extracted_sizes.append([0, 0, 0, 0, "NONE"])


        f_num += 1

    
    cv.destroyAllWindows()

    return (extracted_images, extracted_sizes)


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
            
            extracted_images.append(frame)
            extracted_sizes.append([x1, y1, x2, y2, d['species']])
                
        
        # if len(detection) == 0:
        #     extracted_images.append(frame)
        #     extracted_sizes.append([0, 0, 0, 0, "NONE"])


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
    with device('/device:GPU:0'): 
        sys.exit(main())
    # with open("training_images/annotations.csv") as fp:
    #     for line in fp.readlines():
    #         p = line.rstrip().split(",")
            
    #         if int(p[1]) > int(p[3]):
    #             print(f"x issue {p[1]} > {p[3]} for {p[0]}")
    #             showImage(os.path.join(OUT_PATH, p[0]), int(p[1]), int(p[2]), int(p[3]), int(p[4]) )

    #         if int(p[2]) > int(p[4]):
    #             print(f"y issue {p[2]} > {p[4]} for {p[0]}")