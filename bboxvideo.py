import sys
import os
import argparse
import json
import math
import numpy as np
import cv2 as cv
from tensorflow import device
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img

DATA_PATH = os.path.abspath(os.path.join("..", "..", '..', "Downloads", "train_videos"))
MODEL_PATHS = [os.path.join("models", "bbox_full_apr17", f"model_e{i}.h5") for i in range(1, 11, 2)]
# MODEL_PATHS = [os.path.join("models", "model_e1_neg_aug.h5")] 

def main():
    models = loadModels()
    print(f"{len(models)} models loaded...")
    base = [v.split(".")[0] for v in os.listdir(DATA_PATH) if ".mp4" in v]
    vids = [os.path.join(DATA_PATH, v) for v in os.listdir(DATA_PATH) if ".mp4" in v]
    meta = [os.path.join(DATA_PATH, v) for v in os.listdir(DATA_PATH) if ".json" in v]

    n = 0

    for j in range(len(vids)):
        print(f"Analyzing {base[j]} ({j}/{len(vids)})")

        cur_meta = parseMeta(meta[j])

        images, sizes = analyzeVideo(vids[j], cur_meta, models)
        # break



def loadModels():
    ret = []
    for m in MODEL_PATHS:
        ret.append(load_model(m))
    return ret
            


def emptyDirectory(d):
    for item in os.listdir(d):
        os.remove(os.path.join(d, item))



def showImage(i_path, x1, y1, x2, y2):
    image = cv.imread(i_path)
    cv.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

    cv.imshow('frame', image)
    if cv.waitKey(0) == ord('q'):
        cv.destroyAllWindows()
    
    return





def analyzeVideo(v_path, meta, models):
    cap = cv.VideoCapture(v_path)

    extracted_images = []
    extracted_sizes  = []

    f_num = 0
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            print("Cant recieve frame (video complete)")
            break
        
        h, w = frame.shape[:2]

        print("Detecting meta...")
        meta_detected = False
        mx1, my1, mx2, my2 = 0, 0, 0, 0
        for d in meta['detections']:
            if int(d['frame']) == f_num:
                mx2 = int(d['x'])
                my2 = int(d['y'])
                mx1 = int(d['w'])
                my1 = int(d['h'])
                meta_detected = True
                break
        
        print("Resizing Image...")
        # Prep image for input layer of network
        image = cv.resize(frame, (224,224))
        image = img_to_array(image) / 255.0
        image = np.expand_dims(image, axis=0)

        print("Predicting...")
        # Capture all predected bounding boxes
        bboxes = []
        print(models)
        for mod in models:
            print("Predicting")
            coords = mod.predict(image)[0]
            bboxes.append(coords)
        
        
        if meta_detected:
            cv.rectangle(frame, (mx1, my1), (mx2, my2), (0, 0, 255), 1)
        
        for b in bboxes:
            x1, y1, x2, y2, confidence = b
            x1 = int(x1 * w)
            y1 = int(y1 * h)
            x2 = int(x2 * w)
            y2 = int(y2 * h)
            print(confidence)
            cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
        
        cv.imshow('frame', frame)
        if cv.waitKey(0) == ord('q'):
            break
                
        
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