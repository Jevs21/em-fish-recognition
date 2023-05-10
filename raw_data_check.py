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

    unique_sizes = set()
    for i in range(len(vids)):
        print(f"Analyzing {base[i]}")

        cur_meta = parseMeta(meta[i])
        images, sizes = analyzeVideo(vids[i], cur_meta)
        for s in sizes:
            unique_sizes.add(s)
    
        # saveSummaryImage(base[i], images, sizes)
        # break
    
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

    for d in meta['detections']:
        ret1 = cap.set(cv.CAP_PROP_POS_FRAMES, int(d['frame']))
        # print(ret1)
        ret, frame = cap.read()

        # print(ret)
        # print("-------------", frame.shape)

        #Set grayscale colorspace for the frame. 
        # gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # x2 = int(d['x'])
        # y2 = int(d['y'])
        # x1 = int(d['w'])
        # y1 = int(d['h'])
        # sub = gray[y1:y2, x1:x2]
        
        # extracted_sizes.append((x2-x1, y2-y1))
        extracted_sizes.append((frame.shape[0], frame.shape[1]))

        # cv.rectangle(gray, (x1, y1), (x2, y2), (255, 0, 0), 2)

        #Display the resulting frame
        # cv.imshow(f"Frame {d['frame']}", gray)

        #Set waitKey 
        # if cv.waitKey(0) == ord('q'):
        #     break


        # When everything done, release the capture
    cap.release()
    cv.destroyAllWindows()

    # f_num = 0
    # while cap.isOpened():
    #     ret, frame = cap.read()

    #     if not ret:
    #         print("Cant recieve frame (stream end?)")
    #         break
        
    #     gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    #     detection = list(filter(lambda box: int(box['frame']) == f_num, meta['detections']))

    #     for d in detection:
    #         x2 = int(d['x'])
    #         y2 = int(d['y'])
    #         x1 = int(d['w'])
    #         y1 = int(d['h'])
    #         sub = gray[y1:y2, x1:x2]
            
    #         extracted_images.append(sub)
    #         extracted_sizes.append((x2-x1, y2-y1))

    #         cv.rectangle(gray, (x1, y1), (x2, y2), (255, 0, 0), 2)

    #     # cv.imshow('frame', gray)

    #     f_num += 1

    #     # if cv.waitKey(1) == ord('q'):
    #     #     break
    
    # cv.destroyAllWindows()

    return (extracted_images, extracted_sizes)

if __name__ == '__main__':
    sys.exit(main())