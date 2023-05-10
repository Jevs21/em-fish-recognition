import sys
import os
import argparse
import json
import math
import numpy as np
import cv2 as cv

IMG_PATH  = os.path.join('training_images')

def main():
    images = [os.path.join(IMG_PATH, i) for i in os.listdir(IMG_PATH) if ".jpg" in i]
    annot = os.path.join(IMG_PATH, 'annotations.csv')
    content = []
    with open(annot, "r") as fp:
        content = fp.read().strip().split("\n")
        content = [c.split(",") for c in content]
        fp.close()

    keep = 0
    for im in images:
        basename = os.path.basename(im)
        print(basename, keep)
        for c in content:
            
            if c[0] == basename:
                x1 = int(c[1])
                y1 = int(c[2])
                x2 = int(c[3])
                y2 = int(c[4])

                if abs(x2 - x1) < 40 or abs(y2 - y1) < 40:
                    # print("Too small")
                    image = cv.imread(im)
                    cv.imwrite(os.path.join('training_images_baddetect', c[0]), image)

                    keep += 1
                    if keep > 100: return
                

                break

                # image = cv.imread(im)
                # gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
                # cv.rectangle(gray, (x1, y1), (x2, y2), (255, 0, 0), 2)

                # cv.imshow(f"Frame", gray)

                # key = cv.waitKey(1)
                # if key == ord('q'):
                #     cv.destroyAllWindows()
                #     return
                # elif key == ord('y'):
                #     print("Keep")
                # elif key == ord('n'):
                #     print("Don't keep")


    print(keep, "training images")



        # cv.imread(im)

    # unique_sizes = set()
    # for i in range(len(images)):
    #     cur_meta = parseMeta(meta[i])
    #     images, sizes = analyzeVideo(vids[i], cur_meta)
    #     for s in sizes:
    #         unique_sizes.add(s)
    
    #     # saveSummaryImage(base[i], images, sizes)
    #     # break
    
    # print("OUTPUT SIZES:")
    # for s in unique_sizes:
    #     print(s)





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