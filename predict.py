from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import load_model
# from google.colab.patches import cv2_imshow
from tensorflow import device
import random
import json
import numpy as np
import cv2
import os
import sys

DATA_PATH = os.path.abspath(os.path.join("..", "..", '..', "Downloads", "train_videos"))

test_vids = [os.path.join(DATA_PATH, i) for i in os.listdir(DATA_PATH) if ".mp4"  in i]
annot_p   = [os.path.join(DATA_PATH, i) for i in os.listdir(DATA_PATH) if ".json" in i]
# annot_p = os.path.join(INPUT_IMAGE_PATH, 'annotations.csv')

# random.shuffle(test_vids)
def run(ind, model):
    count  = 0
    c_count = 0
    actual = 0
    print(f"{test_vids[ind]}")

    content = {}
    with open(annot_p[ind], "r") as fp:
        content = json.loads(fp.read())
        fp.close()

    actual = len(content['detections'])
    
    p = test_vids[ind]
    cap = cv2.VideoCapture(p)
    frame = 0
    detected_frames = []
    was_detected = False
    while cap.isOpened():
        ret, image = cap.read()
        if not ret:
            break
        

        h, w = image.shape[:2]
        orig = image.copy()
        
        image = cv2.resize(image, (224,224))
        image = img_to_array(image) / 255.0
        image = np.expand_dims(image, axis=0)

        preds = model.predict(image)[0]
        x1, y1, x2, y2, detection = preds
        # detection = preds
        # print(detection)
        # if detection < 0.999:
        if detection < 0.8:
            cv2.rectangle(orig, (0, 0), (10, 10), (0, 0, 255), 5)
            was_detected = False
            # print(preds)
        else:
            
            cv2.rectangle(orig, (0, 0), (10, 10), (0, 255, 0), 5)
            for det in content['detections']:
                if int(det['frame']) == frame:
                    c_count += 1
                    break
        
            if not was_detected:
                count += 1
            #     print(f"Detected ({detection}) in frame {frame}")
            # else:
            #     print(f"Cont Detected ({detection}) in frame {frame}")

            was_detected = True
            detected_frames.append(frame)
        
        frame += 1
    
    
    d_count = 0
    for det in content['detections']:
        for d in detected_frames:
            if int(det['frame']) == d:
                d_count += 1
                break

    # for i in range(1, len(detected_frames)):
    #   if detected_frames[i - 1] == detected_frames[i] - 1:
    #     continue
    #   else:
    #     c_count += 1
    comp_ret = checkCompressionSuccess(detected_frames, content['detections'])
    # print(detected_frames)
    print(f"Detections: {count} ({c_count}) ({d_count}), Expected: {actual}, Total frames: {frame}, Compress?{comp_ret}")
    cap.release()


def checkCompressionSuccess(detected_frames, actual_detections, padding=5):
    success_count = 0
    for det in actual_detections:
        for f in detected_frames:
            if abs(int(det['frame']) - f) <= padding:
                success_count += 1
                break
    
    return (success_count == len(actual_detections))


def main(m_path, all=False):
    model = load_model(m_path)
    ind = random.randint(0, len(test_vids) - 1)

    for i in range(len(test_vids)):
        run(i, model)

    


if __name__ == '__main__':
    if len(sys.argv) == 2:
        sys.exit(main(sys.argv[1]))
    elif len(sys.argv) == 3:
        sys.exit(main(sys.argv[1], sys.argv[2]))
    else:
        print("wrong args (model path, all?)")
        sys.exit(-1)