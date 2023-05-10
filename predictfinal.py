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
def run(p, a_p, model):
    content = getMetadata(a_p)

    count = 0
    actual = len(content['detections'])
    
    
    cap = cv2.VideoCapture(p)
    frame = 0
    detected_frames = []
    det_vals = []
    was_detected = False
    while cap.isOpened():
        ret, image = cap.read()
        if not ret:
            break
        

        h, w = image.shape[:2]
        # orig = image.copy()
        
        image = cv2.resize(image, (224,224))
        image = img_to_array(image) / 255.0
        image = np.expand_dims(image, axis=0)

        preds = model.predict(image)[0]
        x1, y1, x2, y2, detection = preds

        # print(detection)
        det_vals.append(detection)
        if detection < 0.5:
            # cv2.rectangle(orig, (0, 0), (10, 10), (0, 0, 255), 5)
            was_detected = False
        else:
            # cv2.rectangle(orig, (0, 0), (10, 10), (0, 255, 0), 5)
            if not was_detected:
                count += 1

            was_detected = True
            detected_frames.append(frame)
        
        frame += 1
    

    # for i in range(1, len(detected_frames)):
    #   if detected_frames[i - 1] == detected_frames[i] - 1:
    #     continue
    #   else:
    #     c_count += 1
    comp_ret = checkCompressionSuccess(detected_frames, content['detections'])
    # print(detected_frames)
    print(f"Detections: {count} ({comp_ret['success']}={len(comp_ret['missed_frames'])}), Expected: {actual}, Total frames: {frame}")
    # if not comp_ret['success']:
    #     # print(comp_ret['missed_frames'])
    #     for m in comp_ret['missed_frames']:
    #         print(f"frame {m}: {det_vals[m]}")

    ratio = len(compress(detected_frames)) / frame 
    print(f"Compression: {frame} -> {len(compress(detected_frames))} ({ratio})")
    
    # for d in det_vals:
    #     print(str(d))
    
    cap.release()

    return ratio


def getMetadata(p):
    content = {}
    with open(p, "r") as fp:
        content = json.loads(fp.read())
        fp.close()
    
    if 'detections' not in content:
        print(f"Error getting metadata {p}")
        sys.exit(-1)
    
    return content


def checkCompressionSuccess(detected_frames, actual_detections, padding=5):
    success_count = 0
    missed_frames = []
    for det in actual_detections:
        was_found = False
        for f in detected_frames:
            if abs(int(det['frame']) - f) <= padding:
                success_count += 1
                was_found = True
                break

        if not was_found:
            missed_frames.append(int(det['frame']))
    
    return { 'success': (success_count == len(actual_detections)), 'missed_frames': missed_frames }


def compress(detected_frames, padding=5):
    final_frames = []
    for d in detected_frames:
        for i in range(d-padding, d+padding):
            if i not in final_frames:
                final_frames.append(i)
    
    return sorted(final_frames)


def main(m_path):
    models = []
    if ".h5" in m_path:
        models = [load_model(m_path)]
    else:
        for m in os.listdir(m_path):
            if ".h5" in m:
                print(f"Loading model {m}")
                models.append(load_model(os.path.join(m_path, m)))

    for i in range(len(test_vids)):
        for m in range(len(models)):
            print(f"\nModel {m+1}: {test_vids[i]}")
            run(test_vids[i], annot_p[i], models[m])

    


if __name__ == '__main__':
    if len(sys.argv) == 2:
        sys.exit(main(sys.argv[1]))
    elif len(sys.argv) == 3:
        sys.exit(main(sys.argv[1], sys.argv[2]))
    else:
        print("wrong args (model path, all?)")
        sys.exit(-1)