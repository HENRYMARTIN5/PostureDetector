import cv2
import math
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from model import get_testing_model
from betterlib.config import ConfigFile
from betterlib.logging import Logger
import datetime

# Initialize config and logger
config = ConfigFile("config.json")
log = Logger("./processor.log", "PostureProcessor")

# Initialize colors for visualization
colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0],
          [0, 255, 0], [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255],
          [85, 0, 255], [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

# Function to pad image
def padRightDownCorner(img, stride, padValue):
    h, w = img.shape[:2]
    pad = [0, 0, 0, 0]
    pad[2] = stride - (h % stride) if (h % stride != 0) else 0
    pad[3] = stride - (w % stride) if (w % stride != 0) else 0

    img_padded = cv2.copyMakeBorder(img, 0, pad[2], 0, pad[3], cv2.BORDER_CONSTANT, value=padValue)

    return img_padded, pad

# Function to process input image
def process(input_image, params, model_params):
    oriImg = input_image
    multiplier = [x * model_params['boxsize'] / oriImg.shape[0] for x in params['scale_search']]
    heatmap_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 19))
    paf_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 38))

    for m in range(1):
        scale = multiplier[m]
        imageToTest = cv2.resize(oriImg, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        imageToTest_padded, pad = padRightDownCorner(imageToTest, model_params['stride'], model_params['padValue'])
        input_img = np.transpose(np.float32(imageToTest_padded[:,:,:,np.newaxis]), (3,0,1,2))
        output_blobs = model.predict(input_img)
        heatmap = np.squeeze(output_blobs[1])
        heatmap = cv2.resize(heatmap, (0, 0), fx=model_params['stride'], fy=model_params['stride'], interpolation=cv2.INTER_CUBIC)
        heatmap = heatmap[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3], :]
        heatmap = cv2.resize(heatmap, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)
        paf = np.squeeze(output_blobs[0])
        paf = cv2.resize(paf, (0, 0), fx=model_params['stride'], fy=model_params['stride'], interpolation=cv2.INTER_CUBIC)
        paf = paf[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3], :]
        paf = cv2.resize(paf, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)
        heatmap_avg += heatmap / len(multiplier)
        paf_avg += paf / len(multiplier)

    all_peaks = []
    peak_counter = 0
    
    for part in range(18):
        map_ori = heatmap_avg[:, :, part]
        map = gaussian_filter(map_ori, sigma=3)
        map_left = np.zeros(map.shape)
        map_left[1:, :] = map[:-1, :]
        map_right = np.zeros(map.shape)
        map_right[:-1, :] = map[1:, :]
        map_up = np.zeros(map.shape)
        map_up[:, 1:] = map[:, :-1]
        map_down = np.zeros(map.shape)
        map_down[:, :-1] = map[:, 1:]

        peaks_binary = np.logical_and.reduce((map >= map_left, map >= map_right, map >= map_up, map >= map_down, map > params['thre1']))
        peaks = list(zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]))
        peaks_with_score = [x + (map_ori[x[1], x[0]],) for x in peaks]
        id = range(peak_counter, peak_counter + len(peaks))
        peaks_with_score_and_id = [peaks_with_score[i] + (id[i],) for i in range(len(id))]

        all_peaks.append(peaks_with_score_and_id)
        peak_counter += len(peaks)

    canvas = input_image.copy()

    for i in range(18):
        for j in range(len(all_peaks[i])):
            cv2.circle(canvas, all_peaks[i][j][0:2], 4, colors[i], thickness=-1)

    position = checkPosition(all_peaks)
    return canvas, position

# Function to check position based on keypoints
def checkPosition(all_peaks):
    try:
        if all_peaks[16]:
            a = all_peaks[16][0][0:2]  # Right Ear
        else:
            a = all_peaks[17][0][0:2]  # Left Ear
        b = all_peaks[11][0][0:2]  # Hip
        angle = calcAngle(a, b)
        degrees = round(math.degrees(angle))
        if all_peaks[16]:
            degrees = 180 - degrees
        print("at " + str(degrees))
        if degrees < config.get("webui").get("posture_thresholds").get("rounded"):
            return 1
        elif degrees > config.get("webui").get("posture_thresholds").get("back"):
            return -1
        else:
            return 0
    except Exception as e:
        log.warn("Not in lateral view and unable to detect ears or hip")

# Function to calculate angle between two points
def calcAngle(a, b):
    try:
        ax, ay = a
        bx, by = b
        if ax == bx:
            return 1.570796
        return math.atan2(by - ay, bx - ax)
    except Exception as e:
        log.error("Unable to calculate angle")