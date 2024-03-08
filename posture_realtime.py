import cv2
import math
import time
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from model import get_testing_model
import sys
from betterlib.config import ConfigFile
from betterlib.logging import Logger
import datetime

config = ConfigFile("config.json")
log = Logger("./processor.log", "PostureProcessor")

tic=0
# visualize
colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0],
		  [0, 255, 0], \
		  [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255],
		  [85, 0, 255], \
		  [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

def padRightDownCorner(img, stride, padValue):
    h = img.shape[0]
    w = img.shape[1]

    pad = 4 * [None]
    pad[0] = 0 # up
    pad[1] = 0 # left
    pad[2] = 0 if (h%stride==0) else stride - (h % stride) # down
    pad[3] = 0 if (w%stride==0) else stride - (w % stride) # right

    img_padded = img
    pad_up = np.tile(img_padded[0:1,:,:]*0 + padValue, (pad[0], 1, 1))
    img_padded = np.concatenate((pad_up, img_padded), axis=0)
    pad_left = np.tile(img_padded[:,0:1,:]*0 + padValue, (1, pad[1], 1))
    img_padded = np.concatenate((pad_left, img_padded), axis=1)
    pad_down = np.tile(img_padded[-2:-1,:,:]*0 + padValue, (pad[2], 1, 1))
    img_padded = np.concatenate((img_padded, pad_down), axis=0)
    pad_right = np.tile(img_padded[:,-2:-1,:]*0 + padValue, (1, pad[3], 1))
    img_padded = np.concatenate((img_padded, pad_right), axis=1)

    return img_padded, pad

timestamp_i = 0

def doTimestamp():
	global timestamp_i
	print(timestamp_i)
	timestamp_i += 1
	print(datetime.datetime.now())

def process (input_image, params, model_params):
	global timestamp_i
	''' Start of finding the Key points of full body using Open Pose.'''
	oriImg = input_image # B,G,R order  
	multiplier = [x * model_params['boxsize'] / oriImg.shape[0] for x in params['scale_search']]
	heatmap_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 19))
	paf_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 38))
	for m in range(1):
		scale = multiplier[m]
		imageToTest = cv2.resize(oriImg, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
		imageToTest_padded, pad = padRightDownCorner(imageToTest, model_params['stride'],
														  model_params['padValue'])
		input_img = np.transpose(np.float32(imageToTest_padded[:,:,:,np.newaxis]), (3,0,1,2)) # required shape (1, width, height, channels)
		output_blobs = model.predict(input_img)
		heatmap = np.squeeze(output_blobs[1])  # output 1 is heatmaps
		heatmap = cv2.resize(heatmap, (0, 0), fx=model_params['stride'], fy=model_params['stride'],
							 interpolation=cv2.INTER_CUBIC)
		heatmap = heatmap[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3],
				  :]
		heatmap = cv2.resize(heatmap, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)
		paf = np.squeeze(output_blobs[0])  # output 0 is PAFs
		paf = cv2.resize(paf, (0, 0), fx=model_params['stride'], fy=model_params['stride'],
						 interpolation=cv2.INTER_CUBIC)
		paf = paf[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3], :]
		paf = cv2.resize(paf, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)
		heatmap_avg = heatmap_avg + heatmap / len(multiplier)
		paf_avg = paf_avg + paf / len(multiplier)
	all_peaks = [] #To store all the key points which a re detected.
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

		peaks_binary = np.logical_and.reduce(
			(map >= map_left, map >= map_right, map >= map_up, map >= map_down, map > params['thre1']))
		peaks = list(zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]))  # note reverse
		peaks_with_score = [x + (map_ori[x[1], x[0]],) for x in peaks]
		id = range(peak_counter, peak_counter + len(peaks))
		peaks_with_score_and_id = [peaks_with_score[i] + (id[i],) for i in range(len(id))]

		all_peaks.append(peaks_with_score_and_id)
		peak_counter += len(peaks)


	canvas = input_image # B,G,R order

	for i in range(18): #drawing all the detected key points.
		for j in range(len(all_peaks[i])):
			cv2.circle(canvas, all_peaks[i][j][0:2], 4, colors[i], thickness=-1)
	position = checkPosition(all_peaks)
	timestamp_i = 0
	return canvas, position


def checkPosition(all_peaks):
	try:
		f = 0
		if (all_peaks[16]):
			a = all_peaks[16][0][0:2] #Right Ear
			f = 1
		else:
			a = all_peaks[17][0][0:2] #Left Ear
		b = all_peaks[11][0][0:2] # Hip
		angle = calcAngle(a,b)
		degrees = round(math.degrees(angle))
		if (f):
			degrees = 180 - degrees
		print("at " + str(degrees))
		if (degrees<config.get("webui").get("posture_thresholds").get("rounded")):
			return 1
		elif (degrees > config.get("webui").get("posture_thresholds").get("back")):
			return -1
		else:
			return 0
	except Exception as e:
		log.warn("Not in lateral view and unable to detect ears or hip")

def calcAngle(a, b):
	try:
		ax, ay = a
		bx, by = b
		if (ax == bx):
			return 1.570796
		return math.atan2(by-ay, bx-ax)
	except Exception as e:
		log.error("Unable to calculate angle")

def calcDistance(a,b): #calculate distance between two points.
	try:
		x1, y1 = a
		x2, y2 = b
		return math.hypot(x2 - x1, y2 - y1)
	except Exception as e:
		log.error("Unable to calculate distance")

    

if __name__ == '__main__':
	tic = time.time()
	log.info('start processing...')
	model = get_testing_model()
	model.load_weights('./model/keras/model.h5')
	
	# log.info out possible video inputs
	avail_cams = []
	for i in range(10):
		cap = cv2.VideoCapture(i)
		if cap.isOpened():
			avail_cams.append(i)
			cap.release()
	log.info("Available cameras: " + str(avail_cams))
	defcam = 1
	log.debug("Defaulting to camera " + str(defcam))
	camera_num = defcam
	cap=cv2.VideoCapture(camera_num)
	vi=cap.isOpened()

	if vi:
		cap.set(100,160)
		cap.set(200,120)
			
		while(1):
			tic = time.time()
		
			ret,frame=cap.read()
			# params, model_params = config_reader()
			params = config.get("params")
			model_params = config.get("model")
			canvas, position = process(frame, params, model_params)   
			log.info("Current back position: " + str(position)) 
			cv2.imshow("capture",canvas) 
			if cv2.waitKey(1) & 0xFF==ord('q'):
				break
		cap.release()
	else:
		log.critical("Unable to open camera, whoops")
	cv2.destroyAllWindows()    
