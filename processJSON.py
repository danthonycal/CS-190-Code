from centroidTracker import CentroidTracker
from scipy.spatial import distance as dist
from progressBar import printProgressBar
from collections import OrderedDict
from math import *
import os
import cv2
import json
import glob
import pprint
import numpy as np

def angle_trunc(a):
	while a < 0.0:
		a += pi * 2
	return a

def check_sign(vec, dim):
	# Check sign with respect to origin axis (center of frame)
	if vec == 0: return 0 # if at origin
	elif vec < (dim//2): return -(vec - dim//2) # if negative
	return (vec - dim//2) # if positive

def get_angle_n_magnitude(x1, y1, x2, y2, width, height):
	# Convert image coordinates to cartesian coordinate
	#  checks if x coordinates are positive or negative
	#  with respect to x-axis(horizontal center of frame)
	x1 = check_sign(x1, width)
	x2 = check_sign(x2, width)

	#  checks if y coordinates are positive or negative
	#  with respect to y-axis(vertical center of frame)
	y1 = check_sign(y1, height)
	y2 = check_sign(y2, height)

	# convert points into position vector
	# ⟨→A⟩ minus ⟨→B⟩
	delta_y = y1 - y2
	delta_x = x1 - x2
	
	# get magnitude of position vector 
	# |→BA|
	# magnitude = sqrt((delta_x**2)+(delta_y**2))	
	magnitude = hypot(delta_x, delta_y)

	# get angle of position vector with respect to positive x-axis
	# ⟨→BA⟩
	angle = atan2(delta_y, delta_x) / (pi / 180)
	
	if angle > 0:
		if y1 < y2:
			return angle, magnitude
	
		return (180+angle), magnitude
	
	if (x1 < x2):
		return (180+angle), magnitude
	
	return (360+angle), magnitude

def bbox2points(bbox):
	# From bounding box yolo format
	# to corner points cv2 rectangle
	x, y, w, h = bbox
	xmin = int(round(x - (w / 2)))
	xmax = int(round(x + (w / 2)))
	ymin = int(round(y - (h / 2)))
	ymax = int(round(y + (h / 2)))
	return xmin, ymin, w, h


def computeFlags(obj):
	ang_flag = 0
	mag_flag = 0
	# cnt_flag = 0
	prev_ang = 0
	prev_mag = 0
	# cnt = 0
	for i in reversed(range(10)):
		ang = obj[f"ang_{i}"]
		if i == 0:
			prev_ang = ang
		delta_ang = np.abs(ang-prev_ang)
		a = np.array((obj["center_x"],obj["center_y"]))
		b = np.array((obj["prev_x_10"],obj["prev_y_10"]))
		dist = np.linalg.norm(a-b)
		# print(dist)
		mag = obj[f"mag_{i}"]

		if delta_ang > 60:
			ang_flag += 1
		prev_ang = ang

		if mag < 400 and dist < 400:
			mag_flag += 1
	return ang_flag, mag_flag

def get_fps(cap):
	#Get version of OpenCV
	(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

	if int(major_ver) < 3:
		return cap.get(cv2.cv.CV_CAP_PROP_FPS)
	return cap.get(cv2.CAP_PROP_FPS)

def get_key_from_value(d, val):
    keys = [k for k, v in d.items() if v == val]
    if keys:
        return keys[0]
    return None

def process():
	states   = None
	res_json = []
	fps      = 0
	prev_cx  = 0
	prev_cy  = 0
	vid_name = ""
	frame_no = ""
	result_files = glob.glob(f"{os.getcwd()}\\videos\\frames\\darknet-result-json\\*.JSON")

	# print(result_files[0])
	for ind in range(len(result_files)):
		with open(result_files[ind], 'r') as JSONFile:
			res_json = json.load(JSONFile)
			ct = CentroidTracker()
			state = []

			print(f"Tracking centroids of {result_files[ind]}")
			printProgressBar(0, len(res_json),prefix='Progress:', suffix='Complete', length=50)

			for i in range(len(res_json)):
				vid_name = result_files[ind][-22:-12]
				frame_no = res_json[i]["filename"][55:-4]
				frame_id = res_json[i]["filename"][44:-4]
				
				video_capture = cv2.VideoCapture(f"./videos/{vid_name}.MP4")
				
				height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
				width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
				total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
				
				fps = get_fps(video_capture)
				state.append(OrderedDict())
				
				rects     = []
				angles    = []
				mags      = []
				confs     = []
				angle     = 0
				magnitude = 0

				res_json[i]["objects"] = list(filter(lambda i: i['name'] == 'person', res_json[i]["objects"]))
				for j in range(len(res_json[i]["objects"])):
					object_id = f"{j}_{frame_no}_{vid_name}"
					center_x  = float(res_json[i]["objects"][j]["relative_coordinates"]["center_x"]*width)
					center_y  = float(res_json[i]["objects"][j]["relative_coordinates"]["center_y"]*height)
					b_width   = float(res_json[i]["objects"][j]["relative_coordinates"]["width"]*width)
					b_height  = float(res_json[i]["objects"][j]["relative_coordinates"]["height"]*height)

					rects.append(bbox2points([center_x,center_y,b_width,b_height]))
					
					res_json[i]["objects"][j][f"angle"]     = angle
					res_json[i]["objects"][j][f"magnitude"] = magnitude
					
					confs.append(res_json[i]["objects"][j]["confidence"])
					ct.update(rects, int(fps), confs, int(frame_no))
					# pprint.pprint(dict(ct.bbox))

				state[i]["frame_id"] = frame_id
				state[i]["objects"]  = []
				for key in ct.objects.keys():
					temp = {
						"id"		 : key,
						"center_x"	 : float(ct.objects[key][0][0]/width),
						"center_y"	 : float(ct.objects[key][0][1]/height),
						"prev_x_10"	 : float(ct.objects[key][len(ct.objects[key])-1][0]/width),
						"prev_y_10"	 : float(ct.objects[key][len(ct.objects[key])-1][1]/height),
						"width"	  	 : float(ct.bbox[key][2]/width),
						"height"  	 : float(ct.bbox[key][3]/height),
						"frame_age"  : 0,
						"confidence" : ct.confidences[key]
					}
					
					if len(ct.objects[key]) >= 10:
						temp["prev_x_10"] = float(ct.objects[key][9][0]/width)
						temp["prev_y_10"] = float(ct.objects[key][9][1]/height)

					for age in range(len(ct.objects[key])):
						temp["frame_age"] += 1
										
					for num in range(0,10):
						if (num+1) < len(ct.objects[key]):
							center_x  = float(ct.objects[key][num][0])
							center_y  = float(ct.objects[key][num][1])
							prev_x    = float(ct.objects[key][num+1][0])
							prev_y    = float(ct.objects[key][num+1][1])
							angle, magnitude = get_angle_n_magnitude(center_x,center_y,prev_cx,prev_cy,width,height)
							temp[f"ang_{num}"] = angle
							temp[f"mag_{num}"] = magnitude
							
							continue
						temp[f"ang_{num}"] = 0.0
						temp[f"mag_{num}"] = 0.0

					state[i]["objects"].append(temp)

				printProgressBar(i + 1, len(res_json), prefix='Progress:',suffix='Complete', length=50)
				video_capture.release()

		for i in range(len(state)):
			ang_flag = 0
			mag_flag = 0
			cnt_flag = 0
			ped_no	 = 0 
			for k in range(len(state[i]["objects"])):
				ang_flag, mag_flag = computeFlags(state[i]["objects"][k])
				frame_age = state[i]["objects"][k]["frame_age"]
				ped_no = len(state[i]["objects"])
				if ((ang_flag >= 7) or (mag_flag >= 7)) and (frame_age >= 20) and (ped_no == 1):
					state[i]["objects"][k]["class"] = 1
					continue
				
				state[i]["objects"][k]["class"] = 0

		
		with open(f"./videos/frames/darknet-result-json/processed/{vid_name}_result_processed.json", 'w') as out:
			json.dump(state, out, indent=2)

if __name__ == "__main__":
	process()
