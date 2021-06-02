import sys
import cv2
import numpy as np
import argparse
import time
import datetime
import matplotlib.pyplot as plt
from args import Args
from tqdm import tqdm
from math import *
from skimage.io import imread
from skimage.transform import resize
from collections import deque
from scipy.spatial import distance as dist
from centroidTracker import CentroidTracker
from collections import OrderedDict

def restart_line():
    sys.stdout.write('\r')
    sys.stdout.flush()

def angle_trunc(a):
	while a < 0.0:
		a += pi * 2
	return a

def set_color(object_count):
	if object_count > 1: return (200,0,200) # Purple if more than one person is detected
	return	(0,255,0) #green if only one person is detected

def check_sign(vec, dim):
	# Check sign with respect to origin axis (center of frame)
	if vec == 0: return 0 # if at origin
	elif vec < (dim//2): return -(vec - dim//2) # if negative
	return (vec - dim//2) # if positive

def get_angle_n_magnitude(centroid_a, centroid_b, width, height):
	a = []
	b = []

	a = np.array(centroid_a)
	b = np.array(centroid_b)

	# Convert image coordinates to cartesian coordinate
	#  checks if x coordinates are positive or negative
	#  with respect to x-axis(horizontal center of frame)
	a[0] = check_sign(a[0], width)
	b[0] = check_sign(b[0], width)

	#  checks if y coordinates are positive or negative
	#  with respect to y-axis(vertical center of frame)
	a[1] = check_sign(a[1], height)
	b[1] = check_sign(b[1], height)

	# convert points into position vector
	# ⟨→A⟩ minus ⟨→B⟩
	deltaY = a[1] - b[1]
	deltaX = a[0] - b[0]
	
	# get magnitude of position vector 
	# |→BA|
	magnitude = sqrt((deltaX**2)+(deltaY**2))	

	# get angle of position vector with respect to positive x-axis
	# ⟨→BA⟩
	angle = atan2(deltaY, deltaX) * 180 / pi
	if a[1] > b[1]:
		angle = (2*pi) - angle
	if angle < 0:
		angle += 360 
	
	return angle, magnitude

def get_direction(angle):
	if 75.0 <= angle <= 105.0:
		return f"South ({angle})"
	elif (0.0 <= angle <= 15.0) or (345.0 <= angle <= 360):
		return f"West ({angle})"
	elif 255.0 <= angle <= 285.0:
		return f"North ({angle})"
	elif 165.0 <= angle <= 195.0:
		return f"East ({angle})"
	elif 15.0 < angle < 75.0:
		return f"South-West ({angle})"
	elif 105.0 < angle < 165.0:
		return f"South-East ({angle})"
	elif 195.0 < angle < 255.0:
		return f"North-East ({angle})"
	elif 285.0 < angle < 345.0:
		return f"North-West ({angle})"
	return ""

def bbox2points(bbox):
	# From bounding box yolo format
	# to corner points cv2 rectangle
	x, y, w, h = bbox
	xmin = int(round(x - (w / 2)))
	xmax = int(round(x + (w / 2)))
	ymin = int(round(y - (h / 2)))
	ymax = int(round(y + (h / 2)))
	return xmin, ymin, xmax, ymax

def draw_boxes(detections, image, indexes):
	objects = detections.bbox
	color = (200,0,200)
	if len(detections.objects) == 1:
		color = (0,255,0)
	for (objectID, bbox) in objects.items():
		if objectID in indexes:
			left, top, right, bottom = bbox2points(bbox[0])
			cv2.circle(image, (bbox[0][0],bbox[0][1]), 4, color, -1)
			cv2.rectangle(image, (left, top), (right, bottom), color, 1)
			cv2.putText(image, f"ID: {objectID} ({bbox[0][0]},{bbox[0][1]})",
						(left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
						color, 2)
	return image

def detect_count(detections, image, indexes):
	objects = detections.bbox
	count = 0
	for (objectID, bbox) in objects.items():
		if objectID in indexes:
			count+=1
	cv2.putText(image, f"{count} person(s) detected.", (275, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
	return image

def get_fps(cap):
	#Get version of OpenCV
	(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
	
	if int(major_ver) < 3:
	    return cap.get(cv2.cv.CV_CAP_PROP_FPS)
	return cap.get(cv2.CAP_PROP_FPS)
