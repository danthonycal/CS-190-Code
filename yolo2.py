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

def save_centroid(detections,indexes,frameCount,frameNo,fps):
	csv = Data(int(fps))
	objects = detections.bbox
	angle = 0
	magnitude = 0
	dir_ = ""
	for (objectID, bbox) in objects.items():
		if objectID in indexes:
			x, y, w, h = bbox[0]
			# Get projected direction of object
			if len(bbox) >= 5: 
				angle, magnitude = get_angle_n_magnitude(bbox[0], bbox[4], w, h)
			else:
				angle, magnitude = get_angle_n_magnitude(bbox[0], bbox[len(bbox)-1], w, h)
			
			dir_ = get_direction(angle)
			# print(f"Frame: {int(frameNo)} FPS: {fps} Direction: {dir_}")

			# Write centroid, magnitude, and angle into csv
			# every second (number of frames / fps)
			if (frameCount%fps)!=0:
				# Writing to .csv every frame (for demo purposes)
				csv = csv.update("data.csv", objects, indexes, angle, magnitude)		
				continue
			# Writing to .csv
			csv = csv.update("datapersecond.csv", objects, indexes, angle, magnitude)

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

def load_yolo(weights, cfg, names):
	net = cv2.dnn.readNet(weights, cfg)
	
	# Save all names in a list of classes
	classes = []
	with open(names, "r") as f:
		classes = [line.strip() for line in f.readlines()]
	
	# Get layers of the network
	layer_names = net.getLayerNames()

	# Determine the output layer names from the YOLO model
	output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

	return net, classes, layer_names, output_layers

def detect_centroids(outs, class_ids, classes, boxes, confidences, labels, width, height, ctr):
	for out in outs:
		for detection in out:
			scores = detection[5:]
			class_id = np.argmax(scores)
			confidence = scores[class_id]
			
			# Centroid detected
			center_x = int(detection[0] * width)
			center_y = int(detection[1] * height)
			
			in_contour = cv2.pointPolygonTest(ctr,(center_x,center_y),True) 
			# If the detected object's center is within the area of detection
			if (in_contour >= 0) and ((confidence > 0.45) and (classes[class_id] in labels)):
				# Object detected	
				w = int(detection[2] * width)
				h = int(detection[3] * height)

				# Rectangle coordinates
				x = int(center_x - w / 2)
				y = int(center_y - h / 2)

				if len(boxes) == 0:
					class_ids.append(class_id)
					boxes.append([x, y, w, h])
					confidences.append(float(confidence))
				else:
					ind_ = 0
					if len(boxes) > 1:
						ind_ == 1
					prev_contour = [
						[boxes[ind_][0],boxes[ind_][1]],
						[boxes[ind_][2],boxes[ind_][1]],
						[boxes[ind_][0],boxes[ind_][3]],
						[boxes[ind_][2],boxes[ind_][3]],
					]
					prev_contour = np.array(prev_contour).reshape((-1,1,2)).astype(np.int32)

					prev_ctr = np.array((int(boxes[ind_][0] + (boxes[ind_][2] / 2)), int(boxes[ind_][1] + (boxes[ind_][3] / 2)))) 
					center = np.array((center_x, center_y))

					in_prev_cnt = cv2.pointPolygonTest(prev_contour,(center_x,center_y),True)
					dist_a_b = np.linalg.norm(prev_ctr - center)
					if (in_prev_cnt >= 0) and (dist_a_b > 10):
						class_ids.append(class_id)
						boxes.append([x, y, w, h])
						confidences.append(float(confidence))

	return class_ids, boxes, confidences

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

def set_start_end(args,fps,total_frames):
	start = int(args.time_start*fps)
	if args.duration != -1: return start, int(start+(args.duration*fps))
	return start, total_frames

def yolov3_tracker(path="./videos/GOPR7987.MP4"):
	fps = 0
	labels		 = ['person']
	confidences	 = []
	class_ids	 = []
	boxes		 = []
	objects		 = []
	args	 = Args()
	data	 = OrderedDict()

	# Load Yolo
	print("LOADING YOLO")
	net, classes, layer_names, output_layers = load_yolo(args.weights, args.config, "coco.names")
	print("YOLO LOADED")

	#Capture frame-by-frame
	cap1 = cv2.VideoCapture(args.video_path)
	total_frames = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT)) 

	#Get framerate of video
	fps = get_fps(cap1)
	print(f"FPS: {fps}")
	
	start, end = set_start_end(args,fps,total_frames)

	# Set time start  (in seconds)
	cap1.set(cv2.CAP_PROP_POS_FRAMES, start)
	
	frameNo = 0
	frameCount = frameNo
	centroids = deque(maxlen=args.buffer)

	data["class_ids"] = []
	data["boxes"] = []
	data["confidences"] = []
	data["indexes"] = []

	height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT)*args.scale)
	width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH)*args.scale)

	contour_points=[[int(0.13*width),int(0.77*height)], [int(0.64*width),int(0.94*height)], [int(0.67*width),int(0.37*height)], [int(0.67*width),int(0.31*height)], [int(0.48*width),int(0.31*height)], [int(0.09*width),int(0.62*height)]]
	ctr = np.array(contour_points).reshape((-1,1,2)).astype(np.int32)
	print(f"Total Frames: {end}")
	
	#Initialize centroid tracker
	csv = Data(int(fps))
	ct = CentroidTracker()
	ind=0
	while(cap1.isOpened()):
		_, frame = cap1.read()
		
		sys.stdout.write("Processing: ")
		sys.stdout.write(f"{int(frameCount)}/{end} frames")
		sys.stdout.flush()
		restart_line()

		if not _:
			break
		if frameNo > end:
			break
		if frameNo < start:
			frameNo += 1
			frameCount += 1
			continue
		# if (frameCount%fps) != 0:
		# 	frameNo += 1
		# 	frameCount += 1
		# 	continue
		
		frame = cv2.resize(frame, None, fx=args.scale, fy=args.scale, interpolation=cv2.INTER_LANCZOS4)
		height, width, channels = frame.shape
		# Show area of detection
		# Contour points that make up bounding area
	
		# USing blob function of opencv to preprocess image
		blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)

		#Detecting objects
		net.setInput(blob)
		outs = net.forward(output_layers)

		class_ids, boxes, confidences = detect_centroids(outs, class_ids, classes, boxes, confidences, labels, width, height, ctr)

		#We use NMS function in opencv to perform Non-maximum Suppression
		#we give it score threshold and nms threshold as arguments.
		indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.4)
		
		data["class_ids"].append(class_ids)
		data["boxes"].append(boxes)
		data["confidences"].append(confidences)
		data["indexes"].append(indexes)
		ct = ct.update(tuple(data["boxes"][ind]), int(fps), frameNo)
		# if (frameCount%fps)==0:
		ind += 1
		frameNo += 1
		frameCount += 1

	cap1.release()
	cap2 = cv2.VideoCapture(args.video_path)

	frameNo = 0
	frameCount = frameNo

	# Set time start  (in seconds)
	cap2.set(cv2.CAP_PROP_POS_FRAMES, start)
	print(f"Video start at {str(datetime.timedelta(seconds=args.time_start))}")

	result = 'yolov3-output1.MP4'
	frame_num = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
	frame_width = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
	frame_height = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
	
	sys.stdout.write(f"Video dims: {frame_width}x{frame_height}")
	sys.stdout.flush()

	fourcc = cv2.VideoWriter_fourcc(*'avc1')
	output_vid = cv2.VideoWriter(result,fourcc,fps,(frame_width,frame_height))

	ind = 0
	while True:
		_, frame = cap2.read()
		sys.stdout.write("Writing: ")
		sys.stdout.write(f"{int(frameCount)}/{end} frames")
		sys.stdout.flush()
		restart_line()
		
		if not _:
			break

		frame = cv2.resize(frame, None, fx=args.scale, fy=args.scale, interpolation=cv2.INTER_LANCZOS4)
		height, width, channels = frame.shape
		
		# Draw contour outline to frame
		cv2.drawContours(frame,[ctr],0,(255, 0, 0), 2)
		cv2.putText(frame, f"Frame No.: {int(frameNo)} | Seconds: {datetime.timedelta(seconds=frameNo//fps)}", (275, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
		
		if (frameNo > end) or (ind > (len(data["boxes"])-1)):
			break
		if frameNo < start:
			frameNo += 1
			frameCount += 1
			continue
		
		len_= len(data["boxes"])

		ct = ct.update(tuple(data["boxes"][ind]), int(fps), frameNo)
		objects = ct.objects
		object_count = 0
		
		# frame = draw_boxes(ct, frame, indexes)
		# save_centroid(ct, indexes, frameCount, frameNo, fps)
		for (objectID, centroid) in objects.items():
			width = ct.bbox[objectID][0][2]
			height = ct.bbox[objectID][0][3]

			# if (in_contour >= 0) and (objectID in indexes):
			if (objectID in data["indexes"][ind-1]):
				color = (0,0,0)

				object_count += 1

				color = set_color(object_count)

				cv2.circle(frame, (centroid[0][0], centroid[0][1]), 5, color, -1)
				cv2.rectangle(frame, (int(centroid[0][0]-(width/2.0)), int(centroid[0][1]-(height/2.0))),
											(int(centroid[0][0]+(width/2.0)),	int(centroid[0][1]+(height/2.0))),
											color,1)
				cv2.putText(frame, f"ID: {objectID}: ({centroid[0][0]}, {centroid[0][1]})",
										(int(centroid[0][0]-(width/2.0))-10, int(centroid[0][1]-(height/2.0))-10),
										cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

				angle = 0
				magnitude = 0
				dir_ = ""
				centroid_ind = 4
				# Get projected direction of object
				if len(centroid) < 5:
					centroid_ind = len(centroid)-1
				angle, magnitude = get_angle_n_magnitude(centroid[0], centroid[centroid_ind], width, height)
				cv2.line(frame, (centroid[0][0],centroid[0][1]), (centroid[centroid_ind][0],centroid[centroid_ind][1]), color, 2)
				
				dir_ = get_direction(angle)
			
		frameNo += 1
		frameCount += 1
		# cv2.imshow("Image",frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):  # press q to quit
			print("Keyboard Interrupt")
			break
		output_vid.write(frame)
		if (frameCount%fps)==0: 
			print(f"frameNo: {frameNo} ind: {ind} len data[boxes]: {len_}")
			ind += 1
	cap2.release()
	output_vid.release()
	cv2.destroyAllWindows()

if __name__ == "__main__":
	yolov3_tracker()
