from progressBar import printProgressBar
from vtfArgs import VidToFrameArgs
from yolo2 import *
import os
import cv2
import math
import glob
import numpy as np
import matplotlib.pyplot as plt

def get_vid_paths():
	paths = []
	path = args.videos_path
	for subdir, dirs, files in os.walk(path):
		for filename in files:
			filepath = subdir + os.sep + filename
			if filepath.endswith(args.file_end):
				paths.append(filepath)
	
	return paths

def vidToFrame(vid_file, count, success):
	file_loc = []
	vid_id = vid_file[-14:-4]
	vid_frame_folder = f"{vid_id}-frames"

	if not os.path.exists(f"{os.getcwd()}/frames/{vid_frame_folder}"):
		os.makedirs(f"{os.getcwd()}/frames/{vid_frame_folder}")

	vid_cap = cv2.VideoCapture(vid_file)
	vid_cap.set(cv2.CAP_PROP_POS_FRAMES, 160)
	fps = get_fps(vid_cap)

	print("", end="\n")
	print(vid_file[-14:-4])
	print(f"FPS: {fps}")
	
	total_frames = int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT)) 
	height = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
	width = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	total_files = (total_frames-160)//fps

	printProgressBar(0, total_files, prefix='Progress:', suffix='Complete', length=50)
	i = 0
	while success:
		if (count%fps!=0):
			success, frame = vid_cap.read()
			count += 1
			continue
		success, frame = vid_cap.read()
		output_path = f"/frames/{vid_frame_folder}/{vid_id}-{count}.png"
		save_path = f"/frames/{vid_frame_folder}/{vid_id}-{count}.png\n"
		file_loc.append(save_path)
		plt.imsave(f"{os.getcwd()}/{output_path}", frame, cmap=plt.cm.gray)
		printProgressBar(i + 1, total_files, prefix='Progress:', suffix='Complete', length=50)
		i+=1
		count += 1
	vid_cap.release()
	file_name = f"{vid_frame_folder}-list.txt"

	with open(file_name, 'w') as f:
		for i in range(len(file_loc)):
			f.write(file_loc[i])

def processVidToFrame():
	count = 0
	success = True
	frame_name = "frame"
	vid_paths = get_vid_paths()

	if not os.path.exists(f'{args.res_path}/frames'):
		os.makedirs(f'{args.res_path}/frames')

	print(vid_paths)
	for vid_path in vid_paths:
		vidToFrame(vid_path, count, success)

if __name__ == "__main__":
	global args
	args = VidToFrameArgs()
	processVidToFrame()
