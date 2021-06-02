import os
import cv2
import sys
import time
import glob
import json
import datetime
from args import Args
from yolo2 import *
from progressBar import printProgressBar

def restart_line():
	sys.stdout.write('\r')
	sys.stdout.flush()

def draw_res(frame, coords, obj):
	obj_name = "person"
	conf = obj["confidence"]*100

	color = (0,0,255) if obj["class"] == 1 else (0,255,0)
	left, top, right, bottom = bbox2points(coords)
	
	if obj["class"] == 1:
		cv2.putText(frame, "Suspicious behavior detected.",(10,10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
	
	cv2.rectangle(frame, (left, top),(right, bottom),color, 2)
	obj_id = obj["id"]
	cv2.putText(frame, f"{obj_name} {obj_id}: {conf}%",(left-10,top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

	return frame

def videoWrite():
	args = Args()
	start = args.time_start
	res_paths = None
	vid_paths = None
	
	if args.mode == -1:
		# list all video and json paths
		vid_paths = glob.glob(f"{os.getcwd()}\\videos\\*.MP4")
		res_paths = glob.glob(f"{os.getcwd()}\\darknet-result-json\\processed\\*.JSON")
	else:
		vid_paths = [args.video_path]
		res_paths = [args.res_json]

	for path_idx in range(len(res_paths)):
		print(f"Writing {res_paths[path_idx]}")
		result_name = vid_paths[path_idx][-14:-4]
		video = cv2.VideoCapture(vid_paths[path_idx])
		fps = 0

		#Get version of OpenCV
		(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

		#Get framerate of video
		fps = get_fps(video)

		video.set(cv2.CAP_PROP_POS_FRAMES,  int(start*fps))
		total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT)) - args.time_start

		result = f'{result_name}_result.MP4'
		print(f"Output: {result}")
		frame_num = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
		frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
		frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
		
		sys.stdout.write(f"Video dims: {frame_width}x{frame_height} Total frames: {frame_num}\n")
		sys.stdout.flush()

		fourcc = cv2.VideoWriter_fourcc(*'avc1')
		output_vid = cv2.VideoWriter(result,fourcc,fps,(frame_width,frame_height))
		
		frame_no = int(start*fps)
		frame_count = 0
		# contour_points=[[int(0.13*frame_width),int(0.77*frame_height)], [int(0.64*frame_width),int(0.94*frame_height)], [int(0.67*frame_width),int(0.37*frame_height)], [int(0.67*frame_width),int(0.31*frame_height)], [int(0.48*frame_width),int(0.31*frame_height)], [int(0.09*frame_width),int(0.62*frame_height)]]
		# ctr = np.array(contour_points).reshape((-1,1,2)).astype(np.int32)
		with open(res_paths[path_idx], 'r') as f:
			results_json = json.load(f)
		
			while True:
				_, frame = video.read()
		
				if not _:
					break
				# cv2.drawContours(frame, [ctr], 0, (255, 0, 0), 2)
				# print(f"frame_count: {frame_count-1} len res: {len(results_json)}")
				try:
					for obj in results_json[frame_count-1]["objects"]:
						if (frame_count - 1) == len(results_json):
							break

						coords = [
							int(obj["center_x"]*frame_width),
							int(obj["center_y"]*frame_height),
							int(obj["width"]*frame_width),
							int(obj["height"]*frame_height)
						]

						frame = draw_res(frame, coords, obj)
				except:
					continue
		
				sys.stdout.write(f"{int(frame_count)}/{frame_num//fps} frames")
				sys.stdout.flush()
				restart_line()
		
				if (frame_no%fps==0):
					frame_count += 1
		
				output_vid.write(frame)
				frame_no += 1
		
		video.release()
		output_vid.release()
		cv2.destroyAllWindows()

		print("Completed.")


if __name__ == "__main__":
	videoWrite()	
