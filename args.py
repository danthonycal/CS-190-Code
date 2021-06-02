import argparse

class Args(object):
	def __init__(self):

		parser = argparse.ArgumentParser(description='YOLO v3 Object Tracker')
		parser.add_argument("--time_start", dest="time_start", help="Offset the start of the video in seconds", default=2, type=int)
		parser.add_argument("--duration", dest="duration", help="Duration of video capture", default=30, type=int)
		parser.add_argument("--scale", dest="scale", help="scale dimension of vid, 0 to 1", default=0.4, type=float)
		parser.add_argument("--video", dest="video", help="Video file", default="./videos/1_GP007987.MP4", type=str)
		parser.add_argument("--buffer", dest="buffer", help="max buffer size", type=int, default=32)
		parser.add_argument("--result_json", dest="res_json", help="processed result JSON from Darknet", type=str, default="./darknet-result-json/processed/1_GP007987_result_processed.json")
		parser.add_argument("--mode", dest="mode", help="-1 to process all videos, 0 to process specific video", type=int, default=0)
		
		args = parser.parse_args()
		self.time_start = args.time_start
		self.video_path = args.video
		self.duration   = args.duration
		self.scale	  = args.scale
		self.buffer	 = args.buffer
		self.res_json   = args.res_json
		self.mode 		= args.mode