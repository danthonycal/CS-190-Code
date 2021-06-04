import argparse

class VidToFrameArgs(object):
	def __init__(self):
		parser = argparse.ArgumentParser(description="VidToFrame Arguments")
		parser.add_argument("--videos_path", dest="videos_path", help="Path to video folder", default="", type=str)
		parser.add_argument("--file_end", dest="file_end", help="\".MP4\" or \"8008.MP4\"", default=".MP4", type=str)
		parser.add_argument("--res_path", dest="res_path", help="Path to result", default="", type=str)

		args = parser.parse_args()
		self.videos_path = args.videos_path
		self.file_end = args.file_end
		self.res_path = args.res_path