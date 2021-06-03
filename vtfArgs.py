import argparse

class VidToFrameArgs(object):
	def __init__(self):
		parser = argparse.ArgumentParser(description="VidToFrame Arguments")
		parser.add_argument("--file_end", dest="file_end", help="\".MP4\" or \"8008.MP4\"", default=".MP4", type=str)

		args = parser.parse_args()
		self.file_end = args.file_end