import cv2
import os

import argparse


def get_parser():
	parser = argparse.ArgumentParser(description="convert a video to frames")
	parser.add_argument(
		"--input",
		help="directory of input video frames",
		required=True,
	)
	parser.add_argument(
		"--output",
		help="directory to save output frames",
		required=True,
	)
	return parser


if __name__ == '__main__':
    args = get_parser().parse_args()

    video_file = os.path.basename(args.input)
    video_name = video_file.split('.')[0]
    frame_folder = os.path.join(args.output, video_name + "_f2")

    if not os.path.exists(frame_folder):
        os.makedirs(frame_folder)

    videoCapture = cv2.VideoCapture()
    videoCapture.open(args.input)

    fps = videoCapture.get(cv2.CAP_PROP_FPS)
    num_frames = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
    print("fps=", int(fps), " frames=", int(num_frames))

    for i in range(int(num_frames)):
        ret, frame = videoCapture.read()
        if i % 2 == 0:
            print("convert " + os.path.join(frame_folder, "img_" + f"{i}".zfill(5) + ".jpg"))
            cv2.imwrite(os.path.join(frame_folder, "img_" + f"{i}".zfill(5) + ".jpg"), frame)

    print("Convert Successfully.")
