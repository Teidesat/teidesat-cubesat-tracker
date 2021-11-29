import cv2
from pathlib import Path

INPUT = Path('../data/videos/video1.mp4')
OUTPUT = Path('../data/frames')


def store_frames(video_path, output_dir):
    """ Store frames from a video to process """
    output_subdir = output_dir / INPUT.stem
    output_subdir.mkdir(parents=True, exist_ok=False)

    vidcap = cv2.VideoCapture(str(video_path))
    count = 0
    success, image = vidcap.read()
    while success:
        cv2.imwrite(str(output_subdir / 'frame{:d}.jpg').format(count), image)
        print('\rSaved frame', count)
        success, image = vidcap.read()
        count += 1


if __name__ == '__main__':
    store_frames(INPUT, OUTPUT)
