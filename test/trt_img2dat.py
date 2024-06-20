import argparse
from pathlib import Path

parser = argparse.ArgumentParser(
        description='image to numpy .dat',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-i', '--image', default="data/drum.jpg", help='image file')
parser.add_argument('-s', '--size', default=224, type=int, help='target size (H and W)')
parser.add_argument('-o', '--output_dir', default="tmp/", help='target size (H and W)')
args = parser.parse_args()

image_path = args.image

from PIL import Image
import numpy as np

# prepare the input data
SIZE = args.size

img = Image.open(image_path).resize((SIZE, SIZE))

input_data = np.array(img).transpose(2, 0, 1).astype(np.float32) / 255.0
# input_data -= np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
# input_data /= np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
input_data = input_data[np.newaxis, ...]

input_array = np.zeros((1, 3, SIZE, SIZE), dtype=np.float32)
input_array[:] = input_data

filename = Path(args.output_dir) / Path(Path(image_path).stem + '-%d.dat' % SIZE)
print("write to", filename)
input_array.tofile(filename)
