import numpy as np

# paste trtexec dumpoutput below
string = ""

batch_size = 32
tag_size = 1000
arr = np.fromiter(map(float, string.split(' ')), float)
arr = arr.reshape((batch_size, tag_size))
arr_max = arr.argmax(axis=1)

with open("data/imagenet_classes.txt") as f:
    tags = [x.strip() for x in f.readlines()]
print(arr[np.arange(batch_size), arr_max])
print([tags[x] for x in arr_max])

