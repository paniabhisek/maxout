# https://stackoverflow.com/a/53570674
import gzip
import numpy as np

f = gzip.open('train-images-idx3-ubyte.gz','r')

image_size = 28
num_images = 60000

f.read(16)
buf = f.read(image_size * image_size * num_images)
data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
data = data.reshape(num_images, image_size, image_size, 1)

data = data.reshape((60000, 784))
data /= 255
data_mean = data.mean(0)
data_mean = data.mean()

data_std = data.std(0)
data_std = data.std()

print(data_mean, data_std)      # 0.13066062 0.30810776
