import numpy as np #Исправляет артефакты( учебная программа)
import cv2
from matplotlib import pyplot as plt
input_image = cv2.imread('Input.png',0)
print('Data share and type:',input_image.shape,input_image.dtype)
# plt.figure(figsize=(10,10))
# plt.imshow(input_image, cmap = 'gray')
# plt.show()
float_input_image = np.float32(input_image)
# print('Data share and type:',float_input_image.shape,float_input_image.dtype)
# plt.figure(figsize=(10,10))
# plt.imshow(float_input_image)
# plt.show()
fft_input_image = np.fft.fft2(float_input_image)
# print('Data share and type:',fft_input_image.shape, fft_input_image.dtype)
magnitude = np.log(np.abs(fft_input_image))

# bin_counts, bin_edges, patches = plt.hist(magnitude.ravel())
fft_input_image = np.fft.fft2(float_input_image)
fft_input_image = np.fft.fftshift(fft_input_image)
# print('Data shape and type:', fft_input_image.shape, fft_input_image.dtype)

magnitude = np.abs(fft_input_image)
s_min = magnitude.min()
s_max = magnitude.max()
# print(s_min,s_max)

# plt.figure(figsize=(10, 10))
# plt.imshow(np.log(magnitude))
# plt.show()

reconstructed_image_complex = np.fft.ifft2(fft_input_image)
reconstructed_image = np.abs(reconstructed_image_complex)

# plt.figure(figsize=(10, 10))
# plt.imshow(reconstructed_image)
# plt.show()

fft_input_image_processed = fft_input_image.copy()
fft_input_image_processed[335:350, 0:470] = 0
fft_input_image_processed[335:350, 570:1000] = 0
fft_input_image_processed[0:300, 505:520] = 0
fft_input_image_processed[400:670, 505:520] = 0

fft_input_image_processed[300:340, 520:540] = 0
fft_input_image_processed[360:380, 480:500] = 0
magnitude = np.abs(fft_input_image_processed)
s_max = magnitude.min()
s_min = magnitude.max()
# print(s_min, s_max)

# plt.figure(figsize=(10, 10))
# plt.imshow(np.log(magnitude))
# plt.show()
# Выводим нормальное изображение
reconstructed_image_complex = np.fft.ifft2(fft_input_image_processed)
reconstructed_image = np.abs(reconstructed_image_complex)
reconstructed_image_complex = np.fft.ifft2(fft_input_image_processed)
reconstructed_image = np.abs(reconstructed_image_complex)

plt.figure(figsize=(10,10))
plt.imshow(input_image, cmap = 'gray')

plt.figure(figsize=(10, 10))
plt.imshow(reconstructed_image, cmap = 'gray')
plt.show()