import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import linalg
from skimage.color import rgb2gray
import imageio
import scipy.io.wavfile as sp
import sys

NUM_OF_PIXELS = 256
START_Z_LEVEL = -1
YIQ_MATRIX = np.array([[0.299, 0.587, 0.114], [0.596, -0.275, -0.321], [0.212, -0.523, 0.311]])
RGB_IMAGE_SHAPE_SIZE = 3
x = np.hstack([np.repeat(np.arange(0,50,2),10)[None,:], np.array([255]*6)[None,:]])
grad = np.tile(x,(256,1))/255
np.set_printoptions(threshold=sys.maxsize)

FOURIER_CONST = 2j * np.pi
CHANGE_RATE_FILE = 'change_rate.wav'
CHANGE_SAMPLES_FILE = 'change_samples.wav'

def read_image(filename, representation):
    """
    this function recieves a name of an image file, if the image is colored turns it to grayscale if not
    return the image with type float64
    :param filename: the name of the file
    :param representation: if 1 returns and image is colored returns a grayscale image if 2 returns the colored image
    :return: the imange type float 64
    """
    try:
        im = imageio.imread(filename)
    except IOError:
        return
    if representation != 1 and representation != 2:
        return
    if len(im.shape) == RGB_IMAGE_SHAPE_SIZE and representation == 1:
        img_gray = rgb2gray(im)
        return img_gray
    else:
        return im / (NUM_OF_PIXELS - 1)


def DFT(signal):
    n, = signal.shape
    size = np.arange(n)
    u = np.meshgrid(size, size)[1]
    exponent_signal = np.exp(-(FOURIER_CONST * u * size / n))
    exponent_signal *= signal
    fourier = np.sum(exponent_signal, axis=1)
    return fourier


def IDFT(fourier_signal):
    n, = fourier_signal.shape
    size = np.arange(n)
    u = np.meshgrid(size, size)[1]
    exponent_signal = np.exp(FOURIER_CONST * u * size / n)
    exponent_signal *= fourier_signal
    signal = np.sum(exponent_signal, axis=1) / n
    # return np.real(signal)
    return signal


def DFT2(image):
    return two_d_fourier(image, DFT)


def IDFT2(fourier_image):
    return two_d_fourier(fourier_image, IDFT)


def two_d_fourier(image, func):
    dft = np.zeros(image.shape, dtype=np.complex128)
    for i in range(image.shape[0]):
        dft[i] = func(image[i])
    for i in range(image.shape[1]):
        dft[0:image.shape[0], i] = func(dft[0:image.shape[0], i])
    return dft


def change_rate(filename, ratio):
    rate, data = sp.read(filename)
    rate = int(rate * ratio)
    sp.write(CHANGE_RATE_FILE, rate, data)


def change_samples(filename, ratio):
    rate, data = sp.read(filename)
    data = resize(data, ratio)
    sp.write(CHANGE_SAMPLES_FILE, rate, data.astype(np.int16))


def resize(data, ratio):
    newSize = int(np.floor(len(data) / ratio))
    fourier_data = np.fft.fftshift(DFT(data))
    if newSize > len(data):
        to_add = newSize - len(data)
        left_size = to_add // 2
        right_size = to_add - left_size
        new_data = np.zeros(left_size)
        new_data = np.append(new_data, fourier_data)
        new_data = np.append(new_data, np.zeros(right_size))
        return np.real(IDFT(np.fft.ifftshift(new_data)))
    new_data = np.array([newSize])
    to_del = len(data) - newSize
    left_index = to_del // 2
    right_index = to_del - left_index
    new_data = np.append(new_data, fourier_data[left_index: len(fourier_data) - (right_index + 1)])
    return np.real(IDFT(np.fft.ifftshift(new_data)))


from scipy import signal
from scipy.ndimage.interpolation import map_coordinates


def stft(y, win_length=640, hop_length=160):
    fft_window = signal.windows.hann(win_length, False)

    # Window the time series.
    n_frames = 1 + (len(y) - win_length) // hop_length
    frames = [y[s:s + win_length] for s in np.arange(n_frames) * hop_length]

    stft_matrix = np.fft.fft(fft_window * frames, axis=1)
    return stft_matrix.T


def istft(stft_matrix, win_length=640, hop_length=160):
    n_frames = stft_matrix.shape[1]
    y_rec = np.zeros(win_length + hop_length * (n_frames - 1), dtype=np.float)
    ifft_window_sum = np.zeros_like(y_rec)

    ifft_window = signal.windows.hann(win_length, False)[:, np.newaxis]
    win_sq = ifft_window.squeeze() ** 2

    # invert the block and apply the window function
    ytmp = ifft_window * np.fft.ifft(stft_matrix, axis=0).real

    for frame in range(n_frames):
        frame_start = frame * hop_length
        frame_end = frame_start + win_length
        y_rec[frame_start: frame_end] += ytmp[:, frame]
        ifft_window_sum[frame_start: frame_end] += win_sq

    # Normalize by sum of squared window
    y_rec[ifft_window_sum > 0] /= ifft_window_sum[ifft_window_sum > 0]
    return y_rec


def phase_vocoder(spec, ratio):
    time_steps = np.arange(spec.shape[1]) * ratio
    time_steps = time_steps[time_steps < spec.shape[1]]

    # interpolate magnitude
    yy = np.meshgrid(np.arange(time_steps.size), np.arange(spec.shape[0]))[1]
    xx = np.zeros_like(yy)
    coordiantes = [yy, time_steps + xx]
    warped_spec = map_coordinates(np.abs(spec), coordiantes, mode='reflect', order=1).astype(np.complex)

    # phase vocoder
    # Phase accumulator; initialize to the first sample
    spec_angle = np.pad(np.angle(spec), [(0, 0), (0, 1)], mode='constant')
    phase_acc = spec_angle[:, 0]

    for (t, step) in enumerate(np.floor(time_steps).astype(np.int)):
        # Store to output array
        warped_spec[:, t] *= np.exp(1j * phase_acc)

        # Compute phase advance
        dphase = (spec_angle[:, step + 1] - spec_angle[:, step])

        # Wrap to -pi:pi range
        dphase = np.mod(dphase - np.pi, 2 * np.pi) - np.pi

        # Accumulate phase
        phase_acc += dphase

    return warped_spec

# im = np.exp(2 * np.pi * np.arange(8) / 8)
# im = np.mgrid[:5, :5][0]
# im = read_image('externals/monkey.jpg', 1)
# im = grad
# fft = np.fft.fft2(im)
# print(fft)
# dft = DFT2(im)
# print("------------------------------------------------")
# print(dft)
# print(np.allclose(fft, dft))
# ifft = np.fft.ifft2(fft)
# print(ifft)
# print("-------------------------------------------")
# idft = IDFT2(dft)
# print(idft)
# print(np.allclose(ifft, idft))
change_rate('3.wav',2)
change_samples('3.wav',2)

# im = imageio.imread('externals/monkey.jpg')/255
# plt.imshow(im, cmap="gray")
# plt.show()
# DFT()
# plt.imshow(im_eq, cmap="gray")
# plt.show()
# plt.figure()
# plt.plot(hist.cumsum())
# plt.show()
# plt.figure()
# plt.plot(new_hist.cumsum())
# plt.show()