import numpy as np
from skimage.color import rgb2gray
import imageio
import scipy.io.wavfile as sp
from scipy.signal import convolve2d
from scipy import signal
from scipy.ndimage.interpolation import map_coordinates

NUM_OF_PIXELS = 256
RGB_IMAGE_SHAPE_SIZE = 3
FOURIER_CONST = 2j * np.pi
CHANGE_RATE_FILE = 'change_rate.wav'
CHANGE_SAMPLES_FILE = 'change_samples.wav'
DIFF_ARR = np.array([0.5, 0, -0.5]).reshape(3, 1)

def read_image(filename, representation):
    """
    this function receives a name of an image file, if the image is colored turns it to grayscale if not
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
    """
    this function gets a signal with shape of 1d and calculates the dft of it
    return the dft of the image with type complex128
    :param signal: the data
    :return: the dft of the signal
    """
    n = signal.shape[0]
    signal = signal.T
    size = np.arange(n)
    u = np.meshgrid(size, size)[1]
    exponent_signal = np.exp(-(FOURIER_CONST * u * size / n))
    exponent_signal *= signal
    fourier = np.sum(exponent_signal, axis=1)
    if len(signal.shape) == 2:
        return fourier.reshape(n,1)
    else:
        return fourier


def IDFT(fourier_signal):
    """
    this function gets a fourier signal with shape of 1d and calculates the inverse dft of it
    return the inverse dft of the fourier signal with type complex128
    :param fourier_signal: the data
    :return: the inverse dft of the fourier signal
    """
    n = fourier_signal.shape[0]
    fourier_signal = fourier_signal.T
    size = np.arange(n)
    u = np.meshgrid(size, size)[1]
    exponent_signal = np.exp(FOURIER_CONST * u * size / n)
    exponent_signal *= fourier_signal
    signal = np.sum(exponent_signal, axis=1) / n
    if len(fourier_signal.shape) == 2:
        return signal.reshape(n,1)
    else:
        return signal


def DFT2(image):
    """
    this function gets an image with shape of 2d and calculates the 2d dft of it
    return the 2d dft of the image with type complex128
    :param image: the data
    :return: the 2d dft of the image
    """
    return two_d_fourier(image, DFT)


def IDFT2(fourier_image):
    """
    this function gets a fourier image with shape of 2d and calculates the 2d inverse dft of it
    return the 2d inverse dft of the image with type complex128
    :param fourier_image: the data
    :return: the 2d inverse dft of the image
    """
    return two_d_fourier(fourier_image, IDFT)


def two_d_fourier(image, func):
    """
    a helper func for the dft2 and idft2 gets a func which is dft or idft and executes it on every row and column
    return the result of either the 2d dft or 2d idft
    :param image: the data
    :param func: the function to execute on the data
    :return: the 2d inverse dft of the image or 2d dft
    """
    result = np.zeros(image.shape, dtype=np.complex128)
    for i in range(image.shape[0]):
        result[i] = func(image[i])
    for i in range(image.shape[1]):
        result[0:image.shape[0], i] = func(result[0:image.shape[0], i])
    return result


def change_rate(filename, ratio):
    """
    the function multiplies the rate by the ration and saves the result in a file
    :param filename: the name of the wav file
    :param ratio: the ratio to change the wav file
    """
    rate, data = sp.read(filename)
    rate = int(rate * ratio)
    sp.write(CHANGE_RATE_FILE, rate, data)


def change_samples(filename, ratio):
    """
    the function changes the data of the wav file to fit the new size of the data
    :param filename: the name of the wav file
    :param ratio: the ratio to change the wav file
    :return the new data with the new size
    """
    rate, data = sp.read(filename)
    data = resize(data, ratio)
    sp.write(CHANGE_SAMPLES_FILE, rate, data)
    return data


def resize(data, ratio):
    """
    the function changes the data of the wav file to fit the new size of the data and returns it
    :param data: the data
    :param ratio: the ratio to change the wav file
    :return the new data with the new size
    """
    new_size = int(np.floor(len(data) / ratio))
    fourier_data = np.fft.fftshift(DFT(data))
    if new_size > len(data):  # if new size is larger that the old we pad with zeros
        to_add = new_size - len(data)
        left_size = to_add // 2
        right_size = to_add - left_size
        new_data = np.zeros(left_size)
        new_data = np.append(new_data, fourier_data)
        new_data = np.append(new_data, np.zeros(right_size))
        return np.real(IDFT(np.fft.ifftshift(new_data)))
    to_del = len(data) - new_size  # if not we delete half from each side
    left_index = to_del // 2
    right_index = to_del - left_index
    new_data = fourier_data[left_index: len(fourier_data) - right_index]
    return np.real(IDFT(np.fft.ifftshift(new_data)))


def resize_spectrogram(data, ratio):
    """
    the function changes the data of the wav file to fit the new size of the data using a spectogram
    :param data: the data
    :param ratio: the ratio to change the wav file
    :return the new data with the new size
    """
    spectogram = stft(data)  # gets the spectogram
    new_size = int(np.floor(spectogram.shape[1] / ratio))  # calculate the new data size
    new_spectogram = np.zeros((spectogram.shape[0], new_size))
    for row in range(len(spectogram)):
        new_spectogram[row] = resize(spectogram[row], ratio).T
    return istft(new_spectogram)


def resize_vocoder(data, ratio):
    """
    the function changes the data of the wav file to fit the new size of the data using a spectogram and phase clean
    :param data: the data
    :param ratio: the ratio to change the wav file
    :return the new data with the new size
    """
    spectogram = stft(data)
    return istft(phase_vocoder(spectogram, ratio))


def conv_der(im):
    """
    the function calculates the magnitude of the image using convolution derivatives
    :param im: the image
    :return the magnitude derivative of the image
    """
    diff_x = convolve2d(im, DIFF_ARR, mode="same", boundary="symm")  # calculates diff_x using the kernel DIFF_ARR
    diff_y = convolve2d(im, DIFF_ARR.transpose(), mode="same", boundary="symm")# calculates diff_y using the kernel
    return calc_magnitude(diff_x, diff_y)


def fourier_der(im):
    """
    the function calculates the magnitude of the image using fourier derivatives
    :param im: the image
    :return the magnitude derivative of the image
    """
    fourier = DFT2(im)
    diff_x = get_der(fourier)
    trans_fourier = fourier.transpose()
    diff_y = get_der(trans_fourier).transpose()
    return calc_magnitude(diff_x, diff_y)


def get_der(fourier):
    """
    calculates the diff by multiplying by index and idft2 the result
    :param fourier: the data
    :return the diff of the fourier signal
    """
    n = fourier.shape[0]
    indexes = np.fft.ifftshift(np.arange(n) - n // 2).reshape(n, 1)
    diff = IDFT2(indexes * fourier)
    diff *= FOURIER_CONST / n
    return diff


def calc_magnitude(diff_x, diff_y):
    """
    calculates the magnitude using the formula
    :param diff_x: the derivative in x axis
    :param diff_y: the derivative in y axis
    :return the magnitude
    """
    return np.sqrt(np.abs(diff_x) ** 2 + np.abs(diff_y) ** 2)


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
    num_timesteps = int(spec.shape[1] / ratio)
    time_steps = np.arange(num_timesteps) * ratio

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
