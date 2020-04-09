import numpy as np
from skimage.color import rgb2gray
from imageio import imread
import scipy.io.wavfile as wav
from scipy.signal import convolve2d
from scipy import signal
from scipy.ndimage.interpolation import map_coordinates


####################################################################################################################

def read_image(filename, representation):
    im = imread(filename)
    if (im.dtype == np.uint8 or im.dtype == int or im.np.matrix.max > 1):
        im = im.astype(np.float64) / 255

    if ((representation == 1) and (len(im.shape) >= 3)):  # process image into gray scale

        im = rgb2gray(im)

    return im


## SECTION FROM HELPER:

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


####################################################################################################################
## SECTION 1: ##

# -----------------1.1-----------------
#  Discrete Fourier Transform:
def DFT(signal):
    N = signal.shape[0]
    dft_matrix = (np.exp(-2j * np.pi / N * np.arange(N)).reshape(-1, 1)) ** np.arange(N)
    return np.dot(dft_matrix, signal)


# Inverse DFT
def IDFT(fourier_signal):
    N = fourier_signal.shape[0]
    dft_matrix = (np.exp(-2j * np.pi / N * np.arange(N)).reshape(-1, 1)) ** np.arange(N)
    inverse_dft_matrix = (dft_matrix ** (-1)) / N
    return np.real_if_close(np.dot(inverse_dft_matrix, fourier_signal) )


# -----------------1.2-----------------

def DFT2(image):
    dft_on_row = DFT(image)
    dft_on_col = DFT(dft_on_row.T)
    return dft_on_col.T


def IDFT2(fourier_image):
    idft_on_row = IDFT(fourier_image)
    idft_on_col = IDFT(idft_on_row.T)
    return idft_on_col.T


####################################################################################################################
## SECTION 2 : ##

# -----------------2.1-----------------
def change_rate(filename, ratio):
    rate, data = wav.read(filename)
    wav.write('change_rate.wav', (int)(rate * ratio), data)


# -----------------2.2-----------------

def change_samples(filename, ratio):
    rate, data = wav.read(filename)
    if (ratio != 1):
        new_data = resize(data, ratio)  # manipulate data and reduce samples by frequencies :
    else:  # ratio is 1 so no need to manipulate data
        new_data = data
    wav.write('change_samples.wav', rate, new_data)
    return new_data


def resize(data, ratio):
    new_data = np.fft.fftshift(DFT(data))
    N = data.shape[0]
    # now we change high frequencies according to ratio :

    if (ratio < 1):  # we would like to pad with zeroes:
        samples_to_pad = int(N * ((1 / ratio) - 1) / 2)
        new_data = np.pad(new_data, (samples_to_pad,), 'constant')

    if (ratio > 1):  # we would like to trim high frequencies
        samples_after_trim = int((N * (1 - (1 / ratio)) / 2) + 1)
        new_data = new_data[samples_after_trim:-samples_after_trim]

    new_data = np.fft.ifftshift((new_data))  # shifting back after manipulation
    new_data = IDFT(new_data).astype(data.dtype)  # returning to same data type as the original data
    return new_data


# -----------------2.3-----------------


def resize_spectrogram(data, ratio):
    # computing the spectrogram:
    spectograma = stft(data)
    # changing the number of spectrogram columns:
    spectograma = np.apply_along_axis(resize, 1, spectograma, ratio)
    # creating back the audio
    creating_back = istft(spectograma)
    return creating_back.astype(data.dtype)  # returning to same data type as the original data


# -----------------2.4-----------------
def resize_vocoder(data, ratio):
    spectrograma = phase_vocoder(stft(data), ratio)
    return (istft(spectrograma)).astype(data.dtype)


####################################################################################################################
## SECTION 3 : ##


def conv_der(im):
    convolute_with = np.array([[0.5, 0, -0.5]])
    dx = convolve2d(im, convolute_with, 'same')
    dy = convolve2d(im, convolute_with.T, 'same')
    magnitude = np.sqrt(np.abs(dx) ** 2 + np.abs(dy) ** 2)
    return magnitude


def fourier_der(im):
    rows_number = im.shape[0]
    cols_number = im.shape[1]
    im_fourier = np.fft.fftshift(DFT2(im))

    ##### To compute the x derivative of f: #####
    # 1.Compute the Fourier Transform F
    # 2. Multiply Fourier coefficient F(u,v) by 2𝜋𝑖u/𝑁
    # 3. Compute the Inverse Fourier Transform
    u = np.tile(np.arange(-cols_number / 2, cols_number / 2), (rows_number, 1))
    dx = np.fft.ifftshift(im_fourier * u * (2 * np.pi * 1j) / cols_number)
    image_dx = IDFT2(dx)
    #############################################

    ##### To compute the y derivative of f: #####
    # 1.Compute the Fourier Transform F
    # 2. Multiply Fourier coefficient F(u,v) by 2𝜋𝑖v/𝑁
    # 3. Compute the Inverse Fourier Transform
    v = np.tile(np.arange(-rows_number / 2, rows_number / 2), (cols_number, 1)).T
    dy = np.fft.ifftshift(im_fourier * v * (2 * np.pi * 1j) / rows_number)
    image_dy = IDFT2(dy)
    #############################################

    magnitude = np.sqrt(np.abs(image_dx) ** 2 + np.abs(image_dy) ** 2)
    return magnitude


