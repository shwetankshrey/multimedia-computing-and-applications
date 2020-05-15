import librosa
import matplotlib.pyplot as plt
import numpy as np

def get_fourier_coefficients(samples):
    fourier_coefficients = []
    num_samples = len(samples)
    for i in range(int(num_samples/2)):
        ks = np.arange(0, num_samples, 1)
        coefficient = np.abs(np.sum(samples * np.exp((2j * np.pi * ks * i) / num_samples)) / num_samples) * 2
        fourier_coefficients.append(coefficient)
    return fourier_coefficients

def create_spectrogram(samples, sampling_rate, window_length_ms, overlap, show_spectrogram_plot):
    window_size = int(sampling_rate * window_length_ms / 1000)
    step_size = int(window_size * (1 - overlap))
    start_of_steps = np.arange(0, len(samples), step_size, dtype=int)
    start_of_steps = start_of_steps[start_of_steps + window_size < len(samples)]
    spectrogram = []
    for start_of_step in start_of_steps:
        step = samples[start_of_step:start_of_step + window_size]
        step_frequency_domain = get_fourier_coefficients(step)
        spectrogram.append(step_frequency_domain)
    spectrogram = np.array(spectrogram).T
    assert spectrogram.shape[1] == len(start_of_steps) 
    if show_spectrogram_plot:
        plt.figure(figsize=(15,5))
        plt.plot(np.linspace(0, len(samples) / sampling_rate, num=len(samples)), samples)
        plt.imshow(spectrogram, aspect='auto', origin='lower')
        plt.show()
    return spectrogram

def get_spectrogram(audio_path, sampling_rate = None, window_length_ms = 20, overlap = 0.5, show_spectrogram_plot = False):
    samples, sampling_rate = librosa.load(audio_path, sr=sampling_rate)
    if len(samples) < sampling_rate:
        samples = np.append(samples, np.ones(sampling_rate - len(samples)) * 1e-6)
    elif len(samples) > sampling_rate:
        samples = samples[:sampling_rate]
    return create_spectrogram(samples, sampling_rate, window_length_ms, overlap, show_spectrogram_plot)

def get_noise_augmented_spectrogram(audio_path, noise_path, sampling_rate = None, window_length_ms = 20, overlap = 0.5, show_spectrogram_plot = False):
    samples, sampling_rate = librosa.load(audio_path, sr=sampling_rate)
    noise_samples, noise_sampling_rate = librosa.load(noise_path, sr=sampling_rate)
    assert noise_sampling_rate == sampling_rate
    noise_beginning = np.random.randint(0, len(noise_samples) - len(samples))
    noise_samples = noise_samples[noise_beginning : (noise_beginning + len(samples))]
    alpha = 0.05
    samples = samples * (1 - alpha) + noise_samples * alpha
    if len(samples) < sampling_rate:
        samples = np.append(samples, np.ones(sampling_rate - len(samples)) * 1e-6)
    elif len(samples) > sampling_rate:
        samples = samples[:sampling_rate]
    return create_spectrogram(samples, sampling_rate, window_length_ms, overlap, show_spectrogram_plot)