import numpy as np
import librosa
from scipy import signal

def compare_sounds(audio_path1, audio_path2, sr=22050, threshold=0.8):

    y1, sr = librosa.load(audio_path1, sr=sr)
    y2, _ = librosa.load(audio_path2, sr=sr)

    y1 = y1 / np.max(np.abs(y1))
    y2 = y2 / np.max(np.abs(y2))

    correlation = signal.correlate(y1, y2, mode='valid')

    peaks, properties = signal.find_peaks(correlation, height=threshold*np.max(correlation))

    gap = len(y2) 
    valid_peaks = [peaks[0]] if len(peaks) > 0 else []

    for i in range(1, len(peaks)):
        if peaks[i] - valid_peaks[-1] > gap:
            valid_peaks.append(peaks[i])

    times = np.array(valid_peaks) / sr

    return times, len(valid_peaks)

audio_path1 = 'audio5.mp3' # arquivo de áudio
audio_path2 = 'som.mp3'
times, count = compare_sounds(audio_path1, audio_path2)

formatted_times = [f'{time:.1f}' for time in times]

print(f'A onda sonora do segundo áudio foi reproduzida {count} vezes no primeiro áudio nos seguintes tempos: {formatted_times}')