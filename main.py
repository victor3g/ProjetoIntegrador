import numpy as np
import librosa
from scipy import signal
import threading
import os
import soundfile as sf
import time

def carregar_audio(caminho, sr=22050):
    y, sr = librosa.load(caminho, sr=sr)
    if np.max(np.abs(y)) != 0:
        y = y / np.max(np.abs(y))
    return y, sr

def calcular_correlacao(y1, y2):
    return signal.correlate(y1, y2, mode='valid', method='direct')

def encontrar_picos(correlacao, threshold):
    peaks, _ = signal.find_peaks(correlacao, height=threshold * np.max(correlacao))
    return peaks

def filtrar_picos(peaks, gap):
    valid_peaks = [peaks[0]] if len(peaks) > 0 else []
    for i in range(1, len(peaks)):
        if peaks[i] - valid_peaks[-1] > gap:
            valid_peaks.append(peaks[i])
    return valid_peaks

def converter_indices_para_tempos(indices, sr, offset):
    return (np.array(indices) + offset) / sr

def comparar_audios(audio_path1, audio_path2, sr=22050, threshold=0.8, results=None, index=None, offset=0):
    y1, sr = carregar_audio(audio_path1, sr)
    y2, _ = carregar_audio(audio_path2, sr)

    correlacao = calcular_correlacao(y1, y2)
    peaks = encontrar_picos(correlacao, threshold)
    valid_peaks = filtrar_picos(peaks, len(y2))
    tempos = converter_indices_para_tempos(valid_peaks, sr, offset)

    if results is not None and index is not None:
        results[index] = (tempos, len(valid_peaks))

def dividir_audio(audio_path, num_parts, sr=22050):
    y, sr = carregar_audio(audio_path, sr)
    duration = librosa.get_duration(y=y, sr=sr)
    part_duration = duration / num_parts

    part_paths = []
    offsets = []
    for i in range(num_parts):
        start = i * part_duration
        end = (i + 1) * part_duration if (i + 1) * part_duration < duration else duration
        y_part = y[int(start * sr):int(end * sr)]
        part_path = f"audio_part_{i}.mp3"
        sf.write(part_path, y_part, sr)
        part_paths.append(part_path)
        offsets.append(int(start * sr))

    return part_paths, offsets

def formatar_tempos(tempos):
    return [f'{tempo:.1f}' for tempo in tempos]

def main():
    audio_path1 = 'audio.mp3'
    audio_path2 = 'som.mp3'

    num_threads = 12

    part_paths, offsets = dividir_audio(audio_path1, num_threads)

    threads = []
    results = [None] * num_threads

    start_time = time.time()

    for i in range(num_threads):
        thread = threading.Thread(target=comparar_audios, args=(part_paths[i], audio_path2, 22050, 0.8, results, i, offsets[i]))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    total_times = set()
    total_count = 0

    for result in results:
        if result is not None:
            times, count = result
            for t in times:
                total_times.add(t)
            total_count += count

    total_times = sorted(total_times)
    formatted_times = formatar_tempos(total_times)

    end_time = time.time()

    print(f'O som foi reproduzido {total_count} vezes no áudio nos seguintes tempos: {formatted_times}')
    print(f'Tempo total de execução: {end_time - start_time:.2f} segundos')

    for part_path in part_paths:
        os.remove(part_path)

if __name__ == "__main__":
    main()