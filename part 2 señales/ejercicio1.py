import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.io.wavfile import write

# --- Opción A: Generar un audio sintético (si no tienes archivo .wav) ---
Fs = 44100
duracion = 2.0
t = np.linspace(0, duracion, int(Fs * duracion), endpoint=False)

# Señal con varias frecuencias (simula un audio real)
f_dom = 440   # La4 - nota musical dominante
audio = (0.6 * np.sin(2 * np.pi * f_dom * t) +
         0.3 * np.sin(2 * np.pi * 880 * t) +
         0.1 * np.sin(2 * np.pi * 1320 * t))

# Normalizar y guardar como WAV
audio_int16 = np.int16(audio / np.max(np.abs(audio)) * 32767)
write('audio_sintetico.wav', Fs, audio_int16)

# --- Cargar el archivo ---
Fs, data = wavfile.read('audio_sintetico.wav')

# Si es estéreo, tomar solo un canal
if data.ndim > 1:
    data = data[:, 0]

data = data.astype(float)

# FFT
N = len(data)
X = np.fft.fft(data)
frecuencias = np.fft.fftfreq(N, 1/Fs)[:N//2]
amplitud = (2/N) * np.abs(X[:N//2])

# Frecuencia dominante
idx_max = np.argmax(amplitud)
f_dominante = frecuencias[idx_max]

# Gráficas
fig, axes = plt.subplots(2, 1, figsize=(14, 8))

t_audio = np.arange(N) / Fs
axes[0].plot(t_audio[:2000], data[:2000])
axes[0].set_title('Señal de Audio en el Tiempo')
axes[0].set_xlabel('Tiempo (s)'); axes[0].set_ylabel('Amplitud'); axes[0].grid(True)

axes[1].plot(frecuencias, amplitud, color='steelblue')
axes[1].set_title(f'Espectro de Frecuencias FFT — Frecuencia dominante: {f_dominante:.1f} Hz')
axes[1].set_xlabel('Frecuencia (Hz)'); axes[1].set_ylabel('Amplitud')
axes[1].set_xlim(0, 5000); axes[1].grid(True)
axes[1].axvline(x=f_dominante, color='r', linestyle='--', label=f'{f_dominante:.1f} Hz')
axes[1].legend()

plt.tight_layout()
plt.show()

print(f"Frecuencia de muestreo: {Fs} Hz")
print(f"Frecuencia dominante: {f_dominante:.1f} Hz")