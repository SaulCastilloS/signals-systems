import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.io.wavfile import write

# Cargar/generar señal
Fs = 44100
t = np.linspace(0, 2.0, int(Fs * 2.0), endpoint=False)
audio_limpio = 0.6 * np.sin(2 * np.pi * 440 * t) + 0.3 * np.sin(2 * np.pi * 880 * t)

# Añadir ruido gaussiano
nivel_ruido = 0.3
ruido = nivel_ruido * np.random.randn(len(audio_limpio))
audio_ruidoso = audio_limpio + ruido

# FFT de ambos
N = len(audio_limpio)
mitad = N // 2
frecuencias = np.fft.fftfreq(N, 1/Fs)[:mitad]

amp_limpio  = (2/N) * np.abs(np.fft.fft(audio_limpio)[:mitad])
amp_ruidoso = (2/N) * np.abs(np.fft.fft(audio_ruidoso)[:mitad])

# Gráficas
fig, axes = plt.subplots(2, 2, figsize=(15, 9))

axes[0,0].plot(t[:1000], audio_limpio[:1000])
axes[0,0].set_title('Audio Limpio'); axes[0,0].grid(True)

axes[0,1].plot(t[:1000], audio_ruidoso[:1000], color='tomato')
axes[0,1].set_title(f'Audio con Ruido (nivel={nivel_ruido})'); axes[0,1].grid(True)

axes[1,0].plot(frecuencias, amp_limpio, color='steelblue')
axes[1,0].set_title('Espectro Limpio')
axes[1,0].set_xlim(0, 3000); axes[1,0].grid(True)

axes[1,1].plot(frecuencias, amp_ruidoso, color='tomato', alpha=0.7)
axes[1,1].set_title('Espectro con Ruido')
axes[1,1].set_xlim(0, 3000); axes[1,1].grid(True)

plt.tight_layout()
plt.show()