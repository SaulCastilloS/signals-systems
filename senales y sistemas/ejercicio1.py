import numpy as np
import matplotlib.pyplot as plt

# Parámetros
Fs = 1000          # Frecuencia de muestreo (Hz)
T = 1 / Fs         # Período de muestreo
t = np.arange(0, 1, T)  # Vector de tiempo (1 segundo)

# Señal
f0 = 40
x = np.sin(2 * np.pi * f0 * t)

# FFT
N = len(x)
X = np.fft.fft(x)
frecuencias = np.fft.fftfreq(N, T)

# Solo frecuencias positivas
mitad = N // 2
frecuencias_pos = frecuencias[:mitad]
amplitud_pos = (2 / N) * np.abs(X[:mitad])

# Gráficas
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

axes[0].plot(t[:200], x[:200])
axes[0].set_title('Señal en el Tiempo: x = sin(2π·40·t)')
axes[0].set_xlabel('Tiempo (s)')
axes[0].set_ylabel('Amplitud')
axes[0].grid(True)

axes[1].plot(frecuencias_pos, amplitud_pos)
axes[1].set_title('Espectro de Frecuencias (FFT)')
axes[1].set_xlabel('Frecuencia (Hz)')
axes[1].set_ylabel('Amplitud')
axes[1].set_xlim(0, 200)
axes[1].grid(True)

# Marcar frecuencia dominante
idx_dom = np.argmax(amplitud_pos)
freq_dom = frecuencias_pos[idx_dom]
axes[1].axvline(x=freq_dom, color='r', linestyle='--', label=f'f dominante = {freq_dom:.1f} Hz')
axes[1].legend()

plt.tight_layout()
plt.show()

print(f"Frecuencia dominante: {freq_dom:.1f} Hz")