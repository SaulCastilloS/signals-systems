import numpy as np
import matplotlib.pyplot as plt

Fs = 1000
T = 1 / Fs
t = np.arange(0, 1, T)

# Señal compuesta
f1, f2 = 50, 200
x = np.sin(2 * np.pi * f1 * t) + np.sin(2 * np.pi * f2 * t)

# FFT
N = len(x)
mitad = N // 2
X = np.fft.fft(x)
frecuencias = np.fft.fftfreq(N, T)[:mitad]
amplitud = (2/N) * np.abs(X[:mitad])

# Eliminar frecuencia menor (50 Hz)
X_filtrada = X.copy()
tolerancia = 5  # Hz
for i, f in enumerate(np.fft.fftfreq(N, T)):
    if abs(abs(f) - f1) < tolerancia:
        X_filtrada[i] = 0

x_filtrada = np.real(np.fft.ifft(X_filtrada))
amp_filtrada = (2/N) * np.abs(X_filtrada[:mitad])

# Gráficas
fig, axes = plt.subplots(3, 1, figsize=(12, 10))

axes[0].plot(t[:200], x[:200])
axes[0].set_title('Señal Original: sin(50t) + sin(200t)')
axes[0].set_xlabel('Tiempo (s)'); axes[0].grid(True)

axes[1].plot(frecuencias, amplitud)
axes[1].set_title('Espectro Original')
axes[1].set_xlabel('Frecuencia (Hz)'); axes[1].set_xlim(0, 300); axes[1].grid(True)
axes[1].axvline(x=50,  color='r', linestyle='--', label='50 Hz')
axes[1].axvline(x=200, color='g', linestyle='--', label='200 Hz')
axes[1].legend()

axes[2].plot(frecuencias, amp_filtrada, color='purple')
axes[2].set_title('Espectro tras eliminar 50 Hz (frecuencia menor)')
axes[2].set_xlabel('Frecuencia (Hz)'); axes[2].set_xlim(0, 300); axes[2].grid(True)
axes[2].axvline(x=200, color='g', linestyle='--', label='200 Hz')
axes[2].legend()

plt.tight_layout()
plt.show()  