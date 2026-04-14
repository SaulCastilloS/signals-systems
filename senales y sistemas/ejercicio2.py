import numpy as np
import matplotlib.pyplot as plt

Fs = 1000
T = 1 / Fs
t = np.arange(0, 1, T)

# Señal limpia y con ruido
f0 = 60
x_limpia = np.sin(2 * np.pi * f0 * t)
ruido = 0.5 * np.random.randn(len(t))
x_ruidosa = x_limpia + ruido

# FFT de ambas
N = len(t)
mitad = N // 2

X_limpia = np.fft.fft(x_limpia)
X_ruidosa = np.fft.fft(x_ruidosa)
frecuencias = np.fft.fftfreq(N, T)[:mitad]

amp_limpia  = (2/N) * np.abs(X_limpia[:mitad])
amp_ruidosa = (2/N) * np.abs(X_ruidosa[:mitad])

# Gráficas
fig, axes = plt.subplots(2, 2, figsize=(14, 8))

axes[0,0].plot(t[:300], x_limpia[:300])
axes[0,0].set_title('Señal Limpia (tiempo)')
axes[0,0].set_xlabel('Tiempo (s)'); axes[0,0].grid(True)

axes[0,1].plot(t[:300], x_ruidosa[:300], color='orange')
axes[0,1].set_title('Señal con Ruido (tiempo)')
axes[0,1].set_xlabel('Tiempo (s)'); axes[0,1].grid(True)

axes[1,0].plot(frecuencias, amp_limpia)
axes[1,0].set_title('Espectro Señal Limpia')
axes[1,0].set_xlabel('Frecuencia (Hz)'); axes[1,0].set_xlim(0, 200); axes[1,0].grid(True)
axes[1,0].axvline(x=60, color='r', linestyle='--', label='60 Hz')
axes[1,0].legend()

axes[1,1].plot(frecuencias, amp_ruidosa, color='orange')
axes[1,1].set_title('Espectro Señal con Ruido')
axes[1,1].set_xlabel('Frecuencia (Hz)'); axes[1,1].set_xlim(0, 200); axes[1,1].grid(True)
axes[1,1].axvline(x=60, color='r', linestyle='--', label='60 Hz')
axes[1,1].legend()

plt.tight_layout()
plt.show()