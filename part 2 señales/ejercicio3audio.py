import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write

# Señal de audio sintética
Fs = 44100
t = np.linspace(0, 2.0, int(Fs * 2.0), endpoint=False)

audio = (0.5 * np.sin(2 * np.pi * 440  * t) +  # Componente baja
         0.3 * np.sin(2 * np.pi * 3000 * t) +   # Componente media
         0.2 * np.sin(2 * np.pi * 8000 * t))     # Componente alta

# FFT
N = len(audio)
X = np.fft.fft(audio)
frecuencias_completas = np.fft.fftfreq(N, 1/Fs)
frecuencias = frecuencias_completas[:N//2]
amp_original = (2/N) * np.abs(X[:N//2])

# Frecuencia dominante
f_dom = frecuencias[np.argmax(amp_original)]
print(f"Frecuencia dominante: {f_dom:.1f} Hz")

# Aplicar filtro pasa-bajo (fc = 1000 Hz)
fc = 1000  # Frecuencia de corte
X_filtrado = X.copy()
X_filtrado[np.abs(frecuencias_completas) > fc] = 0
audio_filtrado = np.real(np.fft.ifft(X_filtrado))
amp_filtrado = (2/N) * np.abs(X_filtrado[:N//2])

# Gráficas
fig, axes = plt.subplots(3, 1, figsize=(14, 11))

axes[0].plot(t[:2000], audio[:2000])
axes[0].set_title('Audio Original'); axes[0].grid(True)

axes[1].plot(frecuencias, amp_original, color='steelblue', label='Original')
axes[1].plot(frecuencias, amp_filtrado, color='green',    label=f'Filtrado (fc={fc} Hz)', alpha=0.8)
axes[1].set_title('Espectro: Original vs Filtrado Pasa-Bajo')
axes[1].set_xlim(0, 12000); axes[1].grid(True); axes[1].legend()
axes[1].axvline(x=fc, color='r', linestyle='--', label=f'fc={fc} Hz')

axes[2].plot(t[:2000], audio[:2000],        label='Original',  alpha=0.6)
axes[2].plot(t[:2000], audio_filtrado[:2000], label='Filtrado', color='green', alpha=0.8)
axes[2].set_title('Señal en Tiempo: Original vs Filtrada'); axes[2].grid(True); axes[2].legend()

plt.tight_layout()
plt.show()