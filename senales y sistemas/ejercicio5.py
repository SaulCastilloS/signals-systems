import numpy as np
import matplotlib.pyplot as plt

# Frecuencia aleatoria desconocida
np.random.seed(None)  # Semilla aleatoria
f_desconocida = np.random.randint(50, 201)  # Entre 50 y 200 Hz (entero)
print(f"[SECRETO - no mirar hasta resolver] Frecuencia generada: {f_desconocida} Hz")

Fs = 1000
T = 1 / Fs
t = np.arange(0, 1, T)

x = np.sin(2 * np.pi * f_desconocida * t)

# FFT
N = len(x)
X = np.fft.fft(x)
frecuencias = np.fft.fftfreq(N, T)[:N//2]
amplitud = (2/N) * np.abs(X[:N//2])

# Identificar frecuencia dominante
idx_max = np.argmax(amplitud)
f_identificada = frecuencias[idx_max]

# Gráficas
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

axes[0].plot(t[:100], x[:100])
axes[0].set_title('Señal con Frecuencia Desconocida')
axes[0].set_xlabel('Tiempo (s)'); axes[0].grid(True)

axes[1].plot(frecuencias, amplitud, color='purple')
axes[1].set_title(f'Espectro FFT — Frecuencia identificada: {f_identificada:.1f} Hz')
axes[1].set_xlabel('Frecuencia (Hz)'); axes[1].set_xlim(0, 300); axes[1].grid(True)
axes[1].axvline(x=f_identificada, color='r', linestyle='--',
                label=f'f dominante = {f_identificada:.1f} Hz')
axes[1].legend()

plt.tight_layout()
plt.show()

print(f"Frecuencia identificada por FFT: {f_identificada:.1f} Hz")
print(f"Frecuencia real: {f_desconocida} Hz")
print(f"Error: {abs(f_identificada - f_desconocida):.1f} Hz")