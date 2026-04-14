import numpy as np
import matplotlib.pyplot as plt

# ---- CON Fs = 500 Hz (ALIASING) ----
Fs_bajo = 500
T_bajo = 1 / Fs_bajo
t_bajo = np.arange(0, 1, T_bajo)

f1, f2 = 250, 1000
x_bajo = np.sin(2 * np.pi * f1 * t_bajo) + np.sin(2 * np.pi * f2 * t_bajo)

N_bajo = len(x_bajo)
X_bajo = np.fft.fft(x_bajo)
freqs_bajo = np.fft.fftfreq(N_bajo, T_bajo)[:N_bajo//2]
amp_bajo = (2/N_bajo) * np.abs(X_bajo[:N_bajo//2])

# ---- CON Fs = 3000 Hz (SIN aliasing) ----
Fs_alto = 3000
T_alto = 1 / Fs_alto
t_alto = np.arange(0, 1, T_alto)

x_alto = np.sin(2 * np.pi * f1 * t_alto) + np.sin(2 * np.pi * f2 * t_alto)

N_alto = len(x_alto)
X_alto = np.fft.fft(x_alto)
freqs_alto = np.fft.fftfreq(N_alto, T_alto)[:N_alto//2]
amp_alto = (2/N_alto) * np.abs(X_alto[:N_alto//2])

# Gráficas
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

axes[0].plot(freqs_bajo, amp_bajo, color='red')
axes[0].set_title(f'Espectro con Fs={Fs_bajo} Hz — ALIASING\n(Nyquist={Fs_bajo//2} Hz, 1000 Hz aparece como alias)')
axes[0].set_xlabel('Frecuencia (Hz)'); axes[0].grid(True)
axes[0].axvline(x=Fs_bajo//2, color='k', linestyle=':', label=f'Nyquist = {Fs_bajo//2} Hz')
axes[0].legend()

axes[1].plot(freqs_alto, amp_alto, color='green')
axes[1].set_title(f'Espectro con Fs={Fs_alto} Hz — SIN ALIASING')
axes[1].set_xlabel('Frecuencia (Hz)'); axes[1].grid(True)
axes[1].axvline(x=250,  color='r', linestyle='--', label='250 Hz')
axes[1].axvline(x=1000, color='b', linestyle='--', label='1000 Hz')
axes[1].legend()

plt.tight_layout()
plt.show()

print(f"Con Fs=500 Hz, Nyquist = 250 Hz")
print(f"La componente de 1000 Hz produce aliasing en: {1000 % 500} Hz o {Fs_bajo - (1000 % Fs_bajo)} Hz")   