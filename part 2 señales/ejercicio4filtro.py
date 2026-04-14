import numpy as np
import matplotlib.pyplot as plt
from skimage import data
from skimage.color import rgb2gray
# Cargar imagen (usar imagen incluida en scikit-image)
imagen = rgb2gray(data.astronaut())   # O usar: data.camera()
imagen = imagen[:256, :256]           # Recortar para acelerar

# FFT 2D
F = np.fft.fft2(imagen)
F_centrada = np.fft.fftshift(F)       # Centrar frecuencia cero

# Espectro de amplitud (escala logarítmica para visualizar)
espectro = np.log1p(np.abs(F_centrada))

# Aplicar filtro pasa-bajo circular
filas, cols = imagen.shape
cr, cc = filas // 2, cols // 2       # Centro
radio = 30                            # Radio del filtro (en píxeles de frecuencia)

mascara = np.zeros((filas, cols))
Y, X = np.ogrid[:filas, :cols]
distancia = np.sqrt((X - cc)**2 + (Y - cr)**2)
mascara[distancia <= radio] = 1

F_filtrada = F_centrada * mascara
imagen_filtrada = np.real(np.fft.ifft2(np.fft.ifftshift(F_filtrada)))

# Gráficas
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

axes[0,0].imshow(imagen, cmap='gray')
axes[0,0].set_title('Imagen Original'); axes[0,0].axis('off')

axes[0,1].imshow(espectro, cmap='inferno')
axes[0,1].set_title('Espectro de Amplitud (FFT2D centrada)')
axes[0,1].axis('off')

axes[1,0].imshow(mascara, cmap='gray')
axes[1,0].set_title(f'Máscara Pasa-Bajo (radio={radio} px)'); axes[1,0].axis('off')

axes[1,1].imshow(imagen_filtrada, cmap='gray')
axes[1,1].set_title('Imagen Filtrada (pasa-bajo)'); axes[1,1].axis('off')

plt.tight_layout()
plt.show()