import io
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import zipfile

zip_path_original = 'matrizes_tcc.zip'
zip_path_suavizadas = 'matrizes_suavizadas_tcc.zip'

matrizes = []
matrizes_suavizadas = []

# Abrir matrizes originais
with zipfile.ZipFile(zip_path_original, 'r') as zip_ref:
    npy_arquivos = [nome for nome in zip_ref.namelist() if nome.endswith('.npy')]
    for nome in npy_arquivos:
        with zip_ref.open(nome) as arquivo:
            matriz = np.load(io.BytesIO(arquivo.read()))
            matrizes.append(matriz)

# Abrir matrizes suavizadas
with zipfile.ZipFile(zip_path_suavizadas, 'r') as zip_ref:
    npy_arquivos = [nome for nome in zip_ref.namelist() if nome.endswith('.npy')]
    for nome in npy_arquivos:
        with zip_ref.open(nome) as arquivo:
            matriz = np.load(io.BytesIO(arquivo.read()))
            matrizes_suavizadas.append(matriz)

##############################################
# FUNÇÕES
##############################################

##################################
# histograma
##################################
def exibir_histograma(matriz, titulo='Histograma de Intensidade'):
    plt.figure(figsize=(6, 4))
    plt.hist(matriz.ravel(), bins=50, color='gray', edgecolor='black')
    plt.title(titulo)
    plt.xlabel('Valor de Intensidade')
    plt.ylabel('Frequência')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

##################################
# Calcular desvio padrão do histograma (intensidade)
##################################
def calcular_desvio_padrao(matriz):
    return np.std(matriz)


##################################
# Binarizar matrizes suavizadas
##################################
def binarizar_matrizes(matrizes_suavizadas, limiar):
    binarizadas = []
    for matriz in matrizes_suavizadas:
        binaria = (matriz >= limiar).astype(np.uint8) * 255  # 0 ou 255
        binarizadas.append(binaria)
    return binarizadas


##################################
# Exibir imagens
##################################
def exibir_imagens(originais, suavizadas, binarizadas):
    fig, axes = plt.subplots(1, 3, figsize=(14, 7))
    titulos = ["Original", "Suavizada", "Binarizada"]
    imagens = [originais, suavizadas, binarizadas]
    
    for ax, img, titulo in zip(axes, imagens, titulos):
        ax.imshow(img, cmap='gray')
        ax.set_title(titulo)
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

##############################################
# PROCESSAMENTO DAS IMAGENS
##############################################

# 📐 Médias
media_original = np.mean(matrizes[0])
media = np.mean(matrizes_suavizadas[0])

# 📉 Cálculo do desvio padrão das matrizes originais e suavizadas
desvio_padrao_original = calcular_desvio_padrao(matrizes[0])
desvio_padrao = calcular_desvio_padrao(matrizes_suavizadas[0])

limiar = 6 * desvio_padrao + media  # ➤ Limiar para binarização (0 a 255)

# ⬛ Binarização das matrizes suavizadas com base no limiar
matrizes_binarizadas = binarizar_matrizes(matrizes_suavizadas, limiar)


# 📋 Impressão de resultados
print(f"🎯 Desvio padrão da matriz original: {desvio_padrao_original:.2f}")
print(f"🎯 Desvio padrão da matriz suavizada: {desvio_padrao:.2f}")

print(f"📊 Média da matriz original: {media_original:.2f}")
print(f"📊 Média da matriz suavizada: {media:.2f}")

print(f"📐 Limiar: 6*{desvio_padrao:.2f} + {media:.2f} = {limiar:.2f}")

# 📊 Plot: histograma da matriz suavizada
exibir_histograma(matrizes_suavizadas[0])
# 📊 Plot: imagem original, suavizada e binarizada
exibir_imagens(matrizes[0][0], matrizes_suavizadas[0][0], matrizes_binarizadas[0][0])

