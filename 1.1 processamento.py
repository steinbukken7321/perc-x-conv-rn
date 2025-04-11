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
def exibir_histograma(matriz_original, matriz_suavizada, titulo_original='Histograma Original', titulo_suavizada='Histograma Suavizado'):
    plt.figure(figsize=(12, 4))
    
    # Subplot 1: Histograma da matriz original
    plt.subplot(1, 2, 1)  
    plt.hist(matriz_original.ravel(), bins=50, color='blue', edgecolor='black', alpha=0.7)
    plt.title(titulo_original)
    plt.xlabel('Valor de Intensidade')
    plt.ylabel('Frequência')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Subplot 2: Histograma da matriz suavizada
    plt.subplot(1, 2, 2)  
    plt.hist(matriz_suavizada.ravel(), bins=50, color='green', edgecolor='black', alpha=0.7)
    plt.title(titulo_suavizada)
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

def calcular_desvio_padrao_manual(matriz):
    # Converter a matriz para um array 1D de valores de intensidade
    valores = matriz.ravel()
    
    # 1. Calcular a média
    media = np.mean(valores)
    
    # 2. Calcular a soma dos quadrados das diferenças em relação à média
    soma_quadrados = np.sum((valores - media) ** 2)
    
    # 3. Calcular a variância (média dos quadrados das diferenças)
    variancia = soma_quadrados / len(valores)
    
    # 4. Calcular o desvio padrão (raiz quadrada da variância)
    desvio_padrao_manual = np.sqrt(variancia)
    
    return desvio_padrao_manual

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
# Redução por máscara de blocos
##################################
def reduzir_com_mascara(binary_images, block_size):
    """
    Reduz um conjunto de imagens binarizadas.
    binary_images: array numpy no formato (n_imagens, altura, largura)
    Retorna: array numpy no formato (n_imagens, altura//block_size, largura//block_size)
    """
    n_imagens, h, w = binary_images.shape
    reduced_h = h // block_size
    reduced_w = w // block_size
    reduced_images = np.zeros((n_imagens, reduced_h, reduced_w), dtype=np.uint8)

    for idx in range(n_imagens):  # Processa cada imagem individualmente
        for i in range(reduced_h):
            for j in range(reduced_w):
                block = binary_images[idx, i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
                reduced_images[idx, i, j] = 255 if np.mean(block) > 127.5 else 0
    return reduced_images

##################################
# Exibir imagens (agora com 4 colunas)
##################################
def exibir_imagens(original, suavizada, binarizada, reduzida):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4)) 
    titulos = ["Original", "Suavizada", "Binarizada", "Reduzida"]
    imagens = [original, suavizada, binarizada, reduzida]
    
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
desvio_padrao_original_manual = calcular_desvio_padrao_manual(matrizes[0])
desvio_padrao_manual = calcular_desvio_padrao_manual(matrizes_suavizadas[0])

limiar = 6 * desvio_padrao + media  # ➤ Limiar para binarização (0 a 255)

# ⬛ Binarização das matrizes suavizadas com base no limiar
matrizes_binarizadas = binarizar_matrizes(matrizes_suavizadas, limiar)

# 🔵 Definir o tamanho do bloco para redução 
block_size = 2  # bloco 2x2 = 4 (reduz 4 pixeis para 1)
matrizes_reduzidas = reduzir_com_mascara(matrizes_binarizadas[0], block_size)

'''
# 📋 Impressão de resultados
print(f"🎯 Desvio padrão da matriz original: {desvio_padrao_original:.2f}")
print(f"🎯 Desvio padrão da matriz suavizada: {desvio_padrao:.2f}")
print(f"🎯 Desvio padrão da matriz original com função manual: {desvio_padrao_original_manual:.2f}")
print(f"🎯 Desvio padrão da matriz suavizada com função manual: {desvio_padrao_manual:.2f}")

print(f"📊 Média da matriz original: {media_original:.2f}")
print(f"📊 Média da matriz suavizada: {media:.2f}")

print(f"📐 Limiar: 6*{desvio_padrao:.2f} + {media:.2f} = {limiar:.2f}")

print("Formato da matriz binarizada:", matrizes_binarizadas[0].shape)

# 📊 Plot: histograma da matriz original
exibir_histograma(matrizes[0], matrizes_suavizadas[0])

# 📊 Plot: imagem original, suavizada e binarizada
 exibir_imagens(matrizes[0][0], matrizes_suavizadas[0][0], matrizes_binarizadas[0][0])
'''
# 📊 Plot: imagem original, suavizada, binarizada e reduzida
exibir_imagens(matrizes[0][0], matrizes_suavizadas[0][0], matrizes_binarizadas[0][0], matrizes_reduzidas[0])