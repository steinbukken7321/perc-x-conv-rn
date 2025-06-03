import io
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import zipfile
import os

"""

Este código realiza o processamento de um conjunto de matrizes
(imagens) armazenadas em arquivos `.npy` compactados em arquivos `.zip`.
O processamento envolve análise estatística, binarização e
redução das matrizes, além de visualização e salvamento dos resultados.

🔹 Etapas do Processamento:

1️⃣ **Carregamento das Matrizes**
   - As matrizes são carregadas a partir de dois arquivos `.zip`:
     - `matrizes_tcc.zip` → matrizes originais.
     - `matrizes_suavizadas_tcc.zip` → matrizes suavizadas.
   - Os arquivos `.npy` dentro dos `.zip` são extraídos e carregados como arrays NumPy.

2️⃣ **Análise Estatística**
   - Calcula-se:
     - A média das intensidades dos pixels.
     - O desvio padrão de duas formas:
       🔸 Usando a função `np.std()`.
       🔸 De forma manual, implementando o cálculo matemático do desvio padrão.
   - Essa análise é feita para comparar a matriz original com a suavizada.

3️⃣ **Exibição de Histogramas**
   - Gera histogramas das distribuições de intensidade dos pixels das matrizes original e suavizada.
   - Permite visualizar a dispersão dos valores e os efeitos da suavização.

4️⃣ **Binarização das Matrizes**
   - As matrizes suavizadas são transformadas em imagens binárias (preto e branco).
   - Um **limiar (threshold)** é definido como:
     ```
     limiar = 5 * desvio_padrao + media
     ```
   - Pixels com valor maior ou igual ao limiar recebem o valor 255 (branco) e os demais recebem 0 (preto).

5️⃣ **Redução das Matrizes (Compressão por Blocos)**
   - As imagens binarizadas são reduzidas em resolução utilizando uma máscara de blocos.
   - Para cada bloco (ex.: 2x2 pixels), calcula-se a média dos valores:
     🔸 Se a média for maior que 127.5 → bloco recebe 255 (branco).
     🔸 Caso contrário → bloco recebe 0 (preto).
   - Isso reduz a matriz de tamanho (H, W) para (H//block_size, W//block_size).

6️⃣ **Visualização dos Resultados**
   - Exibe lado a lado:
     🔸 A imagem original.
     🔸 A imagem suavizada.
     🔸 A imagem binarizada.
     🔸 A imagem reduzida.
   - Permite comparar as transformações em cada etapa.

7️⃣ **Salvamento dos Dados**
   - As matrizes binarizadas e reduzidas são salvas em arquivos `.npy`.
   - Esses arquivos são então compactados em `.zip`:
     - `matrizes_binarizadas_tcc.zip`
     - `matrizes_reduzidas_tcc.zip`
   - Após compactar, os arquivos `.npy` são removidos do diretório para economia de espaço.
"""
zip_path_original = 'matrizes_tcc.zip'
zip_path_suavizadas = 'matrizes_suavizadas_tcc.zip'

matrizes = []
matrizes_suavizadas = []

# Abrir matrizes originais
with zipfile.ZipFile(zip_path_original, 'r') as zip_ref:
    npy_arquivos = [nome for nome in zip_ref.namelist()
                    if nome.endswith('.npy')]
    for nome in npy_arquivos:
        with zip_ref.open(nome) as arquivo:
            matriz = np.load(io.BytesIO(arquivo.read()))
            matrizes.append(matriz)

# Abrir matrizes suavizadas
with zipfile.ZipFile(zip_path_suavizadas, 'r') as zip_ref:
    npy_arquivos = [nome for nome in zip_ref.namelist()
                    if nome.endswith('.npy')]
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
    plt.hist(matriz_original.ravel(), bins=50,
             color='blue', edgecolor='black', alpha=0.7)
    plt.title(titulo_original)
    plt.xlabel('Valor de Intensidade')
    plt.ylabel('Frequência')
    plt.grid(True, linestyle='--', alpha=0.5)

    # Subplot 2: Histograma da matriz suavizada
    plt.subplot(1, 2, 2)
    plt.hist(matriz_suavizada.ravel(), bins=50,
             color='green', edgecolor='black', alpha=0.7)
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

"""
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
"""
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
    """
    Reduz um conjunto de imagens binarizadas.
    binary_images: array numpy no formato (n_imagens, altura, largura)
    Retorna: array numpy no formato (n_imagens, altura//block_size, largura//block_size)
    """


def reduzir_com_mascara(binary_images, block_size):
    n_batches, n_imagens, h, w = binary_images.shape

    reduced_h = h // block_size
    reduced_w = w // block_size

    # criar array para armazenar as imagens reduzidas
    reduced_images = np.zeros(
        (n_batches, n_imagens, reduced_h, reduced_w), dtype=np.uint8)

    for b in range(n_batches):
        for idx in range(n_imagens):
            for i in range(reduced_h):
                for j in range(reduced_w):
                    block = binary_images[b, idx, i*block_size:(
                        i+1)*block_size, j*block_size:(j+1)*block_size]
                    reduced_images[b, idx, i, j] = 255 if np.mean(
                        block) > 127.5 else 0

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


##################################
# Salvar matrizes em .npy
##################################
def salvar_matrizes(nome_arquivo, matrizes):
    np.save(nome_arquivo, np.array(matrizes))
    print(f"Matrizes salvas em {nome_arquivo}")

##################################
# Compactar em zip
##################################


def compactar_npy(nome_arquivo_npy, nome_zip):
    with zipfile.ZipFile(nome_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(nome_arquivo_npy)
    print(f"Arquivo compactado salvo como {nome_zip}")

##############################################
# PROCESSAMENTO DAS IMAGENS
##############################################


# 📐 Médias
media_original = np.mean(matrizes[0])
media = np.mean(matrizes_suavizadas[0])

# 📉 Cálculo do desvio padrão das matrizes originais e suavizadas
desvio_padrao_original = calcular_desvio_padrao(matrizes[0])
desvio_padrao = calcular_desvio_padrao(matrizes_suavizadas[0])
#desvio_padrao_original_manual = calcular_desvio_padrao_manual(matrizes[0])
#desvio_padrao_manual = calcular_desvio_padrao_manual(matrizes_suavizadas[0])

limiar = 5 * desvio_padrao + media  # ➤ Limiar para binarização (0 a 255)

# ⬛ Binarização das matrizes suavizadas com base no limiar
matrizes_binarizadas = binarizar_matrizes(matrizes_suavizadas, limiar)
matrizes_binarizadas = np.array(matrizes_binarizadas)

# 🔵 Definir o tamanho do bloco para redução
block_size = 2  # bloco 2x2 = 4 (reduz 4 pixeis para 1)
matrizes_reduzidas = reduzir_com_mascara(matrizes_binarizadas, block_size)

# verificar formato de grupos de imagens
#print(np.array(matrizes).shape)
print(f"Formato da lista das matrizes original: {np.array(matrizes).shape}")
#print(np.array(matrizes_suavizadas).shape)
print(f"Formato da lista das matrizes suavizdas: {np.array(matrizes_suavizadas).shape}")

# 📋 Impressão de resultados
print(f"🎯 Desvio padrão da matriz original: {desvio_padrao_original:.2f}")
print(f"🎯 Desvio padrão da matriz suavizada: {desvio_padrao:.2f}")
#print(f"🎯 Desvio padrão da matriz original com função manual: {desvio_padrao_original_manual:.2f}")
#print(f"🎯 Desvio padrão da matriz suavizada com função manual: {desvio_padrao_manual:.2f}")

print(f"📊 Média da matriz original: {media_original:.2f}")
print(f"📊 Média da matriz suavizada: {media:.2f}")

print(f"📐 Limiar: 5*{desvio_padrao:.2f} + {media:.2f} = {limiar:.2f}")

print(f"Formato da lista das matrizes binarizadas: {matrizes_binarizadas.shape}")
#print(matrizes_binarizadas.shape)  # Verifique se ficou (24, x, y)
print("Formato da matriz binarizada:", matrizes_binarizadas[0].shape)
print("Formato da matriz reduzidas:", matrizes_reduzidas[0].shape)

print(f"Formato da lista das matrizes reduzidas: {matrizes_reduzidas.shape}")
#print(matrizes_reduzidas.shape)  # Verifique se ficou (24, x, y)
print("Formato da matriz binarizada:", matrizes_reduzidas[0].shape)
print("Formato da matriz reduzidas:", matrizes_reduzidas[0].shape)

# 📊 Plot: histograma da matriz original
exibir_histograma(matrizes[0], matrizes_suavizadas[0])

# 📊 Plot: imagem original, suavizada, binarizada e reduzida
exibir_imagens(matrizes[0][0],             # First image (3000, 2000)
               matrizes_suavizadas[0][0],   # First smoothed image (3002, 2002)
               matrizes_binarizadas[0][0],  # First binarized image (3002, 2002)
               matrizes_reduzidas[0][0])    # First reduced image (1501, 1001)

# 💾 Salvamento das matrizes binarizadas em arquivo .npy
npy_path_binarizadas = "matrizes_binarizadas_tcc.npy"
zip_path_binarizadas = "matrizes_binarizadas_tcc.zip"

salvar_matrizes(npy_path_binarizadas, matrizes_binarizadas)
compactar_npy(npy_path_binarizadas, zip_path_binarizadas)
os.remove(npy_path_binarizadas)

# 💾 Salvamento das matrizes reduzidas em arquivo .npy
npy_path_reduzidas = "matrizes_reduzidas_tcc.npy"
zip_path_reduzidas = "matrizes_reduzidas_tcc.zip"

salvar_matrizes(npy_path_reduzidas, matrizes_reduzidas)
compactar_npy(npy_path_reduzidas, zip_path_reduzidas)
os.remove(npy_path_reduzidas)

"""
Formato da lista das matrizes original: (1, 24, 3000, 2000)
Formato da lista das matrizes suavizdas: (1, 24, 3002, 2002)
🎯 Desvio padrão da matriz original: 32.83
🎯 Desvio padrão da matriz suavizada: 25.30
📊 Média da matriz original: 51.05
📊 Média da matriz suavizada: 50.49
📐 Limiar: 5*25.30 + 50.49 = 176.99
Formato da lista das matrizes binarizadas: (1, 24, 3002, 2002)
Formato da matriz binarizada [0]: (24, 3002, 2002)
Formato da lista das matrizes reduzidas: (1, 24, 1501, 1001)
Formato da matriz reduzidas [0]: (24, 1501, 1001)
Matrizes salvas em matrizes_binarizadas_tcc.npy
Arquivo compactado salvo como matrizes_binarizadas_tcc.zip
Matrizes salvas em matrizes_reduzidas_tcc.npy
Arquivo compactado salvo como matrizes_reduzidas_tcc.zip
"""
