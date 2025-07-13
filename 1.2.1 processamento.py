import numpy as np
import zipfile
import io
import matplotlib.pyplot as plt
import os
from scipy.ndimage import correlate, binary_erosion, binary_dilation

##############################################
# Carregar matrizes do ZIP
##############################################
zip_path_reduzidas = "matrizes_reduzidas_tcc.zip"


def carregar_matrizes_zip(zip_path):
    matrizes = []
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        npy_arquivos = [nome for nome in zip_ref.namelist()
                        if nome.endswith('.npy')]
        for nome in npy_arquivos:
            with zip_ref.open(nome) as arquivo:
                matriz = np.load(io.BytesIO(arquivo.read()))
                matrizes.append(matriz)
    matrizes = np.concatenate(matrizes, axis=0)
    return matrizes

##############################################
# Erosão
##############################################


def aplicar_erosao(matrizes, tamanho_kernel):
    b, n, h, w = matrizes.shape
    estrutura = np.ones((tamanho_kernel, tamanho_kernel), dtype=bool)
    resultado = np.zeros_like(matrizes)

    for i in range(b):
        for j in range(n):
            binaria = (matrizes[i, j] == 255)
            erodida = binary_erosion(binaria, structure=estrutura)
            resultado[i, j] = (erodida * 255).astype(np.uint8)

    return resultado

##############################################
# Dilatação
##############################################


def aplicar_dilatacao(matrizes, tamanho_kernel):
    b, n, h, w = matrizes.shape
    estrutura = np.ones((tamanho_kernel, tamanho_kernel), dtype=bool)
    resultado = np.zeros_like(matrizes)

    for i in range(b):
        for j in range(n):
            binaria = (matrizes[i, j] == 255)
            dilatada = binary_dilation(binaria, structure=estrutura)
            resultado[i, j] = (dilatada * 255).astype(np.uint8)

    return resultado

##############################################
# Filtro de esqueletos
##############################################


def aplicar_filtro_esqueleto_binario(matrizes, esqueletos):
    b, n, h, w = matrizes.shape
    resultado = matrizes.copy()

    for esqueleto in esqueletos:
        kernel = (esqueleto == 255).astype(np.uint8)
        soma_kernel = np.sum(kernel)

        for i in range(b):
            for j in range(n):
                img_bin = (resultado[i, j] == 255).astype(np.uint8)
                resposta = correlate(img_bin, kernel, mode='constant', cval=0)
                correspondencias = (resposta == soma_kernel)
                resultado[i, j][correspondencias] = 0

    return resultado


##################################
# Esqueletos
##################################
esqueletos = [
    np.array([[0, 255, 0], [0, 255, 0], [0, 255, 0]], dtype=np.uint8),
    np.array([[0, 0, 0], [255, 255, 255], [0, 0, 0]], dtype=np.uint8),
    np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8),
    np.array([[0, 0, 255], [0, 255, 0], [255, 0, 0]], dtype=np.uint8)
]

##################################
# Salvar e compactar
##################################


def salvar_matrizes(nome_arquivo, matrizes):
    np.save(nome_arquivo, np.array(matrizes))
    print(f"Matrizes salvas em {nome_arquivo}")


def compactar_npy(nome_arquivo_npy, nome_zip):
    with zipfile.ZipFile(nome_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(nome_arquivo_npy)
    print(f"Arquivo compactado salvo como {nome_zip}")


##################################
# PROCESSAMENTO
##################################
# 1. Carregar
matrizes_reduzidas = carregar_matrizes_zip(zip_path_reduzidas)
print(f"Formato das matrizes: {matrizes_reduzidas.shape}")


# 5. Visualizações intermediárias
plt.figure(figsize=(15, 4))
plt.subplot(1, 3, 1)
plt.imshow(matrizes_reduzidas[0, 0], cmap='gray')
plt.title('Após redução')
plt.axis('off')

# 2. Aplicar erosão (rios somem)
matrizes_erosao = aplicar_erosao(matrizes_reduzidas, tamanho_kernel=1)

# 3. Aplicar filtros de esqueletos sobre imagens erodidas
# matrizes_filtradas = aplicar_filtro_esqueleto_binario(matrizes_erosao, esqueletos)

# 4. Aplicar dilatação separadamente (alvos expandem)
matrizes_dilatacao = aplicar_dilatacao(matrizes_erosao, tamanho_kernel=4)

# 5. Visualizações intermediárias
plt.subplot(1, 3, 2)
plt.imshow(matrizes_erosao[0, 0], cmap='gray')
plt.title('Após Erosão')
plt.axis('off')

"""
plt.subplot(1, 3, 2)
plt.imshow(matrizes_filtradas[0, 0], cmap='gray')
plt.title('Após Filtros')
plt.axis('off')
"""

plt.subplot(1, 3, 3)
plt.imshow(matrizes_dilatacao[0, 0], cmap='gray')
plt.title('Após Dilatação')
plt.axis('off')
plt.tight_layout()
plt.show()

# 6. Salvar e compactar
salvar_matrizes("matrizes_erosao.npy", matrizes_erosao)
compactar_npy("matrizes_erosao.npy", "matrizes_erosao.zip")
os.remove("matrizes_erosao.npy")

"""
salvar_matrizes("matrizes_filtradas.npy", matrizes_filtradas)
compactar_npy("matrizes_filtradas.npy", "matrizes_filtradas.zip")
os.remove("matrizes_filtradas.npy")
"""
salvar_matrizes("matrizes_dilatacao.npy", matrizes_dilatacao)
compactar_npy("matrizes_dilatacao.npy", "matrizes_dilatacao.zip")
os.remove("matrizes_dilatacao.npy")
