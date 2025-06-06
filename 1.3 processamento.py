import numpy as np
import zipfile
import io
import matplotlib.pyplot as plt
import os

##############################################
# Carregar matrizes do ZIP
##############################################

zip_path_esqueletos = "matrizes_esqueletos_tcc.zip"
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
    # 🔧 Empilhar os arrays ao longo do primeiro eixo
    matrizes = np.concatenate(matrizes, axis=0)
    return matrizes


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


# Carregar binarizadas e reduzidas
matrizes_reduzidas = carregar_matrizes_zip(zip_path_reduzidas)
matrizes_esqueletos = carregar_matrizes_zip(zip_path_esqueletos)

# verificar formato das matrizes
print(f"Formato das matrizes reduzidas: {matrizes_reduzidas.shape}")
print(f"Formato das matrizes esqueletos: {matrizes_esqueletos.shape}")

# acessar qualquer matriz(img) em formato de img
"""
plt.imshow(matrizes_reduzidas[0][0], cmap='gray')
plt.title('Imagem 1')
plt.axis('off')
plt.show()

plt.imshow(matrizes_esqueletos[0], cmap='gray')
plt.title('Imagem 1')
plt.axis('off')
plt.show()
"""

fig, axs = plt.subplots(1, 2, figsize=(10, 5))

axs[0].imshow(matrizes_reduzidas[0][0], cmap='gray')
axs[0].set_title('Imagem Reduzida 1')
axs[0].axis('off')

axs[1].imshow(matrizes_esqueletos[0][0], cmap='gray')
axs[1].set_title('Esqueleto 1')
axs[1].axis('off')

plt.tight_layout()
plt.show()