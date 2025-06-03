import numpy as np
import zipfile
import io
import matplotlib.pyplot as plt

##############################################
# Carregar matrizes do ZIP
##############################################

zip_path_binarizadas = "matrizes_binarizadas_tcc.zip"
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


# Carregar binarizadas e reduzidas
matrizes_binarizadas = carregar_matrizes_zip(zip_path_binarizadas)
matrizes_reduzidas = carregar_matrizes_zip(zip_path_reduzidas)

# verificar formato das matrizes

print(f"Formato das matrizes binarizadas: {matrizes_binarizadas.shape}")
print(f"Formato das matrizes reduzidas: {matrizes_reduzidas.shape}")


# acessar qualquer matriz(img) da lista matrizes_reduzida
"""
plt.imshow(matrizes_binarizadas[0][14], cmap='gray')
plt.title('Imagem 1')
plt.axis('off')
plt.show()
"""
