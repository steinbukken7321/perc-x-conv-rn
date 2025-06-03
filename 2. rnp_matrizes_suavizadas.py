import numpy as np
import zipfile
import io
import matplotlib.pyplot as plt

##############################################
# Carregar matrizes do ZIP
##############################################
zip_path_binarizadas = "matrizes_binarizadas_tcc.zip"
zip_path_reduzidas = "matrizes_reduzidas_tcc.zip"

matrizes_binarizadas = []
matrizes_reduzidas = []

# Abrir matrizes binarizadas
with zipfile.ZipFile(zip_path_binarizadas, 'r') as zip_ref:
    npy_arquivos = [nome for nome in zip_ref.namelist()
                    if nome.endswith('.npy')]
    for nome in npy_arquivos:
        with zip_ref.open(nome) as arquivo:
            matriz = np.load(io.BytesIO(arquivo.read()))
            matrizes_binarizadas.append(matriz)

# Abrir matrizes reduzidas
with zipfile.ZipFile(zip_path_reduzidas, 'r') as zip_ref:
    npy_arquivos = [nome for nome in zip_ref.namelist()
                    if nome.endswith('.npy')]
    for nome in npy_arquivos:
        with zip_ref.open(nome) as arquivo:
            matriz = np.load(io.BytesIO(arquivo.read()))
            matrizes_reduzidas.append(matriz)

# formato

print(f"Formato da lista das matrizes suavizdas: {np.array(matrizes_binarizadas).shape}")
print(f"Formato da lista das matrizes suavizdas: {np.array(matrizes_reduzidas).shape}")


