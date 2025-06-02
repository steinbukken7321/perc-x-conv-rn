import numpy as np
import zipfile
import io
import matplotlib.pyplot as plt

##############################################
# Carregar matrizes do ZIP
##############################################

def carregar_matrizes_de_zip(caminho_zip):
    matrizes = []
    with zipfile.ZipFile(caminho_zip, 'r') as zip_ref:
        npy_arquivos = [nome for nome in zip_ref.namelist() if nome.endswith('.npy')]
        for nome in npy_arquivos:
            with zip_ref.open(nome) as arquivo:
                matriz = np.load(io.BytesIO(arquivo.read()))
                matrizes.append(matriz)
    return matrizes

arquivo_zip1 = "matrizes_reduzidas_tcc.zip"
arquivo_zip2 = "matrizes_binarizadas_tcc.zip"
