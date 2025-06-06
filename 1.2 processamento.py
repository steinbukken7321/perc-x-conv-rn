import numpy as np
import zipfile
import io
import matplotlib.pyplot as plt
import os

##############################################
# Carregar matrizes do ZIP
##############################################
zip_path_reduzidas = "matrizes_reduzidas_tcc.zip"

def carregar_matrizes_zip(zip_path):
    matrizes = []
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        npy_arquivos = [nome for nome in zip_ref.namelist() if nome.endswith('.npy')]
        for nome in npy_arquivos:
            with zip_ref.open(nome) as arquivo:
                matriz = np.load(io.BytesIO(arquivo.read()))
                matrizes.append(matriz)
    matrizes = np.concatenate(matrizes, axis=0)
    return matrizes

def aplicar_filtro_esqueleto(matrizes, esqueleto):
    """
    Aplica um filtro esqueleto nas matrizes.
    - matrizes: np.array com shape (b, n, h, w)
    - esqueleto: np.array 3x3
    """
    b, n, h, w = matrizes.shape
    resultado = matrizes.copy()

    # Padding apenas nas dimensões espaciais
    padded = np.pad(resultado, ((0, 0), (0, 0), (1, 1), (1, 1)), mode='constant')

    for i in range(b):
        for j in range(n):
            for y in range(1, h + 1):
                for x in range(1, w + 1):
                    vizinhanca = padded[i, j, y - 1:y + 2, x - 1:x + 2]
                    if np.array_equal(vizinhanca, esqueleto):
                        resultado[i, j, y - 1, x - 1] = 0
    return resultado

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
# Esqueletos
##################################
esqueleto_vertical = np.array([
    [0, 255, 0],
    [0, 255, 0],
    [0, 255, 0]
], dtype=np.uint8)

esqueleto_horizontal = np.array([
    [0, 0, 0],
    [255, 255, 255],
    [0, 0, 0]
], dtype=np.uint8)

esqueleto_diagonal_principal = np.array([
    [255, 0, 0],
    [0, 255, 0],
    [0, 0, 255]
], dtype=np.uint8)

esqueleto_diagonal_secundaria = np.array([
    [0, 0, 255],
    [0, 255, 0],
    [255, 0, 0]
], dtype=np.uint8)

##################################
# PROCESSAMENTO
##################################
matrizes_reduzidas = carregar_matrizes_zip(zip_path_reduzidas)
print(f"Formato das matrizes reduzidas: {matrizes_reduzidas.shape}")

# Aplicar filtros sequenciais
matrizes_prefiltradas = matrizes_reduzidas.copy()
for esqueleto in [esqueleto_vertical, esqueleto_horizontal, esqueleto_diagonal_principal, esqueleto_diagonal_secundaria]:
    matrizes_filtradas = aplicar_filtro_esqueleto(matrizes_prefiltradas, esqueleto)

print(f"Formato das matrizes filtradas: {matrizes_filtradas.shape}")

# Visualizar uma das imagens
plt.imshow(matrizes_filtradas[0, 0], cmap='gray')
plt.title('Imagem 0 após filtros')
plt.axis('off')
plt.show()

# Salvar e compactar
npy_path = "matrizes_esqueletos_tcc.npy"
zip_path = "matrizes_esqueletos_tcc.zip"

salvar_matrizes(npy_path, matrizes_filtradas)
compactar_npy(npy_path, zip_path)
os.remove(npy_path)
