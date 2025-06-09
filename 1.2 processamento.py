import numpy as np
import zipfile
import io
import matplotlib.pyplot as plt
import os
from scipy.ndimage import correlate

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

##############################################
# Aplicar filtro por comparação direta
##############################################
def aplicar_filtro_esqueleto_direto(matrizes, esqueleto):
    """
    filtro esqueleto por comparação direta.
    - matrizes: np.array com shape (b, n, h, w)
    - esqueleto: np.array 3x3 com valores 0 ou 255
    """
    b, n, h, w = matrizes.shape
    resultado = matrizes.copy()
    
    # Converter o esqueleto para binário
    kernel_bin = (esqueleto == 255).astype(np.uint8)

    for i in range(b):
        for j in range(n):
            img = (resultado[i, j] == 255).astype(np.uint8)
            
            # Varredura da imagem (evitar bordas)
            for y in range(1, h - 1):
                for x in range(1, w - 1):
                    janela = img[y - 1:y + 2, x - 1:x + 2]
                    if np.array_equal(janela, kernel_bin):
                        resultado[i, j, y, x] = 0  # Zera o pixel central

    return resultado

##############################################
# Aplicar filtro por correlação binária
##############################################
def aplicar_filtro_esqueleto_binario(matrizes, esqueleto):
    """
    filtro esqueleto nas matrizes usando correlação binária
    - matrizes: np.array com shape (b, n, h, w)
    - esqueleto: np.array 3x3
    """
    b, n, h, w = matrizes.shape
    resultado = matrizes.copy()

    # Criar kernel binário com os 255s do esqueleto como 1
    kernel = (esqueleto == 255).astype(np.uint8)
    soma_kernel = np.sum(kernel)

    for i in range(b):
        for j in range(n):
            img = (resultado[i, j] == 255).astype(np.uint8)

            # Aplicar correlação
            resposta = correlate(img, kernel, mode='constant', cval=0)

            # Localizar correspondências completas
            correspondencias = (resposta == soma_kernel)

            # Zerar os pixels correspondentes
            resultado[i, j][correspondencias] = 0

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
matrizes_esqueletos = matrizes_reduzidas.copy()
for esqueleto in [
    esqueleto_vertical, 
    esqueleto_horizontal, 
    esqueleto_diagonal_principal, 
    esqueleto_diagonal_secundaria
]:
    matrizes_esqueletos = aplicar_filtro_esqueleto_binario(matrizes_esqueletos, esqueleto)

print(f"Formato das matrizes filtradas: {matrizes_esqueletos.shape}")

matrizes_esqueletos2 = matrizes_reduzidas.copy()
for esqueleto in [
    esqueleto_vertical, 
    esqueleto_horizontal, 
    esqueleto_diagonal_principal, 
    esqueleto_diagonal_secundaria
]:
    matrizes_esqueletos2 = aplicar_filtro_esqueleto_direto(matrizes_esqueletos2, esqueleto)
print(f"Formato das matrizes filtradas: {matrizes_esqueletos2.shape}")

# Visualizações
plt.figure(figsize=(15, 4))

plt.subplot(1, 3, 1)
plt.imshow(matrizes_reduzidas[0, 0], cmap='gray')
plt.title('Matrizes reduzidas')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(matrizes_esqueletos[0, 0], cmap='gray')
plt.title('Correlacao binaria')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(matrizes_esqueletos2[0, 0], cmap='gray')
plt.title('Comparacao direta')
plt.axis('off')

plt.tight_layout()
plt.show()

"""
# Salvar e compactar
npy_path = "matrizes_esqueletos_tcc.npy"
zip_path = "matrizes_esqueletos_tcc.zip"

salvar_matrizes(npy_path, matrizes_esqueletos)
compactar_npy(npy_path, zip_path)
os.remove(npy_path)
"""
