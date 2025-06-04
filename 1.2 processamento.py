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


def aplicar_filtro_esqueleto(matrizes, esqueleto):
    """
    Aplica um filtro esqueleto nas matrizes.
    
    Parâmetros:
    - matrizes: np.array de formato (n, h, w) contendo as imagens
    - esqueleto: np.array 3x3 com o padrão a ser comparado
    
    Retorna:
    - matrizes filtradas
    """
    # Criar cópia para não modificar o original
    matrizes_filtradas = matrizes.copy()
    
    # Obter dimensões
    n, h, w = matrizes_filtradas.shape
    
    # Pad para lidar com bordas
    padded = np.pad(matrizes_filtradas, ((0, 0), (1, 1), (1, 1)), mode='constant')
    
    # Percorrer cada matriz
    for i in range(n):
        # Percorrer cada pixel (exceto bordas)
        for y in range(1, h+1):
            for x in range(1, w+1):
                # Extrair vizinhança 3x3
                vizinhanca = padded[i, y-1:y+2, x-1:x+2]
                
                # Verificar se corresponde exatamente ao esqueleto
                if np.array_equal(vizinhanca, esqueleto):
                    # Zerar o pixel central na matriz original
                    matrizes_filtradas[i, y-1, x-1] = 0
                    
    return matrizes_filtradas

# Definir os 4 padrões esqueleto
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

# Carregar binarizadas e reduzidas
matrizes_binarizadas = carregar_matrizes_zip(zip_path_binarizadas)
matrizes_reduzidas = carregar_matrizes_zip(zip_path_reduzidas)

# Aplicar todos os filtros sequencialmente
matrizes_esqueletos = matrizes_reduzidas
for esqueleto in [esqueleto_vertical, esqueleto_horizontal, 
                  esqueleto_diagonal_principal, esqueleto_diagonal_secundaria]:
    matrizes_filtradas = aplicar_filtro_esqueleto(matrizes_filtradas, esqueleto)

# verificar formato das matrizes
print(f"Formato das matrizes binarizadas: {matrizes_binarizadas.shape}")
print(f"Formato das matrizes reduzidas: {matrizes_reduzidas.shape}")
print(f"Formato das matrizes esqueletos: {matrizes_esqueletos.shape}")

# acessar qualquer matriz(img) da lista matrizes_reduzida
"""
plt.imshow(matrizes_binarizadas[0][20], cmap='gray')
plt.title('Imagem 1')
plt.axis('off')
plt.show()

plt.imshow(matrizes_reduzidas[0][23], cmap='gray')
plt.title('Imagem 1')
plt.axis('off')
plt.show()
"""
plt.imshow(matrizes_esqueletos[0][23], cmap='gray')
plt.title('Imagem 1')
plt.axis('off')
plt.show()
