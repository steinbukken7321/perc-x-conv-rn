import os
import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import zipfile

"""
🔹 Objetivo:
Este código realiza um pipeline completo de processamento de imagens. 
Ele foi desenvolvido para ler imagens de uma pasta, pré-processá-las e 
gerar saídas prontas para serem utilizadas em treinamentos de modelos, 
análises ou compressões.

✔️ FUNCIONALIDADES DO CÓDIGO:
1️⃣ **Contagem e listagem de imagens na pasta**
   ➤ Filtra apenas arquivos de imagem (formatos: png, jpg, jpeg, bmp).

2️⃣ **Verificação dos tamanhos das imagens**
   ➤ Identifica se as imagens possuem tamanhos diferentes.

3️⃣ **Padronização de formato**
   ➤ Converte todas as imagens da pasta para formato JPG em escala de cinza.

4️⃣ **Conversão para matrizes NumPy**
   ➤ Cada imagem é convertida em uma matriz, onde cada elemento representa a 
      intensidade de cinza (valores de 0 a 255).

5️⃣ **Zero Padding**
   ➤ Adiciona uma borda de zeros nas imagens (tamanho configurável) para 
      preservar as bordas durante operações de filtragem.

6️⃣ **Filtro de Média**
   ➤ Aplica uma máscara NxN para suavizar a imagem, reduzindo ruídos. 
      A média da vizinhança de cada pixel substitui o valor do pixel central.

7️⃣ **Cálculo de estatísticas**
   ➤ Mede a média e o desvio padrão da imagem original e da suavizada.

8️⃣ **Binarização**
   ➤ Converte a imagem suavizada para uma imagem binária (preto e branco), 
      utilizando um limiar adaptativo calculado como:
      ➤ limiar = média + (5 * desvio padrão)

9️⃣ **Exibição dos resultados**
   ➤ Mostra a imagem original, suavizada e binarizada lado a lado.
   ➤ Mostra também o histograma de intensidades da imagem suavizada.

🔟 **Salvamento dos dados**
   ➤ As matrizes das imagens originais e suavizadas são salvas em arquivos `.npy`.
   ➤ Esses arquivos são compactados em `.zip` para fácil armazenamento e transporte.
   ➤ Após compactar, os arquivos `.npy` são apagados do diretório para não ocupar espaço extra.

⚙️ PARÂMETROS CONFIGURÁVEIS:
- `pasta_imagens` → caminho onde estão as imagens.
- `size_padding` → tamanho do zero padding aplicado nas bordas (ex.: 1, 2...).
- `filtro_size` → tamanho da máscara do filtro de média (ex.: 3x3, 5x5...).

"""
##################################
# Contar imagens na pasta
##################################
def contareler_imagens(pasta):
    formatos_validos = {"png", "jpg", "jpeg", "bmp"}
    imagens = [f for f in os.listdir(pasta) if f.split(".")[-1].lower() in formatos_validos]
    return len(imagens), imagens

'''
se
"gato.png"
"cachorro.JPG"
"documento.pdf"
"letra.txt"
"pato.jpeg"

retorna:
quantidade de imagens (nesse caso seriam 3) e imagens:
["gato.png", "cachorro.JPG", "pato.jpeg"]

'''
##################################
# Obter tamanho das imagens
##################################
def obter_tamanho_imagens(pasta):
    _, imagens = contareler_imagens(pasta)
    tamanhos = set()
    
    for imagem in imagens:
        caminho = os.path.join(pasta, imagem)  
        with Image.open(caminho) as img:
            tamanhos.add(img.size)  # (largura, altura)
    
    return tamanhos

'''
path.join:
pasta = "C:/imagens"
imagem = "gato.jpg"
caminho = os.path.join(pasta, imagem)  # Resultado: "C:/imagens/gato.jpg"
'''

'''
se:
foto1.png	(28, 28)
foto2.png	(64, 64)
foto3.png	(28, 28)
foto4.png	(128, 128)
retorna:
{(28, 28), (64, 64), (128, 128)}
'''
##################################
# Padronizar formatos para JPG
##################################
def padronizar_formatos(pasta):
    _, imagens = contareler_imagens(pasta)
    formatos = set()
    
    for imagem in imagens:
        caminho = os.path.join(pasta, imagem)
        with Image.open(caminho) as img:
            formatos.add(img.format.lower())
    
    if len(formatos) > 1:
        for imagem in imagens:
            caminho = os.path.join(pasta, imagem)
            with Image.open(caminho) as img:
                novo_caminho = os.path.splitext(caminho)[0] + ".jpg"
                img.convert("L").save(novo_caminho, "JPEG")
                os.remove(caminho)



##################################
# Converter imagens para matrizes de intensidade
##################################
def converter_para_matriz(pasta):
    _, imagens = contareler_imagens(pasta)
    matrizes = []
    
    for imagem in imagens:
        caminho = os.path.join(pasta, imagem)
        with Image.open(caminho) as img:
            matriz = np.array(img, dtype=np.uint8) # transforma os pixels da imagem em uma matriz de números inteiros (entre 0 e 255)
            matrizes.append(matriz)
    
    return matrizes # escala de cinza

'''
np.array(img) → converte a imagem para uma matriz.
dtype=np.uint8 → define que os valores vão de 0 a 255.

Retorna a lista matrizes, que contém todas as imagens convertidas em matrizes NumPy
'''
##################################
# Obter tamanho de uma matriz
##################################
def obter_tamanho_matriz(matriz):
    linhas, colunas = matriz.shape
    return linhas, colunas, linhas * colunas

'''
verificar se tamanho da matriz é = ao da imagem
'''
##################################
# Adicionar zero padding
##################################
def zero_padding(matrizes, size):
    return [np.pad(matriz, pad_width=size, mode='constant', constant_values=0) for matriz in matrizes]

'''
[[1, 2, 3],
 [4, 5, 6],
 [7, 8, 9]]

 size = 1
 [[0, 0, 0, 0, 0],
 [0, 1, 2, 3, 0],
 [0, 4, 5, 6, 0],
 [0, 7, 8, 9, 0],
 [0, 0, 0, 0, 0]]
'''

##################################
# Aplicar filtro de média NxN
##################################
def filtro_media(matrizes, filtro_size):
    pad = filtro_size // 2
    suavizadas = []

    for matriz in matrizes:
        altura, largura = matriz.shape
        nova_matriz = np.copy(matriz)

        # Percorre apenas os pixels que NÃO são do padding
        for i in range(pad, altura - pad):
            for j in range(pad, largura - pad):
                # Extrai a região NxN ao redor do pixel atual
                vizinhanca = matriz[i-pad:i+pad+1, j-pad:j+pad+1]
                media = np.sum(vizinhanca) / (filtro_size ** 2)     # Calcula a média
                nova_matriz[i, j] = int(media)                      # Atualiza o pixel

        suavizadas.append(nova_matriz.astype(np.uint8))

    return suavizadas


'''

size = 1
[[0, 0, 0, 0, 0],
[0, 1, 2, 3, 0],
[0, 4, 5, 6, 0],
[0, 7, 8, 9, 0],
[0, 0, 0, 0, 0]]

(1,1) (1,2) (1,3)
(2,1) (2,2) (2,3)
(3,1) (3,2) (3,3)

Centrado no matriz[1][1] = 1
[ [0, 0, 0],
  [0, 1, 2],
  [0, 4, 5] ]
Soma: 0+0+0+0+1+2+0+4+5 = 12, média = 12 / 9 = 1.33 arredonda p/ 1
Resumo p/ exemplo:
(0,0) (0,1) (0,2) (0,3) (0,4)
(1,0)  ✔️    ✔️    ✔️   (1,4)
(2,0)  ✔️    ✔️    ✔️   (2,4)
(3,0)  ✔️    ✔️    ✔️   (3,4)
(4,0) (4,1) (4,2) (4,3) (4,4)
'''


##################################
# Binarizar matrizes suavizadas
##################################
def binarizar_matrizes(matrizes, limiar=128):
    binarizadas = []
    for matriz in matrizes:
        binaria = (matriz >= limiar).astype(np.uint8) * 255  # 0 ou 255
        binarizadas.append(binaria)
    return binarizadas

##################################
# Exibir imagens
##################################
def exibir_imagens(originais, suavizadas, binarizadas):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    titulos = ["Original", "Suavizada", "Binarizada"]
    imagens = [originais, suavizadas, binarizadas]
    
    for ax, img, titulo in zip(axes, imagens, titulos):
        ax.imshow(img, cmap='gray')
        ax.set_title(titulo)
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()


##################################
# histograma
##################################
def exibir_histograma(matriz, titulo='Histograma de Intensidade'):
    plt.figure(figsize=(6, 4))
    plt.hist(matriz.ravel(), bins=50, color='gray', edgecolor='black')
    plt.title(titulo)
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
# CONFIGURAÇÕES INICIAIS (PARÂMETROS AJUSTÁVEIS)
##############################################
pasta_imagens = r"C:\\Users\\rafae\\Desktop\\perc-x-conv-rn\\img"  # Caminho da pasta com imagens
size_padding = 1             # ➤ Tamanho do zero padding (ex: 1, 2, ...)
filtro_size = 3              # ➤ Tamanho da máscara do filtro de média (ex: 3, 5, ...)

##############################################
# PROCESSAMENTO DAS IMAGENS
##############################################
'''
# ⏱️ Início do temporizador
tempo_inicio = time.time()
'''
# 📸 Contagem de imagens e padronização de formatos
quantidade, _ = contareler_imagens(pasta_imagens)
padronizar_formatos(pasta_imagens)

# 🔄 Conversão das imagens para matrizes
matrizes = converter_para_matriz(pasta_imagens)

# ➕ Aplicação de zero padding
padded_matrices = zero_padding(matrizes, size_padding)

# 🧹 Aplicação do filtro de média
matrizes_suavizadas = filtro_media(padded_matrices, filtro_size)

# 📐 Médias
media_original = np.mean(matrizes[0])
media = np.mean(matrizes_suavizadas[0])
# 📉 Cálculo do desvio padrão da imagem suavizada
desvio_padrao = calcular_desvio_padrao(matrizes_suavizadas[0])
desvio_padrao_original = calcular_desvio_padrao(matrizes[0])


limiar = 5 * desvio_padrao + media  # ➤ Limiar para binarização (0 a 255)

# ⬛ Binarização das matrizes suavizadas com base no limiar
matrizes_binarizadas = binarizar_matrizes(matrizes_suavizadas, limiar)


# 📏 Obtenção e exibição dos tamanhos das matrizes (original e com padding)
tamanho_original = obter_tamanho_matriz(matrizes[0])
tamanho_padded = obter_tamanho_matriz(padded_matrices[0])



'''
# ⏱️ Fim do temporizador e cálculo do tempo total
tempo_fim = time.time()
tempo_total = tempo_fim - tempo_inicio
'''
# 📋 Impressão de resultados
print(f"📂 Total de imagens: {quantidade}")

print(f"📐 Tamanho original da matriz: {tamanho_original[0]}x{tamanho_original[1]}")
print(f"📐 Tamanho após zero padding: {tamanho_padded[0]}x{tamanho_padded[1]}")

print(f"🎯 Desvio padrão da matriz original: {desvio_padrao_original:.2f}")
print(f"🎯 Desvio padrão da matriz suavizada: {desvio_padrao:.2f}")

print(f"📊 Média da matriz original: {media_original:.2f}")
print(f"📊 Média da matriz suavizada: {media:.2f}")

print(f"📐 Limiar: 3*{desvio_padrao:.2f} + {media:.2f} = {limiar:.2f}")



# 📊 Plot: histograma da matriz suavizada
exibir_histograma(matrizes_suavizadas[0])
# 📊 Plot: imagem original, suavizada e binarizada
exibir_imagens(matrizes[0], matrizes_suavizadas[0], matrizes_binarizadas[0])



'''
print(f"⏳ Tempo total de execução: {tempo_total:.2f} segundos")



'''

# 💾 Salvamento das matrizes (lidas do zip) em arquivo .npy
npy_path_matrizes = "matrizes_tcc.npy"
zip_path_matrizes = "matrizes_tcc.zip"

salvar_matrizes(npy_path_matrizes, matrizes)
compactar_npy(npy_path_matrizes, zip_path_matrizes)
os.remove(npy_path_matrizes)

# 💾 Salvamento das matrizes suavizadas em arquivo .npy
npy_path_suavizadas = "matrizes_suavizadas_tcc.npy"
zip_path_suavizadas = "matrizes_suavizadas_tcc.zip"

salvar_matrizes(npy_path_suavizadas, matrizes_suavizadas)
compactar_npy(npy_path_suavizadas, zip_path_suavizadas)
os.remove(npy_path_suavizadas)

