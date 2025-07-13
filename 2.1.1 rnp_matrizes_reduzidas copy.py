import numpy as np
import zipfile
import io
from scipy.ndimage import label

##############################################
# Parâmetros ajustáveis
##############################################
tamanho_janela = 3
bias = 1.0
num_camadas_ocultas = 2
neuronios_ocultos = 256
num_epochs = 5
limiar_alvo = 220
taxa_aprendizado = 0.1
arquivo_matrizes = "matrizes_tcc.npy"
zip_path_matrizes = "matrizes_tcc.zip"

##############################################
# Função de ativação e derivada (Sigmoid)
##############################################
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def derivada_sigmoid(x):
    sx = sigmoid(x)
    return sx * (1 - sx)

##############################################
# Carregar matrizes suavizadas do arquivo
##############################################
def carregar_matrizes_zip(zip_path):
    matrizes = []
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        npy_arquivos = [nome for nome in zip_ref.namelist() if nome.endswith('.npy')]
        for nome in npy_arquivos:
            with zip_ref.open(nome) as arquivo:
                matriz = np.load(io.BytesIO(arquivo.read())).astype(np.float32)
                matrizes.append(matriz)
    matrizes = np.concatenate(matrizes, axis=0)
    return matrizes

##############################################
# Geração das janelas e rótulos (dataset)
##############################################
def gerar_dados_treino(matrizes, tamanho_janela, limiar_alvo):
    X = []
    y = []
    pad = tamanho_janela // 2

    for matriz in matrizes:
        for i in range(pad, matriz.shape[0] - pad):
            for j in range(pad, matriz.shape[1] - pad):
                janela = matriz[i-pad:i+pad+1, j-pad:j+pad+1]
                vetor = janela.flatten()
                media = np.mean(vetor)
                X.append(vetor)
                y.append(1 if media > limiar_alvo else 0)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32).reshape(-1, 1)

##############################################
# Inicialização de pesos
##############################################
def inicializar_pesos(entrada, camadas_ocultas, saida):
    pesos = []
    pesos.append(np.random.randn(entrada, neuronios_ocultos).astype(np.float32))
    for _ in range(camadas_ocultas - 1):
        pesos.append(np.random.randn(neuronios_ocultos, neuronios_ocultos).astype(np.float32))
    pesos.append(np.random.randn(neuronios_ocultos, saida).astype(np.float32))
    return pesos

##############################################
# Feedforward
##############################################
def feedforward(x, pesos):
    ativacoes = [x]
    entrada = x
    for w in pesos:
        entrada = sigmoid(np.dot(entrada, w) + bias)
        ativacoes.append(entrada)
    return ativacoes

##############################################
# Backpropagation
##############################################
def backpropagation(pesos, ativacoes, y_real):
    gradientes = [None] * len(pesos)
    erro = ativacoes[-1] - y_real
    delta = erro * derivada_sigmoid(ativacoes[-1])
    for i in reversed(range(len(pesos))):
        gradientes[i] = np.dot(ativacoes[i].T, delta)
        if i > 0:
            delta = np.dot(delta, pesos[i].T) * derivada_sigmoid(ativacoes[i])
    return gradientes

##############################################
# Treinamento
##############################################
def treinar(X, y, pesos, epocas):
    for epoca in range(epocas):
        ativacoes = feedforward(X, pesos)
        gradientes = backpropagation(pesos, ativacoes, y)
        for i in range(len(pesos)):
            pesos[i] -= taxa_aprendizado * gradientes[i]
        if epoca % 1 == 0:
            pred = (ativacoes[-1] > 0.5).astype(int)
            acuracia = np.mean(pred == y)
            print(f"Época {epoca+1}/{epocas} - Acurácia: {acuracia:.4f}")
    return pesos

##############################################
# Contar alvos com janelas em lote e regiões conectadas
##############################################
def contar_alvos_rapido(matriz_teste, pesos, tamanho_janela):
    pad = tamanho_janela // 2
    janelas = []
    posicoes = []

    for i in range(pad, matriz_teste.shape[0] - pad):
        for j in range(pad, matriz_teste.shape[1] - pad):
            janela = matriz_teste[i-pad:i+pad+1, j-pad:j+pad+1].flatten()
            janelas.append(janela)
            posicoes.append((i, j))

    janelas = np.array(janelas, dtype=np.float32)
    saidas = feedforward(janelas, pesos)[-1]

    mapa_binario = np.zeros_like(matriz_teste, dtype=np.uint8)
    for idx, (i, j) in enumerate(posicoes):
        if saidas[idx] > 0.5:
            mapa_binario[i, j] = 1

    estrutura = np.ones((3, 3), dtype=np.uint8)
    mapa_rotulado, num_alvos = label(mapa_binario, structure=estrutura)
    return num_alvos

##############################################
# Execução
##############################################
if __name__ == "__main__":
    print("🔍 Carregando matrizes...")
    matrizes = carregar_matrizes_zip(zip_path_matrizes)

    print("📦 Gerando dados de treino...")
    X, y = gerar_dados_treino(matrizes, tamanho_janela, limiar_alvo)

    print("🧠 Inicializando rede neural...")
    pesos = inicializar_pesos(X.shape[1], num_camadas_ocultas, 1)

    print("🏋️ Treinando rede neural...")
    pesos = treinar(X, y, pesos, num_epochs)

    print("🧪 Testando em nova imagem...")
    imagem_teste = matrizes[0]
    total_alvos = contar_alvos_rapido(imagem_teste, pesos, tamanho_janela)
    print(f"✅ Total de alvos detectados: {total_alvos}")
