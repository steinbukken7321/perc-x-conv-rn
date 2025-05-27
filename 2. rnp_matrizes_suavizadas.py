import numpy as np
import zipfile
import io

##############################################
# Parâmetros ajustáveis
##############################################

tamanho_janela = 3
num_camadas_ocultas = 2
neuronios_ocultos = 256
num_epochs = 10
limiar_alvo = 180
taxa_aprendizado = 0.01
arquivo_zip = "matrizes_reduzidas_tcc.zip"

##############################################
# Função de ativação e derivada
##############################################

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def derivada_sigmoid(x):
    return x * (1 - x)

##############################################
# Carregar matrizes de um arquivo ZIP
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

##############################################
# Geração das janelas e rótulos considerando os frames
##############################################

def gerar_dados_treino(matrizes, tamanho_janela, limiar_alvo):
    X = []
    y = []
    pad = tamanho_janela // 2

    for matriz in matrizes:
        num_frames = matriz.shape[0]  # Eixo dos frames
        for frame in range(num_frames):
            img = matriz[frame]
            for i in range(pad, img.shape[0] - pad):
                for j in range(pad, img.shape[1] - pad):
                    janela = img[i-pad:i+pad+1, j-pad:j+pad+1]
                    vetor = janela.flatten()
                    media = np.mean(vetor)
                    X.append(vetor)
                    y.append(1 if media > limiar_alvo else 0)

    return np.array(X), np.array(y).reshape(-1, 1)

##############################################
# Inicialização de pesos e bias
##############################################

def inicializar_pesos_e_bias(entrada, camadas_ocultas, saida):
    pesos = []
    bias = []

    # entrada -> oculta1
    pesos.append(np.random.randn(entrada, neuronios_ocultos) * 0.01)
    bias.append(np.zeros((1, neuronios_ocultos)))

    # camadas ocultas
    for _ in range(camadas_ocultas - 1):
        pesos.append(np.random.randn(neuronios_ocultos, neuronios_ocultos) * 0.01)
        bias.append(np.zeros((1, neuronios_ocultos)))

    # última oculta -> saída
    pesos.append(np.random.randn(neuronios_ocultos, saida) * 0.01)
    bias.append(np.zeros((1, saida)))

    return pesos, bias

##############################################
# Feedforward
##############################################

def feedforward(x, pesos, bias):
    ativacoes = [x]
    entrada = x

    for w, b in zip(pesos, bias):
        z = np.dot(entrada, w) + b
        saida = sigmoid(z)
        ativacoes.append(saida)
        entrada = saida

    return ativacoes

##############################################
# Backpropagation
##############################################

def backpropagation(pesos, bias, ativacoes, y_real):
    grad_pesos = [np.zeros_like(w) for w in pesos]
    grad_bias = [np.zeros_like(b) for b in bias]

    erro = ativacoes[-1] - y_real
    delta = erro * derivada_sigmoid(ativacoes[-1])

    for i in reversed(range(len(pesos))):
        grad_pesos[i] = np.dot(ativacoes[i].T, delta)
        grad_bias[i] = np.sum(delta, axis=0, keepdims=True)

        if i > 0:
            delta = np.dot(delta, pesos[i].T) * derivada_sigmoid(ativacoes[i])

    return grad_pesos, grad_bias

##############################################
# Treinamento
##############################################

def treinar(X, y, pesos, bias, epocas):
    for epoca in range(epocas):
        ativacoes = feedforward(X, pesos, bias)
        grad_pesos, grad_bias = backpropagation(pesos, bias, ativacoes, y)

        for i in range(len(pesos)):
            pesos[i] -= taxa_aprendizado * grad_pesos[i]
            bias[i] -= taxa_aprendizado * grad_bias[i]

        pred = (ativacoes[-1] > 0.5).astype(int)
        acuracia = np.mean(pred == y)
        print(f"Época {epoca+1}/{epocas} - Acurácia: {acuracia:.4f}")

    return pesos, bias

##############################################
# Contagem de alvos
##############################################

def contar_alvos(matriz_teste, pesos, bias, tamanho_janela):
    pad = tamanho_janela // 2
    contagem = 0
    num_frames = matriz_teste.shape[0]

    for frame in range(num_frames):
        img = matriz_teste[frame]
        for i in range(pad, img.shape[0] - pad):
            for j in range(pad, img.shape[1] - pad):
                janela = img[i-pad:i+pad+1, j-pad:j+pad+1].flatten().reshape(1, -1)
                saida = feedforward(janela, pesos, bias)[-1]
                if saida > 0.5:
                    contagem += 1

    return contagem

##############################################
# Execução principal
##############################################

if __name__ == "__main__":
    print("🔍 Carregando matrizes do zip...")
    matrizes = carregar_matrizes_de_zip(arquivo_zip)

    print(f"Total de matrizes carregadas: {len(matrizes)}")
    print(f"Shape da primeira matriz: {matrizes[0].shape}")

    """
    24 "imagens" dentro da mesma matriz.
    Cada imagem tem 1501 linhas e 1001 colunas.
    """
    
    print("📦 Gerando dados de treino...")
    X, y = gerar_dados_treino(matrizes, tamanho_janela, limiar_alvo)

    print(f"Dados de treino gerados: {X.shape[0]} amostras, {X.shape[1]} features.")

    print("🧠 Inicializando rede neural...")
    pesos, bias = inicializar_pesos_e_bias(X.shape[1], num_camadas_ocultas, 1)

    print("🏋️ Treinando rede neural...")
    pesos, bias = treinar(X, y, pesos, bias, num_epochs)

    print("🧪 Testando em uma matriz de exemplo...")
    matriz_teste = matrizes[0]  # Usando a primeira matriz para teste
    total_alvos = contar_alvos(matriz_teste, pesos, bias, tamanho_janela)

    print(f"✅ Total de alvos detectados na matriz de teste: {total_alvos}")