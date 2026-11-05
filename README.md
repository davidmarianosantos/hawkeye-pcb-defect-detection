# Hawkeye: detecção de defeitos em PCB com YOLOv8

Detecção de quatro tipos de defeito em placas de circuito impresso com YOLOv8n, e análise de underfitting, overfitting e generalização fora da distribuição de treino.

## Resultados

| Cenário | Conjunto avaliado | mAP50 | Recall | Precisão |
|---|---|---:|---:|---:|
| Baseline (100 épocas, early stopping) | teste original (262 imagens) | 92,9% | 90,0% | 91,6% |
| Underfitting (2 épocas) | teste original | 46,1% | 31,4% | 75,3% |
| Treino longo (300 épocas) | validação (447 imagens) | 93,3% | 93,3% | 92,4% |
| **Baseline, imagens com matiz deslocada** | **teste alterado (262 imagens)** | **3,0%** | **6,7%** | **2,4%** |

mAP50 por classe no baseline (teste original):

| missing_hole | mouse_bite | open_circuit | short |
|---:|---:|---:|---:|
| 97,4% | 97,3% | 89,1% | 88,0% |

mAP50-95 do baseline: 50,0%.

O treino longo partiu dos pesos do modelo de 2 épocas, e as métricas dele são as da validação final do Ultralytics. Os demais cenários foram avaliados no conjunto de teste.

## O achado principal: o modelo aprendeu cor, não defeito

Para testar a generalização, o conjunto de teste foi reprocessado com deslocamento de matiz no espaço HSV (+100 no canal H), o que transforma as placas verdes em vermelhas. Os rótulos ficaram os mesmos, porque a posição dos defeitos não muda.

O modelo baseline marca 92,9% de mAP50 no teste original. Nas mesmas 262 imagens com a cor alterada, ele cai para 3,0%. Na inferência com confiança mínima de 0,25, ele não detectou nada em nenhuma das 262 imagens.

O dataset de treino só tem placas verdes. O resultado indica que a rede usou a cor da placa como atalho em vez da geometria do defeito (furos, cortes). Acurácia alta no teste não garante robustez: o conjunto de teste vem da mesma distribuição do treino, e não mede o que acontece quando essa distribuição muda.

O treino longo, por outro lado, não mostrou overfitting clássico: as métricas de validação continuaram altas e não houve divergência entre as curvas de treino e validação. Neste experimento, a falha de generalização veio do viés dos dados, não do número de épocas.

## Dataset e abordagem

**Dados**
- [PCB_detection](https://universe.roboflow.com/object-defect-1xn3t/pcb_detection-kecsf) (Roboflow Universe), versão 3, público.
- 1.807 imagens: 1.098 de treino (61%), 447 de validação (25%) e 262 de teste (14%).
- Classes: `missing_hole`, `mouse_bite`, `open_circuit`, `short`.
- Sem pré-processamento nem data augmentation, de propósito, para observar o comportamento "puro" da rede.

**Modelo e treino**
- YOLOv8n com transfer learning a partir dos pesos pré-treinados (`yolov8n.pt`).
- Baseline: 100 épocas, imagem 640, early stopping com `patience=10`.
- Underfitting: 2 épocas.
- Treino longo: 300 épocas.

**Ambiente**: Google Colab, GPU Tesla T4, Ultralytics 8.3.235, PyTorch 2.9.0+cu126.

**Pesos treinados** (versionados neste repositório):

| Arquivo | Treino | Cenário |
|---|---|---|
| `weights/baseline/best.pt` | `hawkeye_v1` | Baseline |
| `weights/underfit/best.pt` | `hawkeye_v3Under` | Underfitting |
| `weights/overfit/last.pt` | `hawkeye_v2` | Treino longo |

## Análise das métricas

Precisão e recall foram escolhidos pelo custo operacional de cada tipo de erro numa linha de inspeção:

- **Precisão baixa** gera falsos alarmes, que param a linha de produção sem necessidade.
- **Recall baixo** deixa placas defeituosas passarem e chegarem ao cliente.

Na matriz de confusão do baseline, a classe `short` teve 34 falsos alarmes: o modelo confundiu o fundo da placa com curto-circuito. Isso explica a precisão mais baixa dessa classe (83,8%).

O modelo de 2 épocas mostra o mesmo trade-off pelo lado oposto: precisão de 75,3% com recall de 31,4%. As poucas detecções que ele faz costumam estar certas, mas ele deixa passar cerca de dois terços dos defeitos.

## Como reproduzir

O notebook [`hawkeye_pcb_detection.ipynb`](hawkeye_pcb_detection.ipynb) foi executado no Google Colab. As células de treino são opcionais: depois de cada uma, há uma célula que carrega os pesos já treinados de `weights/`.

### 1. Chave do Roboflow

O download do dataset precisa de uma chave de API do Roboflow. O notebook lê a chave nesta ordem:

1. variável de ambiente `ROBOFLOW_API_KEY`;
2. no Colab, o secret `ROBOFLOW_API_KEY` (ícone de chave na barra lateral, com acesso liberado para o notebook).

Nunca coloque a chave diretamente no código.

### 2a. No Google Colab

Selecione um ambiente com GPU e rode, no início do notebook:

```python
!git clone https://github.com/davidmarianosantos/hawkeye-pcb-defect-detection.git
%cd hawkeye-pcb-defect-detection
```

Os caminhos dos pesos são relativos à raiz do repositório.

### 2b. Localmente

```bash
git clone https://github.com/davidmarianosantos/hawkeye-pcb-defect-detection.git
cd hawkeye-pcb-defect-detection
pip install ultralytics roboflow jupyter
export ROBOFLOW_API_KEY="sua-chave"
jupyter notebook hawkeye_pcb_detection.ipynb
```

No notebook, pule as células `!pip install` se as dependências já estiverem instaladas.

## Contexto acadêmico e autoria

Trabalho avaliativo da disciplina de Inteligência Artificial (DEC000068) da Universidade Estadual de Santa Cruz (UESC), Ilhéus, 2025.

O enunciado trazia um cenário industrial **fictício**, uma fábrica que inspeciona manualmente cerca de 2.000 placas por dia, e pedia uma análise de underfitting e overfitting. O cenário de negócio e o tema da análise vieram do enunciado.

**Equipe**
- Cibelle Sousa Rodrigues
- David Júnio Mariano dos Santos
- Laisa Pereira da França

## Relatório técnico

O relatório completo, com a fundamentação teórica, a matriz de confusão e a discussão de cada cenário, está em [`docs/relatorio-tecnico.pdf`](docs/relatorio-tecnico.pdf).
