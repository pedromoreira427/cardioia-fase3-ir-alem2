# Relatório comparativo: Regressão Logística x Rede Neuromórfica LIF em batimentos de ECG

**CardioIA Fase 3, Ir Além 2 | Pedro Gustavo França Moreira (RM 568262)**

## 1. Objetivo

Comparar um classificador tradicional (regressão logística) com uma rede neuromórfica simples (neurônios *Leaky Integrate-and-Fire*, LIF) na tarefa de classificar batimentos cardíacos em normal ou anormal, olhando desempenho, custo e o que cada abordagem permite fazer num produto como o wearable do CardioIA.

## 2. Dados e preparação

- **Base:** PTB Diagnostic ECG Database (mesma do Ir Além 2 da Fase 2), baixada de um espelho público no GitHub porque o Kaggle costuma ser bloqueado. São 14.552 batimentos: 4.046 normais e 10.506 anormais. Cada batimento tem 187 amostras normalizadas entre 0 e 1.
- **Divisão:** 70% treino (10.186), 15% validação (2.183) e 15% teste (2.183), estratificada, semente 42. A validação foi usada só para escolher a melhor época da rede LIF. O teste foi usado uma vez, no final, para os dois modelos.
- **Desbalanceamento:** os dois modelos usam peso de classe balanceado. A métrica principal, além da acurácia, é o **recall da classe anormal**, porque deixar passar um batimento anormal é o erro mais caro em triagem.

## 3. Métodos

**Regressão logística.** `StandardScaler` seguido de `LogisticRegression` (scikit-learn), sobre os 187 valores brutos. Cada posição do sinal recebe um peso fixo e o modelo traça uma fronteira linear. Variei a regularização C de 0,01 a 100 na validação e a acurácia ficou entre 76,8% e 78,4%, então mantive C = 1.

**Codificação em spikes.** Chips neuromórficos trabalham com eventos. Usei codificação por cruzamento de nível: 16 níveis de amplitude, cada um com um canal "sobe" e um "desce", totalizando 32 canais. Um canal dispara quando o sinal cruza seu nível naquela direção. Em média, um batimento gera 76 spikes de entrada, o que ocupa só 1,3% das 5.984 posições possíveis (187 passos x 32 canais).

**Rede LIF.** 32 canais de entrada, 64 neurônios LIF e uma leitura linear para 2 classes. Cada neurônio segue `v[t] = β·v[t-1] + I[t] - θ·s[t-1]` e dispara quando `v ≥ θ`. O β de cada neurônio foi sorteado entre 0,85 e 0,98 (memórias de durações diferentes) e não é treinado. A leitura conta os spikes de cada neurônio em 8 janelas de tempo, para a posição do evento dentro do batimento contar. O treino usa **gradiente substituto** (o spike não tem derivada; no passo para trás ele é trocado por uma curva suave em volta do limiar) com retropropagação no tempo e Adam, tudo escrito com NumPy, porque o TensorFlow não instala na minha máquina. Foram 30 épocas, guardando os pesos da época com melhor F1 na validação (época 20).

**Controle.** Para saber se o ganho vem do neurônio ou só da codificação, treinei a mesma regressão logística sobre a contagem de spikes de entrada por janela (256 atributos).

## 4. Resultados (base de teste, 2.183 batimentos, 1.576 anormais)

| Métrica | Regressão Logística | LR sobre os spikes (controle) | Rede LIF |
|---|---|---|---|
| Acurácia | 78,3% | 81,4% | **91,0%** |
| Recall anormal | 78,4% | 79,5% | **92,8%** |
| Precisão anormal | 90,3% | 93,8% | **94,6%** |
| F1 anormal | 0,839 | 0,861 | **0,937** |
| Parâmetros treináveis | 188 | 257 | 3.138 |
| Tempo de treino | 0,4 a 0,7 s | não medido | 47 a 65 s |
| Inferência (1000 batimentos, CPU) | 0,8 a 1,7 ms | não medido | 101 a 155 ms |

As métricas são idênticas em todas as execuções do notebook (semente fixa). Os tempos variam com a carga do computador; a tabela mostra a faixa observada em quatro execuções.

Em números absolutos: dos 1.576 batimentos anormais do teste, a regressão logística deixou passar 341 e a rede LIF, 113.

**Custo por batimento.** A logística faz cerca de 562 operações. A LIF simulada em CPU faz a conta em todos os 187 passos, cerca de 420 mil operações. Se só os eventos custassem (como num chip neuromórfico), a estimativa cai para cerca de 19,5 mil. Na camada oculta, 11,1% dos neurônios disparam por passo, em média.

## 5. Discussão

**Por que a LIF foi melhor.** A logística olha cada posição separadamente. Batimentos têm duração e alinhamento um pouco diferentes, então o mesmo pico pode cair na posição 60 num e na 65 em outro, e para ela são coisas diferentes. O neurônio LIF integra a entrada no tempo e esquece aos poucos: responde ao formato e ao ritmo da subida e da descida, e os β diferentes funcionam como janelas de memória de tamanhos variados. Além disso o disparo é não linear, então a rede não fica presa a uma fronteira reta.

**O que o controle mostrou.** A codificação em eventos sozinha levou a logística de 78,3% para 81,4%. O resto do ganho, quase 10 pontos, vem da dinâmica dos neurônios. Sem esse controle eu poderia atribuir tudo à "rede neuromórfica" quando parte era só a forma de representar o sinal.

**Eficiência: a promessa depende do hardware.** Na CPU a LIF foi cerca de 100 vezes mais lenta na inferência e treinou em cerca de 1 minuto contra menos de 1 segundo. Isso confirma o que o capítulo 3 mostra com o estudo de Freitas et al.: SNN rodando em arquitetura Von Neumann não traz economia de energia, porque a CPU processa todos os passos, com ou sem evento. A vantagem aparece em chips orientados a eventos como o Intel Loihi e o IBM TrueNorth. A esparsidade medida (1,3% de ocupação na entrada, 11,1% na camada oculta) é justamente o que esses chips aproveitam. A estimativa de 19,5 mil operações por evento é um modelo simples de contagem, não uma medição de energia.

**Interpretabilidade.** Os 188 pesos da logística mostram quanto cada instante do batimento puxa para anormal. Na LIF dá para ver o raster de spikes, mas explicar uma decisão é bem mais difícil. Em saúde isso pesa.

## 6. Vantagens e limitações

| | Regressão Logística | Rede LIF |
|---|---|---|
| Vantagens | minúscula, rápida, estável, explicável, cabe no ESP32 | capta padrões temporais, trabalha com eventos esparsos, muito mais precisa aqui |
| Limitações | fronteira linear, sem noção de tempo, erra 1 em cada 5 anormais | treino difícil (gradiente substituto e BPTT à mão), mais hiperparâmetros, validação oscila entre épocas, lenta sem hardware neuromórfico, menos explicável |

Usei LIF e não FitzHugh-Nagumo porque o LIF tem uma variável de estado e conta barata. O FHN tem duas variáveis e um termo cúbico: imita melhor o potencial de ação biológico, mas isso não ajuda a classificar e deixaria a simulação mais cara.

**Cuidados com os números:**

- O PTB vem em batimentos soltos, sem identificação do paciente. Batimentos da mesma pessoa provavelmente estão no treino e no teste, o que deixa os dois modelos otimistas. O correto seria separar por paciente.
- Uma divisão e uma semente só, sem validação cruzada e sem intervalo de confiança. A diferença de 12,7 pontos de acurácia é grande, mas o valor exato mudaria com outra divisão.
- Ajuste limitado da LIF: testei poucas combinações na validação (16 ou 24 níveis, 64 ou 128 neurônios, 4 ou 8 janelas).
- Não é o melhor resultado possível nessa base: a MLP sobre imagens da Fase 2 chegou a 97,5% de acurácia. O objetivo aqui era comparar as duas abordagens pedidas, não bater recorde.

## 7. Conclusão

Para o wearable do CardioIA hoje, com ESP32, a regressão logística é a opção realista na borda: roda no microcontrolador, responde na hora e o alerta é explicável. A rede LIF foi claramente superior em detectar batimentos anormais (92,8% contra 78,4% de recall) e trabalha com o tipo de dado esparso que um chip neuromórfico processa com pouca energia, mas só vale a pena de verdade com esse hardware. Uma arquitetura híbrida faz sentido: triagem simples na borda e o modelo temporal mais forte na névoa ou na nuvem, revendo os casos que a borda marcar como duvidosos.
