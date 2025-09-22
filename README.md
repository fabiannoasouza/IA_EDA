<div align="center">

# Análise Preditiva da Doença de Alzheimer 

*Um projeto de Machine Learning para explorar e prever o diagnóstico da Doença de Alzheimer a partir de dados clínicos.*

</div>

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![Kaggle](https://img.shields.io/badge/Dataset-Kaggle-blue?style=for-the-badge&logo=kaggle)](https://www.kaggle.com/datasets/sagnikseal/alzheimers-disease-dataset)
[![Status](https://img.shields.io/badge/Status-Em%20Desenvolvimento-yellow?style=for-the-badge)]()
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](https://opensource.org/licenses/MIT)

</div>

---

### Descrição do Processo

O processo que vamos seguir se divide em duas grandes partes, exatamente como você planejou: a **Análise Exploratória de Dados (EDA)** e a **Modelagem Preditiva**. A EDA é como um trabalho de detetive, onde vamos investigar os dados para entender suas características antes de treinar qualquer modelo.

Para organizar nosso trabalho, podemos seguir estes passos:

1.  **Configuração**: Preparar o ambiente no Colab e carregar o dataset.
2.  **Análise Exploratória (EDA)**: Criar e interpretar os 4 gráficos que você precisa.
3.  **Pré-processamento**: Preparar os dados para que os modelos possam aprender com eles.
4.  **Treinamento e Avaliação**: Aplicar os 5 modelos e gerar as métricas de desempenho.

### Carregando o Dataset no Google Colab

Vamos carregar o arquivo para o ambiente do Colab e transformá-lo em uma tabela (que em Python chamamos de DataFrame) para podermos trabalhar.

* **Dataset**: `alzheimers_disease_dataset.csv`

#### Código em Python:
```python
import pandas as pd

# Define o caminho do arquivo dentro do seu Google Drive
path = '/content/drive/MyDrive/alzheimers_disease_dataset.csv'

# Carrega os dados para o dataframe 'df'
df = pd.read_csv(path)

# Mostra as 5 primeiras linhas
print("DataFrame carregado com sucesso!")
display(df.head())
```
---

### Análise Exploratória (EDA) - Primeira Olhada

Agora que os dados estão carregados, começa a parte divertida da investigação.

> O primeiro passo de qualquer análise é obter uma "visão panorâmica" do dataset. Queremos responder a perguntas como:

* **Estrutura**: Quantas linhas e colunas temos?
* **Dados Faltantes**: Há dados faltando em alguma coluna?
* **Tipos de Dados**: Quais são os tipos de dados (números, texto, etc.)?
* **Estatísticas**: Qual a média, o valor máximo e mínimo das colunas numéricas?

O `pandas` tem duas funções mágicas para isso: `.info()` e `.describe()`. Elas nos dão um resumo técnico e estatístico completo com apenas uma linha de código cada.

#### Código em Python:
```python
# Mostra um resumo técnico do dataframe
print("--- Informações Gerais (Tipos e Nulos) ---")
df.info()

print("\n\n--- Resumo Estatístico das Colunas Numéricas ---")
display(df.describe())
```

### Interpretando os Resultados Iniciais

#### O que observar:
1.  **Na tabela do `df.info()`**: Olhe para a coluna `"Non-Null Count"` (Contagem não nula). Se algum número aqui for menor que o total de entradas (que é mostrado na primeira linha), significa que temos **dados faltando** naquela coluna. Veja também a coluna `"Dtype"`, que mostra o tipo de cada dado.
2.  **Na tabela do `df.describe()`**: Esta tabela te dá um resumo rápido dos dados numéricos. Por exemplo, você pode ver a idade (`Age`) média dos pacientes, o maior e o menor Índice de Massa Corporal (`BMI`), e assim por diante.

 **O que `Dtypes: float64(12), int64(22), object(1)` nos diz?**
> * Temos **34 colunas** com dados numéricos (`float` para decimais e `int` para inteiros).
> * Temos **1 coluna** do tipo `object`, que o pandas usa para armazenar texto.

Isso é um ótimo sinal, pois modelos de machine learning trabalham primariamente com números.

Agora, vamos focar em dois pontos críticos da análise:

1.  **Dados Faltando**: No resultado do `df.info()`, a contagem de "non-null" (não nulos) foi a mesma para todas as colunas? Se sim, ótimo! Se não, precisamos decidir o que fazer com eles.
2.  **A Coluna de Texto (`object`)**: Precisamos investigar o que há nessa coluna para decidir se ela é útil para a nossa análise ou se devemos removê-la.

#### Código em Python:
```python
# 1. Verifica se há algum valor nulo em qualquer coluna do dataframe
print("--- Verificação de Dados Faltando ---")
print(df.isnull().sum())

# 2. Identifica o nome da coluna de texto e mostra seus valores únicos
coluna_texto = df.select_dtypes(include=['object']).columns[0]
print(f"\n\n--- Análise da Coluna de Texto: '{coluna_texto}' ---")
print("Valores únicos nesta coluna:")
print(df[coluna_texto].unique())
```
### Análise da Verificação e Ação

#### O que este código fez?
1.  O comando `df.isnull().sum()` contou quantos valores nulos (vazios) existem em **cada coluna**. O resultado "0" para todas é a confirmação que esperávamos.
2.  O restante do código encontrou automaticamente qual era a coluna do tipo `object` e nos mostrou todos os valores únicos dentro dela para entendermos do que se trata.

#### Conclusões:
* **1. Nenhum dado nulo**: Isso é excelente! Significa que não precisamos nos preocupar em preencher ou remover dados faltantes. Nosso dataset está limpo nesse aspecto.
* **2. Coluna `DoctorInCharge`**: A mensagem `['XXXConfid']` significa que a única informação presente em todas as linhas dessa coluna é o texto "XXXConfid".

> **O que isso significa para a nossa análise?**

Imagine que você está tentando prever se vai chover e uma das suas informações é "o céu é azul". Se o céu for **sempre** azul em todos os seus registros (dias com e sem chuva), essa informação não ajuda em nada a sua previsão, certo?

> A coluna `DoctorInCharge` é a mesma coisa. Como ela tem o mesmo valor para todos os pacientes, ela **não tem poder preditivo** e não nos ajuda a diferenciar um paciente com Alzheimer de um sem.

Portanto, a melhor coisa a fazer é simplesmente **removê-la**.

#### Ação: Removendo a Coluna
O código abaixo executa a remoção da coluna `DoctorInCharge`.

```python
# Remove a coluna 'DoctorInCharge' do dataframe 'df'
df = df.drop(columns=['DoctorInCharge'])

# Confirma que a coluna foi removida, mostrando as 5 primeiras linhas novamente
print("Coluna 'DoctorInCharge' removida com sucesso!")
display(df.head())
```

---

### Análise Exploratória Visual (EDA)

Com os dados limpos, podemos finalmente começar a criar nossos gráficos!

O gráfico mais importante para começar é o da nossa variável-alvo: a coluna `Diagnosis`. Precisamos saber quantas pessoas no nosso dataset foram diagnosticadas com Alzheimer e quantas não foram. Isso nos diz se o nosso problema está **"balanceado"**.

#### Gráfico 1: Distribuição dos Diagnósticos
Vamos criar um gráfico de barras para visualizar a contagem de cada diagnóstico.

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Configura o estilo dos gráficos para uma melhor visualização
sns.set_style("whitegrid")

# Cria uma figura e eixos para o nosso gráfico
plt.figure(figsize=(8, 6))

# Usa a função countplot do seaborn para criar o gráfico de barras
# A variável 'x' é a coluna que queremos contar
ax = sns.countplot(x='Diagnosis', data=df, palette='viridis')

# Adiciona um título e rótulos para os eixos
plt.title('Gráfico 1: Distribuição dos Diagnósticos', fontsize=16)
plt.xlabel('Diagnóstico (0 = Não Alzheimer, 1 = Alzheimer)', fontsize=12)
plt.ylabel('Quantidade de Pacientes', fontsize=12)

# Exibe o gráfico
plt.show()
```
<br>
<div align="center">
<img width="608" height="455" alt="Image" src="https://github.com/user-attachments/assets/ee2c7218-0c26-4ecc-a43b-5cc672cf6a76" />
</div>
<br><br>

**Insight Importante: Dataset Desbalanceado**

A visualização anterior confirma que nosso dataset é **desbalanceado**. Isso é muito importante porque, se não tomarmos cuidado, nosso modelo de machine learning poderia ficar "preguiçoso". Ele poderia aprender a chutar sempre "Não Alzheimer" e ainda assim ter uma acurácia alta, simplesmente porque essa é a maioria. Vamos manter isso em mente para a etapa de treinamento dos modelos.

#### Gráfico 2: Distribuição de Idade por Diagnóstico
Agora que entendemos a nossa variável alvo (`Diagnosis`), vamos investigar as outras características. Uma das mais importantes, quando se fala em Alzheimer, é a idade (`Age`).

**Pergunta de investigação:** Será que a distribuição de idade é visivelmente diferente entre o grupo com diagnóstico e o grupo sem diagnóstico?

> Para responder a isso, um **gráfico de densidade (KDE)** é perfeito. Ele é como um histograma "suavizado" e nos permite comparar a forma das distribuições de idade para os dois grupos no mesmo gráfico.

```python
# Cria a figura para o gráfico
plt.figure(figsize=(10, 6))

# Usa a função kdeplot do seaborn para criar o gráfico de densidade
# 'hue' separa o gráfico por diagnóstico, criando uma curva para cada grupo
sns.kdeplot(data=df, x='Age', hue='Diagnosis', fill=True, common_norm=False, palette='plasma')

# Adiciona título e rótulos
plt.title('Gráfico 2: Distribuição de Idade por Diagnóstico', fontsize=16)
plt.xlabel('Idade', fontsize=12)
plt.ylabel('Densidade', fontsize=12)
plt.legend(title='Diagnóstico', labels=['Alzheimer', 'Não Alzheimer']) # Ajusta a legenda

# Exibe o gráfico
plt.show()
```
<br>
<div align="center">
<img width="667" height="455" alt="Image" src="https://github.com/user-attachments/assets/ff898dcb-3f99-41f5-a5fd-4c52f62d8ce7" />
</div><br><br>
**Analisando as Curvas:**

Observe novamente o gráfico:
* O pico da curva dos pacientes sem Alzheimer e o pico da curva dos pacientes com Alzheimer estão exatamente no mesmo ponto de idade?
* Ou o pico de um dos grupos está visivelmente deslocado para a direita (idades mais avançadas)?

O que acabamos de observar é:

* O grupo sem Alzheimer tem um pico mais forte e concentrado em uma idade mais jovem (por volta dos 65-75 anos).
* O grupo com Alzheimer é mais "distribuído", o que significa que ele está mais espalhado por idades mais avançadas. O centro dessa distribuição está claramente mais à direita (mais velho) que o outro grupo.

> **Conclusão Chave:** Acabamos de confirmar visualmente que, neste dataset, a **idade avançada está associada ao diagnóstico de Alzheimer.** Ótima análise!

#### Gráfico 3: Boxplot
Agora que vimos um fator demográfico (Idade), que tal analisarmos um dado clínico? A coluna MMSE (Mini-Mental State Examination) é a pontuação de um famoso teste cognitivo.

Nossa hipótese: Pacientes com Alzheimer devem ter, em média, uma pontuação MMSE mais baixa do que pacientes saudáveis.

Para visualizar isso, um Box Plot é a ferramenta ideal. Ele vai nos mostrar a mediana, a variação e os valores extremos das pontuações para os dois grupos, lado a lado.

> Vamos criar esse gráfico
```python
# Cria a figura para o gráfico
plt.figure(figsize=(8, 7))

# Usa a função boxplot do seaborn
sns.boxplot(x='Diagnosis', y='MMSE', data=df)

# Adiciona título e rótulos
plt.title('Gráfico 3: Box Plot do Score MMSE por Diagnóstico', fontsize=16)
plt.xlabel('Diagnóstico (0 = Não Alzheimer, 1 = Alzheimer)', fontsize=12)
plt.ylabel('Pontuação no Teste MMSE', fontsize=12)

# Exibe o gráfico
plt.show()
```
<br>
<div align="center">
<img width="590" height="532" alt="Image" src="https://github.com/user-attachments/assets/511c5cda-1872-49f0-91cc-9c9960e48fe6" />
</div>
<br><br>
  
> Como ler este gráfico:
* A linha no meio de cada caixa é a mediana (o valor do meio).
* A caixa em si representa 50% dos pacientes, mostrando a faixa de pontuação mais comum.
* As linhas verticais (ou "bigodes") mostram o alcance geral dos dados.

A caixa do grupo "Alzheimer" (1) está visivelmente mais abaixo, o que significa que as pontuações MMSE para esse grupo são, em geral, bem mais baixas.

> Conclusão: Confirmamos nossa hipótese. Uma pontuação baixa no teste MMSE está fortemente associada ao diagnóstico de Alzheimer neste dataset. Esta será, muito provavelmente, uma das informações mais importantes para os nossos futuros modelos de previsão.

** Pergunta de investigação:** Como a pontuação cognitiva (MMSE) se comporta com o avanço da Idade? Será que existe um padrão diferente para quem tem e quem não tem o diagnóstico?

Um gráfico de dispersão (scatter plot) é ideal para isso. Cada paciente será um ponto no gráfico, posicionado de acordo com sua idade e sua nota no teste. A cor do ponto nos dirá se ele tem o diagnóstico ou não.

#### Gráfico 4: Gráfico de Dispersão
Este gráfico de dispersão é excelente para "caçar" padrões entre três variáveis de uma só vez (Idade, MMSE e Diagnóstico).

```python
# Cria a figura para o gráfico
plt.figure(figsize=(10, 7))

# Usa a função scatterplot do seaborn
# 'hue' novamente colore os pontos de acordo com o diagnóstico
sns.scatterplot(data=df, x='Age', y='MMSE', hue='Diagnosis', alpha=0.7)

# Adiciona título e rótulos
plt.title('Gráfico 4: Relação entre Idade, MMSE e Diagnóstico', fontsize=16)
plt.xlabel('Idade', fontsize=12)
plt.ylabel('Pontuação no Teste MMSE', fontsize=12)

# Exibe o gráfico
plt.show()
```
<br>
<div align="center">
<img width="645" height="432" alt="Image" src="https://github.com/user-attachments/assets/6d9fffa0-3613-4dc9-ba13-6a598f8d9b8b" />
</div>
<br><br>
## Resumo da Análise Exploratória (EDA)
> Concluímos a primeira grande etapa do projeto! Com estes 4 gráficos, descobrimos informações valiosas:

* Nosso dataset é desbalanceado (temos mais casos de não-Alzheimer).
* A idade avançada está associada ao diagnóstico.
* Uma pontuação MMSE baixa é um forte indicador da doença.
* A combinação de idade e MMSE nos ajuda a separar visualmente os dois grupos.
