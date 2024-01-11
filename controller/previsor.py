import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import pickle
import json
import numpy as np
import tqdm
import config.environment as environment 
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os

class Previsor:
    def __init__(self):
        self.modelos = {} 
        self.modelo = None
        self.X = None
        self.df = None
        self.df = None
        self.scaler = None
        self.modelo_escolhido = None
        self.tipo = None
        self.diretorio = None
    

    def buscaResultadoAlgoritimo(self, resultado=False): 
        try:
            with open(f'{environment.algoritimos_dir}{self.tipo}/{environment.resultado_completo_df}.pickle', 'rb') as file:
                dados =  pickle.load(file)
                return dados
        except:
            return print('Não foi possível carregar o resultado_final do algoritimo')
        
    def verificaDiretorio(self):
        self.diretorio =  f'{environment.resultado_dir}{self.tipo}/'
        if not os.path.exists(self.diretorio):
            os.makedirs(self.diretorio)

    def verificaDiretorioScore(self):
        self.diretorio =  f'{environment.score_dir}{self.tipo}/'
        if not os.path.exists(self.diretorio):
            os.makedirs(self.diretorio)

    def carregarModelo(self):
        self.tipo = input('O modelo que quer usar ? Ex: (s_10k) (p_100k) ' )
        resultados_formatados = self.buscaResultadoAlgoritimo()
        self.verificaDiretorio()
        print(json.dumps(resultados_formatados, indent=4))
        self.previsoesMetricas()
        for modelo in tqdm.tqdm(resultados_formatados['resultados'].keys(), desc="Carregando modelos"):
            caminho_modelo = f"{environment.algoritimos_dir}{self.tipo}/{modelo}.pickle"
            with open(caminho_modelo, 'rb') as file:
                self.modelos[modelo] = pickle.load(file)

        self.carregarDados()
        self.adicionarPredicoesAoDataFrame()

    def carregarDados(self):
        with open(environment.teste_dir + environment.previsor_utilizado, 'rb') as file:
            self.X = pickle.load(file)

        with open(environment.teste_dir + environment.previsores, 'rb') as file:
            self.df = pickle.load(file)
        self.df = pd.DataFrame(self.df)
        
    def prever(self):
        if self.modelo is None:
            raise ValueError("Nenhum modelo foi carregado.")
        return self.modelo.predict(self.X)

    def preverProba(self):
        if self.modelo is None:
            raise ValueError("Nenhum modelo foi carregado.")
        return self.modelo.predict_proba(self.X)[:, 1]

    def adicionarPredicoesAoDataFrame(self):
        barra_progresso = tqdm.tqdm(total=len(self.modelos), desc="Iniciando previsões", unit="modelo")

        for nome_modelo, modelo in self.modelos.items():
            barra_progresso.set_description(f"Previsões modelo: {nome_modelo}...")

            self.df[environment.predicao] = modelo.predict(self.X)
            self.df[environment.score] = modelo.predict_proba(self.X)[:, 1]

            self.salvarDataFrame(nome_modelo)
            barra_progresso.update(1)
            self.rodarTudoEjogarAsImagensNaPasta(nome_modelo)
        barra_progresso.close()
        self.analise()

        
    def analise(self):
        RESULTADOS = self.buscaResultadoAlgoritimo(True)

        relatorio = {}
        try:
            with open(f'{self.diretorio}{environment.resultado_completo_df}.pickle', 'rb') as file:
                resultados_completos = pickle.load(file)
        except FileNotFoundError:
            resultados_completos = {"resultados": {}}
        # Iterando sobre cada modelo e suas métricas
        for modelo, metricas in tqdm.tqdm(RESULTADOS['resultados'].items(), desc="Processando modelos"):
            if modelo in resultados_completos['resultados']:
                print(f"Resultados do modelo {modelo} já existem. Pulando para o próximo.")
                continue
            df = pd.read_csv(f'{self.diretorio}{modelo}_{self.tipo}.csv', sep=',')
            df.drop('Unnamed: 0', axis=1, inplace=True)
            # Obtendo as primeiras previsões e as previsões com score maior que 0.3
            primeiras_previsoes = df.head(3)
            previsoes_positivas = df.query('predicao > 0').head(3)
            scores_altos = df.query('score > 0.3').sort_values('score', ascending=False).head(3)

            relatorio[modelo] = {
                'metricas': metricas,
                'primeiras_previsoes': primeiras_previsoes,
                'previsoes_positivas': previsoes_positivas,
                'scores_altos': scores_altos
            }
            
        def imprimir_dataframe(df):
            # Calculando a largura máxima para cada coluna
            larguras_colunas = []
            for col in df.columns:
                if df[col].empty:
                    larguras_colunas.append(len(col))
                else:
                    larguras_colunas.append(max([len(str(x)) for x in df[col]]))
            
            
            # Formatando e imprimindo o cabeçalho
            cabecalho = " | ".join([f"{col:<{larguras_colunas[i]}}" for i, col in enumerate(df.columns)])
            print(cabecalho)
            print("-" * len(cabecalho))

            # Formatando e imprimindo cada linha
            for _, row in df.iterrows():
                linha_formatada = " | ".join([f"{str(valor):<{larguras_colunas[i]}}" for i, valor in enumerate(row)])
                print(linha_formatada)

        # Exemplo de uso:
        for modelo, info in relatorio.items():
            print(f"Modelo: {modelo}")
            print("Métricas:")
            for metrica, valor in info['metricas'].items():
                print(f"{metrica}: {valor}")

            print("\nPrimeiras Previsões:")
            imprimir_dataframe(info['primeiras_previsoes'])

            print("\nPrevisões Positivas:")
            imprimir_dataframe(info['previsoes_positivas'])

            print("\nScores Altos:")
            imprimir_dataframe(info['scores_altos'])
            print("\n")

    def salvarDataFrame(self, nome):
        resultados_completos = {
            "resultados": nome,
        }
        with open(f'{self.diretorio}{environment.resultado_completo_df}.pickle', 'wb') as file:
            pickle.dump(resultados_completos, file)
        caminho_completo = self.diretorio + nome
        df2 = pd.DataFrame(self.X)
        df2.to_csv(caminho_completo + '.csv' )

    def previsoesMetricas(self):
        print('accuracy: Proporção de previsões corretas sobre o total. Mede a eficácia geral do modelo.\n')
        print('precision: Proporção de previsões positivas corretas. Indica a qualidade dos resultados positivos do modelo.\n')
        print('cv_Accuracy: Média da acurácia do modelo em diferentes subconjuntos do conjunto de treinamento. Oferece uma estimativa mais robusta da performance do modelo.\n')
        print('recall: Proporção de casos positivos reais identificados corretamente. Mede a capacidade do modelo de detectar resultados positivos.\n')
        print('f1_score: Média harmônica de precisão e recall. Combina precisão e revocação em uma única métrica, útil quando o equilíbrio entre estas é importante.\n')
        print('roc_auc: Medida de quão bem o modelo distingue entre classes. Quanto maior, melhor o modelo em diferenciar entre positivo e negativo.\n')
        print('confusion_matrix: Tabela que mostra os acertos e erros do modelo comparados com a realidade. Ajuda a entender os tipos de erro e acerto.\n')

    def rodarTudoEjogarAsImagensNaPasta(self, nome):
     
        score_boa_vista = pd.read_csv(f'{environment.base_dir}score.txt', sep=';')
        
        modelo_treinado = pd.read_csv(f'{self.diretorio}{nome}.csv')

        if isinstance(modelo_treinado, dict):
            # Converta o dicionário em DataFrame, se necessário
            modelo_treinado = pd.DataFrame([modelo_treinado])

        if len(modelo_treinado) == len(score_boa_vista):
            modelo_treinado['ScoreBVS'] = score_boa_vista['ScoreBVS'] * 0.2
            modelo_treinado['ID_VINCULO'] = score_boa_vista['ID_VINCULO']
            modelo_treinado['score'] = modelo_treinado['score'] * 100
            modelo_treinado['pagou'] = score_boa_vista['dt_rec'].apply(lambda x: 0 if pd.isna(x) else 1)
            modelo_treinado.rename(columns={'score': 'MunaScore'}, inplace=True)
            modelo_treinado = modelo_treinado[['ID_VINCULO', 'co_ope', 'dt_base', 'vlr_base', 'dt_nasc', 'sexo', 'renda', 'renda_liq', 'dt_relac', 'co_cart', 'co_prof', 'cep', 'ScoreBVS', 'predicao', 'MunaScore', 'pagou']]
        else:
            print("Os DataFrames não têm o mesmo número de linhas e não podem ser combinados diretamente.")
        
        self.montaGraficoComParametro(modelo_treinado)

    def montaGraficoComParametro(self, modelo_treinado):
        self.verificaDiretorioScore()
        # Função auxiliar para formatar o eixo y e os rótulos das barras
        def k_m_formatter(x, pos):
            if x >= 1000000:  # Se o valor for igual ou maior que 1 milhão
                return f'{int(x/1000000)}m'
            else:  # Se o valor for menor que 1 milhão
                return f'{int(x/1000)}k'

        # [Supondo que você já tenha o DataFrame 'modelo_treinado' carregado aqui]
        print(modelo_treinado.columns) 

        # Definindo as faixas de score
        bins = [0, 24.99, 49.99, 74.99, 100]

        # Criando as faixas de score para 'ScoreBVS' e 'MunaScore'
        modelo_treinado['faixa_ScoreBVS'] = pd.cut(modelo_treinado['ScoreBVS'], bins, labels=["0-25", "25-50", "50-75", "75-100"])
        modelo_treinado['faixa_MunaScore'] = pd.cut(modelo_treinado['MunaScore'], bins, labels=["0-25", "25-50", "50-75", "75-100"])

        # Filtrar o DataFrame para incluir quem pagou e quem não pagou
        modelo_treinado_pagou = modelo_treinado[modelo_treinado['pagou'] == 1]
        modelo_treinado_nao_pagou = modelo_treinado[modelo_treinado['pagou'] == 0]

        # Agrupando e contando por faixas para quem pagou e não pagou
        contagem_ScoreBVS_pagou = modelo_treinado_pagou.groupby(['faixa_ScoreBVS']).size()
        contagem_MunaScore_pagou = modelo_treinado_pagou.groupby(['faixa_MunaScore']).size()
        contagem_ScoreBVS_nao_pagou = modelo_treinado_nao_pagou.groupby(['faixa_ScoreBVS']).size()
        contagem_MunaScore_nao_pagou = modelo_treinado_nao_pagou.groupby(['faixa_MunaScore']).size()

        # Criação das barras agrupadas
        labels = ["0-25", "25-50", "50-75", "75-100"]  # Rótulos das faixas
        x = np.arange(len(labels))  # Localizações dos grupos
        width = 0.35  # Largura das barras

        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 5))

        # Barras para 'ScoreBVS'
        bars_pagou = axes[0].bar(x - width/2, contagem_ScoreBVS_pagou, width, label='Pagou', color='lightblue')
        bars_nao_pagou = axes[0].bar(x + width/2, contagem_ScoreBVS_nao_pagou, width, label='Não Pagou', color='lightcoral')

        # Barras para 'MunaScore'
        bars_pagou_2 = axes[1].bar(x - width/2, contagem_MunaScore_pagou, width, label='Pagou', color='lightblue')
        bars_nao_pagou_2 = axes[1].bar(x + width/2, contagem_MunaScore_nao_pagou, width, label='Não Pagou', color='lightcoral')

        # Função para adicionar rótulos
        def add_bar_labels(bars, ax):
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{k_m_formatter(height, None)}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')

        # Adicionando rótulos de valor nas barras
        add_bar_labels(bars_pagou, axes[0])
        add_bar_labels(bars_nao_pagou, axes[0])
        add_bar_labels(bars_pagou_2, axes[1])
        add_bar_labels(bars_nao_pagou_2, axes[1])

        # Adicionando detalhes aos gráficos
        axes[0].set_title('ScoreBVS')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(labels)
        axes[0].legend()

        axes[1].set_title('MunaScore')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(labels)
        axes[1].legend()
        
        # Formatação do eixo y para ambos os gráficos
        axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(k_m_formatter))
        axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(k_m_formatter))

        # Ajustar layout
        plt.tight_layout()
        # Salvar a figura
        if not os.path.exists(f'{environment.score_dir}{self.tipo}'):
             os.makedirs(f'{environment.score_dir}{self.tipo}')
        plt.savefig( f'{environment.score_dir}{self.tipo}/{self.modelo}.png', format='png')
        plt.show()

