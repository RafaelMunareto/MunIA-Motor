import pandas as pd
import pickle

import tqdm
import config.environment as environment 
from sklearn.preprocessing import StandardScaler
import json

class Previsor:
    def __init__(self):
        self.modelos = {} 
        self.modelo = None
        self.X = None
        self.df = None
        self.df = None
        self.scaler = None
        self.modelo_escolhido = None
    

    def carregarModelo(self):
        with open(f'{environment.algoritimos_dir}{environment.resultado_completo_df}', 'rb') as file:
            resultados_formatados = pickle.load(file)
        print(json.dumps(resultados_formatados, indent=4))

        for modelo in tqdm.tqdm(resultados_formatados['resultados'].keys(), desc="Carregando modelos"):
            caminho_modelo = environment.algoritimos_dir + modelo + '.pickle'
            with open(caminho_modelo, 'rb') as file:
                self.modelos[modelo] = pickle.load(file)

        print("Modelos Carregados")
        self.carregarDados()
        self.adicionarPredicoesAoDataFrame()

    def carregarDados(self):
        with open(environment.teste_dir + environment.previsor_utilizado, 'rb') as file:
            self.X = pickle.load(file)

        with open(environment.teste_dir + environment.previsores, 'rb') as file:
            self.df = pickle.load(file)
        self.df = pd.DataFrame(self.df)
        print("Dados Carregados \n")
        print(json.dumps(self.X.head().to_dict(), indent=4))
        
    def prever(self):
        if self.modelo is None:
            raise ValueError("Nenhum modelo foi carregado.")
        return self.modelo.predict(self.X)

    def preverProba(self):
        if self.modelo is None:
            raise ValueError("Nenhum modelo foi carregado.")
        return self.modelo.predict_proba(self.X)[:, 1]

    def adicionarPredicoesAoDataFrame(self):
        for nome_modelo, modelo in tqdm.tqdm(self.modelos.items(), desc="Executando previsões"):
            print(f"Executando previsões para o modelo: {nome_modelo}")
            self.df[environment.predicao] = modelo.predict(self.X)
            if hasattr(modelo, 'predict_proba'):
                self.df[environment.score] = modelo.predict_proba(self.X)[:, 1]

            self.salvarDataFrame(self.df, nome_modelo)

        self.analise()

    def analise(selv):
        # Carregando os resultados dos algoritmos
        with open(environment.algoritimos_dir + environment.resultado_completo_df, 'rb') as file:
            RESULTADOS = pickle.load(file)

        relatorio = {}

        # Iterando sobre cada modelo e suas métricas
        for modelo, metricas in tqdm(RESULTADOS['resultados'].items(), desc="Processando modelos"):
            df = pd.read_csv(f'{environment.resultado_dir}{modelo}.csv', sep=',')
            df.drop('Unnamed: 0', axis=1, inplace=True)
            # Obtendo as primeiras previsões e as previsões com score maior que 0.3
            primeiras_previsoes = df.head(3)
            previsoes_positivas = df.query('predicao > 0').head(5)
            scores_altos = df.query('score > 0.3').sort_values('score', ascending=False).head(3)

            relatorio[modelo] = {
                'metricas': metricas,
                'primeiras_previsoes': primeiras_previsoes,
                'previsoes_positivas': previsoes_positivas,
                'scores_altos': scores_altos
            }
            
        def imprimir_dataframe(df):
            # Calculando a largura máxima para cada coluna
            larguras_colunas = [max([len(str(x)) for x in df[col]]) for col in df.columns]
            
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

    @staticmethod
    def salvarDataFrame(X, nome):
        caminho_completo = environment.resultado_dir + nome
        df2 = pd.DataFrame(X)
        df2.to_csv(caminho_completo + '.csv' )
        print(f"DataFrame salvo com sucesso em {caminho_completo}.csv")
        
   