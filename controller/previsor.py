import pandas as pd
import pickle
import warnings
warnings.filterwarnings("ignore")
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
        self.tipo = None
    

    def buscaResultado(self): 
        try:
            with open(f'{environment.algoritimos_dir}{environment.resultado_completo_df}_{self.tipo}.pickle', 'rb') as file:
                return pickle.load(file)
        except:
            return print('Não foi possível carregar o resultado_final do algoritimo')

    def carregarModelo(self):
        self.tipo = input('O modelo que quer usar ? Ex: (s_1000) (p_10000) ' )
        resultados_formatados = self.buscaResultado()
      
        print(json.dumps(resultados_formatados, indent=4))
        self.previsoesMetricas()
        for modelo in tqdm.tqdm(resultados_formatados['resultados'].keys(), desc="Carregando modelos"):
            caminho_modelo = environment.algoritimos_dir + modelo + '_' + self.tipo + '.pickle'
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
        print("Dados Carregados \n")
        print(json.dumps(self.X.head(3).to_dict(), indent=4))
        
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
        try:
            with open(f'{environment.algoritimos_dir}{environment.resultado_completo_df}_{self.tipo}.pickle', 'rb') as file:
                resultados_completos = pickle.load(file)
        except FileNotFoundError:
            resultados_completos = {"resultados": {}}

        for nome_modelo, modelo in self.modelos.items():
            if nome_modelo in resultados_completos['resultados']:
                print(f"Resultados do modelo {nome_modelo} já existem. Pulando para o próximo.")
                continue
            barra_progresso.set_description(f"Previsões modelo: {nome_modelo}...")

            self.df[environment.predicao] = modelo.predict(self.X)
            self.df[environment.score] = modelo.predict_proba(self.X)[:, 1]

            self.salvarDataFrame(self.df, nome_modelo, self.tipo)
            barra_progresso.update(1)

        barra_progresso.close()
        self.analise()

        
    def analise(self):
        RESULTADOS = self.buscaResultado()

        relatorio = {}

        # Iterando sobre cada modelo e suas métricas
        for modelo, metricas in tqdm.tqdm(RESULTADOS['resultados'].items(), desc="Processando modelos"):
            df = pd.read_csv(f'{environment.resultado_dir}{modelo}_{self.tipo}.csv', sep=',')
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

    @staticmethod
    def salvarDataFrame(X, nome, tipo):
        resultados_completos = {
            "resultados": nome,
        }
        with open(f'{environment.resultado_dir}{environment.resultado_completo_df}_{tipo}.pickle', 'wb') as file:
            pickle.dump(resultados_completos, file)
        caminho_completo = environment.resultado_dir + nome + '_' + tipo
        df2 = pd.DataFrame(X)
        df2.to_csv(caminho_completo + '.csv' )
    
    def previsoesMetricas(self):
        print('accuracy: Proporção de previsões corretas sobre o total. Mede a eficácia geral do modelo.\n')
        print('precision: Proporção de previsões positivas corretas. Indica a qualidade dos resultados positivos do modelo.\n')
        print('cv_Accuracy: Média da acurácia do modelo em diferentes subconjuntos do conjunto de treinamento. Oferece uma estimativa mais robusta da performance do modelo.\n')
        print('recall: Proporção de casos positivos reais identificados corretamente. Mede a capacidade do modelo de detectar resultados positivos.\n')
        print('f1_score: Média harmônica de precisão e recall. Combina precisão e revocação em uma única métrica, útil quando o equilíbrio entre estas é importante.\n')
        print('roc_auc: Medida de quão bem o modelo distingue entre classes. Quanto maior, melhor o modelo em diferenciar entre positivo e negativo.\n')
        print('confusion_matrix: Tabela que mostra os acertos e erros do modelo comparados com a realidade. Ajuda a entender os tipos de erro e acerto.\n')
