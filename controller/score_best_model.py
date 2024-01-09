import json
import pickle
import pandas as pd
import config.environment as environment 
import tqdm

class ScoreBestModel:
    def __init__(self):
        self.df = None
        self.modelo = None
        self.results = None
        self.bm = None

    def juntarComBestModel(self):
        print('Qual modelo quer juntar com a previsao do BestModel para melhorar o score?')
        with open(f'{environment.algoritimos_dir}/{environment.resultado_completo_df}', 'rb') as file:
            resultados_formatados = pickle.load(file)
        print(json.dumps(resultados_formatados, indent=4))

        self.modelo = self.pergunta()

        try:
            self.results = pd.read_csv(environment.resultado_dir + self.modelo + '.csv', sep=',')
        except FileNotFoundError:
            print(f'Erro: arquivo de resultados do modelo {self.modelo} não encontrado.')
            return  # Encerra a execução da função

        try:
            self.bm = pd.read_csv(environment.resultado_dir + environment.bm + '.csv', sep=',')
        except FileNotFoundError:
            print('Erro: arquivo de resultados do BestModel não encontrado.')
            return  # Encerra a execução da função

        print("Bases carregadas")
        self.refazerScoreComBestModel()

    def refazerScoreComBestModel(self):
        # Verificações das colunas
        if 'predicao' not in self.results.columns:
            raise ValueError(f"Coluna 'predicao' não encontrada no DataFrame 'results' do modelo {self.modelo}.")
        if 'predicao' not in self.bm.columns:
            raise ValueError("Coluna 'predicao' não encontrada no DataFrame 'bm' do BestModel.")

        # Renomear colunas para evitar conflitos
        self.results.rename(columns={'predicao': 'predicao_results', 'score': 'score_results'}, inplace=True)
        self.bm.rename(columns={'predicao': 'predicao_bm', 'score': 'score_bm'}, inplace=True)

        # Juntar DataFrames
        combined_df = pd.merge(self.results, self.bm, left_index=True, right_index=True, how='left')

        # Ajustar scores
        for i, row in tqdm.tqdm(combined_df.iterrows(), total=combined_df.shape[0], desc="Criando Score"):
            if row['predicao_bm'] == 1 and row['score_results'] < 70:
                aumento = row['score_results'] * 0.5  # Aumento de 10% do score atual
                novo_score = min(100, row['score_results'] + aumento)  # Garantir que o score não ultrapasse 100%
                combined_df.at[i, 'score_results'] = novo_score

        # Selecionar todas as colunas do DataFrame 'results' e adicionar o score ajustado
        self.df = self.results.copy()
        self.df.drop(columns=["Unnamed: 0", "score_results"], inplace=True)
        self.df.rename(columns={'score_ajustado': 'score', 'predicao_results': 'predicao'}, inplace=True)
        print(self.df.head())
        score = self.df.query('score > 0.7').sort_values('score', ascending=True)
        print(f'10 Score > 0.7 \n {score}')
        self.salvarBestModel()

    def salvarBestModel(self):
        self.df.to_csv(environment.resultado_dir + 'resultado_final.csv')
        print('Resultado_final salvo !!!')
    
    @staticmethod
    def pergunta():
        while True:
            print("\nEscolha uma opção:")
            print("BM - Best Model")
            print("NB - Naive Bayes")
            print("ET - ExtraTrees")
            print("LR - Logistic Regression")
            print("KNN - KNeighbors")
            print("GB - GradientBoosting")
            print("AB - AdaBoost")
            print("RF - RandomForest")
            print("S - Sair")
            modelo = input('Qual o modelo a ser usado para juntar com o BestModel? ' ).lower()
            if modelo == 'nb':
                return environment.nb
            elif modelo == 'et':
                return environment.et
            elif modelo == 'lr':
                return environment.lr
            elif modelo == 'knn':
                return environment.knn
            elif modelo == 'gb':
                return environment.gb
            elif modelo == 'ab':
                return environment.ab
            elif modelo == 'rf':
                return environment.rf
            elif modelo == 'bm':
                return environment.bm
            else:
                print("Opção inválida. Por favor, escolha uma opção válida.")


