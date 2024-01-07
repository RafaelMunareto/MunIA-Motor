import numpy as np
import pandas as pd
from interativo_base_utilizacao import InterativoBaseUtilizacao
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import IncrementalPCA
import pickle
import config.environment as environment


class TratamentoVariaveisBaseUtilizacao:
    def __init__(self):
        self.previsores = None
        self.previsores_scalonados = None
        self.pca_model = None
        self.variance_explained = None

    def capturaDadosUtilizacao(self): 
        nome  = input("Qual base vc quer utilizar - digite o nome completo com a extensão? ")
        separado = input('É separado por , ou ; o arquivo ? ')
        try:
            self.df = pd.read_csv(environment.teste_dir + nome, sep=separado)
            print("Base carregada")
        except:
            print('Base não encontrada ou diretório inexistente')
       
        with open(f'{environment.variaveis_dir}{environment.previsores}', 'rb') as file:
            previsores = pickle.load(file)
        print('Deve ter essas colunas e não pode conter NAN ou NULL')
        print(f'{previsores.columns}')
        self.tratamentoVariaveis()
        

    def tratamentoVariaveis(self): 
        tratamento = InterativoBaseUtilizacao(self.df)
        self.previsores = tratamento.processar()
        print(f' previsores retornados \n {self.previsores}')
        self.pca()
        self.escalonarPrevisores()


    def escalonarPrevisores(self):
        scaler = StandardScaler()
        self.previsores_scalonados = scaler.fit_transform(self.previsores)
        print("Variáveis escalonadas")

    def pca(self, variance_threshold=0.90, batch_size=None):
        n_components = 0
        variance_explained = 0

        while variance_explained < variance_threshold and n_components < self.previsores.shape[1]:
            n_components += 1
            ipca = IncrementalPCA(n_components=n_components, batch_size=batch_size)
            transformed_data = ipca.fit_transform(self.previsores)
            variance_explained = np.sum(ipca.explained_variance_ratio_)

        self.pca_model = ipca
        self.variance_explained = variance_explained
        self.pca_model = transformed_data
        print("Redução de dimensionalidade concluída")
        print(f'Fazendo o algoritimo {n_components} de {self.previsores.shape}')
        print(f'Variância de {variance_explained}')


    def salvarVariaveisBaseUtilizacao(self):
        pickle_files = {
            environment.previsores: pd.DataFrame(self.previsores),
            environment.previsores_scalonados: pd.DataFrame(self.previsores_scalonados),
            environment.previsores_pca: self.pca_model,
        }        
        self.verificacao()
        for filename, data in pickle_files.items():
            with open(f'{environment.teste_dir}{filename}', 'wb') as file:
                pickle.dump(data, file)
        print("Variáveis salvas.")

    def verificacao(self):
        with open(f'{environment.variaveis_dir}{environment.previsores}', 'rb') as file:
            previsores_bench = pickle.load(file)

        print('Verificação: \n')
        print('Previsores originais: \n')
        print(f'{previsores_bench.head(5)} \n')
        print(f'{previsores_bench.isna().sum()} \n')
        print('Previsores processados agora: \n')
        print(f'{pd.DataFrame(self.previsores).head(5)} \n')
        print(f'{pd.DataFrame(self.previsores).isna().sum()} \n')
        print('Lembrando que devem estar iguais, mesmas quantidade e colunas iguais, sem NAN ou NULL')
        correto = input('Está correto ? (S) (N)').strip().lower()
        if(correto == 'n'):
            retorna = TratamentoVariaveisBaseUtilizacao()
            retorna.capturaDadosUtilizacao()