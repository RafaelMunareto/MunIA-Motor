import pickle
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from datetime import datetime
import json
import os

class LoopingAlgoritmos:
    def __init__(self, config):
        self.config = config
        self.alvo = None
        self.previsores = None
        self.modelos = {}
        self.resultados = {}
    
    def carregarDados(self, escolha, tamanho):
        if escolha not in ['p', 's', 'pca']:
            raise ValueError("Escolha inválida. Deve ser 'p', 's' ou 'pca'.")
        
        previsor_utilizado = self.config['previsores'][escolha]
        alvo_path = os.path.join(self.config['variaveis_dir'], self.config['alvo'])
        previsores_path = os.path.join(self.config['variaveis_dir'], previsor_utilizado)
        
        if not os.path.exists(alvo_path) or not os.path.exists(previsores_path):
            raise FileNotFoundError("Arquivo de dados não encontrado.")
        
        with open(alvo_path, 'rb') as file:
            self.alvo = pickle.load(file)[:tamanho]
        with open(previsores_path, 'rb') as file:
            self.previsores = pickle.load(file)[:tamanho]
            
    def treinarModelos(self):
        X_train, X_test, y_train, y_test = train_test_split(
            self.previsores, self.alvo, test_size=0.3, random_state=42
        )
        y_train = y_train.values.ravel()
        
        # Criação e treinamento de modelos
        algoritmos = {
            'nb': GaussianNB(),
            'et': ExtraTreesClassifier(),
            'lr': LogisticRegression(max_iter=1000),
            'knn': KNeighborsClassifier(),
            'gb': GradientBoostingClassifier(),
            'ab': AdaBoostClassifier(),
            'rf': RandomForestClassifier(),
        }
        resultados = {}
        inicio_treinamento = datetime.now()

        for nome, modelo in algoritmos.items():
            cv_scores = cross_val_score(modelo, X_train, y_train, cv=self.config['cv'])
            cv_accuracy = np.mean(cv_scores)
            cv_std = np.std(cv_scores)
            modelo.fit(X_train, y_train)
            y_pred = modelo.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            self.modelos[nome] = modelo
            resultados[nome] = {
                "accuracy": acc,
                "cv_accuracy": cv_accuracy,
                "cv_std": cv_std
            }
            with open(os.path.join(self.config['algoritimos_dir'], f'{nome}.pickle'), 'wb') as file:
                pickle.dump(modelo, file)

        fim_treinamento = datetime.now()
        resultados_completos = {
            "resultados": resultados,
            "inicio": inicio_treinamento.strftime('%Y-%m-%d %H:%M:%S'),
            "fim": fim_treinamento.strftime('%Y-%m-%d %H:%M:%S')
        }
        with open(os.path.join(self.config['algoritimos_dir'], self.config['resultado_completo_df']), 'wb') as file:
            pickle.dump(resultados_completos, file)

        self.resultados = resultados_completos

    def obterResultados(self):
        return self.resultados
