import pickle
import pandas as pd
from sklearn.ensemble import VotingClassifier
import config.environment as environment
from tqdm import tqdm
from datetime import datetime
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import json

class MaquinaDeComites:
    def __init__(self):
        self.resultados = None
        self.previsores = None
        self.alvo = None
        self.modelos = {}

    def carregar_dados(self, caminho):
        with open(caminho, 'rb') as file:
            return pickle.load(file)

    def carregarResultados(self):
        try:
            with open(f'{environment.algoritimos_dir}{environment.resultado_completo_df}', 'rb') as file:
                self.resultados = pickle.load(file)
            print("Resultado dos algoritmos carregados.")
        except FileNotFoundError:
            print(f"Arquivo {environment.resultado_completo_df} não encontrado.")
            self.resultados = None

    def carregarModelos(self):
        if not self.resultados or 'resultados' not in self.resultados:
            print("Resultados não disponíveis ou mal formatados.")
            return
        
        for nome in tqdm(self.resultados['resultados'], desc="Carregando modelos", unit="modelo"):
            try:
                with open(f'{environment.algoritimos_dir}{nome}.pickle', 'rb') as file:
                    self.modelos[nome] = pickle.load(file)
            except FileNotFoundError:
                print(f"Modelo {nome} não encontrado.")
                self.modelos[nome] = None
        print("Todos os modelos de treino carregados.")

    def criarComite(self):
        X_train = self.carregar_dados(f'{environment.variaveis_dir}X_train.pickle')
        y_train = self.carregar_dados(f'{environment.variaveis_dir}y_train.pickle')
        X_test = self.carregar_dados(f'{environment.variaveis_dir}X_test.pickle')
        y_test = self.carregar_dados(f'{environment.variaveis_dir}y_test.pickle')

        self.carregarResultados()
        if self.resultados is None:
            print("Não é possível criar o comitê sem resultados.")
            return

        self.carregarModelos()
        if not self.modelos:
            print("Nenhum modelo válido foi carregado. Não é possível criar o comitê.")
            return

        modelos_para_comite = []
        for nome, modelo in tqdm(self.modelos.items(), desc="Preparando modelos para o comitê", unit="modelo"):
            if modelo is not None:
                modelos_para_comite.append((nome, modelo))
        
        if not modelos_para_comite:
            print("Nenhum modelo válido para formar o comitê.")
            return

        voting = VotingClassifier(estimators=modelos_para_comite, voting='hard')
        voting.fit(X_train, y_train)
        print("Criação do comitê de algoritmos concluída.")

        with open(f'{environment.algoritimos_dir}{environment.bm}.pickle', 'wb') as file:
            pickle.dump(voting, file)

        self.adicionarResultadosComite(voting, X_train, y_train, X_test, y_test)
        return voting

    def adicionarResultadosComite(self, comite, X_train, y_train, X_test, y_test):
        inicio_mc = datetime.now()

        cv_scores = cross_val_score(comite, X_train, y_train, cv=environment.cv)
        cv_accuracy = cv_scores.mean()
        cv_std = cv_scores.std()

        y_pred_test = comite.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred_test)

        fim_mc = datetime.now()

        comite_resultados = {
            "cv_accuracy": cv_accuracy,
            "cv_std": cv_std,
            "test_accuracy": test_accuracy,
            "inicio_mc": inicio_mc.strftime('%Y-%m-%d %H:%M:%S'),
            "fim_mc": fim_mc.strftime('%Y-%m-%d %H:%M:%S')
        }

        try:
            with open(f'{environment.algoritimos_dir}{environment.resultado_completo_df}', 'rb') as file:
                resultados_existentes = pickle.load(file)

            resultados_existentes['best_model_comite'] = comite_resultados

            with open(f'{environment.algoritimos_dir}{environment.resultado_completo_df}', 'wb') as file:
                pickle.dump(resultados_existentes, file)
            print(json.dumps(resultados_existentes, indent=4))
        except Exception as e:
            print(f"Ocorreu um erro: {e}")
