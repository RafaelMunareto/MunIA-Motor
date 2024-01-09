import pickle
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from datetime import datetime

from xgboost import XGBClassifier
import config.environment as environment 
from tqdm import tqdm
import json

class LoopingAlgoritmos:
    def __init__(self):
        self.alvo = None
        self.previsores = None
        self.modelos = {}
        self.resultados = {}

    
    def carregarDados(self):
        escolha = input('Qual previsor deseja escolher? Previsor (P), Scanolado (S) ou PCA (PCA)? ').lower()
        if escolha == 'p':
            environment.previsor_utilizado = environment.previsores
        elif escolha == 's':
            environment.previsor_utilizado = environment.previsores_scalonados
        elif escolha == 'pca':
            environment.previsor_utilizado = environment.previsores_pca
        tamanho = int(input('Tamanho Máximo do previsor e alvo? '))
        with open(f'{environment.variaveis_dir}{environment.alvo}', 'rb') as file:
            self.alvo = pickle.load(file)
            if tamanho < len(self.alvo): 
                self.alvo = self.alvo[:tamanho]
                environment.tamanho = tamanho
        with open(f'{environment.variaveis_dir}{environment.previsor_utilizado}', 'rb') as file:
            self.previsores = pickle.load(file)
            if tamanho < len(self.previsores):  
                self.previsores = self.previsores[:tamanho]
            
        
    def treinarModelos(self):
        print(f'Previsor utilizado {environment.previsor_utilizado}')
        inicio_treinamento = datetime.now()
        X_train, X_test, y_train, y_test = train_test_split(
            self.previsores, self.alvo, test_size=0.3, random_state=42
        )

        y_train = y_train.values.ravel()
        print("Número de linhas em X:", X_train.shape[0])
        print("Número de linhas em y:", y_train.shape[0])
        print("Terminou a divisão treino e teste!")
        
        with open(f'{environment.variaveis_dir}X_test.pickle', 'wb') as file:
            pickle.dump(X_test, file)
        with open(f'{environment.variaveis_dir}Y_test.pickle', 'wb') as file:
            pickle.dump(y_test, file)
        with open(f'{environment.variaveis_dir}X_train.pickle', 'wb') as file:
            pickle.dump(X_train, file)
        with open(f'{environment.variaveis_dir}Y_train.pickle', 'wb') as file:
            pickle.dump(y_train, file)
        
        algoritmos = {
            environment.xgb: XGBClassifier(objective="binary:logistic", n_estimators=200, learning_rate=0.05, max_depth=6, subsample=0.8, random_state=42),
            environment.et: ExtraTreesClassifier(n_estimators=150, max_features='sqrt', max_depth=10, random_state=42),
            environment.lr: LogisticRegression(max_iter=1000, C=1.0, solver='saga', random_state=42),
            environment.knn: KNeighborsClassifier(n_neighbors=5, weights='distance'),
            environment.gb: GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42),
            environment.ab: AdaBoostClassifier(n_estimators=100, learning_rate=0.01, random_state=42),
            environment.rf: RandomForestClassifier(n_estimators=200, max_features='sqrt', max_depth=10, random_state=42),
        }

        resultados = {}
        barra_progresso = tqdm(total=len(algoritmos), unit="modelo")

        for nome, modelo in algoritmos.items():
            barra_progresso.set_description(f"Treinando modelo {nome}")
            cv_scores = cross_val_score(modelo, X_train, y_train, cv=environment.cv)
            cv_accuracy = np.mean(cv_scores)
            modelo.fit(X_train, y_train)
            y_pred = modelo.predict(X_test)
            
            acc = accuracy_score(y_test, y_pred)
            precision, recall, fscore, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
            roc_auc = roc_auc_score(y_test, modelo.predict_proba(X_test)[:, 1])
        
            cm = confusion_matrix(y_test, y_pred)
            barra_progresso.update(1)
            
            resultados[nome] = {
                "accuracy": acc,
                "precision": precision,
                "cv_Accuracy": cv_accuracy,
                "recall": recall,
                "f1_score": fscore,
                "roc_auc": roc_auc,
                "confusion_matrix": cm.tolist()  
            }
            
            with open(f'{environment.algoritimos_dir}{nome}.pickle', 'wb') as file:
                pickle.dump(modelo, file)
        barra_progresso.close()
        fim_treinamento = datetime.now()
        
        resultados_completos = {
            "resultados": resultados,
            "inicio": inicio_treinamento.strftime('%Y-%m-%d %H:%M:%S'),
            "fim": fim_treinamento.strftime('%Y-%m-%d %H:%M:%S')
        }
        print(json.dumps(resultados_completos, indent=4))
        with open(f'{environment.algoritimos_dir}{environment.resultado_completo_df}', 'wb') as file:
            pickle.dump(resultados_completos, file)

        self.resultados = resultados_completos

        
    def obterResultados(self):
        return self.resultados
