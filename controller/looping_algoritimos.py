import pickle
import numpy as np
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from datetime import datetime
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
import config.environment as environment 
from tqdm import tqdm
import json
import os 

class LoopingAlgoritmos:
    def __init__(self):
        self.alvo = None
        self.escolha = None
        self.tamanho = None
        self.previsores = None
        self.modelos = {}
        self.resultados = {}
        self.diretorio = None


    def verificaDiretorio(self):
        self.diretorio =  f'{environment.algoritimos_dir}{self.escolha}_{self.k_m_formatter(self.tamanho)}/'
        if not os.path.exists(self.diretorio):
            os.makedirs(self.diretorio)
    
    def carregarDados(self):
        self.escolha = input('\nQual previsor deseja escolher? Previsor (P), Scanolado (S) ou PCA (PCA)? ').lower()
        if self.escolha == 'p':
            environment.previsor_utilizado = environment.previsores
        elif self.escolha == 's':
            environment.previsor_utilizado = environment.previsores_scalonados
        elif self.escolha == 'pca':
            environment.previsor_utilizado = environment.previsores_pca
        self.tamanho = int(input('Tamanho Máximo do previsor e alvo? '))
        with open(f'{environment.variaveis_dir}{environment.alvo}', 'rb') as file:
            self.alvo = pickle.load(file)
            if self.tamanho < len(self.alvo): 
                self.alvo = self.alvo[:self.tamanho]
                environment.tamanho = self.tamanho
        with open(f'{environment.variaveis_dir}{environment.previsor_utilizado}', 'rb') as file:
            self.previsores = pickle.load(file)
            if self.tamanho < len(self.previsores):  
                self.previsores = self.previsores[:self.tamanho]
        self.verificaDiretorio()
            
    def k_m_formatter(self, x):
        if x >= 1000000:  
            return f'{int(x/1000000)}m'
        else:  
            return f'{int(x/1000)}k'
        
    def treinarModelos(self):
        print(f'\nPrevisor utilizado: {environment.previsor_utilizado}')
        inicio_treinamento = datetime.now()
        X_train, X_test, y_train, y_test = train_test_split(
            self.previsores, self.alvo, test_size=0.3, random_state=42
        )

        y_train = y_train.values.ravel()
        print("Número de linhas em X:", X_train.shape[0])
        print("Número de linhas em y:", y_train.shape[0])
        print("Terminou a divisão treino e teste!\n")

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
            environment.mlp: MLPClassifier(hidden_layer_sizes=(100,), random_state=42, max_iter=500),
            environment.lr: LogisticRegression(max_iter=1000, C=1.0, solver='saga', random_state=42),
            environment.knn: KNeighborsClassifier(n_neighbors=5, weights='distance'),
            environment.gb: GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42),
            environment.ab: AdaBoostClassifier(n_estimators=200, learning_rate=0.05, random_state=42),
            environment.rf: RandomForestClassifier(n_estimators=200, max_features='sqrt', max_depth=10, random_state=42),
        }

        resultados = {}
      
        try:
            with open(f'{self.diretorio}{environment.resultado_completo_df}.pickle', 'rb') as file:
                resultados_completos = pickle.load(file)
        except FileNotFoundError:
            resultados_completos = {"resultados": {}, "inicio": inicio_treinamento.strftime('%Y-%m-%d %H:%M:%S')}
        barra_progresso = tqdm(total=len(algoritmos), unit="modelo")
        for nome, modelo in algoritmos.items():
            if nome in resultados_completos['resultados']:
                print(f"Resultados do modelo {nome} já existem. Pulando para o próximo.")
                continue
            barra_progresso.set_description(f"Treinando {nome}...")
         
            cv_scores = cross_val_score(modelo, X_train, y_train, cv=environment.cv)
            cv_accuracy = np.mean(cv_scores)
            modelo.fit(X_train, y_train)
            y_pred = modelo.predict(X_test)
            
            acc = accuracy_score(y_test, y_pred)
            precision, recall, fscore, _ = precision_recall_fscore_support(y_test, y_pred, average='binary', zero_division=0)
            roc_auc = roc_auc_score(y_test, modelo.predict_proba(X_test)[:, 1])
        
            cm = confusion_matrix(y_test, y_pred)
           
            fim_treinamento = datetime.now()
            resultados[nome] = {
                "inicio": inicio_treinamento.strftime('%Y-%m-%d %H:%M:%S'),
                "accuracy": acc,
                "precision": precision,
                "cv_Accuracy": cv_accuracy,
                "recall": recall,
                "f1_score": fscore,
                "roc_auc": roc_auc,
                "confusion_matrix": cm.tolist(),
                "fim": fim_treinamento.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            with open(f'{self.diretorio}{nome}.pickle', 'wb') as file:
                pickle.dump(modelo, file)
            
           
            resultados_completos = {
                "resultados": resultados,
            }
            with open(f'{self.diretorio}{environment.resultado_completo_df}.pickle', 'wb') as file:
                pickle.dump(resultados_completos, file)
            barra_progresso.update(1)
            
        barra_progresso.close()
      
        
        print(json.dumps(resultados_completos, indent=4))
        self.previsoesMetricas()
       
        self.resultados = resultados_completos

        
    def obterResultados(self):
        return self.resultados


    def previsoesMetricas(self):
        print('accuracy: Proporção de previsões corretas sobre o total. Mede a eficácia geral do modelo.\n')
        print('precision: Proporção de previsões positivas corretas. Indica a qualidade dos resultados positivos do modelo.\n')
        print('cv_Accuracy: Média da acurácia do modelo em diferentes subconjuntos do conjunto de treinamento. Oferece uma estimativa mais robusta da performance do modelo.\n')
        print('recall: Proporção de casos positivos reais identificados corretamente. Mede a capacidade do modelo de detectar resultados positivos.\n')
        print('f1_score: Média harmônica de precisão e recall. Combina precisão e revocação em uma única métrica, útil quando o equilíbrio entre estas é importante.\n')
        print('roc_auc: Medida de quão bem o modelo distingue entre classes. Quanto maior, melhor o modelo em diferenciar entre positivo e negativo.\n')
        print('confusion_matrix: Tabela que mostra os acertos e erros do modelo comparados com a realidade. Ajuda a entender os tipos de erro e acerto.\n')
