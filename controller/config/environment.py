

# Diretórios de arquivos
model = 'model/'
base_dir = model + 'base/'
teste_dir = model + 'teste/'
resultado_dir = model + 'resultados/'
algoritimos_dir = model + 'algoritimos/'
variaveis_dir = model + 'variaveis/'
variaveis_csv_file = base_dir + 'IA_TUDO_EM_DIA_CONTROLE.txt'
max_alvo_previsor = ':'

#variaveis 
alvo = 'alvo.pickle'
previsores = 'previsores.pickle'
previsores_scalonados = 'previsores_scalonados.pickle'
previsores_pca = 'previsores_pca.pickle'
df = 'df.pickle'
df_com_previsao = './algoritmos/df_com_previsao.pickle'
resultado_completo_df = 'resultado_completos_metrica_algoritimos'
results_df = resultado_dir + 'df_com_previsao.pickle'
respostas_tratamento_base = 'respostas_tratamento_base.pickle'

#colunas previsores 
predicao = 'predicao'
score = 'score'
#alvo = 'alvo'
faixa_score = 'Faixa_Score'

#modelos de treinos
rf =  "RandomForest"
lr = "LogisticRegression"
knn = "KNeighbors"
nb = "GaussianNB"
bm = "BestModel"
sgd = "SGDClassifier"
gb = "GradientBoosting"
ab = "AdaBoost"
dt = "DecisionTree"
et = "ExtraTrees"
ct = "CatBoost"
xgb = "XBoost"
svm = "SVM"
mlp = "Rede_Neural"

#previsor_utilizado e modelo utilizado
previsor_utilizado = previsores_scalonados
cv = 10
X_test = 'X_test.pickle'
Y_test = 'Y_test.pickle'
X_train = 'X_train.pickle'
Y_train = 'Y_train.pickle'
modelo_aplicado = algoritimos_dir + bm + ".pickle" 


