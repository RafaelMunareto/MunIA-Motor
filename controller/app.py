
from tratamento_base_utilizacao import TratamentoVariaveisBaseUtilizacao
from looping_algoritimos import LoopingAlgoritmos
from maquina_comites import MaquinaDeComites
from previsor import Previsor

def processarBase():
    data_model = TratamentoVariaveisBaseUtilizacao(True)
    data_model.capturaDadosUtilizacao()  
    data_model.salvarVariaveisBaseUtilizacao()
        
def processarBaseUtilizacao():
    data_utilizacao = TratamentoVariaveisBaseUtilizacao(False)
    data_utilizacao.capturaDadosUtilizacao()  
    data_utilizacao.salvarVariaveisBaseUtilizacao()
        
def rodarModelos():
    loop = LoopingAlgoritmos()
    loop.carregarDados()
    loop.treinarModelos()
    loop.obterResultados()

def maquinaComites():
    comites = MaquinaDeComites()
    comites.criarComite()

def previsao():
    preditor = Previsor()
    preditor.carregarModelo()

def menu_principal():
    while True:
        print("\nEscolha uma opção:")
        print("B - Processar base de treino")
        print("U - Processar base de teste")
        print("R - Rodar modelos")
        print("C - Máquina de comitês")
        print("P - Predict e score")
        print("S - Sair")
    
        escolha = input("Digite sua escolha? ").lower()
    
        if escolha == "b":
            processarBase()
            pass
        elif escolha == "u":
            processarBaseUtilizacao()
        elif escolha == "r":
            rodarModelos()
            pass
        elif escolha == "c":
            maquinaComites()
            pass
        elif escolha == "p":
            previsao()
            pass 
        elif escolha == "s":
            print("Saindo do programa.")
            break
        else:
            print("Escolha inválida. Tente novamente.")

menu_principal()
