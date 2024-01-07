
from tratamento_base_utilizacao import TratamentoVariaveisBaseUtilizacao
from tratamento_variaveis import TratamentoVariaveis
from looping_algoritimos import LoopingAlgoritmos
from maquina_comites import MaquinaDeComites
from previsor import Previsor
from score_best_model import ScoreBestModel

def processarBase():
    data_processor = TratamentoVariaveis()
    data_processor.capturaDados()  
    data_processor.salvarVariaveis()
        
def processarBaseUtilizacao():
    data_utilizacao = TratamentoVariaveisBaseUtilizacao()
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

def score():     
    analise = ScoreBestModel()
    analise.juntarComBestModel()

def menu_principal():
    while True:
        print("\nEscolha uma opção:")
        print("P - Processar a base")
        print("M - Rodar modelos")
        print("C - Máquina de comitês")
        print("PR - Propensão")
        print("A - Score")
        print("T - Tudo")
        print("E - Preparar Base para utilização")
        print("S - Sair")
    
        escolha = input("Digite sua escolha? ").lower()
    
        if escolha == "p":
            processarBase()
            pass
        elif escolha == "m":
            rodarModelos()
            pass
        elif escolha == "c":
            maquinaComites()
            pass
        elif escolha == "pr":
            previsao()
            pass
        elif escolha == "a":
            score()
            pass
        elif escolha == "t":
            processarBase()
            rodarModelos()
            maquinaComites()
            previsao()
            score()
            pass
        elif escolha == "e":
            processarBaseUtilizacao()
        elif escolha == "s":
            print("Saindo do programa.")
            break
        else:
            print("Escolha inválida. Tente novamente.")

menu_principal()
