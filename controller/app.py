
from tratamento_base_utilizacao import TratamentoVariaveisBaseUtilizacao
from looping_algoritimos import LoopingAlgoritmos
from maquina_comites import MaquinaDeComites
from previsor import Previsor
import config.environment as environment 
import os 
import shutil

class app():

    def __init__(self):
        self.caminho_projeto = environment.model

    def processarBase(self):
        data_model = TratamentoVariaveisBaseUtilizacao(True)
        data_model.capturaDadosUtilizacao()  
        data_model.salvarVariaveisBaseUtilizacao()
            
    def processarBaseUtilizacao(self):
        data_utilizacao = TratamentoVariaveisBaseUtilizacao(False)
        data_utilizacao.capturaDadosUtilizacao()  
        data_utilizacao.salvarVariaveisBaseUtilizacao()
            
    def rodarModelos(self):
        loop = LoopingAlgoritmos()
        loop.carregarDados()
        loop.treinarModelos()
        loop.obterResultados()

    def maquinaComites(self):
        comites = MaquinaDeComites()
        comites.criarComite()

    def previsao(self):
        preditor = Previsor()
        preditor.carregarModelo()

    def projeto(self):
        caminho_modelos = environment.model
        pastas_excluidas = ["resultados", "analise_score", "teste", "base", "algoritimos", "variaveis"]

        if not os.path.exists(caminho_modelos):
            print("Pasta de modelos não encontrada.")
            return

        pastas = [pasta for pasta in os.listdir(caminho_modelos) 
                if os.path.isdir(os.path.join(caminho_modelos, pasta)) and pasta not in pastas_excluidas]

        if not pastas:
            print("Não há nenhum projeto salvo")
        else:
            print("Projetos disponíveis:")
            for i, pasta in enumerate(pastas):
                print(f"{i+1}. {pasta}")

        print(f"{len(pastas) + 1}. Criar um novo projeto")
        print(f"{len(pastas) + 2}. Excluir um projeto existente")

        escolha = int(input("Escolha um projeto pelo número, crie um novo ou exclua um existente: "))

        if escolha == len(pastas) + 1:
            nova_pasta = input("Digite o nome do novo projeto: ")
            novo_caminho = os.path.join(caminho_modelos, nova_pasta)
            os.makedirs(novo_caminho, exist_ok=True)
            environment.atualiza_caminho_modelo(pasta_escolhida)
            print(f"Novo projeto criado: {nova_pasta}")
            self.menu_principal()
        elif escolha == len(pastas) + 2:
            print("Escolha um projeto para excluir:")
            for i, pasta in enumerate(pastas):
                print(f"{i+1}. {pasta}")
            escolha_exclusao = int(input("Digite o número do projeto a ser excluído: "))
            if 1 <= escolha_exclusao <= len(pastas):
                pasta_para_excluir = pastas[escolha_exclusao - 1]
                confirmacao = input(f"Tem certeza que deseja excluir o projeto '{pasta_para_excluir}'? Digite 'sim' para confirmar: ")
                if confirmacao.lower() == 'sim':
                    shutil.rmtree(os.path.join(caminho_modelos, pasta_para_excluir))
                    print(f"Projeto '{pasta_para_excluir}' excluído.")
                    self.projeto()
                else:
                    print("Exclusão cancelada.")
                    self.projeto()
            else:
                print("Escolha inválida para exclusão.")
                self.projeto()
        elif 1 <= escolha <= len(pastas):
            pasta_escolhida = pastas[escolha - 1]
            print(f"Projeto escolhido: {pasta_escolhida}")
            environment.atualiza_caminho_modelo(pasta_escolhida)
            self.menu_principal()
        else:
            print("Escolha inválida.")
            self.projeto()
    

    def menu_principal(self):

        while True:
            print("\nEscolha uma opção:")
            print("B - Processar base de treino")
            print("U - Processar base de teste")
            print("R - Rodar modelos")
            print("C - Máquina de comitês")
            print("P - Predict e score")
            print("T - Todas as etapas")
            print("V - Voltar")
        
            escolha = input("Digite sua escolha? ").lower()
        
            if escolha == "b":
                self.processarBase()
                pass
            elif escolha == "u":
                self.processarBaseUtilizacao()
            elif escolha == "r":
                self.rodarModelos()
                pass
            elif escolha == "c":
                self.maquinaComites()
                pass
            elif escolha == "p":
                self.previsao()
                pass 
            elif escolha == "t":
                self.processarBase()
                self.processarBaseUtilizacao()
                self.rodarModelos()
                self.maquinaComites()
                self.previsao()
                pass 
            elif escolha == "v":
                self.projeto()
                break
            else:
                print("Escolha inválida. Tente novamente.")


iniciar = app()
iniciar.projeto()
