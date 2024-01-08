
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import config.environment as environment
from sklearn.preprocessing import LabelEncoder

class InterativoBaseUtilizacao:

    def __init__(self, df, tipo):
        self.tipo = tipo
        self.df = df
        self.alvo = None
        self.previsores = None
        self.respostas = {}

    def solicitarEntradaValida(self, pergunta, funcao_validacao):
        while True:
            resposta = input(pergunta).strip().lower()
            if funcao_validacao(resposta):
                return resposta
            else:
                print("Resposta inválida, tente novamente.")

    def processarColunas(self):
        for coluna in list(self.df.columns):
            print(f"Amostra da coluna '{coluna}':\n{self.df[coluna].head(3)}")

            # Solicitando a entrada e convertendo para minúscula para padronização
            if self.tipo == True:
                resposta = self.solicitarEntradaValida(
                    f"Essa coluna '{coluna}' é previsor ou descartar? (A/P/D): ",
                    lambda x: x.lower() in ['a', 'p', 'd']
                ).lower()
                if resposta == 'd':
                    self.df.drop(coluna, axis=1, inplace=True)
                elif resposta == 'a':
                    self.definirAlvo(coluna)
                elif resposta == 'p':
                    novo_nome = input(f"Deseja renomear a coluna '{coluna}'? Deixe em branco para manter ou digite o novo nome: ")
                    if novo_nome:
                        self.df.rename(columns={coluna: novo_nome}, inplace=True)
                        coluna = novo_nome
                    self.tratarPrevisor(coluna)
            else: 
                resposta = self.solicitarEntradaValida(
                    f"Essa coluna '{coluna}' é previsor ou descartar? (P/D): ",
                    lambda x: x.lower() in [ 'p', 'd']
                ).lower()
                if resposta == 'd':
                    self.df.drop(coluna, axis=1, inplace=True)
                elif resposta == 'p':
                    novo_nome = input(f"Deseja renomear a coluna '{coluna}'? Deixe em branco para manter ou digite o novo nome: ")
                    if novo_nome:
                        self.df.rename(columns={coluna: novo_nome}, inplace=True)
                        coluna = novo_nome
                    self.tratarPrevisor(coluna)

                    print(f"Coluna {coluna} - NAN: {self.df[coluna].isna().sum()}")

    def definirAlvo(self, coluna):
        # Preenche valores NaN com 0
        self.df[coluna].fillna(0, inplace=True)
        # Verifica se os valores são numéricos ou strings
        if self.df[coluna].dtype == 'object':
            # Transforma todas as strings não nulas em 1
            self.alvo = self.df[coluna].apply(lambda x: 0 if x == 0 else 1)
        else:
            # Para valores numéricos, transforma todos os não zeros em 1
            self.alvo = self.df[coluna].apply(lambda x: 1 if x != 0 else 0)

        # Adiciona a coluna 'alvo' transformada ao DataFrame e remove a coluna original
        self.df['alvo'] = self.alvo
        self.df.drop(columns=coluna, inplace=True)

        # Armazena informações da coluna original
        self.respostas['alvo'] = {'coluna_original': coluna}

    def tratarQuantitativo(self, coluna):
        if self.df[coluna].isna().sum() > 0:
            escolha = self.solicitarEntradaValida(
            f"Para NaN/Null na coluna '{coluna}', escolha (media/mediana/moda/0/1/descartar): ",
            lambda x: x in ['media', 'mediana', 'moda', '0', '1', 'descartar'])

            self.aplicarTratamentoNaN(coluna, escolha)
       
    def tratarPrevisor(self, coluna):
        tipo_dados = self.solicitarEntradaValida(
            f"Qual o tipo de dados da coluna '{coluna}'? (QT/QL/DT/CEP): ",
            lambda x: x in ['qt', 'ql', 'dt', 'cep']
        ).lower()
    
        if tipo_dados == 'qt':
            if self.df[coluna].dtype == object: 
                self.df[coluna] = pd.to_numeric(self.df[coluna].str.replace(',', '.'), errors='coerce')
            print(f'isna {coluna} : {self.df[coluna].isna().sum()}')
            self.tratarQuantitativo(coluna)
        elif tipo_dados == 'ql':
            print(f'isna {coluna} : {self.df[coluna].isna().sum()}')
            self.tratarQualitativo(coluna)
        elif tipo_dados == 'dt':
            print(f'isna {coluna} : {self.df[coluna].isna().sum()}')
            self.tratarData(coluna)
        elif tipo_dados == 'cep':
            print(f'isna {coluna} : {self.df[coluna].isna().sum()}')
            self.tratarCEP(coluna)

    def tratarQualitativo(self, coluna):
        label_encoder = LabelEncoder()
        self.df[coluna] = label_encoder.fit_transform(self.df[coluna].astype(str))
        if self.df[coluna].isna().sum() > 0:
            escolha = self.solicitarEntradaValida(
                f"Para NaN/Null na coluna '{coluna}', escolha (DESC/PRE): ",
                lambda x: x in ['desc', 'pre'] 
            )
            self.aplicarTratamentoNaN(coluna, escolha)

    def tratarCEP(self, coluna):
        if(self.df[coluna].isna().sum() > 0):
            digitos = int(self.solicitarEntradaValida(
                "Quantos dígitos do CEP deseja usar para representar a região? ",
                lambda x: x.isdigit() and int(x) > 0
            ))
       
            escolha_preenchimento = self.solicitarEntradaValida(
                f"Para NaN/Null na coluna '{coluna}', escolha entre descartar os registros com NaN ou preencher com um valor padrão (desc/pre): ",
                lambda x: x in ['desc', 'pre']
            )

            if escolha_preenchimento == 'pre':
                self.df[coluna] = self.df[coluna].astype(str).str[:digitos]

                label_encoder = LabelEncoder()
                self.df[coluna] = label_encoder.fit_transform(self.df[coluna])
                
            elif escolha_preenchimento == 'desc':
                self.df.dropna(subset=[coluna], inplace=True)
    
    def tratarData(self, coluna):
        self.df[coluna] = pd.to_datetime(self.df[coluna], errors='coerce')
        escolha_data = self.solicitarEntradaValida(
            f"Como deseja tratar a coluna de data '{coluna}'? (dias/meses/anos): ",
            lambda x: x in ['dias', 'meses', 'anos']
        )

        if escolha_data == 'dias':
            data_referencia = pd.Timestamp('1900-01-01')
            self.df[coluna]  = (self.df[coluna] - data_referencia).dt.days
        elif escolha_data == 'meses':
            self.df[coluna]  = self.df[coluna].dt.year * 12 + self.df[coluna].dt.month - 190001
        elif escolha_data == 'anos':
            self.df[coluna]  = self.df[coluna].dt.year

        if self.df[coluna].isna().sum() > 0:
            escolha = self.solicitarEntradaValida(
                f"Para NaN/Null na coluna '{coluna}', escolha (media/mediana/moda/0/1/descartar): ",
                lambda x: x in ['media', 'mediana', 'moda', '0', '1', 'descartar']
            )
            self.aplicarTratamentoNaN(coluna, escolha)

    def aplicarTratamentoNaN(self, coluna, escolha):
        self.df[coluna] = pd.to_numeric(self.df[coluna], errors='coerce')
        if escolha == 'media':
            self.df[coluna].fillna(self.df[coluna].mean(), inplace=True)
        elif escolha == 'mediana':
            self.df[coluna].fillna(self.df[coluna].median(), inplace=True)
        elif escolha == 'moda':
            moda = self.df[coluna].mode()
            if len(moda) > 0:
                self.df[coluna].fillna(moda[0], inplace=True)
        elif escolha in ['0', '1']:
            self.df[coluna].fillna(int(escolha), inplace=True)
        elif escolha == 'descartar':
            self.df.dropna(subset=[coluna], inplace=True)
        elif escolha == 'pre':
            preenchimento = input(f"Escolha o valor para preencher NaN/Null na coluna '{coluna}': ")
            self.df[coluna].fillna(preenchimento, inplace=True)

    def salvarRespostas(self):
        data_atual = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.respostas['data'] = data_atual
        with open(environment.respostas_tratamento_base, 'wb') as file:
            pickle.dump(self.respostas, file)
        print(f"Respostas salvas em {environment.respostas_tratamento_base} em {data_atual}.")

    def processar(self):
        self.processarColunas()
        self.previsores = self.df
        print(f'previsores:\n {self.previsores}')
        return pd.DataFrame(self.previsores)
