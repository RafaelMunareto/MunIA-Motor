
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import config.environment as environment
from sklearn.preprocessing import LabelEncoder

class InterativoBaseUtilizacao:
    def __init__(self, df):
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
            resposta = self.solicitarEntradaValida(
                f"Essa coluna '{coluna}' é previsor ou descartar? (P/D): ",
                lambda x: x.lower() in ['p', 'd']
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
                
                nan_count = self.df[coluna].isna().sum()
                if nan_count > 0:
                    self.tratarPrevisor(coluna)

                print(f"Coluna {coluna} - NAN: {nan_count}")
        
    def tratarQuantitativo(self, coluna):
        if self.df[coluna].isna().sum() == 0:
            return
        label_encoder = LabelEncoder()
        self.df[coluna] = self.df[coluna].astype(str)
        self.df[coluna] = label_encoder.fit_transform(self.df[coluna])
        escolha = self.solicitarEntradaValida(
            f"Para NaN/Null na coluna '{coluna}', escolha (media/mediana/moda/0/1/descartar): ",
            lambda x: x in ['media', 'mediana', 'moda', '0', '1', 'descartar']
        )
        self.aplicarTratamentoNaN(coluna, escolha)
        
    def tratarPrevisor(self, coluna):
        tipo_dados = self.solicitarEntradaValida(
            f"Qual o tipo de dados da coluna '{coluna}'? (QT/QL/DT/CEP): ",
            lambda x: x in ['qt', 'ql', 'dt', 'cep']
        ).lower()
        print(f'isna {coluna} : {self.df[coluna].isna().sum()}')
        if tipo_dados == 'qt':
            if self.df[coluna].dtype == object: 
                self.df[coluna] = self.df[coluna].str.replace(',', '.').astype(float)
            self.tratarQuantitativo(coluna)
        elif tipo_dados == 'ql':
            self.tratarQualitativo(coluna)
        elif tipo_dados == 'dt':
            self.tratarData(coluna)
        elif tipo_dados == 'cep':
            self.tratarCEP(coluna)

    def tratarQualitativo(self, coluna):
        if self.df[coluna].isna().sum() == 0:
            return
        escolha = self.solicitarEntradaValida(
            f"Para NaN/Null na coluna '{coluna}', escolha (descartar/preencher): ",
            ['descartar', 'preencher']
        )
        self.aplicarTratamentoNaN(coluna, escolha)
        self.df[coluna] = pd.Categorical(self.df[coluna]).codes

    def tratarCEP(self, coluna):
        digitos = int(self.solicitarEntradaValida(
            "Quantos dígitos do CEP deseja usar para representar a região? ",
            lambda x: x.isdigit() and int(x) > 0
        ))

        escolha_preenchimento = self.solicitarEntradaValida(
            f"Para NaN/Null na coluna '{coluna}', escolha entre descartar os registros com NaN ou preencher com um valor padrão (descartar/preencher): ",
            lambda x: x in ['descartar', 'preencher']
        )

        if escolha_preenchimento == 'preencher':
            while True:
                valor_preenchimento = input(f"Digite o valor numérico (com exatamente {digitos} dígitos) para preencher NaN/Null na coluna '{coluna}': ")
                if valor_preenchimento.isdigit() and len(valor_preenchimento) == digitos:
                    self.df[coluna].fillna(valor_preenchimento, inplace=True)
                    break
                print(f"Por favor, digite um número com exatamente {digitos} dígitos.")
        elif escolha_preenchimento == 'descartar':
            self.df.dropna(subset=[coluna], inplace=True)

        # Truncate the CEP column
        self.df[coluna] = self.df[coluna].astype(str).str[:digitos]

        # Apply label encoding
        label_encoder = LabelEncoder()
        self.df[coluna] = label_encoder.fit_transform(self.df[coluna])


    def tratarQualitativo(self, coluna):
        escolha = self.solicitarEntradaValida(
            f"Para NaN/Null na coluna '{coluna}', escolha (descartar/preencher): ",
            lambda x: x in ['descartar', 'preencher']
        )
        self.aplicarTratamentoNaN(coluna, escolha)
        self.df[coluna] = pd.Categorical(self.df[coluna]).codes
       

    def tratarData(self, coluna):
        escolha_data = self.solicitarEntradaValida(
            f"Como deseja tratar a coluna de data '{coluna}'? (dias/meses/anos): ",
            lambda x: x in ['dias', 'meses', 'anos']
        )
        coluna_data = pd.to_datetime(self.df[coluna], errors='coerce')
        if escolha_data == 'dias':
            data_referencia = pd.Timestamp('1900-01-01')
            self.df[coluna] = (coluna_data - data_referencia).dt.days
        elif escolha_data == 'meses':
            self.df[coluna] = coluna_data.dt.year * 12 + coluna_data.dt.month - 190001
        elif escolha_data == 'anos':
            self.df[coluna] = coluna_data.dt.year

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
        elif escolha == 'preencher':
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
        print(f'previsores:\n{self.previsores}')
        return pd.DataFrame(self.previsores)
