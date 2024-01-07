# Projeto Propensão

Este projeto é uma aplicação  para processamento e análise de dados para utilizando de IA Supervisionada de classificação, utilizando diversas técnicas de machine learning e oferecendo uma interface web para interação com os usuários.

## Estrutura do Projeto

A aplicação está estruturada da seguinte forma:
propensao/
├── pycache/
├── config/
│ ├── init.py
│ └── enviroment.py
├── controller/
│ ├── init.py
│ ├── app.py
│ ├── interativo_base_utilizacao.py
│ ├── interativo_tratamento_variaveis.py
│ ├── looping_algoritimos.py
│ ├── maquina_comites.py
│ ├── previsor.py
│ ├── score_best_model.py
│ └── tratamento_base_utilizacao.py
├── env/
├── model/
├── start.sh
└── view/

### Descrição dos Diretórios

- `__pycache__`: Contém arquivos compilados que o Python cria para acelerar o carregamento de módulos.
- `config`: Guarda arquivos de configuração, como variáveis de ambiente e configurações gerais.
- `controller`: Contém a lógica de controle que faz a ponte entre o modelo e as visualizações. Inclui:
  - `app.py`: Ponto de entrada principal da aplicação web.
  - `interativo_base_utilizacao.py`: Controla a interação com a base de utilização.
  - `interativo_tratamento_variaveis.py`: Gerencia a interação com o tratamento de variáveis.
  - `looping_algoritimos.py`: Executa o loop dos algoritmos de machine learning.
  - `maquina_comites.py`: Implementa a lógica para a máquina de comitês.
  - `previsor.py`: Controla a funcionalidade de previsão.
  - `score_best_model.py`: Responsável pelo cálculo de score utilizando o melhor modelo.
  - `tratamento_base_utilizacao.py`: Trata a base de utilização.
- `env`: Ambiente virtual Python contendo todas as dependências do projeto.
- `model`: Armazena os modelos de dados, algoritmos e qualquer lógica de negócios relacionada.
- `view`: Contém arquivos relacionados à interface do usuário, como templates HTML.

## Configuração e Instalação

Instruções para configurar o ambiente e instalar as dependências necessárias.

### Pré-requisitos

- Python 3.x
- Outras dependências devem ser instaladas pelo requirementes.txt


### Instalação

Passo a passo para instalar e configurar o projeto.

1. Clone o repositório:

    git clone https://github.com/RafaelMunareto/propens-o_ia_adimp.git

2.  Instale as dependências:
    basta executar o arquivo install.sh - no terminal com ./install.sh

3. Crie uma .env com o comando
    pip venv 

3. execute o arquivo start.sh - no terminal com ./start.sh
