# Instruções

Este arquivo tem o propósito de explicar a organização e execução do trabalho prático.

## Arquivos

Existem dois diretórios `documentação` e `datasets`. O diretório `documentação` contém os arquivos referentes ao relatório em dois formatos: um em LaTeX (com os respectivos arquivos necessários para compilação) e outro em PDF. O relatório se encontra no padrão referente ao usado nos artigos SBC. O diretório `datasets` contém todos os arquivos tsp usados nos experimentos. No diretório raiz, se encontra o arquivo fonte `tsp.py` em Python que contém todas as implementações dos algoritmos para resolver as instâncias do problema. Além disso, também se encontra o arquivo `resultados` em formatos csv e xlsx contendo uma tabela com os resultados dos experimentos.

## Execução

Especificações do ambiente de execução dos experimentos: sistema operacional Windows 10 com uso do recurso Windows Subsystem for Linux (WSL) na distribuição Ubuntu.

Execução na linha de comando:
python3 tsp.py nome_do_dataset

Observações: 
- O dataset deve estar no diretório raiz para o comando acima.
- O código para executar o algoritmo Branch-And-Bound está comentando no arquivo fonte devido ao tempo não hábil de execução.