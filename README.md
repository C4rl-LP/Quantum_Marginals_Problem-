# Problema das Marginais Quânticas

Este projeto estuda o Problema das Marginais Quânticas (QMP) por meio de experimentos computacionais em Python. O objetivo é investigar quando matrizes densidade reduzidas são compatíveis com um estado global e como reconstruir estados que reproduzam essas marginais. O código usa programação semidefinida para estudar a compatibilidade e otimização dual com regularização entrópica para selecionar estados globalmente compatíveis segundo um critério de energia e entropia.

## Instalação

Na raiz do repositório, crie e ative um ambiente virtual com Python 3.13:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requeriments.txt
```

O arquivo `requeriments.txt` fixa as versões das dependências diretas e instala o pacote local `modulos` em modo editável. Assim, o ambiente virtual usa as alterações feitas no código local sem precisar reinstalar o pacote.

Para rodar o experimento de correlações entre dois qubits:

```bash
python tests/qubits/Experimento_1.py
```
