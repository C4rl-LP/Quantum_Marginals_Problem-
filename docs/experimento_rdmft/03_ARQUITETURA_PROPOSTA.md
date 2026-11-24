# 4. Arquitetura proposta

Usar funções. A primeira implementação recebe H, V e marginais por parâmetros explícitos. O estado reutilizável de uma varredura cabe em arrays locais como H, V, dimensões, observáveis e o último vetor dual. Uma classe de funcional acrescentaria gestão de estado e cache sem benefício comprovado. A classe DualEntropico mantém seu papel atual de objeto matemático que recebe um problema por vez.

| Local futuro | Responsabilidade |
|---|---|
| modulos/osciladores.py | Mover a implementação harmônica reutilizável e reunir estados de Fock e modos normais. O script existente conserva wrappers compatíveis mediante aprovação |
| modulos/marginais.py, modulos/operadores.py e modulos/validacoes.py | Gerar restrições, contrair tensores e validar entradas. Manter o contrato geral de marginais sobrepostas |
| modulos/dual_entropico.py e modulos/otimizacao.py | Avaliação matemática, gradiente e algoritmos. Preservar a convenção atual do dual |
| modulos/funcional_rdm.py | Compor Hamiltoniano, restrições e solver. Calcular observáveis e diagnósticos na primeira entrega. Extrair observáveis genéricos para outro módulo só se surgirem consumidores |
| exemplos/energia_marginais_osciladores.py | Script de experimentos, grades, armazenamento de escalares e gráficos. Começar com um arquivo no diretório existente |
| tests/test_osciladores.py e tests/test_funcional_rdm.py | Testes físicos de operadores e estados, e testes de integração da busca restrita |

Fluxo: estados locais e H₀,V → restrições M,b → DualEntropico → otimizar_dual → diagnóstico do estado bruto e observáveis → varredura, visualização ou busca externa. A comparação SDP fica em um caminho de validação separado, opcional. A camada de visualização usa Matplotlib, com QuTiP carregado apenas no caminho de Wigner. Notebooks podem consumir as funções depois da validação por scripts, sem reimplementar o solver.

Para generalizar a N, usar dims e um mapeamento com uma marginal por índice desde o começo. O recorte analítico de térmicas iguais, a bipartição padrão e a busca externa continuam especializados em dois modos até haver um caso de teste para a extensão.
