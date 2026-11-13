# Registro de decisões

## DEC-001 — Formular a existência de extensão como factibilidade SDP

Status: Confirmada
Data: não identificada
Contexto: O problema pergunta pela existência de uma matriz global que satisfaça positividade, normalização e marginais prescritas.
Decisão: Usar restrições semidefinidas e igualdades lineares para resolver a factibilidade.
Justificativa: É a formulação descrita no `GUIA_DO_PROJETO.md` e implementada em `build_qmp_sdp`.
Consequências: O status numérico deve ser acompanhado por diagnóstico da solução e reconstrução de marginais.
Alternativas consideradas: não documentadas.
Evidências: `GUIA_DO_PROJETO.md`; `modulos/`.

## DEC-002 — Codificar igualdades em base Hermitiana

Status: Confirmada
Data: não identificada
Contexto: A variável e marginais são operadores Hermitianos; a formulação usa SpaceCore/SDPLab.
Decisão: Representar restrições em coordenadas de base Hermitiana ortonormal.
Justificativa: Explicitada no guia e refletida em `base_hermitiana` e `build_qmp_sdp`.
Consequências: Convenções de produto interno e transposição do operador SDPLab são críticas.
Alternativas consideradas: não documentadas.
Evidências: `GUIA_DO_PROJETO.md`; `modulos/`.

## DEC-003 — Usar ordem tensorial explícita para rotular subsistemas

Status: Confirmada
Data: não identificada
Contexto: Traço parcial e embeddings dependem da ordenação dos fatores.
Decisão: Usar a ordem de Kronecker/tensor; em três qubits, `A=0`, `B=1`, `C=2`.
Justificativa: Convenção declarada no AGENTS e no guia.
Consequências: Mudanças em parser, dimensões ou chaves precisam validar índices e ordenação.
Alternativas consideradas: não documentadas.
Evidências: `AGENTS.md`; `GUIA_DO_PROJETO.md`.

## DEC-004 — Separar a API reutilizável por responsabilidade

Status: Confirmada
Data: 2026-11-12
Contexto: O módulo único misturava primitivas matriciais, SDP, validação e geradores experimentais, com imports/dependências sem necessidade para todas as funções.
Decisão: Organizar a implementação no pacote `modulos` (`estados`, `operadores`, `validacao`, `sdp`, `experimentos`) e manter `funcoes_fundamentais.py` como reexportação compatível.
Justificativa: Facilita localizar, documentar e ampliar cada área, preservando os consumidores que ainda importem o módulo antigo.
Consequências: Novos códigos devem usar `from modulos import ...`; a camada antiga pode ser removida em uma migração futura coordenada.
Alternativas consideradas: mover todas as funções para um único arquivo dentro do pacote.
Evidências: `modulos/`; notebooks de exemplos migrados.

## DEC-005 — Implementar o dual entrópico pela exponencial hermitiana

Status: Confirmada
Data: 2026-11-12
Contexto: O dual de energia com regularizador `Tr(Gamma log(Gamma) - Gamma)` exige avaliar uma exponencial matricial global e suas expectativas.
Decisão: Calcular `Gamma_u = exp((sum_i u_i Q_i - H)/epsilon)` por diagonalização espectral do operador hermitiano efetivo, sem normalização e sem deslocamento `-I`.
Justificativa: A fórmula segue a convenção entrópica especificada; a diagonalização usa apenas NumPy e conserva a estrutura hermitiana.
Consequências: A restrição de traço não é implícita; deve ser incluída em `Q, q` quando fizer parte do problema. `avaliar` retorna valor e gradiente a partir de uma única exponencial.
Alternativas consideradas: exponencial genérica de SciPy; normalização do estado exponencial.
Evidências: `modulos/dual_entropico.py`; `tomography.ipynb` para o uso de exponenciais matriciais no projeto.
