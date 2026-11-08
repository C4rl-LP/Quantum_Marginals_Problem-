# Registro de decisões

## DEC-001 — Formular a existência de extensão como factibilidade SDP

Status: Confirmada
Data: não identificada
Contexto: O problema pergunta pela existência de uma matriz global que satisfaça positividade, normalização e marginais prescritas.
Decisão: Usar restrições semidefinidas e igualdades lineares para resolver a factibilidade.
Justificativa: É a formulação descrita no `GUIA_DO_PROJETO.md` e implementada em `build_qmp_sdp`.
Consequências: O status numérico deve ser acompanhado por diagnóstico da solução e reconstrução de marginais.
Alternativas consideradas: não documentadas.
Evidências: `GUIA_DO_PROJETO.md`; `funcoes_fundamentais.py`.

## DEC-002 — Codificar igualdades em base Hermitiana

Status: Confirmada
Data: não identificada
Contexto: A variável e marginais são operadores Hermitianos; a formulação usa SpaceCore/SDPLab.
Decisão: Representar restrições em coordenadas de base Hermitiana ortonormal.
Justificativa: Explicitada no guia e refletida em `base_hermitiana` e `build_qmp_sdp`.
Consequências: Convenções de produto interno e transposição do operador SDPLab são críticas.
Alternativas consideradas: não documentadas.
Evidências: `GUIA_DO_PROJETO.md`; `funcoes_fundamentais.py`.

## DEC-003 — Usar ordem tensorial explícita para rotular subsistemas

Status: Confirmada
Data: não identificada
Contexto: Traço parcial e embeddings dependem da ordenação dos fatores.
Decisão: Usar a ordem de Kronecker/tensor; em três qubits, `A=0`, `B=1`, `C=2`.
Justificativa: Convenção declarada no AGENTS e no guia.
Consequências: Mudanças em parser, dimensões ou chaves precisam validar índices e ordenação.
Alternativas consideradas: não documentadas.
Evidências: `AGENTS.md`; `GUIA_DO_PROJETO.md`.
