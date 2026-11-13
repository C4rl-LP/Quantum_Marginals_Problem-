"""Testes numéricos determinísticos do dual entrópico e restrições marginais."""

import unittest
import warnings

import numpy as np

from modulos.dual_entropico import DualEntropico
from modulos.marginais import construir_restricoes_marginais
from modulos.operadores import partial_trace
from modulos.otimizacao import otimizar_dual


class TestDualEntropico(unittest.TestCase):
    def test_gradiente_analitico_por_diferencas_finitas(self):
        rng = np.random.default_rng(712)
        d, n = 3, 4

        def hermitian_random():
            matrix = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
            return (matrix + matrix.conj().T) / 2

        problem = DualEntropico(
            H=hermitian_random(),
            Q=[hermitian_random() for _ in range(n)],
            q=rng.normal(size=n),
            epsilon=0.8,
        )
        u = rng.normal(scale=0.08, size=n)
        _, analytic = problem.avaliar(u)
        step = 1e-6
        finite_difference = np.empty(n)
        for index in range(n):
            direction = np.zeros(n)
            direction[index] = step
            finite_difference[index] = (
                problem.valor(u + direction) - problem.valor(u - direction)
            ) / (2 * step)
        np.testing.assert_allclose(analytic, finite_difference, rtol=2e-6, atol=2e-7)

    def test_estado_exponencial_e_hermitiano_positivo_definido(self):
        sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
        sigma_z = np.diag([1.0, -1.0])
        problem = DualEntropico(
            H=0.2 * sigma_z,
            Q=[sigma_x, sigma_z],
            q=[0.0, 0.0],
            epsilon=0.7,
        )
        gamma = problem.estado([0.1, -0.2])
        np.testing.assert_allclose(gamma, gamma.conj().T, atol=1e-13)
        self.assertGreater(float(np.min(np.linalg.eigvalsh(gamma))), 0.0)

    def test_construcao_de_marginais_sobrepostas_e_redundancias(self):
        rho_a = np.array([[0.7, 0.1], [0.1, 0.3]], dtype=complex)
        rho_b = np.array([[0.4, 0.08j], [-0.08j, 0.6]], dtype=complex)
        rho_c = np.array([[0.6, 0.0], [0.0, 0.4]], dtype=complex)
        rho_abc = np.kron(np.kron(rho_a, rho_b), rho_c)
        rho_ab = partial_trace(rho_abc, keep=[0, 1], dims=[2, 2, 2])
        rho_bc = partial_trace(rho_abc, keep=[1, 2], dims=[2, 2, 2])

        restrictions = construir_restricoes_marginais(
            [2, 2, 2], [rho_ab, rho_bc], [[0, 1], [1, 2]]
        )
        self.assertGreater(len(restrictions.indices_redundantes), 0)
        self.assertEqual(len(restrictions.Q), len(restrictions.q))
        self.assertEqual(restrictions.quantidade_original,
                         len(restrictions.Q) + len(restrictions.indices_redundantes))

        # As restrições compactadas continuam descrevendo a extensão produto.
        problem = DualEntropico(np.zeros((8, 8)), restrictions.Q, restrictions.q, 1.0)
        optimized = otimizar_dual(
            problem,
            u_inicial=np.full(len(problem.Q), 0.05),
            options={"maxiter": 1200},
        )
        self.assertTrue(optimized["restricoes_satisfeitas"], optimized["mensagem"])
        np.testing.assert_allclose(optimized["estado"], rho_abc, atol=2e-6, rtol=2e-6)

    def test_reconstrucao_produto_de_marginais_de_posto_completo(self):
        rho_a = np.array([[0.7, 0.1], [0.1, 0.3]], dtype=complex)
        rho_b = np.array([[0.4, 0.08j], [-0.08j, 0.6]], dtype=complex)
        restrictions = construir_restricoes_marginais(
            [2, 2], [rho_a, rho_b], [[0], [1]]
        )
        problem = DualEntropico(np.zeros((4, 4)), restrictions.Q, restrictions.q, 1.0)
        result = otimizar_dual(problem)
        self.assertTrue(result["convergencia_numerica"], result["mensagem"])
        self.assertTrue(result["restricoes_satisfeitas"], result["mensagem"])
        np.testing.assert_allclose(result["estado"], np.kron(rho_a, rho_b), atol=2e-7, rtol=2e-7)

    def test_dois_qubits_com_interacao_ising_compara_solucao_analitica(self):
        identity = np.eye(2, dtype=complex)
        sigma_z = np.diag([1.0, -1.0])
        zz = np.kron(sigma_z, sigma_z)
        J, epsilon = 0.7, 1.2
        maximally_mixed = identity / 2
        restrictions = construir_restricoes_marginais(
            [2, 2], [maximally_mixed, maximally_mixed], [[0], [1]]
        )
        problem = DualEntropico(-J * zz, restrictions.Q, restrictions.q, epsilon)
        result = otimizar_dual(problem, tolerancia_residuos=2e-8)

        analytic = (np.eye(4) + np.tanh(J / epsilon) * zz) / 4
        self.assertTrue(result["convergencia_numerica"], result["mensagem"])
        self.assertTrue(result["restricoes_satisfeitas"], result["mensagem"])
        np.testing.assert_allclose(result["estado"], analytic, atol=2e-8, rtol=2e-8)
        exact_dual = -epsilon * np.log(4 * np.cosh(J / epsilon)) - epsilon
        self.assertAlmostEqual(result["valor_dual"], exact_dual, places=8)
        self.assertAlmostEqual(np.trace(result["estado"]).real, 1.0, places=9)
        self.assertAlmostEqual(np.trace(zz @ result["estado"]).real,
                               np.tanh(J / epsilon), places=8)

        rho_a = partial_trace(result["estado"], keep=[0], dims=[2, 2])
        rho_b = partial_trace(result["estado"], keep=[1], dims=[2, 2])
        np.testing.assert_allclose(rho_a, maximally_mixed, atol=2e-8)
        np.testing.assert_allclose(rho_b, maximally_mixed, atol=2e-8)

    def test_otimizacao_identifica_multiplicadores_redundantes(self):
        identity = np.eye(2, dtype=complex)
        problem = DualEntropico(np.zeros((2, 2)), [identity, identity], [1.0, 1.0], 1.0)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = otimizar_dual(problem)
        self.assertTrue(any("redundantes" in str(item.message) for item in caught))
        report = result["redundancia_multiplicadores"]
        self.assertEqual(report["nulidade"], 1)
        self.assertTrue(report["alvos_consistentes"])

    def test_marginais_singulares_e_nao_atingimento_potencial(self):
        pure_zero = np.array([[1, 0], [0, 0]], dtype=complex)
        restrictions = construir_restricoes_marginais([2], [pure_zero], [[0]])
        problem = DualEntropico(np.zeros((2, 2)), restrictions.Q, restrictions.q, 1.0)

        # Para todo u finito, exp de um operador hermitiano tem posto completo;
        # portanto não pode reproduzir exatamente uma marginal pura.
        finite_state = problem.estado(np.zeros(len(problem.Q)))
        self.assertGreater(float(np.min(np.linalg.eigvalsh(finite_state))), 0.0)
        finite_marginal = partial_trace(finite_state, keep=[0], dims=[2])
        self.assertGreater(float(np.min(np.linalg.eigvalsh(finite_marginal))), 0.0)

        result = otimizar_dual(
            problem,
            options={"maxiter": 30},
            tolerancia_residuos=1e-12,
        )
        self.assertEqual(
            result["restricoes_satisfeitas"],
            result["norma_residuos_inf"] <= result["tolerancia_residuos"],
        )
        # Sucesso numérico do método, se ocorrer, não declara por si só exatidão.
        self.assertIn("convergencia_numerica", result)
        self.assertIn("restricoes_satisfeitas", result)

    def test_marginais_sobrepostas_incompativeis_sao_rejeitadas(self):
        rho_a = np.diag([1.0, 0.0])
        rho_ab = np.eye(4, dtype=complex) / 4
        with self.assertRaisesRegex(ValueError, "incompatíveis"):
            construir_restricoes_marginais(
                [2, 2], [rho_a, rho_ab], [[0], [0, 1]]
            )

    def test_validacao_de_dimensao_hermiticidade_e_epsilon(self):
        with self.assertRaisesRegex(ValueError, "hermitiana"):
            DualEntropico(np.array([[0, 1], [0, 0]]), [], [], 1.0)
        with self.assertRaisesRegex(ValueError, "dimensão"):
            DualEntropico(np.eye(2), [np.eye(3)], [0.0], 1.0)
        with self.assertRaisesRegex(ValueError, "epsilon"):
            DualEntropico(np.eye(2), [], [], 0.0)

    def test_exponencial_instavel_levanta_erro_numerico_explicito(self):
        identity = np.eye(2, dtype=complex)
        problem = DualEntropico(np.zeros((2, 2)), [identity], [1.0], 1.0)
        with self.assertRaisesRegex(OverflowError, "exponencial|Gamma_u"):
            problem.estado([1000.0])


if __name__ == "__main__":
    unittest.main()
