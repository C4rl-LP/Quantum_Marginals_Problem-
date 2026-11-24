import unittest

import numpy as np

from modulos.dual_entropico import DualEntropico
from modulos.otimizacao import otimizar_dual


class TestOtimizacaoDualProduto(unittest.TestCase):
    def test_marginais_de_dois_qubits_reconstroem_produto(self):
        identidade = np.eye(2, dtype=complex)
        sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
        sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        sigma_z = np.diag([1, -1]).astype(complex)
        paulis = (sigma_x, sigma_y, sigma_z)

        rho_a = np.diag([0.7, 0.3]).astype(complex)
        rho_b = np.array([[0.6, 0.1], [0.1, 0.4]], dtype=complex)
        
        # Impõe o traço global e as três componentes locais de cada qubit.
        operadores = [np.eye(4, dtype=complex)]
        operadores.extend(np.kron(sigma, identidade) for sigma in paulis)
        operadores.extend(np.kron(identidade, sigma) for sigma in paulis)
        alvos = np.array(
            [1.0]
            + [np.trace(rho_a @ sigma).real for sigma in paulis]
            + [np.trace(rho_b @ sigma).real for sigma in paulis]
        )

        problema = DualEntropico(
            H=np.zeros((4, 4), dtype=complex),
            Q=operadores,
            q=alvos,
            epsilon=1.0,
        )
        resultado = otimizar_dual(
            problema, maxiter=5000, tolerancia=1e-10
        )
        produto_esperado = np.kron(rho_a, rho_b)

        self.assertTrue(resultado["convergiu"])
        self.assertLess(np.linalg.norm(resultado["residuos"]), 1e-5)
        self.assertTrue(
            np.allclose(resultado["estado"], produto_esperado, atol=2e-6, rtol=0)
        )


if __name__ == "__main__":
    unittest.main()
