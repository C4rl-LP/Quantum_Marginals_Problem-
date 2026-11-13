import numpy as np
from scipy.linalg import expm
from .validacoes import validar_dados_dual, validar_vetor_real


class DualEntropico:

    def __init__(self, H, Q, q, epsilon):
        self.H, self.Q, self.q, self.epsilon = validar_dados_dual(H, Q, q, epsilon)

    def estado(self, u):
        u = validar_vetor_real(u, "u", len(self.Q))
        A = -np.asarray(self.H, dtype=complex).copy()

        for ui, Qi in zip(u, self.Q):
            A += ui * Qi

        return expm(A / self.epsilon)

    def avaliar(self, u):
        Gamma = self.estado(u)

        valor = (
            np.dot(u, self.q)
            - self.epsilon * np.trace(Gamma).real
        )

        gradiente = np.array([
            qi - np.trace(Qi @ Gamma).real
            for qi, Qi in zip(self.q, self.Q)
        ])
        return valor, gradiente

    def residuos(self, u):
        _, gradiente = self.avaliar(u)
        return -gradiente
        
