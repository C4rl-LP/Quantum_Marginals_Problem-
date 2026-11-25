import numpy as np
from scipy.linalg import expm
from .validacao import validar_dados_dual, validar_vetor_real


class DualEntropico:

    def __init__(self, H, Q, q, epsilon, forma="exponencial"):
        self.H, self.Q, self.q, self.epsilon = validar_dados_dual(H, Q, q, epsilon)

        if forma not in ("exponencial", "logaritmico"):
            raise ValueError(
                "forma deve ser 'exponencial' ou 'logaritmico'."
            )

        self.forma = forma

    def matriz_efetiva(self, u):
        """A(u) = Q(u) - H."""

        u = validar_vetor_real(u, "u", len(self.Q))

        A = -np.asarray(self.H, dtype=complex).copy()

        for ui, Qi in zip(u, self.Q):
            A += ui * Qi

        return A

    def _estado_exponencial(self, A):
        Gamma = expm(A / self.epsilon)
        termo = self.epsilon * np.trace(Gamma).real

        return Gamma, termo

    def _estado_logaritmico(self, A):
        """Calcula o estado normalizado utilizando Log-Sum-Exp."""

        autovalores, V = np.linalg.eigh(A)

        a = autovalores[-1]

        pesos = np.exp((autovalores - a) / self.epsilon)

        Z = pesos.sum()

        probabilidades = pesos / Z

        Gamma = (V * probabilidades) @ V.conj().T

        termo = a + self.epsilon * np.log(Z)

        return Gamma, termo
    
    def _calcular_estado(self, u):
        """Seleciona a formulação do dual."""

        A = self.matriz_efetiva(u)

        if self.forma == "exponencial":
            return self._estado_exponencial(A)

        return self._estado_logaritmico(A)
    
    def estado(self, u):
        Gamma, _ = self._calcular_estado(u)
        return Gamma

    def avaliar(self, u, mask=None):

        Gamma, termo = self._calcular_estado(u)

        valor = (
            np.dot(u, self.q)
            - termo
        )
        if mask is None:
            mask = np.arange(len(self.q))
        gradiente = np.array([
            self.q[i] - np.einsum("ij,ji->", self.Q[i], Gamma).real
            for i in mask
        ])
        return valor, gradiente

    def residuos(self, u):
        _, gradiente = self.avaliar(u)
        return -gradiente
        
