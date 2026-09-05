mport numpy as np

def subst_retroativa(A, b):
    A = np.array(A)
    b = np.array(b).flatten()

    n = A.shape[0]

    if A.shape[0] != A.shape[1]:
        raise ValueError("Erro: A matriz A deve ser quadrada (n x n).")

    if len(b) != n:
        raise ValueError(f"Erro: O tamanho do vetor b ({len(b)}) deve ser igual ao número de linhas de A ({n}).")

    # Matriz Triangular?
    if not np.all(A[np.tril_indices(n, -1)] == 0):
        raise ValueError("Erro: A matriz fornecida não é triangular superior.")

    if np.any(np.diag(A) == 0):
        raise ValueError("Erro: A matriz possui elementos nulos na diagonal principal (sistema sem solução única).")

    x = np.zeros(n) # vetor resposta inicia zerado

    for i in range(n - 1, -1, -1):
        # Soma os termos já calculados: A[i, i+1:] * x[i+1:]
        soma = np.dot(A[i, i + 1:], x[i + 1:])
        x[i] = (b[i] - soma) / A[i, i]
        print(f"o valor de x[{i}] = {x[i]}")

    return x


if __name__ == "__main__":
    matriz_A = [
        [2, 4, -2],
        [0, 1, 9],
        [0, 0, 8]
    ]
    # Vetor de termos independentes
    vetor_b = [2, 4, 10]

    solucao = subst_retroativa(matriz_A, vetor_b)
    print(f"A solução do sistema é o vetor x = {solucao}")