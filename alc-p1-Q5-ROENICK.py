import numpy as np

def resolver_sistema_lu_estendida(A, b):
    # Converter entradas para arrays do NumPy em ponto flutuante
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)

    # Garantir que b seja explicitamente um vetor coluna (n, 1)
    if b.ndim == 1:
        b = b.reshape(-1, 1)

    n = A.shape[0]

    # Validação de dimensões
    if A.shape[0] != A.shape[1]:
        raise ValueError("A matriz A deve ser quadrada.")
    if b.shape[0] != n or b.shape[1] != 1:
        raise ValueError("O vetor b deve possuir dimensões (n, 1).")

    # 1. Construção da Matriz Estendida [A | I]
    AI = np.zeros((n, 2 * n), dtype=float)

    for i in range(n):
        for j in range(n):
            AI[i, j] = A[i, j]

    I = np.eye(n, dtype=float)
    for i in range(n):
        for j in range(n):
            AI[i, n + j] = I[i, j]

    # Inicialização da matriz L como a matriz identidade
    L = np.eye(n, dtype=float)

    # 2. Eliminação de Gauss para transformar o lado esquerdo de [A | I] na matriz U
    for i in range(n):
        # Verificação de pivô nulo
        if AI[i, i] == 0:
            raise ZeroDivisionError(
                f"Pivô nulo encontrado na posição ({i}, {i}). "
                "A decomposição LU sem pivoteamento falhou.\n"
                "Sugestão de função alternativa: Utilize a decomposição PLU (com pivoteamento parcial) "
                "via `scipy.linalg.lu_factor` / `scipy.linalg.lu_solve` ou a função `numpy.linalg.solve`."
            )

        # Eliminação abaixo da diagonal principal
        for k in range(i + 1, n):
            # Coeficiente de ajuste (multiplicador)
            fator = AI[k, i] / AI[i, i]

            # O coeficiente de ajuste com sinal invertido em relação à operação compõe L
            L[k, i] = fator

            # Operação de linha na matriz estendida [A | I]
            for j in range(i, 2 * n):
                AI[k, j] = AI[k, j] - fator * AI[i, j]

    # 3. Extração da Matriz U (submatriz n x n do lado esquerdo de [A | I])
    U = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            U[i, j] = AI[i, j]

    # 4. Substituição Progressiva: Resolver Ly = b (y como vetor coluna)
    y = np.zeros((n, 1), dtype=float)
    for i in range(n):
        soma = 0.0
        for j in range(i):
            soma += L[i, j] * y[j, 0]
        y[i, 0] = b[i, 0] - soma

    # 5. Substituição Regressiva: Resolver Ux = y (x como vetor coluna)
    x = np.zeros((n, 1), dtype=float)
    for i in range(n - 1, -1, -1):
        soma = 0.0
        for j in range(i + 1, n):
            soma += U[i, j] * x[j, 0]
        x[i, 0] = (y[i, 0] - soma) / U[i, i]

    # Retorno na ordem exigida: Matriz L, Matriz U e Vetor Coluna Solução x
    return L, U, x


# Exemplo de execução:
if __name__ == "__main__":
    A = np.array([[1,-2, 1], [2, -3, 1], [1, 4, 2]], dtype=float)

    # Definição explícita do vetor b como vetor coluna (n, 1)
    b = np.array([[-1], [-3], [7]], dtype=float)

    L, U, x = resolver_sistema_lu_estendida(A, b)

    print("Matriz L:")
    print(L)
    print("\nMatriz U:")
    print(U)
    print("\nVetor Solução x (coluna):")
    print(x)
