import math
import numpy as np

def posicao_efetuador(theta1_deg, theta2_deg):
    L1 = 20.0
    L2 = 15.0

    # Converter os ângulos de graus => pacote math)
    t1 = math.radians(theta1_deg)
    t2 = math.radians(theta2_deg)

    # Cálculo das coordenadas
    X_U = L1 * math.cos(t1) + L2 * math.cos(t1 + t2)
    Y_U = L1 * math.sin(t1) + L2 * math.sin(t1 + t2)

    return round(X_U, 1), round(Y_U, 1)

def matriz_transformacao(theta1_deg, theta2_deg):
    L1 = 20.0
    L2 = 15.0

    t1 = np.radians(theta1_deg)
    t2 = np.radians(theta2_deg)
    phi = t1 + t2

    # Coordenadas de posição do efetuador final (Xu, Yu)
    X_U = L1 * np.cos(t1) + L2 * np.cos(phi)
    Y_U = L1 * np.sin(t1) + L2 * np.sin(phi)

    # Montagem da Matriz de Transformação Homogênea 3x3
    T = np.array(
        [
            [np.cos(phi), -np.sin(phi), X_U],
            [np.sin(phi), np.cos(phi), Y_U],
            [0.0, 0.0, 1.0],
        ]
    )

    return T

if __name__ == "__main__":
    x, y = posicao_efetuador(30, 45)
    print(f"X_U: {x} cm, Y_U: {y} cm")
    T = matriz_transformacao(x, y)
    print(f"T: {T}")