import numpy as np
from scipy.spatial.distance import squareform


# utworzenie losowego grafu pełnego o zadanej liczbie wierzchołków
def create_and_save_random_data_model(N, filename):
    model = squareform(np.random.randint(0, 2000, N*(N-1)//2))
    with open(filename, 'wb') as f:
        np.save(f, model)
    return model


# odczyt grafu z pliku
def read_data_model(filename):
    model = None
    with open(filename, 'rb') as f:
        model = np.load(f)
    return model
