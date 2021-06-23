import networkx as nx
import numpy as np
import time
import json
from recordclass import recordclass
from dwave.system import LeapHybridSampler
import dwave_networkx.algorithms.tsp as dnx


from random_graph import create_and_save_random_data_model, read_data_model
from google_solver import solve_TSP


Experiment = recordclass('Experiment',
                         'tsp_calculation_start_time tsp_calculation_end_time N method path cost distance_matrix')
MAX_ATTEMPTS=10
path_to_experiments_results = "/home/knalecz/Pulpit/QuantumComputing/quantum_gsp/results/"
path_to_data = "/home/knalecz/Pulpit/QuantumComputing/quantum_gsp/data/random_complete_graphs/"


# from mstechly TSP_utilities.calculate_cost
def calculate_cost(distance_matrix, path):
    cost = 0
    for i in range(len(path)):
        a = i%len(path)
        b = (i+1)%len(path)
        cost += distance_matrix[path[a]][path[b]]
    return cost
    

def serialize_numpy_array(obj):
    if type(obj).__module__ == np.__name__:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj.item()
    raise TypeError('Unknown type:', type(obj))  


def save_experiment(exp, json_filename):
    with open(path_to_experiments_results + json_filename, 'w') as outfile:
        json.dump(exp.__dict__, outfile, default=serialize_numpy_array)


for N in [50]:#, 100]: #TODO [5, 10, 20, 50, 100, 200, 500]:
    print(f'\nCalculations for N={N}:')
    #model = create_and_save_random_data_model(N, path_to_data + f'random_complete_graph_{N}.npy')
    model = read_data_model(path_to_data + 'random_complete_graph_50.npy')
    G = nx.from_numpy_matrix(model)
    assert(len(G.edges) == N*(N-1)//2)

    print('\n----------------- CLASSICAL CALCULATIONS -----------------')
    start = time.time()
    google_path = solve_TSP(model)[0][:-1]
    google_cost = calculate_cost(model, google_path)
    end = time.time()
    exp = Experiment(tsp_calculation_start_time = start,
                     tsp_calculation_end_time = end,
                     N = N,
                     method = "Google OR-Tools",
                     path = google_path,
                     cost = google_cost,
                     distance_matrix = model)
    print(f'google_path: {google_path}, google_cost: {google_cost}')                     
    save_experiment(exp, f'exp_classical.N{N}.json')


    print('\n------------------ QUANTUM CALCULATIONS ------------------')
    for max_time in [36]: # dla N=50 time limit musi być > 6 [6, 12]:
        print(f'max_time={max_time}')
        for attempt in range(MAX_ATTEMPTS):
            start = time.time()
            dwave_path = dnx.traveling_salesperson(G, LeapHybridSampler(), start=0, time_limit=max_time)
            dwave_cost = calculate_cost(model, dwave_path)
            end = time.time()            
            exp = Experiment(tsp_calculation_start_time = start,
                             tsp_calculation_end_time = end,
                             N = N,
                             method = "D-Wave LeapHybridSampler",
                             path = dwave_path,
                             cost = dwave_cost,
                             distance_matrix = model)
            print(f'dwave_path: {dwave_path}, dwave_cost: {dwave_cost}, attempt={attempt}')                     
            save_experiment(exp, f'exp_quantum.t{max_time}.a{attempt}.N{N}.json')                        
            if dwave_cost <= google_cost:
                print(f'Quantum algorithm has caught up with or surpassed the classical one! time_limit={max_time}, N={N}')
                break
        else:
            continue
        break


