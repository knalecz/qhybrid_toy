from dimod import ConstrainedQuadraticModel, Binary, Integer, quicksum, sampleset
from dwave.system import LeapHybridCQMSampler
import numpy as np
from scipy.spatial.distance import squareform


cqm = ConstrainedQuadraticModel()

# Prepare TSP problem instance
nodes_num = 5
edges_num = nodes_num * (nodes_num - 1) // 2
# distance_matrix = squareform(np.random.randint(0, 2000, edges_num))

distance_matrix = [
    [0, 469, 1619, 448, 806],
    [469, 0, 83, 185, 1690],
    [1619, 83, 0, 1672, 87],
    [448, 185, 1672, 0, 1000],
    [806, 1690, 87, 1000, 0],
]

# Declare variables
x = [[Binary(f"x_{i}_{j}") for j in range(nodes_num)] for i in range(nodes_num)]

u = [
    Integer(f"u_{i}", lower_bound=2, upper_bound=nodes_num - 1)
    for i in range(1, nodes_num)
]

# --------------------------------- Objectives --------------------------------

# Build and set an objective - minimalization of cost function
objective = quicksum(
    distance_matrix[i][j] * x[i][j] for i in range(nodes_num) for j in range(nodes_num)
)
cqm.set_objective(objective)

# -------------------------------- Constraints --------------------------------

# Add the first assignment constraint: every node can only appear once in the cycle
for i in range(nodes_num):
    cqm.add_constraint(quicksum(x[i]) == 1, label=f"node_{i}_appearing_once_in_cycle")

# Add the second assignment constraint: for each time a node has to occur in the cycle
for j in range(nodes_num):
    cqm.add_constraint(
        quicksum(np.array(x)[:, j]) == 1, label=f"node_{j}_has_to_be_used"
    )

print(distance_matrix)

# Add arc constraints
for i in range(1, nodes_num):
    for j in range(1, nodes_num):
        cqm.add_constraint(u[i - 1] - u[j - 1] + (nodes_num - 1) * x[i][j] <= nodes_num - 2)

# ------------------------- Submit to the CQM Sampler -------------------------

cqm_sampler = LeapHybridCQMSampler()
sampleset = cqm_sampler.sample_cqm(cqm, label="TSP - MTZ formulation")
print(sampleset.info)

# ---------------------------- Process the results-- --------------------------

feasible_solutions = np.where(sampleset.record.is_feasible == True)
