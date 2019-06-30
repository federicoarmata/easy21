from src.algos import *
from src.utils import *

import random
import time


#timing the script
t0 = time.time()
random.seed(2)

gamma = 1

### Linear function approximation:
number_of_episodes = 200000
lamdba = 0.1
lamdba_vector = np.linspace(0,1,11)
alpha = 0.01      # step size
epsilon = 0.05    # exploration probability

Q_lfa, _ = linear_function_approximation(
    number_of_episodes, 
    lamdba, 
    gamma,
    alpha,
    epsilon,
    learning_curve=False)

plot_optimal_value_function(Q_lfa, 'lfa', number_of_episodes, lamdba)
plot_optimal_policy(Q_lfa, 'lfa', number_of_episodes, lamdba)

lamdba_vector = np.linspace(0,1,11)

plot_learning_curve_lfa(
    10000, 
    lamdba_vector, 
    gamma,
    alpha,
    epsilon,
    'lfa')

t1 = time.time()

total = t1-t0
print(total)