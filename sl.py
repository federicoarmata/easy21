from src.algos import *
from src.utils import *

import random
import time

#timing the script
t0 = time.time()
random.seed(2)

gamma = 1
N0 = 100


### Sarsa lambda:
number_of_episodes = 500000
lamdba = 0.1
lamdba_vector = np.linspace(0,1,11)

Q_sarsa, _ = sarsa_lambda(
    number_of_episodes, 
    lamdba,
    N0,
    gamma,
    learning_curve=False)

plot_optimal_value_function(Q_sarsa, 'sarsa', number_of_episodes, lamdba)
plot_optimal_policy(Q_sarsa, 'sarsa', number_of_episodes, lamdba)

plot_learning_curve(
    10000, 
    lamdba_vector, 
    N0, 
    gamma,
    'sl')


t1 = time.time()

total = t1-t0
print(total)