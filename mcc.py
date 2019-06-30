from src.algos import *
from src.utils import *

import random
import time


#timing the script
t0 = time.time()
random.seed(2)

gamma = 1
N0 = 100


### Monte Carlo Control:
number_of_episodes = 500000

Q_mcc = monte_carlo_control(
    number_of_episodes, 
    N0)

plot_optimal_value_function(Q_mcc, 'mcc', number_of_episodes)
plot_optimal_policy(Q_mcc, 'mcc', number_of_episodes)


t1 = time.time()

total = t1-t0
print(total)