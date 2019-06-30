# easy21

Assignment of David Silver's course on Reinforcement Learning

Course website: http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html


## Monte Carlo Control:

`python mcc.py`

Selecting 10 million episodes with gamma = 1 and N0 = 100,
the following optimal value function and optimal policy are obtained:

![mcc_value_func_10m](results/optimal_value_function_mcc_10milions.pdf)

![mcc_optimal_policy_10m](results/optimal_policy_mcc_10milions.pdf)

While, selecting 500 thousands episodes (same values for the other parameters), 
the optimal value function and optimal policy result:

![mcc_value_func_500k](results/optimal_value_function_mcc_500k.pdf)

![mcc_optimal_policy_500k](results/optimal_policy_mcc_500k.pdf)

## Sarsa(λ):

`python sl.py`

Selecting 500 thousands episodes with gamma = 1, N0 = 100 and lambda = 0.1, 
the following optimal value function and optimal policy are obtained:

![sl_optimal_policy_500k](results/optimal_value_function_sarsa0.1_500k.pdf)

![sl_value_func_500k](results/optimal_policy_sarsa0.1_500k.pdf)

It is then showed the Mean Squared Error (MSE) between the action value function 
obtained with sarsa(λ) and the optimal action value function obtained from Monte
Carlo Control: 

-MSE against lambda, for each lambda 10000 episodes have been evaluated:

![mse_lambda_10000_sl](results/mean_squared_error_against_lambda_10000_sl.pdf)

-MSE against the number of episodes:

![mse_epi_sl](results/mean_squared_error_against_episodes_sl.pdf)


## Linear Function Approximation:

`python lfa.py`

Selecting 200 thousands episodes with gamma = 1, alpha = 0.01, epsilon (step 
size) = 0.05, epsilon (exploration probability) = 0.05 and lambda = 0,
the following optimal value function and optimal policy are obtained:

![lfa0_value_func_200k](results/optimal_value_function_lfa0_200000.pdf)

![lfa0_optimal_policy_200k](results/optimal_policy_lfa0_200000.pdf)

Changing lamdba to 1 (same choice for the other parameters):

![lfa1_value_func_200k](results/optimal_value_function_lfa1_200000.pdf)

![lfa1_optimal_policy_200k](results/optimal_policy_lfa1_200000.pdf)

It is then showed the MSE between the action value function obtained with 
sarsa(λ) with a linear function approximation and the optimal action value 
function obtained from Monte Carlo Control.

-MSE against lambda, for each lambda 10000 episodes have been evaluated:

![mse_epi_lfa](results/mean_squared_error_against_lambda_lfa.pdf)

-MSE against the number of episodes (lambda has been set 
to 0.1):

![mse_epi_lfa](results/mean_squared_error_against_episodes_lfa.pdf)














