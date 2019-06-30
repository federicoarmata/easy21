##################################################################
### RL: Easy21 algorithms:
###     1) Monte Carlo Control
###     2) Sarsa Lambda
###     3) Linear Function Approximation        
##################################################################

import random
import numpy as np
import pickle


def generate_new_card():
    """
       Generates the new card called either by the 
       dealer or the player.

       Return: 
       - tuple object containing the value of the new card new_card_value
        and its color (new_card_colour).
    """
    
    new_card_value = random.randint(1, 10)
    new_card_colour = np.random.choice(["Blue", "Red"], 1, p=[2/3, 1/3])

    return new_card_value, str(new_card_colour[0])


def adding_card(variable_to_sum, new_card_value, new_card_colour):
    """
       Adds the new card to the current sum. 

       Parameters: 
       - variable_to_sum: quantity we want to sum, will be either player_sum 
         or dealer_sum;
       - new_card_value: value of the new card called;
       - new_card_colour: colour of the new card following the probability 
         distribution given;
         
       Return:
       - variable_to_sum: scalar number representing the sum of the value of 
         the cards called.
    """

    if new_card_colour=="Blue":
        variable_to_sum += new_card_value
    else:
        variable_to_sum -= new_card_value

    return variable_to_sum


def initialise_variables():
    """
       Initiliase actions, states, Q and V 
       (lookup tables).

       Return: 
       - Q: action value function, numpy matrices;
       - N: numpy matrix;
       - d_states, d_actions: dictionaries for states
         and actions;
       - states: numpy matrix;
       - actions: numpy array.
    """

    actions = np.array(['hit', 'stick'])
    states = np.zeros(shape=(11,22), dtype=object)
    states = create_states(states)

    Q = np.zeros((len(actions), states.shape[0]*states.shape[1]), dtype=object)
    N = np.zeros((len(actions), states.shape[0]*states.shape[1]), dtype=object)

    values_states = list(range(0, states.shape[0]*states.shape[1]))
    d_states = dict(
        zip(
            values_states,
            states.reshape(1, states.shape[0]*states.shape[1]).T.tolist()
            ))
    d_states = {y[0]:x for x, y in d_states.items()}

    values_actions = [0, 1]
    d_actions = {"hit":0, "stick":1}

    return Q, N, d_states, d_actions, states, actions


def step(state, action):
    """
       Creates the environment for the game.

       Parameters: 
       - state: tuple object which contains the dealer first card and the sum 
         of the cards of the player: (dealer_first_card, player_sum);
       - action: string giving the player's action, which can be
         "hit" or "stick";

       Return:
       - next_state: tuple object giving the updated state;
       - reward: 
                 +1 = The player wins
                 0 = Draw
                 -1 = The player loses
       - terminal_state: boolean, if True next_state is terminal.
    """

    next_state = ()
    cards_called_by_dealer = []

    lower_bound = 0
    dealer_upper_bound = 17
    upper_bound = 21

    reward = 0
    dealer_sum = 0

    (dealer_first_card, player_sum) = state

    if action == "hit":
        new_card_value, new_card_colour = generate_new_card()
        player_sum = adding_card(player_sum, new_card_value, new_card_colour)

        if not (lower_bound < player_sum <= upper_bound):
            reward = -1
            terminal_state = True

        else:
            reward = 0 
            next_state = (dealer_first_card, player_sum)
            terminal_state = False

    elif action == "stick":
        terminal_state = True
        dealer_sum += dealer_first_card

        while lower_bound < dealer_sum  < dealer_upper_bound:
            new_card_value, new_card_colour = generate_new_card()
            cards_called_by_dealer.append((new_card_value, new_card_colour))
            dealer_sum = adding_card(dealer_sum, new_card_value, new_card_colour)

        if not (lower_bound < dealer_sum <= upper_bound) \
            or player_sum > dealer_sum:
            reward = 1

        elif dealer_sum == player_sum:
            reward = 0

        elif dealer_sum > player_sum:
            reward = -1
        
        #print("sequence of cards called by dealer and sum:")
        #print(cards_called_by_dealer, dealer_sum)

    return next_state, reward, terminal_state


def create_states(states):

    for index, _ in np.ndenumerate(states):
        states[index] = index

    return states[1:,1:]


def generate_random_action():

    A = np.random.choice(["hit", "stick"], 1, p=[0.5, 0.5])

    return str(A[0])


def generate_greedy_action(S, Q, d_states, d_actions):

    #check in case for state S, Q is still empty
    if not np.any(Q[:, d_states[S]]) == True:
        A = generate_random_action()

    else:
        action_index = np.argmax(Q[:, d_states[S]])

        for action, index in d_actions.items():   
            if index == action_index:
                A = action

    return A


def initialise_state():
    
    first_card_dealer, _ = generate_new_card()
    first_card_player, _ = generate_new_card()

    S = (first_card_dealer, first_card_player)

    return S


def grouped(iterable, n):
    """s -> (s0,s1,s2,...sn-1), 
    (sn,sn+1,sn+2,...s2n-1), 
    (s2n,s2n+1,s2n+2,...s3n-1), ..."""
    return zip(*[iter(iterable)]*n)


def generate_episode(Q, N, N0, d_states, d_actions):
    """
       Generates an episode.
       (Used for Monte Carlo Control)

       Parameters:
       - Q: action value fucntion, numpy matrix;
       - N: numpy matrix;
       - N0: parameter;
       - d_states, d_actions: dictionaries for states
         and actions;

       Return:
       - episode: list object;
       - N: numpy matrix.
    """

    episode=[]
    terminal_state = False

    S1 = initialise_state()

    while not terminal_state:

        epsilon = N0/(N0 + sum(N[:, d_states[S1]]))
        policy = np.random.choice(["random", "greedy"], 1, p=[epsilon, 1-epsilon])

        if str(policy[0]) == "random":
            A = generate_random_action()

        elif str(policy[0]) == "greedy":
            A = generate_greedy_action(S1, Q, d_states, d_actions)

        N[d_actions[A], d_states[S1]] += 1

        S, R, terminal_state = step(S1, A)

        episode.extend([S1, A, R])

        S1 = S

    return episode, N


def monte_carlo_control(
    number_of_episodes, 
    N0=100):
    """
       Implements Monte Carlo Control.

       Parameters: 
       - number_of_episodes: parameter;
       - N0: parameter, default=100;

       Return:
       - Q: action value function, numpy matrix.
    """

    Q, N, d_states, d_actions, _, _ = initialise_variables()
    wins = 0

    for k in range(0, number_of_episodes):
        
        episode, N = generate_episode(Q, N, N0, d_states, d_actions)

        numeric_types = [int, float, complex]
        episode_states_and_actions = [x for x in episode if type(x) not in numeric_types]
        episode_rewards = [x for x in episode if type(x) in numeric_types]  

        G = sum(episode_rewards)

        if 1 in episode_rewards:
            wins += 1

        for s, a in grouped(episode_states_and_actions, 2):
            Q[d_actions[a], d_states[s]] +=  (1/N[d_actions[a], d_states[s]])*(G - Q[d_actions[a], d_states[s]])

        if k % 100000 == 0:
            print("Episode %i, Wins %.3f"%(k, wins/(k+1)))

    #save the final Q to file
    #add path if required: path + 'results/Q_star'
    pickle.dump(Q, 
    open('results/Q_star', 'wb'))
    
    return Q


def sarsa_lambda(
    number_of_episodes, 
    lamdba, 
    N0 = 100, 
    gamma = 1,
    learning_curve=False):

    """
       Implements Sarsa Lambda.

       Parameters: 
       - number_of_episodes, lamdba, N0, 
         gamma: parameters;
       - learning_curve: boolean, True if we want to calculate the 
         mean squared error with the optimal Q generated from monte
         carlo control;

       Return:
       - Q: action value function, numpy matrix;
       - mse: list, mean squared error.
    """

    Q, N, d_states, d_actions, _, _ = initialise_variables()

    mse = []
    wins = 0

    for k in range(0, number_of_episodes):

        #eligibility trace for each state-action pair
        E = np.zeros((len(d_actions), len(d_states)), dtype=object)

        #initialise state, action pair
        S = initialise_state()
        A = generate_random_action()

        #initialise terminal state and wins
        terminal_state = False
        Aprime = A

        while not terminal_state:

            #calling enviroment
            Sprime, R, terminal_state = step(S, A)

            #counting wins
            if R == 1:
                wins += 1

            #generate next epsilon greedy action and calculate td error
            if not terminal_state:
                epsilon = N0/(N0+ sum(N[:, d_states[S]]))
                policy = np.random.choice(["random", "greedy"], 1, p=[epsilon, 1-epsilon])

                if str(policy[0]) == "random":
                    Aprime = generate_random_action()
                else:
                    Aprime = generate_greedy_action(Sprime, Q, d_states, d_actions)

                delta = R + gamma*Q[d_actions[Aprime], d_states[Sprime]] - Q[d_actions[A], d_states[S]]

            else:
                #td error when S (state) is terminal
                delta = R - Q[d_actions[A], d_states[S]]

            N[d_actions[A], d_states[S]] += 1
            E[d_actions[A], d_states[S]] += 1
            alpha = (1/N[d_actions[A], d_states[S]])

            #updating action value function and eligibility trace
            Q += alpha*delta*E
            E *= gamma*lamdba

            S = Sprime
            A = Aprime

        #importing Qstar from monte carlo control to calculate mse
        if learning_curve is True:
            Q_star = pickle.load(open('results/Q_star', 'rb'))
            mse_value = (np.square(Q - Q_star)).mean(axis=None)
            mse.append(mse_value)

        if k % 100000 == 0:
            if learning_curve is True:
                print("Lambda=%.1f Episode %d, MSE %5.3f, Wins %.3f"%(lamdba, k, mse_value, wins/(k+1)))
            else:
                print("Lambda=%.1f Episode %d, Wins %.3f"%(lamdba, k, wins/(k+1)))


    return Q, mse


def reset():
    theta = np.random.randn(3*6*2).reshape(-1,1)
    wins = 0

    return theta, wins


def feature_vector(S, a):
    """
       Creates the feature vector for lfa.
       x_space, y_space and actions define the cuboids.

       Parameter: 
       - S: state, tuple object = (delear first card, player sum);
       - a: 0 or 1. Note 0 == "hit, 1 == "stick";

       Return:
       - feature_vector: numpy array of shape (1, 36).
    """

    feature = np.ones([3, 6, 2])
    x, y = S
    feature[:, :, 1 - a] = 0

    x_space = [[1, 4], [4, 7], [7, 10]]
    y_space = [[1, 6], [4, 9], [7, 12], [10, 15], [13, 18], [16, 21]]

    for i in range(len(x_space)):
        if x < x_space[i][0] or x > x_space[i][1]:
            feature[i, :, :] = 0

    for i in range(len(y_space)):
        if y < y_space[i][0] or y > y_space[i][1]:
            feature[:, i, :] = 0

    return feature.flatten().reshape(1,-1)


def approximate_Q(S, a, theta):
    """
       Calculates the scalar product between 
       the feature vector and theta (the weights).

       Parameters:
       - S: state, tuple object = (delear first card, player sum);
       - a: 0 or 1. Note 0 == "hit, 1 == "stick";
       - theta: vector of weights, shape (36, 1);

       Return: scalar number.
    """
    return np.dot(feature_vector(S, a), theta).flatten()[0]


def Q_table(Q, theta, d_states, states, actions):
    """
       Creates lookup table for the action value function.
       Each element given by the result of the approximated value
       coming from lfa. 

       Parameters:
       - Q: action value function, numpy matrix;
       - theta: vector of weights, shape (36, 1);
       - d_states: dictionary for states;
       - states: numoy ndarray of states;
       - actions: numpy array of actions;

       Return: 
       - Q: action value function, numpy matrix.
    """

    for i in range(1, states.shape[0]+1):
            for j in range(1, states.shape[1]+1):
                state_index = d_states[(i, j)]
                for k in range(len(actions)):
                    Q[k, state_index] = approximate_Q((i,j), k, theta)

    return Q


def generate_greedy_action_lfa(S, theta):

    """
       Generates greedy action for lfa.

       Parameters: 
       - S: state, tuple object = (delear first card, player sum);
       - theta: vector of weights, shape (36, 1);

       Return:
       -A: action, string 
    """

    q0 = approximate_Q(S, 0, theta)
    q1 = approximate_Q(S, 1, theta)

    if q0 == q1:
        A = generate_random_action()
    elif q0 > q1:
        A = "hit"
    else:
        A = "stick"
    
    return A

def linear_function_approximation(
    number_of_episodes, 
    lamdba, 
    gamma=1,
    alpha = 0.01,
    epsilon = 0.05,
    learning_curve=False):
    
    """
       Implements linear function approximation (lfa) for Sarsa lambda.

       Parameters: 
       - number_of_episodes, lamdba, gamma, alpha (step size), 
         epsilon (exploration probability): parameters, scalar numbers;
       - learning_curve: boolean, True if we want to calculate the 
         mean squared error with the optimal Q generated from monte
         carlo control;

       Return:
       - Q: action value function, numpy matrix;
       - mse: list, mean squared error.
    """

    Q, N, d_states, d_actions, states, actions = initialise_variables()
    
    mse = []
    theta, wins = reset()

    for k in range(0, number_of_episodes):

        #eligibility trace for each state-action pair
        E = np.zeros_like(theta)

        #initialise state, action pair
        S = initialise_state()
        A = generate_random_action()

        #initialise terminal state and wins
        terminal_state = False
        Aprime = A

        while not terminal_state:

            #calling enviroment
            Sprime, R, terminal_state = step(S, A)

            #generate next epsilon greedy action and calculate td error
            if not terminal_state:
                policy = np.random.choice(["random", "greedy"], 1, p=[epsilon, 1-epsilon])

                if str(policy[0]) == "random":
                    Aprime = generate_random_action()
                else:
                    Aprime = generate_greedy_action_lfa(Sprime, theta)

                delta = R + gamma*approximate_Q(Sprime, d_actions[Aprime], theta) - approximate_Q(S, d_actions[A], theta)

            else:
                #td error when S (state) is terminal
                delta = R - approximate_Q(S, d_actions[A], theta)

            E = gamma*lamdba*E + feature_vector(S, d_actions[A]).T
            gradient = alpha*delta*E
            theta += gradient

            S = Sprime
            A = Aprime
        
        #counting wins
        if R == 1:
            wins += 1

        Q = Q_table(Q, theta, d_states, states, actions)
        #print(Q)

        #importing Qstar from monte carlo control to calculate mse
        if learning_curve is True:
            Q_star = pickle.load(open('results/Q_star', 'rb'))
            mse_value = (np.square(Q - Q_star)).mean(axis=None)
            mse.append(mse_value)

        if k % 100000 == 0:
            if learning_curve is True:
                print("Lambda=%.1f Episode %d, MSE %5.3f, Wins %.3f"%(lamdba, k, mse_value, wins/(k+1)))
            else:
                print("Lambda=%.1f Episode %d, Wins %.3f"%(lamdba, k, wins/(k+1)))


    return Q, mse











