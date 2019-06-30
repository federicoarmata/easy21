from src.algos import *

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_optimal_value_function(
    Q, 
    algorithm,
    number_of_episodes,
    parameter=None):

    # Optimal value function:
    V = np.zeros(shape=(10,21), dtype=float)
    V = np.max(Q, axis=0).reshape((10, 21))

    nx, ny = 10, 21
    x = range(nx)
    y = range(ny)


    fig = plt.figure(figsize=(10,6))
    ax = fig.gca(projection='3d')

    # `plot_surface` expects `x` and `y` data to be 2D
    X, Y = np.meshgrid(x, y)  

    # Plot the surface
    surf = ax.plot_surface(X.T, Y.T, V, rstride = 1, cstride = 1, cmap='jet',
                        linewidth=0, antialiased=False, alpha=0.7)

    fig.suptitle('Optimal value function', fontsize=20)
    ax.set_xlabel('Dealer initial card', fontsize=15)
    ax.set_ylabel('Player cards value', fontsize=15)
    ax.set_zlabel('Value function', fontsize=15)

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

    if parameter is None:
        parameter = ''
    else:
        parameter = round(parameter,1)

    fig.savefig(
        'results/optimal_value_function_' 
        + algorithm + str(parameter) 
        + '_' + str(number_of_episodes)
        + '.pdf')


def plot_optimal_policy(
    Q, 
    algorithm, 
    number_of_episodes,
    parameter=None):

    # Optimal policy
    optimal_policy = np.argmax(Q, axis=0).reshape((10, 21))

    fig = plt.figure()
    plt.imshow(optimal_policy, cmap=plt.cm.bone, aspect='auto')

    fig.suptitle('Optimal policy', fontsize=20)
    plt.xlabel('Player cards value', fontsize=15)
    plt.ylabel('Dealer initial card', fontsize=15)

    plt.colorbar()
    plt.show()

    if parameter is None:
        parameter = ''
    else:
        parameter = round(parameter,1)

    fig.savefig(
        'results/optimal_policy_' 
        + algorithm + str(parameter) 
        + '_' + str(number_of_episodes)
        + '.pdf')
        

def plot_learning_curve(
    number_of_episodes, 
    lamdba, 
    N0, 
    gamma,
    algorithm=None):
    
    n = np.linspace(
        1, number_of_episodes,
        num=number_of_episodes, 
        endpoint=number_of_episodes)

    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()

    mse_final = []

    for x in lamdba:
        
        _, mse = sarsa_lambda(
            number_of_episodes, x, 
            N0, gamma, 
            learning_curve=True)

        mse_final.append(mse[-1])

        ax1.plot(n, mse, label=round(x, 1))

    ax2.plot(lamdba, mse_final, label=round(x, 1))

    ax1.set(
        xlabel='number of episodes', ylabel='mse',
        title='Mean squared error against number of episodes')

    ax2.set(
        xlabel='lambda', ylabel='mse',
        title='Mean squared error against lambda')
    
    ax1.grid()
    ax2.grid()
    ax1.legend(fancybox=True, framealpha=1, shadow=True, borderpad=1)
    #if you want legend box outside plot:
    #(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()

    if algorithm is None:
        algorithm = ''
    else:
        algorithm = '_' + algorithm


    fig1.savefig(
        'results/mean_squared_error_against_episodes' + algorithm + '.pdf')
    fig2.savefig('results/mean_squared_error_against_lambda' + algorithm + '.pdf')


def plot_learning_curve_lfa(
    number_of_episodes, 
    lamdba, 
    gamma,
    alpha,
    epsilon,
    algorithm=None):
    
    n = np.linspace(
        1, number_of_episodes,
        num=number_of_episodes, 
        endpoint=number_of_episodes)

    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()

    mse_final = []

    for x in lamdba:
        
        _, mse = linear_function_approximation(
            number_of_episodes, 
            x, 
            gamma,
            alpha,
            epsilon,
            learning_curve=True)

        mse_final.append(mse[-1])

        ax1.plot(n, mse, label=round(x, 1))

    ax2.plot(lamdba, mse_final, label=round(x, 1))

    ax1.set(
        xlabel='number of episodes', ylabel='mse',
        title='Mean squared error against number of episodes')

    ax2.set(
        xlabel='lambda', ylabel='mse',
        title='Mean squared error against lambda')
    
    ax1.grid()
    ax2.grid()
    ax1.legend(fancybox=True, framealpha=1, shadow=True, borderpad=1)
    #if you want legend box outside plot:
    #(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()

    if algorithm is None:
        algorithm = ''
    else:
        algorithm = '_' + algorithm


    fig1.savefig(
        'results/mean_squared_error_against_episodes' + algorithm + '.pdf')
    fig2.savefig('results/mean_squared_error_against_lambda' + algorithm + '.pdf')