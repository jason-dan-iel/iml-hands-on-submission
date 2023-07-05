#importing required libraries
import numpy as np
import random

# following the website link and sir's for coding this part
def coin_em(rolls, theta_A=None, theta_B=None, maxiter=10):
    theta_A = theta_A or random.random()
    theta_B = theta_B or random.random()
    thetas = [(theta_A, theta_B)]

    for c in range(maxiter): 
        print("#%d:\t%0.2f %0.2f" % (c, theta_A, theta_B))
        heads_A, tails_A, heads_B, tails_B = e_step(rolls, theta_A, theta_B)
        theta_A, theta_B = m_step(heads_A, tails_A, heads_B, tails_B)

    thetas.append((theta_A,theta_B))
    return thetas, (theta_A,theta_B) 

def e_step(rools, theta_A, theta_B):
    heads_A, tails_A = 0,0
    heads_B, tails_B = 0,0
    for trial in rools:
        likelihood_A = coin_likelihood(trial, theta_A)
        likelihood_B = coin_likelihood(trial, theta_B)
        p_A = likelihood_A / (likelihood_A + likelihood_B) 
        p_B = likelihood_B / (likelihood_A + likelihood_B)
        heads_A += p_A * trial.count("H")
        tails_A += p_A * trial.count("T")
        heads_B += p_B * trial.count("H")
        tails_B += p_B * trial.count("T")
    return heads_A, tails_A, heads_B, tails_B

def m_step(heads_A, tails_A, heads_B, tails_B):
    theta_A = heads_A / (heads_A + tails_A)
    theta_B = heads_B / (heads_B + tails_B)
    return theta_A, theta_B

def coin_likelihood(roll, bias): 
    numHeads = roll.count("H")
    flips = len(roll)
    return pow(bias, numHeads) * pow(1-bias, flips-numHeads)

from matplotlib import pyplot as plt 
import matplotlib as mpl

def plot_coin_likelihood(rolls, thetas=None):
    xvals = np.linspace(0.01,0.99,100)
    yvals = np.linspace(0.01,0.99,100)
    X,Y = np.meshgrid(xvals, yvals)

    Z = []
    for i,r in enumerate(X):
        z = []
        for j,c in enumerate(r):
            z.append(coin_marginal_likelihood(rolls,c,Y[i][j]))
        Z.append(z)

    plt.figure(figsize=(10,8))
    C = plt.contour(X,Y,Z,150)
    cbar = plt.colorbar(C)
    plt.title(r"Likelihood $\log p(\mathcal{X}|\theta_A,\theta_B)$", fontsize=20) 
    plt.xlabel(r"$\theta_A$", fontsize=20)
    plt.ylabel(r"$\theta_B$", fontsize=20)

    if thetas is not None:
        thetas = np.array(thetas)
        plt.plot(thetas[:,0], thetas[:,1], '-k', lw=2.0)
        plt.plot(thetas[:,0], thetas[:,1], 'ok', ms=5.0)


def coin_marginal_likelihood(rolls, biasA, biasB):
    trials = []
    for roll in rolls:
        h = roll.count("H")
        t = roll.count("T")
        llhA = coin_likelihood(roll, biasA)
        llhB = coin_likelihood(roll, biasB)
        trials.append(np.log(0.5 * (llhA + llhB)))
    return sum(trials)

XX = ['HHHHHTTTTTHHH', 'HHTHHTHTHTHHT', 'HHTHTTHTHTHTT', 'TTTHTTHTHTHTT', 'THTHTTHTHTHHH'] 
thetas1, _ = coin_em(XX, 0.6, 0.5, maxiter=6)
plot_coin_likelihood(XX, thetas1)
plt.savefig("pltCoinLikelihood.png")