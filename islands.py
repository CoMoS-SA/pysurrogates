import pandas as pd
import numpy as np


def problem():
    problem_islands = {
    'abm_name': 'Islands',
    'num_vars': 7,
    'names': ['rho', 'alpha', 'phi','pi', 'eps', 'N','lambda'],
    'bounds': [[0,1], [0.8,2], [0.0,1.0], [0.0,1.0], [0.0,1.0], [3,15], [0.0,1.0]]
    }

    return problem_islands


""" Islands ABM """
def model(params,
               T=200):
    """ Islands growth model

    Parameters
    ----------

    rho : 
    
    alpha : 
    
    phi : float, required
    
    eps : 
    
    lambda_param: (Default = 1)
        
    T : int, required
    The number of periods for the simulation
    
    N : int, optional (Default = 50)
    Number of firms
    
    _RNG_SEED : int, optional (Default = 0)
    Random number seen


    Output
    ------

    GDP : array, length = [,T]
    Simulated GPD

    """
    # Set random number seed
    #np.random.seed(_RNG_SEED)

    rho, alpha, phi, pi, eps, N, lambda_param = params

    N = int(N)
    T = int(T)
    T_2 = int(T/2) #Half the duration of the simulation (what for??)
    #Maybe this is the size of the lattice because it is the farthest that an explorer could get. 
    #Anyway, the center is in T_2...

    GDP = np.zeros((T, 1))

    # Distributions
    # Precompute random binomial draws
    xy = np.random.binomial(1, np.clip(pi,0,1), (T, T))   #Fixed if _RNG_SEED is defined
    xy[T_2, T_2] = 1

    # Containers
    s = np.zeros((T, T)) #productivities of islands
    A = np.ones((N, 6)) #For each firm, a vector of 6 ones: ['state (1: miner)','pos x','pos y',
    # 'prod. of its island (s)','output of its island (Q = s m^alpha)','memory of last quantity produced']

    # # Initializations
    A[:, 1] = T_2
    A[:, 2] = T_2
    m = np.zeros((T, T)) #Population lattice, how many firms in each place
    m[T_2, T_2] = N
    dest = np.zeros((N, 2)) # Eventually used for storing destinations when forms imitate

    """ Begin ABM Code """
    for t in range(T): #for every timestep..
        w = np.zeros((N, N))
        signal = np.zeros((N, N)) #interactions between firms?

        for i in range(N): #for each firm
            for j in range(N): 
                if i != j: #for all its 'other firms'
                    if A[j, 0] == 1: # i.e. it's a 'miner'
                        #Probability for signal of island i to j
                        w[i, j] = m[int(A[i,1]), int(A[i,2])]/N * np.exp(-rho*
                                         (np.abs(A[j,1]-A[i,1])+ np.abs(A[j,2]-A[i,2]))) 
                        #The m_j/m factor was missing: m[A[i,1], A[i,2]]/N * np. exp(...)
                        if np.random.rand() < w[i,j]:
                            signal[i,j]=s[int(A[j, 1]), int(A[j, 2])] #signal tells productivity s of current island

            if A[i,0] == 1:
    #             A[i,4] = s[int(A[i,1]),int(A[i,2])] * m[int(A[i,1]),int(A[i,2])]**alpha #output of island j 
    #            # (because m tells how many firms there are there)
                A[i,4] = s[int(A[i,1]),int(A[i,2])] * m[int(A[i,1]),int(A[i,2])]**(alpha - 1) 
                    # the '-1' means divided by m, i.e. output per firm!
                A[i,3] = s[int(A[i,1]),int(A[i,2])] #productivity of island

            if A[i, 0] == 3: #explorer. 
                A[i, 4] = 0 #you don't produce
                #Move in some of the four directions
                rnd = np.random.rand() 
                if rnd <= 0.25:
                    A[i,1] += 1
                else:
                    if rnd <= 0.5:
                        A[i,1] -= 1
                    else:
                        if rnd <= 0.75:
                            A[i,2] += 1
                        else:
                            A[i,2] -= 1
                #you moved.

                if xy[int(A[i,1]),int(A[i,2])] == 1: #if it's an island...
                    A[i,0] = 1 #Change status to miner
                    m[int(A[i,1]),int(A[i,2])] += 1 #New island has plus one firm
                    if m[int(A[i,1]),int(A[i,2])] == 1: #If you're the only one, set up island prod. level s
                        s[int(A[i,1]),int(A[i,2])] = \
                            (1+int(np.random.poisson(lambda_param)))* \
                            (np.abs(A[i,1] - T_2) +np.abs(A[i,2] - T_2)+phi*A[i,5]+np.random.randn()) #There was a parenthesis missing here.
                            #Also we should do the distance to the starting position T_2 and not the absolute position. 
    #                         Why cast the poisson to int in this case? 
    

            if (A[i,0] == 1) and (np.random.rand() <= eps): #If you happen to go exploring...
                A[i,0] = 3 #Change status to explorer
                A[i,5] = A[i,4] #Remember how much your island produced (note this is output of island with m firms: s m^a, 
    #             and not s m^(a-1) (i.e. output per firm) as in the paper)
                m[int(A[i,1]),int(A[i,2])] -= 1 #Islands has lost you as firm

            if t > T/100: #For the ~99% last part of the simulation (why?). Also note eg 199/100 = 1.
                if A[i,0] == 2: #If your imitating (i.e. moving to the better island)
                    A[i,4] = 0
                    #Go towards your destination.
                    if dest[i,0] != A[i,1]:
                        if dest[i,0] > A[i,1]:
                            A[i,1] += 1
                        else:
                            A[i,1] -= 1
                    else:
                        if dest[i,1] != A[i,2]:
                            if dest[i,1] > A[i,2]:
                                A[i,2] += 1
                            else:
                                A[i,2] -= 1
                    #if you reached the destination...
                    if (dest[i,0] == A[i,1]) and (dest[i,1] == A[i,2]):
                        A[i,0] = 1
                        m[int(dest[i,0]),int(dest[i,1])] += 1
                        #new firm in destination island

                if A[i,0] == 1: #If you're mining
                    best_sig = np.max(signal[i,:])
                    #if you see a better island
                    if best_sig > s[int(A[i,1]),int(A[i,2])]:
                        A[i,0] = 2 #change to imitator
                        A[i,5] = A[i,4] #remember your last prod
                        m[int(A[i,1]),int(A[i,2])] -= 1 #island lost a firm
                        index = np.where(signal[i,:] == best_sig)[0]#where is the best signal
                        if index.shape[0] > 1:
                            ind = int(index[int(np.random.uniform(0, len(index)))]) # if more than one match, random choice
                        else:
                            ind = int(index)
                        dest[i,0] = A[ind,1]
                        dest[i,1] = A[ind,2] #set destination

        GDP[t, 0] = np.sum(A[:, 4]) #Again this is not production of firms, but that of islands!!

    log_GDP = np.log10(GDP)

    T = log_GDP.shape[0]
    log_GDP = log_GDP[~np.isinf(log_GDP)]
    log_GDP = log_GDP[~np.isnan(log_GDP)]
    if log_GDP.shape[0] > 0:
        GDP_growth_rate = (log_GDP[-1] - log_GDP[0]) / T
    else:
        GDP_growth_rate = 0

    return GDP_growth_rate





