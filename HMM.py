########################################
# CS/CNS/EE 155 2018
# Problem Set 6
#
# Author:       Andrew Kang
# Description:  Set 6 skeleton code
########################################

# You can use this (optional) skeleton code to complete the HMM
# implementation of set 5. Once each part is implemented, you can simply
# execute the related problem scripts (e.g. run 'python 2G.py') to quickly
# see the results from your code.
#
# Some pointers to get you started:
#
#     - Choose your notation carefully and consistently! Readable
#       notation will make all the difference in the time it takes you
#       to implement this class, as well as how difficult it is to debug.
#
#     - Read the documentation in this file! Make sure you know what
#       is expected from each function and what each variable is.
#
#     - Any reference to "the (i, j)^th" element of a matrix T means that
#       you should use T[i][j].
#
#     - Note that in our solution code, no NumPy was used. That is, there
#       are no fancy tricks here, just basic coding. If you understand HMMs
#       to a thorough extent, the rest of this implementation should come
#       naturally. However, if you'd like to use NumPy, feel free to.
#
#     - Take one step at a time! Move onto the next algorithm to implement
#       only if you're absolutely sure that all previous algorithms are
#       correct. We are providing you waypoints for this reason.
#
# To get started, just fill in code where indicated. Best of luck!

import random
import numpy as np


class HiddenMarkovModel:
    '''
    Class implementation of Hidden Markov Models.
    '''

    def __init__(self, A, O):
        '''
        Initializes an HMM. Assumes the following:
            - States and observations are integers starting from 0. 
            - There is a start state (see notes on A_start below). There
              is no integer associated with the start state, only
              probabilities in the vector A_start.
            - There is no end state.

        Arguments:
            A:          Transition matrix with dimensions L x L.
                        The (i, j)^th element is the probability of
                        transitioning from state i to state j. Note that
                        this does not include the starting probabilities.

            O:          Observation matrix with dimensions L x D.
                        The (i, j)^th element is the probability of
                        emitting observation j given state i.

        Parameters:
            L:          Number of states.
            
            D:          Number of observations.
            
            A:          The transition matrix.
            
            O:          The observation matrix.
            
            A_start:    Starting transition probabilities. The i^th element
                        is the probability of transitioning from the start
                        state to state i. For simplicity, we assume that
                        this distribution is uniform.
        '''

        self.L = len(A)
        self.D = len(O[0])
        self.A = A
        self.O = O
        self.A_start = [1. / self.L for _ in range(self.L)]

    def viterbi(self, x):
        '''
        Uses the Viterbi algorithm to find the max probability state 
        sequence corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

        Returns:
            max_seq:    State sequence corresponding to x with the highest
                        probability.
        '''

        M = len(x)  # Length of sequence.

        # The (i, j)^th elements of probs and seqs are the max probability
        # of the prefix of length i ending in state j and the prefix
        # that gives this probability, respectively.
        #
        # For instance, probs[1][0] is the probability of the prefix of
        # length 1 ending in state 0.
        probs = [[0. for _ in range(self.L)] for _ in range(M + 1)]
        seqs = [['' for _ in range(self.L)] for _ in range(M + 1)]

        ###
        ###
        ### 
        ### TODO: Insert Your Code Here (2A)
        ###
        ###
        ###

        # first row in probs, Y1
        for j in range(self.L):
            probs[1][j] = self.O[j][x[0]] * self.A_start[j]
            seqs[1][j] = str(j)

        for i in range(2, M + 1):

            for j in range(self.L):

                prefix = ''
                max_prob = 0

                # max
                for t in range(self.L):
                    prob = probs[i - 1][t] * self.O[j][x[i - 1]] * self.A[t][j]
                    if prob > max_prob:
                        max_prob = prob
                        prefix = seqs[i - 1][t]

                probs[i][j] = max_prob
                seqs[i][j] = prefix + str(j)

        max_seq = seqs[M][probs[M].index(max(probs[M]))]
        return max_seq

    def forward(self, x, normalize=False):
        '''
        Uses the forward algorithm to calculate the alpha probability
        vectors corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

            normalize:  Whether to normalize each set of alpha_j(i) vectors
                        at each i. This is useful to avoid underflow in
                        unsupervised learning.

        Returns:
            alphas:     Vector of alphas.

                        The (i, j)^th element of alphas is alpha_j(i),
                        i.e. the probability of observing prefix x^1:i
                        and state y^i = j.

                        e.g. alphas[1][0] corresponds to the probability
                        of observing x^1:1, i.e. the first observation,
                        given that y^1 = 0, i.e. the first state is 0.
        '''

        M = len(x)  # Length of sequence.
        alphas = [[0. for _ in range(self.L)] for _ in range(M + 1)]

        ###
        ###
        ### 
        ### TODO: Insert Your Code Here (2Bi)
        ###
        ###
        ###
        # Compute alphas for i = 1
        for j in range(self.L):
            alphas[1][j] = self.O[j][x[0]] * self.A_start[j]

        # Recursively solve for each i > 1
        for i in range(2, M + 1):
            for j in range(self.L):
                sum_probs = 0
                for k in range(self.L):
                    sum_probs += alphas[i - 1][k] * self.A[k][j]
                alphas[i][j] = self.O[j][x[i - 1]] * sum_probs
            if normalize:
                alphas[i] = [a / sum(alphas[i]) for a in alphas[i]]

        return alphas

    def backward(self, x, normalize=False):
        '''
        Uses the backward algorithm to calculate the beta probability
        vectors corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

            normalize:  Whether to normalize each set of alpha_j(i) vectors
                        at each i. This is useful to avoid underflow in
                        unsupervised learning.

        Returns:
            betas:      Vector of betas.

                        The (i, j)^th element of betas is beta_j(i), i.e.
                        the probability of observing prefix x^(i+1):M and
                        state y^i = j.

                        e.g. betas[M][0] corresponds to the probability
                        of observing x^M+1:M, i.e. no observations,
                        given that y^M = 0, i.e. the last state is 0.
        '''

        M = len(x)  # Length of sequence.
        betas = [[0. for _ in range(self.L)] for _ in range(M + 1)]

        ###
        ###
        ### 
        ### TODO: Insert Your Code Here (2Bii)
        ###
        ###
        ###

        # Compute betas for i = M
        betas[M] = [1. for _ in range(self.L)]

        for i in range(M - 1, -1, -1):
            for j in range(self.L):
                for k in range(self.L):
                    betas[i][j] += betas[i + 1][k] * self.O[k][x[i]] * self.A[j][k]
            if normalize:
                betas[i] = [b / sum(betas[i]) for b in betas[i]]

        return betas

    def supervised_learning(self, X, Y):
        '''
        Trains the HMM using the Maximum Likelihood closed form solutions
        for the transition and observation matrices on a labeled
        datset (X, Y). Note that this method does not return anything, but
        instead updates the attributes of the HMM object.

        Arguments:
            X:          A dataset consisting of input sequences in the form
                        of lists of variable length, consisting of integers 
                        ranging from 0 to D - 1. In other words, a list of
                        lists.

            Y:          A dataset consisting of state sequences in the form
                        of lists of variable length, consisting of integers 
                        ranging from 0 to L - 1. In other words, a list of
                        lists.

                        Note that the elements in X line up with those in Y.
        '''

        # # Calculate each element of A using the M-step formulas.

        N = len(X)
        # ###
        # ###
        # ### 
        # ### TODO: Insert Your Code Here (2C)
        # ###
        # ###
        # ###
        for b in range(self.L):
            for a in range(self.L):

                numer = 0.0
                denom = 0.0

                for j in range(N):
                    for i in range(len(Y[j]) - 1):
                        if Y[j][i] == b:
                            denom += 1
                            if Y[j][i + 1] == a:
                                numer += 1
                self.A[b][a] = numer / denom

        # Calculate each element of O using the M-step formulas.

        ###
        ###
        ### 
        ### TODO: Insert Your Code Here (2C)
        ###
        ###
        ###
        for z in range(self.L):
            for w in range(self.D):

                numer = 0.0
                denom = 0.0

                for j in range(N):
                    for i in range(len(Y[j])):
                        if Y[j][i] == z:
                            denom += 1
                            if X[j][i] == w:
                                numer += 1
                self.O[z][w] = numer / denom

    def unsupervised_learning(self, X, N_iters):
        '''
        Trains the HMM using the Baum-Welch algorithm on an unlabeled
        datset X. Note that this method does not return anything, but
        instead updates the attributes of the HMM object.

        Arguments:
            X:          A dataset consisting of input sequences in the form
                        of lists of length M, consisting of integers ranging
                        from 0 to D - 1. In other words, a list of lists.

            N_iters:    The number of iterations to train on.
        '''

        ###
        ###
        ### 
        ### TODO: Insert Your Code Here (2D)
        ###
        ###
        ###


        for iteration in range(N_iters):
            # print('iteration %d'%iteration)
            # print()

            A_numer = [[0. for i in range(self.L)] for j in range(self.L)]
            A_denom = [0. for i in range(self.L)]
            O_numer = [[0. for i in range(self.D)] for j in range(self.L)]
            O_denom = [0. for i in range(self.L)]

            # E-step:  use alphas, betas, and the current O and A matrices to calculate the marginals
            for x in X:
                M = len(x)

                alphas = self.forward(x, True)
                betas = self.backward(x, True)

                # compute P(y^i|x) and update A_denom, O_numer and O_denom
                for i in range(1, M + 1):

                    # compute P(y^i|x)
                    ps_joint_yix = [alphas[i][k] * betas[i][k] for k in range(self.L)]
                    p_x = sum(ps_joint_yix)

                    ps_yi_given_x = [ p_joint_yix/p_x for p_joint_yix in ps_joint_yix]

                    # update A_denom, O_numer and O_denom
                    for s in range(self.L):
                        # A_denom
                        if i < M:
                            A_denom[s] += ps_yi_given_x[s]
                        # O_numer
                        O_numer[s][x[i - 1]] += ps_yi_given_x[s]
                        # O_denom
                        O_denom[s] += ps_yi_given_x[s]

                # compute P(y^(i+1),y^i|x) and update A_numer
                for i in range(1, M):
                    ps_yi_ynext_x = [[alphas[i][curr] * self.A[curr][nxt] * self.O[nxt][x[i]] * betas[i + 1][nxt]
                                    for nxt in range(self.L)] for curr in range(self.L)]

                    p_x = 0
                    for row in ps_yi_ynext_x:
                        p_x += sum(row)

                    ps_yi_ynext_x = [[ e/p_x for e in row] for row in ps_yi_ynext_x]

                    # update A_numer
                    for curr in range(self.L):
                        for nxt in range(self.L):
                            A_numer[curr][nxt] += ps_yi_ynext_x[curr][nxt]
            # M-step:
            # update A
            for curr in range(self.L):
                for nxt in range(self.L):
                    self.A[curr][nxt] = A_numer[curr][nxt] / A_denom[curr]
            # update O
            for curr in range(self.L):
                for xi in range(self.D):
                    self.O[curr][xi] = O_numer[curr][xi] / O_denom[curr]

        #
        # marginal2 = [[[[0.0 for i in range(L)] for j in range(L)] for _ in range(len(X[i]))] for i in range(len(X))]
        #
        # marginal1 = [[[0.0 for j in range(L)] for _ in range(len(X[i]))] for i in range(len(X))]
        #
        # for iteration in range(N_iters):
        #     print('iteration %d' % iteration)
        #     print()
        #     # E-step:  use alphas, betas, and the current O and A matrices to calculate the marginals
        #     alphas = [self.forward(X[j], True) for j in range(len(X))]
        #     betas = [self.backward(X[j], True) for j in range(len(X))]
        #
        #     for j in range(len(X)):
        #
        #         # for each data X[j]
        #         for i in range(len(X[j])):
        #
        #             for a in range(L):
        #                 # compute marginal2
        #                 for b in range(L):
        #
        #                     numer = alphas[j][i][a] * A[a][b] * O[b][X[j][i]] * betas[j][i+1][b]
        #                     denon = 0.0
        #
        #                     for m in range(L):
        #                         for n in range(L):
        #                             denon += alphas[j][i][m] * A[m][n] * O[n][X[j][i]] * betas[j][i+1][n]
        #
        #                     if denon == 0:
        #                         marginal2[j][i-1][a][b] = float(numer)
        #                     else:
        #                         marginal2[j][i-1][a][b] = float(numer) / float(denon)
        #
        #                 # compute marginal1
        #                 numer = alphas[j][i][a] * betas[j][i][a]
        #                 denon = 0
        #                 for k in range(L):
        #                     denon += alphas[j][i][k] * betas[j][i][k]
        #
        #                 if denon == 0:
        #                     marginal1[j][i][a] = float(numer)
        #                 else:
        #                     marginal1[j][i][a] = float(numer) / float(denon)
        #
        #     # M-step:
        #     # update A
        #     for a in range(L):
        #         for b in range(L):
        #             numer = 0.0
        #             denon = 0.0
        #             for j in range(len(X)):
        #                 for i in range(len(X[j])):
        #                     numer += marginal2[j][i][a][b]
        #                     denon += marginal1[j][i][a]
        #             A[a][b] = float(numer) / float(denon)
        #     # update O
        #     for z in range(L):
        #         for w in range(D):
        #             numer = 0.0
        #             denon = 0.0
        #             for j in range(len(X)):
        #                 for i in range(len(X[j])):
        #                     if X[j][i - 1] == w:
        #                         numer += marginal1[j][i][z]
        #
        #                     denon += marginal1[j][i][z]
        #             O[z][w] = float(numer) / float(denon)
        #
        # self.O = O
        # self.A = A

    def generate_emission(self, M):
        '''
        Generates an emission of length M, assuming that the starting state
        is chosen uniformly at random. 

        Arguments:
            M:          Length of the emission to generate.

        Returns:
            emission:   The randomly generated emission as a list.

            states:     The randomly generated states as a list.
        '''

        emission = []
        states = []

        ###
        ###
        ### 
        ### TODO: Insert Your Code Here (2F)
        ###
        ###
        ###

        # first state and observation
        state = np.random.choice(range(self.L))
        states.append(state)
        x0 = np.random.choice(range(self.D), p=self.O[state])
        emission.append(x0)

        for i in range(M - 1):
            state = np.random.choice(range(self.L), p=self.A[state])
            states.append(state)

            xi = np.random.choice(range(self.D), p=self.O[state])
            emission.append(xi)

        return emission, states

    def probability_alphas(self, x):
        '''
        Finds the maximum probability of a given input sequence using
        the forward algorithm.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

        Returns:
            prob:       Total probability that x can occur.
        '''

        # Calculate alpha vectors.
        alphas = self.forward(x)

        # alpha_j(M) gives the probability that the state sequence ends
        # in j. Summing this value over all possible states j gives the
        # total probability of x paired with any state sequence, i.e.
        # the probability of x.
        prob = sum(alphas[-1])
        return prob

    def probability_betas(self, x):
        '''
        Finds the maximum probability of a given input sequence using
        the backward algorithm.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

        Returns:
            prob:       Total probability that x can occur.
        '''

        betas = self.backward(x)

        # beta_j(1) gives the probability that the state sequence starts
        # with j. Summing this, multiplied by the starting transition
        # probability and the observation probability, over all states
        # gives the total probability of x paired with any state
        # sequence, i.e. the probability of x.
        prob = sum([betas[1][j] * self.A_start[j] * self.O[j][x[0]] \
                    for j in range(self.L)])

        return prob


def supervised_HMM(X, Y):
    '''
    Helper function to train a supervised HMM. The function determines the
    number of unique states and observations in the given data, initializes
    the transition and observation matrices, creates the HMM, and then runs
    the training function for supervised learning.

    Arguments:
        X:          A dataset consisting of input sequences in the form
                    of lists of variable length, consisting of integers 
                    ranging from 0 to D - 1. In other words, a list of lists.

        Y:          A dataset consisting of state sequences in the form
                    of lists of variable length, consisting of integers 
                    ranging from 0 to L - 1. In other words, a list of lists.
                    Note that the elements in X line up with those in Y.
    '''
    # Make a set of observations.
    observations = set()
    for x in X:
        observations |= set(x)

    # Make a set of states.
    states = set()
    for y in Y:
        states |= set(y)

    # Compute L and D.
    L = len(states)
    D = len(observations)

    # Randomly initialize and normalize matrix A.
    A = [[random.random() for i in range(L)] for j in range(L)]

    for i in range(len(A)):
        norm = sum(A[i])
        for j in range(len(A[i])):
            A[i][j] /= norm

    # Randomly initialize and normalize matrix O.
    O = [[random.random() for i in range(D)] for j in range(L)]

    for i in range(len(O)):
        norm = sum(O[i])
        for j in range(len(O[i])):
            O[i][j] /= norm

    # Train an HMM with labeled data.
    HMM = HiddenMarkovModel(A, O)
    HMM.supervised_learning(X, Y)

    return HMM


def unsupervised_HMM(X, n_states, N_iters):
    '''
    Helper function to train an unsupervised HMM. The function determines the
    number of unique observations in the given data, initializes
    the transition and observation matrices, creates the HMM, and then runs
    the training function for unsupervised learing.

    Arguments:
        X:          A dataset consisting of input sequences in the form
                    of lists of variable length, consisting of integers 
                    ranging from 0 to D - 1. In other words, a list of lists.

        n_states:   Number of hidden states to use in training.
        
        N_iters:    The number of iterations to train on.
    '''

    # Make a set of observations.
    observations = set()
    for x in X:
        observations |= set(x)

    # Compute L and D.
    L = n_states
    D = len(observations)

    random.seed(2020)
    # Randomly initialize and normalize matrix A.
    A = [[random.random() for i in range(L)] for j in range(L)]

    for i in range(len(A)):
        norm = sum(A[i])
        for j in range(len(A[i])):
            A[i][j] /= norm
    random.seed(155)
    # Randomly initialize and normalize matrix O.
    O = [[random.random() for i in range(D)] for j in range(L)]

    for i in range(len(O)):
        norm = sum(O[i])
        for j in range(len(O[i])):
            O[i][j] /= norm

    # Train an HMM with unlabeled data.
    HMM = HiddenMarkovModel(A, O)
    HMM.unsupervised_learning(X, N_iters)

    return HMM
