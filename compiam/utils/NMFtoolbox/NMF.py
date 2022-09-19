"""
    Name: NMF
    Date: Jun 2019
    Programmer: Christian Dittmar, Yiğitcan Özer

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    If you use the 'NMF toolbox' please refer to:
    [1] Patricio López-Serrano, Christian Dittmar, Yiğitcan Özer, and Meinard
        Müller
        NMF Toolbox: Music Processing Applications of Nonnegative Matrix
        Factorization
        In Proceedings of the International Conference on Digital Audio Effects
        (DAFx), 2019.

    License:
    This file is part of 'NMF toolbox'.
    https://www.audiolabs-erlangen.de/resources/MIR/NMFtoolbox/
    'NMF toolbox' is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by the
    the Free Software Foundation, either version 3 of the License, or (at
    your option) any later version.

    'NMF toolbox' is distributed in the hope that it will be useful, but
    WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
    Public License for more details.

    You should have received a copy of the GNU General Public License along
    with 'NMF toolbox'. If not, see http://www.gnu.org/licenses/.
"""

import numpy as np
from copy import deepcopy
from tqdm import tnrange

from NMFtoolbox.utils import EPS


def NMF(V, parameter):
    """Given a non-negative matrix V, find non-negative templates W and activations
    H that approximate V.

    References
    ----------
    [2] Lee, DD & Seung, HS. "Algorithms for Non-negative Matrix Factorization"

    [3] Andrzej Cichocki, Rafal Zdunek, Anh Huy Phan, and Shun-ichi Amari
    "Nonnegative Matrix and Tensor Factorizations: Applications to
    Exploratory Multi-Way Data Analysis and Blind Source Separation"
    John Wiley and Sons, 2009.

    Parameters
    ----------
    V: array-like
        K x M non-negative matrix to be factorized

    parameter: dict
        costFunc      Cost function used for the optimization, currently
                      supported are:
                      'EucDdist' for Euclidean Distance
                      'KLDiv' for Kullback Leibler Divergence
                      'ISDiv' for Itakura Saito Divergence
        numIter       Number of iterations the algorithm will run.
        numComp       The rank of the approximation

    Returns
    -------
    W: array-like
        K x R non-negative templates
    H: array-like
        R x M non-negative activations
    nmfV: array-like
        List with approximated component matrices
    """
    parameter = init_parameters(parameter)

    # get important params
    K, M = V.shape
    R = parameter['numComp']
    L = parameter['numIter']

    # initialization of W and H
    if isinstance(parameter['initW'], list):
        W = np.array(parameter['initW'])
    else:
        W = deepcopy(parameter['initW'])

    H = deepcopy(parameter['initH'])

    # create helper matrix of all ones
    onesMatrix = np.ones((K, M))

    # normalize to unit sum
    V /= (EPS + V.sum())

    # main iterations
    for iter in tnrange(L, desc='Processing'):

        # compute approximation
        Lambda = EPS + W @ H

        # switch between pre-defined update rules
        if parameter['costFunc'] == 'EucDist':  # euclidean update rules
            if not parameter['fixW']:
                W *= (V @ H.T / (Lambda @ H.T + EPS))

            H *= (W.T @ V / (W.T @ Lambda + EPS))

        elif parameter['costFunc'] == 'KLDiv':  # Kullback Leibler divergence update rules
            if not parameter['fixW']:
                W *= ((V / Lambda) @ H.T) / (onesMatrix @ H.T + EPS)

            H *= (W.T @ (V / Lambda)) / (W.T @ onesMatrix + EPS)

        elif parameter['costFunc'] == 'ISDiv':  # Itakura Saito divergence update rules
            if not parameter['fixW']:
                W *= ((Lambda ** -2 * V) @ H.T) / ((Lambda ** -1) @ H.T + EPS)

            H *= (W.T @(Lambda ** -2 * V)) / (W.T @ (Lambda ** -1) + EPS)

        else:
            raise ValueError('Unknown cost function')

        # normalize templates to unit sum
        if not parameter['fixW']:
            normVec = W.sum(axis=0)
            W *= 1.0 / (EPS + normVec)

    nmfV = list()

    # compute final output approximation
    for r in range(R):
        nmfV.append(W[:, r].reshape(-1, 1) @ H[r, :].reshape(1, -1))

    return W, H, nmfV


def init_parameters(parameter):
    """Auxiliary function to set the parameter dictionary

    Parameters
    ----------
    parameter: dict
        See the above function inverseSTFT for further information

    Returns
    -------
    parameter: dict
    """
    parameter['costFunc'] = 'KLDiv' if 'costFunc' not in parameter else parameter['costFunc']
    parameter['numIter'] = 30 if 'numIter' not in parameter else parameter['numIter']
    parameter['fixW'] = False if 'fixW' not in parameter else parameter['fixW']

    return parameter
