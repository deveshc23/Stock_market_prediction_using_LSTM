from colorsys import yiq_to_rgb

import numpy as np
from RNN_utils import *

def rnn_cell_forward(xt,a_prev,parameters):
    Wax=parameters['Wax']
    Waa=parameters['Waa']
    Wya=parameters['Wya']
    ba=parameters['ba']
    by=parameters['by']

    a_next=np.tanh(np.matmul(Waa,a_prev)+np.matmul(Wax,xt)+ba)
    yt_pred=softmax(np.matmul(Wya,a_next)+by)

    cache=(a_next,a_prev,xt,parameters)
    return a_next,yt_pred, cache

def rnn_cell_backward(da_next,cache):
    (a_next,a_prev,xt,parameters)=cache
    Wax = parameters['Wax']
    Waa = parameters['Waa']
    Wya = parameters['Wya']
    ba = parameters['ba']
    by = parameters['by']

    dtanh=(1-a_next**2)*da_next

    dxt=np.dot(Wax.T,dtanh)
    dWax=np.dot(dtanh,xt.T)
    da_prev = np.dot(Waa.T, dtanh)
    dWaa = np.dot(dtanh, a_prev.T)

    dba=np.sum(dtanh,1,keepdims=True)
    gradients = {"dxt": dxt, "da_prev": da_prev, "dWax": dWax, "dWaa": dWaa, "dba": dba}
    return gradients
