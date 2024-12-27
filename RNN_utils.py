import numpy as np

def softmax(x):
    e_x=np.exp(x-np.max(x))
    return e_x/e_x.sum(axis=0)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def initialize_adam(parameters):
    """Initializing momentum and RMS_prop into two dictionaries
    -keys=dW1,db1....dWL,dbL where W are the weights and the b are bias

    parameters--- the dictionary containing the values
    parameters["dW"+str(l)]=dWl """
    L=len(parameters)//2
    """For the number of the layers in neural network"""
    v={}
    s={}

    for l in range(L):
        v["dW"+str(l+1)]=np.zeroes(parameters["W"+str(l+1)].shape())
        v["db"+str(l+1)]=np.zeroes(parameters["W"+str(l+1)].shape())
        s["dW"+str(l+1)]=np.zeroes(parameters["W"+str(l+1)].shape())
        s["dW"+str(l+1)]=np.zeroes(parameters["W"+str(l+1)].shape())
    return v,s

def update_parameters_with_adam(parameters,grad,v,s,t,alpha=0.01,beta1=0.9,beta2=0.999,epsilon=1e-8):
    L=len(parameters)//2
    v_corrected={}
    s_corrected={}

    for l in range(L):
        v['dW'+str(l+1)]=beta1*v['dW'+str(l+1)]+(1-beta1)*grad['dW'+str(l+1)]
        v['db'+str(l+1)]=beta1*v['db'+str(l+1)]+(1-beta1)*grad['db'+str(l+1)]

        v_corrected['dW'+str(l+1)]=v['dW'+str(l+1)]/(1-beta1**t)
        v_corrected['db'+str(l+1)]=v['db'+str(l+1)]/(1-beta1**t)

        s['dW' + str(l + 1)] = beta1 * s['dW' + str(l + 1)] + (1 - beta1) * grad['dW' + str(l + 1)]
        s['db' + str(l + 1)] = beta1 * s['db' + str(l + 1)] + (1 - beta1) * grad['db' + str(l + 1)]

        s_corrected['dW' + str(l + 1)] = s['dW' + str(l + 1)] / (1 - beta1 ** t)
        s_corrected['db' + str(l + 1)] = s['db' + str(l + 1)] / (1 - beta1 ** t)

        parameters['W'+str(l+1)]=parameters['W'+str(l+1)]-alpha*v_corrected['dW'+str(l+1)]/np.sqrt(s_corrected['dW'+str(l+1)]+epsilon)
        parameters['b'+str(l+1)]=parameters['b'+str(l+1)]-alpha*v_corrected['db'+str(l+1)]/np.sqrt(s_corrected['db'+str(l+1)]+epsilon)
    return parameters,v,s