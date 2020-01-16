import numpy as np
import random

from funcs4nn import activation, derivative
def one_hot(str):

    vocabulary=["a", "b"]
    index=vocabulary.index(str)
    zeros=np.zeros(len(vocabulary))
    zeros[index]=1
    return zeros

def sequence(M):
    random.seed(2)
    T=8 # str length (time dimension)
    n=2 # length of one hot vector
    data_set=[]
    for i in range(M):
        num1=random.randint(1,4)
        num2=random.randint(1,4)
        str="a"*num1+"b"*num2
        data_set.append(str)
    print data_set
    matrix=np.zeros((n, M, T))
    for m in range(len(data_set)):
        word=np.zeros((n, T))
        str=data_set[m]
        for t in range(len(str)):
            letter=str[t]
            word[:,t]=one_hot(letter)
        matrix[:,m,:]=word
    return matrix

def gates_func(xt, prev_a, w_gate, b_gate, func):
   """calculates the gate value with respect to the given parameters
   :parameter
   xt- the current x value
   prev_a- the previous prediction
   w_gate- the gate's weight
   b_gate- the gate's bias
   func- activation function
   :returns
   gate_value- the value of the gate
   """
   print prev_a.shape
   print xt.shape
   concat = np.concatenate((prev_a, xt))  # concatenated array of a_prev and xt
   print (concat.shape)
   gate_value = activation(func, np.dot(w_gate, concat) + b_gate)
   return gate_value


def forward_propagation( x, params):
   """
   :param x: input values
   :param first_c: the first cell value
   :param rnn_params:
   :param lstm_params:
   :return:
   """

   n_y, n_a = params["wy"].shape
   n_x, m, T_x = x.shape

   values = {
               "forget": np.zeros((n_a, m, T_x)),
               "input":  np.zeros((n_a, m, T_x)),
               "output": np.zeros((n_a, m, T_x)),
               "candidate_cell": np.zeros((n_a, m, T_x))
           }

   c = np.zeros((n_a, m, T_x + 1))  # lstm cell***
   a = np.zeros((n_a, m, T_x + 1))  # hidden states***
   y = np.zeros((n_y, m, T_x))  # output array
   a[:, :, 0] = np.random.randn(n_a, m)  # setting the first cell value
   for t in range(T_x):
       # calculate lstm gates values

       current_x = x[:, :, t]  # the current x value
       prev_a = a[:, :, t] # the previous value of the hidden state*
       prev_c = c[:, :, t]
       #1
       values["forget"][:, :, t] = gates_func(current_x, prev_a, params["wf"], params["bf"], "sigmoid")  # forget gate value
       values["input"][:, :, t] = gates_func(current_x, prev_a, params["wi"], params["bi"], "sigmoid")  # input  gate value
       values["output"][:, :, t] = gates_func(current_x, prev_a, params["wo"], params["bo"], "sigmoid")  # output gate value
       values["candidate_cell"][:,:,t] = gates_func(current_x, prev_a, params["wc"], params["bc"], "tanh")  # the candidate of the new cell value
       #2
       new_cell = values["forget"][:,:,t] * prev_c + values["input"] [:,:,t]* values["candidate_cell"][:,:,t]  # calculate the next cell state
       #3
       new_a = values["output"][:,:,t] * activation("tanh", new_cell)  # the new hidden state value
       #4
       new_y = activation("softmax", np.dot(params["wy"], new_a) + params["by"]) # calculate possibilities
       a[:, :, t + 1] = new_a
       c[:, :, t + 1] = new_cell
       y[:, :, t] = new_y

   values["x"]=x
   values["y"]=y
   values["a"]=a
   values["c"]=c
   return values
def init_params():
    params={}
    n_x=2
    m=5
    n_a=50
    n_y=2

    params["wf"]=np.random.randn(n_a+n_x, m)
    params["wi"]=np.random.randn(n_a+n_x, m)
    params["wo"]=np.random.randn(n_a+n_x, m)
    params["wy"]=np.random.randn(n_a+n_x, m)
    params["wc"]=np.random.randn(n_a+n_x, m)

    params["bc"]=np.zeros((n_a,1))
    params["by"] = np.zeros ((n_a, 1))
    params["bi"] = np.zeros ((n_a, 1))
    params["bo"] = np.zeros ((n_a, 1))
    params["bf"] = np.zeros ((n_a, 1))
    return params
def main():
    input=sequence(5)
    values=forward_propagation(input, init_params())
if __name__ == '__main__':
    main()