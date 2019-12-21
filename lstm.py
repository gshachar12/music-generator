import numpy as np
from funcs4nn import activation, derivative

""" LSTM model-version 1.0
*General explanation:


*variables and dimensions:


x-(n_x, m, T_x)
hidden state (a)-(n_a, m, T_x+1)
cell-(n_a, m, T_x+1)
y-(n_y, m, T_x)
parameters:

    weights-

    biases-
    
reference:
-LSTM
    https://www.geeksforgeeks.org/deep-learning-introduction-to-long-short-term-memory/
    https://www.geeksforgeeks.org/long-short-term-memory-networks-explanation/
-back propagation equations
    https://www.quora.com/How-can-I-derive-the-LSTM-back-propagation-formulas
    http://arunmallya.github.io/writeups/nn/lstm/index.html#/3
"""


def gates_func(xt, a_prev, w_gate, b_gate, func):
    """calculates the gate value with respect to the given parameters
    :parameter
    xt- the current x value
    a_prev- the previous prediction
    w_gate- the gate's weight
    b_gate- the gate's bias
    func- activation function
    :returns
    gate_value- the value of the gate
    """
    concat = np.concatenate((a_prev, xt))  # concatenated array of a_prev and xt
    gate_value = activation(func, np.dot(w_gate, concat) + b_gate)
    return gate_value


def forward_propagation(first_a, x, params):
    """

    :param first_a: the first hidden state value
    :param x: input values
    :param first_c: the first cell value
    :param rnn_params:
    :param lstm_params:
    :return:
    """
    n_y, n_a = params["wy"].shape
    n_x, m, T_x = x.shape

    c = np.zeros((n_a, m, T_x+1)) # lstm cell***
    a = np.zeros((n_a, m, T_x+1)) # hidden states***
    y = np.zeros((n_y, m, T_x))   # output array
    a[:,:,0] = first_a              # setting the first cell value

    for t in range(T_x):
        # calculate lstm gates values

        current_x = x[:, :, t] # the current x value
        prev_a=a[:,:,t]
        prev_c=c[:,:,t]
        forget = gates_func(current_x, prev_a, params["wf"], params["bf"], "sigmoid") # forget gate value
        input  = gates_func(current_x, prev_a, params["wi"], params["bi"], "sigmoid")  # input  gate value
        output = gates_func(current_x, prev_a, params["wo"], params["bo"], "sigmoid") # output gate value

        candidate_cell = gates_func(current_x, prev_a, params["wc"], params["bc"], "tanh")  # the candidate of the new cell value
        new_cell = forget* prev_c + input*candidate_cell # calculate the next cell state
        new_a = output * activation("tanh", new_cell)  # the new hidden state value
        new_y = activation("softmax", np.dot(params["wy"], new_a)+params["by"])
        a[:, :, t+1] =new_a
        c[:,:, t+1]=new_cell
        y[:,:,t]=new_y
        prev_a=a[:,:,t]  # the previous value of the hidden state*
        prev_c=c[:,:,t]
    return a,y,c#{"a":a, "y":y, "c":c, "forget":forget, "input": input:
def gates_func_backward(d_gate, xt, a_prev, w_gate, b_gate, func):

    concat = np.concatenate((a_prev, xt))  # concatenated array of a_prev and xt
    =d_forget * np.dot(concat, derivative("sigmoid", forget))
    return
def back_propagation(params, dh, x, c):
    """ back propagation"""

    ny, na = params["w_ya"].shape
    n_x, m, T_x = x.shape

    for t in reversed(range(T_x)):
        ###

        ### computing the gradient with respect to the output gate and the new_cell
        new_cell=c[:,:,t+1] # the last cell value
        d_output=activation("tanh", new_cell)
        d_new_cell=output*derivative("tanh",new_cell)
        ###

        ###compute the gradient with respect to the forget and input gates
        ###new_cell = forget* prev_c + input*candidate_cell
        prev_c=c[:,:,t]
        d_input=d_new_cell*candidate_cell
        d_forget=d_new_cell*prev_c
        d_prev_c=d_new_cell*forget
        d_candidate=d_new_cell*input
        ###

        ###compute the gradient with respect to the gates' parameters
        # gate_value = activation(func, np.dot(w_gate, concat) + b_gate)
        xt=x[:,:,t]
        dwf = d_forget * np.dot(w.T,derivative("sigmoid", forget))
        dwi = d_input * np.dot(w.T, derivative("sigmoid", forget))
        dwo = d_output * np.dot(w.T, derivative("sigmoid", forget))
        dwc = d_candidate * np.dot( derivative("tanh", forget))

        dbf = d_forget * np.dot(w.T,derivative("sigmoid", forget))
        dbi = d_input * np.dot(w.T, derivative("sigmoid", forget))
        dbo = d_output * np.dot(w.T, derivative("sigmoid", forget))
        dbc = d_output * np.dot(w.T, derivative("sigmoid", forget))










def cost_function():
    pass


# -----------------------------------------------------------------------------------------------------------------------

"""data preparation """
def reduce(text):
    reduced = list(sorted(set(text)))

    return reduced

def one_hot(char, text):
    zeros=np.zeros((len(reduce(text)), 1)) # an array of zeros
    zeros[reduce(text).index(char)]=1
    return zeros


def prepare_data(data_set, seq_length=2):
    with open(data_set, "r") as data:
        text = "abcde"#data.read().lower()

        x_vals = []
        y_vals = []
        print (reduce(text))

        for i in range(len(text) - seq_length+1):

            x_vals.append([one_hot(char, text) for char in text[i:i + seq_length]])
            y_vals.append(one_hot(text[i+seq_length-1], text))

        # reshaping x values to the form (n_x, m, T_x)
        m=len(x_vals) # the number of training examples
        T_x=seq_length
        n_x=len(reduce(text))
        x_vals=np.reshape(x_vals,(m, T_x, n_x)) # create 3d data set
        y_vals=np.reshape()
        x_vals=x_vals/float(m) # normalize

        return x_vals


def main():


    # initializing first hidden state and first cell value
    h0=np.zeros((n_h, m))
    c0=np.zeros((n_h, m))
    #print (one_hot("t", "i to"))
    """text="i to"
    x_vals=prepare_data("c://project/lstm/alice.txt")
    m, T_x, n_x=x_vals.shape
    print( (x_vals/float(len(reduce(text)))))
"""
    np.random.seed(1)
    x = np.random.randn(3, 10, 7)
    a0 = np.random.randn(5, 10)
    Wf = np.random.randn(5, 5 + 3)
    bf = np.random.randn(5, 1)
    Wi = np.random.randn(5, 5 + 3)
    bi = np.random.randn(5, 1)
    Wo = np.random.randn(5, 5 + 3)
    bo = np.random.randn(5, 1)
    Wc = np.random.randn(5, 5 + 3)
    bc = np.random.randn(5, 1)
    Wy = np.random.randn(2, 5)
    by = np.random.randn(2, 1)

    parameters = {"wf": Wf, "wi": Wi, "wo": Wo, "wc": Wc, "wy": Wy, "bf": bf, "bi": bi, "bo": bo, "bc": bc, "by": by}

    a, y, c = forward_propagation(a0, x, parameters)
    print("a[4][3][6] = ", a[4][3][7])
    print("a.shape = ", a.shape)
    print("y[1][4][3] =", y[1][4][3])
    print("y.shape = ", y.shape)

if __name__ == '__main__':
    main()