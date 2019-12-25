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
   concat = np.concatenate((prev_a, xt))  # concatenated array of a_prev and xt
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

   values = {
               "forget": np.zeros((n_a, m, T_x)),
               "input":  np.zeros((n_a, m, T_x)),
               "output": np.zeros((n_a, m, T_x)),
               "candidate_cell": np.zeros((n_a, m, T_x))
           }

   c = np.zeros((n_a, m, T_x + 1))  # lstm cell***
   a = np.zeros((n_a, m, T_x + 1))  # hidden states***
   y = np.zeros((n_y, m, T_x))  # output array
   a[:, :, 0] = first_a  # setting the first cell value
   for t in range(T_x):
       # calculate lstm gates values

       current_x = x[:, :, t]  # the current x value
       prev_a = a[:, :, t]# the previous value of the hidden state*
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
       new_y = activation("softmax", np.dot(params["wy"], new_a) + params["by"])
       a[:, :, t + 1] = new_a
       c[:, :, t + 1] = new_cell
       y[:, :, t] = new_y

   values["x"]=x
   values["y"]=y
   values["a"]=a
   values["c"]=c
   return values


def back_propagation(params, values,da):
   """ back propagation"""
   x = values["x"]
   y = values["y"]
   a = values["a"]
   c = values["c"]

   # gates
   output = values["output"]
   input = values["input"]
   forget= values["forget"]
   candidate = values["candidate_cell"]

   n_x, m, T_x = x.shape
   n_a, m, T_x = da.shape
   gradients={
   "da_prev":np.zeros((n_a,m)),
   "dwf" : np.zeros((n_a, n_a+n_x)),
   "dwi" :np.zeros((n_a, n_a+n_x)),
   "dwo" : np.zeros((n_a, n_a+n_x)),
   "dwc" : np.zeros((n_a, n_a+n_x)),

   "dbf" : np.zeros((n_a, 1)),
   "dbi" : np.zeros((n_a, 1)),
   "dbo" : np.zeros((n_a, 1)),
   "dbc" : np.zeros((n_a, 1))
   }
   T_x=4
   for t in reversed(range(T_x)):

       ### computing the gradient with respect to the output gate and the new_cell 3
       #values["output"][:,:,t] * activation("tanh", new_cell)
       new_cell = c[:, :, t + 1]  # the last cell value
       d_output =   (da[:,:,t]+gradients["da_prev"])*derivative("tanh", new_cell)
       d_new_cell = (da[:,:,t]+gradients["da_prev"]) * derivative("tanh", output[:,:,t])
       ###compute the gradient with respect to the forget and input gates 2
       ###new_cell = forget* prev_c + input*candidate_cell

       d_input  = d_new_cell * candidate[:,:,t]
       d_forget = d_new_cell * forget[:,:,t]
       d_prev_c = d_new_cell * c[:,:,t]
       d_candidate = d_new_cell * input[:,:,t]


       ###compute the gradient with respect to the gates' parameters
       # gate_value = activation(func, np.dot(w_gate, concat) + b_gate)
       xt = x[:, :, t]
       prev_a=a[:,:,t]
       concat = np.concatenate ((prev_a, xt))  # concatenated array of a_prev and xt
       gradients["dwf"] += np.dot(d_forget * derivative("sigmoid", forget[:,:,t]), concat.T)
       gradients["dwi"] += np.dot(d_input  * derivative("sigmoid",input[:,:,t]), concat.T)
       gradients["dwo"] += np.dot(d_output * derivative("sigmoid", output[:,:,t]), concat.T)
       gradients["dwc"] += np.dot(d_candidate * derivative("tanh", candidate[:,:,t]), concat.T)

       gradients["dbf"] += np.sum(d_forget * derivative("sigmoid", forget[:,:,t]), axis=1, keepdims=True)
       gradients["dbi"] += np.sum(d_input  * derivative("sigmoid", input[:,:,t]),  axis=1, keepdims=True)
       gradients["dbo"] += np.sum(d_output * derivative("sigmoid", output[:,:,t]), axis=1, keepdims=True)
       gradients["dbc"] += np.sum(d_output * derivative("tanh",    candidate[:,:,t]), axis=1, keepdims=True)
       gradients["da_prev"] += np.dot(derivative("tanh", candidate[:,:,t]))
   
       gradients["d"] += np.dot(d_forget * derivative("sigmoid", forget[:,:,t]), concat.T)
   return gradients
def cost_function():
   pass


# -----------------------------------------------------------------------------------------------------------------------

"""data preparation """


def reduce(text):
   reduced = list(sorted(set(text)))

   return reduced


def one_hot(char, text):
   zeros = np.zeros((len(reduce(text)), 1))  # an array of zeros
   zeros[reduce(text).index(char)] = 1
   return zeros


def prepare_data(data_set, seq_length=2):
   with open(data_set, "r") as data:
       text = "abcde"  # data.read().lower()

       x_vals = []
       y_vals = []
       print(reduce(text))

       for i in range(len(text) - seq_length + 1):
           x_vals.append([one_hot(char, text) for char in text[i:i + seq_length]])
           y_vals.append(one_hot(text[i + seq_length - 1], text))

       # reshaping x values to the form (n_x, m, T_x)
       m = len(x_vals)  # the number of training examples
       T_x = seq_length
       n_x = len(reduce(text))
       x_vals = np.reshape(x_vals, (m, T_x, n_x))  # create 3d data set
       y_vals = np.reshape()
       x_vals = x_vals / float(m)  # normalize

       return x_vals


def main():
   # print (one_hot("t", "i to"))
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

   caches = forward_propagation(a0, x, parameters)
   print (caches["a"].shape)
   da = np.random.randn(5, 10, 7)
   gradients = back_propagation(parameters, caches, da)
   """
   print("gradients[\"dx\"][1][2] =", gradients["dx"][1][2])
   print("gradients[\"dx\"].shape =", gradients["dx"].shape)
   print("gradients[\"da0\"][2][3] =", gradients["da0"][2][3])
   print("gradients[\"da0\"].shape =", gradients["da0"].shape)
   """
   print("gradients[\"dWf\"][3][1] =", gradients["dwf"][3][1])
   print("gradients[\"dWf\"].shape =", gradients["dwf"].shape)
   print("gradients[\"dWi\"][1][2] =", gradients["dwi"][1][2])
   print("gradients[\"dWi\"].shape =", gradients["dwi"].shape)
   print("gradients[\"dWc\"][3][1] =", gradients["dwc"][3][1])
   print("gradients[\"dWc\"].shape =", gradients["dwc"].shape)
   print("gradients[\"dWo\"][1][2] =", gradients["dwo"][1][2])
   print("gradients[\"dWo\"].shape =", gradients["dwo"].shape)
   print("gradients[\"dbf\"][4] =", gradients["dbf"][4])
   print("gradients[\"dbf\"].shape =", gradients["dbf"].shape)
   print("gradients[\"dbi\"][4] =", gradients["dbi"][4])
   print("gradients[\"dbi\"].shape =", gradients["dbi"].shape)
   print("gradients[\"dbc\"][4] =", gradients["dbc"][4])
   print("gradients[\"dbc\"].shape =", gradients["dbc"].shape)
   print("gradients[\"dbo\"][4] =", gradients["dbo"][4])

if __name__ == '__main__':
    main()



