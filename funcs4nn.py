import numpy as np


def init_params(L, layers):
    # initialize random weights and biases

    params = {}
    for l in range(1, L):
        params["w" + str(l)] = np.random.randn(layers[l],layers[l - 1]) * 0.1

        params["b" + str(l)] = np.zeros((layers[l], 1))

    return params


def activation( activation, z):

    # activation functions that can be used for forward propagation
    activations = {"sigmoid": np.exp(z)/ (np.exp(z)+1), "relu": np.maximum(0, z), "tanh": np.tanh(z), "softmax": np.exp(z - np.max(z))/ np.sum(np.exp(z- np.max(z)), axis=0)}
    return activations[activation]


def forward_prop(L, x, params):

    a_values = {}  # y_hat parameters
    z_values = {}
    current_a = x  # the current input

    a_values["a" + str(0)] = x
    for l in range(1, L):
        w = params["w" + str(l)]
        b = params["b" + str(l)]
        z = np.dot(w, current_a) + b  # linear function

        z_values["z" + str(l)] = z

        if l == L - 1:
            current_a = activation("sigmoid", z)

        else:
            current_a = activation("relu", z)

        a_values["a" + str(l)] = current_a
    print ("a_va", a_values)
    return a_values, z_values


def cost_func(y_hat, y):
    m = y.shape[1]
    epsilon = 1e-4  # small value in order to prevent division by 0

    cost = np.squeeze(
        - (np.dot(y, np.log(y_hat + epsilon).T) + np.dot(1 - y, np.log(1 - y_hat + epsilon).T)) / m)
    return cost


def relu_back( a):
    s=a
    s[s>0] = 1
    return s


def derivative(activation, a):

    activations = {"sigmoid": a * (1 - a),
                   "tanh": a**2-1}
                 #  "relu": relu_back(a)


    return activations[activation]


def backward_prop(L, params, a_values, y):
    derivatives = {}
    a = a_values["a" + str(L - 1)]  # the final output of the network
    # self.y = self.y.reshape(a.shape)
    epsilon = 1e-5  # small value in order to prevent division by 0
    m = a.shape[1]
    # da =  np.divide(a-self.y,(a+epsilon)*(1-a+epsilon)) for some reason it doesn't work
    da = -y / (a + epsilon) + (1 -y) / (1 - (a + epsilon))  # np.divide((a+epsilon)*(1-a+epsilon),a-self.y )
    dz = da * derivative("sigmoid", a)  # the derivative of sigmoid
    for l in reversed(range(L - 1)):
        a = a_values["a" + str(l)]
        m = a.shape[1]

        derivatives["dw" + str(l + 1)] = dz.dot(a.T) / float(m)

        derivatives["db" + str(l + 1)] = np.sum(dz, axis=1, keepdims=True) / float(m)
        w = params["w" + str(l + 1)]

        da = w.T.dot(dz)
        dz = da*derivative("relu", a)  # the derivative of relu
    return derivatives


def update_parameters(L, learning_rate, params, derivatives):
    for l in range(L - 1):
        w = params["w" + str(l + 1)]
        b = params["b" + str(l + 1)]

        dw = derivatives["dw" + str(l + 1)]
        db = derivatives["db" + str(l + 1)]

        params["w" + str(l + 1)] = w - learning_rate * dw

        params["b" + str(l + 1)] = b - learning_rate * db