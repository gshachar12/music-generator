def one_hot(vocabulary_size, input_val):
    """
    :
    build a one hot vector from the given input

    :param:
     vocabulary_size: the number of unique characters in the database
     input_val: the index of the value in the list of unique characters
    :return:
    """
    one_hot=np.zeros((vocabulary_size, 1)) # a new array of zeros
    one_hot[input]=1

    return one_hot

