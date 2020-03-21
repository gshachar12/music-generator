import os
from music21 import *
import numpy as np
from keras.layers import *
from keras.models import Sequential, Model
import tensorflow as tf
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.callbacks.callbacks import ModelCheckpoint
# network parameters
import tqdm
notes = ['a', 'a#', 'b', 'c', 'c#', 'd', 'd#', 'e', 'f', 'f#', 'g', 'g#']
num_notes = 78 # number of notes used in each vector
epochs=400
num_cells=256
# create an LSTM cell which returns the hidden state value (which makes it possible to generate notes) and the cell state value.
np.random.seed(0)
softmax = Dense(num_notes, activation='softmax')  # performs matrix multiplication using softmax activation function

T_x = 30  # number of time steps
m=100  # number of training examples
T_y= 1  # the length of an output notes sequence

def download_midifiles():
    pass

def scale_augmentation():

    pass

def one_hot_encoding(list):
    #unique=sorted(set(list)) # returning only the unique values in the list
    vectors=[] # a list that will include the one hot vectors
    for event in list:
        notes_vector=np.zeros(num_notes)
        event=event.split('.')
        for note in event:

            octave=int(note[-1]) # the octave of the note
            note=note[:-1].lower()
            if "-" in note: # flat to sharp conversion
                note=notes[notes.index(note[0])-1]
            note_index=notes.index(note)+12*(octave-1)

            notes_vector[note_index]=1
            vectors.append(notes_vector)
    return vectors

def event_list():
    download_midifiles()

    file = "c://project/music/well tempered/fine.mid"
    components = instrument.partitionByInstrument(converter.parse (file)) # convert the midi file into a music21 stream
    events = []
    for event in components.recurse():

        if type(event) is note.Note: # single note object
            events.append (str(event.pitch))
        if type(event) is chord.Chord:
            events.append('.'.join([str(n) for n in event.pitches]))

    return events


def get_dataset(dataset_file='c://project/dataset.npz'):

        if not os.path.exists (dataset_file):
            dataset = prepare_data ()
            print ("created")


        dataset = np.load (dataset_file)
        return dataset

def prepare_data(dataset='c://project/dataset.npz'):
    events = one_hot_encoding(event_list())
    input_seq = []  # notes input sequence with length of sequence length
    output_seq = []

    #try:
    for i in range(m):
        input_seq.append(events[i:i+T_x])
        output_seq.append(events[i+T_x:i+T_x+T_y])
    # transforming the inputs to one hot vectors-

    input_seq = np.reshape(input_seq, (m, T_x, num_notes) )
    print (len(input_seq))
    output_seq = np.reshape(output_seq, (m, num_notes) )

    np.savez(dataset, name1=input_seq, name2=output_seq)
    #except:
    #    print ("sequence length is too large")

def vectors2stream(vectors):

    piece = stream.Score ()
    part= stream.Part()

    for vector in vectors:
        note=vector2note(vector)
        part.append(note)

    piece.insert(part)
    s = midi.realtime.StreamPlayer(piece)
    while True:
        s.play()

def build_lstm_model(X, Y):
    # utilizes keras built LSTM model for sequence generation
    # the function creates a general model, which will be used for music generation


    # useful functions
    lstm_model=Sequential()
    lstm_model.add(LSTM(128, input_shape=(T_x, num_notes)))  # insert 128 cells of LSTM that will be applied on the given input
    lstm_model.add(softmax) # add a Softmax layer
    lstm_model.add(Dropout())
    optimization(lstm_model)
    lstm_model.summary() # print model description
    if not os.path.isdir("c://project/lstm_model"):
        os.makedirs("c://project/lstm_model")
    check_point=ModelCheckpoint("c://project/lstm_model/music_generator-{loss:.2f}.h5", verbose=1) # save the model after each epoch

    lstm_model.fit (X, Y, batch_size=128, epochs=epochs, callbacks=[check_point])
    return lstm_model


# GRADED FUNCTION: music_inference_model

def prediction(length):
    lstm_model=Sequential()
    lstm_model.add(LSTM(128, input_shape=(1, num_notes)))  # insert 128 cells of LSTM that will be applied on the given input
    lstm_model.add(softmax) # add a Softmax layer

    lstm_model.load_weights("c://project/lstm_model/music_generator-3.65.h5")
    x_initializer = np.zeros ((1, 1, num_notes)) # first note to be fed into the LSTM
    note_index=38
    x_initializer[:,:,note_index]=1 # initialize the first note
    outputs=[]
    for i in range(length):
        next=lstm_model.predict(x_initializer) # predict the next x value
        # transform into one hot vector type
        x_initializer = np.zeros ((1, 1, num_notes))  # first note to be fed into the LSTM
        note_index=np.argmax(next) # the index of the turned on note
        x_initializer[:, :, note_index] = 1  # initialize the first note
        outputs.append(x_initializer)
    return outputs


def optimization(model):
    # calculate optimization and compiles the model
    from keras.optimizers import SGD
    opt = SGD (lr=0.01)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy']) #!!!!!!!
    # the Adam parameters are


def vector2note(vector):
    notes_list=[]
    vector=vector.reshape(num_notes) # convert the vector to a useable shape
    print (vector.shape)
    for i in tqdm.tqdm(range(len(vector))):
         if vector[i]==1:  # note on event
             octave, note=divmod(i, 12) # calculate the octave and the number of the note
             octave+=1 # human readable octave
             note= notes[note]+str(octave) # the string which fits the note
             notes_list.append(note)
    return chord.Chord(notes_list)

def form_melody():
    pass
def main():
    dataset=get_dataset()
    X=dataset["name1"] # the input sequences
    Y=dataset["name2"] # the output sequences
    build_lstm_model(X, Y)

    outputs=prediction(100)
    vectors2stream(outputs)


if __name__ == '__main__':
    main()