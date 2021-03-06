from keras.models import Sequential
from keras.layers.core import Dense
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
import os
from datetime import datetime
from csv import writer
import gc

y_estimate = {
    'categorical': {
        'file': 'movies_metadata_y_categorical.csv',
        'loss_function': 'categorical_crossentropy',
        'ol_activation': 'softmax'
    },
    'profit': {
        'file': 'movies_metadata_y_profit.csv',
        'loss_function': 'binary_crossentropy',
        'ol_activation': 'sigmoid'
    },
    'minmax': {
        'file': 'movies_metadata_y_minmax.csv',
        'loss_function': 'mean_squared_error',
        'ol_activation': 'linear'
    }
}
y_estimate_selected = input("Y estimate: ")


# Evaluate model
def eval_model(loss_fn, opt_fn, hl_activation, ol_activation, nb_epochs, batch_size, hidden_layer_neurons):
    model = Sequential()

    model.add(Dense(X.shape[1], input_dim=X.shape[1], activation = hl_activation))

    # hidden layers
    nb_layers = 0
    neurons = X.shape[1]
    while neurons > Y.shape[1]*2:
        # add a layer with half the neurons
        if hidden_layer_neurons == 'case_1':
            neurons = neurons//2
        elif hidden_layer_neurons == 'case_2':
            neurons = neurons//2
            neurons = neurons + neurons//2
        # add layer
        model.add(Dense(neurons, activation=hl_activation))
        nb_layers = nb_layers + 1

    # output layer
    model.add(Dense(Y.shape[1], activation=ol_activation))
    # compile the model
    model.compile(loss=loss_fn, optimizer='sgd', metrics=['accuracy'])

    # fit the model
    model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=nb_epochs, batch_size=batch_size)

    # evaluate the model
    eval_model=model.evaluate(X_train, Y_train)
    print(eval_model)
    print("Accuracy eval: %.2f%%" %(eval_model[1]*100))
    del model
    print(gc.collect())
    return [datetime.now().strftime("%m/%d/%Y, %H:%M:%S"), y_estimate_selected, loss_fn, opt_fn, hl_activation, ol_activation, nb_epochs, batch_size, hidden_layer_neurons, nb_layers, eval_model[0], eval_model[1]]

def append_row_to_file(list_of_elem):
    # Open file in append mode
    with open('./stats/testing.csv', 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        csv_writer = writer(write_obj)
        # Add contents of list as last row in the csv file
        csv_writer.writerow(list_of_elem)

# seed for reproducing same results
seed = 9
np.random.seed(seed)

# categories are in the first 4 columns
y_file = y_estimate[y_estimate_selected]['file']
Y = pd.read_csv('./dataset/trimmed/' + y_file, low_memory=False)

X = pd.read_csv('./dataset/trimmed/movies_metadata_x.csv', low_memory=False)

# split the data into training (67%) and testing (33%)
(X_train, X_test, Y_train, Y_test) = train_test_split(X, Y, test_size=0.33, random_state=seed)

# initialize csv
if os.path.exists('./stats/testing.csv') == False:
    new_file = pd.DataFrame(columns=['current_time', 'y_estimate_type', 'loss_fn', 'optimizer', 'hl_activation', 'ol_activation', 'nb_epochs', 'batch_size', 'hidden_layer_neurons', 'loss', 'accuracy'])
    new_file.to_csv('./stats/testing.csv', index=False)

# find the best model
# curr_model = eval_model('relu', 'relu', 'softmax', 100, 10, 'case_1')
# print(curr_model)
# append_row_to_file(curr_model)
# need to iterate over all different parameters to find the best model

# parameters
batch_sizes = [128, 64, 32, 10]
np_epoches = [10, 100, 500, 750]
activations = ['sigmoid', 'tanh', 'relu']
hidden_layer_neurons = ['case_1', 'case_2']
optimizer_fn = ['adam', 'sgd', 'rmsprop']

for b_size in batch_sizes:
    for np_epoch in np_epoches:
        for hl_neuron in hidden_layer_neurons:
            for hl_activation in activations:
                for opt_fn in optimizer_fn:
                    print(y_estimate[y_estimate_selected]['loss_function'], opt_fn, hl_activation, y_estimate[y_estimate_selected]['ol_activation'], np_epoch, b_size, hl_neuron)
                    curr_model = eval_model(y_estimate[y_estimate_selected]['loss_function'], opt_fn, hl_activation, y_estimate[y_estimate_selected]['ol_activation'], np_epoch, b_size, hl_neuron)
                    append_row_to_file(curr_model)

# # create the model
# model = Sequential()
# # input layer
# model.add(Dense(X.shape[1], input_dim=X.shape[1], activation = 'relu'))

# # hidden layers
# neurons = X.shape[1]
# while neurons > Y.shape[1]*2:
#     # add a layer with half the neurons
#     neurons = neurons//2
#     neurons = neurons + neurons//2
#     model.add(Dense(neurons, activation='relu'))

# # output layer
# model.add(Dense(Y.shape[1], activation='softmax'))

# # compile the model
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# # fit the model
# model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=100, batch_size=10)

# # evaluate the model
# eval_model=model.evaluate(X_train, Y_train)
# print(eval_model)
# print("Accuracy eval: %.2f%%" %(eval_model[1]*100))