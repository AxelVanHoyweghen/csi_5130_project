from keras.models import Sequential
from keras.layers.core import Dense
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import r2_score
import numpy as np

# seed for reproducing same results
seed = 9
np.random.seed(seed)

# load pima indians dataset
dataset = np.loadtxt('./dataset/trimmed/movies_metadata.csv', delimiter=',', skiprows=1)

# split into input and output variables
final_column = len(dataset[0]) - 1
print(final_column)
X = dataset[:,1:final_column]
print(X[0])
# revenue is in the first column
Y = dataset[:,0]

# split the data into training (67%) and testing (33%)
(X_train, X_test, Y_train, Y_test) = train_test_split(X, Y, test_size=0.33, random_state=seed)

# Feature Scaling
# X_train = preprocessing.scale(X_train)
# X_test =  preprocessing.scale(X_test)

# create the model
model = Sequential()
# input layer
model.add(Dense(final_column-1, input_dim=final_column-1, kernel_initializer = 'uniform', activation = 'tanh'))

# hidden layers
neurons = final_column-1
while neurons > 6:
    # add a layer with half the neurons
    neurons = neurons//2
    neurons = neurons + neurons//2
    model.add(Dense(neurons, kernel_initializer = 'uniform', activation='linear'))
    print(neurons)

# output layer
model.add(Dense(1, kernel_initializer = 'uniform', activation='relu'))

# compile the model
model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mean_absolute_error'])

checkpoint_name = 'Weights-{epoch:03d}--{val_loss:.5f}.hdf5' 
checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose = 1, save_best_only = True, mode ='auto')
callbacks_list = [checkpoint]

# fit the model
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), nb_epoch=250, batch_size=5)

# evaluate the model
scores = model.evaluate(X_test, Y_test)
print("Accuracy eval: %.2f%%" %(scores[1]*100))

y_pred = model.predict(X_test)
score = r2_score(Y_test, y_pred) 
print("Accuracy: %.2f%%" %(score))