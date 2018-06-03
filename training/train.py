# This code based on Siraj Raval's code (https://github.com/llSourcell/How_to_simulate_a_self_driving_car)

"""
Training module. Based on "End to End Learning for Self-Driving Cars" research paper by Nvidia.
"""

import argparse

import h5py
import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split  # to split out training and testing data

# path with training files
from data_collection.data_collect import path
from training.model import build_model
# helper class
from training.utils import batch_generator

# for debugging, allows for reproducible (deterministic) results
np.random.seed(0)


def load_data(args):
    """
    Load training data and split it into training and validation set
    """
    data = h5py.File(path, 'r')
    # list of all possible indexes
    indexes = list(range(data['img'].shape[0]))
    # split the data into a training (80), testing(20), and validation set
    indexes_train, indexes_valid = train_test_split(indexes, test_size=args.test_size, random_state=0)

    return data, indexes_train, indexes_valid


def load_weights(model):
    """
    Load weights from previously trained model
    """
    prev_model = load_model("..\\training\\base_model.h5")
    model.set_weights(prev_model.get_weights())

    return model


def train_model(model, args, data, indexes_train, indexes_valid):
    """
    Train the model
    """
    # Saves the model after every epoch.
    # quantity to monitor, verbosity i.e logging mode (0 or 1),
    # if save_best_only is true the latest best model according to the quantity monitored will not be overwritten.
    # mode: one of {auto, min, max}. If save_best_only=True, the decision to overwrite the current save file is
    # made based on either the maximization or the minimization of the monitored quantity. For val_acc,
    # this should be max, for val_loss this should be min, etc. In auto mode, the direction is automatically
    # inferred from the name of the monitored quantity.
    checkpoint = ModelCheckpoint('model-{epoch:03d}.h5',
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only=args.save_best_only,
                                 mode='auto')

    # calculate the difference between expected steering angle and actual steering angle
    # square the difference
    # add up all those differences for as many data points as we have
    # divide by the number of them
    # that value is our mean squared error! this is what we want to minimize via
    # gradient descent
    model.compile(loss='mean_squared_error', optimizer=Adam(lr=args.learning_rate))

    # Fits the model on data generated batch-by-batch by a Python generator.

    # The generator is run in parallel to the model, for efficiency.
    # For instance, this allows you to do real-time data augmentation on images on CPU in
    # parallel to training your model on GPU.
    # so we reshape our data into their appropriate batches and train our model simultaneously
    model.fit_generator(batch_generator(data, indexes_train, args.batch_size, True),
                        steps_per_epoch=len(indexes_train) / args.batch_size,
                        epochs=args.nb_epoch,
                        max_queue_size=1,
                        validation_data=batch_generator(data, indexes_valid, args.batch_size, False),
                        validation_steps=len(indexes_valid) / args.batch_size,
                        callbacks=[checkpoint],
                        verbose=1)


# for command line args
def s2b(s):
    """
    Converts a string to boolean value
    """
    s = s.lower()
    return s == 'true' or s == 'yes' or s == 'y' or s == '1'


def main():
    """
    Load train/validation data set and train the model
    """
    # The argparse module makes it easy to write user-friendly command-line interfaces.
    parser = argparse.ArgumentParser(description='Behavioral Cloning Training Program')
    parser.add_argument('-d', help='data directory', dest='data_dir', type=str, default=path)
    parser.add_argument('-t', help='test size fraction', dest='test_size', type=float, default=0.2)
    parser.add_argument('-k', help='drop out probability', dest='keep_prob', type=float, default=0.5)
    parser.add_argument('-n', help='number of epochs', dest='nb_epoch', type=int, default=200)
    parser.add_argument('-b', help='batch size', dest='batch_size', type=int, default=500)
    parser.add_argument('-o', help='save best models only', dest='save_best_only', type=s2b, default='true')
    parser.add_argument('-l', help='learning rate', dest='learning_rate', type=float, default=1.0e-4)
    args = parser.parse_args()

    # print parameters
    print('-' * 30)
    print('Parameters')
    print('-' * 30)
    for key, value in vars(args).items():
        print('{:<20} := {}'.format(key, value))
    print('-' * 30)

    # load data
    data = load_data(args)
    # build model
    model = build_model(args)
    # load previous weights
    model = load_weights(model)
    # train model on data, it saves as model.h5
    train_model(model, args, *data)


if __name__ == '__main__':
    main()
