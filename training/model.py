"""
NN model
"""

from keras.layers import Lambda, Conv2D, Dropout, Dense, Flatten, Concatenate, Input, MaxPooling2D
from keras.models import Model

from training.utils import INPUT_SHAPE, RADAR_SHAPE


# original Nvidia model
# def build_model(args):
#     """
#     NVIDIA model used
#     Image normalization to avoid saturation and make gradients work better.
#     Convolution: 5x5, filter: 24, strides: 2x2, activation: ELU
#     Convolution: 5x5, filter: 36, strides: 2x2, activation: ELU
#     Convolution: 5x5, filter: 48, strides: 2x2, activation: ELU
#     Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
#     Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
#     Drop out (0.5)
#     Fully connected: neurons: 100, activation: ELU
#     Fully connected: neurons: 50, activation: ELU
#     Fully connected: neurons: 10, activation: ELU
#     Fully connected: neurons: 1 (output)
#     # the convolution layers are meant to handle feature engineering
#     the fully connected layer for predicting the steering angle.
#     dropout avoids overfitting
#     ELU(Exponential linear unit) function takes care of the Vanishing gradient problem.
#     """
#     model = Sequential()
#     model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=INPUT_SHAPE))
#     model.add(Conv2D(24, (5, 5), activation='elu', strides=(2, 2)))
#     model.add(Conv2D(36, (5, 5), activation='elu', strides=(2, 2)))
#     model.add(Conv2D(48, (5, 5), activation='elu', strides=(2, 2)))
#     model.add(Conv2D(64, (3, 3), activation='elu'))
#     model.add(Conv2D(64, (3, 3), activation='elu'))
#     model.add(Dropout(args.keep_prob))
#     model.add(Flatten())
#     model.add(Dense(100, activation='elu'))
#     model.add(Dense(50, activation='elu'))
#     model.add(Dense(10, activation='elu'))
#     model.add(Dense(1))
#     model.summary()
#
#     return model


# original + radar added
# def build_model(args):
#     # image model
#     img_input = Input(shape=INPUT_SHAPE)
#     img_model = (Lambda(lambda x: x / 127.5 - 1.0, input_shape=INPUT_SHAPE))(img_input)
#     img_model = (Conv2D(24, (5, 5), activation='elu', strides=(2, 2)))(img_model)
#     img_model = (Conv2D(36, (5, 5), activation='elu', strides=(2, 2)))(img_model)
#     img_model = (Conv2D(48, (5, 5), activation='elu', strides=(2, 2)))(img_model)
#     img_model = (Conv2D(64, (3, 3), activation='elu'))(img_model)
#     img_model = (Conv2D(64, (3, 3), activation='elu'))(img_model)
#     img_model = (Dropout(args.keep_prob))(img_model)
#     img_model = (Flatten())(img_model)
#     img_model = (Dense(100, activation='elu'))(img_model)
#
#     # radar model
#     radar_input = Input(shape=RADAR_SHAPE)
#     radar_model = (Conv2D(10, (5, 5), activation='elu'))(radar_input)
#     radar_model = (MaxPooling2D((2, 2)))(radar_model)
#     radar_model = (Conv2D(20, (5, 5), activation='elu'))(radar_model)
#     radar_model = (MaxPooling2D((2, 2)))(radar_model)
#     radar_model = (Dropout(args.keep_prob / 2))(radar_model)
#     radar_model = (Flatten())(radar_model)
#     radar_model = (Dense(30, activation='elu'))(radar_model)
#
#     # combined model
#     out = Concatenate()([img_model, radar_model])
#     out = (Dense(50, activation='elu'))(out)
#     out = (Dense(10, activation='elu'))(out)
#     out = (Dense(1))(out)
#
#     final_model = Model(inputs=[img_input, radar_input], outputs=out)
#     final_model.summary()
#
#     return final_model


# original + radar and speed info added
def build_model(args):
    # image model
    img_input = Input(shape=INPUT_SHAPE)
    img_model = (Lambda(lambda x: x / 127.5 - 1.0, input_shape=INPUT_SHAPE))(img_input)
    img_model = (Conv2D(24, (5, 5), activation='elu', strides=(2, 2)))(img_model)
    img_model = (Conv2D(36, (5, 5), activation='elu', strides=(2, 2)))(img_model)
    img_model = (Conv2D(48, (5, 5), activation='elu', strides=(2, 2)))(img_model)
    img_model = (Conv2D(64, (3, 3), activation='elu'))(img_model)
    img_model = (Conv2D(64, (3, 3), activation='elu'))(img_model)
    img_model = (Dropout(args.keep_prob))(img_model)
    img_model = (Flatten())(img_model)
    img_model = (Dense(100, activation='elu'))(img_model)

    # radar model
    radar_input = Input(shape=RADAR_SHAPE)
    radar_model = (Conv2D(32, (5, 5), activation='elu'))(radar_input)
    radar_model = (MaxPooling2D((2, 2), strides=(2, 2)))(radar_model)
    radar_model = (Conv2D(64, (5, 5), activation='elu'))(radar_model)
    radar_model = (MaxPooling2D((2, 2), strides=(2, 2)))(radar_model)
    radar_model = (Dropout(args.keep_prob / 2))(radar_model)
    radar_model = (Flatten())(radar_model)
    radar_model = (Dense(10, activation='elu'))(radar_model)

    # speed
    speed_input = Input(shape=(1,))

    # combined model
    out = Concatenate()([img_model, radar_model])
    out = (Dense(50, activation='elu'))(out)
    out = Concatenate()([out, speed_input])
    out = (Dense(10, activation='elu'))(out)
    out = (Dense(1))(out)

    final_model = Model(inputs=[img_input, radar_input, speed_input], outputs=out)
    final_model.summary()

    return final_model

# original + throttle control
# def build_model(args):
#     """
#     NVIDIA model used
#     Image normalization to avoid saturation and make gradients work better.
#     Convolution: 5x5, filter: 24, strides: 2x2, activation: ELU
#     Convolution: 5x5, filter: 36, strides: 2x2, activation: ELU
#     Convolution: 5x5, filter: 48, strides: 2x2, activation: ELU
#     Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
#     Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
#     Drop out (0.5)
#     Fully connected: neurons: 100, activation: ELU
#     Fully connected: neurons: 50, activation: ELU
#     Fully connected: neurons: 10, activation: ELU
#     Fully connected: neurons: 1 (output)
#     # the convolution layers are meant to handle feature engineering
#     the fully connected layer for predicting the steering angle.
#     dropout avoids overfitting
#     ELU(Exponential linear unit) function takes care of the Vanishing gradient problem.
#     """
#     # image model
#     img_input = Input(shape=INPUT_SHAPE)
#     img_model = (Lambda(lambda x: x / 127.5 - 1.0, input_shape=INPUT_SHAPE))(img_input)
#     img_model = (Conv2D(24, (5, 5), activation='elu', strides=(2, 2)))(img_model)
#     img_model = (Conv2D(36, (5, 5), activation='elu', strides=(2, 2)))(img_model)
#     img_model = (Conv2D(48, (5, 5), activation='elu', strides=(2, 2)))(img_model)
#     img_model = (Conv2D(64, (3, 3), activation='elu'))(img_model)
#     img_model = (Conv2D(64, (3, 3), activation='elu'))(img_model)
#     img_model = (Dropout(args.keep_prob))(img_model)
#     img_model = (Flatten())(img_model)
#     img_model = (Dense(100, activation='elu'))(img_model)
#
#     # speed and direction model
#     metrics_input = Input(shape=(2,))
#     metrics_model = Dense(2, activation='elu')(metrics_input)
#
#     # combined model
#     out = Concatenate()([img_model, metrics_model])
#     out = (Dense(50, activation='elu'))(out)
#     out = (Dense(10, activation='elu'))(out)
#     out = (Dense(2))(out)
#
#     final_model = Model(inputs=[img_input, metrics_input], outputs=out)
#     final_model.summary()
#
#     return final_model
