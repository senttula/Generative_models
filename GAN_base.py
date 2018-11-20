import time
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

from keras.layers import Lambda, Input, Dense, Conv2D, UpSampling2D, Flatten, Reshape, Dropout, LeakyReLU, \
    AveragePooling2D
from keras.models import Model, clone_model
from keras.optimizers import Adam, RMSprop, Adadelta, SGD
from keras.datasets import mnist
from keras import backend as K


class Args:
    # network parameters
    batch_size = 128
    number_of_batches = 50000
    log_frequency = 1000

    latent_dim = 2
    intermediate_dim = 32
    node_count = 8

    original_dim = None
    input_shape = None

    temp_folder = 'temp_folder'


class DataContainer:
    def __init__(self):
        # MNIST dataset
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        image_size = x_train.shape[1]
        Args.original_dim = image_size * image_size
        Args.input_shape = (x_train.shape[1], x_train.shape[2], 1)
        # x_train = np.reshape(x_train, [-1, original_dim])
        # x_test = np.reshape(x_test, [-1, original_dim])
        x_train = x_train[:, :, :, np.newaxis]
        x_test = x_test[:, :, :, np.newaxis]
        self.x_train = x_train.astype('float32') / 255
        self.x_test = x_test.astype('float32') / 255

    def data_generator(self, batch_size=Args.batch_size):
        batch_indices = np.random.randint(0, self.x_train.shape[0], size=batch_size)
        return self.x_train[batch_indices]

    def binary_noise(self, max_noise=0.3, size=Args.batch_size):
        binary = np.random.choice([-1, 1-max_noise, 0-max_noise/2], size=(size, Args.latent_dim,),
                                  p=[0.45, 0.45, 0.1])

        noisy_binary = binary + np.random.random_sample(size=binary.shape)*max_noise

        return noisy_binary


class ModelContainer:
    def __init__(self):
        self.full_gan = None
        self.generator = None

        self.discriminator_frozen = None

        self.discriminator = None

    def make_model(self):
        def base_conv(x, nodes=Args.intermediate_dim, mode=None, dropout=True):
            if dropout:
                x = Dropout(.25)(x)
            x = Conv2D(nodes, (3, 3), padding='same')(x)
            # x = BatchNormalization()(x)
            x = LeakyReLU()(x)
            if mode == 'up':
                x = UpSampling2D()(x)
            elif mode == 'down':
                x = AveragePooling2D()(x)
            return x

        def base_dense(x, nodes=Args.intermediate_dim, dropout=True):
            if dropout:
                x = Dropout(.25)(x)
            x = Dense(nodes)(x)
            # x = BatchNormalization()(x)
            x = LeakyReLU()(x)
            return x

        def make_generator(latent_inputs):
            layer = base_dense(latent_inputs, 7 * 7 * Args.node_count * 4, dropout=False)
            layer = Reshape((7, 7, Args.node_count * 4))(layer)
            layer = base_conv(layer, Args.node_count * 4, 'up')
            layer = base_conv(layer, Args.node_count * 2, 'up')
            layer = base_conv(layer, Args.node_count)
            output_img = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(layer)
            return output_img

        def make_discriminator(image_input):
            layer = base_conv(image_input, Args.node_count, 'down', dropout=False)
            layer = base_conv(layer, Args.node_count * 2, 'down')
            layer = base_conv(layer, Args.node_count * 4)
            layer = Flatten()(layer)
            layer = base_dense(layer, Args.node_count)
            chance_of_real = Dense(1, activation='sigmoid', name='chance_of_real', use_bias=False)(layer)
            return chance_of_real

        latent_inputs = Input(shape=(Args.latent_dim,), name='latent_in')
        latent_inputs2 = Input(shape=(Args.latent_dim,), name='latent_in2')
        image_input = Input(shape=Args.input_shape, name='image_input')

        # generator
        output_img = make_generator(latent_inputs)
        self.generator = Model(latent_inputs, output_img, name='generator')

        # discriminator
        chance_of_real = make_discriminator(image_input)
        self.discriminator = Model(image_input, chance_of_real, name='discriminator')

        # frozen discriminator won't be trained, but lets gradient flow through to train generator
        # its weigths are updated after each batch
        self.discriminator_frozen = clone_model(self.discriminator)
        self.discriminator_frozen.name = 'discrimnator_clone'
        self.discriminator_frozen.trainable = False

        gradient_stop_layer = Lambda(lambda x: K.stop_gradient(x), name='stop_gradient')

        self.validity_of_generator = self.discriminator_frozen(self.generator(latent_inputs))
        self.validity_of_trues = self.discriminator(image_input)

        stopped_gradient = gradient_stop_layer(self.generator(latent_inputs2))
        self.validity_of_fakes = self.discriminator(stopped_gradient)

        self.full_gan = Model([image_input, latent_inputs, latent_inputs2],
                        [self.validity_of_trues, self.validity_of_fakes, self.validity_of_generator])

        self.discriminator.summary()
        self.generator.summary()
        self.full_gan.summary()

    def make_loss(self):
        predict_loss_1 = K.mean(-K.log(self.validity_of_trues))
        predict_loss_2 = K.mean(-K.log(1-self.validity_of_fakes))

        predict_loss_3 = K.mean(-K.log(self.validity_of_generator))

        # TODO regularization

        gan_loss = K.sum(predict_loss_1 + predict_loss_2 + predict_loss_3)

        self.full_gan.add_loss(gan_loss)

    def compile_gan(self):
        opt = Adam(lr=10**-4, clipvalue=0.5)
        #opt = RMSprop(lr=10**-4)
        # opt = Adadelta(lr=10**-3)
        # opt = SGD(lr=10**-8)
        self.full_gan.compile(optimizer=opt)

    def update_frozen(self):
        self.discriminator_frozen.set_weights(self.discriminator.get_weights())  # updates frozen weigths


class GAN:
    def __init__(self):
        self.dg = DataContainer()
        self.md = ModelContainer()

        self.md.make_model()
        self.md.make_loss()
        self.md.compile_gan()

    def main(self):
        self.train()
        self.draw()

    def progress_print(self, loss=0):
        predicts = self.md.full_gan.predict([self.dg.x_test,
                                             self.dg.binary_noise(size=self.dg.x_test.shape[0]),
                                             self.dg.binary_noise(size=self.dg.x_test.shape[0])])

        print('time:%5d, trues: %.4f, fakes: %.4f, generator: %.4f, batch_loss: %4.2f' %
              (time.clock(), np.mean(predicts[0]), np.mean(predicts[1]), np.mean(predicts[2]), loss))

    def train(self):
        for i in range(Args.number_of_batches):
            batch_loss = self.md.full_gan.train_on_batch([self.dg.data_generator(),
                                                self.dg.binary_noise(), self.dg.binary_noise()], None)
            self.md.update_frozen()

            if i % Args.log_frequency == 0:
                self.progress_print(batch_loss)
                self.draw()
                if np.isnan(batch_loss):
                    print('encountered nan, ending...')
                    quit()

    def draw(self):
        if Args.latent_dim != 2:
            return
        else:
            filename = os.path.join(Args.temp_folder, str(int(time.clock())).zfill(5)+"digits_over_latent.png")
            # display a 30x30 2D manifold of digits
            n = 16
            digit_size = 28
            figure = np.zeros((digit_size * n, digit_size * n))
            # linearly spaced coordinates corresponding to the 2D plot
            # of digit classes in the latent space
            grid_x = np.linspace(-1, 1, n)
            grid_y = np.linspace(-1, 1, n)[::-1]

            for i, yi in enumerate(grid_y):
                for j, xi in enumerate(grid_x):
                    z_sample = np.array([[xi, yi]])
                    x_decoded = self.md.generator.predict(z_sample)
                    digit = x_decoded[0].squeeze()
                    figure[i * digit_size: (i + 1) * digit_size,
                           j * digit_size: (j + 1) * digit_size] = digit

            plt.figure(figsize=(10, 10))
            start_range = digit_size // 2
            end_range = n * digit_size + start_range + 1
            pixel_range = np.arange(start_range, end_range, digit_size)
            sample_range_x = np.round(grid_x, 1)
            sample_range_y = np.round(grid_y, 1)
            plt.xticks(pixel_range, sample_range_x)
            plt.yticks(pixel_range, sample_range_y)
            plt.imshow(figure, cmap='Greys_r')

            plt.savefig(filename)
            plt.close()
            # plt.show()


def make_gif():
    try:
        import imageio
        import datetime
        images = []
        files = glob.glob(os.path.join(Args.temp_folder, '*.png'))
        for filename in files:
            images.append(imageio.imread(filename))
        output_file = 'Gif-%s.gif' % datetime.datetime.now().strftime('%d-%H-%M-%S')
        imageio.mimwrite(output_file, images, duration=0.1)
    except Exception as e:
        print(e)

def clean_folder():
    os.makedirs(Args.temp_folder, exist_ok=True)
    for the_file in os.listdir(Args.temp_folder):
        file_path = os.path.join(Args.temp_folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)

if __name__ == '__main__':
    clean_folder()

    gan_class = GAN()
    gan_class.main()
    make_gif()
