# 2. Library
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

import time
start = time.time()             # 시작 시간

# 3. Data Loading
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_test = (x_test.astype(np.float32) - 127.5)/127.5   # Normalization: -1.0 ~ 1.0
mnist_data = x_test.reshape(10000, 784)              # to 1 column
# print(mnist_data.shape)
# print(len(mnist_data))

# 4. Making of Generator NN
def create_generator():
    generator = Sequential()
    generator.add(Dense(units=256, input_dim=100))
    generator.add(LeakyReLU(alpha=0.2))     # slope in minus area
    generator.add(Dense(units=512))
    generator.add(LeakyReLU(0.2))
    generator.add(Dense(units=784, activation='tanh'))
    return generator
g = create_generator()
g.summary()

# 5. Making of Discriminator NN
def create_discriminator():
    discriminator = Sequential()
    discriminator.add(Dense(units=512, input_dim=784))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dense(units=256))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dense(units=1, activation='sigmoid'))
    discriminator.compile(loss='binary_crossentropy', \
                          optimizer=Adam(learning_rate=0.0002,beta_1=0.5))
    return discriminator
d = create_discriminator()
d.summary()

# 6. Making of GAN Function, Train for generator
def create_gan(discriminator, generator):
    discriminator.trainable = False
    gan_input = Input(shape=(100,)) # generator input 100x10000 noise pixels
    x = generator(gan_input)        # generator output 784 pixels, Train for generator
    gan_output = discriminator(x)   # discriminator output 1(=T/F)
    
    gan = Model(inputs=gan_input, outputs=gan_output)
    gan.compile(loss='binary_crossentropy', optimizer='adam')
    return gan
gan = create_gan(d, g)
gan.summary()

# 7. Function of producing an image and Showing
def plot_generated_images(generator):
    # noise = np.random.normal(loc=0, scale=1, size=[100, 100])   # mistake, sd=1.0
    noise = 2 * np.random.rand(100, 100) - 1                    # -1.0 ~ 1.0
    generated_images = generator.predict(noise)
    generated_images = generated_images.reshape(100, 28, 28)
    plt.figure(figsize=(10, 10))
    for i in range(generated_images.shape[0]):
        plt.subplot(10, 10, i+1)
        plt.imshow(generated_images[i], interpolation='nearest')
        plt.axis('off')
    plt.tight_layout()
    
# plot_generated_images(g)              # Showing the test noise image
# plt.show()
    
# 8. Training of GAN
batch_size = 128
epochs = 5001
for e in tqdm(range(epochs)):
    # noise = np.random.normal(0, 1, [batch_size, 100])   # generated image, mistake, sd=1.0
    noise = 2 * np.random.rand(batch_size, 100) - 1     # -1.0 ~ 1.0
    generated_images = g.predict(noise)
    image_batch = mnist_data[np.random.randint(low=0, \
                        high=mnist_data.shape[0],size=batch_size)] # label images
        
    X = np.concatenate([image_batch, generated_images]) # X data for training    
    y_dis = np.zeros(2*batch_size)                      # y data for training
    y_dis[:batch_size] = 1
    
    d.trainable = True                      # Train for discriminator, first
    d.train_on_batch(X, y_dis)
    
    # noise = np.random.normal(0, 1, [batch_size, 100])   # mistake, sd=1.0
    noise = 2 * np.random.rand(batch_size, 100) - 1     # -1.0 ~ 1.0
    y_gen = np.ones(batch_size)
    d.trainable = False
    gan.train_on_batch(noise, y_gen)        # Train for generator
    
    if e % 1000 == 0:
        print("\nEpoch is ", e)
        plot_generated_images(g)
        plt.show()
        
print("\nProcessed time: ", time.time() - start, " sec") # 현재 시각 - 시잣 시간 = 실행 시간


    
    
    
    
    