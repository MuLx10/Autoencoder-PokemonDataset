
# coding: utf-8

# # My Session 3

# # Alternate

# In[5]:

import os

import numpy as np
import matplotlib.pyplot as plt

# from skimage.transform import resize
# from skimage import data

from scipy.misc import imresize
# import IPython.display as ipyd

import tensorflow as tf

from libs import utils, gif, datasets, dataset_utils, vae, dft

# get_ipython().magic(u'matplotlib inline')
plt.style.use('ggplot')


# ## Load

# In[10]:

data_directory = './data/pokemon/jpeg/'

output_directory = './output'
model_dir = './stored_model'

if not os.path.exists(output_directory):
    os.makedirs(output_directory)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)


experiment_title = 'lr-0.005-itr-2500'


# In[11]:

filenames = [os.path.join(data_directory, file_i)
              for file_i in os.listdir(data_directory)
              if '.jpg' in file_i]



filenames = filenames
# np.random.shuffle(filenames)

nb_clip = 100
filenames = filenames[:nb_clip]
imgs = [plt.imread(f) for f in filenames]

Xs = np.array(imgs)

# print(Xs.shape)


# In[13]:

# plt.figure(figsize=(10, 10))
# plt.imshow(utils.montage(imgs).astype(np.uint8))


# In[14]:

# get_ipython().magic(u'pinfo datasets.Dataset')


# In[15]:

ds = datasets.Dataset(Xs)


# In[16]:

mean_img = ds.mean().astype(np.uint8)
# plt.imshow(mean_img)
# print(ds.mean().shape)


# In[17]:

std_img = ds.std() #.astype(np.uint8)
# plt.imshow(std_img)
# print(std_img.shape)


# In[18]:

std_img = np.mean(std_img, axis=2)
# plt.imshow(std_img)


# In[19]:

# plt.imshow(ds.X[0])
# print(ds.X[0].shape)
# print(ds.X.shape)


# In[20]:

# Write a function to preprocess/normalize an image, given its dataset object
# (which stores the mean and standard deviation!)
def preprocess(img, ds):
    norm_img = (img - ds.mean()) / ds.std()
    return norm_img

# Write a function to undo the normalization of an image, given its dataset object
# (which stores the mean and standard deviation!)
def deprocess(norm_img, ds):
    img = norm_img * ds.std() + ds.mean()
    return img

# Just to make sure that you've coded the previous two functions correctly:
assert(np.allclose(deprocess(preprocess(ds.X[0], ds), ds), ds.X[0]))
# plt.imshow(deprocess(preprocess(ds.X[0], ds), ds).astype(np.uint8))


# In[21]:

# Calculate the number of features in your image.
# This is the total number of pixels, or (height x width x channels).
height = ds.X[0].shape[0]
width = ds.X[0].shape[1]
channels = ds.X[0].shape[2]

n_features = height * width * channels
print(height,width,channels)


# In[22]:

# print(64*64*3)


# In[23]:

# encoder_dimensions = [1024, 256, 64, 2]

# encoder_dimensions = [1024, 64, 16, 2]
encoder_dimensions = [1024, 64, 4]
# encoder_dimensions = [1024, 512, 256, 128, 64, 32, 16, 8]


# In[24]:

tf.reset_default_graph()


# In[25]:

X = tf.placeholder(tf.float32, shape = (None, n_features), name = "X")
                   
assert(X.get_shape().as_list() == [None, n_features])


# In[26]:

def encode(X, dimensions, activation=tf.nn.tanh):
    # We're going to keep every matrix we create so let's create a list to hold them all
    Ws = []

    # We'll create a for loop to create each layer:
    for layer_i, n_output in enumerate(dimensions):

        # This will simply prefix all the variables made in this scope
        # with the name we give it.  Make sure it is a unique name
        # for each layer, e.g., 'encoder/layer1', 'encoder/layer2', or
        # 'encoder/1', 'encoder/2',... 
        with tf.variable_scope("encode/layer" + str(layer_i + 1)):

            # Create a weight matrix which will increasingly reduce
            # down the amount of information in the input by performing
            # a matrix multiplication.  You can use the utils.linear function.
            h, W = utils.linear(X, dimensions[layer_i])

            # Finally we'll store the weight matrix.
            # We need to keep track of all
            # the weight matrices we've used in our encoder
            # so that we can build the decoder using the
            # same weight matrices.
            Ws.append(W)
            
            # Replace X with the current layer's output, so we can
            # use it in the next layer.
            X = h
    
    z = X
    return Ws, z


# In[27]:

# Then call the function
Ws, z = encode(X, encoder_dimensions)

# And just some checks to make sure you've done it right.
# assert(z.get_shape().as_list() == [None, 2])
# assert(len(Ws) == len(encoder_dimensions))


# In[28]:

# We'll first reverse the order of our weight matrices
decoder_Ws = Ws[::-1]

# then reverse the order of our dimensions
# appending the last layers number of inputs.
decoder_dimensions = encoder_dimensions[::-1][1:] + [n_features]
# print(decoder_dimensions)

assert(decoder_dimensions[-1] == n_features)


# In[29]:

def decode(z, dimensions, Ws, activation=tf.nn.tanh):
    current_input = z
    for layer_i, n_output in enumerate(dimensions):
        # we'll use a variable scope again to help encapsulate our variables
        # This will simply prefix all the variables made in this scope
        # with the name we give it.
        with tf.variable_scope("decoder/layer/{}".format(layer_i)):

            # Now we'll grab the weight matrix we created before and transpose it
            # So a 3072 x 784 matrix would become 784 x 3072
            # or a 256 x 64 matrix, would become 64 x 256
            W = tf.transpose(Ws[layer_i])

            # Now we'll multiply our input by our transposed W matrix
            h = tf.matmul(current_input, W)

            # And then use a relu activation function on its output
            current_input = activation(h)

            # We'll also replace n_input with the current n_output, so that on the
            # next iteration, our new number inputs will be correct.
            n_input = n_output
    Y = current_input
    return Y


# In[30]:

Y = decode(z, decoder_dimensions, decoder_Ws)


# In[31]:

Y.get_shape().as_list()


# In[32]:

# Calculate some measure of loss, e.g. the pixel to pixel absolute difference or squared difference
loss = tf.squared_difference(X, Y)

# Now sum over every pixel and then calculate the mean over the batch dimension (just like session 2!)
# hint, use tf.reduce_mean and tf.reduce_sum
cost = tf.reduce_sum(loss)


# In[33]:

learning_rate = 0.001
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)


# In[34]:

from libs import tboard
tboard.show_graph(tf.get_default_graph().as_graph_def())


# In[36]:

# Create a tensorflow session and initialize all of our weights:
sess = tf.Session()

saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())

# Some parameters for training
batch_size = 100
n_epochs = 251
step = 10

# We'll try to reconstruct the same first 100 images and show how
# The network does over the course of training.
examples = ds.X[:100]

# We have to preprocess the images before feeding them to the network.
# I'll do this once here, so we don't have to do it every iteration.
test_examples = preprocess(examples, ds).reshape(-1, n_features)

# If we want to just visualize them, we can create a montage.
test_images = utils.montage(examples).astype(np.uint8)

# Store images so we can make a gif
gifs = []

# Now for our training:
for epoch_i in range(n_epochs):
    
    # Keep track of the cost
    this_cost = 0
    
    # Iterate over the entire dataset in batches
    for batch_X, _ in ds.train.next_batch(batch_size = batch_size):
        
        # Preprocess and reshape our current batch, batch_X:
        this_batch = preprocess(batch_X, ds).reshape(-1, n_features)
        
        # Compute the cost, and run the optimizer.
        this_cost += sess.run([cost, optimizer], feed_dict = {X: this_batch})[0]
    
    # Average cost of this epoch
    avg_cost = this_cost / ds.X.shape[0] / batch_size
    print ("Iteration: {}, Loss: {}".format(epoch_i, avg_cost))
    # print(epoch_i, avg_cost)
    
    # Let's also try to see how the network currently reconstructs the input.
    # We'll draw the reconstruction every `step` iterations.
    if epoch_i % step == 0:
        
        # Ask for the output of the network, Y, and give it our test examples
        recon = sess.run(Y, feed_dict = {X: test_examples})
                         
        # Resize the 2d to the 4d representation:
        rsz = recon.reshape(examples.shape)

        # We have to unprocess the image now, removing the normalization
        unnorm_img = deprocess(rsz, ds)
                         
        # Clip to avoid saturation
        clipped = np.clip(unnorm_img, 0, 255)

        # And we can create a montage of the reconstruction
        recon = utils.montage(clipped).astype(np.uint8)
        
        # Store for gif
        gifs.append(recon)

        # fig, axs = plt.subplots(1, 2, figsize=(10, 10))
        # axs[0].imshow(test_images)
        # axs[0].set_title('Original')
        # axs[1].imshow(recon)
        # axs[1].set_title('Synthesis')
        # fig.canvas.draw()
        # plt.show()


# In[37]:

save_path = saver.save(sess, model_dir+'/'+experiment_title+"-pokedex.data")
print("Model saved in file: %s" % save_path)

fig, axs = plt.subplots(1, 2, figsize=(10, 10))
# axs[0].imshow(test_images)
# axs[0].set_title('Original')
# axs[1].imshow(recon)
# axs[1].set_title('Synthesis')
# fig.canvas.draw()
# plt.show()
plt.imsave(arr=test_images, fname=os.path.join(output_directory, experiment_title + '-pokemon-test.png'))
plt.imsave(arr=recon, fname=os.path.join(output_directory, experiment_title +'-pokemon-recon.png'))


# In[38]:

gif.build_gif(gifs, interval=0.2, saveto=os.path.join(output_directory, experiment_title +'-reconstruction.gif'))
# ipyd.Image(url=os.path.join(output_directory, experiment_title +'-reconstruction.gif?{}').format(np.random.rand()),
#            height=500, width=500)


# In[ ]:



