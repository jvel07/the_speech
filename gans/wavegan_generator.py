import tensorflow as tf
from IPython.display import display, Audio
import numpy as np

# Load the graph
tf.reset_default_graph()
saver = tf.train.import_meta_graph('D:/VHD/infer/infer.meta')
graph = tf.get_default_graph()
sess = tf.InteractiveSession()
saver.restore(sess, 'D:/VHD/models/model.ckpt')

# Create 50 random latent vectors z
_z = (np.random.rand(50, 100) * 2.) - 1

# Synthesize G(z)
z = graph.get_tensor_by_name('z:0')
G_z = graph.get_tensor_by_name('G_z:0')
_G_z = sess.run(G_z, {z: _z})

# Play audio in notebook
display(Audio(_G_z[0, :, 0], rate=16000))