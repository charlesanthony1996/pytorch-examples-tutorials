import os

os.environ["keras_backend"] = "tensorflow"

import tensorflow as tf

tf.random.set_seed(42)

import keras
from keras import layers

import os
import glob
import imageio.v2 as imageio
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


# initialize global variables
auto = tf.data.AUTOTUNE
batch_size = 5
num_samples = 32
pos_encode_dims = 16
epochs = 20


print(auto)

# download and load the data
url = ("http://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20/nerf/tiny_nerf_data.npz")

data = keras.utils.get_file(origin=url)
data = np.load(data)
images = data["images"]
im_shape = images.shape
(num_images, h, w, _) = images.shape
(poses, focal) = (data["poses"], data["focal"])

plt.imshow(images[np.random.randint(low=0, high=num_images)])
# plt.show()

def encode_position(x):
    positions = [x]
    for i in range(pos_encode_dims):
        for fn in [tf.sin, tf.cos]:
            positions.append(fn(2.0 ** 1 * x))
    return tf.concat(positions, axis= 1)

    

def get_rays(height, width, focal, pose):
    i, j = tf.meshgrid(
        tf.range(width, dtype=tf.float32),
        tf.range(height, dtype=tf.float32),
        indexing="xy"
    )

    # normalize the x axis coordinates
    transformed_i = (i - width * 0.5) / focal

    # normalize the y axis coordinates
    transformed_j = (j - height * 0.5) / focal

    # create the direction unit vectors
    directions = tf.stack([transformed_i, -transformed_j, -tf.ones_like(i)], axis=-1)

    # get the camera matrix
    camera_matrix = pose[:3, :3]
    height_width_focal = pose[:3, -1]

    # get origins and directions for the rays
    transformed_dirs = directions[..., None, :]
    camera_dirs = transformed_dirs * camera_matrix
    ray_directions = tf.reduce_sum(camera_dirs, axis= -1)
    ray_origins = tf.broadcast_to(height_width_focal, tf.shape(ray_directions))

    # return the origins and directions
    return (ray_origins, ray_directions)


def render_flat_rays(ray_origins, ray_directions, near, far, num_samples, rand=False):
    t_vals = tf.linspace(near, far, num_samples)
    if rand:
        shape = list(ray_origins.shape[:-1]) + [num_samples]
        noise = tf.random.uniform(shape=shape) * (far - near) / num_samples
        t_vals = t_vals + noise


    # equation: r(t) = o + td -> building the "r" here
    rays = ray_origins[..., None, :] + (ray_origins[..., None, :] * t_vals[..., None])

    rays_flat = tf.reshape(rays, [-1, 3])
    rays_flat = encode_position(rays_flat)
    return (rays_flat, t_vals)


def map_fn(pose):
    (ray_origins, ray_directions) = get_rays(height=h, width=w, focal=focal, pose=pose)
    (rays_flat, t_vals) = render_flat_rays(
        ray_origins= ray_origins,
        ray_directions=ray_directions,
        near = 2.0,
        far = 6.0,
        num_samples = num_samples,
        rand=True
    )
    return (rays_flat, t_vals)


# create the training split
split_index = int(num_images * 0.8)

# split the images into training and validation
train_images = images[:split_index]
val_images = images[split_index:]

# split the poses into training and validation
train_poses = poses[:split_index]
val_poses = poses[split_index:]

# make the training pipeline
train_img_ds = tf.data.Dataset.from_tensor_slices(train_images)
train_pose_ds = tf.data.Dataset.from_tensor_slices(train_poses)
train_ray_ds = train_pose_ds.map(map_fn, num_parallel_calls=auto)
training_ds = tf.data.Dataset.zip((train_img_ds, train_ray_ds))
train_ds = (
    training_ds.shuffle(batch_size)
    .batch(batch_size, drop_remainder=True, num_parallel_calls=auto)
    .prefetch(auto)
)



print("train img ds: ", len(train_img_ds))
print("valid img ds: ", len(train_pose_ds))


# make the validation pipeline
val_img_ds = tf.data.Dataset.from_tensor_slices(val_images)
val_pose_ds = tf.data.Dataset.from_tensor_slices(val_poses)
val_ray_ds = val_pose_ds.map(map_fn, num_parallel_calls=auto)
validation_ds = tf.data.Dataset.zip((val_img_ds, val_ray_ds))
val_ds = (
    validation_ds.shuffle(batch_size)
    .batch(batch_size, drop_remainder=True, num_parallel_calls=auto)
    .prefetch(auto)
)
print(len(val_img_ds))
print(len(val_pose_ds))

def get_nerf_model(num_layers, num_pos):
    inputs = keras.Input(shape=(num_pos, 2 * 3 * pos_encode_dims + 3))
    x = inputs
    for i in range(num_layers):
        x = layers.Dense(units=64, activation="relu")(x)
        if i % 4 == 0 and i > 0:
            # inject residual connection
            x = layers.concatenate([x, inputs], axis= -1)
    outputs = layers.Dense(units=4)(x)

    return keras.Model(inputs=inputs, outputs=outputs)


def render_rgb_depth(model, rays_flat, t_vals, rand=True, train=True):
    if train:
        predictions = model(rays_flat)

    else:
        predictions = model.predict(rays_flat)

    predictions = tf.reshape(predictions, shape=(batch_size, h, w, num_samples, 4))

    # slice the predictions into rgb and sigma
    rgb = tf.sigmoid(predictions[..., :-1])
    sigma_a = tf.nn.relu(predictions[..., -1])

    # get the distance of adjacent intervals
    delta = t_vals[..., -1:] - t_vals[..., -1]
    # delta shape = (num_samples)

    if rand:
        delta = tf.concat([delta, tf.broadcast_to([1e10], shape=(batch_size, h, 1, 1))], axis= 1)
        alpha = 1.0 - tf.exp(-sigma_a * delta)

    else:
        delta = tf.concat([delta, tf.broadcast_to([1e10], shape=(batch_size, h, w, 1))], axis= -1)
        alpha = 1.0 - tf.exp(-sigma_a * delta[:, None, None, :])


    # get transmittance
    exp_term = 1.0 - alpha
    epsilon = 1e-10
    transmittance = tf.math.cumprod(exp_term + epsilon, axis= -1,exclusive=True)
    weights = alpha * transmittance
    rgb = tf.reduce_sum(weights[..., None] * rgb, axis=-2)

    if rand:
        depth_map = tf.reduce_sum(weights * t_vals, axis = -1)

    else:
        depth_map = tf.reduce_sum(weights * t_vals, axis = -1)

    return (rgb, depth_map)

    

