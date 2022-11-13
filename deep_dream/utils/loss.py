import tensorflow as tf

def calc_loss(img, model):
    # Pass forward the image through the model to retrieve the activations.
    # Converts the image into a batch of size 1.
    img_batch = tf.expand_dims(img, axis=0)
    prediction = model(img_batch)
    prediction = tf.nn.softmax(prediction)[..., 0]

    return tf.math.reduce_mean(prediction)
