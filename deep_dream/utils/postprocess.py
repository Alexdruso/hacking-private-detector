import tensorflow as tf

def postprocess_for_evaluation(image: tf.Tensor,
                              dtype: tf.dtypes.DType) -> tf.Tensor:
    image *= 128
    image += 128

    image = tf.cast(image, dtype)

    return image