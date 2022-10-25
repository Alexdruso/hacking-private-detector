from deep_dream.deep_dream import DeepDream
import argparse
from pathlib import Path
from typing import List

import tensorflow as tf
from private_detector.utils.preprocess import preprocess_for_evaluation


# TODO refactor to remove code duplication :)
def read_image(filename: str) -> tf.Tensor:
    """
    Load and preprocess image for inference with the Private Detector

    Parameters
    ----------
    filename : str
        Filename of image

    Returns
    -------
    image : tf.Tensor
        Image ready for inference
    """
    image = tf.io.read_file(filename)
    image = tf.io.decode_jpeg(image, channels=3)

    image = preprocess_for_evaluation(
        image,
        480,
        tf.float16
    )

    return image


def lewdify(
        restore_path: str,
        image_paths: List[str]
) -> None:
    # TODO implement and test model loading (tf.saved_model.load(model) does not restore custom gradients :))
    # for image_path in image_paths:
    #    image = read_image(image_path)
    return


if __name__ == '__main__':
    # TODO support command line with argparse
    # TODO support lewd image generation from random noise
    lewdify(
        restore_path='saved_checkpoint/ckpt-0.9375-14.index',
        image_paths=['sonny.jpg']
    )
