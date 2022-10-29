import argparse
import os

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
        input_path: str,
        output_path: str
) -> None:
    # TODO implement and test model loading (tf.saved_model.load(model) does not restore custom gradients :))
    #images_names = os.listdir(input_path)
    # images_paths = map(lambda name: os.path.join(input_path, name), images_names)
    # for image_path in image_paths:
    #    image = read_image(image_path)

    os.makedirs(output_path, exist_ok=True)
    return


if __name__ == '__main__':
    # TODO support lewd image generation from random noise

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--restore_path',
        type=str,
        help='Location of SavedModel to load',
        default='saved_checkpoint/ckpt-0.9375-14.index'
    )
    parser.add_argument(
        '--input_path',
        type=str,
        help='Paths to image paths to predict for',
        default='sonny.jpg'
    )
    parser.add_argument(
        '--output_path',
        type=str,
        help='Paths to image paths to predict for',
        default='output'
    )
    args = parser.parse_args()

    lewdify(
        restore_path=args.restore_path,
        input_path=args.input_path,
        output_path=args.output_path
    )
