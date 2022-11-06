import argparse
import os

import tensorflow as tf
from private_detector.private_detector import PrivateDetector

from private_detector.utils.preprocess import preprocess_for_evaluation
from deep_dream.deep_dream import DeepDream


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
        tf.float32
    )

    return image


def lewdify(
        restore_path: str,
        input_path: str,
        output_path: str
) -> None:
    #images_names = os.listdir(input_path)
    # images_paths = map(lambda name: os.path.join(input_path, name), images_names)
    # for image_path in image_paths:
    #    image = read_image(image_path)

    os.makedirs(output_path, exist_ok=True)

    model = PrivateDetector(
        initial_learning_rate=1.,
        class_labels=['lewd', 'not_lewd'],
        checkpoint_dir='',
        batch_size=1,
        reg_loss_weight=1.,
        use_fp16=True,
        tensorboard_log_dir='',
        eval_threshold=2
    )

    restore_path = tf.train.latest_checkpoint(restore_path)
    model.restore(restore_path=restore_path)

    model = DeepDream(model.model)


    images_names = os.listdir(input_path)

    images_paths = map(lambda name: os.path.join(input_path, name), images_names)
    for image_path in images_paths:
        image = read_image(image_path)

        new_image, loss = model(
            image,
            step_size=tf.constant(1., dtype=tf.float32),
            steps=tf.constant(1, dtype=tf.int32)
        )

        print(new_image.shape)
        print(loss)

    return


if __name__ == '__main__':
    # TODO support lewd image generation from random noise

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--restore_path',
        type=str,
        help='Location of SavedModel to load',
        default='saved_checkpoint'
    )
    parser.add_argument(
        '--input_path',
        type=str,
        help='Paths to image paths to predict for',
        default='input'
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
