"""Export the trained model for TF Serving, with JPEG decoding and resizing inside the model."""
import tensorflow as tf


def save_for_serving(model, path, img_size):
    """Save a SavedModel whose default signature takes raw JPEG bytes.

    Decode and bilinear resize match image_dataset_from_directory, so serving matches training.
    """
    @tf.function(input_signature=[tf.TensorSpec([None], tf.string)])
    def serve_jpeg(image_bytes):
        def decode(b):
            img = tf.io.decode_image(b, channels=3, expand_animations=False)
            return tf.image.resize(img, (img_size, img_size))
        images = tf.map_fn(decode, image_bytes,
                           fn_output_signature=tf.TensorSpec([img_size, img_size, 3], tf.float32))
        return {"probability": model(images, training=False)}

    model.save(path, signatures={"serving_default": serve_jpeg})
