"""VGG16 transfer learning, cats-vs-dogs binary classification — minimal reproducible baseline.

SageMaker conventions: data is read from SM_CHANNEL_*, the model is saved to SM_MODEL_DIR.
"""
import argparse
import os

import mlflow
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input

from export import save_for_serving

IMG_SIZE = 224
CLASS_NAMES = ["cats", "dogs"]   # cats=0, dogs=1


def load(data_dir, batch_size, shuffle):
    """Read {data_dir}/{cats,dogs}/*.jpg and resize to 224x224."""
    return tf.keras.utils.image_dataset_from_directory(
        data_dir,
        label_mode="binary",
        class_names=CLASS_NAMES,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=batch_size,
        shuffle=shuffle,
        seed=42,                     # fixed shuffle order, independent of global RNG state
    )


def main():
    # Fully reproducible experiments
    tf.keras.utils.set_random_seed(42)              # Python / NumPy / TensorFlow seeds
    tf.config.experimental.enable_op_determinism()  # force deterministic GPU ops

    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--learning-rate", type=float, default=0.001)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--hidden-units", type=int, default=10)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--model-dir", default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    p.add_argument("--model_dir", dest="sagemaker_model_dir", help=argparse.SUPPRESS)   # injected by tuning jobs; unused, own dest so it cannot overwrite --model-dir
    p.add_argument("--train", default=os.environ.get("SM_CHANNEL_TRAIN"))
    p.add_argument("--validation", default=os.environ.get("SM_CHANNEL_VALIDATION"))
    args = p.parse_args()

    train_ds = load(args.train, args.batch_size, shuffle=True)
    val_ds = load(args.validation, args.batch_size, shuffle=False)

    # End-to-end model: raw pixels -> VGG16 preprocessing -> VGG16 (frozen) -> flatten -> standardise -> hidden layer -> probability
    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = layers.Lambda(preprocess_input)(inputs)
    base = VGG16(include_top=False, weights="imagenet", input_tensor=x)
    base.trainable = False                                  # freeze; train only the classifier head below
    x = layers.Flatten()(base.output)

    # Feature-wise standardisation (like StandardScaler), statistics from the training set only
    norm = layers.Normalization()
    features = Model(inputs, x).predict(load(args.train, args.batch_size, shuffle=False), verbose=0)
    norm.adapt(features)
    x = norm(x)

    x = layers.Dense(args.hidden_units, activation="relu")(x)   # hidden layer
    x = layers.Dropout(args.dropout)(x)                         # 0.0 = no-op
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = Model(inputs, outputs)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
                  loss="binary_crossentropy", metrics=["accuracy"])
    model.summary()

    # Log each epoch's metrics to MLflow under consistent train_/val_ names (step = epoch number, from 1)
    log_epoch = tf.keras.callbacks.LambdaCallback(
        on_epoch_end=lambda epoch, logs: mlflow.log_metrics({
            "train_loss": logs["loss"],
            "train_accuracy": logs["accuracy"],
            "val_loss": logs["val_loss"],
            "val_accuracy": logs["val_accuracy"],
        }, step=epoch + 1)
    )

    # Stop when val_loss stops improving and roll back to the best epoch's weights
    early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=args.patience,
                                                  restore_best_weights=True, verbose=1)

    with mlflow.start_run(run_name=os.environ.get("TRAINING_JOB_NAME")):   # name the run after the SageMaker job
        mlflow.log_params({"epochs": args.epochs, "batch_size": args.batch_size,
                           "learning_rate": args.learning_rate, "patience": args.patience,
                           "hidden_units": args.hidden_units, "dropout": args.dropout})
        history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, verbose=2,
                            callbacks=[log_epoch, early_stop])

        # The saved model is the best epoch, not the last one: log its scores explicitly
        best = early_stop.best_epoch                       # 0-based index of the best epoch
        mlflow.log_metrics({
            "best_epoch": best + 1,                        # 1-based, consistent with the metric steps
            "best_val_loss": early_stop.best,
            "best_val_accuracy": history.history["val_accuracy"][best],
        })
        print(f"best_val_loss={early_stop.best:.6f}")              # parsed from the log by the SageMaker tuner

    # Save as a TF SavedModel (under version dir "1", for TF Serving); its entry point takes JPEG bytes
    save_for_serving(model, os.path.join(args.model_dir, "1"), IMG_SIZE)
    print("Model saved.")


if __name__ == "__main__":
    main()
