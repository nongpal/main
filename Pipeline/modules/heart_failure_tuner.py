from typing import NamedTuple, Dict, Any, Text
from heart_failure_transform import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    LABEL_KEY,
    transfomed_name,
)

import keras_tuner as kt
from keras_tuner import GridSearch
from keras_tuner.engine import base_tuner
import tensorflow as tf
import tensorflow_transform as tft

TunerFnResult = NamedTuple(
    "TunerFnResult",
    [
        ("tuner", base_tuner.BaseTuner),
        ("best_hyperparameters", Dict[Text, Any]),
    ],
)

es = tf.keras.callbacks.EarlyStopping(
    monitor="val_binary_accuracy",
    mode="max",
    patience=4,
    restore_best_weights=True,
)

def gzip_reader_fn(filenames):
    """
    Small utility returning a record reader that can read gzip'ed files.
    
    Args:
        filenames: directory or list of file names to read from.
    
    Returns:
        TFRecord: Compressed data.
    """
    return tf.data.TFRecordDataset(filenames, compression_type="GZIP")

def input_fn(file_pattern, tf_transform_outout, batch_size=32):
    """
    Generates a file reader that can read input files from a given file pattern and transform them into output files. 
    The transform function returns a tuple containing the input file name and the output file name.

    Args:
        file_pattern: input file pattern
        tf_transform_outout: a TFTransformOutput
        batch_size (int, optional): Defaults to 32.
        
    Returns:
        A tf.data.Dataset of (input_file_name, transformed_features).
    """
    
    transform_feature_spec = tf_transform_outout.transformed_features_spec().copy()
    
    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transform_feature_spec,
        reader=gzip_reader_fn,
        label_key=transfomed_name(LABEL_KEY),
    )
    
    return dataset

def get_model(hp):
    """
    Builds a Keras model for classification with keras tuner.

    Args:
        hp (tuner.HyperParameters): Hyperparameter object.
    
    Returns:
        Model object
    """
    
    num_layers = hp.Int("num_layers", min_value=1, max_value=5, step=1)

    dense_units = hp.Int("dense_units", min_value=32, max_value=256, step=32)

    learning_rate = hp.Float("learning_rate", min_value=0.0001, max_value=0.01, log=True)

    input_features = []

    for key, dim in CATEGORICAL_FEATURES.items():
        input_features.append(
            tf.keras.layers.Input(
                shape=(dim + 1),
                name=transfomed_name(key),
                dtype="int64"
            )
        )
        
    for feature in NUMERICAL_FEATURES:
        input_features.append(
            tf.keras.layers.Input(
                shape=(1,),
                name=transfomed_name(feature),
                dtype="float32"
            )
        )
        
    concat = tf.keras.layers.Concatenate(input_features)

    x = tf.keras.layers.Dense(
        dense_units, tf.nn.relu, name="dense"
    )(concat)
    
    x = tf.keras.layers.BatchNormalization()(x)

    for _ in range(num_layers - 1):
        x = tf.keras.layers.Dense(
            dense_units, tf.nn.relu, name="dense"
        )(x)
        
    outputs = tf.keras.layers.Dense(
        1, tf.nn.sigmoid, name="output"
    )(x)
    
    model = tf.keras.Model(inputs=input_features, outputs=outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.BinaryAccuracy()],
    )
    
    model.summary()
    
    return model

def tuner_fn(fn_args):
    """
    Tuner function for keras tuner.
    
    Args:
        fn_args (TunerFnArgs): Tuner function arguments.
    
    Returns:
        TunerFnResult
    """
    
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)
    
    train_dataset = input_fn(fn_args.train_files, tf_transform_output)
    eval_dataset = input_fn(fn_args.eval_files, tf_transform_output)
    
    tuner = GridSearch(
        hypermodel=get_model,
        objective=kt.Objective("val_binary_accuracy", "max"),
        seed=42,
        max_trials=10,
        project_name="heart_failure",
        directory=fn_args.working_dir,
        overwrite=True,
    )
    
    return TunerFnResult(
        tuner=tuner,
        fit_kwargs={
            "x": train_dataset,
            "validation_data": eval_dataset,
            "callbacks": [es],
            "epochs": 10,
            "steps_per_epoch": fn_args.train_steps,
            "validation_steps": fn_args.eval_steps,
        }
    )