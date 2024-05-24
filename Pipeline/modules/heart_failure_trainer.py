import os

import tensorflow as tf
from keras.utils.vis_utils import plot_model
import tensorflow_transform as tft

from heart_failure_transform import (
    CATEGORICAL_FEATURES,
    LABEL_KEY,
    NUMERICAL_FEATURES,
    transfomed_name,
)

from heart_failure_tuner import es, gzip_reader_fn


def get_model(hp):
    """
    Returns a compiled Keras model.
    """
    
    input_features = []
    for key, dim in CATEGORICAL_FEATURES.items():
        input_features.append(
            tf.keras.layers.Input(
                shape=(dim + 1),
                name=transfomed_name(key)
            )
        )
        
    for feature in NUMERICAL_FEATURES.items():
        input_features.append(
            tf.keras.layers.Input(
                shape=(1,),
                name=transfomed_name(feature)
            )
        )
        
        concat = tf.keras.layers.Concatenate(input_features)
        x = tf.keras.layers.Dense(hp.get("dense_units"), tf.nn.relu)(concat)
        x = tf.keras.layers.BatchNormalization()(x)
        
        for _ in range(hp.get("num_layers")):
            x = tf.keras.layers.Dense(hp.get("dense_units"), tf.nn.relu)(x)
            
        outputs = tf.keras.layers.Dense(1, tf.nn.sigmoid)(x)
        
        model = tf.keras.Model(inputs=input_features, outputs=outputs)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(hp.get("learning_rate")),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[tf.keras.metrics.BinaryAccuracy()]
        )
        
        model.summary()
        
        return model
    

    def get_serve_tf_example_fn(model, tf_transform_output):
        """Returns a function that parses a serialized tf.Example and applies TFT."""
        
        model.tft_layer = tf_transform_output.transform_features_layer()
        
        @tf.function
        def serve_tf_examples_fn(serialized_tf_examples):
            """Returns the output to be used in the serving signature."""
            feature_spec = tf_transform_output.raw_feature_spec()
            feature_spec.pop(LABEL_KEY)
            parsed_features = tf.io.parse_example(
                serialized_tf_examples, 
                feature_spec
                )
            transformed_features = model.tft_layer(parsed_features)
            outputs = model(transformed_features)
            return {"outputs": outputs}
        
        return serve_tf_examples_fn
    
    def input_fn(file_pattern, tf_transform_output, batch_size=32):
        """
        Generates features and labels for tuning and training

        Args:
            file_pattern: input tfrecord file pattern
            tf_transform_output: A TFTransformOutput
            batch_size: representing the number of consecutive elements of returned dataset to combine
            
        Returns:
            A dataset that contains (features, indices) tuple where features is a dictionary of
            Tensors, and indices is a single Tensor of label indices.
        """
        transformed_features_specs = tf_transform_output.transformed_features_specs().copiy()
        
        dataset = tf.data.experimental.make_batched_features_dataset(
            file_pattern=file_pattern,
            batch_size=batch_size,
            features=transformed_features_specs,
            label_key=transfomed_name(LABEL_KEY),
            reader=gzip_reader_fn,
        )
        
        return dataset
    
    
    def run_fn(fn_args):
        """
        This function use to train the model.

        Args:
            fn_args: use to train the model
        """
        
        tf_transform_output = tft.TFTransformOutput(fn_args.tranform_output)
        
        train_dataset = input_fn(fn_args.train_files, tf_transform_output)
        eval_dataset = input_fn(fn_args.eval_files, tf_transform_output)
        
        hp = fn_args.hyperparameters["values"]
        
        model = get_model(hp)
        
        log_dir = os.path.join(os.path.dirname(fn_args.serving_model_dir), "logs")
        tb = tf.keras.callbacks.TensorBoard(
            log_dir=log_dir,
            update_freq="batch"
        )
        mdcp = tf.keras.callbacks.ModelCheckpoint(
                    fn_args.serving_model_dir,
                    monitor="val_binary_accuracy",
                    mode="max",
                    save_best_only=True,
                )

        model.fit(
            train_dataset,
            steps_per_epoch=fn_args.train_steps,
            validation_data=eval_dataset,
            validation_steps=fn_args.eval_steps,
            callbacks=[es, tb, mdcp],
            epochs=10
        )
        
        signatures = {
            "serving_default": get_serve_tf_example_fn(
                model, tf_transform_output
                ).get_concrete_function(
                    tf.TensorSpec(
                        shape=[None], 
                        dtype=tf.string,
                        name="examples"
                    )
                )
        }
        
        model.save(
            fn_args.serving_model_dir, save_format="tf", signature=signatures
        )
        
        plot_model(
            model,
            to_file=os.path.join(
                os.path.dirname(fn_args.serving_model_dir),
                "model.png"
            )
        )