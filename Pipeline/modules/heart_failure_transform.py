import tensorflow as tf
import tensorflow_transform as tft

CATEGORICAL_FEATURES = {
    "anemia": 2,
    "diabetes": 2,
    "high_blood_pressure": 2,
    "smoking": 2,
    "sex": 2,
}

NUMERICAL_FEATURES = [
    "age", "creatinine_phosphokinase", "ejection_fraction", "platelets", "serum_creatinine", "serum_sodium", "time"
]

LABEL_KEY = "DEATH_EVENT"

def transfomed_name(key):
    """
    Renaming the label to transformed feature name

    Args:
        str: key 
    
    Returns:
        str: transformed feature name
    """
    
    return key + "_xf"

def convert_num_to_one_hot(label_tensor, num_labels=2):
    """
    Convert numerical label to one-hot label

    Args:
        label_tensor (int): label tensor (0 or 1)
        
    Returns:
        label tensor
    """
    one_hot_tensor = tf.one_hot(label_tensor, num_labels)
    return tf.reshape(one_hot_tensor, [-1, num_labels])

def preprocessing_fn(inputs):
    """
    Preprocessing function for transform module to preprocess input data into transformed features

    Args:
        inputs: map from feature keys to raw features
        
    Returns:
        outputs: map from feature keys to transformed feature
    """
    
    outputs = {}
    
    for key in CATEGORICAL_FEATURES:
        dim = CATEGORICAL_FEATURES[key]
        int_value = tft.compute_and_apply_vocabulary(
            inputs[key],
            top_k=dim + 1,
        )
        outputs[transfomed_name(key)] = convert_num_to_one_hot(
            int_value,
            num_labels=dim + 1,
        )
        
    for feature in NUMERICAL_FEATURES:
        outputs[transfomed_name(feature)] = tft.scale_to_0_1(inputs[feature])
        
    outputs[transfomed_name(LABEL_KEY)] = tf.cast(inputs[LABEL_KEY], tf.int64)
    
    return outputs