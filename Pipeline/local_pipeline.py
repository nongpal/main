import os

from absl import logging
from tfx.orchestration import metadata, pipeline
from tfx.orchestration.beam.beam_dag_runner import BeamDagRunner

PIPELINE_NAME = "naufal_hafish-pipeline"

DATA_ROOT = "data"
TRANSFORM_MODULE_FILE = "modules/heart_failure_transform.py"
TUNER_MODULE_FILE = "modules/heart_failure_tuner.py"
TRAINER_MODULE_FILE = "modules/heart_failure_trainer.py"

OUTPUT_BASE = "output"
serving_model_dir = os.path.join(OUTPUT_BASE, "serving_model")
pipeline_root = os.path.join(OUTPUT_BASE, PIPELINE_NAME)
metadata_path = os.path.join(pipeline_root, "metadata.sqlite")

def init_local_pipeline(
    components, pipeline_root, 
) -> pipeline.Pipeline:
    
    """
    This function use to run the local pipeline
    
    Args:
        components: list of components
        pipeline_root: containing the pipeline root directory for outputs

    Returns:
       pipeline.Pipeline: the pipeline object created from the components list
    """
    
    logging.info(f"Pipeline root set to: {pipeline_root}")
    beam_args = [
        "--runner=DirectRunner",
        "--direct_running_mode=multi_processing"
        "--direct_num_workers=0",
    ]
    
    return pipeline.Pipeline(
        pipeline_name=PIPELINE_NAME,
        pipeline_root=pipeline_root,
        metadata_connection_config=metadata.sqlite_metadata_connection_config(
            metadata_path
        ),
        components=components,
        beam_pipeline_args=beam_args,
        enable_cache=True,
        enable_beam_ui=True,
    )
    
if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)

    from modules.components import init_components
    
    components = init_components(
        DATA_ROOT,
        transform_module=TRANSFORM_MODULE_FILE,
        tuner_module=TUNER_MODULE_FILE,
        training_module=TRAINER_MODULE_FILE,
        training_steps=1000,
        eval_steps=100,
        serving_model_dir=serving_model_dir,
    )
    
    _pipeline = init_local_pipeline(components, pipeline_root)
    BeamDagRunner().run(pipeline=_pipeline)