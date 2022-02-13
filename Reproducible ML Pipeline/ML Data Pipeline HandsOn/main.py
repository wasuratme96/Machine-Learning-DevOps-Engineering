import mlflow
import os
import hydra
from omegaconf import DictConfig, OmegaConf


# This automatically reads in the configuration
# from config.yml 
@hydra.main(config_name='config')
def go(config: DictConfig):

    # Setup the wandb experiment. All runs will be grouped under this name
    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    # You can get the path at the root of the MLflow project with this:
    root_path = hydra.utils.get_original_cwd()

    # Check which steps we need to execute
    if isinstance(config["main"]["execute_steps"], str):
        # This was passed on the command line as a comma-separated list of steps
        steps_to_execute = config["main"]["execute_steps"].split(",")
    else:
        steps_to_execute = list(config["main"]["execute_steps"])

    # Download step
    if "download" in steps_to_execute:
        _ = mlflow.run(
            os.path.join(root_path, "download"),
            "main",
            parameters={
                "file_url": config["data"]["file_url"],
                "artifact_name": "raw_data.parquet",
                "artifact_type": "raw_data",
                "artifact_description": "Data as downloaded"
            },
        )
    # Preprocess step
    if "preprocess" in steps_to_execute:
        _ = mlflow.run(
            os.path.join(root_path, "preprocess"),
            "main",
            parameters = {
                "input_artifact": "raw_data.parquet:latest",
                "artifact_name" : "preprocessed_data.csv",
                "artifact_type" : "preprocessed_data",
                "artifact_description" : "Data with preprocessing applied"
            }
        )

    # Check data step
    if "checkdata" in steps_to_execute:
        _ = mlflow.run(
            os.path.join(root_path, "checkdata"),
            "main",
            parameters = {
                "reference_artifact" : config["data"]["reference_dataset"],
                "sample_artifact" : "preprocessed_data.csv:latest",
                "ks_alpha" : config["data"]["ks_alpha"]
            }
        )
    # Segregate step
    if "segregate" in steps_to_execute:
        _ = mlflow.run(
            os.path.join(root_path, "segregate"),
            "main",
            parameters = {
                "input_artifact" : "preprocessed_data.csv:latest",
                "artifact_root" : "data",
                "artifact_type" : "sergregated_data",
                "test_size" : config["data"]["test_size"],
                "stratify" : config["data"]["stratify"]
            }
        )

if __name__ == "__main__":
    go()