#!/usr/bin/env python
"""
Download from W&B the raw dataset and apply some basic data cleaning, exporting the result to a new artifact
"""
import argparse
import logging
import wandb

import pandas as pd
import os


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    logger.info("Download arifact from artifact store")
    artifact_local_path = run.use_artifact(args.input_artifact).file()

    df = pd.read_csv(artifact_local_path)

    # Drop outliers from dataset
    ## Price
    logger.info("Drop outliers from dataset")
    idx_price = df["price"].between(args.min_price, args.max_price)
    df = df[idx_price].copy()

    ## longitude
    idx = df['longitude'].between(-74.25, -73.50) & df['latitude'].between(40.5, 41.2)
    df = df[idx].copy()

    # Save cleaned data
    logger.info("Cleaned data saving")
    saved_filename = "clean_sample.csv"
    df.to_csv(saved_filename, index = False)

    artifact = wandb.Artifact(
        args.output_artifact,
        type = args.output_type,
        description = args.output_description
    )
    artifact.add_file(saved_filename)
    run.log_artifact(artifact)
    artifact.wait()

    logger.info("Cleaned data uploaded into Weight and Bias")
    os.remove(saved_filename)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A very basic data cleaning")

    parser.add_argument(
        "--input_artifact", 
        type=str, 
        help="Name of input artifact",
        required=True
        )

    parser.add_argument(
        "--output_artifact", 
        type=str,
        help="Name of output artifact",
        required=True
    )

    parser.add_argument(
        "--output_type", 
        type=str,
        help="Data type for output artifact",
        required=True
    )

    parser.add_argument(
        "--output_description", 
        type=str,
        help="Description for output artifact",
        required=True
    )

    parser.add_argument(
        "--min_price", 
        type=float,
        help="Minimum price to consider",
        required=True
    )

    parser.add_argument(
        "--max_price", 
        type=float,
        help="Maximum price to consider",
        required=True
    )

    args = parser.parse_args()

    go(args)
