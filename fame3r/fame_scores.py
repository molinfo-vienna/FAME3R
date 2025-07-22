"""Module for computing Fame Scores."""

import glob
import os

import numpy as np
import pandas as pd


class FAMEScores:
    """Class for computing Fame Scores."""

    def __init__(self, model_folder, num_nearest_neighbors=3):
        """
        Initialize the FameScores class.

        Args:
            model_folder (str): Path to the model folder where the reference descriptors are stored.
            num_nearest_neighbors (int): Number of nearest neighbors to consider.
        """
        csv_files = glob.glob(os.path.join(model_folder, "*descriptors.csv"))
        if len(csv_files) == 1:
            reference_descriptors_df = pd.read_csv(csv_files[0])
        else:
            raise FileNotFoundError(
                f"Expected one CSV file ending with 'descriptors.csv', but found {len(csv_files)}."
            )
        # Drop all columns from the reference descriptors dataframe that don't contain "AtomType"
        # This removes the molecule and atom identifiers, true labels, and physicochemical and topological
        # descriptors, leaving only the circular fingerprints.
        atomtype_columns = [col for col in reference_descriptors_df.columns if "AtomType" in col]
        ref_fps = reference_descriptors_df[atomtype_columns].values

        self.ref_fps = ref_fps
        self.num_nearest_neighbors = num_nearest_neighbors

    def compute_fame_scores(self, fps):
        """
        Compute the fame scores based on the provided fingerprints.
        The fame score is the mean of the top 3 nearest neighbor similarities to the reference set.

        Args:
            fps (np.ndarray): 2D array of circular fingerprints (rows: sample index, columns: circular fingerprint index).

        Returns:
            np.ndarray: 1D array of fame scores.
        """

        ab = np.matmul(self.ref_fps, fps.T)
        a = np.sum(self.ref_fps * self.ref_fps, axis=1)
        b = np.sum(fps * fps, axis=1)
        div = a[:,None] + b[None,:] - ab

        pairwise_tanimoto_similarities = np.divide(
            ab, 
            div, 
            out=np.zeros_like(ab), 
            where=div != 0
        )

        fame_scores = np.mean(np.sort(pairwise_tanimoto_similarities, axis=0)[-3:], axis=0)

        return fame_scores
