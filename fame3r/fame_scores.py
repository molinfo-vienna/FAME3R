"""Module for computing Fame Scores."""

from statistics import mean

import numpy as np
from CDPL import Descr, Math


class FAMEScores:
    """Class for computing Fame Scores."""

    def __init__(self, train_descriptors, num_nearest_neighbors=3):
        """
        Initialize the FameScores class.

        Args:
            data (np.ndarray): 2D array of descriptors (rows: samples, columns: descriptors).
            num_nearest_neighbors (int): Number of nearest neighbors to consider.
        """
        self.train_descriptors = train_descriptors
        self.num_nearest_neighbors = num_nearest_neighbors

        # Initialize bulk similarity calculator for
        # calculating the similarity between
        # double precision floating point descriptor vectors.
        # Default similarity measure is Tanimoto.
        self.sim_calc = Descr.DVectorBulkSimilarityCalculator()

        # Add descriptors to the similarity calculator
        for i in range(self.train_descriptors.shape[0]):
            descriptor = Math.DVector(self.train_descriptors[i, :])

            # Check if the descriptor is already in the calculator
            # to avoid duplicates.
            if descriptor in self.sim_calc:
                continue

            self.sim_calc.addDescriptor(descriptor)

    def compute_fame_scores(self, data):
        """
        Compute the fame scores based on the provided data.

        Args:
            data (np.ndarray): 2D array of descriptors (rows: samples, columns: descriptors).

        Returns:
            np.ndarray: 1D array of fame scores.
        """
        fame_scores = []

        # Iterate over each descriptor in the data
        for i in range(data.shape[0]):
            query = Math.DVector(data[i, :])

            # Calculate similarities and order results
            # (enabled by sec. arg. == True) by descending value
            self.sim_calc.calculate(query, True)
            temp_fame_scores = []
            for k in range(self.num_nearest_neighbors):
                res = self.sim_calc[k]
                temp_fame_scores.append(res[1])
                # res[0] is the index of the datapoint and
                # res[1] is the similarity score
            fame_scores.append(mean(temp_fame_scores))

        # Return the fame scores as a numpy array
        return np.array(fame_scores, dtype=float)
