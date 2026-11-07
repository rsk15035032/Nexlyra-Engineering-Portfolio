import numpy as np
import torch

from torch.utils.data import Dataset


class TBMSequenceDataset(Dataset):

    def __init__(
        self,
        X,
        y,
        sequence_length=30
    ):

        self.X = np.asarray(
            X,
            dtype=np.float32
        )

        self.y = np.asarray(
            y,
            dtype=np.float32
        )

        self.sequence_length = (
            sequence_length
        )

    def __len__(self):

        return (
            len(self.X)
            - self.sequence_length
            + 1
        )

    def __getitem__(
        self,
        index
    ):

        start = index

        end = (
            index
            + self.sequence_length
        )

        X_sequence = (
            self.X[start:end]
        )

        # Target = last timestep
        target = self.y[end - 1]

        return (

            torch.tensor(
                X_sequence,
                dtype=torch.float32
            ),

            torch.tensor(
                target,
                dtype=torch.float32
            )
        )