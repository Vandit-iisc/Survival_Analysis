"""
Data loader for Azure Predictive Maintenance Dataset
Implements data preprocessing and loading for DDRSA experiments

Dataset description from paper:
This is a dataset from a guide provided by Microsoft as a case study in failure prediction.
The data comes from multiple sources including time-series of voltage, rotation, pressure
and vibration measurements collected from 100 machines in real time averaged over every hour,
error logs, machine information (type, age etc.) and 720 failure records.

We down-sample the data to take one sample every 15 hours.
"""

import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder


class AzurePMDataLoader:
    """Loads and preprocesses Azure Predictive Maintenance dataset"""

    def __init__(self, data_path, lookback_window=128, pred_horizon=100, downsample_factor=5):
        """
        Args:
            data_path: Path to the data directory (containing features.csv)
            lookback_window: Number of time steps to look back (K in the paper)
            pred_horizon: Maximum prediction horizon (L_max in the paper)
            downsample_factor: Down-sample factor (5 = every 15 hours from 3-hour data)
        """
        self.data_path = data_path
        self.lookback_window = lookback_window
        self.pred_horizon = pred_horizon
        self.downsample_factor = downsample_factor

        # Sensor/feature columns
        self.sensor_cols = [
            'voltmean', 'rotatemean', 'pressuremean', 'vibrationmean',
            'voltsd', 'rotatesd', 'pressuresd', 'vibrationsd'
        ]

        # Error count columns
        self.error_cols = [
            'error1count', 'error2count', 'error3count',
            'error4count', 'error5count'
        ]

        # Machine info columns
        self.machine_cols = ['model', 'age']

        # All feature columns (model will be one-hot encoded)
        self.feature_cols = self.sensor_cols + self.error_cols + ['age']

    def load_data(self):
        """Load features.csv which contains pre-processed data"""
        filepath = f"{self.data_path}/features.csv"

        # Load data
        df = pd.read_csv(filepath)

        # Parse datetime
        df['datetime'] = pd.to_datetime(df['datetime'])

        # Sort by machine and time
        df = df.sort_values(['machineID', 'datetime']).reset_index(drop=True)

        return df

    def preprocess_data(self, df):
        """Preprocess the Azure PM data"""

        # One-hot encode model column
        model_dummies = pd.get_dummies(df['model'], prefix='model')
        df = pd.concat([df, model_dummies], axis=1)

        # Update feature columns to include one-hot encoded model
        self.model_cols = list(model_dummies.columns)
        self.all_feature_cols = self.sensor_cols + self.error_cols + ['age'] + self.model_cols

        # Convert failure to binary
        df['failure_binary'] = (df['failure'] == True).astype(int) | (df['failure'] == 'TRUE').astype(int)

        # Down-sample to every 15 hours (take every 5th sample)
        # Group by machine and take every Nth sample
        downsampled_dfs = []
        for machine_id in df['machineID'].unique():
            machine_df = df[df['machineID'] == machine_id].copy()
            downsampled_df = machine_df.iloc[::self.downsample_factor].copy()
            downsampled_dfs.append(downsampled_df)

        df = pd.concat(downsampled_dfs, ignore_index=True)

        # Add time step index within each machine
        df['time_step'] = df.groupby('machineID').cumcount()

        # Calculate RUL (time to next failure)
        df = self._calculate_rul(df)

        return df

    def _calculate_rul(self, df):
        """
        Calculate Remaining Useful Life for each sample.
        RUL is the number of time steps until the next failure.
        """
        df = df.copy()
        df['RUL'] = np.nan

        for machine_id in df['machineID'].unique():
            machine_mask = df['machineID'] == machine_id
            machine_df = df[machine_mask].copy()

            # Find all failure time steps
            failure_times = machine_df[machine_df['failure_binary'] == 1]['time_step'].values

            if len(failure_times) == 0:
                # No failures for this machine - RUL is very large (censored)
                df.loc[machine_mask, 'RUL'] = len(machine_df)
            else:
                # Calculate RUL for each time step
                ruls = []
                for idx, row in machine_df.iterrows():
                    current_time = row['time_step']
                    # Find next failure after current time
                    future_failures = failure_times[failure_times >= current_time]

                    if len(future_failures) > 0:
                        # RUL is time to next failure
                        rul = future_failures[0] - current_time
                    else:
                        # No more failures - censored
                        rul = len(machine_df) - current_time

                    ruls.append(rul)

                df.loc[machine_mask, 'RUL'] = ruls

        return df

    def normalize_data(self, train_df, test_df=None, val_df=None, use_minmax=True):
        """
        Normalize features

        Args:
            train_df: Training dataframe
            test_df: Test dataframe (optional)
            val_df: Validation dataframe (optional)
            use_minmax: If True, use MinMaxScaler to [-1, 1] (paper requirement)
        """
        # Choose scaler
        if use_minmax:
            scaler = MinMaxScaler(feature_range=(-1, 1))
        else:
            scaler = StandardScaler()

        # Fit on training data
        train_df[self.all_feature_cols] = scaler.fit_transform(train_df[self.all_feature_cols])

        # Transform other sets
        if test_df is not None:
            test_df[self.all_feature_cols] = scaler.transform(test_df[self.all_feature_cols])

        if val_df is not None:
            val_df[self.all_feature_cols] = scaler.transform(val_df[self.all_feature_cols])

        if test_df is not None and val_df is not None:
            return train_df, val_df, test_df, scaler
        elif test_df is not None:
            return train_df, test_df, scaler
        elif val_df is not None:
            return train_df, val_df, scaler

        return train_df, scaler


class AzurePMDataset(Dataset):
    """PyTorch Dataset for Azure PM DDRSA training"""

    def __init__(self, df, feature_cols, lookback_window=128, pred_horizon=100, is_train=True):
        """
        Args:
            df: Preprocessed dataframe
            feature_cols: List of feature column names
            lookback_window: Number of past time steps to use (K)
            pred_horizon: Maximum prediction horizon (L_max)
            is_train: Whether this is training data
        """
        self.df = df
        self.feature_cols = feature_cols
        self.lookback_window = lookback_window
        self.pred_horizon = pred_horizon
        self.is_train = is_train

        # Prepare sequences
        self.sequences, self.targets, self.censoring = self._create_sequences()

    def _create_sequences(self):
        """Create sequences for DDRSA training"""
        sequences = []
        targets = []
        censoring = []

        # Group by machineID
        grouped = self.df.groupby('machineID')

        for machine_id, group in grouped:
            group = group.sort_values('time_step')
            features = group[self.feature_cols].values
            rul = group['RUL'].values

            # Create sliding window sequences
            for i in range(len(group)):
                # Get lookback window
                start_idx = max(0, i - self.lookback_window + 1)
                seq = features[start_idx:i+1]

                # Pad if necessary
                if len(seq) < self.lookback_window:
                    padding = np.zeros((self.lookback_window - len(seq), len(self.feature_cols)))
                    seq = np.vstack([padding, seq])

                # Get hazard labels for future time steps
                current_rul = int(rul[i])

                # Create hazard rate labels
                hazard_labels = np.zeros(self.pred_horizon)
                censored = np.ones(self.pred_horizon)  # 1 = censored, 0 = event observed

                if current_rul < self.pred_horizon:
                    # Event occurs within prediction horizon
                    event_time = current_rul
                    if event_time >= 0 and event_time < self.pred_horizon:
                        hazard_labels[event_time] = 1  # Event at this time step
                        censored[event_time] = 0  # Not censored
                        censored[:event_time] = 0  # All time steps before event are observed

                sequences.append(seq)
                targets.append(hazard_labels)
                censoring.append(censored)

        return np.array(sequences), np.array(targets), np.array(censoring)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = torch.FloatTensor(self.sequences[idx])
        target = torch.FloatTensor(self.targets[idx])
        censored = torch.FloatTensor(self.censoring[idx])

        return sequence, target, censored


def get_azure_pm_dataloaders(data_path, batch_size=32, lookback_window=128,
                              pred_horizon=100, num_workers=4,
                              random_seed=42, use_minmax=True, downsample_factor=5):
    """
    Create train, validation, and test dataloaders for Azure PM dataset

    Uses paper methodology:
    - 70% of machines for training
    - 30% of machines for testing
    - From training machines, 30% for validation

    Args:
        data_path: Path to data directory (containing features.csv)
        batch_size: Batch size for training
        lookback_window: Number of past time steps
        pred_horizon: Maximum prediction horizon
        num_workers: Number of workers for data loading
        random_seed: Random seed for reproducibility
        use_minmax: If True, use MinMaxScaler to [-1, 1]
        downsample_factor: Down-sample to every Nth sample (5 = 15 hours)

    Returns:
        train_loader, val_loader, test_loader, scaler
    """
    np.random.seed(random_seed)

    # Load and preprocess data
    loader = AzurePMDataLoader(data_path, lookback_window, pred_horizon, downsample_factor)
    df = loader.load_data()
    df = loader.preprocess_data(df)

    # Get all unique machine IDs
    all_machines = df['machineID'].unique()
    np.random.shuffle(all_machines)

    # Split: 70% train+val, 30% test (at machine level)
    n_machines = len(all_machines)
    n_train_val = int(0.7 * n_machines)

    train_val_machines = all_machines[:n_train_val]
    test_machines = all_machines[n_train_val:]

    # From train_val, 30% for validation
    np.random.shuffle(train_val_machines)
    n_val = int(0.3 * len(train_val_machines))

    val_machines = train_val_machines[:n_val]
    train_machines = train_val_machines[n_val:]

    print(f"\nAzure PM Dataset - Paper Split Statistics:")
    print(f"  Total machines: {n_machines}")
    print(f"  Training machines: {len(train_machines)} ({len(train_machines)/n_machines*100:.1f}%)")
    print(f"  Validation machines: {len(val_machines)} ({len(val_machines)/n_machines*100:.1f}%)")
    print(f"  Test machines: {len(test_machines)} ({len(test_machines)/n_machines*100:.1f}%)")

    # Create dataframes for each split
    train_df = df[df['machineID'].isin(train_machines)].copy()
    val_df = df[df['machineID'].isin(val_machines)].copy()
    test_df = df[df['machineID'].isin(test_machines)].copy()

    # Normalize
    train_df, val_df, test_df, scaler = loader.normalize_data(
        train_df, test_df, val_df, use_minmax=use_minmax
    )

    print(f"\nNormalization: {'MinMaxScaler [-1, 1]' if use_minmax else 'StandardScaler (z-score)'}")
    print(f"Number of features: {len(loader.all_feature_cols)}")
    print(f"Feature columns: {loader.all_feature_cols}")

    # Create datasets
    train_dataset = AzurePMDataset(train_df, loader.all_feature_cols, lookback_window, pred_horizon, is_train=True)
    val_dataset = AzurePMDataset(val_df, loader.all_feature_cols, lookback_window, pred_horizon, is_train=True)
    test_dataset = AzurePMDataset(test_df, loader.all_feature_cols, lookback_window, pred_horizon, is_train=False)

    print(f"\nDataset Sizes:")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader, scaler


if __name__ == '__main__':
    # Test the data loader
    import os

    data_path = '../AMLWorkshop/Data'

    if os.path.exists(data_path):
        print("Testing Azure PM Data Loader...")
        train_loader, val_loader, test_loader, scaler = get_azure_pm_dataloaders(
            data_path=data_path,
            batch_size=32,
            lookback_window=128,
            pred_horizon=100
        )

        # Test a batch
        batch = next(iter(train_loader))
        sequences, targets, censoring = batch
        print(f"\nBatch shapes:")
        print(f"  Sequences: {sequences.shape}")
        print(f"  Targets: {targets.shape}")
        print(f"  Censoring: {censoring.shape}")
    else:
        print(f"Data path not found: {data_path}")
        print("Please provide the correct path to Azure PM data.")
