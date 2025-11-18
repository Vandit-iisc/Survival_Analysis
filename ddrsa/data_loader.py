"""
Data loader for NASA Turbofan Engine Degradation Dataset (PHM08)
Implements data preprocessing and loading for DDRSA experiments
"""

import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class TurbofanDataLoader:
    """Loads and preprocesses NASA Turbofan dataset"""

    def __init__(self, data_path, lookback_window=128, pred_horizon=100):
        """
        Args:
            data_path: Path to the data directory
            lookback_window: Number of time steps to look back (K in the paper)
            pred_horizon: Maximum prediction horizon (L_max in the paper)
        """
        self.data_path = data_path
        self.lookback_window = lookback_window
        self.pred_horizon = pred_horizon

        # Column names for the dataset
        self.index_names = ['unit_id', 'time_cycles']
        self.setting_names = ['setting_1', 'setting_2', 'setting_3']
        self.sensor_names = [f'sensor_{i}' for i in range(1, 22)]  # 21 sensors
        self.col_names = self.index_names + self.setting_names + self.sensor_names

    def load_data(self, filename):
        """Load data from txt file"""
        filepath = f"{self.data_path}/{filename}"
        data = pd.read_csv(filepath, sep='\s+', header=None, names=self.col_names)
        return data

    def add_remaining_useful_life(self, df):
        """Add RUL (Remaining Useful Life) column to dataframe"""
        # Group by unit_id and get max time_cycles for each unit
        max_cycles = df.groupby('unit_id')['time_cycles'].max().reset_index()
        max_cycles.columns = ['unit_id', 'max_cycles']

        # Merge back to original dataframe
        df = df.merge(max_cycles, on='unit_id', how='left')

        # Calculate RUL
        df['RUL'] = df['max_cycles'] - df['time_cycles']
        df = df.drop('max_cycles', axis=1)

        return df

    def normalize_data(self, train_df, test_df=None, val_df=None, use_minmax=True):
        """
        Normalize sensor and setting data

        Args:
            train_df: Training dataframe
            test_df: Test dataframe (optional)
            val_df: Validation dataframe (optional)
            use_minmax: If True, use MinMaxScaler to [-1, 1] (paper requirement)
                       If False, use StandardScaler (z-score)
        """
        # Features to normalize
        feature_cols = self.setting_names + self.sensor_names

        # Choose scaler based on paper requirements
        if use_minmax:
            # MinMax scaling to [-1, 1] range (as per paper)
            scaler = MinMaxScaler(feature_range=(-1, 1))
        else:
            # Standard scaling (z-score)
            scaler = StandardScaler()

        # Fit scaler on training data
        train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])

        # Transform test data if provided
        if test_df is not None:
            test_df[feature_cols] = scaler.transform(test_df[feature_cols])

        # Transform validation data if provided
        if val_df is not None:
            val_df[feature_cols] = scaler.transform(val_df[feature_cols])

        if test_df is not None and val_df is not None:
            return train_df, val_df, test_df, scaler
        elif test_df is not None:
            return train_df, test_df, scaler
        elif val_df is not None:
            return train_df, val_df, scaler

        return train_df, scaler

    def prepare_dataset(self, train_file='train.txt', test_file='test.txt'):
        """Prepare train and test datasets"""
        # Load data
        train_df = self.load_data(train_file)
        test_df = self.load_data(test_file)

        # Add RUL
        train_df = self.add_remaining_useful_life(train_df)
        test_df = self.add_remaining_useful_life(test_df)

        # Normalize
        train_df, test_df, scaler = self.normalize_data(train_df, test_df)

        return train_df, test_df, scaler


class TurbofanDataset(Dataset):
    """PyTorch Dataset for DDRSA training"""

    def __init__(self, df, lookback_window=128, pred_horizon=100, is_train=True):
        """
        Args:
            df: Preprocessed dataframe
            lookback_window: Number of past time steps to use (K)
            pred_horizon: Maximum prediction horizon (L_max)
            is_train: Whether this is training data
        """
        self.df = df
        self.lookback_window = lookback_window
        self.pred_horizon = pred_horizon
        self.is_train = is_train

        # Feature columns (settings + sensors)
        self.feature_cols = [col for col in df.columns
                            if col.startswith('setting_') or col.startswith('sensor_')]

        # Prepare sequences
        self.sequences, self.targets, self.censoring = self._create_sequences()

    def _create_sequences(self):
        """Create sequences for DDRSA training"""
        sequences = []
        targets = []
        censoring = []

        # Group by unit_id
        grouped = self.df.groupby('unit_id')

        for unit_id, group in grouped:
            group = group.sort_values('time_cycles')
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
                current_rul = rul[i]

                # Create hazard rate labels
                hazard_labels = np.zeros(self.pred_horizon)
                censored = np.ones(self.pred_horizon)  # 1 = censored, 0 = event observed

                if current_rul < self.pred_horizon:
                    # Event occurs within prediction horizon
                    event_time = int(current_rul)
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


def get_dataloaders(data_path, batch_size=32, lookback_window=128,
                   pred_horizon=100, train_file='train.txt', test_file='test.txt',
                   val_split=0.2, num_workers=4, use_paper_split=False,
                   random_seed=42, use_minmax=True):
    """
    Create train, validation, and test dataloaders

    Args:
        data_path: Path to data directory
        batch_size: Batch size for training
        lookback_window: Number of past time steps
        pred_horizon: Maximum prediction horizon
        train_file: Training data filename
        test_file: Test data filename
        val_split: Fraction of training data to use for validation
        num_workers: Number of workers for data loading
        use_paper_split: If True, use paper's splitting methodology (70/30 split at unit level)
        random_seed: Random seed for reproducibility
        use_minmax: If True, use MinMaxScaler to [-1, 1] (paper default)

    Returns:
        train_loader, val_loader, test_loader, scaler
    """
    loader = TurbofanDataLoader(data_path, lookback_window, pred_horizon)

    if use_paper_split:
        # Paper methodology: Split at unit/sequence level
        # 70% of units for train, 30% for test
        # From train units, 30% for validation
        return get_dataloaders_paper_split(
            data_path, batch_size, lookback_window, pred_horizon,
            train_file, num_workers, random_seed, use_minmax
        )
    else:
        # Original implementation: Split at sample level
        train_df, test_df, scaler = loader.prepare_dataset(train_file, test_file)

        # Create datasets
        full_train_dataset = TurbofanDataset(train_df, lookback_window, pred_horizon, is_train=True)

        # Split train into train and validation
        train_size = int((1 - val_split) * len(full_train_dataset))
        val_size = len(full_train_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_train_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(random_seed)
        )

        test_dataset = TurbofanDataset(test_df, lookback_window, pred_horizon, is_train=False)

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


def get_dataloaders_paper_split(data_path, batch_size=32, lookback_window=128,
                                pred_horizon=100, data_file='train.txt',
                                num_workers=4, random_seed=42, use_minmax=True):
    """
    Create dataloaders using the paper's splitting methodology:
    - 70% of units (sequences) for training
    - 30% of units for testing
    - From training units, 30% for validation (i.e., 21% of total, 49% final training)

    This ensures that sequences from the same unit don't appear in both train and test,
    which would cause data leakage.

    Args:
        data_path: Path to data directory
        batch_size: Batch size for training
        lookback_window: Number of past time steps
        pred_horizon: Maximum prediction horizon
        data_file: Data filename (default uses train.txt which has all labeled data)
        num_workers: Number of workers for data loading
        random_seed: Random seed for reproducibility
        use_minmax: Use MinMaxScaler to [-1, 1] range (paper requirement)

    Returns:
        train_loader, val_loader, test_loader, scaler
    """
    np.random.seed(random_seed)

    loader = TurbofanDataLoader(data_path, lookback_window, pred_horizon)

    # Load all data
    df = loader.load_data(data_file)
    df = loader.add_remaining_useful_life(df)

    # Get all unique unit IDs
    all_units = df['unit_id'].unique()
    np.random.shuffle(all_units)

    # Split 1: 70% train, 30% test (at unit level)
    n_units = len(all_units)
    n_train_val = int(0.7 * n_units)

    train_val_units = all_units[:n_train_val]
    test_units = all_units[n_train_val:]

    # Split 2: From train_val, take 30% for validation (70% for final training)
    np.random.shuffle(train_val_units)
    n_val = int(0.3 * len(train_val_units))

    val_units = train_val_units[:n_val]
    train_units = train_val_units[n_val:]

    print(f"\nPaper Split Statistics:")
    print(f"  Total units: {n_units}")
    print(f"  Training units: {len(train_units)} ({len(train_units)/n_units*100:.1f}%)")
    print(f"  Validation units: {len(val_units)} ({len(val_units)/n_units*100:.1f}%)")
    print(f"  Test units: {len(test_units)} ({len(test_units)/n_units*100:.1f}%)")

    # Create dataframes for each split
    train_df = df[df['unit_id'].isin(train_units)].copy()
    val_df = df[df['unit_id'].isin(val_units)].copy()
    test_df = df[df['unit_id'].isin(test_units)].copy()

    # Normalize: fit on train, transform val and test
    train_df, val_df, test_df, scaler = loader.normalize_data(
        train_df, test_df, val_df, use_minmax=use_minmax
    )

    print(f"\nNormalization: {'MinMaxScaler [-1, 1]' if use_minmax else 'StandardScaler (z-score)'}")

    # Create datasets
    train_dataset = TurbofanDataset(train_df, lookback_window, pred_horizon, is_train=True)
    val_dataset = TurbofanDataset(val_df, lookback_window, pred_horizon, is_train=True)
    test_dataset = TurbofanDataset(test_df, lookback_window, pred_horizon, is_train=False)

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
