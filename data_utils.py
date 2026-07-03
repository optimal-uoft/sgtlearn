from sklearn.datasets import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

import numpy as np
import numpy as np
import pandas as pd
import os
from sklearn.datasets import fetch_openml
import tarfile
import tempfile
import json

from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
pd.options.mode.chained_assignment = None
import sys
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def process_dataframe(df):
    """
    Process a dataframe with a mix of numerical, binary, and categorical variables.

    Args:
        df (pd.DataFrame): Input dataframe.
        target_column (str): Name of the target column (if any). If provided, it will be excluded from processing.
        binary_threshold (int): Threshold for treating numerical columns as binary.

    Returns:
        pd.DataFrame: Processed dataframe with one-hot encoded categorical variables.
        dict: Dictionary mapping original features to relevant columns.
    """
    df = df.copy()

    categorical_columns = df.select_dtypes(include=['object', 'category']).columns

    numerical_columns = [col for col in df.columns if col not in categorical_columns]

    # One-hot encode categorical variables
    df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
    df_encoded_columns = list(df_encoded.columns)

    mapping = {}

    for col in numerical_columns:
        mapping[col] = [df_encoded_columns.index(col)]
    
    
    for col in categorical_columns:
        mapping[col] = [df_encoded_columns.index(c) for c in df_encoded_columns if c.startswith(f"{col}_")]


    # Return processed dataframe, target, and mapping
    return df_encoded, mapping

class DataFactory_clf:
    def __init__(self, 
                 dataset,
                 train_ratio = 0.8,
                 val_ratio = 1.0/8.0, # results in a 65/15/20 train/val/test split
                 seeds = [42, 63, 84, 56, 33],
                 cache = True,
                 cache_dir = 'data'):
        self.dataset = dataset
        self.cache = cache
        self.cache_dir = cache_dir
        fetch_func = self.fetch_function(dataset)
        if self.cache:
            # check if we have the data cached
            # get current working dir
            cache_file = f'{self.cache_dir}/{dataset}.tar.gz'
            if os.path.exists(cache_file):
                self.X, self.y, self.feature_dict = self._load_from_tar(cache_file)
            else:
                self.X, self.y, self.feature_dict = fetch_func()
                if not os.path.exists(self.cache_dir):
                    os.makedirs(self.cache_dir)
                self._save_to_tar(cache_file, self.X, self.y, self.feature_dict)
        else:
            self.X, self.y, self.feature_dict = fetch_func()
        self.le = LabelEncoder()
        self.y = self.le.fit_transform(self.y)
        self.class_names = list(self.le.classes_)
        self.n_features = self.X.shape[1]
        self.n_classes = len(self.le.classes_)
        self.n_points = self.X.shape[0]
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.random_seeds = seeds
        self.n_folds = len(self.random_seeds)
        
    def _save_to_tar(self, tar_path, X, y, feature_dict):
        """Save data to a compressed tar.gz file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save X as CSV
            X_path = os.path.join(tmpdir, 'X.csv')
            X.to_csv(X_path, index=False)

            # Save y as npy
            y_path = os.path.join(tmpdir, 'y.npy')
            np.save(y_path, y)

            # Save feature_dict as JSON
            feature_dict_path = os.path.join(tmpdir, 'feature_dict.json')
            with open(feature_dict_path, 'w') as f:
                json.dump(feature_dict, f)

            # Create tar.gz archive
            with tarfile.open(tar_path, 'w:gz') as tar:
                tar.add(X_path, arcname='X.csv')
                tar.add(y_path, arcname='y.npy')
                tar.add(feature_dict_path, arcname='feature_dict.json')

    def _load_from_tar(self, tar_path):
        """Load data from a compressed tar.gz file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Extract tar.gz archive
            with tarfile.open(tar_path, 'r:gz') as tar:
                tar.extractall(tmpdir)

            # Load X from CSV
            X_path = os.path.join(tmpdir, 'X.csv')
            X = pd.read_csv(X_path)

            # Load y from npy
            y_path = os.path.join(tmpdir, 'y.npy')
            y = np.load(y_path, allow_pickle=True)

            # Load feature_dict from JSON
            feature_dict_path = os.path.join(tmpdir, 'feature_dict.json')
            with open(feature_dict_path, 'r') as f:
                feature_dict = json.load(f)

            # Convert feature_dict keys back to original types if needed
            # JSON converts int keys to strings, so convert them back
            feature_dict_converted = {}
            for key, value in feature_dict.items():
                try:
                    key_converted = int(key)
                except (ValueError, TypeError):
                    key_converted = key
                feature_dict_converted[key_converted] = value
    
            return X, y, feature_dict_converted

    def get_data(self, fold_idx=0):
        assert fold_idx < len(self.random_seeds), f'fold_idx must be less than {len(self.random_seeds)}'
        X_train_val, X_test, y_train_val, y_test = train_test_split(self.X, self.y, train_size=self.train_ratio, random_state=self.random_seeds[fold_idx], stratify=self.y)
        if self.val_ratio > 0.0:
            X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, train_size=1.0-self.val_ratio, random_state=self.random_seeds[fold_idx], stratify=y_train_val)
        else:
            X_train = X_train_val.copy()
            X_val = X_train_val.copy()
            y_train = y_train_val.copy()
            y_val = y_train_val.copy()
        X_train = X_train.values.astype(np.float64, copy=False)
        X_val = X_val.values.astype(np.float64, copy=False)
        X_test = X_test.values.astype(np.float64, copy=False)
        # get all non-binary and non-categorical variables 
        categorical_or_binary = []
        continous = []
        for feature, idxs in self.feature_dict.items():
            if len(idxs) > 1:
                categorical_or_binary.extend(idxs)
            else:
                rel_X = X_train[:, idxs]
                if len(np.unique(rel_X)) <= 2:
                    categorical_or_binary.extend(idxs)
                else:
                    continous.extend(idxs)

        if len(continous) > 0:
            scaler = StandardScaler()
            X_train[:, continous] = scaler.fit_transform(X_train[:, continous])
            X_val[:, continous] = scaler.transform(X_val[:, continous])
            X_test[:, continous] = scaler.transform(X_test[:, continous])
            X_train[:, categorical_or_binary] = X_train[:, categorical_or_binary].astype(int)
            X_val[:, categorical_or_binary] = X_val[:, categorical_or_binary].astype(int)
            X_test[:, categorical_or_binary] = X_test[:, categorical_or_binary].astype(int)
        return X_train, y_train, X_val, y_val, X_test, y_test

    

    def fetch_function(self, dataset):
        function_dict = {
            'monks-1': self.fetch_monk1,
            'monks-2': self.fetch_monk2,
            'monks-3': self.fetch_monk3,

            # qbnb benchmark
            'room': self.fetch_room,
            'bean': self.fetch_bean,
            'eye-state': self.fetch_eye_state,
            'avila': self.fetch_avila,
            'magic': self.fetch_magic,
            'htru': self.fetch_htru,
            'occupancy': self.fetch_occupancy,
            'skin': self.fetch_skin,
            'fault': self.fetch_fault,
            'segment': self.fetch_segment,
            'page': self.fetch_page,
            'bidding': self.fetch_bidding,
            'raisin': self.fetch_raisin,
            'rice': self.fetch_rice,
            'wilt': self.fetch_wilt,
            'bank': self.fetch_bank,

            # classification datasets from ShapeCART paper            
            'adult': self.fetch_adult,
            'covtype': self.fetch_covtype,
            'electricity': self.fetch_electricity,
            'eucalyptus': self.fetch_eucalyptus,
            'eye-movements': self.fetch_eye_movements, # REALLY GOOD
            'gas-drift': self.fetch_gasdrift, 
            'higgs': self.fetch_higgs,
            'mini-boone': self.fetch_mini_boone,
            'mushroom': self.fetch_mushroom,
            'pendigits': self.fetch_pendigits,

            # # Random
            # 'heloc': self.fetch_heloc,
            # 'compas': self.fetch_compas,
            # 'spambase': self.fetch_spambase,
            # 'bank-marketing': self.fetch_bank_marketing,
            
        }
        if dataset in function_dict:
            return function_dict[dataset]
        else:
            raise ValueError(f"Dataset {dataset} not found")

    def _fetch_openml_dataset(self, data_id):
        X, y = fetch_openml(data_id=data_id, as_frame=True, return_X_y=True)
        y = y.values.ravel()
        X, mapping = process_dataframe(X)
        return X, y, mapping

    def fetch_monk1(self):
        return self._fetch_openml_dataset(data_id=333)
    def fetch_monk2(self):
        return self._fetch_openml_dataset(data_id=334)
    def fetch_monk3(self):
        return self._fetch_openml_dataset(data_id=335)

    def fetch_heloc(self):
        return self._fetch_openml_dataset(data_id=45023)
    
    def fetch_compas(self):
        X, y = fetch_openml(data_id=44162, as_frame=True, return_X_y=True)
        y = y.values.ravel()
        X = X.drop(columns=['v_type_of_assessment'])
        X, mapping = process_dataframe(X)
        return X, y, mapping
        # return self._fetch_openml_dataset(data_id=44162)
    
    def fetch_bank_marketing(self):
        return self._fetch_openml_dataset(data_id=1558)

    def fetch_spambase(self):
        return self._fetch_openml_dataset(data_id=44)

    def fetch_higgs(self):
        return self._fetch_openml_dataset(data_id=45570)
    
    def fetch_bank(self):
        return self._fetch_openml_dataset(data_id=1462)

    def fetch_eucalyptus(self):
        return self._fetch_openml_dataset(data_id=43924)

    def fetch_eye_state(self):
        return self._fetch_openml_dataset(data_id=1471)

    def fetch_mini_boone(self):
        return self._fetch_openml_dataset(data_id=41150)
    
    def fetch_electricity(self):
        return self._fetch_openml_dataset(data_id=151)
    
    def fetch_eye_movements(self):
        return self._fetch_openml_dataset(data_id=1044)

    def fetch_gasdrift(self):
        return self._fetch_openml_dataset(data_id=1476)

    def fetch_pendigits(self):
        return self._fetch_openml_dataset(data_id=32)

    def fetch_htru(self):
        return self._fetch_openml_dataset(data_id=45558)
    
    def fetch_magic(self):
        return self._fetch_openml_dataset(data_id=44125)
    def fetch_skin(self):
        return self._fetch_openml_dataset(data_id=1502)
    def fetch_fault(self):
        return self._fetch_openml_dataset(data_id=40982)
    def fetch_page(self):
        return self._fetch_openml_dataset(data_id=30)
    def fetch_segment(self):
        return self._fetch_openml_dataset(data_id=40984)
    def fetch_wilt(self):
        return self._fetch_openml_dataset(data_id=40983)
    
    def fetch_avila(self):
        X,_ = fetch_openml(data_id=42932, as_frame=True, return_X_y=True)
        y = X[['10']]
        X = X.drop(columns = ['10', 'train', 'test'])
        y = y.values.ravel()
        X, mapping = process_dataframe(X)
        return X, y, mapping
    
    def fetch_adult(self):

        X, y = fetch_openml(data_id=179, as_frame=True, return_X_y=True)
        y = y.astype(str)       
        X.drop(columns=['fnlwgt'], inplace=True)
        # X.dropna(inplace=True)
        #
        nan_mask = X.isna().any(axis=1)
        X = X[~nan_mask]
        y = y[~nan_mask]
        y = y.values.ravel()
        X, mapping = process_dataframe(X)
        return X, y, mapping

    def fetch_covtype(self):
        X,y = fetch_openml(data_id=150, as_frame=True, return_X_y=True)
        y = y.values.ravel()
        X_numeric = X.iloc[:, list(range(10))]
        X_wilderness = X.iloc[:, list(range(10, 14))]
        X_soil = X.iloc[:, list(range(14, 54))]

        X_wilderness = X_wilderness.astype(int)
        X_soil = X_soil.astype(int)


        X_numeric['Wilderness_Area'] = X_wilderness.idxmax(axis=1)
        X_numeric['Soil_Type'] = X_soil.idxmax(axis=1)
        X, mapping = process_dataframe(X_numeric)
        return X, y, mapping

    def fetch_mushroom(self):
        mushroom = fetch_ucirepo(id=73) 
        
        # data (as pandas dataframes) 
        X = mushroom.data.features 
        X = X.drop(columns=['veil-type'])
        y = mushroom.data.targets
        y = y.values.ravel()
        X, mapping = process_dataframe(X)
        return X, y, mapping

    def fetch_bidding(self):
        X, y = fetch_openml(data_id=42889, as_frame=True, return_X_y=True)
        y = y.values.ravel()
        X = X.drop(columns=['Record_ID', 'Auction_ID', 'Bidder_ID'])
        X, mapping = process_dataframe(X)
        return X, y, mapping


    def fetch_raisin(self):

        # fetch dataset
        raisin = fetch_ucirepo(id=850)

        # data (as pandas dataframes)
        X = raisin.data.features
        y = raisin.data.targets.values.ravel()
        X, mapping = process_dataframe(X)
        return X, y, mapping

    def fetch_rice(self):

        # fetch dataset
        rice = fetch_ucirepo(id=545)

        # data (as pandas dataframes)
        X = rice.data.features
        y = rice.data.targets.values.ravel()
        X, mapping = process_dataframe(X)
        return X, y, mapping

    def fetch_bean(self):
        
        # fetch dataset 
        dry_bean = fetch_ucirepo(id=602) 
        
        # data (as pandas dataframes) 
        X = dry_bean.data.features 
        # y = dry_bean.data.targets 
        
        y = dry_bean.data.targets.values.ravel()
        X, mapping = process_dataframe(X)

        return X, y, mapping

    def fetch_room(self):
        
        # fetch dataset 
        room = fetch_ucirepo(id=864) 
        
        # data (as pandas dataframes) 
        X = room.data.features 
        X = X.drop(columns=['Date', 'Time'])
        y = room.data.targets.values.ravel()
        X, mapping = process_dataframe(X)

        return X, y, mapping

    def fetch_occupancy(self):
        
        # fetch dataset 
        occupancy = fetch_ucirepo(id=357) # occupancy detection dataset
        
        # data (as pandas dataframes) 
        X = occupancy.data.features
        X = X.drop(columns=['date'])
        mask = X['Temperature'] != 'Humidity'
        X = X[mask]
        X = X.astype(float)
        y = occupancy.data.targets
        y = y[mask].values.ravel()
        X, mapping = process_dataframe(X)

        return X, y, mapping