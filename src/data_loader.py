"""
Data Loading Module for Revenue Optimization System

This module provides utilities for loading and managing data files.
"""

import pandas as pd
import os
from typing import Dict, Optional


class DataLoader:
    """
    A class for loading data files from the data directory.
    
    Attributes:
        data_dir (str): Path to the data directory
        files (Dict[str, str]): Dictionary mapping file names to their paths
    """
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize the DataLoader.
        
        Args:
            data_dir: Path to the directory containing data files
        """
        self.data_dir = data_dir
        self.files = {
            "sales": "sales.csv",
            "online": "online.csv",
            "markdowns": "markdowns.csv",
            "price_history": "price_history.csv",
            "discounts_history": "discounts_history.csv",
            "actual_matrix": "actual_matrix.csv",
            "catalog": "catalog.csv",
            "stores": "stores.csv",
            "test": "test.csv",
            "sample_submission": "sample_submission.csv",
        }
    
    def get_file_path(self, name: str) -> str:
        """
        Get the full path for a file by name.
        
        Args:
            name: The name key from the files dictionary
            
        Returns:
            Full path to the file
        """
        if name not in self.files:
            raise ValueError(f"Unknown file: {name}")
        return os.path.join(self.data_dir, self.files[name])
    
    def load_csv(self, name: str, **kwargs) -> pd.DataFrame:
        """
        Load a CSV file by name.
        
        Args:
            name: The name key from the files dictionary
            **kwargs: Additional arguments to pass to pd.read_csv
            
        Returns:
            Loaded DataFrame
        """
        file_path = self.get_file_path(name)
        return pd.read_csv(file_path, **kwargs)
    
    def load_all(self) -> Dict[str, pd.DataFrame]:
        """
        Load all available data files.
        
        Returns:
            Dictionary mapping file names to DataFrames
        """
        data = {}
        for name in self.files:
            try:
                data[name] = self.load_csv(name)
            except FileNotFoundError:
                print(f"Warning: {self.files[name]} not found, skipping...")
        return data
    
    def list_available_files(self) -> list:
        """
        List all available files in the data directory.
        
        Returns:
            List of available file names
        """
        available = []
        for name, filename in self.files.items():
            if os.path.exists(os.path.join(self.data_dir, filename)):
                available.append(name)
        return available
