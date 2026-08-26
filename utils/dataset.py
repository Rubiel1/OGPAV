# -*- coding: utf-8 -*-
"""
dataset.py

Custom data loader library for OGPAV.
Allows loading raw fiber datasets dynamically from a local directory or a .zip archive.
Supports `.npy`, `.csv`, and `.txt` files.
Implements lazy-loading iterators and optimized `get_fiber_lengths()` for the GPAV pipeline.
"""

from __future__ import annotations

import os
import zipfile
import tempfile
import warnings
import shutil
import numpy as np
from typing import Iterator, Sequence, Dict, Optional, List, Union


class CustomFiberDataset(Sequence):
    """
    A sequence-like object that loads fiber datasets (R_i) from a directory or zip file lazily.
    """

    def __init__(
        self,
        source_path: str,
        node_to_file: Optional[Dict[int, str]] = None,
        delimiter: str = ",",
    ):
        """
        Parameters
        ----------
        source_path : str
            Path to a directory containing fiber files, or a .zip archive.
        node_to_file : Dict[int, str], optional
            Mapping from Q node index to the specific filename (e.g., {0: 'r0.npy', 1: 'r1.csv'}).
            If None, all valid files in the source are sorted alphabetically and mapped 0 to m-1.
        delimiter : str, optional
            Delimiter to use when parsing .csv or .txt files. Default is ",".
        """
        self.source_path = source_path
        self.node_to_file = node_to_file
        self.delimiter = delimiter
        self._lengths_cache: Dict[int, int] = {}
        
        self._temp_dir = None
        self._data_dir = source_path

        # 1. Handle Zip Extraction
        if zipfile.is_zipfile(source_path):
            self._temp_dir = tempfile.mkdtemp(prefix="ogpav_dataset_")
            with zipfile.ZipFile(source_path, 'r') as z:
                z.extractall(self._temp_dir)
            
            # Find the actual root (some zips extract into a single subfolder)
            extracted_items = os.listdir(self._temp_dir)
            if len(extracted_items) == 1 and os.path.isdir(os.path.join(self._temp_dir, extracted_items[0])):
                self._data_dir = os.path.join(self._temp_dir, extracted_items[0])
            else:
                self._data_dir = self._temp_dir
        elif not os.path.isdir(source_path):
            raise ValueError(f"source_path must be a directory or a valid .zip file. Got: {source_path}")

        # 2. Discover/Validate Files
        valid_extensions = {".npy", ".csv", ".txt"}
        available_files = [f for f in os.listdir(self._data_dir) 
                           if os.path.isfile(os.path.join(self._data_dir, f)) 
                           and os.path.splitext(f)[1].lower() in valid_extensions]

        if len(available_files) == 0:
            self._cleanup()
            raise FileNotFoundError(f"No valid data files (.npy, .csv, .txt) found in {self._data_dir}")

        # 3. Apply Mapping or Fallback to Alphabetical
        self._file_paths: List[str] = []
        
        if self.node_to_file is not None:
            # Validate mapping keys are contiguous 0...m-1
            m = len(self.node_to_file)
            expected_keys = set(range(m))
            actual_keys = set(self.node_to_file.keys())
            
            if actual_keys != expected_keys:
                self._cleanup()
                raise ValueError(f"node_to_file keys must be contiguous from 0 to {m-1}. Got: {sorted(list(actual_keys))}")
            
            for i in range(m):
                fname = self.node_to_file[i]
                fpath = os.path.join(self._data_dir, fname)
                if not os.path.exists(fpath):
                    self._cleanup()
                    raise FileNotFoundError(f"Mapped file not found: {fpath}")
                self._file_paths.append(fpath)
                
        else:
            # Fallback: Sort Alphabetically
            available_files.sort()
            warnings.warn(
                f"No `node_to_file` mapping provided. Defaulting to sorting files alphabetically. "
                f"Node 0 maps to '{available_files[0]}', ..., Node {len(available_files)-1} maps to '{available_files[-1]}'.",
                UserWarning
            )
            self._file_paths = [os.path.join(self._data_dir, f) for f in available_files]
            
        self.num_fibers = len(self._file_paths)

        warnings.warn(
            "CustomFiberDataset loaded successfully. Reminder: OperadicGPAV asserts "
            "NO order inside a fiber unless you say so. Pass `f` (a comparator per "
            "fiber, or one for all) or assume_component_wise=True; otherwise each "
            "fiber is treated as an antichain and only Q constrains the fit.",
            UserWarning
        )

    def _load_file(self, path: str, mmap_mode: Optional[str] = None) -> np.ndarray:
        ext = os.path.splitext(path)[1].lower()
        if ext == ".npy":
            return np.load(path, mmap_mode=mmap_mode)
        else:
            # For csv/txt, load fully. mmap not generally supported for text parsing.
            return np.loadtxt(path, delimiter=self.delimiter)

    def __len__(self) -> int:
        return self.num_fibers

    def __getitem__(self, i: int) -> np.ndarray:
        if i < 0 or i >= self.num_fibers:
            raise IndexError(f"Fiber index {i} out of range [0, {self.num_fibers-1}]")
            
        path = self._file_paths[i]
        data = self._load_file(path)
        self._lengths_cache[i] = len(data)
        return data

    def __iter__(self) -> Iterator[np.ndarray]:
        for i in range(self.num_fibers):
            yield self[i]

    def get_fiber_lengths(self) -> List[int]:
        """
        Calculates memory-efficient fiber lengths.
        Leverages `mmap_mode='r'` for `.npy` files to read headers without loading arrays.
        """
        lens = []
        for i in range(self.num_fibers):
            if i in self._lengths_cache:
                lens.append(self._lengths_cache[i])
            else:
                path = self._file_paths[i]
                ext = os.path.splitext(path)[1].lower()
                
                if ext == ".npy":
                    # Mmap mode only reads the header to get shape, super fast.
                    arr = self._load_file(path, mmap_mode='r')
                    length = arr.shape[0]
                    self._lengths_cache[i] = length
                    lens.append(length)
                else:
                    # For CSV/TXT, count lines efficiently
                    with open(path, 'r', encoding='utf-8') as f:
                        lines = sum(1 for line in f if line.strip())
                    self._lengths_cache[i] = lines
                    lens.append(lines)
        return lens

    def _cleanup(self):
        """Remove temporary directory if a zip was extracted."""
        if self._temp_dir and os.path.exists(self._temp_dir):
            try:
                shutil.rmtree(self._temp_dir)
            except Exception as e:
                warnings.warn(f"Failed to clean up temporary directory {self._temp_dir}: {e}")

    def __del__(self):
        self._cleanup()
