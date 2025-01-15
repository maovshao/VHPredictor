import os
import torch
import json
import pickle
import numpy as np
from pathlib import Path
from logzero import logger

def get_index_protein_dic(protein_list):
    return {index: protein for index, protein in enumerate(protein_list)}

def get_protein_index_dic(protein_list):
    return {protein: index for index, protein in enumerate(protein_list)}

def make_parent_dir(path):
    filepath = Path(path)
    filepath.parent.mkdir(parents=True, exist_ok=True)

def tensor_to_list(tensor):
    decimals = 4
    numpy_array = tensor.cpu().numpy()
    return np.round(numpy_array, decimals=decimals).tolist()

def load_config(config_path):
    """
    Load configuration from a JSON file.

    Parameters:
    - config_path (str): Path to the JSON configuration file.

    Returns:
    - config_dict (dict): Dictionary containing configuration parameters.
    """
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    return config_dict

def update_args_from_config(args, config_dict):
    """
    Update argparse Namespace with values from config dictionary.

    Parameters:
    - args (argparse.Namespace): Parsed argparse arguments.
    - config_dict (dict): Dictionary containing configuration parameters.

    Returns:
    - args (argparse.Namespace): Updated argparse arguments.
    """
    for key, value in config_dict.items():
        if hasattr(args, key):
            setattr(args, key, value)
        else:
            logger.warning(f"Ignoring unknown config parameter: {key}")
    return args

def load_label_index(label_index_path):
    """
    Load the label index from a file in the specified directory.
    Assumes each line in the file is formatted as "host_name\tindex".

    Parameters:
    - directory (str): The directory where the label index file is located.
                       Defaults to "vhpredictor_data/vhdb/host_label_index".

    Returns:
    - label_index_path (dict): A dictionary mapping each host label to its unique index.
    """
    label_index = {}
    file_path = Path(label_index_path)
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Label index file not found at {file_path}")
    
    with open(file_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            parts = line.strip().split('\t')
            if len(parts) != 2:
                print(f"Warning: Line {line_num} in {file_path} is malformed. Skipping.")
                continue
            host, idx_str = parts
            try:
                idx = int(idx_str)
            except ValueError:
                print(f"Warning: Invalid index on line {line_num} in {file_path}. Skipping.")
                continue
            label_index[host] = idx
    
    print(f"Label index successfully loaded from {file_path}")
    return label_index

def load_ic_score(ic_score_path):
    """
    Load the Information Content (IC) scores from a file in the specified directory.
    Assumes each line in the file is formatted as "host_name\tic_score".

    Parameters:
    - directory (str): The directory where the IC scores file is located.
                       Defaults to "vhpredictor_data/vhdb/host_ic_score".

    Returns:
    - ic_scores (dict): A dictionary mapping each host label to its IC score.
    """
    ic_scores = {}
    file_path = Path(ic_score_path)
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"IC scores file not found at {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            parts = line.strip().split('\t')
            if len(parts) != 2:
                logger.warning(f"Line {line_num} in {file_path} is malformed. Skipping.")
                continue
            host, ic_str = parts
            try:
                ic = float(ic_str)
            except ValueError:
                logger.warning(f"Invalid IC score on line {line_num} in {file_path}. Skipping.")
                continue
            ic_scores[host] = ic
    
    logger.info(f"IC scores successfully loaded from {file_path}")
    return ic_scores

def get_virus_name(virusprotein_fullname):
    # Assume that virusprotein_fullnames are in the format 'virusprotein_name virus_name'
    virus_name = None
    parts = virusprotein_fullname.split(' ')
    if len(parts) > 1:
        virus_name = ' '.join(parts[1:])
    else:
        assert("bad virus_name")
    return virus_name

def load_embeddings(embedding_path):
    """
    Load embeddings from a pickle file.

    Parameters:
    - embedding_path (str): Path to the embedding file.

    Returns:
    - embeddings (dict): Dictionary mapping sample IDs to embedding vectors.
    """
    if not os.path.exists(embedding_path):
        logger.error(f"Embedding file {embedding_path} not found!")
        raise FileNotFoundError(f"Embedding file {embedding_path} not found!")
    with open(embedding_path, 'rb') as handle:
        embeddings = pickle.load(handle)
    logger.info(f'Embeddings Loaded from {embedding_path}.')
    return embeddings

def cos_similarity(z1, z2):
    """
    Compute cosine similarity between all rows of z1 and all rows of z2.

    Parameters
    ----------
    z1 : torch.Tensor, shape [N, dim]
    z2 : torch.Tensor, shape [M, dim]

    Returns
    -------
    sim_mt : torch.Tensor, shape [N, M]
        The matrix of cosine similarities, value in [-1, 1]
    """
    eps = 1e-8
    z1_n, z2_n = z1.norm(dim=1)[:, None], z2.norm(dim=1)[:, None]
    z1_norm = z1 / torch.max(z1_n, eps * torch.ones_like(z1_n))
    z2_norm = z2 / torch.max(z2_n, eps * torch.ones_like(z2_n))
    sim_mt = torch.mm(z1_norm, z2_norm.transpose(0, 1))
    return sim_mt