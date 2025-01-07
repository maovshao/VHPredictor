import numpy as np
import scipy.sparse as ssp
from logzero import logger
from sklearn.metrics import average_precision_score, precision_recall_curve

def read_predicted_probabilities(predict_file, label_index, num_labels):
    """
    Read predicted probabilities from a file.

    Parameters:
    - predict_file (str): Path to the predictions file.
    - label_index (dict): Mapping from label names to indices.
    - num_labels (int): Number of labels.

    Returns:
    - sample_ids (list): List of sample identifiers.
    - scores (np.ndarray): Array of predicted scores with shape [num_samples, num_labels].
    """
    sample_ids_set = set()
    with open(predict_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) != 3:
                continue
            sample_id = parts[0]
            sample_ids_set.add(sample_id)
    sample_ids = sorted(list(sample_ids_set))
    num_samples = len(sample_ids)
    sample_id_to_idx = {sample_id: idx for idx, sample_id in enumerate(sample_ids)}

    # Initialize scores array
    scores = np.zeros((num_samples, num_labels))

    # Second pass: fill in the scores
    with open(predict_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) != 3:
                continue
            sample_id, host_label, probability = parts
            probability = float(probability)
            sample_idx = sample_id_to_idx[sample_id]
            if host_label in label_index:
                label_idx = label_index[host_label]
                scores[sample_idx, label_idx] = probability
            else:
                logger.warning(f"Host label '{host_label}' not found in label_index.")
    return sample_ids, scores

def load_ground_truth(key_ids, key_host_dict_path, label_index):
    """
    Load virus-level ground truth labels.

    Parameters:
    - key_ids (list): List of virus identifiers.
    - key_host_dict_path (str): Path to the key-host annotation file.
    - label_index (dict): Mapping from host labels to indices.

    Returns:
    - ground_truth (numpy.ndarray): Binary matrix of shape [num_keys, num_labels].
    """
    # Initialize the binary matrix
    num_keys = len(key_ids)
    num_labels = len(label_index)
    ground_truth = np.zeros((num_keys, num_labels), dtype=int)

    # Create a mapping from key to its hosts
    key_to_hosts = {}
    with open(key_host_dict_path, 'r') as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                key = parts[0]
                hosts = parts[1:]
                key_to_hosts[key] = hosts

    # Populate the binary matrix
    for i, key in enumerate(key_ids):
        hosts = key_to_hosts.get(key, [])
        for host in hosts:
            if host in label_index:
                label_idx = label_index[host]
                ground_truth[i, label_idx] = 1
            else:
                logger.warning(f"Host '{host}' for virus '{key}' not found in label_index.")

    return ground_truth

def get_topk_sparse(scores: np.ndarray, top: int):
    """
    Convert scores to a sparse matrix keeping only top K scores per sample.

    Parameters:
    - scores (np.ndarray): Predicted scores for each label.
    - top (int): Number of top predictions to keep per sample.

    Returns:
    - ssp.csr_matrix: Sparse matrix with only top K scores per sample.
    """
    n_samples, n_labels = scores.shape
    if top >= n_labels:
        return ssp.csr_matrix(scores)
    
    # Find the indices of the top K scores for each sample
    indices = np.argpartition(-scores, top, axis=1)[:, :top]
    rows = np.repeat(np.arange(n_samples), top)
    cols = indices.flatten()
    data = scores[np.arange(n_samples)[:, None], indices].flatten()
    
    return ssp.csr_matrix((data, (rows, cols)), shape=scores.shape)

def fmax(targets: ssp.csr_matrix, scores: ssp.csr_matrix):
    """
    Compute the maximum F1 score (Fmax) over all possible thresholds.
    Note: We do not use IA weighting here, just normal F1 computation.
    We return a single value computed over the entire dataset, not per-sample.

    Parameters:
    - targets (ssp.csr_matrix): Binary ground truth labels.
    - scores (ssp.csr_matrix): Predicted scores for each label.

    Returns:
    - fmax_score (float): The best F1 score over the entire dataset.
    - best_threshold (float): The threshold corresponding to the best F1 score.
    """
    fmax_score = 0.0
    best_threshold = 0.0

    for cut in (c / 100 for c in range(101)):
        # Binarize predictions based on the threshold
        binarized = scores.copy()
        binarized.data = (binarized.data >= cut).astype(int)
        binarized.eliminate_zeros()

        correct = binarized.multiply(targets).sum()
        predicted = binarized.sum()
        actual = targets.sum()

        # Compute precision and recall for the entire dataset
        precision = correct / predicted if predicted > 0 else 0.0
        recall = correct / actual if actual > 0 else 0.0

        f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        if f1_score > fmax_score:
            fmax_score = f1_score
            best_threshold = cut

    return fmax_score, best_threshold

def smin(targets: ssp.csr_matrix, scores: ssp.csr_matrix, ic_scores: dict, label_index: list):
    """
    Compute the Smin score using IC values instead of IA.
    We use the given formula for RU and MI directly over the entire dataset.
    According to the formula:
    ru(tau) = (1/n) * sum_{i=1}^n sum_q IC(q)*1(q in T_i but not in P_i(tau))
    mi(tau) = (1/n) * sum_{i=1}^n sum_q IC(q)*1(q in P_i(tau) but not in T_i)
    
    We consider all samples together and return a single Smin value and threshold.

    Parameters:
    - targets (ssp.csr_matrix): Binary ground truth labels, shape [n_samples, n_labels].
    - scores (ssp.csr_matrix): Predicted scores for each label, shape [n_samples, n_labels].
      This is a sparse matrix.
    - ic_scores (dict): IC scores for each label (label name as key).
    - label_index (list): List of label names corresponding to label indices.

    Returns:
    - smin_value (float): The minimal semantic distance (S_min) over the entire dataset.
    - best_threshold (float): The threshold corresponding to the S_min value.
    """
    n_samples, n_labels = targets.shape

    # Verify IC availability
    missing_ic = [label for label in label_index if label not in ic_scores]
    if missing_ic:
        logger.error(f"The following labels are missing in ic_scores: {missing_ic}")
        raise ValueError(f"The following labels are missing in ic_scores: {missing_ic}")

    # Create array of IC values indexed by label indices
    ic_array = np.zeros(n_labels)
    for idx, label_name in enumerate(label_index):
        ic_array[idx] = ic_scores[label_name]

    smin_value = float('inf')
    best_threshold = 0.0

    for cut in (c / 100 for c in range(101)):
        # Binarize predictions at the current threshold
        binarized = scores.copy()
        binarized.data = (binarized.data >= cut).astype(int)
        binarized.eliminate_zeros()

        # Overlap: labels that are both predicted and true
        overlap = targets.multiply(binarized)

        # True but not predicted: (targets - overlap)
        true_not_pred = targets - overlap
        # Predicted but not true: (binarized - overlap)
        pred_not_true = binarized - overlap

        # Compute RU(tau) and MI(tau)
        # RU(tau): (1/n)*sum of IC(q) for q in T_i but not in P_i(tau)
        RU = true_not_pred.dot(ic_array).sum() / n_samples
        # MI(tau): (1/n)*sum of IC(q) for q in P_i(tau) but not in T_i
        MI = pred_not_true.dot(ic_array).sum() / n_samples

        # Compute S(tau)
        s_dist = np.sqrt(RU**2 + MI**2)

        if s_dist < smin_value:
            smin_value = s_dist
            best_threshold = cut

    return smin_value, best_threshold

def pair_aupr(targets: ssp.csr_matrix, scores: ssp.csr_matrix):
    """
    Compute the Area Under the Precision-Recall Curve (AUPR).
    Also return precision and recall values for plotting the curve.

    Parameters:
    - targets (ssp.csr_matrix): Binary ground truth labels.
    - scores (ssp.csr_matrix): Predicted scores for each label.

    Returns:
    - aupr_score (float): AUPR score.
    - precision (np.ndarray): Precision values for the curve.
    - recall (np.ndarray): Recall values for the curve.
    - thresholds (np.ndarray): Thresholds used to compute precision and recall.
    """
    y_true = targets.toarray().flatten()
    y_scores = scores.toarray().flatten()
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    aupr_score = average_precision_score(y_true, y_scores)
    return aupr_score, precision, recall, thresholds

def evaluate_metrics(targets: ssp.csr_matrix, scores: np.ndarray, ic_scores=None, label_index=None, top=None):
    """
    Compute all evaluation metrics, optionally considering only top predictions.

    Parameters:
    - targets (ssp.csr_matrix): Binary ground truth labels.
    - scores (np.ndarray): Predicted scores for each label.
    - ic_scores (dict, optional): Information Content scores for each label.
    - label_index (list, optional): List of label names corresponding to label indices.
    - top (int, optional): Number of top predictions to consider per sample. If None, consider all.

    Returns:
    - Tuple containing:
        - fmax_score (float): Fmax score.
        - best_threshold (float): Best threshold for Fmax.
        - aupr_score (float): AUPR score.
        - smin_score (float): Smin score.
        - precision (np.ndarray): Precision values for AUPR curve.
        - recall (np.ndarray): Recall values for AUPR curve.
    """
    # Instead of converting to array for Smin, we keep it as sparse (ssp.csr_matrix).
    if top is not None:
        scores_sparse = get_topk_sparse(scores.copy(), top)
    else:
        scores_sparse = ssp.csr_matrix(scores.copy())

    # Compute Fmax and AUPR using the sparse matrix
    fmax_score, best_threshold = fmax(targets, scores_sparse)
    aupr_score, precision_values, recall_values, _ = pair_aupr(targets, scores_sparse)

    # Compute Smin if IC scores and label names are provided
    if ic_scores is not None and label_index is not None:
        if len(label_index) != targets.shape[1]:
            logger.error("The length of label_index does not match the number of labels.")
            raise ValueError("The length of label_index does not match the number of labels.")
        # Check if all label_index are present in ic_scores
        missing_ic = [label for label in label_index if label not in ic_scores]
        if missing_ic:
            logger.error(f"The following labels are missing in ic_scores: {missing_ic}")
            raise ValueError(f"The following labels are missing in ic_scores: {missing_ic}")

        smin_score, best_threshold_smin = smin(targets, scores_sparse, ic_scores, label_index)
    else:
        smin_score = None
        logger.warning("IC scores or label_index not provided. Smin score will not be computed.")

    return (fmax_score, best_threshold, aupr_score, smin_score, precision_values, recall_values)