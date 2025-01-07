import os
import sys
import pickle
import torch
import torch.nn.functional as F
import scipy.sparse as ssp
import argparse
from logzero import logger
from vhpredictor_util.util import load_label_index, load_ic_score, make_parent_dir, get_virus_name, load_embeddings, load_config, update_args_from_config
from vhpredictor_util.vhpredictor_label_model import load_dataset, vhpredictor_label_model
from vhpredictor_util.evaluation import evaluate_metrics, load_ground_truth
import numpy as np

model_type_list = ["Protein-Central", "Virus-Central"]

def load_torch_model(model_path, input_dim, output_dim, device):
    """
    Load PyTorch model from saved state dict.

    Parameters:
    - model_path (str): Path to the model file.
    - input_dim (int): Input dimension for the model.
    - output_dim (int): Output dimension for the model.
    - device (torch.device): Device to load the model onto.

    Returns:
    - model (nn.Module): Loaded PyTorch model.
    """
    model = vhpredictor_label_model(input_dim, output_dim)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    logger.info('PyTorch model loaded successfully.')
    return model

def eval_torch_model(model, dataloader, device, ic_scores, label_index, top=None):
    """
    Evaluate the PyTorch model and compute evaluation metrics.

    Parameters:
    - model (nn.Module): The PyTorch model to evaluate.
    - dataloader (DataLoader): DataLoader for the evaluation dataset.
    - device (torch.device): Device to perform computations on.
    - ic_scores (dict): Information content scores for labels.
    - label_index (dict): Mapping from labels to indices.
    - top (int, optional): Number of top predictions to consider.

    Returns:
    - fmax (float): Fmax score.
    - best_threshold (float): Best threshold found during evaluation.
    - aupr_score (float): Area Under the Precision-Recall Curve.
    - smin (float or None): Smin score, if applicable.
    - virusprotein_outputs (numpy.ndarray): Model output scores for all samples.
    """
    virusprotein_outputs = []
    virusprotein_targets = []
    with torch.no_grad():
        for data in dataloader:
            x, _, y = data
            outputs, _ = model(x.to(device))
            outputs = torch.sigmoid(outputs).cpu().numpy()
            virusprotein_outputs.append(outputs)
            virusprotein_targets.append(y.cpu().numpy())

    virusprotein_outputs = np.vstack(virusprotein_outputs)
    virusprotein_targets = np.vstack(virusprotein_targets)
    virusprotein_targets_sparse = ssp.csr_matrix(virusprotein_targets)
    fmax, best_threshold, aupr_score, smin, _, _ = evaluate_metrics(
        targets=virusprotein_targets_sparse,
        scores=virusprotein_outputs,
        ic_scores=ic_scores,
        label_index=label_index,
        top=top
    )
    return fmax, best_threshold, aupr_score, smin, virusprotein_outputs

def aggregate_predictions(sample_ids, virusprotein_outputs, aggregation_method='average'):
    """
    Aggregate protein-level predictions to virus-level predictions.

    Parameters:
    - sample_ids (list): List of protein sample identifiers.
    - virusprotein_outputs (numpy.ndarray): Protein-level prediction scores.
    - aggregation_method (str): Method to aggregate predictions ('average', 'max', 'product', 'majority').

    Returns:
    - virus_ids (list): List of virus identifiers.
    - virus_outputs (numpy.ndarray): Virus-level aggregated prediction scores.
    """
    # Assume that sample_ids are in the format 'virusprotein_id virus_name'
    virus_to_proteins = {}
    for idx, sample_id in enumerate(sample_ids):
        virus_name = get_virus_name(sample_id)
        if virus_name not in virus_to_proteins:
            virus_to_proteins[virus_name] = []
        virus_to_proteins[virus_name].append(idx)

    virus_ids = []
    virus_outputs = []
    for virus_id, indices in virus_to_proteins.items():
        protein_outputs = virusprotein_outputs[indices]  # Shape: [num_proteins, num_labels]
        if aggregation_method == 'average':
            aggregated_output = protein_outputs.mean(axis=0)
        elif aggregation_method == 'max':
            aggregated_output = protein_outputs.max(axis=0)
        elif aggregation_method == 'product':
            aggregated_output = 1 - np.prod(1 - protein_outputs, axis=0)
        elif aggregation_method == 'majority':
            aggregated_output = (protein_outputs >= 0.1).mean(axis=0)
            aggregated_output = (aggregated_output > 0.1).astype(float)
        else:
            raise ValueError("Unsupported aggregation method")
        virus_ids.append(virus_id)
        virus_outputs.append(aggregated_output)
    virus_outputs = np.array(virus_outputs)
    return virus_ids, virus_outputs

def vhpredictor_label_test(model_path, model_type, embedding_path, label_index_path, virusprotein_test_path=None, ic_score_path=None, virus_test_path=None, top=10, threshold=0.1, output_path=None):
    fmax = aupr_score = smin = virus_fmax = virus_aupr = virus_smin = virusprotein_embeddings = None
    logger.info(f"# Starting prediction for VHPredictor model in {model_path}")
    if model_type not in model_type_list:
        logger.error(f"Wrong Model Type!!!")
    else:
        logger.info(f"Model Type = {model_type}")

    # Load embeddings
    esm_embedding = load_embeddings(embedding_path)

    # Load label index mapping
    label_index = load_label_index(label_index_path)
    logger.info(f'# Total label size: {len(label_index)}')

    # Load dataset
    dataset = None
    if (model_type == "Protein-Central") and virusprotein_test_path:
        dataset = load_dataset(esm_embedding, label_index, virusprotein_test_path)
    elif (model_type == "Virus-Central") and virus_test_path:
        dataset = load_dataset(esm_embedding, label_index, virus_test_path)
    else:
        dataset = load_dataset(esm_embedding, label_index, None)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1000, shuffle=False)

    # Set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        logger.info("Using GPU for prediction")
    else:
        logger.info("Using CPU for prediction")

    # Get input and output dimensions
    input_dim = dataset.get_dim()
    output_dim = dataset.get_class_num()

    # Load PyTorch model
    model = load_torch_model(model_path, input_dim, output_dim, device)

    # Initialize lists to collect outputs
    virusprotein_outputs = []
    virusprotein_ids = []
    virus_ids = []
    virus_outputs = np.array([])
    virusprotein_embeddings = {}

    # Perform prediction and collect embeddings
    with torch.no_grad():
        for data in dataloader:
            x, sample_id, _ = data

            logits, embeddings = model(x.to(device))
            outputs = torch.sigmoid(logits).cpu().numpy()
            virusprotein_outputs.append(outputs)
            virusprotein_ids.extend(sample_id)
            # Collect embeddings
            for sid, emb in zip(sample_id, embeddings):
                virusprotein_embeddings[sid] = emb.to(device).clone().cpu()  # Store tensor directly

    if virusprotein_outputs:
        virusprotein_outputs = np.vstack(virusprotein_outputs)
    else:
        virusprotein_outputs = np.array([])

    if (model_type == "Protein-Central"):
        virus_ids, virus_outputs = aggregate_predictions(virusprotein_ids, virusprotein_outputs, aggregation_method='average')
    elif (model_type == "Virus-Central"):
        virus_ids = virusprotein_ids
        virus_outputs = virusprotein_outputs
    else:
        logger.error(f"Wrong Model Type!!!")

    # If labels are provided, evaluate the model
    if ic_score_path:
        ic_scores = load_ic_score(ic_score_path)
        if virusprotein_test_path:
            # Generate IC scores based on train, validation, and test label files
            
            virusprotein_targets = load_ground_truth(virusprotein_ids, virusprotein_test_path, label_index)
            virusprotein_targets_sparse = ssp.csr_matrix(virusprotein_targets)

            # Compute evaluation metrics
            fmax, best_threshold, aupr_score, smin, _, _ = evaluate_metrics(
                targets=virusprotein_targets_sparse,
                scores=virusprotein_outputs,
                ic_scores=ic_scores,
                label_index=label_index,
                top=top
            )
            # logger.info("# Evaluation Metrics for Test Set (Protein-level):")
            # logger.info(f"Protein Fmax: {fmax:.3f}")
            # logger.info(f"Protein Best_threshold: {best_threshold:.3f}")
            # logger.info(f"Protein AUPR: {aupr_score:.3f}")
            # logger.info(f"Protein Smin: {smin:.3f}" if smin is not None else "Protein Smin: N/A")

        if virus_test_path:
            # If virus-level labels are provided, evaluate at virus level
            # Load virus-level ground truth labels
            virus_targets = load_ground_truth(virus_ids, virus_test_path, label_index)

            # Convert to sparse matrix
            virus_targets_sparse = ssp.csr_matrix(virus_targets)

            # Compute evaluation metrics
            virus_fmax, virus_best_threshold, virus_aupr, virus_smin, _, _ = evaluate_metrics(
                targets=virus_targets_sparse,
                scores=virus_outputs,
                ic_scores=ic_scores,
                label_index=label_index,
                top=top
            )
            logger.info("# Evaluation Metrics for Test Set (Virus-level):")
            logger.info(f"Virus Fmax: {virus_fmax:.3f}")
            logger.info(f"Virus Best_threshold: {virus_best_threshold:.3f}")
            logger.info(f"Virus AUPR: {virus_aupr:.3f}")
            logger.info(f"Virus Smin: {virus_smin:.3f}" if virus_smin is not None else "Virus Smin: N/A")

    if output_path:
        # Extract model name from model_path
        model_name = os.path.basename(model_path.rstrip('/'))

        # Create a reverse mapping from index to label
        index_to_label = {idx: label for label, idx in label_index.items()}

        if virusprotein_test_path:
            # Prepare output file path for protein-level predicted probabilities
            output_file_protein_prob = os.path.join(output_path, model_name + "_virusprotein_probability")
            output_file_protein_label = os.path.join(output_path, model_name + "_virusprotein_label")
            make_parent_dir(output_file_protein_prob)

            with open(output_file_protein_prob, 'w') as f_out_prob:
                for sample_id, preds in zip(virusprotein_ids, virusprotein_outputs):
                    # Get top-n indices
                    if top:
                        top_n_indices = preds.argsort()[-top:][::-1]
                    else:
                        top_n_indices = preds.argsort()[::-1]
                    top_n_labels = [index_to_label[idx] for idx in top_n_indices]
                    top_n_scores = preds[top_n_indices]
                    # Write each label and score on a separate line
                    for label, score in zip(top_n_labels, top_n_scores):
                        f_out_prob.write(f"{sample_id}\t{label}\t{score:.4f}\n")

            logger.info(f"Protein-level predicted probabilities saved to {output_file_protein_prob}")

            # Apply threshold to get predicted labels
            predicted_labels_bool = virusprotein_outputs >= threshold  # Shape: [num_samples, num_labels]

            # Write protein-level predicted labels to the output file
            with open(output_file_protein_label, 'w') as f_out_label:
                for sample_id, preds in zip(virusprotein_ids, predicted_labels_bool):
                    labels = [index_to_label[idx] for idx, val in enumerate(preds) if val]
                    labels_str = "\t".join(labels)
                    f_out_label.write(f"{sample_id}\t{labels_str}\n")

            logger.info(f"Protein-level predicted labels saved to {output_file_protein_label}") 

        # Prepare output file path for virus-level predictions
        output_file_virus = os.path.join(output_path, model_name + "_virus_label")
        output_file_virus_prob = os.path.join(output_path, model_name + "_virus_probability")
        make_parent_dir(output_file_virus)
        # Apply threshold to get predicted labels
        predicted_labels_bool_virus = virus_outputs >= threshold  # Shape: [num_viruses, num_labels]

        # Write virus-level predicted labels to the output file
        with open(output_file_virus, 'w') as f_out:
            for virus_id, preds in zip(virus_ids, predicted_labels_bool_virus):
                labels = [index_to_label[idx] for idx, val in enumerate(preds) if val]
                labels_str = "\t".join(labels)
                f_out.write(f"{virus_id}\t{labels_str}\n")

        logger.info(f"Virus-level predicted labels saved to {output_file_virus}")

        # Write virus-level probabilities
        with open(output_file_virus_prob, 'w') as f_out_virus_prob:
            for virus_id, preds in zip(virus_ids, virus_outputs):
                # Get top-n indices
                if top:
                    top_n_indices_virus = preds.argsort()[-top:][::-1]
                else:
                    top_n_indices_virus = preds.argsort()[::-1]
                top_n_labels_virus = [index_to_label[idx] for idx in top_n_indices_virus]
                top_n_scores_virus = preds[top_n_indices_virus]
                # Write each label and score on a separate line
                for label, score in zip(top_n_labels_virus, top_n_scores_virus):
                    f_out_virus_prob.write(f"{virus_id}\t{label}\t{score:.4f}\n")

        logger.info(f"Virus-level predicted probabilities saved to {output_file_virus_prob}")

        # Save virusprotein embeddings
        embedding_file = os.path.join(output_path, model_name + "_virusprotein_label_embedding.pkl")
        with open(embedding_file, 'wb') as f_emb:
            pickle.dump(virusprotein_embeddings, f_emb, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"Virusprotein MLP embeddings saved to {embedding_file}")

    # Return metrics
    return fmax, aupr_score, smin, virus_fmax, virus_aupr, virus_smin


if __name__ == "__main__":
    """
    Main function to test the VHPredictor model and evaluate it.
    """

    parser = argparse.ArgumentParser('Script for testing VHPredictor model')

    # Configuration file
    parser.add_argument('-c', '--config', type=str, help='Path to JSON config file')

    # Input paths
    parser.add_argument('-m', '--model_path', type=str)
    parser.add_argument('-mt', '--model_type', type=str, default='Virus-Central')
    parser.add_argument('-ep', '--embedding_path', type=str)
    parser.add_argument('-lip', '--label_index_path', type=str)
    parser.add_argument('-vptp', '--virusprotein_test_path', type=str)
    parser.add_argument('-icp', '--ic_score_path', type=str)
    parser.add_argument('-vtp', '--virus_test_path', type=str)

    # Test parameters
    parser.add_argument('--top', type=int, default=10, help='Number of top predictions to consider (default: 10)')
    parser.add_argument('--threshold', type=float, default=0.1, help='Threshold of predictions to output as label (default: 0.1)')
    parser.add_argument('-olp', '--output_path', type=str, default='vhpredictor_data/experiment/test_result/')

    args = parser.parse_args()
    #make_parent_dir(args.output_path)

    # If a config file is provided, load and update arguments
    if args.config:
        if not os.path.exists(args.config):
            logger.error(f"Config file {args.config} does not exist.")
            sys.exit(1)
        config_dict = load_config(args.config)
        args = update_args_from_config(args, config_dict)

    logger.info(f'# Testing VHPredictor: model_path={args.model_path}, model_type={args.model_type}, output_path={args.output_path}, top={args.top}')

    vhpredictor_label_test(
        model_path=args.model_path,
        model_type=args.model_type,
        embedding_path=args.embedding_path,
        label_index_path=args.label_index_path,
        virusprotein_test_path=args.virusprotein_test_path,
        ic_score_path=args.ic_score_path,
        virus_test_path=args.virus_test_path,
        top=args.top,
        threshold=args.threshold,
        output_path=args.output_path
    )