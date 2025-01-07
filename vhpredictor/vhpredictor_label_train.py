import os
import sys
import argparse
import tempfile
import torch
import torch.nn.functional as F
from tqdm import tqdm
import wandb  # Import W&B
from vhpredictor_util.util import load_label_index, load_ic_score, load_config, update_args_from_config, load_embeddings
from vhpredictor_util.vhpredictor_label_model import vhpredictor_label_model, load_dataset
from vhpredictor_util.evaluation import evaluate_metrics
from torch.optim.lr_scheduler import StepLR
import scipy.sparse as ssp
import json  # For serializing IC scores
from logzero import logger

def train_ss(model, train_dataloader, device, optimizer):
    """
    Train the model for one epoch.

    Parameters:
    - model (nn.Module): The neural network model.
    - train_dataloader (DataLoader): DataLoader for training data.
    - device (torch.device): Device to run the training on.
    - optimizer (torch.optim.Optimizer): Optimizer for training.

    Returns:
    - avg_loss (float): Average training loss for the epoch.
    """
    model.train()
    total_loss = 0
    total_samples = 0
    for x, _, y in train_dataloader:
        x = x.to(device)
        y = y.to(device)
        outputs, _ = model(x)
        label_loss = F.binary_cross_entropy_with_logits(outputs, y)
        total_loss_batch = label_loss

        optimizer.zero_grad()
        total_loss_batch.backward()
        optimizer.step()

        total_loss += total_loss_batch.item() * x.size(0)
        total_samples += x.size(0)
    avg_loss = total_loss / total_samples
    return avg_loss

def eval_ss(model, dataloader, device, ic_scores = None, label_index = None, top = None):
    """
    Evaluate the model on a given dataset.

    Parameters:
    - model (nn.Module): The neural network model.
    - dataloader (DataLoader): DataLoader for evaluation data.
    - device (torch.device): Device to run the evaluation on.
    - ic_scores (dict): Information Content scores for each label.
    - label_index (list): List of label names corresponding to label indices.
    - top (int, optional): Number of top predictions to consider per sample. If None, consider all.

    Returns:
    - loss (float): Average loss over the dataset.
    - fmax (float): Fmax metric over the dataset.
    - aupr (float): AUPR metric over the dataset.
    - smin (float): Smin metric over the dataset.
    """
    model.eval()
    total_loss = 0
    total_samples = 0
    all_outputs = []
    all_targets = []
    with torch.no_grad():
        for x, _, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            outputs, _ = model(x)
            loss = F.binary_cross_entropy_with_logits(outputs, y)
            total_loss += loss.item() * x.size(0)
            total_samples += x.size(0)
            all_outputs.append(outputs.cpu())
            all_targets.append(y.cpu())
    loss = total_loss / total_samples
    # Concatenate all outputs and targets
    all_outputs = torch.sigmoid(torch.cat(all_outputs)).cpu().numpy()
    all_targets = torch.cat(all_targets).cpu().numpy()
    # Convert to sparse matrix
    all_targets_sparse = ssp.csr_matrix(all_targets)
    # Compute evaluation metrics
    fmax, _, aupr_score, smin, _, _ = evaluate_metrics(
        targets=all_targets_sparse,
        scores=all_outputs,
        ic_scores=ic_scores,
        label_index = label_index,
        top=top
    )
    return loss, fmax, aupr_score, smin

def main():
    """
    Main function to train the VHPredictor model and evaluate it.
    """

    parser = argparse.ArgumentParser('Script for training VHPredictor')

    # Configuration file
    parser.add_argument('-c', '--config', type=str, help='Path to JSON config file')

    # Input paths
    # protein_central
    parser.add_argument('-ep', '--embedding_path', type=str)
    parser.add_argument('-trp', '--train_path', type=str)
    parser.add_argument('-vap', '--validation_path', type=str)
    parser.add_argument('-tep', '--test_path', type=str)
    parser.add_argument('-lip', '--label_index_path', type=str)
    parser.add_argument('-icp', '--ic_score_path', type=str)
    parser.add_argument('-smp', '--save_model_path', type=str)

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=1000, help='Minibatch size for training (default: 1000)')
    parser.add_argument('--epochs', type=int, default=3000, help='Number of epochs (default: 1000)')
    parser.add_argument('--step', type=int, default=2500, help='Epoch to change the learning rate from 1e-4 to 1e-5 (default: 1000)')
    parser.add_argument('--top', type=int, default=10, help='Number of top predictions to consider for metrics (default: 10)')

    args = parser.parse_args()

    # If a config file is provided, load and update arguments
    if args.config:
        if not os.path.exists(args.config):
            logger.error(f"Config file {args.config} does not exist.")
            sys.exit(1)
        config_dict = load_config(args.config)
        args = update_args_from_config(args, config_dict)

    logger.info(f'# Training VHPredictor: batch_size={args.batch_size}, epochs={args.epochs}, step={args.step}, top={args.top}')

    # Initialize W&B run
    wandb.init(project="vhpredictor_label_training", config={
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "step": args.step,
        "top": args.top,
    })
    config = wandb.config

    # load label index
    label_index = load_label_index(args.label_index_path)
    logger.info(f'# Total label size: {len(label_index)}')

    esm_embedding = load_embeddings(args.embedding_path)

    # Load datasets
    train_dataset = load_dataset(esm_embedding, label_index, args.train_path)
    validation_dataset = load_dataset(esm_embedding, label_index, args.validation_path)
    test_dataset = load_dataset(esm_embedding, label_index, args.test_path)

    # Get dimensions
    input_dim = train_dataset.get_dim()
    output_dim = train_dataset.get_class_num()

    # Create data loaders
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=config.batch_size, shuffle=False)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    # Set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        logger.info("Using GPU for training")
    else:
        logger.info("Using CPU for training")

    # Create model
    model = vhpredictor_label_model(input_dim, output_dim).to(device)

    # Define optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    scheduler = StepLR(optimizer, step_size=args.step, gamma=0.1)

    # Watch model gradients and parameters with W&B
    wandb.watch(model, log="all")

    best_fmax_val = 0
    best_test_metrics = {
        "fmax_test": 0.0,
        "aupr_test": 0.0,
        "smin_test": 0.0,
    }

    # Load IC scores
    ic_scores = load_ic_score(args.ic_score_path)

    # Serialize IC scores to a temporary JSON file
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as tmpf:
        json.dump(ic_scores, tmpf)
        tmpf_path = tmpf.name

    # Create a W&B artifact for IC scores
    ic_artifact = wandb.Artifact('host_ic_scores', type='ic_scores')
    ic_artifact.add_file(tmpf_path, name='host_ic_scores.json')

    # Log the artifact to W&B
    wandb.log_artifact(ic_artifact)

    # Remove the temporary file
    os.remove(tmpf_path)

    # Training loop
    for epoch in range(config.epochs):
        wandb_log = {}
        train_loss = train_ss(model, train_dataloader, device, optimizer)
        wandb_log["train_loss"] = train_loss
        # Step the scheduler at the end of each epoch
        scheduler.step()

        # Log the current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        wandb_log["learning_rate"] = current_lr

        if epoch % 100 == 0:
            logger.info(f"\n-------------------------------\nEpoch {epoch}\n-------------------------------")
            # Evaluate on validation set
            validation_loss, fmax_val, aupr_val, smin_val = eval_ss(model, validation_dataloader, device, ic_scores, label_index, top=config.top)
            logger.info(f"Validation Loss: {validation_loss:.6f}")
            logger.info(f"Validation Fmax: {fmax_val:.3f}")
            logger.info(f"Validation AUPR: {aupr_val:.3f}")
            logger.info(f"Validation Smin: {smin_val:.3f}" if smin_val is not None else "Validation Smin: N/A")
            wandb_log["validation_loss"] = validation_loss
            wandb_log["validation_Fmax"] = fmax_val
            wandb_log["validation_AUPR"] = aupr_val
            wandb_log["validation_Smin"] = smin_val if smin_val is not None else 0.0

            # Check for improvement based on Fmax
            if fmax_val > best_fmax_val:
                best_fmax_val = fmax_val
                # Evaluate on test set
                test_loss, fmax_test, aupr_test, smin_test = eval_ss(model, test_dataloader, device, ic_scores, label_index, top=config.top)
                best_test_metrics = {
                    "fmax_test": fmax_test,
                    "aupr_test": aupr_test,
                    "smin_test": smin_test if smin_test is not None else 0.0,
                }
                logger.info(f"New Best Validation Fmax: {best_fmax_val:.3f}")
                logger.info(f"Test Loss: {test_loss:.6f}")
                logger.info(f"Test Fmax: {fmax_test:.3f}")
                logger.info(f"Test AUPR: {aupr_test:.3f}")
                logger.info(f"Test Smin: {smin_test:.3f}" if smin_test is not None else "Test Smin: N/A")
                wandb_log["test_loss"] = test_loss
                wandb_log["test_Fmax"] = fmax_test
                wandb_log["test_AUPR"] = aupr_test
                wandb_log["test_Smin"] = smin_test if smin_test is not None else 0.0

                # Save the model in ONNX format
                # should be saved to different path
                torch.save(model.state_dict(), args.save_model_path)
                wandb.save(args.save_model_path)
        wandb.log(wandb_log)

    # After training completes
    logger.info(f"Final Best Test Metrics:")
    logger.info(f"Fmax: {best_test_metrics['fmax_test']:.3f}")
    logger.info(f"AUPR: {best_test_metrics['aupr_test']:.3f}")
    logger.info(f"Smin: {best_test_metrics['smin_test']:.3f}")
    wandb.finish()  # Mark the W&B run as finished
    logger.info("Training Complete!")

if __name__ == "__main__":
    main()
