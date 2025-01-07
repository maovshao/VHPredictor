import os
from tqdm import tqdm
import statistics
import numpy as np
import pandas as pd
import matplotlib.patches as mpatches
import seaborn as sns
import scipy.sparse as ssp
from collections import Counter
from logzero import logger
import matplotlib.pyplot as plt
from collections import defaultdict
from vhpredictor_util.evaluation import evaluate_metrics, load_ground_truth, read_predicted_probabilities
from vhpredictor_util.util import load_label_index, load_ic_score, get_virus_name, load_embeddings, make_parent_dir
from sklearn import manifold
import math
import matplotlib.colors as mcolors
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import silhouette_score

def multi_label_analyze(dict_file, plot = False):
    plt.rcParams.update(plt.rcParamsDefault)
    host_dict = {}  # List to store number of host per virus
    virus_dict = {}  # Dictionary to store number of viruses per host

    # Read the virusprotein_host_pairs file
    with open(dict_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')

            # Get the host (starting from the second column)
            virus = parts[0]
            host = parts[1:]

            for single_host in host:
                host_dict.setdefault(virus,set())
                host_dict[virus].add(single_host)
                virus_dict.setdefault(single_host,set())
                virus_dict[single_host].add(virus)

    # Print results
    print(f"Total number of unique host: {len(virus_dict)}")

    if plot:
        virus_num = len(host_dict)
        host_num_dic = {}
        for virus in host_dict:
            host_num_dic[virus] = len(host_dict[virus])
        host_num_dic = dict(sorted(host_num_dic.items(), key=lambda item: item[1], reverse=True))
        host_num_array = np.asarray(list(host_num_dic.values()))
        max_host_num = np.max(host_num_array)

        # make data:
        x = 0.5 + np.arange(virus_num)

        # plot
        fig, ax = plt.subplots()
        ax.bar(x, host_num_array, width=1, linewidth=0)
        ax.set_xlabel('Virus list', fontsize=18)
        ax.set_ylabel('Number of host', fontsize=18)
        plt.yscale('log')
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        ax.set(xlim=(0, virus_num), ylim=(0.9, 10**(int(np.log10(max_host_num))+1)), yticks=10**np.arange(0, (int(np.log10(max_host_num))+1)))
        ax.set_xticklabels(['{:,}'.format(int(label)) for label in ax.get_xticks()])

        plt.show()
        plt.close()

        host_num = len(virus_dict)
        virus_num_dic = {}
        for host in virus_dict:
            virus_num_dic[host] = len(virus_dict[host])
        virus_num_dic = dict(sorted(virus_num_dic.items(), key=lambda item: item[1], reverse=True))
        virus_num_array = np.asarray(list(virus_num_dic.values()))
        max_virus_num = np.max(virus_num_array)

        # make data:
        x = 0.5 + np.arange(host_num)

        # plot
        fig, ax = plt.subplots()
        ax.bar(x, virus_num_array, width=1, linewidth=0)
        ax.set_xlabel('host list', fontsize=18)
        ax.set_ylabel('Number of annotated virus', fontsize=18)
        plt.yscale('log')
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        ax.set(xlim=(0, host_num), ylim=(0.9, 10**(int(np.log10(max_virus_num))+1)), yticks=10**np.arange(0, (int(np.log10(max_virus_num))+1)))
        ax.set_xticklabels(['{:,}'.format(int(label)) for label in ax.get_xticks()])

        plt.show()
        plt.close()
    return virus_dict

def fmax_aupr_smin_plot_simple(df_dict, color_dict, legend):
    """
    Plot a bar chart where 'Fmax' and 'AUPR' share the left y-axis,
    and 'Smin' is on a separate right y-axis. Both plots share the same x-axis
    with metric categories: ['Fmax', 'AUPR', 'Smin'].

    Parameters
    ----------
    df_dict : pd.DataFrame
        A DataFrame containing columns 'metric', 'score', 'method'.
    color_dict : dict
        Dictionary mapping method names to their corresponding plot colors.
    legend : bool
        Whether to show the legend or not.

    Notes
    -----
    - Make sure the 'metric' column contains exactly 'Fmax', 'AUPR', and 'Smin'.
    - The code uses Seaborn barplot twice on the same figure:
      one for (Fmax, AUPR) on ax1, one for (Smin) on ax2.
    - The order of bars on the x-axis is controlled by `order=['Fmax','AUPR','Smin']`.
    """

    # Split the data by metric so we can plot them on different axes
    df_fmax_aupr = df_dict[df_dict['metric'].isin(['Fmax', 'AUPR'])]
    df_smin = df_dict[df_dict['metric'] == 'Smin']

    # Create a figure and two y-axes
    fig, ax1 = plt.subplots(figsize=(8, 6))
    ax2 = ax1.twinx()

    # Define the order of the metrics on the x-axis
    metrics_order = ['Fmax', 'AUPR', 'Smin']

    # Clear ax1 to avoid duplicate plots if re-running in certain environments
    ax1.clear()

    # Plot Fmax and AUPR on the left axis (ax1)
    sns.barplot(
        data=df_fmax_aupr,
        x="metric",
        y="score",
        hue="method",
        palette=list(color_dict.values()),
        ax=ax1,
        order=metrics_order  # ensures x-axis shows Fmax, AUPR, Smin in this order
    )

    # Plot Smin on the right axis (ax2)
    # Note that we still use the same metrics_order to keep x-axis alignment,
    # but df_smin only has "Smin" as actual data.
    sns.barplot(
        data=df_smin,
        x="metric",
        y="score",
        hue="method",
        palette=list(color_dict.values()),
        ax=ax2,
        order=metrics_order
    )

    # Remove the second legend from ax2 to avoid duplication
    if ax2.get_legend():
        ax2.get_legend().remove()

    # Handle legend display
    if legend:
        # Move the legend of ax1 to the right side
        sns.move_legend(ax1, "upper left", bbox_to_anchor=(1.1, 1),  fontsize=14)
        legend_obj = ax1.get_legend()
        # Remove the legend title if it exists
        if legend_obj is not None:
            legend_obj.set_title(None)
    else:
        ax1.get_legend().remove()

    # Labeling and style
    ax1.set_xlabel('')
    ax1.set_ylabel('Fmax and AUPR', fontsize=18)  # Left y-axis
    ax2.set_ylabel('Smin', fontsize=18)           # Right y-axis

    ax1.set_ylim(0, 1.0)

    ax1.tick_params(axis='x', labelsize=18)
    ax1.tick_params(axis='y', labelsize=18)
    ax2.tick_params(axis='y', labelsize=18)

    plt.xticks(fontsize=18)
    # We only set the y-ticks for ax1 explicitly; ax2 will adapt accordingly.
    ax1.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    # Finally, show the figure
    plt.show()
    plt.close()

# Compute metrics
def compute_metrics(
    label_index_path,
    ic_score_path,
    predict_file,
    ground_truth_file,
):
    """
    Compute evaluation metrics and generate plots.

    Parameters:
    - label_index_path (str): Path to the label index file.
    - ic_score_path (str): Path to the IC scores file.
    - predict_file (str): Path to the predictions file.
    - ground_truth_file (str): Path to the ground truth file.

    Returns:
    - Tuple containing Fmax, AUPR, and Smin scores.
    """
    # Load label_index
    label_index = load_label_index(label_index_path)
    max_idx = max(label_index.values())
    num_labels = max_idx + 1
    label_index_list = [''] * num_labels
    for label, idx in label_index.items():
        label_index_list[idx] = label

    # Load ic_scores
    ic_scores = load_ic_score(ic_score_path)

    # Read predicted probabilities for virus proteins
    sample_ids, scores = read_predicted_probabilities(
        predict_file, label_index, num_labels
    )

    # Load protein-level ground truth targets using load_ground_truth
    targets = load_ground_truth(
        key_ids=sample_ids,
        key_host_dict_path=ground_truth_file,
        label_index=label_index
    )

    # Convert protein_targets to ssp.csr_matrix
    targets_sparse = ssp.csr_matrix(targets)

    # Compute evaluation metrics for protein-level predictions
    (fmax_score, best_threshold, aupr_score, smin_score, 
     precision_values, recall_values) = evaluate_metrics(
        targets=targets_sparse,
        scores=scores,
        ic_scores=ic_scores,
        label_index=label_index_list,
        top=None
    )

    logger.info(f"Fmax: {fmax_score:.3f}")
    logger.info(f"AUPR: {aupr_score:.3f}")
    if smin_score is not None:
        logger.info(f"Smin: {smin_score:.3f}")
    else:
        logger.info("Smin: N/A")
    
    # Return the metrics and plotting data
    return (fmax_score, aupr_score, smin_score, precision_values, recall_values)

def compute_metrics_for_best(
    label_index_path: str,
    ic_score_path: str,
    virus_host_dict_train: str,
    virus_host_dict_test: str
):
    """
    Compute the 'Best' performance reference based on hosts seen in the training set.
    Returns Fmax, AUPR, and Smin for the 'Best' scenario, 
    along with the precision/recall values for plotting.

    Parameters:
    - label_index_path (str): Path to the label index file.
    - ic_score_path (str): Path to the IC scores file.
    - virus_host_dict_train (str): Path to the training set virus-host file.
    - virus_host_dict_test (str): Path to the testing set virus-host file.

    Returns:
    - best_fmax (float)
    - best_aupr (float)
    - best_smin (float)
    - precision_values (np.ndarray)
    - recall_values (np.ndarray)
    """
    # 1) Load label_index
    label_index = load_label_index(label_index_path)
    max_idx = max(label_index.values())
    num_labels = max_idx + 1
    label_index_list = [''] * num_labels
    for label, idx in label_index.items():
        label_index_list[idx] = label

    # 2) Load ic_scores
    ic_scores = load_ic_score(ic_score_path)

    # 3) Read all hosts from the training set
    hosts_train = set()
    with open(virus_host_dict_train, 'r') as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                for h in parts[1:]:
                    hosts_train.add(h)

    # 4) Parse the test set: read virus IDs and their hosts
    key_ids_test = []
    key_to_hosts_test = {}
    with open(virus_host_dict_test, 'r') as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                key = parts[0]
                key_ids_test.append(key)
                hosts_test_line = parts[1:]
                key_to_hosts_test[key] = hosts_test_line

    # 5) Count host stats
    hosts_test = set()
    for _, test_host_list in key_to_hosts_test.items():
        for h in test_host_list:
            hosts_test.add(h)
    intersection_hosts = hosts_train.intersection(hosts_test)
    logger.info(f"Number of hosts in TRAIN: {len(hosts_train)}")
    logger.info(f"Number of hosts in TEST: {len(hosts_test)}")
    logger.info(f"Number of hosts in intersection: {len(intersection_hosts)}")

    # 6) Build ground truth
    ground_truth_test = load_ground_truth(key_ids_test, virus_host_dict_test, label_index)
    ground_truth_test_sparse = ssp.csr_matrix(ground_truth_test)

    # 7) Construct the "best_predict" matrix: 
    #    For each test sample, set true hosts seen in TRAIN to 1 - 1e-5, else 0
    best_predict = np.zeros((len(key_ids_test), num_labels), dtype=float)
    for i, virus in enumerate(key_ids_test):
        true_hosts = key_to_hosts_test.get(virus, [])
        for host in true_hosts:
            if host in label_index:
                host_idx = label_index[host]
                if host in hosts_train:
                    best_predict[i, host_idx] = 1 - 1e-5  # Predict correctly if seen in TRAIN
                else:
                    best_predict[i, host_idx] = 0.0  # Do not predict if not seen in TRAIN
            else:
                logger.warning(f"host '{host}' for virus '{virus}' not found in label_index.")
    # Convert to sparse
    best_predict_sparse = ssp.csr_matrix(best_predict)

    # 8) Use evaluate_metrics to compute metrics
    #    Pass the sparse matrix 'best_predict_sparse' instead of the dense array
    (best_fmax, best_threshold,
     best_aupr, best_smin,
     best_precision, best_recall) = evaluate_metrics(
        targets=ground_truth_test_sparse,
        scores=best_predict_sparse,
        ic_scores=ic_scores,
        label_index=label_index_list,
        top=None
    )

    # 9) Additional evaluation for Correct, Incorrect, and No answer
    correct_count = 0
    no_answer_count = 0
    total_count = len(key_ids_test)

    for virus, test_host_list in key_to_hosts_test.items():
        # If this virus has no hosts at all, count it as 'no answer'
        if len(test_host_list) == 0:
            continue
        # If ALL test hosts are in the train set => correct
        if any(host in hosts_train for host in test_host_list):
            correct_count += 1
        else:
            # As soon as we find one test host not in the train set => incorrect
            no_answer_count += 1

    correct_ratio = correct_count / total_count
    no_answer_ratio = no_answer_count / total_count
    incorrect_ratio = 1.0 - correct_ratio - no_answer_ratio

    return (
        best_fmax,
        best_aupr,
        best_smin,
        best_precision,
        best_recall,
        correct_ratio,
        incorrect_ratio,
        no_answer_ratio
    )

def fmax_aupr_smin_plot(fmax_aupr_smin_plot_dict):
    """
    Modified plotting function.
    For Fmax and Smin:
    - Plot them on the same figure with two x-axis ticks: ['Fmax', 'Smin'].
    - Fmax is shown on the left y-axis, Smin on the right y-axis.
    - Different methods are represented by different colors, and their names shown in the legend.

    For AUPR:
    - Plot the Precision-Recall curves in a separate figure.
    - Different methods are represented by different colors, and their names shown in the legend.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Extract all methods and their values
    methods = list(fmax_aupr_smin_plot_dict.keys())
    # Prepare arrays for plotting
    fmax_values = []
    smin_values = []
    precision_curves = {}
    recall_curves = {}
    aupr_values = {}

    for method_name, results in fmax_aupr_smin_plot_dict.items():
        # results expected to be (fmax_score, aupr_score, smin_score, precision_values, recall_values)
        fmax_score, aupr_score, smin_score, precision_values, recall_values = results
        
        fmax_values.append(fmax_score)
        smin_values.append(smin_score)
        precision_curves[method_name] = precision_values
        recall_curves[method_name] = recall_values
        aupr_values[method_name] = aupr_score

    # X positions for the metrics
    x_positions = np.array([0, 1])  # 0 for Fmax, 1 for Smin
    width = 0.8 / len(methods)  # width of each bar
    # We'll offset bars for different methods
    offsets = np.linspace(-0.4+width*0.5, 0.4-width*0.5, len(methods))

    # --- Plot Fmax and Smin in one figure ---
    fig, ax1 = plt.subplots(figsize=(8, 6))
    ax2 = ax1.twinx()

    # Set x-axis ticks
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels(['Fmax', 'Smin'], fontsize=18)

    # Colors for methods
    cmap = plt.get_cmap('tab10')
    colors = [cmap(i) for i in range(len(methods))]

    # Plot Fmax (on ax1) and Smin (on ax2)
    bars_fmax = []
    bars_smin = []
    for i, method_name in enumerate(methods):
        # Fmax bar at x=0 with offset
        bar_f = ax1.bar(x_positions[0] + offsets[i], fmax_values[i], width=width, color=colors[i], label=method_name if i == 0 else "")
        # Smin bar at x=1 with offset
        bar_s = ax2.bar(x_positions[1] + offsets[i], smin_values[i], width=width, color=colors[i], label=method_name if i == 0 else "")
        bars_fmax.append(bar_f)
        bars_smin.append(bar_s)

    ax1.set_ylabel('Fmax', fontsize=18)
    ax2.set_ylabel('Smin', fontsize=18)
    ax1.tick_params(axis='y', labelsize=18)
    ax2.tick_params(axis='y', labelsize=18)

    # Create a combined legend for methods
    # We only need one legend entry per method, so let's use bars_fmax for legend entries
    ax1.legend([b[0] for b in bars_fmax], methods, bbox_to_anchor=(1.1, 1), loc='upper left', fontsize=14)

    plt.title('Fmax and Smin Comparison', fontsize=18)
    plt.show()
    plt.close()

    # --- Plot AUPR curves in another figure ---
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, method_name in enumerate(methods):
        ax.plot(recall_curves[method_name], precision_curves[method_name],
                color=colors[i],
                label=f'{method_name} (AUPR={aupr_values[method_name]:.3f})', lw=2)

    ax.set_xlabel('Recall', fontsize=18)
    ax.set_ylabel('Precision', fontsize=18)
    ax.set_title('Precision-Recall Curve', fontsize=18)
    ax.legend(fontsize=10, loc='upper right')
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.show()
    plt.close()

# Plot T-SNE
# Modify labels according to the specified rules
def process_labels(labels_dict):
    processed_labels = {}
    for key, label in labels_dict.items():
        if 'ssRNA-RT' in label:
            new_label = 'ssRNA-RT'
        elif 'dsDNA-RT' in label:
            new_label = 'dsDNA-RT'
        elif 'ssRNA' in label:
            new_label = 'ssRNA'
        elif 'dsRNA' in label:
            new_label = 'dsRNA'
        elif 'ssDNA' in label:
            new_label = 'ssDNA'
        elif 'dsDNA' in label:
            new_label = 'dsDNA'
        else:
            new_label = 'other'
        processed_labels[key] = new_label
    return processed_labels

def load_classification(classification_path):
    """
    Load classification labels from a file.

    Parameters:
    - classification_path (str): Path to the classification file.

    Returns:
    - classification_dict (dict): Dictionary mapping sample IDs to classifications.
    """
    classification_dict = {}
    with open(classification_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                sample_id = parts[0]
                classification = parts[1]
                classification_dict[sample_id] = classification
    return classification_dict

def create_tsne_plot(embedding_path, classification_path, title, figure_path):
    """
    Create a t-SNE plot for the given embeddings and labels.

    Parameters:
    - embedding_path (dict): Dictionary mapping sample IDs to embeddings.
    - classification_path (dict): Dictionary mapping sample IDs to class labels.
    - title (str): Title of the plot.
    - figure_path: Path to save figure
    """

    # Load embeddings
    embeddings = load_embeddings(embedding_path)

    # Load classifications
    classification = load_classification(classification_path)

    # Processed classifications
    if (title[1] == 'Genome_Type'):
        processed_classification = process_labels(classification)
    else:
        processed_classification = classification

    #########
    # Count the number of viruses per label
    # Keep only labels with more than 10 samples
    label_counts = Counter(processed_classification.values())
    processed_classification = {
        key: label for key, label in processed_classification.items()
        if label_counts[label] > 10
    }
    label_counts = Counter(processed_classification.values())
    logger.info(f"Filtered virus count: {label_counts}")

    # Choose a suitable seaborn color palette and fix the color for each label
    labels_list = list(label_counts.keys())
    palette_colors = sns.color_palette('tab10', n_colors=len(labels_list))
    label_to_color = dict(zip(labels_list, palette_colors))

    ###########
    # 1. Virus embeddings t-SNE plot (MLP) with processed labels
    # Filter out embeddings without classification
    filtered_embeddings = {}
    if (title[0] == 'Virus'):
        filtered_embeddings = {key: embeddings[key] for key in processed_classification}
    else:
        all_keys = list(processed_classification.keys())
        subset_size = max(1, len(all_keys) // 10)
        subset_keys = set(all_keys[:subset_size])
        
        filtered_embeddings = {
            key: embeddings[key]
            for key in embeddings
            if get_virus_name(key) in subset_keys
        }

    # Extract sample IDs and corresponding embeddings
    sample_ids = list(filtered_embeddings.keys())
    embedding_vectors = np.array([filtered_embeddings[sid] for sid in sample_ids])
    logger.info(f"Embedding vectors shape: {embedding_vectors.shape}")

    if (title[0] == 'Virus'):
        class_labels = [processed_classification[sid] for sid in sample_ids]
    else:
        class_labels = [processed_classification[get_virus_name(sid)] for sid in sample_ids]

    # Perform t-SNE dimensionality reduction
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
    X_tsne = tsne.fit_transform(embedding_vectors)
    
    # Normalize the t-SNE embeddings
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)

    # Create a DataFrame for plotting
    df = pd.DataFrame({
        'X': X_norm[:, 0],
        'Y': X_norm[:, 1],
        'Class': class_labels
    })

    # >>> MODIFIED/NEW <<<
    # Encode class_labels to numeric values for silhouette_score
    le = LabelEncoder()
    numeric_labels = le.fit_transform(class_labels)
    # Compute the silhouette score
    s_score = silhouette_score(X_norm, numeric_labels, metric='euclidean')
    logger.info(f"Silhouette Coefficient: {s_score:.4f}")

    # Plot using seaborn
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df, x='X', y='Y', hue='Class', s=10, palette=label_to_color)

    plt.title(title, fontsize=18)
    plt.xlabel('t-SNE Dimension 1', fontsize=18)
    plt.ylabel('t-SNE Dimension 2', fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(title='Classification', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=18, title_fontsize=18)
    plt.tight_layout()

    # >>> MODIFIED/NEW <<<
    # Add text box for silhouette coefficient (below the legend)
    stats = f"Silhouette Coefficient = {s_score:.4f}"
    bbox = dict(boxstyle='round', fc='blanchedalmond', ec='orange', alpha=0.5)
    from matplotlib.font_manager import FontProperties
    font = FontProperties(family='monospace', size=15)
    # Adjust coordinates as needed; using transform=ax.transAxes for relative positioning
    ax = plt.gca()
    plt.text(0.025, 0.03, stats, font=font, bbox=bbox, transform=ax.transAxes)

    filename = '_'.join(title) + '.svg'
    save_path = os.path.join(figure_path, filename)
    make_parent_dir(save_path)
    plt.savefig(save_path, format='svg')

    # Ensure the output directory exists
    plt.show()
    plt.close()

def create_violinplot_plot(virus_host_dict_path, virusprotein_host_dict_path):
    """
    Reads virus-host and virusprotein-host dictionaries, counts the number of hosts and
    virus proteins per virus, generates violin plots for each distribution, and
    calculates and outputs the mean, maximum values, and the virus_name corresponding 
    to the maximum values for each dataset. Additionally, generates violin plots
    excluding the top 5% of data to remove outliers.
    
    Parameters:
        virus_host_dict_path (str): Path to the virus_host_dict file.
        virusprotein_host_dict_path (str): Path to the virusprotein_host_dict file.
    """
    plt.rcParams.update(plt.rcParamsDefault)
    
    # Step 1: Read virus_host_dict and count host names per virus
    virus_host_counts = {}
    try:
        with open(virus_host_dict_path, 'r') as f:
            for line in f:
                columns = line.strip().split('\t')
                if not columns:
                    continue  # Skip empty lines
                virus_name = columns[0]
                host_names = columns[1:]
                virus_host_counts[virus_name] = len(host_names)
    except FileNotFoundError:
        print(f"File not found: {virus_host_dict_path}")
        return
    except Exception as e:
        print(f"Error reading {virus_host_dict_path}: {e}")
        return

    # Step 2: Read virusprotein_host_dict and count virus proteins per virus
    # Initialize counts with zero for all virus_names in virus_host_counts
    virus_protein_counts = {virus_name: 0 for virus_name in virus_host_counts}
    
    try:
        with open(virusprotein_host_dict_path, 'r') as f:
            for line in f:
                columns = line.strip().split('\t')
                if len(columns) < 1:
                    continue  # Skip empty lines
                virusprotein_fullname = columns[0]
                try:
                    virus_name = get_virus_name(virusprotein_fullname)
                    if virus_name in virus_protein_counts:
                        virus_protein_counts[virus_name] += 1
                except ValueError:
                    # Skip lines with bad format
                    continue
    except FileNotFoundError:
        print(f"File not found: {virusprotein_host_dict_path}")
        return
    except Exception as e:
        print(f"Error reading {virusprotein_host_dict_path}: {e}")
        return

    # Step 3: Prepare data for plotting
    # Prepare data for host Names per Virus
    host_data = {
        'Count': list(virus_host_counts.values()),
        'Dataset': ['host Names per Virus'] * len(virus_host_counts)
    }
    
    # Prepare data for Virus Proteins per Virus
    protein_data = {
        'Count': list(virus_protein_counts.values()),
        'Dataset': ['Virus Proteins per Virus'] * len(virus_protein_counts)
    }

    # Step 4: Calculate and Output Statistics
    # For host Names per Virus
    host_mean = statistics.mean(host_data['Count']) if host_data['Count'] else 0
    host_max = max(host_data['Count']) if host_data['Count'] else 0
    # Find virus_name(s) with the maximum host count
    max_host_viruses = [virus for virus, count in virus_host_counts.items() if count == host_max]
    
    print("host Names per Virus:")
    print(f"  Mean: {host_mean}")
    print(f"  Maximum: {host_max}")
    print(f"  Virus Name(s) with Maximum hosts: {', '.join(max_host_viruses)}\n")
    
    # For Virus Proteins per Virus
    protein_mean = statistics.mean(protein_data['Count']) if protein_data['Count'] else 0
    protein_max = max(protein_data['Count']) if protein_data['Count'] else 0
    # Find virus_name(s) with the maximum protein count
    max_protein_viruses = [virus for virus, count in virus_protein_counts.items() if count == protein_max]
    
    print("Virus Proteins per Virus:")
    print(f"  Mean: {protein_mean}")
    print(f"  Maximum: {protein_max}")
    print(f"  Virus Name(s) with Maximum Proteins: {', '.join(max_protein_viruses)}\n")
    
    # Step 5: Plot host Names per Virus Violin Plot
    plt.figure(figsize=(8, 6))
    sns.violinplot(data=host_data, x='Dataset', y='Count', inner='box', color="skyblue")
    plt.xticks(fontsize=12)
    plt.ylabel('Number of host Names', fontsize=14)
    plt.xlabel('')
    plt.title('Distribution of host Names per Virus', fontsize=16)
    plt.show()
    plt.close()
    
    # Step 6: Plot Virus Proteins per Virus Violin Plot
    plt.figure(figsize=(8, 6))
    sns.violinplot(data=protein_data, x='Dataset', y='Count', inner='box', color="lightgreen")
    plt.xticks(fontsize=12)
    plt.ylabel('Number of Virus Proteins', fontsize=14)
    plt.xlabel('')
    plt.title('Distribution of Virus Proteins per Virus', fontsize=16)
    plt.show()
    plt.close()
    
    # Step 7: Filter Data to Exclude Top 5% (95th Percentile) as Outliers
    def filter_top_5_percent(data_list):
        """
        Filters out the top 5% of data based on the 95th percentile.
        
        Parameters:
            data_list (list of int): The list of counts to filter.
        
        Returns:
            list of int: The filtered list with values <= 95th percentile.
        """
        if not data_list:
            return []
        percentile_95 = statistics.quantiles(data_list, n=100)[94]  # 95th percentile
        filtered_data = [x for x in data_list if x <= percentile_95]
        return filtered_data

    filtered_host_counts = filter_top_5_percent(host_data['Count'])
    filtered_protein_counts = filter_top_5_percent(protein_data['Count'])

    # Prepare data for plotting filtered host Names per Virus
    filtered_host_data = {
        'Count': filtered_host_counts,
        'Dataset': ['host Names per Virus'] * len(filtered_host_counts)
    }
    
    # Prepare data for plotting filtered Virus Proteins per Virus
    filtered_protein_data = {
        'Count': filtered_protein_counts,
        'Dataset': ['Virus Proteins per Virus'] * len(filtered_protein_counts)
    }

    # Step 8: Plot Filtered host Names per Virus Violin Plot
    plt.figure(figsize=(8, 6))
    sns.violinplot(data=filtered_host_data, x='Dataset', y='Count', inner='box', color="skyblue")
    plt.xticks(fontsize=12)
    plt.ylabel('Number of host Names', fontsize=14)
    plt.xlabel('')
    plt.title('Distribution of host Names per Virus (≤95th Percentile)', fontsize=16)
    plt.show()
    plt.close()
    
    # Step 9: Plot Filtered Virus Proteins per Virus Violin Plot
    plt.figure(figsize=(8, 6))
    sns.violinplot(data=filtered_protein_data, x='Dataset', y='Count', inner='box', color="lightgreen")
    plt.xticks(fontsize=12)
    plt.ylabel('Number of Virus Proteins', fontsize=14)
    plt.xlabel('')
    plt.title('Distribution of Virus Proteins per Virus (≤95th Percentile)', fontsize=16)
    plt.show()
    plt.close()

# iphop test
def read_virus_host_dict(file_path):
    """
    Reads the virus_host_dict file and returns a dictionary
    mapping virus names to a list of host names.
    In addition, returns a DataFrame with a single column 'virus_name'.
    The input file format is:
    virus_name<TAB>host_name_1<TAB>host_name_2<...>
    Example line:
    Gordonia phage Anon    Gordonia terrae
    """
    virus_names = []
    virus_host_dict = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue  # Skip empty lines
            parts = line.split('\t')
            if len(parts) < 2:
                print(f"Warning: Line {line_num} in {file_path} does not have enough columns.")
                continue
            virus_name = parts[0].strip()
            host_names = parts[1:]
            # store them in a list or set
            virus_host_dict[virus_name] = host_names
            virus_names.append(virus_name)

    # create a DataFrame for the virus_name column
    return pd.DataFrame({"virus_name": virus_names}), virus_host_dict

def collect_iphop_prediction_result(
    virus_host_dict,
    predict_file,
    top=10,
    Therehold=0,
    taxonomy_file=None,
    classify_level=None
):
    """
    Collects iPHoP prediction results from the given predict_file and calculates
    correct_ratio, incorrect_ratio, and no_answer_ratio.

    Parameters
    ----------
    virus_host_dict : dict
        A dictionary mapping virus_name -> list of host_names (ground truth).
    predict_file : str
        The file path to the prediction results. The file format is:
        virus_name<TAB>host_name<TAB>probability
    top : int or None
        If top is not None, only the top 'top' predictions (by probability) are considered for each virus_name.
    Therehold : float
        If top is None, only predictions whose probability > Therehold will be considered,
        but only the highest-probability (Top-1) prediction is used to determine correctness.
        If no prediction exceeds Therehold => "No answer".
        If the Top-1 prediction is not correct => "Incorrect".
        If both top and Therehold are provided, top overrides Therehold.
    taxonomy_file : str or None
        The file path to the host taxonomy data. If None, no taxonomy-based comparison is used.
        Format (tab-separated), containing header:
            host    species    genus    family    order    class    phylum    kingdom
    classify_level : str or None
        One of ["host", "species", "genus", "family", "order", "class", "phylum", "kingdom"].
        If taxonomy_file is not None and classify_level is None, defaults to "species".
        If taxonomy_file and classify_level are both None, the original logic (exact host match) is used.

    Returns
    -------
    correct_ratio : float
    incorrect_ratio : float
    no_answer_ratio : float

    Explanation:
    1) We group predictions by virus_name.
    2) For each virus_name, we sort predictions by probability descending.
       - if top is not None, we take the first 'top' predictions
       - else (using Therehold), we only consider predictions with probability > Therehold
         but only take the highest-probability (Top-1) prediction.
         If no prediction exceeds Therehold => "No answer".
         If the Top-1 prediction is incorrect => "Incorrect".
    3) We then check if any (or the only one in the threshold case) predicted host_name is in
       the ground-truth list stored in virus_host_dict[virus_name]. The logic of checking
       depends on taxonomy_file and classify_level:
         - if taxonomy_file and classify_level are both None:
             If the predicted host is exactly in the ground-truth list, it's correct.
         - else:
             Compare the predicted host and ground-truth host at the given classify_level.
             If they match on that level, it's correct.
       If none is correct => "Incorrect".
       If there are no predicted results (filtered is empty) => "No_answer".
    4) correct_ratio = correct_count / total_viruses, etc.
       no_answer_ratio = 1 - correct_ratio - incorrect_ratio
    """
    import pandas as pd
    
    # -------------------------------------------------------------------------
    # A) Read the taxonomy_file (if provided) and build a dictionary
    # -------------------------------------------------------------------------
    host_taxonomy_dict = {}
    valid_levels = ["host", "species", "genus", "family", "order", "class", "phylum", "kingdom"]
    
    if taxonomy_file is not None:
        # Default classify_level to "species" if not provided
        if classify_level is None:
            classify_level = "species"

        if classify_level not in valid_levels:
            raise ValueError(f"classify_level must be one of {valid_levels}, got {classify_level}")

        # Load the taxonomy file
        df_tax = pd.read_csv(taxonomy_file, sep="\t", header=0, dtype=str).fillna("_")

        # Build a dictionary: host_name -> { "host":..., "species":..., ... }
        for idx, row in df_tax.iterrows():
            host_name = row["host"]
            level_dict = {}
            for level in valid_levels:
                value = row[level] if row[level] != "_" else host_name
                level_dict[level] = value
            host_taxonomy_dict[host_name] = level_dict

    # -------------------------------------------------------------------------
    # B) A helper function to get the classification level of a host
    # -------------------------------------------------------------------------
    def get_host_level_name(host, level):
        """
        Return the classification name of 'host' at taxonomy level 'level'.
        If the host is not found in host_taxonomy_dict, or taxonomy_file is None,
        return the host itself.
        """
        if (taxonomy_file is None) or (host not in host_taxonomy_dict):
            return host
        return host_taxonomy_dict[host][level]

    # -------------------------------------------------------------------------
    # C) Load the prediction file into a DataFrame
    # -------------------------------------------------------------------------
    cols = ["virus_name", "predicted_host_name", "probability"]
    df_pred = pd.read_csv(predict_file, sep="\t", names=cols)

    # 2) Group by virus_name
    grouped = df_pred.groupby("virus_name")

    correct_count = 0
    incorrect_count = 0
    total_viruses = len(virus_host_dict)

    # -------------------------------------------------------------------------
    # D) Iterate over each virus_name's predictions
    # -------------------------------------------------------------------------
    for virus_name, group in grouped:
        # Only consider virus_names that appear in virus_host_dict
        if virus_name not in virus_host_dict:
            continue

        # Sort predictions by probability descending
        group_sorted = group.sort_values(by="probability", ascending=False)

        # ---------------------------------------------------------------------
        # Filter based on top or threshold:
        # ---------------------------------------------------------------------
        if top is not None:
            # Original behavior: Take the top 'top' predictions
            filtered = group_sorted.head(top)
        else:
            # 1) Only consider predictions with probability > Therehold
            # 2) From those, take the highest-probability (Top-1)
            filtered = group_sorted[group_sorted["probability"] > Therehold].head(1)

        # If no valid predictions after filtering => "No_answer"
        if len(filtered) == 0:
            continue  # Will be counted as "No_answer" later

        # Ground truth hosts
        ground_truth_hosts = virus_host_dict[virus_name]

        # E) Check if ANY (or the only one if threshold is used) predicted host matches the ground-truth
        if taxonomy_file is not None:
            ground_truth_level_names = set(
                get_host_level_name(h, classify_level) for h in ground_truth_hosts
            )
        else:
            ground_truth_level_names = set(ground_truth_hosts)

        # Get the predicted hosts
        predicted_hosts = filtered["predicted_host_name"].tolist()

        found_correct = False
        for ph in predicted_hosts:
            predicted_level_name = get_host_level_name(ph, classify_level) \
                if taxonomy_file is not None else ph

            if predicted_level_name in ground_truth_level_names:
                found_correct = True
                break

        if found_correct:
            correct_count += 1
        else:
            incorrect_count += 1

    # -------------------------------------------------------------------------
    # F) Compute the final ratios
    # -------------------------------------------------------------------------
    if total_viruses > 0:
        correct_ratio = correct_count / total_viruses
        incorrect_ratio = incorrect_count / total_viruses
        no_answer_count = total_viruses - (correct_count + incorrect_count)
        no_answer_ratio = no_answer_count / total_viruses
    else:
        correct_ratio = 0.0
        incorrect_ratio = 0.0
        no_answer_ratio = 0.0

    return correct_ratio, incorrect_ratio, no_answer_ratio

def collect_iphop_baseline_result(df_test, 
                                  df_virus_list, 
                                  df_baseline_single, 
                                  df_baseline_other, 
                                  method_name_dict):
    """
    This function collects the number of correct and incorrect predictions for each method,
    then calculates the ratio (correct_count / total_viruses_found) and (incorrect_count / total_viruses_found).
    
    Additionally, we now compute the no_answer_ratio = 1 - correct_ratio - incorrect_ratio 
    for each method, based on the viruses found in df_test.

    Parameters
    ----------
    df_test : pd.DataFrame
        DataFrame for iphop_test (contains the virus_name in the first column).
    df_virus_list : pd.DataFrame
        DataFrame for virus_list.csv (has 'Virus Id' and 'Virus name' columns).
    df_baseline_single : pd.DataFrame
        DataFrame for baseline_result_single.csv (must include 'Genome', 'Method', 'Result' columns).
    df_baseline_other : pd.DataFrame
        DataFrame for result_baseline_other.csv (must include 'Virus', 'Method', 'Prediction type' columns).
    method_name_dict : dict
        Dictionary mapping the method-key (e.g., "Blast") to the final method legend name (e.g., "blast_Virulent").

    Returns
    -------
    (dict, dict, dict)
        Three dictionaries:
            method_correct_ratio : {method_name: float}
                The ratio of correct predictions for each method.
            method_incorrect_ratio : {method_name: float}
                The ratio of incorrect predictions for each method.
            method_no_answer_ratio : {method_name: float}
                The ratio of no_answer for each method = 1 - correct_ratio - incorrect_ratio.
    """
    # 1) Build a mapping from virus_name -> virus_id based on virus_list
    virus_name_to_id = dict(zip(df_virus_list["Virus name"], df_virus_list["Virus Id"]))

    # 2) For each virus_name in df_test, try to find corresponding Virus Id
    found_virus_ids = []
    not_found_list = []
    for vname in df_test["virus_name"]:
        if vname in virus_name_to_id:
            found_virus_ids.append(virus_name_to_id[vname])
        else:
            print(f"error: {vname} not found in virus_name_to_id")
            not_found_list.append(vname)

    # The number of viruses we successfully found
    found_count = len(found_virus_ids)
    print(f"Number of virus names found: {found_count}")

    # 3) Initialize counters for correct/incorrect for each method_key
    method_counter = {}
    for mkey in method_name_dict.keys():
        method_counter[mkey] = {"Correct": 0, "Incorrect": 0}

    # 4) Filter baseline_result_single.csv to rows that matter
    df_single_filtered = df_baseline_single[
        (df_baseline_single["Genome"].isin(found_virus_ids)) &
        (df_baseline_single["Result"].isin(["Correct", "Incorrect"]))
    ]

    # 5) Count correct/incorrect in baseline_result_single.csv
    for _, row in df_single_filtered.iterrows():
        method = row["Method"]
        result = row["Result"]
        if method in method_counter:
            method_counter[method][result] += 1

    # 6) Filter result_baseline_other.csv
    df_other_filtered = df_baseline_other[
        (df_baseline_other["Virus"].isin(found_virus_ids)) &
        (df_baseline_other["Prediction type"].isin(["Correct", "Incorrect"]))
    ]

    # 7) Count correct/incorrect in result_baseline_other.csv
    for _, row in df_other_filtered.iterrows():
        method = row["Method"]
        result = row["Prediction type"]
        if method in method_counter:
            method_counter[method][result] += 1

    # 8) Convert the counts to ratio:
    method_correct_ratio = {}
    method_incorrect_ratio = {}
    method_no_answer_ratio = {}

    for mkey, counts in method_counter.items():
        method_name = method_name_dict[mkey]
        correct_count = counts["Correct"]
        incorrect_count = counts["Incorrect"]

        if found_count > 0:
            correct_ratio = correct_count / found_count
            incorrect_ratio = incorrect_count / found_count
        else:
            correct_ratio = 0.0
            incorrect_ratio = 0.0

        # no_answer_ratio
        no_answer_ratio = 1.0 - correct_ratio - incorrect_ratio

        method_correct_ratio[method_name] = correct_ratio
        method_incorrect_ratio[method_name] = incorrect_ratio
        method_no_answer_ratio[method_name] = no_answer_ratio

    return method_correct_ratio, method_incorrect_ratio, method_no_answer_ratio

def plot_iphop_result(method_correct_ratio, method_incorrect_ratio):
    """
    This function creates a stacked bar chart using correct and incorrect ratios for each method.
    Additionally, use different colors for each method's 'Correct' portion, and grey for 'Incorrect'.
    Font sizes are set to 14, and legend is placed on the right side (outside).

    Parameters
    ----------
    method_correct_ratio : dict
        {method_name: float} for correct ratio.
    method_incorrect_ratio : dict
        {method_name: float} for incorrect ratio.
    """

    # Convert to a DataFrame
    data_rows = []
    for method_name in method_correct_ratio.keys():
        data_rows.append({
            "method_name": method_name,
            "correct_ratio": method_correct_ratio[method_name],
            "incorrect_ratio": method_incorrect_ratio[method_name]
        })
    df_plot = pd.DataFrame(data_rows)

    # Plot: stacked bar chart
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))

    x_positions = range(len(df_plot))
    correct_vals = df_plot["correct_ratio"].values
    incorrect_vals = df_plot["incorrect_ratio"].values

    # We assign a distinct color for each method's 'Correct' portion
    palette = sns.color_palette("hls", n_colors=len(df_plot))

    bars_correct = ax.bar(
        x_positions,
        correct_vals,
        color=palette  # different color for each bar
    )

    bars_incorrect = ax.bar(
        x_positions,
        incorrect_vals,
        bottom=correct_vals,
        color="lightgray"  # grey color for the 'Incorrect' portion
    )

    patches = []
    for method_name, color in zip(df_plot["method_name"], palette):
        patch = mpatches.Patch(facecolor=color, label=method_name)
        patches.append(patch)

    incorrect_patch = mpatches.Patch(facecolor="lightgray", label="Incorrect")
    patches.append(incorrect_patch)

    ax.legend(
        handles=patches,
        loc='upper left',
        bbox_to_anchor=(1.05, 1),
        fontsize=14
    )

    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(df_plot["method_name"], rotation=45, ha="right")

    # set y limit to [0,1]
    ax.set_ylim(0, 1)

    # Set font sizes
    ax.set_title("Correct and Incorrect Ratio", fontsize=14)
    ax.set_xlabel("Method", fontsize=14)
    ax.set_ylabel("Ratio", fontsize=14)
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)

    plt.tight_layout()
    plt.show()
    plt.close()

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from collections import defaultdict

def host_predict_result_analyze(
    dict_file, 
    dict_test_file, 
    ic_score_file, 
    predict_file,
    output_file,
    top=10  # parameter for top-k host predictions
):
    """
    Analyze host prediction results and output a confusion matrix for the top 20 hosts.

    Parameters
    ----------
    dict_file : str
        Path to vhpredictor_data/vhdb/virus_host_dict.
    dict_test_file : str
        Path to vhpredictor_data/vhdb/virus_host_dict_test (subset of dict_file).
    ic_score_file : str
        Path to vhpredictor_data/vhdb/host_ic_score.
    predict_file : str
        Path to vhpredictor_data/experiment/test_result/vhpredictor_virus_central_virus_probability.
    output_file : str
        Output path for sorted host data.
    top : int, optional
        For each virus, only consider the top `top` hosts by probability, by default 10.

    Returns
    -------
    None
    """

    # --------------------------------------------------------------------------
    # Step 1: Read virus_host_dict and count occurrences of each host.
    # --------------------------------------------------------------------------
    host_count_dict = {}
    with open(dict_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 2:
                continue
            virus_name = parts[0]
            hosts = parts[1:]
            for h in hosts:
                host_count_dict[h] = host_count_dict.get(h, 0) + 1

    # --------------------------------------------------------------------------
    # Step 2: Read host_ic_score, build a mapping host->ic_score.
    # --------------------------------------------------------------------------
    host_ic_dict = {}
    with open(ic_score_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 2:
                continue
            host_name = parts[0]
            ic_score = float(parts[1])
            host_ic_dict[host_name] = ic_score

    # --------------------------------------------------------------------------
    # Step 3: Read dict_test_file to get (virus, host) pairs in the test set.
    # --------------------------------------------------------------------------
    host_count_test = {}
    virus_host_test = set()
    with open(dict_test_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 2:
                continue
            virus_name = parts[0]
            hosts = parts[1:]
            for h in hosts:
                virus_host_test.add((virus_name, h))
                host_count_test[h] = host_count_test.get(h, 0) + 1

    # --------------------------------------------------------------------------
    # Step 4: Read predict_file, but only keep the top hosts for each virus.
    # --------------------------------------------------------------------------
    predicted_dict = defaultdict(list)
    with open(predict_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            # We expect three columns: virus_name, host_name, probability
            if len(parts) < 3:
                continue
            virus_name = parts[0]
            host_name = parts[1]
            probability = float(parts[2])
            predicted_dict[virus_name].append((host_name, probability))

    # For each virus, sort hosts by descending probability, then select up to `top`
    # Build a set of predicted (virus, host)
    predict_set = set()
    for virus_name, host_probs in predicted_dict.items():
        host_probs.sort(key=lambda x: x[1], reverse=True)
        top_hosts = host_probs[:top]
        for host_name, prob in top_hosts:
            predict_set.add((virus_name, host_name))

    # --------------------------------------------------------------------------
    # Step 5: Calculate recall for each host in the test set.
    # --------------------------------------------------------------------------
    correct_count_dict = {}
    for (virus, host) in virus_host_test:
        if (virus, host) in predict_set:
            correct_count_dict[host] = correct_count_dict.get(host, 0) + 1

    host_recall_dict = {}
    for host, denom in host_count_test.items():
        numerator = correct_count_dict.get(host, 0)
        recall_value = numerator / denom if denom > 0 else 0.0
        host_recall_dict[host] = recall_value

    # --------------------------------------------------------------------------
    # Step 6: Gather data only for hosts that appear in the test set.
    #         Sort by occurrences (descending) => sorted_items
    # --------------------------------------------------------------------------
    hosts_in_test = list(host_count_test.keys())
    data_dict = {}
    for h in hosts_in_test:
        occ = host_count_dict.get(h, 0)       # occurrences in dict_file
        ic = host_ic_dict.get(h, 0.0)         # IC score
        rec = host_recall_dict.get(h, 0.0)    # recall
        data_dict[h] = (occ, ic, rec)

    sorted_items = sorted(data_dict.items(), key=lambda x: x[1][0], reverse=True)
    host_names   = [item[0] for item in sorted_items]
    occ_array    = np.array([item[1][0] for item in sorted_items])
    ic_array     = np.array([item[1][1] for item in sorted_items])
    recall_array = [item[1][2] for item in sorted_items]

    # Write sorted data to output file
    with open(output_file, 'w', encoding='utf-8') as out_f:
        for item in sorted_items:
            out_f.write(f"{item}\n")  # each item in one line

    num_hosts = len(sorted_items)
    if num_hosts == 0:
        print("No hosts found in the test set.")
        return

    # --------------------------------------------------------------------------
    # Confusion Matrix for Top 20 hosts
    # --------------------------------------------------------------------------
    # 1) Get the top 20 hosts from sorted_items
    top_20_hosts = [item[0] for item in sorted_items[:20]]
    top_20_index = {h: i for i, h in enumerate(top_20_hosts)}

    # 2) Initialize the confusion matrix: 20 x 20
    conf_matrix = np.zeros((20, 20), dtype=int)

    # 3) For each (virus, true_host) in the test set, determine predicted host
    #    only if 'true_host' is in top 20. Otherwise, we skip it for the 20x20 matrix.
    for (virus, true_host) in virus_host_test:
        if true_host not in top_20_index:
            continue

        # find the top-k predicted hosts for this virus
        host_probs = predicted_dict.get(virus, [])
        # sort by probability descending
        host_probs.sort(key=lambda x: x[1], reverse=True)
        top_hosts = [hp[0] for hp in host_probs[:top]]

        # if true_host is within top-k, predicted = true_host
        if true_host in top_hosts:
            pred_host = true_host
        else:
            # if not in top-k, predicted = top-1 host (the highest probability)
            if len(top_hosts) > 0:
                pred_host = top_hosts[0]
            else:
                # if no predictions available, skip
                continue

        # if the predicted host is also among top 20, we update the confusion matrix
        if pred_host in top_20_index:
            i_true = top_20_index[true_host]
            i_pred = top_20_index[pred_host]
            conf_matrix[i_true, i_pred] += 1

    # 4) Plot confusion matrix with seaborn
    plt.figure(figsize=(10, 8))

    # Create a custom colormap based on "Blues", but darker overall
    original_cmap = plt.cm.get_cmap("Blues")  # or plt.colormaps["Blues"] in newer Matplotlib
    colors = [original_cmap(0.3 + 0.7 * i / 255) for i in range(256)]
    new_cmap = mcolors.ListedColormap(colors)

    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt='d',
        cmap=new_cmap,  # use the custom darker "Blues"
        xticklabels=top_20_hosts,
        yticklabels=top_20_hosts
    )

    plt.title("Confusion Matrix (Top 20 hosts)", fontsize=20)
    plt.tight_layout()
    plt.show()
    plt.close()

    # --------------------------------------------------------------------------
    # Step 7 : Rescale and create bar plot (same as original code)
    # --------------------------------------------------------------------------
    sns.set_theme(style="whitegrid")

    fig, ax = plt.subplots(figsize=(12, 8))

    x_positions = np.arange(num_hosts)

    # Top side (occurrences): log10 transform
    top_heights = np.log10(occ_array + 1.0)  
    max_log_occ = np.max(top_heights) if len(top_heights) else 1

    # Bottom side (IC score): we want the maximum negative to be -max_log_occ
    max_ic = np.max(ic_array) if len(ic_array) else 1
    scale = max_log_occ / max_ic
    bottom_heights = -ic_array * scale  # negative side

    # Set consistent y-limits
    y_max = max_log_occ
    y_min = -max_log_occ
    ax.set_ylim(y_min, y_max)

    # Create a custom colormap for recall to avoid extreme whiteness at 0
    original_cmap = plt.colormaps["RdPu"]
    # We cut the colormap to start from 0.5 -> 1.0 (for example)
    colors = [original_cmap(0.5 + 0.5 * i/255) for i in range(256)]
    new_cmap = mcolors.ListedColormap(colors)

    # Prepare normalizer for recall in [0,1]
    norm = plt.Normalize(0, 1)

    # Plot bars
    for i in range(num_hosts):
        r = recall_array[i]
        color = new_cmap(norm(r))

        # Top bar
        ax.bar(
            x_positions[i], 
            top_heights[i],
            width=0.8,
            color=color,
            linewidth=0
        )
        # Bottom bar
        ax.bar(
            x_positions[i],
            bottom_heights[i],
            width=0.8,
            color=color,
            linewidth=0
        )

    # Build y-ticks so the negative side shows IC scores, 
    # the positive side shows log-scale for occurrences.
    top_tick_int = int(math.floor(max_log_occ)) + 1
    top_ticks = list(range(0, top_tick_int + 1))  # e.g. [0,1,2,...]
    top_tick_labels = [r"$10^{%d}$" % t for t in top_ticks]

    # Negative side ticks for IC scores
    num_bottom_ticks = 5
    ic_values = np.linspace(0, max_ic, num_bottom_ticks+1)  # e.g. [0, ..., max_ic]
    ic_tick_positions = -ic_values * scale
    ic_tick_labels = [f"{v:.1f}" for v in ic_values]
    ic_tick_positions = ic_tick_positions[::-1]
    ic_tick_labels = ic_tick_labels[::-1]

    combined_ticks = list(ic_tick_positions[:-1]) + top_ticks
    combined_labels = ic_tick_labels[:-1] + top_tick_labels

    ax.set_yticks(combined_ticks)
    ax.set_yticklabels(combined_labels)

    ax.axhline(0, color='black', linewidth=1)

    ax.set_xlim(-0.5, num_hosts)
    ax.set_xticks([num_hosts])
    ax.set_xticklabels([f"{num_hosts}"])
    ax.set_xlabel("host List", fontsize=20)

    sm = plt.cm.ScalarMappable(cmap=new_cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.03)
    cbar.set_label("Recall", fontsize=20)
    cbar.ax.tick_params(labelsize=18)

    ax.set_ylabel("")
    ax.text(
        -0.07, 
        1.0, 
        "Number of annotated virus",
        rotation=90,
        ha="right",
        va="top",
        transform=ax.transAxes,
        fontsize=18
    )
    ax.text(
        -0.07,
        0.3,
        "IC Score",
        rotation=90,
        ha="right",
        va="top",
        transform=ax.transAxes,
        fontsize=18
    )

    ax.set_title("host Label Analysis", fontsize=20)

    sns.despine(left=False, bottom=False, right=True, top=True)
    ax.tick_params(axis='both', labelsize=18)

    plt.tight_layout()
    plt.show()
    plt.close()

def correct_incorrect_control(
    virus_host_dict,
    predict_file,
    taxonomy_file=None,
    classify_level=None
):
    """
    This function tests thresholds ranging from 0.00 to 1.00 in increments of 0.01.
    For each threshold, it calls `collect_iphop_prediction_result` to get the correct_ratio,
    incorrect_ratio, and then plots them (Correct ratio on the X-axis, Incorrect ratio on the Y-axis)
    using Seaborn.

    Additionally, it prints out the Correct, Incorrect, and No_answer ratio when the threshold
    is in [0, 0.2, 0.4, 0.6, 0.8, 1.0].

    Parameters
    ----------
    virus_host_dict : dict
        Mapping from virus_name -> list of ground-truth host names.
    predict_file : str
        Prediction results file path (virus_name<TAB>host_name<TAB>probability).
    taxonomy_file : str or None
        Taxonomy file path (if available).
    classify_level : str or None
        Classification level to compare ("host", "species", "genus", etc.).

    Returns
    -------
    None
        Displays a scatter plot using Seaborn, with correct ratio vs. incorrect ratio,
        and prints certain threshold metrics.
    """

    # Lists for storing thresholds, correct ratios, and incorrect ratios for plotting.
    thresholds_list = []
    correct_ratios = []
    incorrect_ratios = []

    # First loop: vary threshold from 0.00 to 1.00 in increments of 0.01 for the scatter plot
    for i in tqdm(range(101), desc="Processing thresholds"):
        t = i / 100.0
        cr, ir, _ = collect_iphop_prediction_result(
            virus_host_dict=virus_host_dict,
            predict_file=predict_file,
            top=None,                  # We set top=None to respect the threshold filtering
            Therehold=t,              # Varying threshold
            taxonomy_file=taxonomy_file,
            classify_level=classify_level
        )
        thresholds_list.append(t)
        correct_ratios.append(cr)
        incorrect_ratios.append(ir)

    # Create a scatter plot with a figure and axes to avoid colorbar issues
    fig, ax = plt.subplots(figsize=(8, 6))

    # Scatter plot: X = correct ratio, Y = incorrect ratio, hue = threshold
    scatter = sns.scatterplot(
        ax=ax,
        x=correct_ratios,
        y=incorrect_ratios,
        hue=thresholds_list,
        palette='viridis',
        #legend=False
    )

    # Remove the original legend
    if scatter.legend_:
        scatter.legend_.remove()

    # Set axis labels and title
    scatter.set_xlabel("Correct Ratio")
    scatter.set_ylabel("Incorrect Ratio")
    scatter.set_title("Correct vs. Incorrect Ratio by Threshold")

    # Create a colorbar for threshold
    norm = plt.Normalize(min(thresholds_list), max(thresholds_list))
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array([])

    # Important: use fig.colorbar(...) instead of plt.colorbar(...) to fix ValueError
    cbar = fig.colorbar(sm, ax=ax, fraction=0.1, pad=0.1)
    cbar.set_label("Threshold", fontsize=20)
    cbar.ax.tick_params(labelsize=18)

    # Show the scatter plot
    plt.show()

    # Additional step: print out Correct, Incorrect, and No_answer for thresholds [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    special_thresholds = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    print("\nDetailed ratios for selected thresholds:")
    for t in special_thresholds:
        cr, ir, _ = collect_iphop_prediction_result(
            virus_host_dict=virus_host_dict,
            predict_file=predict_file,
            top=None,
            Therehold=t,
            taxonomy_file=taxonomy_file,
            classify_level=classify_level
        )
        no_answer = 1 - cr - ir
        print(f"Threshold = {t:.1f}: Correct = {cr:.4f}, Incorrect = {ir:.4f}, No_answer = {no_answer:.4f}")