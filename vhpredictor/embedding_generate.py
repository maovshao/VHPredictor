import argparse
import torch
import time
import pickle
from logzero import logger
from vhpredictor_util.util import make_parent_dir, get_virus_name, load_embeddings

def esm_embedding_generate(esm_model_path, fasta, embedding, nogpu, onehot):
    """
    This function generates embeddings using ESM model if esm_model_path != 'onehot'.
    Otherwise, it calls onehot_embedding_generate for 'onehot' embeddings.
    """
    from esm import FastaBatchedDataset, pretrained

    print(f"With GPU: {torch.cuda.is_available()}")
    esm_model, alphabet = pretrained.load_model_and_alphabet(esm_model_path)
    esm_model.eval()

    if torch.cuda.is_available() and not nogpu:
        esm_model = esm_model.cuda()
        print("Transferred model to GPU")

    dataset = FastaBatchedDataset.from_file(fasta)
    batches = dataset.get_batch_indices(16384, extra_toks_per_seq=1)
    data_loader = torch.utils.data.DataLoader(
        dataset, collate_fn=alphabet.get_batch_converter(1022), batch_sampler=batches
    )
    print(f'Read {fasta} with {len(dataset)} sequences')
    embedding_dic = {}
    with torch.no_grad():
        for batch_idx, (labels, strs, toks) in enumerate(data_loader):
            print(
                f"Processing {batch_idx + 1} of {len(batches)} batches ({toks.size(0)} sequences)"
            )

            if onehot:
                print("Using onehot embedding method.")
                onehot_embedding_dimension = len(alphabet.all_toks)
                onehot_embedding = torch.nn.functional.one_hot(toks, num_classes=onehot_embedding_dimension).float()
                for i, label in enumerate(labels):
                    embedding_dic[label] = onehot_embedding[i, 1 : len(strs[i]) + 1].mean(0).clone().cpu()
                continue

            if torch.cuda.is_available() and not nogpu:
                toks = toks.to(device="cuda", non_blocking=True)

            print(f"Get embedding from layer {esm_model.num_layers}")
            out = esm_model(toks, repr_layers=[esm_model.num_layers], return_contacts=False)["representations"][esm_model.num_layers]

            for i, label in enumerate(labels):
                # get mean embedding
                esm_embedding = out[i, 1 : len(strs[i]) + 1].mean(0).clone().cpu()
                embedding_dic[label] = esm_embedding
        
        with open(embedding, 'wb') as handle:
            pickle.dump(embedding_dic, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # input
    # parser.add_argument('-emp', '--esm_model_path', type=str, default='vhpredictor_data/model/esm1b/esm1b_t33_650M_UR50S.pt', help="ESM model location")
    parser.add_argument('-emp', '--esm_model_path', type=str, default='vhpredictor_data/model/esm2/esm2_t12_35M_UR50D.pt', help="ESM model location")
    
    parser.add_argument('-f', '--fasta', type=str, help="Fasta file to generate embedding")

    # output
    parser.add_argument('-pe', '--protein_embedding', type=str, help="Protein embedding result")
    parser.add_argument('-ve', '--virus_embedding', type=str, help="Virus embedding result")

    # parameter
    parser.add_argument("--nogpu", action="store_true", help="Do not use GPU even if available")
    parser.add_argument("--onehot", action="store_true", help="Use onehot embedding")
    args = parser.parse_args()

    time_start = time.time()
    make_parent_dir(args.protein_embedding)

    # Step 1: Generate protein embedding (using ESM or onehot)
    esm_embedding_generate(args.esm_model_path, args.fasta, args.protein_embedding, args.nogpu, args.onehot)

    time_end = time.time()
    logger.info('Protein embedding generation time cost:', time_end - time_start, 's')

    # Step 2: Load protein embeddings
    embeddings_dict = load_embeddings(args.protein_embedding)

    # Total number of virus proteins originally
    total_virus_proteins_original = len(embeddings_dict)

    # Step 3: Map virus names to their corresponding protein IDs
    virus_to_proteins = {}
    total_virus_proteins_collected = 0
    virus_embeddings = {}

    for virusprotein_fullname in embeddings_dict.keys():
        virus_name = get_virus_name(virusprotein_fullname)
        virus_to_proteins.setdefault(virus_name, []).append(virusprotein_fullname)
        total_virus_proteins_collected += 1

    # Step 4: Compute mean embeddings for each virus using PyTorch
    for virus_name, protein_ids in virus_to_proteins.items():
        protein_embeddings = []
        for pid in protein_ids:
            protein_embeddings.append(embeddings_dict.get(pid))
        if protein_embeddings:
            stacked_embeddings = torch.stack(protein_embeddings)
            mean_embedding = torch.mean(stacked_embeddings, dim=0)
            virus_embeddings[virus_name] = mean_embedding
        else:
            logger.warning(f"No embeddings found for proteins of virus '{virus_name}'.")

    logger.info(f"Computed mean embeddings for {len(virus_embeddings)} viruses.")

    # Step 5: Save virus embeddings
    with open(args.virus_embedding, 'wb') as f_emb:
        pickle.dump(virus_embeddings, f_emb, protocol=pickle.HIGHEST_PROTOCOL)

    # Output the counts
    logger.info(f"Total number of virus proteins originally: {total_virus_proteins_original}")
    logger.info(f"Total number of virus proteins collected: {total_virus_proteins_collected}")
    logger.info(f"Total number of viruses with computed embeddings: {len(virus_embeddings)}")
