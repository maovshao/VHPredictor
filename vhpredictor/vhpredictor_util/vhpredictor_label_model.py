import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from vhpredictor_util.util import get_virus_name

class vhpredictor_label_model(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(vhpredictor_label_model, self).__init__()
        self.linear1 = nn.Linear(input_dim, 960)
        self.linear2 = nn.Linear(960, 480)
        self.linear3 = nn.Linear(480, 240)
        self.classifier = nn.Linear(240, output_dim)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear1(x)))
        x = self.dropout(F.relu(self.linear2(x)))
        x = self.dropout(F.relu(self.linear3(x)))
        embeddings = x  # Embeddings before the classifier
        logits = self.classifier(x)
        return logits, embeddings  # Return both logits and embeddings

class load_dataset(Dataset):
    def __init__(self, esm_embedding, label_index, host_dict_path = None):
        # Read virusprotein_host_dict
        virusprotein_hosts = {}
        self.host_dict_path = host_dict_path
        self.label_index = label_index
        if self.host_dict_path:
            with open(self.host_dict_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    # First, split the line by tab to separate virus_name and hosts
                    parts = line.split('\t')
                    if len(parts) < 2:
                        continue  # Skip lines that do not have enough parts
                    virusprotein_fullname = parts[0]
                    if virusprotein_fullname in esm_embedding:
                        virusprotein_hosts[virusprotein_fullname] = parts[1:]
                    else:
                        assert("Difference between label and embedding")
        else:
            for virusprotein_fullname in esm_embedding:
                virusprotein_hosts[virusprotein_fullname] = None


        self.num_hosts = len(self.label_index)

        # Build x, y, and virus_names
        self.embeddings = []
        self.virusprotein_ids = []
        self.virus_names = {}
        self.hosts = []

        for virusprotein_id in virusprotein_hosts:
            self.embeddings.append(esm_embedding[virusprotein_id])
            self.virusprotein_ids.append(virusprotein_id)
            self.virus_names[virusprotein_id] = get_virus_name(virusprotein_id)
            if self.host_dict_path:
                y_vector = torch.zeros(self.num_hosts)
                for host in virusprotein_hosts[virusprotein_id]:
                    if host in self.label_index:
                        idx = self.label_index[host]
                        y_vector[idx] = 1
                self.hosts.append(y_vector)
            else:
                self.hosts.append(torch.zeros(self.num_hosts))

        self.embeddings = torch.stack(self.embeddings)
        self.hosts = torch.stack(self.hosts)
        # if self.host_dict_path:
        #     self.hosts = torch.stack(self.hosts)
        # else:
        #     self.hosts = torch.empty()

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.virusprotein_ids[idx], self.hosts[idx]  # Return virus_names

    def get_dim(self):
        return int(self.embeddings[0].shape[0])

    def get_class_num(self):
        return self.num_hosts
    
    def get_virus_name(self, virusprotein_id):
        return self.virus_names[virusprotein_id]