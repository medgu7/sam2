import torch
import torch.nn.functional as F

class MemoryBank:
    def __init__(self, max_size: int, device: str = 'cuda'):
        self.max_size = max_size
        self.bank = []  # List of [feature, pos_enc, iou, image_embed]
        self.device = device

    def add(self, feature, pos_enc, iou, image_embed):
        """Add a new memory entry or replace an existing one"""
        entry = [feature.detach(), pos_enc.detach(), iou.detach(), image_embed.detach()]
        if len(self.bank) < self.max_size:
            self.bank.append(entry)
        else:
            self._replace(entry)

    def _replace(self, new_entry):
        """Replace least useful entry based on similarity and iou"""
        new_feat_flat = F.normalize(new_entry[0].reshape(-1), p=2, dim=0).to(self.device)

        # Build similarity matrix between current bank entries
        bank_flattened = [F.normalize(e[0].reshape(-1), p=2, dim=0) for e in self.bank]
        bank_matrix = torch.stack(bank_flattened)
        sim_scores = torch.matmul(bank_matrix, new_feat_flat)

        # Find the most similar bank entry and its most redundant neighbor
        min_sim_idx = torch.argmin(sim_scores)
        similarity_matrix = torch.matmul(bank_matrix, bank_matrix.T)
        similarity_matrix[torch.arange(len(self.bank)), torch.arange(len(self.bank))] = float('-inf')
        max_sim_idx = torch.argmax(similarity_matrix[min_sim_idx])

        # Soft threshold on IoU
        if new_entry[2] > self.bank[max_sim_idx][2] - 0.1:
            self.bank.pop(max_sim_idx)
            self.bank.append(new_entry)

    def get_memory(self):
        """Returns stacked memory features, pos_enc, and image_embed"""
        if not self.bank:
            return None, None, None

        features = torch.stack([e[0] for e in self.bank], dim=0)
        pos_encs = torch.stack([e[1] for e in self.bank], dim=0)
        image_embeds = torch.stack([e[3] for e in self.bank], dim=0)
        return features, pos_encs, image_embeds

    def __len__(self):
        return len(self.bank)

    def clear(self):
        self.bank.clear()
