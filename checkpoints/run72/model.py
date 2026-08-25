import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as checkpoint
from typing import Literal
from collections import defaultdict


class DistanceEmbedding(nn.Module):
    centers: torch.Tensor
    def __init__(self, dist_dim: int = 16, r_min: float = 0.0, r_max: float = 15.0):
        super().__init__()
        centers = torch.linspace(r_min, r_max, dist_dim)
        self.register_buffer('centers', centers)
        self.gamma: float = -4.0 * math.log(0.5) / ((centers[1].item() - centers[0].item()) ** 2) # midpoint = 0.5

    def forward(self, coords: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # coords: [B, N, 4, 3]
        B, N, _, _ = coords.shape
        flat_coords = coords.view(B, N * 4, 3)
        dists = torch.cdist(flat_coords, flat_coords).view(B, N, 4, N, 4).permute(0,1,3,2,4) # [B, N, N, 4, 4]
        rbfs = torch.exp(((dists.flatten(-2).unsqueeze(-1) - self.centers) ** 2) * -self.gamma).flatten(-2) # [B, N, N, Sd]
        return rbfs, dists[...,3,3]

class AngleEmbedding(nn.Module):
    centers: torch.Tensor
    def __init__(self, angle_dim: int = 8):
        super().__init__()
        centers = torch.linspace(-1.0, 1.0, angle_dim)
        self.register_buffer('centers', centers)
        self.gamma: float = -4.0 * math.log(0.5) / ((centers[1].item() - centers[0].item()) ** 2) # midpoint = 0.5

    def forward(self, coords: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # coords: [B, N, 4, 3]
        
        ca, c, n, sc = coords.unbind(2) # [B, N, 3], [B, N, 3], [B, N, 3], [B, N, 3]
        
        v1 = F.normalize(c - ca, dim=-1) # [B, N, 3]
        v2 = F.normalize(n - ca, dim=-1) # [B, N, 3]
        v3 = F.normalize(sc - ca, dim=-1) # [B, N, 3]
        v4 = F.normalize(torch.cross(v1, v2, dim=-1), dim=-1) # [B, N, 3]
        
        frame = torch.stack([v1, v2, v4], dim=-1) # [B, N, 3, 3]
        
        diff = ca[:, None, :, :] - ca[:, :, None, :] # [B, N, N, 3]
        dir = diff / (diff.norm(dim=-1, keepdim=True) + 1e-8) # [B, N, N, 3]
        
        angs = torch.stack([
            (v1.unsqueeze(2) * dir).sum(dim=-1), # C_i • dir_ij
            (v2.unsqueeze(2) * dir).sum(dim=-1), # N_i • dir_ij
            (v3.unsqueeze(2) * dir).sum(dim=-1), # SC_i • dir_ij
            (v1.unsqueeze(1) * dir).sum(dim=-1), # C_j • dir_ij
            (v2.unsqueeze(1) * dir).sum(dim=-1), # N_j • dir_ij
            (v3.unsqueeze(1) * dir).sum(dim=-1), # SC_j • dir_ij
            (v1.unsqueeze(2) * v1.unsqueeze(1)).sum(dim=-1), # C_i • C_j
            (v2.unsqueeze(2) * v2.unsqueeze(1)).sum(dim=-1), # N_i • N_j
            (v3.unsqueeze(2) * v3.unsqueeze(1)).sum(dim=-1), # SC_i • SC_j
            (v4.unsqueeze(2) * v4.unsqueeze(1)).sum(dim=-1), # Plane_i • Plane_j
        ], dim=-1) # [B, N, N, 10]
        
        rbfs = torch.exp(((angs.unsqueeze(-1) - self.centers) ** 2) * -self.gamma).flatten(-2) # [B, N, N, Sa]
        return rbfs, dir, frame


class SpatialAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.norm = nn.LayerNorm(hidden_dim)
        self.query_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.scalar_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.vector_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.bias_mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_heads, bias=False)
        )
        nn.init.zeros_(self.bias_mlp[-1].weight)
        
        self.dropout = nn.Dropout(dropout)

        self.node_proj = nn.Linear(hidden_dim * 4, hidden_dim)
        nn.init.zeros_(self.node_proj.weight)
        nn.init.zeros_(self.node_proj.bias)
        
        self.node_update = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        nn.init.zeros_(self.node_update[-1].weight)
        nn.init.zeros_(self.node_update[-1].bias)
        
        self.edge_proj = nn.Linear(hidden_dim, hidden_dim)
        nn.init.zeros_(self.edge_proj.weight)
        nn.init.zeros_(self.edge_proj.bias)
        self.edge_gate = nn.Linear(hidden_dim, hidden_dim)
        nn.init.zeros_(self.edge_gate.weight)
        nn.init.zeros_(self.edge_gate.bias)

        self.edge_update = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        nn.init.zeros_(self.edge_update[-1].weight)
        nn.init.zeros_(self.edge_update[-1].bias)
        
    def forward(self, node_embed: torch.Tensor, edge_embed: torch.Tensor, dir: torch.Tensor, edge_mask: torch.Tensor, frame: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # node_embed: [B, N, D]
        # edge_embed: [B, N, N, D]
        # dir: [B, N, N, 3]
        # edge_mask: [B, N, N]
        # frame: [B, N, 3, 3]
        
        B, N, _ = edge_mask.shape
        
        node_norm = self.norm(node_embed)
        query = self.query_proj(node_norm).view(B, N, self.num_heads, self.head_dim) # [B, N, H, C]
        key = self.key_proj(node_norm).view(B, N, self.num_heads, self.head_dim) # [B, N, H, C]
        scalar = self.scalar_proj(node_norm).view(B, N, self.num_heads, self.head_dim) # [B, N, H, C]
        vector = self.vector_proj(node_norm).view(B, N, self.num_heads, self.head_dim) # [B, N, H, C]

        logits = query.permute(0,2,1,3) @ key.permute(0,2,3,1) / math.sqrt(self.head_dim) # [B, H, N, N]
        bias = self.bias_mlp(edge_embed).permute(0, 3, 1, 2) # [B, H, N, N]
        logits = (logits + bias).masked_fill(~edge_mask.unsqueeze(1), -1e9) # [B, H, N, N]
            
        weights = self.dropout(torch.softmax(logits, dim=-1)) # [B, H, N, N]
        scalar_embed = (weights @ scalar.transpose(1,2)).transpose(1,2).reshape(B, N, -1) # [B, N, D]
        vector_embed = dir.permute(0,3,1,2).unsqueeze(2) * weights.unsqueeze(1) # [B, 3, H, N, N]
        vector_embed = ((vector_embed @ vector.transpose(1,2).unsqueeze(1)).permute(0,3,2,4,1).reshape(B,N,-1,3) @ frame).view(B,N,-1) # [B, N, D * 3]

        node_embed = node_embed + self.node_proj(torch.concat([scalar_embed, vector_embed], dim=-1)) # [B, N, D]
        node_embed = node_embed + self.node_update(node_embed) # [B, N, D]
        
        edge_embed = edge_embed + self.edge_proj(node_embed).unsqueeze(2) * self.edge_gate(node_embed).sigmoid().unsqueeze(1) # [B, N, N, D]
        edge_embed = edge_embed + self.edge_update(edge_embed) # [B, N, N, D]
        
        return node_embed, edge_embed

class SpatialEncoder(nn.Module):
    def __init__(self, hidden_dim: int = 128, res_dim: int = 16, dist_dim: int = 16, angle_dim: int = 8, num_layers: int = 2, num_heads: int = 8, r_max: float = 15.0, dropout: float = 0.1):
        super().__init__()
        self.r_max = r_max
        self.dist_embedding = DistanceEmbedding(dist_dim=dist_dim, r_max=r_max)
        self.angle_embedding = AngleEmbedding(angle_dim=angle_dim)
        self.res_embedding = nn.Embedding(32, res_dim)
        
        self.geom_proj = nn.Sequential(
            nn.Linear(16 * dist_dim + 10 * angle_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        self.node_proj = nn.Sequential(
            nn.Linear(res_dim + 5, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        self.edge_proj = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        self.attn_layers = nn.ModuleList([
            SpatialAttention(hidden_dim, num_heads=num_heads, dropout=dropout) for _ in range(num_layers)
        ])

        self.sym_proj = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
    def forward(self, coords: torch.Tensor, residues: torch.Tensor, props: torch.Tensor, valid_masks: torch.Tensor):
        # coords: [B, N, 4, 3]
        # residues: [B, N]
        # props: [B, N, 5]
        # valid_masks: [B, N]

        B, N = residues.shape
        
        res_embed = self.res_embedding(residues.long()) # [B, N, R]
        node_embed = self.node_proj(torch.concat([res_embed, props], dim=-1)) # [B, N, D]

        dist_embed, dist = self.dist_embedding(coords) # [B, N, N, Sd], [B, N, N]
        angle_embed, dir, frame = self.angle_embedding(coords) # [B, N, N, Sa], [B, N, N, 3], [B, N, 3, 3]
        geom_embed = self.geom_proj(torch.concat([dist_embed, angle_embed], dim=-1)) # [B, N, N, D]

        edge_embed = self.edge_proj(torch.concat([
            geom_embed,
            node_embed.unsqueeze(2).expand(-1, -1, N, -1),
            node_embed.unsqueeze(1).expand(-1, N, -1, -1)
        ], dim=-1)) # [B, N, N, D]

        edge_mask = valid_masks.unsqueeze(2) & valid_masks.unsqueeze(1) & (dist <= self.r_max) # [B, N, N]

        for layer in self.attn_layers:
            node_embed, edge_embed = layer(node_embed, edge_embed, dir, edge_mask, frame) # [B, N, D], [B, N, N, D]
        
        edge_embed = self.sym_proj(torch.concat([
            edge_embed + edge_embed.transpose(-2,-3),
            edge_embed * edge_embed.transpose(-2,-3),
            torch.abs(edge_embed - edge_embed.transpose(-2,-3)),
        ], dim=-1))
        
        return node_embed, edge_embed, geom_embed, dist


class TeacherModel(nn.Module):
    def __init__(self, hidden_dim: int = 128, **kwargs):
        super().__init__()
        self.spatial_encoder = SpatialEncoder(hidden_dim=hidden_dim, **kwargs)

        self.enthalpy_gate = nn.Sequential(
            nn.LayerNorm(2 * hidden_dim),
            nn.Linear(2 * hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        nn.init.zeros_(self.enthalpy_gate[-1].weight)
        nn.init.constant_(self.enthalpy_gate[-1].bias, -1.0)

        self.entropy_gate = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        nn.init.zeros_(self.entropy_gate[-1].weight)
        nn.init.constant_(self.entropy_gate[-1].bias, -1.0)

        self.enthalpy_mlp = nn.Sequential(
            nn.LayerNorm(2 * hidden_dim),
            nn.Linear(2 * hidden_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(hidden_dim * 4, hidden_dim) 
        )
        self.entropy_mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim) 
        )
        
        self.enthalpy_head = nn.Linear(hidden_dim, 1, bias=False)
        self.entropy_head = nn.Linear(hidden_dim, 1, bias=False)
        self.interaction_head = nn.Linear(hidden_dim, 4, bias=False)
        
        self.log_beta = nn.Parameter(torch.tensor([4.0])) # log(1 / (kB * T))

    def forward(self, coords: torch.Tensor, residues: torch.Tensor, props: torch.Tensor, valid_masks: torch.Tensor, core_masks: torch.Tensor, ligand_masks: torch.Tensor, mode: Literal['inference', 'pretrain', 'train', 'embed'] = 'inference', chunk_size: int = 1, **kwargs):
        # coords: [B, T, N, 4, 3]
        # residues: [B, N]
        # props: [B, N, 5]
        # valid_masks: [B, N]
        # core_masks: [B, N]
        # ligand_masks: [B, N]

        B, T, N, _, _ = coords.shape

        coords_flat = coords.view(B * T, N, 4, 3)
        residues_flat = residues.repeat_interleave(T, 0)
        props_flat = props.repeat_interleave(T, 0)
        valid_masks_flat = valid_masks.repeat_interleave(T, 0)
        core_masks_flat = core_masks.repeat_interleave(T, 0)
        ligand_masks_flat = ligand_masks.repeat_interleave(T, 0)

        def process_chunk(coords_chunk, residues_chunk, props_chunk, valid_masks_chunk, core_masks_chunk, ligand_masks_chunk):
            node_embeds_chunk, edge_embeds_chunk, geom_embeds_chunk, dists_chunk = self.spatial_encoder(coords_chunk, residues_chunk, props_chunk, valid_masks_chunk) # [C, N, D], [C, N, N, D], [C, N, N, D], [C, N, N]
            
            edge_embeds_chunk = torch.concat([edge_embeds_chunk, geom_embeds_chunk], -1) # [C, N, N, 2 * D]
            enthalpy_embeds_chunk = self.enthalpy_mlp(edge_embeds_chunk) # [C, N, N, D]
            interactions_chunk = self.interaction_head(enthalpy_embeds_chunk) if mode == 'pretrain' or mode == 'train' else None # [C, N, N, 4]
            if mode == 'pretrain': return {'interactions': interactions_chunk}
            edge_masks_chunk = core_masks_chunk[:,None,:] & core_masks_chunk[:,:,None] & (dists_chunk <= 15.0) & (ligand_masks_chunk[:,None,:] != ligand_masks_chunk[:,:,None]) # [C, N, N]
            enthalpies_chunk = (self.enthalpy_head(enthalpy_embeds_chunk) * self.enthalpy_gate(edge_embeds_chunk).sigmoid() * edge_masks_chunk[:,:,:,None]).sum((-2,-3)) # [C, 1]
            
            return {'node_embeds': node_embeds_chunk, 'enthalpies': enthalpies_chunk, **({'interactions': interactions_chunk} if mode == 'train' else {}), **({'enthalpy_embeds': enthalpy_embeds_chunk} if mode == 'embed' else {})}

        out = defaultdict(list)
        for start in range(0, B * T, chunk_size):
            end = min(start + chunk_size, B * T)
            coords_chunk = coords_flat[start:end]
            residues_chunk = residues_flat[start:end]
            props_chunk = props_flat[start:end]
            valid_masks_chunk = valid_masks_flat[start:end]
            core_masks_chunk = core_masks_flat[start:end]
            ligand_masks_chunk = ligand_masks_flat[start:end]

            if self.training: chunk_out = checkpoint(process_chunk, coords_chunk, residues_chunk, props_chunk, valid_masks_chunk, core_masks_chunk, ligand_masks_chunk, use_reentrant=False)
            else: chunk_out = process_chunk(coords_chunk, residues_chunk, props_chunk, valid_masks_chunk, core_masks_chunk, ligand_masks_chunk)
            
            for k, v in chunk_out.items(): out[k].append(v) # type: ignore
        out = {k: torch.concat(v, dim=0).unflatten(0, (B,T)) for k, v in out.items()}

        interactions = out.get('interactions') # [B, T, N, N, 4]
        if mode == 'pretrain': return {'interactions': interactions}
        
        enthalpies = out['enthalpies'] # [B, T, 1]
        probs = F.softmax(torch.exp(self.log_beta) * enthalpies, dim=-2) # [B, T, 1]
        enthalpy = (enthalpies * probs).sum(dim=1) # [B, 1]
        
        node_embeds = out['node_embeds'] # [B, T, N, D]
        entropy_embeds = self.entropy_mlp(node_embeds) # [B, T, N, D]
        entropy_embed = ((probs[:,:,None,:,None] * probs[:,None,:,:,None]) * torch.abs(entropy_embeds[:,:,None,:,:] - entropy_embeds[:,None,:,:,:])).sum((-3,-4)) # [B, N, D]
        node_embed = (node_embeds * probs[:,:,None,:]).sum(-3) # [B, N, D]
        entropy = (self.entropy_head(entropy_embed) * self.entropy_gate(node_embed).sigmoid() * core_masks[:,:,None]).sum(-2) # [B, 1]

        affinity = (enthalpy + entropy).squeeze(-1) # [B]

        if mode == 'inference': return affinity
        if mode == 'train': return {'affinities': affinity, 'interactions': interactions}
        
        enthalpy_embeds = out['enthalpy_embeds'] # [B, T, N, N, D]
        enthalpy_embed = (enthalpy_embeds * probs[:,:,None,None,:]).sum(-4) # [B, N, N, D]
        return {'enthalpy_embed': enthalpy_embed, 'entropy_embed': entropy_embed}


class StudentModel(nn.Module):
    def __init__(self, hidden_dim: int = 128, noise: float = 0.01, **kwargs):
        super().__init__()
        self.noise = noise
        self.spatial_encoder = SpatialEncoder(hidden_dim=hidden_dim, **kwargs)

        self.enthalpy_gate = nn.Sequential(
            nn.LayerNorm(2 * hidden_dim),
            nn.Linear(2 * hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        nn.init.zeros_(self.enthalpy_gate[-1].weight)
        nn.init.constant_(self.enthalpy_gate[-1].bias, -1.0)

        self.entropy_gate = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        nn.init.zeros_(self.entropy_gate[-1].weight)
        nn.init.constant_(self.entropy_gate[-1].bias, -1.0)

        self.enthalpy_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(hidden_dim * 4, hidden_dim) 
        )

        self.entropy_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim) 
        )

        self.enthalpy_head = nn.Linear(hidden_dim, 1, bias=False)
        self.entropy_head = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, coords: torch.Tensor, residues: torch.Tensor, props: torch.Tensor, valid_masks: torch.Tensor, core_masks: torch.Tensor, ligand_masks: torch.Tensor, mode: Literal['inference', 'train'] = 'inference', **kwargs):
        # coords: [B, N, 4, 3] or [B, 1, N, 4, 3]
        # residues: [B, N]
        # props: [B, N, 5]
        # valid_masks: [B, N]
        # core_masks: [B, N]
        # ligand_masks: [B, N]

        if coords.ndim == 5: coords = coords.squeeze(1) # [B, N, 4, 3]
        B, N, _, _ = coords.shape

        if self.training:
            coords = coords + self.noise * torch.randn_like(coords) * valid_masks.view(B, N, 1, 1) # [B, N, 4, 3]
            
        node_embed, edge_embed, geom_embed, dist = self.spatial_encoder(coords, residues, props, valid_masks) # [B, N, D], [B, N, N, D], [B, N, N, D]
        edge_embed = torch.concat([edge_embed, geom_embed], -1) # [B, N, N, 2 * D]
        enthalpy_embed = self.enthalpy_mlp(edge_embed) # [B, N, D]
        entropy_embed = self.entropy_mlp(node_embed) # [B, N, D]
        edge_mask = core_masks[:,None,:,None] & core_masks[:,:,None,None] & (dist <= 15.0) & (ligand_masks[:,None,:,None] != ligand_masks[:,:,None,None])
        enthalpies = (self.enthalpy_head(enthalpy_embed) * self.enthalpy_gate(edge_embed).sigmoid() * edge_mask).sum(dim=(1,2,3)) # [B]
        entropies = (self.entropy_head(entropy_embed) * self.entropy_gate(node_embed).sigmoid() * core_masks[:,:,None]).sum(dim=(1,2)) # [B]

        affinities = enthalpies + entropies # [B]
        if mode == 'inference': return affinities
        
        return {
            'affinities': affinities, 
            'node_embed': node_embed,
            'edge_embed': edge_embed,
            'enthalpy_embed': enthalpy_embed,
            'entropy_embed': entropy_embed
        }


class InteractionLoss(nn.Module):
    def __init__(self, vdw_std: float = 1.0, elec_std: float = 1.0, solv_std: float = 1.0, hbond_std: float = 1.0, active_weight: float = 1.0, inactive_weight: float = 0.01):
        super().__init__()
        self.vdw_std = vdw_std
        self.elec_std = elec_std
        self.solv_std = solv_std
        self.hbond_std = hbond_std
        self.active_weight = active_weight
        self.inactive_weight = inactive_weight

    def forward(self, interactions, interaction_targets, coords, core_masks, **kwargs):
        B, T, N, _, _ = interactions.shape
        interaction_targets = torch.where(interaction_targets > 10.0, 10.0 + torch.log(1 + interaction_targets.clamp(min=10.0) - 10.0), interaction_targets)
        stds = torch.tensor([self.vdw_std, self.elec_std, self.solv_std, self.hbond_std], device=interactions.device) # [4]
        losses = F.huber_loss(interactions / stds, interaction_targets / stds, reduction='none', delta=1.0) # [B, T, N, N, 4]

        dists = (coords[:,:,None,:,3] - coords[:,:,:,None,3]).norm(dim=-1, keepdim=True) # [B, T, N, N, 1]
        edge_mask = core_masks[:,None,None,:,None] & core_masks[:,None,:,None,None] & (dists <= 15.0) # [B, T, N, N, 1]
        active_mask = (interaction_targets.abs() > 0.01) & edge_mask # [B, T, N, N, 4]
        inactive_mask = (interaction_targets.abs() <= 0.01) & edge_mask # [B, T, N, N, 4]

        active_loss = self.active_weight * (losses * active_mask).sum((0,1,2,3)) / (active_mask.sum((0,1,2,3)) + 1e-8) # [4]
        inactive_loss = self.inactive_weight * (losses * inactive_mask).sum((0,1,2,3)) / (inactive_mask.sum((0,1,2,3)) + 1e-8) # [4]

        loss = active_loss.sum() + inactive_loss.sum() # []
        return loss, {'vdw': active_loss[0].item(), 'elec': active_loss[1].item(), 'solv': active_loss[2].item(), 'hbond': active_loss[3].item(), 'inactive': inactive_loss.sum().item()}

class AffinityLoss(nn.Module):
    def __init__(self, affinity_weight: float = 0.1, rank_weight: float = 1.0):
        super().__init__()
        self.affinity_weight = affinity_weight
        self.rank_weight = rank_weight
        
    def forward(self, affinities, affinity_targets, **kwargs):
        affinity_loss = F.smooth_l1_loss(affinities, affinity_targets) * self.affinity_weight # []
        
        if len(affinities) > 1:
            r, c = torch.triu_indices(len(affinities), len(affinities), offset=1, device=affinities.device) # [B * (B - 1) / 2], [B * (B - 1) / 2]
            pred_diff = affinities[r] - affinities[c] # [B * (B - 1) / 2]
            target_diff = affinity_targets[r] - affinity_targets[c] # [B * (B - 1) / 2]
            rank_loss = F.binary_cross_entropy_with_logits(pred_diff, torch.sigmoid(target_diff)) * self.rank_weight # []
        else: rank_loss = torch.tensor(0., device=affinities.device, dtype=affinities.dtype)

        loss = affinity_loss + rank_loss # []
        return loss, {'affinity': affinity_loss.item(), 'rank': rank_loss.item()}

class DistillationLoss(nn.Module):
    def __init__(self, node_weight: float = 0.1, edge_weight: float = 0.1, enthalpy_weight: float = 1.0, entropy_weight: float = 1.0):
        super().__init__()
        self.node_weight = node_weight
        self.edge_weight = edge_weight
        self.enthalpy_weight = enthalpy_weight
        self.entropy_weight = entropy_weight

    def forward(self, node_embed, edge_embed, enthalpy_embed, entropy_embed, node_embed_targets, edge_embed_targets, enthalpy_embed_targets, entropy_embed_targets, **kwargs):
        loss_node = F.mse_loss(node_embed, node_embed_targets) * self.node_weight # []
        loss_edge = F.mse_loss(edge_embed, edge_embed_targets) * self.edge_weight # []
        loss_enthalpy = F.mse_loss(enthalpy_embed, enthalpy_embed_targets) * self.enthalpy_weight # []
        loss_entropy = F.mse_loss(entropy_embed, entropy_embed_targets) * self.entropy_weight # []
        
        loss = loss_node + loss_edge + loss_enthalpy + loss_entropy # []
        return loss, {'node': loss_node.item(), 'edge': loss_edge.item(), 'enthalpy': loss_enthalpy.item(), 'entropy': loss_entropy.item()}

class JointLoss(nn.Module):
    def __init__(self, loss_fns: list[tuple[nn.Module, float]]):
        super().__init__()
        self.loss_fns = nn.ModuleList([fn for fn, w in loss_fns])
        self.weights = [w for fn, w in loss_fns]

    def forward(self, **kwargs):
        total_loss, total_dict = 0.0, {}
        for loss_fn, weight in zip(self.loss_fns, self.weights):
            loss, loss_dict = loss_fn(**kwargs)
            total_loss = total_loss + loss * weight
            total_dict.update({k: l * weight for k, l in loss_dict.items()})
        return total_loss, total_dict
