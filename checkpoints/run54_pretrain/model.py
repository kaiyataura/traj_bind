import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as checkpoint
from typing import Literal


class DistanceEmbedding(nn.Module):
    centers: torch.Tensor
    def __init__(self, dist_dim: int = 32, r_min: float = 2.0, r_max: float = 15.0):
        super().__init__()
        centers = torch.linspace(r_min, r_max, dist_dim)
        self.register_buffer('centers', centers)
        self.gamma: float = -4.0 * math.log(0.5) / ((centers[1].item() - centers[0].item()) ** 2) # midpoint = 0.5

    def forward(self, coords: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # coords: [B, N, 4, 3]

        c1, c2 = coords[:,:,None,:,None,:], coords[:,None,:,None,:,:] # [B, N, 1, 4, 1, 3], [B, 1, N, 1, 4, 3]
        dists = torch.sqrt(((c1 - c2) ** 2).sum(dim=-1) + 1e-8) # [B, N, N, 4, 4]
        rbfs = torch.exp(((dists.flatten(-2).unsqueeze(-1) - self.centers) ** 2) * -self.gamma).flatten(-2) # [B, N, N, Sd]
        return rbfs, dists[...,3,3]


class AngleEmbedding(nn.Module):
    centers: torch.Tensor
    def __init__(self, angle_dim: int = 8):
        super().__init__()
        centers = torch.linspace(-1.0, 1.0, angle_dim)
        self.register_buffer('centers', centers)
        self.gamma: float = 1.0 / ((centers[1].item() - centers[0].item()) ** 2)

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
    def __init__(self, hidden_dim: int, geom_dim: int, num_heads: int = 8, dropout: float = 0.1):
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

        self.edge_norm = nn.LayerNorm(hidden_dim)
        self.edge_update = nn.Sequential(
            nn.Linear(hidden_dim + geom_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        nn.init.zeros_(self.edge_update[-1].weight)
        nn.init.zeros_(self.edge_update[-1].bias)
        
    def forward(self, node_embed: torch.Tensor, edge_embed: torch.Tensor, geom_embed: torch.Tensor, dir: torch.Tensor, mask: torch.Tensor, frame: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # node_embed: [B, N, D]
        # edge_embed: [B, N, N, D]
        # geom_embed: [B, N, N, Sd + Sa]
        # dir: [B, N, N, 3]
        # mask: [B, N, N]
        # frame: [B, N, 3, 3]
        
        B, N, _ = mask.shape
        
        node_norm = self.norm(node_embed)
        query = self.query_proj(node_norm).view(B, N, self.num_heads, self.head_dim) # [B, N, H, C]
        key = self.key_proj(node_norm).view(B, N, self.num_heads, self.head_dim) # [B, N, H, C]
        scalar = self.scalar_proj(node_norm).view(B, N, self.num_heads, self.head_dim) # [B, N, H, C]
        vector = self.vector_proj(node_norm).view(B, N, self.num_heads, self.head_dim) # [B, N, H, C]

        logits = query.permute(0,2,1,3) @ key.permute(0,2,3,1) / math.sqrt(self.head_dim) # [B, H, N, N]
        bias = self.bias_mlp(edge_embed).permute(0, 3, 1, 2) # [B, H, N, N]
        logits = (logits + bias).masked_fill(~mask.unsqueeze(1), -1e9) # [B, H, N, N]
            
        weights = self.dropout(torch.softmax(logits, dim=-1)) # [B, H, N, N]
        scalar_embed = (weights @ scalar.transpose(1,2)).transpose(1,2).reshape(B, N, -1) # [B, N, D]
        vector_embed = dir.permute(0,3,1,2).unsqueeze(2) * weights.unsqueeze(1) # [B, 3, H, N, N]
        vector_embed = ((vector_embed @ vector.transpose(1,2).unsqueeze(1)).permute(0,3,2,4,1).reshape(B,N,-1,3) @ frame).view(B,N,-1) # [B, N, D * 3]

        node_embed = node_embed + self.node_proj(torch.concat([scalar_embed, vector_embed], dim=-1)) # [B, N, D]
        node_embed = node_embed + self.node_update(node_embed) # [B, N, D]
        
        edge_embed = edge_embed + self.edge_proj(node_embed).unsqueeze(2) * self.edge_gate(node_embed).sigmoid().unsqueeze(1) # [B, N, N, D]
        edge_embed = edge_embed + self.edge_update(torch.concat([self.edge_norm(edge_embed), geom_embed], dim=-1)) # [B, N, N, D]
        
        return node_embed, edge_embed

class SpatialEncoder(nn.Module):
    def __init__(self, hidden_dim: int = 128, res_dim: int = 16, dist_dim: int = 16, angle_dim: int = 8, num_layers: int = 2, num_heads: int = 8, r_max: float = 15.0, dropout: float = 0.1):
        super().__init__()
        self.r_max = r_max
        self.dist_embedding = DistanceEmbedding(dist_dim=dist_dim, r_max=r_max)
        self.angle_embedding = AngleEmbedding(angle_dim=angle_dim)
        self.res_embedding = nn.Embedding(32, res_dim)
        self.geom_dim = 16 * dist_dim + 10 * angle_dim
        
        self.node_proj = nn.Sequential(
            nn.Linear(res_dim + 5, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        self.edge_proj = nn.Sequential(
            nn.Linear(self.geom_dim + 2 * hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        self.attn_layers = nn.ModuleList([
            SpatialAttention(hidden_dim, self.geom_dim, num_heads=num_heads, dropout=dropout) for _ in range(num_layers)
        ])

        # self.sym_proj = nn.Sequential(
        #     nn.Linear(3 * hidden_dim, hidden_dim),
        #     nn.LayerNorm(hidden_dim)
        # )
        
    def forward(self, coords: torch.Tensor, residues: torch.Tensor, props: torch.Tensor, masks: torch.Tensor):
        # coords: [B, N, 4, 3]
        # residues: [B, N]
        # props: [B, N, 5]
        # masks: [B, N]

        B, N = residues.shape
        
        # res_embed = self.res_embedding(residues.long()) # [B, N, R]
        # node_embed = self.node_proj(torch.concat([res_embed, props], dim=-1)) # [B, N, D]

        dist_embed, dist = self.dist_embedding(coords) # [B, N, N, Sd], [B, N, N]
        angle_embed, dir, frame = self.angle_embedding(coords) # [B, N, N, Sa], [B, N, N, 3], [B, N, 3, 3]
        geom_embed = torch.concat([dist_embed, angle_embed], dim=-1) # [B, N, N, Sd + Sa]
        return None, None, geom_embed

        # edge_embed = self.edge_proj(torch.concat([
        #     geom_embed,
        #     node_embed.unsqueeze(2).expand(-1, -1, N, -1),
        #     node_embed.unsqueeze(1).expand(-1, N, -1, -1)
        # ], dim=-1)) # [B, N, N, D]

        # edge_mask = masks.unsqueeze(2) & masks.unsqueeze(1) & (dist < self.r_max) # [B, N, N]

        # for layer in self.attn_layers:
        #     node_embed, edge_embed = layer(node_embed, edge_embed, geom_embed, dir, edge_mask, frame) # [B, N, D], [B, N, N, D]
        
        # edge_embed = self.sym_proj(torch.concat([
        #     edge_embed + edge_embed.transpose(-2,-3),
        #     edge_embed * edge_embed.transpose(-2,-3),
        #     torch.abs(edge_embed - edge_embed.transpose(-2,-3)),
        # ], dim=-1))
        
        # return node_embed, edge_embed, geom_embed


class TeacherModel(nn.Module):
    def __init__(self, hidden_dim: int = 128, **kwargs):
        super().__init__()
        self.spatial_encoder = SpatialEncoder(hidden_dim=hidden_dim, **kwargs)
        geom_dim = self.spatial_encoder.geom_dim

        self.edge_gate = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        nn.init.zeros_(self.edge_gate[-1].weight)
        nn.init.zeros_(self.edge_gate[-1].bias)

        self.node_gate = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        nn.init.zeros_(self.node_gate[-1].weight)
        nn.init.zeros_(self.node_gate[-1].bias)

        self.enthalpy_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim) 
        )
        self.entropy_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim) 
        )
        
        self.enthalpy_head = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.enthalpy_head.weight)
        nn.init.zeros_(self.enthalpy_head.bias)
        
        self.entropy_head = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.entropy_head.weight)
        nn.init.zeros_(self.entropy_head.bias)
        
        self.log_beta = nn.Parameter(torch.zeros(1)) # log(1 / (kB * T))

        self.interaction_head = nn.Sequential(
            nn.Linear(geom_dim, hidden_dim),
            # nn.Linear(hidden_dim + geom_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 4)
        )

    def forward(self, coords: torch.Tensor, residues: torch.Tensor, props: torch.Tensor, masks: torch.Tensor, core_masks: torch.Tensor, mode: Literal['inference', 'pretrain', 'train', 'embed'] = 'inference', chunk_size: int = 2, **kwargs):
        # coords: [B, T, N, 4, 3]
        # residues: [B, N]
        # props: [B, N, 5]
        # masks: [B, N]
        # core_masks: [B, N]

        B, T, N, _, _ = coords.shape

        coords_flat = coords.view(B * T, N, 4, 3) # [B * T, N, 4, 3]
        residues_flat = residues.repeat_interleave(T, 0) # [B * T, N]
        props_flat = props.repeat_interleave(T, 0) # [B * T, N, 5]
        masks_flat = masks.repeat_interleave(T, 0) # [B * T, N]
        core_masks_flat = core_masks.repeat_interleave(T, 0) # [B * T, N]

        node_embed_flat, edge_embed_flat, geom_embed_flat = self.spatial_encoder(coords_flat, residues_flat, props_flat, masks_flat) # [B * T, N, D], [B * T, N, N, D], [B * T, N, N, Sd + Sa]

        interactions = self.interaction_head(geom_embed_flat).view(B, T, N, N, -1) # [B, T, N, N, 4]
        # interactions = self.interaction_head(torch.concat([edge_embed_flat, geom_embed_flat], dim=-1)).view(B, T, N, N, -1) # [B, T, N, N, 4]
        # return {'interactions': interactions, 'node_embed': node_embed_flat.view(B, T, N, -1), 'edge_embed': edge_embed_flat.view(B, T, N, N, -1), 'geom_embed': geom_embed_flat.view(B, T, N, N, -1)}
        return {'interactions': interactions}

        
        def _ckpt_forward(coord_chunk, residue_chunk, prop_chunk, mask_chunk, core_mask_chunk):
            C = mask_chunk.shape[0]
            node_embed_chunk, edge_embed_chunk = self.spatial_encoder(coord_chunk, residue_chunk, prop_chunk, mask_chunk) # [C, N, D], [C, N, N, D]
            
            enthalpy_emb = self.enthalpy_mlp(edge_embed_chunk) # [C, N, N, D]
            enthalpies = self.enthalpy_head(enthalpy_emb) # [C, N, N, 1]

            edge_mask = core_mask_chunk[:,None,:,None] & core_mask_chunk[:,:,None,None] # [C, N, N, 1]
            edge_weight = self.edge_gate(edge_embed_chunk).sigmoid() * edge_mask # [C, N, N, 1]
            edge_graph_embeds = (edge_weight * edge_embed_chunk).sum((1,2)) / 100. # [C, D]
            
            interactions = self.interaction_head(edge_embed_chunk) if mode == 'train' else None # [C, N, N, 4]
            return node_embed_chunk, edge_graph_embeds, interactions

        all_node_graph_embeds, all_edge_graph_embeds, all_interactions = [], [], []
        for i in range(0, B * T, chunk_size):
            C = min(chunk_size, B * T - i)
            coord_chunk = coords_flat[i:i+C] # [C, N, 4, 3]
            residue_chunk = residues_flat[i:i+C] # [C, N]
            prop_chunk = props_flat[i:i+C] # [C, N, 5]
            mask_chunk = masks_flat[i:i+C] # [C, N]
            core_mask_chunk = core_masks_flat[i:i+C] # [C, N]
            
            if self.training: node_embed_chunk, edge_graph_embeds, interactions = checkpoint(_ckpt_forward, coord_chunk, residue_chunk, prop_chunk, mask_chunk, core_mask_chunk, use_reentrant=False) # [C, D], [C, N, N, 4] # type: ignore
            else: node_embed_chunk, edge_graph_embeds, interactions = _ckpt_forward(coord_chunk, residue_chunk, prop_chunk, mask_chunk, core_mask_chunk) # [C, D], [C, N, N, 4]
            
            node_mask = core_mask_chunk[:,:,None] # [C, N, 1]
            node_weight = self.node_gate(node_embed_chunk).sigmoid() * node_mask # [C, N, 1]
            node_graph_embeds = (node_weight * node_embed_chunk).sum(1) / 10. # [C, D]
            
            all_node_graph_embeds.append(node_graph_embeds)
            all_edge_graph_embeds.append(edge_graph_embeds)
            all_interactions.append(interactions)

        node_graph_embeds = torch.concat(all_node_graph_embeds, dim=0).view(B, T, -1) # [B, T, D]
        edge_graph_embeds = torch.concat(all_edge_graph_embeds, dim=0).view(B, T, -1) # [B, T, D]
        interactions = torch.concat(all_interactions, dim=0).view(B, T, N, N, 4) if mode == 'train' else None # [B, T, N, N, 4]
            
        enthalpy_embeds = self.enthalpy_mlp(edge_graph_embeds) # [B, T, D]        
        energies = self.enthalpy_head(enthalpy_embeds).squeeze(2) # [B, T]
        probs = F.softmax(-torch.exp(self.log_beta) * energies, dim=1).view(B, T, 1) # [B, T, 1]
        enthalpy_embed = (enthalpy_embeds * probs).sum(1) # [B, D]
        
        entropy_embeds = self.entropy_mlp(node_graph_embeds) # [B, T, D]
        mean_entropy_embed = (entropy_embeds * probs).sum(1, keepdim=True) # [B, 1, D]
        var_entropy_embed = (((entropy_embeds - mean_entropy_embed) ** 2) * probs).sum(1) # [B, D]
        entropy_embed = 0.5 * torch.log(var_entropy_embed + 1e-8) # [B, D]

        affinities = (self.enthalpy_head(enthalpy_embed) + self.entropy_head(entropy_embed)).squeeze(1) # [B]
        
        if mode == 'inference': return affinities
        if mode == 'train': return {'affinities': affinities, 'interactions': interactions}
        
        D = entropy_embed.shape[1]
        node_embed = torch.zeros(B, N, D, device=coords.device, dtype=coords.dtype) # [B, N, D]
        edge_embed = torch.zeros(B, N, N, D, device=coords.device, dtype=coords.dtype) # [B, N, N, D]
        
        indices = torch.arange(B, device=coords.device).repeat_interleave(T) # [B * T]
        probs_flat = probs.view(B * T) # [B * T]
        
        for i in range(0, B * T, chunk_size):
            C = min(chunk_size, B * T - i)
            coord_chunk = coords_flat[i:i+C] # [C, N, 4, 3]
            residue_chunk = residues_flat[i:i+C] # [C, N]
            prop_chunk = props_flat[i:i+C] # [C, N, 5]
            mask_chunk = masks_flat[i:i+C] # [C, N]
            index_chunk = indices[i:i+C] # [C]
            
            node_embed_chunk, edge_embed_chunk = self.spatial_encoder(coord_chunk, residue_chunk, prop_chunk, mask_chunk) # [C, N, D], [C, N, N, D]
            node_embed.index_add_(0, index_chunk, node_embed_chunk * probs_flat[i:i+C].view(C, 1, 1)) # [B, N, D]
            edge_embed.index_add_(0, index_chunk, edge_embed_chunk * probs_flat[i:i+C].view(C, 1, 1, 1)) # [B, N, N, D]

        return {
            'node_embed': node_embed,
            'edge_embed': edge_embed,
            'enthalpy_embed': enthalpy_embed,
            'entropy_embed': entropy_embed
        }

class StudentModel(nn.Module):
    def __init__(self, hidden_dim: int = 128, **kwargs):
        super().__init__()
        self.spatial_encoder = SpatialEncoder(hidden_dim=hidden_dim, **kwargs)

        self.edge_gate = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        nn.init.zeros_(self.edge_gate[-1].weight)
        nn.init.zeros_(self.edge_gate[-1].bias)

        self.node_gate = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        nn.init.zeros_(self.node_gate[-1].weight)
        nn.init.zeros_(self.node_gate[-1].bias)

        self.enthalpy_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim) 
        )

        self.entropy_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim) 
        )

        self.enthalpy_head = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.enthalpy_head.weight)
        nn.init.zeros_(self.enthalpy_head.bias)
        
        self.entropy_head = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.entropy_head.weight)
        nn.init.zeros_(self.entropy_head.bias)

    def forward(self, coords: torch.Tensor, residues: torch.Tensor, props: torch.Tensor, masks: torch.Tensor, core_masks: torch.Tensor, mode: Literal['inference', 'train'] = 'inference', **kwargs):
        # coords: [B, N, 4, 3] or [B, 1, N, 4, 3]
        # residues: [B, N]
        # props: [B, N, 5]
        # masks: [B, N]
        # core_masks: [B, N]

        if coords.ndim == 5: coords = coords.squeeze(1) # [B, N, 4, 3]
        B, N, _, _ = coords.shape

        if self.training:
            coords = coords + 0.01 * torch.randn_like(coords) * masks.view(B, N, 1, 1) # [B, N, 4, 3]
            
        node_embed, edge_embed = self.spatial_encoder(coords, residues, props, masks) # [B, N, D], [B, N, N, D]
        
        edge_mask = core_masks[:,None,:,None] & core_masks[:,:,None,None] # [B, N, N, 1]
        edge_weight = self.edge_gate(edge_embed).sigmoid() * edge_mask # [B, N, N, 1]
        enthalpy_embed = self.enthalpy_mlp((edge_weight * edge_embed).sum((1,2)) / 100.) # [B, D]
        
        node_mask = core_masks[:,:,None] # [B, N, 1]
        node_weight = self.node_gate(node_embed).sigmoid() * node_mask # [B, N, 1]
        entropy_embed = self.entropy_mlp((node_weight * node_embed).sum(1) / 10.) # [B, D]

        affinities = (self.enthalpy_head(enthalpy_embed) + self.entropy_head(entropy_embed)).squeeze(1) # [B]
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
        interaction_targets = interaction_targets.clamp(min=-10.0, max=10.0)
        stds = torch.tensor([self.vdw_std, self.elec_std, self.solv_std, self.hbond_std], device=interactions.device) # [4]
        losses = F.huber_loss(interactions / stds, interaction_targets / stds, reduction='none', delta=1.0) # [B, T, N, N, 4]

        dists = (coords[:,:,None,:,3] - coords[:,:,:,None,3]).norm(dim=-1, keepdim=True) # [B, T, N, N, 1]
        mask = core_masks[:,None,None,:,None] & core_masks[:,None,:,None,None] & (dists < 15.0) # [B, T, N, N, 1]
        active_mask = (interaction_targets.abs() > 0.01) & mask # [B, T, N, N, 4]
        inactive_mask = (interaction_targets.abs() <= 0.01) & mask # [B, T, N, N, 4]

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
        affinity_loss = ((affinities - affinity_targets) ** 2).mean() * self.affinity_weight # []

        if affinities.shape[0] > 1:
            pred_diff = affinities.unsqueeze(1) - affinities.unsqueeze(0) # [B, B]
            label_diff = affinity_targets.unsqueeze(1) - affinity_targets.unsqueeze(0) # [B, B]
            mask = (label_diff > 0).to(affinities.dtype) # [B, B]
            rank_loss = (-F.logsigmoid(pred_diff) * mask).sum() / (mask.sum() + 2e-5) * self.rank_weight # []
        else: rank_loss = torch.tensor(0.0, device=affinities.device) # []

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
