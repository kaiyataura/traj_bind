import os
import shutil
import warnings
from collections import defaultdict
from typing import Literal
import time, datetime
import random
import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import *
from dataset import *


def save_checkpoint(path, model=None, optimizer=None, scheduler=None, epoch=1, best_loss=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {
        **({'model_state_dict': model.state_dict()} if model else {}),
        **({'optimizer_state_dict': optimizer.state_dict()} if optimizer else {}),
        **({'scheduler_state_dict': scheduler.state_dict()} if scheduler else {}),
        'torch_rng_state': torch.get_rng_state(),
        'numpy_rng_state': np.random.get_state(),
        'random_rng_state': random.getstate(),
        'epoch': epoch,
        **({'best_loss': best_loss} if best_loss is not None else {}),
    }
    torch.save(checkpoint, path)

def load_checkpoint(path, model=None, optimizer=None, scheduler=None):
    if not os.path.exists(path):
        warnings.warn(f"[\u26A0] WARNING: Checkpoint '{path}' not found. Starting from scratch.")
        return 0, None
    
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    
    if 'model_state_dict' in checkpoint and model: model.load_state_dict(checkpoint['model_state_dict'])
    if 'optimizer_state_dict' in checkpoint and optimizer: optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if 'scheduler_state_dict' in checkpoint and scheduler: scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    if 'torch_rng_state' in checkpoint: torch.set_rng_state(checkpoint['torch_rng_state'])
    if 'numpy_rng_state' in checkpoint: np.random.set_state(checkpoint['numpy_rng_state'])
    if 'random_rng_state' in checkpoint: random.setstate(checkpoint['random_rng_state'])
        
    epoch = checkpoint.get('epoch', 0)
    best_loss = checkpoint.get('best_loss', None)
    
    return epoch, best_loss


def param_stats(model: torch.nn.Module) -> str:
    lines = ["\n" + "="*122]
    lines.append(f"{'Layer Name':<70} | {'Param Norm':<10} | {'Param Mean':<10} | {'Grad Norm':<10} | {'Grad Mean':<10}")
    lines.append("-"*122)
    
    for name, param in model.named_parameters():
        p_norm = f"{param.norm().item():<10.4f}"
        p_mean = f"{param.mean().item():< 10.2e}"
        
        if param.grad is not None:
            g = param.grad
            g_norm = f"{g.norm().item():<10.4f}"
            g_mean = f"{g.mean().item():< 10.2e}"
        else:
            g_norm = f"{'None':<10}"
            g_mean = f"{'-':<10}"
            
        lines.append(f"{name:<70} | {p_norm} | {p_mean} | {g_norm} | {g_mean}")
            
    lines.append("="*122 + "\n")
    return "\n".join(lines)


def embed_dataset(data_dir, embeds_file, model, num_frames, device, dtype, chunk_size):
    dataset = TrajectoryDataset(data_dir, num_frames=num_frames)
    dataloader = DataLoader(
        dataset=dataset,
        collate_fn=TrajectoryDataset.collate_fn, 
        batch_size=8, 
        shuffle=False,
        num_workers=4
    )

    embeds = {
        'node_embed_targets': {},
        'edge_embed_targets': {},
        'enthalpy_embed_targets': {},
        'entropy_embed_targets': {}
    }

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing Teacher Embeddings", leave=False, smoothing=0.0):
            batch = {k:(x.to(device, dtype) if x.is_floating_point() else x.to(device)) if isinstance(x, torch.Tensor) else x for k, x in batch.items()}
            out_dict = model(**batch, mode='embed', chunk_size=chunk_size)
            
            node_embed = out_dict['node_embed'].cpu()
            edge_embed = out_dict['edge_embed'].cpu()
            enthalpy_embed = out_dict['enthalpy_embed'].cpu()
            entropy_embed = out_dict['entropy_embed'].cpu()
            
            for i, idx in enumerate(batch['indices'].cpu().tolist()):
                n = batch['masks'][i].sum().item()
                embeds['node_embed_targets'][idx] = node_embed[i, :n].clone()
                embeds['edge_embed_targets'][idx] = edge_embed[i, :n, :n].clone()
                embeds['enthalpy_embed_targets'][idx] = enthalpy_embed[i].clone()
                embeds['entropy_embed_targets'][idx] = entropy_embed[i].clone()
                
            if device == 'mps': torch.mps.empty_cache()

    torch.save(embeds, embeds_file)


def evaluate(model, dataloader, loss_fn, mode, device, dtype, chunk_size, desc=None, **kwargs):
    model.eval()
    total_loss = 0.0
    total_dict = defaultdict(float)
    
    with tqdm(dataloader, desc=desc, leave=False, smoothing=0.0) as pbar:
        with torch.no_grad():
            for i, batch in enumerate(pbar):
                batch = {k:(x.to(device, dtype) if x.is_floating_point() else x.to(device)) if isinstance(x, torch.Tensor) else x for k, x in batch.items()}
                out_dict = model(**batch, mode='pretrain' if mode == 'pretrain' else 'train', chunk_size=chunk_size)
                loss, loss_dict = loss_fn(**out_dict, **batch, **kwargs)
                total_loss += loss.item()
                for k, l in loss_dict.items(): total_dict[k] += l

                if device == 'mps': torch.mps.empty_cache()
                pbar.set_postfix({'loss': f'{loss.item():.4f}', **{k:f'{x:.4f}' for k,x in loss_dict.items()}})
            
    model.train()
    
    avg_loss = total_loss / max(1, len(dataloader))
    avg_dict = {k: l / max(1, len(dataloader)) for k, l in total_dict.items()}
    
    return avg_loss, avg_dict


def train(data_dir: str, mode: Literal['pretrain', 'teacher', 'student'] = 'pretrain', 
          device: str = 'mps', dtype: torch.dtype = torch.bfloat16, 
          num_epochs: int = 100, batch_size: int = 32, accum_size: int = 1, chunk_size: int = 1, num_frames: int = 16, valid_frac: float = 0.1,
          lr_head: float = 1e-4, lr_gate: float = 5e-3, lr_scaffold: float = 1e-3, warmup_epochs: int = 1, stop_lr: float | None = 1e-6,
          weight_decay: float = 1e-4, patience: int = 5,
          profile_time: bool = False, param_every: int = 0, eval_every: int = 0, checkpoint_every: int = 0,
          out_dir: str = '.', best_path: str | None = None, param_path: str | None = None, eval_path: str | None = None, copy_dir: str | None = None, 
          resume_ckpt: str | None = None, base_ckpt: str | None = None, **kwargs):

    best_path = best_path or os.path.join(out_dir, 'best.pt')
    param_path = param_path or os.path.join(out_dir, 'params.txt')
    eval_path = eval_path or os.path.join(out_dir, 'evals.txt')
    copy_dir = copy_dir or out_dir

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(best_path) or '.', exist_ok=True)
    os.makedirs(os.path.dirname(param_path) or '.', exist_ok=True)
    os.makedirs(os.path.dirname(eval_path) or '.', exist_ok=True)
    os.makedirs(copy_dir, exist_ok=True)

    if eval_every == 0: warnings.warn("Best model will not be saved as the model evaluation is disabled.")

    if resume_ckpt is None:
        if any(os.path.exists(os.path.join(copy_dir, f)) for f in ['model.py', 'train.py', 'dataset.py']):
            if input(f"WARNING: Backup files already exist in '{copy_dir}'. Press Enter to overwrite, or type anything to abort: ").strip():
                print("Aborting.")
                return

        for f in ['model.py', 'train.py', 'dataset.py']:
            if not os.path.exists(f): continue
            dest = os.path.join(copy_dir, f)
            if os.path.exists(dest): os.chmod(dest, 0o666)
            shutil.copy2(f, dest)
            os.chmod(dest, 0o444)

    if mode == 'pretrain':
        if base_ckpt is not None: warnings.warn("Argument `base_ckpt` will be ignored in pretrain mode.")
        model = TeacherModel(**kwargs).to(device).to(dtype)
        loss_fn = InteractionLoss(5.76699671, 1.51246251, 0.84149043, 0.52675139)
        dataset = TrajectoryDataset(data_dir, num_frames=1)

    elif mode == 'teacher':
        model = TeacherModel(**kwargs).to(device).to(dtype)
        if base_ckpt is not None: load_checkpoint(base_ckpt, model=model)
        loss_fn = JointLoss([(InteractionLoss(5.76699671, 1.51246251, 0.84149043, 0.52675139), 1.0), (AffinityLoss(), 1.0)]).to(dtype)
        dataset = TrajectoryDataset(data_dir, num_frames=num_frames)
        
    elif mode == 'student':
        if base_ckpt is None: raise ValueError("Argument `base_ckpt` must be defined for student mode.")
        model = StudentModel(**kwargs).to(device).to(dtype)
        loss_fn = JointLoss([(DistillationLoss(), 1.0), (AffinityLoss(), 1.0)]).to(dtype)
        
        teacher = TeacherModel(**kwargs).to(device).to(dtype)
        load_checkpoint(base_ckpt, model=teacher)
        teacher.eval()

        embeds_file = os.path.join(os.path.dirname(base_ckpt), 'embeds.pt')
        if not os.path.exists(embeds_file): embed_dataset(data_dir, embeds_file, teacher, num_frames, device, dtype, chunk_size)

        dataset = TrajectoryDataset(data_dir, num_frames=1, embeds_file=embeds_file)
        
    else: raise ValueError("Argument `mode` must be 'teacher' or 'student' or 'pretrain'.")
    
    train_dataset, valid_dataset = dataset.split([1.0 - valid_frac, valid_frac], seed=0xDEADBEEF)
    valid_dataset.eval()

    train_dataloader = DataLoader(
        dataset=train_dataset,
        collate_fn=TrajectoryDataset.collate_fn, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=4,
        persistent_workers=True
    )

    valid_dataloader = DataLoader(
        dataset=valid_dataset,
        collate_fn=TrajectoryDataset.collate_fn, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=4,
        persistent_workers=True
    )
    
    decay_params, nodecay_params = {'head': [], 'gate': [], 'scaffold': []}, {'head': [], 'gate': [], 'scaffold': []}
    for name, p in model.named_parameters():
        if not p.requires_grad: continue
        group = 'head' if 'head' in name else 'gate' if 'gate' in name or 'log_beta' in name else 'scaffold'
        if p.ndim < 2 or 'bias' in name or 'norm' in name: nodecay_params[group].append(p)
        else: decay_params[group].append(p)

    optimizer = AdamW([
        {'params': decay_params['head'], 'weight_decay': weight_decay, 'lr': lr_head},
        {'params': nodecay_params['head'], 'weight_decay': 0.0, 'lr': lr_head},
        {'params': decay_params['gate'], 'weight_decay': weight_decay, 'lr': lr_gate},
        {'params': nodecay_params['gate'], 'weight_decay': 0.0, 'lr': lr_gate},
        {'params': decay_params['scaffold'], 'weight_decay': weight_decay, 'lr': lr_scaffold},
        {'params': nodecay_params['scaffold'], 'weight_decay': 0.0, 'lr': lr_scaffold}
    ])
    for g in optimizer.param_groups: g['initial_lr'] = g['lr']

    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=patience)

    start_epoch, best_loss = 1, None
    if resume_ckpt is not None:
        start_epoch, best_loss = load_checkpoint(resume_ckpt, model, optimizer, scheduler)
        start_epoch += 1

    model.train()
    start_time = time.time()
    
    with tqdm(range(start_epoch, num_epochs + 1), initial=start_epoch, total=num_epochs, desc="Epochs", leave=False) as pbar1:
        for epoch in pbar1:
            running_loss, running_dict, smoothing = 0.0, defaultdict(float), 0.02
            timer = defaultdict(float)
            
            with tqdm(train_dataloader, desc="Batches", leave=False, smoothing=0.0) as pbar2:
                t0 = time.time()
                for i, batch in enumerate(pbar2):
                    t1 = time.time()
                    
                    batch = {k:(x.to(device, dtype=dtype) if x.is_floating_point() else x.to(device)) if isinstance(x, torch.Tensor) else x for k, x in batch.items()}
                    
                    if i % accum_size == 0:
                        optimizer.zero_grad()
                    
                    out_dict = model(**batch, mode='pretrain' if mode == 'pretrain' else 'train', chunk_size=chunk_size)
                    loss, loss_dict = loss_fn(**out_dict, **batch)
                    
                    if profile_time and device == 'mps': torch.mps.synchronize()
                    t2 = time.time()

                    (loss / accum_size).backward()

                    if (i + 1) % accum_size == 0 or (i + 1) == len(train_dataloader):
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
                        
                        if epoch <= warmup_epochs:
                            scale = (epoch - 1 + (i + 1) / len(train_dataloader)) / warmup_epochs
                            for g in optimizer.param_groups: g['lr'] = g['initial_lr'] * scale
                            
                        optimizer.step()

                    if profile_time and device == 'mps': torch.mps.synchronize()
                    t3 = time.time()
                    running_loss = loss.item() * smoothing + running_loss * (1 - smoothing) if i else loss.item()
                    for k, l in loss_dict.items(): running_dict[k] = l * smoothing + running_dict[k] * (1 - smoothing) if i else l

                    t4 = time.time()
                    if device == 'mps': torch.mps.empty_cache()
                    t5 = time.time()
                    
                    if profile_time:
                        timer['load'] += (t1 - t0) * 1000
                        timer['fwd'] += (t2 - t1) * 1000
                        timer['bwd'] += (t3 - t2) * 1000
                        timer['cache'] += (t5 - t4) * 1000

                    postfix_dict = {'loss': f'{running_loss:.4f}', **{k:f'{x:.4f}' for k,x in running_dict.items()}, 'lr': f'{optimizer.param_groups[-1]["lr"]:.1e}'}
                    if profile_time:
                        postfix_dict['load'] = f'{(t1 - t0) * 1000:.1f}'
                        postfix_dict['fwd'] = f'{(t2 - t1) * 1000:.1f}'
                        postfix_dict['bwd'] = f'{(t3 - t2) * 1000:.1f}'
                        postfix_dict['cache'] = f'{(t5 - t4) * 1000:.1f}'
                    pbar2.set_postfix(postfix_dict)
                    t0 = time.time()

            postfix_dict = {
                'loss': f'{running_loss:.4f}',
                **{k: f'{l:.4f}' for k, l in running_dict.items()},
                'lr': f'{optimizer.param_groups[-1]["lr"]:.1e}'
            }
            if profile_time: postfix_dict.update({k: f'{t / max(1, len(train_dataloader)):.1f}' for k, t in timer.items()})
            pbar1.set_postfix(postfix_dict)

            if param_every and epoch % param_every == 0:
                param_str = f"\n--- Parameter Statistics (Epoch {epoch}) ---\n" + param_stats(model)
                if param_path:
                    with open(param_path, "a") as f: f.write(f"{param_str}\n")
                else: tqdm.write(param_str)
            
            if eval_every and epoch % eval_every == 0:
                eval_loss, eval_dict = evaluate(model, valid_dataloader, loss_fn, mode, device, dtype, chunk_size, desc="Validating")
                res_str = f"Epoch {epoch} - " \
                          f"Train Loss: {running_loss:.4f} ({', '.join(f'{k}: {l:.4f}' for k, l in running_dict.items())}) | " \
                          f"Validation Loss: {eval_loss:.4f} ({', '.join(f'{k}: {l:.4f}' for k, l in eval_dict.items())}) | " \
                          f"LR: {optimizer.param_groups[-1]['lr']:.1e}"
                tqdm.write(f"Evaluation Results at {res_str}")
                if eval_path is not None:
                    with open(eval_path, "a") as f: f.write(f"{res_str}\n")
                if best_loss is None or eval_loss < best_loss:
                    best_loss = eval_loss
                    save_checkpoint(best_path, model, optimizer, scheduler, epoch, best_loss)
                scheduler.step(eval_loss)

            if checkpoint_every and epoch % checkpoint_every == 0:
                save_checkpoint(os.path.join(out_dir, f"epoch_{epoch}.pt"), model, optimizer, scheduler, epoch, best_loss)
            
            if stop_lr and optimizer.param_groups[-1]['lr'] < stop_lr: break
    
    end_time = time.time()
    print(f"Training Completed in {datetime.timedelta(seconds=round(end_time - start_time))}")


if __name__ == '__main__':
    # train(
    #     data_dir='data/dynarepo/dataset.h5', mode='pretrain',
    #     device='mps', dtype=torch.float32, num_epochs=1000, valid_frac=0.1,
    #     batch_size=2, accum_size=8, chunk_size=2, lr_head=1e-4, lr_gate=5e-3, lr_scaffold=1e-3, warmup_epochs=1, patience=10,
    #     param_every=1, eval_every=1, checkpoint_every=10,
    #     out_dir='checkpoints/run58_pretrain',
    # )

    # train(
    #     data_dir='data/dynarepo/dataset.h5', mode='teacher',
    #     device='mps', dtype=torch.float32, num_epochs=5, valid_frac=0.1,
    #     batch_size=8, accum_size=1, chunk_size=2, num_frames=4, lr_head=1e-4, lr_gate=5e-3, lr_scaffold=1e-3, warmup_epochs=1, patience=5,
    #     param_every=1, eval_every=1, checkpoint_every=5,
    #     base_ckpt='checkpoints/run64_58_warmup/epoch_0.pt',
    #     out_dir='checkpoints/run64_58_warmup',
    # )

    # train(
    #     data_dir='data/dynarepo/dataset.h5', mode='pretrain',
    #     device='mps', dtype=torch.float32, num_epochs=1000, valid_frac=0.1,
    #     batch_size=8, accum_size=1, chunk_size=1, num_frames=1, lr_head=1e-3, lr_gate=1e-3, lr_scaffold=1e-3, warmup_epochs=2, patience=10,
    #     param_every=1, eval_every=1, checkpoint_every=5,
    #     out_dir='checkpoints/run73_pretrain',
    # )

    train(
        data_dir='data/dynarepo/dataset.h5', mode='teacher',
        device='mps', dtype=torch.float32, num_epochs=1000, valid_frac=0.1,
        batch_size=8, accum_size=1, chunk_size=1, num_frames=4, lr_head=1e-3, lr_gate=5e-4, lr_scaffold=1e-4, warmup_epochs=5, patience=20, weight_decay=1e-4,
        param_every=1, eval_every=1, checkpoint_every=1,
        base_ckpt='checkpoints/run79/epoch_0.pt',
        out_dir='checkpoints/run79',
    )