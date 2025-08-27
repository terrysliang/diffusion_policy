if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hydra
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import copy
import random
# init wandb only on rank 0 later
import wandb
import tqdm
import numpy as np
import shutil
from contextlib import nullcontext
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.model.common.lr_scheduler import get_scheduler

OmegaConf.register_new_resolver("eval", eval, replace=True)

# ---------------------- DDP helpers ----------------------
def setup_ddp():
    """Initialize DDP if launched with torchrun; return local_rank (int)."""
    if dist.is_available() and dist.is_initialized():
        return int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        return local_rank
    return 0

def is_distributed():
    return dist.is_available() and dist.is_initialized()

def is_main_process():
    return (not is_distributed()) or dist.get_rank() == 0
# --------------------------------------------------------


class TrainDiffusionUnetImageWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model (unwrapped here)
        self.model: DiffusionUnetImagePolicy = hydra.utils.instantiate(cfg.policy)

        self.ema_model: DiffusionUnetImagePolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        # optimizer
        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=self.model.parameters())

        # state
        self.global_step = 0
        self.epoch = 0

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        # -------- DDP setup + device --------
        local_rank = setup_ddp()
        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

        # resume (safe to do on all ranks)
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                print(f"[rank {dist.get_rank() if is_distributed() else 0}] Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)

        # configure dataset
        dataset: BaseImageDataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseImageDataset)

        # train loader with DistributedSampler when DDP
        train_sampler = DistributedSampler(dataset, shuffle=cfg.dataloader.shuffle, drop_last=True) if is_distributed() else None
        train_dataloader = DataLoader(
            dataset,
            batch_size=cfg.dataloader.batch_size,
            sampler=train_sampler,
            shuffle=(train_sampler is None and cfg.dataloader.shuffle),
            num_workers=cfg.dataloader.num_workers,
            pin_memory=cfg.dataloader.pin_memory,
            persistent_workers=cfg.dataloader.persistent_workers
        )
        normalizer = dataset.get_normalizer()

        # validation only on main process to avoid duplication
        if is_main_process():
            val_dataset = dataset.get_validation_dataset()
            val_dataloader = DataLoader(
                val_dataset,
                batch_size=cfg.val_dataloader.batch_size,
                shuffle=cfg.val_dataloader.shuffle,
                num_workers=cfg.val_dataloader.num_workers,
                pin_memory=cfg.val_dataloader.pin_memory,
                persistent_workers=cfg.val_dataloader.persistent_workers
            )
        else:
            val_dataset, val_dataloader = None, None

        # set normalizers (model not wrapped yet)
        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        # lr scheduler (per-rank; step every batch)
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
                len(train_dataloader) * cfg.training.num_epochs) \
                    // cfg.training.gradient_accumulate_every,
            last_epoch=self.global_step-1
        )

        # ema
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(cfg.ema, model=self.ema_model)

        # env runner — run only on rank 0
        if is_main_process():
            env_runner: BaseImageRunner = hydra.utils.instantiate(
                cfg.task.env_runner, output_dir=self.output_dir)
            assert isinstance(env_runner, BaseImageRunner)
        else:
            env_runner = None

        # logging (rank 0 only)
        if is_main_process():
            wandb_run = wandb.init(
                dir=str(self.output_dir),
                config=OmegaConf.to_container(cfg, resolve=True),
                **cfg.logging
            )
            wandb.config.update({"output_dir": self.output_dir})
        else:
            class _NoOpWandB:
                def log(self, *a, **k): pass
                def __getattr__(self, name): return self  # swallow other accesses
            wandb_run = _NoOpWandB()

        # checkpoints (rank 0 only)
        if is_main_process():
            topk_manager = TopKCheckpointManager(
                save_dir=os.path.join(self.output_dir, 'checkpoints'),
                **cfg.checkpoint.topk
            )
        else:
            topk_manager = None

        # device transfer + DDP wrap
        self.model.to(device)
        if cfg.training.use_ema:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)

        if is_distributed():
            self.model = DDP(self.model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

        # save batch for sampling (rank 0 only)
        train_sampling_batch = None

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1

        # training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        json_logger_ctx = JsonLogger(log_path) if is_main_process() else nullcontext()
        with json_logger_ctx as json_logger:
            for local_epoch_idx in range(cfg.training.num_epochs):
                step_log = dict()

                if train_sampler is not None:
                    train_sampler.set_epoch(self.epoch)

                # ========= train for this epoch ==========
                if cfg.training.freeze_encoder:
                    # access underlying module if wrapped
                    target_model = self.model.module if isinstance(self.model, DDP) else self.model
                    target_model.obs_encoder.eval()
                    target_model.obs_encoder.requires_grad_(False)

                train_losses = list()
                with tqdm.tqdm(
                    train_dataloader,
                    desc=f"Training epoch {self.epoch} (rank {dist.get_rank() if is_distributed() else 0})",
                    leave=False,
                    mininterval=cfg.training.tqdm_interval_sec,
                    disable=not is_main_process()
                ) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        # device transfer
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        if is_main_process() and train_sampling_batch is None:
                            # keep a copy only on rank 0
                            train_sampling_batch = batch

                        # compute loss
                        # loss should be computed on the DDP-wrapped model
                        raw_loss = (self.model.module if isinstance(self.model, DDP) else self.model).compute_loss(batch)
                        loss = raw_loss / cfg.training.gradient_accumulate_every
                        loss.backward()

                        # step optimizer
                        if (batch_idx + 1) % cfg.training.gradient_accumulate_every == 0:
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            lr_scheduler.step()
                        
                        # update ema (use the *unwrapped* src model)
                        if cfg.training.use_ema:
                            src_model = self.model.module if isinstance(self.model, DDP) else self.model
                            ema.step(src_model)

                        # logging (rank 0 only)
                        raw_loss_cpu = raw_loss.item()
                        if is_main_process():
                            tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                            train_losses.append(raw_loss_cpu)
                            step_log = {
                                'train_loss': raw_loss_cpu,
                                'global_step': self.global_step,
                                'epoch': self.epoch,
                                'lr': lr_scheduler.get_last_lr()[0]
                            }
                            is_last_batch = (batch_idx == (len(train_dataloader)-1))
                            if not is_last_batch:
                                if wandb_run is not None:
                                    wandb_run.log(step_log, step=self.global_step)
                                if json_logger is not None:
                                    json_logger.log(step_log)

                        self.global_step += 1

                        if (cfg.training.max_train_steps is not None) \
                            and batch_idx >= (cfg.training.max_train_steps-1):
                            break

                # at the end of each epoch: rank 0 aggregates/prints
                if is_main_process():
                    train_loss = float(np.mean(train_losses)) if len(train_losses) > 0 else float('nan')
                    step_log['train_loss'] = train_loss

                # ========= eval for this epoch (rank 0 only) ==========
                # choose policy (ema or model)
                policy = (self.ema_model if cfg.training.use_ema else (self.model.module if isinstance(self.model, DDP) else self.model))
                policy.eval()

                # rollout on env (rank 0 only)
                if is_main_process() and (self.epoch % cfg.training.rollout_every) == 0 and env_runner is not None:
                    runner_log = env_runner.run(policy)
                    step_log.update(runner_log)

                # validation (rank 0 only)
                if is_main_process() and (self.epoch % cfg.training.val_every) == 0 and val_dataloader is not None:
                    with torch.no_grad():
                        val_losses = list()
                        with tqdm.tqdm(
                            val_dataloader,
                            desc=f"Validation epoch {self.epoch}",
                            leave=False,
                            mininterval=cfg.training.tqdm_interval_sec,
                            disable=not is_main_process()
                        ) as tepoch:
                            for batch_idx, batch in enumerate(tepoch):
                                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                                vloss = (self.model.module if isinstance(self.model, DDP) else self.model).compute_loss(batch)
                                val_losses.append(vloss.item())
                                if (cfg.training.max_val_steps is not None) \
                                    and batch_idx >= (cfg.training.max_val_steps-1):
                                    break
                        if len(val_losses) > 0:
                            step_log['val_loss'] = float(np.mean(val_losses))

                # sampling on a training batch (rank 0 only)
                if is_main_process() and (self.epoch % cfg.training.sample_every) == 0 and train_sampling_batch is not None:
                    with torch.no_grad():
                        batch = dict_apply(train_sampling_batch, lambda x: x.to(device, non_blocking=True))
                        obs_dict = batch['obs']
                        gt_action = batch['action']
                        result = policy.predict_action(obs_dict)
                        pred_action = result['action_pred']
                        mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                        step_log['train_action_mse_error'] = mse.item()
                        del batch, obs_dict, gt_action, result, pred_action, mse
                
                # checkpoint (rank 0 only)
                if is_main_process() and (self.epoch % cfg.training.checkpoint_every) == 0:
                    if cfg.checkpoint.save_last_ckpt:
                        self.save_checkpoint()
                    if cfg.checkpoint.save_last_snapshot:
                        self.save_snapshot()

                    # sanitize metric names
                    metric_dict = {k.replace('/', '_'): v for k, v in step_log.items()}

                    if topk_manager is not None:
                        topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)
                        if topk_ckpt_path is not None:
                            self.save_checkpoint(path=topk_ckpt_path)

                # back to train
                (self.model.module if isinstance(self.model, DDP) else self.model).train()

                # end of epoch logging (rank 0 only)
                if is_main_process():
                    if wandb_run is not None:
                        wandb_run.log(step_log, step=self.global_step)
                    if json_logger is not None:
                        json_logger.log(step_log)

                self.epoch += 1

        # clean up process group on exit
        if is_distributed():
            dist.barrier()
            dist.destroy_process_group()


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainDiffusionUnetImageWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
