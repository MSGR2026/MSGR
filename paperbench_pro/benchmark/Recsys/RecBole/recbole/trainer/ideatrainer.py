# -*- coding: utf-8 -*-
# @Time   : 2025/01/10
# @Author : Integrated from IDEA implementation

r"""
IDEATrainer
################################################
Custom trainer for IDEAGRU model with two-phase training:
1. Pretraining phase: Train the recommendation model
2. Adversarial training phase: Alternative training between environment generator and recommendation model
"""

import torch
import torch.cuda.amp as amp
from logging import getLogger
from time import time
from tqdm import tqdm
import numpy as np

from recbole.trainer import Trainer
from recbole.utils import (
    set_color,
    get_gpu_usage,
    early_stopping,
    dict2str,
)
from torch.nn.utils.clip_grad import clip_grad_norm_


class IDEATrainer(Trainer):
    r"""IDEATrainer implements a two-phase training strategy for IDEAGRU.
    
    Phase 1: Pretraining - Train the base recommendation model
    Phase 2: Adversarial Training - Jointly train environment generator and recommendation model
    """

    def __init__(self, config, model):
        super().__init__(config, model)
        
        # Separate parameters for environment generator and recommendation model
        env_params = []
        erm_params = []
        for name, params in self.model.named_parameters():
            if 'env' in name:
                env_params.append(params)
            else:
                erm_params.append(params)
        
        # Create separate optimizers
        self.optimizer_rec = self._build_optimizer(params=erm_params)
        self.optimizer_env = self._build_optimizer(params=env_params)
        
        # Training phase control
        self.env_update_interval = config.get('env_update_interval', 5)

    def _train_epoch(self, optimizer, loss_func, train_data, epoch_idx, show_progress=False):
        """Standard training epoch (used in pretraining phase)."""
        self.model.train()
        
        total_loss = None
        iter_data = (
            tqdm(
                train_data,
                total=len(train_data),
                ncols=100,
                desc=set_color(f"Train {epoch_idx:>5}", "pink"),
            )
            if show_progress
            else train_data
        )
        
        if not self.config["single_spec"] and train_data.shuffle:
            train_data.sampler.set_epoch(epoch_idx)

        scaler = amp.GradScaler(enabled=self.enable_scaler)
        
        for _, interaction in enumerate(iter_data):
            interaction = interaction.to(self.device)
            optimizer.zero_grad()
            
            sync_loss = 0
            if not self.config["single_spec"]:
                self.set_reduce_hook()
                sync_loss = self.sync_grad_loss()

            with torch.autocast(device_type=self.device.type, enabled=self.enable_amp):
                losses = loss_func(interaction)

            if isinstance(losses, tuple):
                loss = sum(losses)
                loss_tuple = tuple(per_loss.item() for per_loss in losses)
                total_loss = (
                    loss_tuple
                    if total_loss is None
                    else tuple(map(sum, zip(total_loss, loss_tuple)))
                )
            else:
                loss = losses
                total_loss = (
                    losses.item() if total_loss is None else total_loss + losses.item()
                )
                
            self._check_nan(loss)
            scaler.scale(loss + sync_loss).backward()
            
            if self.clip_grad_norm:
                clip_grad_norm_(self.model.parameters(), **self.clip_grad_norm)
                
            scaler.step(optimizer)
            scaler.update()
            
            if self.gpu_available and show_progress:
                iter_data.set_postfix_str(
                    set_color("GPU RAM: " + get_gpu_usage(self.device), "yellow")
                )
                
        return total_loss

    def _adtrain_epoch(self, loss_func1, loss_func2, train_data, epoch_idx, show_progress=False):
        """Adversarial training epoch with alternating optimization."""
        self.model.train()
        total_loss = None
        
        iter_data = (
            tqdm(
                train_data,
                total=len(train_data),
                ncols=100,
                desc=set_color(f"AdTrain {epoch_idx:>5}", "pink"),
            )
            if show_progress
            else train_data
        )
        
        if not self.config["single_spec"] and train_data.shuffle:
            train_data.sampler.set_epoch(epoch_idx)

        scaler = amp.GradScaler(enabled=self.enable_scaler)
        
        for batch_idx, interaction in enumerate(iter_data):
            interaction = interaction.to(self.device)
            
            # Update recommendation model
            self.optimizer_rec.zero_grad()
            sync_loss = 0
            if not self.config["single_spec"]:
                self.set_reduce_hook()
                sync_loss = self.sync_grad_loss()

            with torch.autocast(device_type=self.device.type, enabled=self.enable_amp):
                losses = loss_func1(interaction)  # Invariant learning loss

            if isinstance(losses, tuple):
                loss = sum(losses)
                loss_tuple = tuple(per_loss.item() for per_loss in losses)
                total_loss = (
                    loss_tuple
                    if total_loss is None
                    else tuple(map(sum, zip(total_loss, loss_tuple)))
                )
            else:
                loss = losses
                total_loss = (
                    losses.item() if total_loss is None else total_loss + losses.item()
                )
                
            self._check_nan(loss)
            scaler.scale(loss + sync_loss).backward()
            
            if self.clip_grad_norm:
                clip_grad_norm_(self.model.parameters(), **self.clip_grad_norm)
                
            scaler.step(self.optimizer_rec)
            scaler.update()
            
            # Update environment generator every T batches
            if (batch_idx + 1) % self.env_update_interval == 0:
                self.optimizer_env.zero_grad()
                inv_loss = loss_func2(interaction)  # Environment loss
                inv_loss.backward()
                self.optimizer_env.step()

            if self.gpu_available and show_progress:
                iter_data.set_postfix_str(
                    set_color("GPU RAM: " + get_gpu_usage(self.device), "yellow")
                )
                
        return total_loss

    def fit(self, train_data, valid_data=None, verbose=True, saved=True, show_progress=False, callback_fn=None):
        """Standard fitting (pretraining phase)."""
        return super().fit(
            train_data, 
            valid_data=valid_data, 
            verbose=verbose, 
            saved=saved, 
            show_progress=show_progress, 
            callback_fn=callback_fn
        )

    def adfit(self, train_data, loss_func1, loss_func2, valid_data=None, verbose=True, saved=True, show_progress=False, callback_fn=None):
        """Adversarial training phase."""
        if saved and self.start_epoch >= self.epochs:
            self._save_checkpoint(-1, verbose=verbose)
            
        self.best_valid_score = -np.inf if self.valid_metric_bigger else np.inf
        self.best_valid_result = None
        self.eval_collector.data_collect(train_data)
        
        if self.config["train_neg_sample_args"].get("dynamic", False):
            train_data.get_model(self.model)
            
        valid_step = 0
        
        for epoch_idx in range(self.start_epoch, self.epochs):
            # Train
            training_start_time = time()
            train_loss = self._adtrain_epoch(
                loss_func1, loss_func2, train_data, epoch_idx, show_progress=show_progress
            )
            self.train_loss_dict[epoch_idx] = (
                sum(train_loss) if isinstance(train_loss, tuple) else train_loss
            )
            training_end_time = time()
            train_loss_output = self._generate_train_loss_output(
                epoch_idx, training_start_time, training_end_time, train_loss
            )
            if verbose:
                self.logger.info(train_loss_output)
            self._add_train_loss_to_tensorboard(epoch_idx, train_loss)
            self.wandblogger.log_metrics(
                {"epoch": epoch_idx, "train_loss": train_loss, "train_step": epoch_idx},
                head="train",
            )

            # Evaluation
            if self.eval_step <= 0 or not valid_data:
                if saved:
                    self._save_checkpoint(epoch_idx, verbose=verbose)
                continue
                
            if (epoch_idx + 1) % self.eval_step == 0:
                valid_start_time = time()
                valid_score, valid_result = self._valid_epoch(
                    valid_data, show_progress=show_progress
                )

                (
                    self.best_valid_score,
                    self.cur_step,
                    stop_flag,
                    update_flag,
                ) = early_stopping(
                    valid_score,
                    self.best_valid_score,
                    self.cur_step,
                    max_step=self.stopping_step,
                    bigger=self.valid_metric_bigger,
                )
                valid_end_time = time()
                valid_score_output = (
                    set_color("epoch %d evaluating", "green")
                    + " ["
                    + set_color("time", "blue")
                    + ": %.2fs, "
                    + set_color("valid_score", "blue")
                    + ": %f]"
                ) % (epoch_idx, valid_end_time - valid_start_time, valid_score)
                valid_result_output = (
                    set_color("valid result", "blue") + ": \n" + dict2str(valid_result)
                )
                if verbose:
                    self.logger.info(valid_score_output)
                    self.logger.info(valid_result_output)
                self.tensorboard.add_scalar("Vaild_score", valid_score, epoch_idx)
                self.wandblogger.log_metrics(
                    {**valid_result, "valid_step": valid_step}, head="valid"
                )

                if update_flag:
                    if saved:
                        self._save_checkpoint(epoch_idx, verbose=verbose)
                    self.best_valid_result = valid_result

                if callback_fn:
                    callback_fn(epoch_idx, valid_score)

                if stop_flag:
                    stop_output = "Finished training, best eval result in epoch %d" % (
                        epoch_idx - self.cur_step * self.eval_step
                    )
                    if verbose:
                        self.logger.info(stop_output)
                    break

                valid_step += 1

        self._add_hparam_to_tensorboard(self.best_valid_score)
        return self.best_valid_score, self.best_valid_result

    def idea_fit(self, train_data, valid_data=None, verbose=True, saved=True, show_progress=False, callback_fn=None):
        """Two-phase training for IDEAGRU: pretraining + adversarial training."""
        
        # Phase 1: Pretraining
        if verbose:
            self.logger.info(set_color("Phase 1: Pretraining", "cyan"))
        
        # Freeze environment generator
        for name, params in self.model.named_parameters():
            params.requires_grad = True
            if 'env' in name:
                params.requires_grad = False
        
        self.start_epoch = 0
        self.train_loss_dict = dict()
        
        # Pretrain with standard loss
        _, _ = super().fit(
            train_data, 
            valid_data=valid_data,
            verbose=verbose,
            saved=saved,
            show_progress=show_progress,
            callback_fn=callback_fn
        )
        
        # Phase 2: Adversarial Training
        if verbose:
            self.logger.info(set_color("Phase 2: Adversarial Training", "cyan"))
        
        # Unfreeze all parameters
        for params in self.model.parameters():
            params.requires_grad = True
        
        # Reset training state
        self.start_epoch = 0
        self.train_loss_dict = dict()
        
        # Adversarial training
        best_valid_score, best_valid_result = self.adfit(
            train_data, 
            loss_func1=self.model.calculate_loss_il,
            loss_func2=self.model.calculate_loss_env,
            valid_data=valid_data,
            verbose=verbose,
            saved=saved,
            show_progress=show_progress,
            callback_fn=callback_fn
        )
        
        return best_valid_score, best_valid_result
