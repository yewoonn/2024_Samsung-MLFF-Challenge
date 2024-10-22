"""
Copyright (C) 2023 Samsung Electronics Co. LTD

This software is a property of Samsung Electronics.
No part of this software, either material or conceptual may be copied or distributed, transmitted,
transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
electronic, mechanical, manual or otherwise, or disclosed
to third parties without the express written permission of Samsung Electronics.
"""

"""
Reference : ocp/ocpmodels/trainers/base_trainer.py

The following items are modified and they can be claimed as properties of Samsung Electronics. 

(1) Support more MLFF models (BPNN, NequIP, Allegro, and MACE)
(2) Support simulation indicators for the benchmark evaluation on simulations (RDF, ADF, EoS, PEW)
(3) Support more loss functions and metrics (loss.py and metric_evaluator.py in src/modules/)
(4) Support more learning rate schedulers (scheduler.py in src/modules/)
(5) Support normalization of per-atom energy (NormalizerPerAtom in src/modules/normalizer.py)
(6) Some different featurs are as follows:
    (a) Print training results using PrettyTable
    (b) Use a benchmark logger (named bm_logging) instead of the root logger (named logging in OCP)
    (c) Remove features that includes to save prediction results and make the corresponding directory named 'results'
    (d) Remove features related to HPO
    (e) Set the identifier of an experiment using the starting time
"""

"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
import datetime
import errno
import os
import random
import yaml
import json
import math

from pathlib import Path
from abc import ABC, abstractmethod
from collections import defaultdict
from tqdm import tqdm
from prettytable import PrettyTable

import numpy as np
import torch
import torch.nn as nn
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.utils.data import DataLoader

import ocpmodels
from ocpmodels.common import distutils, gp_utils
from ocpmodels.common.data_parallel import BalancedBatchSampler, OCPDataParallel
from ocpmodels.common.registry import registry
from ocpmodels.common.utils import load_state_dict, save_checkpoint
from ocpmodels.modules.exponential_moving_average import ExponentialMovingAverage
from ocpmodels.modules.scaling.compat import load_scales_compat
from ocpmodels.modules.scaling.util import ensure_fitted

from src.common.utils import bm_logging, get_device
from src.common.logger import parse_logs
from src.modules.loss import initiate_loss
from src.modules.scheduler import LRScheduler
from src.modules.metric_evaluator import MetricEvaluator
from src.common.collaters.parallel_collater import ParallelCollater
from src.modules.normalizer import log_and_check_normalizers


@registry.register_trainer("base")
class BaseTrainer(ABC):
    @property
    def _unwrapped_model(self):
        module = self.model
        while isinstance(module, (OCPDataParallel, DistributedDataParallel)):
            module = module.module
        return module

    def __init__(self, config):
        assert config is not None

        # set mode
        self.mode = config["mode"]
        self.flag_loading_dataset = (self.mode == "train" or self.mode == "fit-scale")

        # debug mode
        self.is_debug = config["is_debug"]

        self.task_name = "s2ef"
        self.config = self._parse_config(config)

        self.epoch = 0
        self.step = 0

        self.cpu = self.config["gpus"] == 0
        self.device = get_device(self.config)

        # use float32 as default precision
        torch.set_default_dtype(torch.float32) 

        # AMP Scaler
        self.scaler = torch.cuda.amp.GradScaler() if self.config["amp"] else None

        if "SLURM_JOB_ID" in os.environ and "folder" in self.config["slurm"]:
            if "SLURM_ARRAY_JOB_ID" in os.environ:
                self.config["slurm"]["job_id"] = "%s_%s" % (
                    os.environ["SLURM_ARRAY_JOB_ID"],
                    os.environ["SLURM_ARRAY_TASK_ID"],
                )
            else:
                self.config["slurm"]["job_id"] = os.environ["SLURM_JOB_ID"]
            self.config["slurm"]["folder"] = self.config["slurm"][
                "folder"
            ].replace("%j", self.config["slurm"]["job_id"])

        # set a dataset normalizer (for energy and force)
        self.normalizer = self.config.get("normalizer", self.config["dataset"])

        # make directories to save checkpoint and log
        if (self.mode == "train"
            and not self.is_debug 
            and distutils.is_master()
        ):
            os.makedirs(self.config["cmd"]["checkpoint_dir"], exist_ok=True)
            os.makedirs(self.config["cmd"]["logs_dir"], exist_ok=True)

        # determine whether to use stress
        self.use_stress = config['model'].get('regress_stress', False)

        # set various modules used in the trainer        
        self._inititiate()

        # logging the local config
        if self.mode == "train":
            bm_logging.info(f"\n{yaml.dump(self.config, default_flow_style=False)}")

    def _parse_config(self, config):
        logger_name = config["logger"] if isinstance(config["logger"], str) else config["logger"]["name"]
        timestamp_id = config["timestamp_id"]
        trainer_config = {
            "task": config["task"],
            "model_name": config["model"].pop("name"),
            "model_attributes": config["model"],
            "optim": config["optim"],
            "logger": logger_name,
            "amp": config["amp"],
            "gpus": distutils.get_world_size() if not config["cpu"] else 0,
            "cmd": {
                "identifier": config["identifier"],
                "print_every": config["print_every"],
                "seed": config["seed"],
                "timestamp_id": timestamp_id,
                "checkpoint_dir": os.path.join(config["run_dir"], "checkpoints", timestamp_id),
                "logs_dir": os.path.join(config["run_dir"], "logs", logger_name, timestamp_id),
                "show_eval_progressbar": config.get("show_eval_progressbar", False),
                "resume": config.get("resume", False),
            },
            "local_rank": config["local_rank"],
            "slurm": config["slurm"],
            "noddp": config["noddp"],
            "save_ckpt_every_epoch": config["save_ckpt_every_epoch"],
        }

        # set data configuration style
        if config["task"]["dataset"] == "lmdb":
            trainer_config["data_config_style"] = "OCP"
        elif config["task"]["dataset"] == "lmdb_sait":
            trainer_config["data_config_style"] = "SAIT"

        # specify dataset path
        dataset = config["dataset"]
        if isinstance(dataset, list):
            if len(dataset) > 0:
                trainer_config["dataset"] = dataset[0]
            if len(dataset) > 1:
                trainer_config["val_dataset"] = dataset[1]
            if len(dataset) > 2:
                trainer_config["test_dataset"] = dataset[2]
        elif isinstance(dataset, dict):
            trainer_config["dataset"] = dataset.get("train", None)
            trainer_config["val_dataset"] = dataset.get("valid", None)
            trainer_config["test_dataset"] = dataset.get("test", None)
            if trainer_config["data_config_style"] == "SAIT":
                trainer_config["normalizer"] = dataset.get("normalize", {})
        else:
            trainer_config["dataset"] = dataset
            
        return trainer_config

    def _inititiate(self):
        self._set_seed_from_config()
        self._set_logger()
        self._set_datasets_and_generate_loaders_samplers()
        self._set_normalizer()
        self._set_task()
        self._set_model()
        self._set_loss()
        self._set_optimizer_and_lr_scheduler()
        self._set_ema()
        self._set_evaluator()
        self._set_extras()

    def _set_seed_from_config(self):
        # https://pytorch.org/docs/stable/notes/randomness.html
        seed = self.config["cmd"]["seed"]
        if seed is None:
            bm_logging.warning("It is recommended to set a seed for reproducing results")
        else:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def _set_logger(self):
        self.logger = None
        if (self.mode == "train"
            and not self.is_debug 
            and distutils.is_master()
        ):
            logger_class = registry.get_logger_class(self.config["logger"])
            self.logger = logger_class(self.config)

    def get_sampler(self, dataset, batch_size, shuffle):
        if "load_balancing" in self.config["optim"]:
            balancing_mode = self.config["optim"]["load_balancing"]
            force_balancing = True
        else:
            balancing_mode = "atoms"
            force_balancing = False

        if gp_utils.initialized():
            num_replicas = gp_utils.get_dp_world_size()
            rank = gp_utils.get_dp_rank()
        else:
            num_replicas = distutils.get_world_size()
            rank = distutils.get_rank()

        sampler = BalancedBatchSampler(
            dataset,
            batch_size=batch_size,
            num_replicas=num_replicas,
            rank=rank,
            device=self.device,
            mode=balancing_mode,
            shuffle=shuffle,
            force_balancing=force_balancing,
        )
        return sampler

    def get_dataloader(self, dataset, sampler, collater):
        loader = DataLoader(
            dataset,
            collate_fn=collater,
            num_workers=self.config["optim"]["num_workers"],
            pin_memory=True,
            batch_sampler=sampler,
        )
        return loader
    
    def initiate_collater(self):
        return ParallelCollater(
            num_gpus=0 if self.cpu else 1,
            otf_graph=self.config["model_attributes"].get("otf_graph", False),
        )

    def _do_data_related_settings(self):
        """ After setting dataset and loader, this function is called."""
        if self.config["model_name"] == "bpnn":
            # scale and pca should be used in BPNN
            if self.config.get("data_config_style", "OCP") == "SAIT":
                assert ("dataset_path" in self.config["model_attributes"] or "src" in self.normalizer)
            dataset_path = self.config["model_attributes"].get("dataset_path", self.normalizer["src"]) # train dataset
            if os.path.isfile(dataset_path):
                # single lmdb file
                pca_path = Path(dataset_path).parent / "BPNN_pca.pt"
            elif os.path.isdir(dataset_path):
                # multi lmdb files
                pca_path = os.path.join(dataset_path, "BPNN_pca.pt")
            else:
                raise RuntimeError("Error")
            self.config["model_attributes"]["pca_path"] = pca_path
            self.config["model_attributes"]["dataset_path"] = dataset_path

    def _set_datasets_and_generate_loaders_samplers(self):
        self.parallel_collater = self.initiate_collater()

        if self.mode == "validate":
            # in validate mode, skip this
            return

        dataset_class = registry.get_dataset_class(self.config["task"]["dataset"])
        # print(f"Dataset_class: {dataset_class}/n")
        assert "batch_size" in self.config["optim"], "Specify batch_size"
        train_local_batch_size = self.config["optim"]["batch_size"]
        # print(f"train_local_batch_size: {train_local_batch_size}/n")
        eval_local_batch_size = self.config["optim"].get("eval_batch_size", train_local_batch_size)
        # print(f"eval_local_batch_size: {eval_local_batch_size}/n")

        self.train_dataset = self.val_dataset = self.test_dataset = None
        self.train_loader = self.val_loader = self.test_loader = None
        
        # train set
        if self.config.get("dataset", None) and self.flag_loading_dataset:
            bm_logging.info(f"Loading train dataset (type: {self.config['task']['dataset']})")
            if self.config.get('data_config_style', 'OCP') == 'OCP':
                bm_logging.info(f" - {self.config['dataset']['src']}")
            self.train_dataset = dataset_class(self.config["dataset"])
            # print(f"train_dataset: {self.train_dataset}")

            self.train_sampler = self.get_sampler(
                dataset=self.train_dataset,
                batch_size=train_local_batch_size,
                shuffle=True, 
            )
            # print(f"train_sampler: {self.train_sampler}")
            
            self.train_loader = self.get_dataloader(
                dataset=self.train_dataset,
                sampler=self.train_sampler,
                collater=self.parallel_collater,
            )
            # print(f"train_loader: {self.train_loader}")

            self.config["optim"]["num_train"] = len(self.train_dataset)

        # validation set
        if self.config.get("val_dataset", None) and self.flag_loading_dataset:
            bm_logging.info(f"Loading validation dataset (type: {self.config['task']['dataset']})")
            
            if self.config.get('data_config_style', 'OCP') == 'OCP':
                bm_logging.info(f" - {self.config['val_dataset']['src']}")
            
            self.val_dataset = dataset_class(self.config["val_dataset"])
            print(f"val_dataset: {self.val_dataset}")

            self.val_sampler = self.get_sampler(
                dataset=self.val_dataset,
                batch_size=eval_local_batch_size,
                shuffle=True,
            )
            print(f"val_sampler: {self.val_sampler}")
            
            self.val_loader = self.get_dataloader(
                dataset=self.val_dataset,
                sampler=self.val_sampler,
                collater=self.parallel_collater,
            )
            print(f"val_loader: {self.val_loader}")

            self.config["optim"]["num_val"] = len(self.val_dataset)
            print("===", self.config["optim"]["num_val"])

        # test set
        if self.config.get("test_dataset", None) and self.flag_loading_dataset:
            bm_logging.info(f"Loading test dataset (type: {self.config['task']['dataset']})")
            
            if self.config.get('data_config_style', 'OCP') == 'OCP':
                bm_logging.info(f" - {self.config['test_dataset']['src']}")
            
            self.test_dataset = dataset_class(self.config["test_dataset"])
            print(f"test_dataset: {self.test_dataset}")

            self.test_sampler = self.get_sampler(
                dataset=self.test_dataset,
                batch_size=eval_local_batch_size,
                shuffle=True,
            )
            print(f"test_sampler: {self.test_sampler}")
            
            self.test_loader = self.get_dataloader(
                dataset=self.test_dataset,
                sampler=self.test_sampler,
                collater=self.parallel_collater,
            )
            print(f"test_loader: {self.test_loader}")
            
            self.config["optim"]["num_test"] = len(self.test_dataset)
            print("===", self.config["optim"]["num_val"])

        # data-related setting
        self._do_data_related_settings()

    @abstractmethod
    def _set_normalizer(self):
        """Energy normalizer needs factors calculated by using forces, thereby that the function is implemented in ForcesTrainer"""

    @abstractmethod
    def _set_task(self):
        """Initialize task-specific information. Derived classes should implement this function."""

    def _set_model(self):
        # Build model
        bm_logging.info(f"Loading model: {self.config['model_name']}")
        model_class = registry.get_model_class(self.config["model_name"])
        self.model = model_class(
            num_atoms = None, # useless
            bond_feat_dim = None, # useless
            num_targets = self.num_targets,
            **self.config["model_attributes"],
        ).to(self.device)

        bm_logging.info(f"Loaded {self.model.__class__.__name__} with {self.model.num_params} parameters.")

        if self.logger:
            self.logger.watch(self.model)

        # wrapping OCPDataParallel even when using single GPU
        self.model = OCPDataParallel(
            module=self.model, # MACE Wrap /data1/project/yewon/MLFF-Framework/src/models/mace/mace.py
            output_device=self.device,
            num_gpus=1 if not self.cpu else 0,
        )
        if distutils.initialized() and not self.config["noddp"]:
            # wrapping pytorch DDP
            if (self.config["model_name"]=='bpnn' 
                or self.config["model_name"]=='allegro'
            ):
                self.model = DistributedDataParallel(
                    module=self.model, 
                    device_ids=[self.device],
                    find_unused_parameters=True # it makes computation somewhat slow
                )
            else:
                self.model = DistributedDataParallel(
                    module=self.model, 
                    device_ids=[self.device],
                )

    def _set_loss(self):
        # initiate loss which is wrapped with DDPLoss in default
        self.loss_fn = {
            "energy": initiate_loss(self.config["optim"].get("loss_energy", "energy_per_atom_mse")),
            "force" : initiate_loss(self.config["optim"].get("loss_force", "force_per_dim_mse")),
            "stress" : initiate_loss(self.config["optim"].get("loss_stress", "mse"))
        }

    def _split_trainable_params_optimizer_weight_decay(self):
        # as in OCP code
        params_decay = []
        params_no_decay = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if "embedding" in name:
                    params_no_decay += [param]
                elif "frequencies" in name:
                    params_no_decay += [param]
                elif "bias" in name:
                    params_no_decay += [param]
                else:
                    params_decay += [param]
        return params_decay, params_no_decay

    def _set_optimizer_and_lr_scheduler(self):
        if self.mode != "train":
            return

        # optimizer
        optimizer = self.config["optim"].get("optimizer", "AdamW")
        optimizer_class = getattr(torch.optim, optimizer)
        # Arguments of optimizer are listed in self.config['optim']['optimizer_params'],
        # except weight decay that is given as self.config['optim']['weight_decay']
        weight_decay = self.config["optim"].get("weight_decay", 0)
        if  weight_decay > 0:
            # Do not regularize bias etc.
            params_decay, params_no_decay = self._split_trainable_params_optimizer_weight_decay()
            self.optimizer = optimizer_class(
                params=[
                    {"params": params_no_decay, "weight_decay": 0},
                    {"params": params_decay, "weight_decay": weight_decay},
                ],
                lr=self.config["optim"]["lr_initial"],
                **self.config["optim"].get("optimizer_params", {}),
            )
        else:
            self.optimizer = optimizer_class(
                params=self.model.parameters(),
                lr=self.config["optim"]["lr_initial"],
                **self.config["optim"].get("optimizer_params", {}),
            )

        # lr scheduler
        self.scheduler = LRScheduler(self.optimizer, self.config["optim"])

    def _set_ema(self):
        # Exponential Moving Average (EMA)
        self.ema = None
        self.ema_decay = self.config["optim"].get("ema_decay", None)
        if self.ema_decay is not None:
            self.ema = ExponentialMovingAverage(
                self.model.parameters(),
                self.ema_decay,
            )

    def _generate_evaluator(self):
        task_metrics = self.config["task"].get("metrics", None) # check list!
        task_attributes = self.config["task"].get("attributes", None)
        task_primary_metric = self.config["task"].get("primary_metric", None)

        return MetricEvaluator(
            task=self.task_name, 
            task_metrics=task_metrics,
            task_attributes=task_attributes,
            task_primary_metric=task_primary_metric,
            device=self.device,
        )

    def _set_evaluator(self):
        # build an evaluator
        self.evaluator = self._generate_evaluator()

    def _set_extras(self):
        self.clip_grad_norm = self.config["optim"].get("clip_grad_norm", None)

    def load_checkpoint(self, checkpoint_path):
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                errno.ENOENT, "Checkpoint file not found", checkpoint_path
            )

        log_temp_str = ""
        if self.mode == "train":
            if self.config["cmd"]["resume"]:
                log_temp_str = "resuming the training"
            else:
                log_temp_str = "finetuning"
        if self.mode == "validate":
            log_temp_str = "validation"
        bm_logging.info(f"Loading checkpoint from {checkpoint_path} (used for `{log_temp_str}`)")
        
        map_location = torch.device("cpu") if self.cpu else self.device
        checkpoint = torch.load(checkpoint_path, map_location=map_location)

        # load the training config saved in the checkpoint if resuming the training
        if self.config["cmd"]["resume"]:
            self.config = checkpoint.get("config", None)
            assert self.config is not None

            self.epoch = checkpoint.get("epoch", 0)
            self.step = checkpoint.get("step", 0)
            self.best_val_metric = checkpoint.get("best_val_metric", None)
            self.primary_metric = checkpoint.get("primary_metric", None)

            if "optimizer" in checkpoint and self.mode == "train":
                self.optimizer.load_state_dict(checkpoint["optimizer"])

            if "scheduler" in checkpoint and checkpoint["scheduler"] is not None and self.mode == "train":
                self.scheduler.scheduler.load_state_dict(checkpoint["scheduler"])

        # Match the "module." count in the keys of model and checkpoint state_dict
        # DataParallel model has 1 "module.",  DistributedDataParallel has 2 "module."
        # Not using either of the above two would have no "module."
        ckpt_key_count = next(iter(checkpoint["state_dict"])).count("module")
        mod_key_count = next(iter(self.model.state_dict())).count("module")
        key_count_diff = mod_key_count - ckpt_key_count
        if key_count_diff > 0:
            new_dict = {
                key_count_diff * "module." + k: v
                for k, v in checkpoint["state_dict"].items()
            }
        elif key_count_diff < 0:
            new_dict = {
                k[len("module.") * abs(key_count_diff) :]: v
                for k, v in checkpoint["state_dict"].items()
            }
        else:
            new_dict = checkpoint["state_dict"]

        strict = self.config["task"].get("strict_load", True)
        load_state_dict(self.model, new_dict, strict=strict)
            
        if "ema" in checkpoint and checkpoint["ema"] is not None:
            self.ema.load_state_dict(checkpoint["ema"])
        else:
            self.ema = None

        scale_dict = checkpoint.get("scale_dict", None)
        if scale_dict:
            logging.info(
                "Overwriting scaling factors with those loaded from checkpoint. "
                "If you're generating predictions with a pretrained checkpoint, this is the correct behavior. "
                "To disable this, delete `scale_dict` from the checkpoint. "
            )
            load_scales_compat(self._unwrapped_model, scale_dict)

        if len(checkpoint["normalizers"]) > 0:
            for key in checkpoint["normalizers"]:
                if key in self.normalizers:
                    self.normalizers[key].load_state_dict(checkpoint["normalizers"][key])
                else:
                    bm_logging.warning(f"{key} in checkpoint cannot be used")
            log_and_check_normalizers(self.normalizers["target"], self.normalizers["grad_target"], loaded=True)

        if self.scaler and checkpoint["amp"]:
            self.scaler.load_state_dict(checkpoint["amp"])

        if self.config["model_name"] == "bpnn":
            self._unwrapped_model.pca = checkpoint["pca"]

    def make_checkpoint_dict(self, metrics, training_state):
        if self.config["model_name"] == "bpnn":
            if "dataset_path" in self.config["model_attributes"]:
                del self.config["model_attributes"]["dataset_path"]
                del self.config["model_attributes"]["pca_path"]

        normalizers = {key: value.state_dict() for key, value in self.normalizers.items()}
        amp_scaler = self.scaler.state_dict() if self.scaler else None

        if training_state:
            ckpt_dict = {
                "epoch": self.epoch,
                "step": self.step,
                "state_dict": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.scheduler.state_dict(),
                "normalizers": normalizers,
                "config": self.config,
                "val_metrics": metrics,
                "ema": self.ema.state_dict() if self.ema else None,
                "amp": amp_scaler,
                "best_val_metric": self.best_val_metric,
                "primary_metric": self.config["task"].get(
                    "primary_metric",
                    self.evaluator.task_primary_metric[self.task_name],
                ),
            }
        else:
            ckpt_dict = {
                "state_dict": self.model.state_dict(),
                "normalizers": normalizers,
                "config": self.config,
                "val_metrics": metrics,
                "amp": amp_scaler,
                "trained_epochs": self.epoch+1,
            }
        
        if self.config["model_name"] == "bpnn":
            ckpt_dict["pca"] = self._unwrapped_model.pca

        return ckpt_dict

    def save(
        self,
        metrics=None,
        checkpoint_file="checkpoint.pt",
        training_state=True,
    ):
        if not self.is_debug and distutils.is_master():
            if (not training_state) and self.ema:
                self.ema.store()
                self.ema.copy_to()

            ckpt_path = save_checkpoint(
                self.make_checkpoint_dict(metrics, training_state),
                checkpoint_dir=self.config["cmd"]["checkpoint_dir"],
                checkpoint_file=checkpoint_file,
            )

            if (not training_state) and self.ema:
                self.ema.restore()
            return ckpt_path
        else:
            return None

    @abstractmethod
    def train(self):
        """Derived classes should implement this function."""

    @torch.no_grad()
    def validate(self, split="val", loader=None, log_flag=True):
        if loader is None:
            # set a dataloader corresponding to the given data split
            if split == "train":
                loader = self.train_loader
            elif split == "val":
                loader = self.val_loader
            elif split == "test":
                loader = self.test_loader
            else:
                raise ValueError(f"Split {split} is not supported")

        # set the model as the eval mode
        ensure_fitted(self._unwrapped_model, warn=True)
        self.model.eval()
        if self.ema:
            self.ema.store()
            self.ema.copy_to()

        # evaluate
        rank = distutils.get_rank()
        metrics = {}
        evaluator = self._generate_evaluator()
        for i, batch in tqdm(
            enumerate(loader),
            total=len(loader),
            position=rank,
            desc="device {}".format(rank),
            disable=not self.config["cmd"].get("show_eval_progressbar", False),
        ):
            # Forward.
            with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                out = self._forward(batch)
            loss = self._compute_loss(out, batch)

            # Compute metrics.
            metrics = self._compute_metrics(out, batch, evaluator, metrics)
            metrics = evaluator.update("loss", loss.item(), metrics)

        # aggregate evaluation results
        aggregated_metrics = evaluator.aggregate(metrics)

        # logging
        log_dict = {k: aggregated_metrics[k]["metric"] for k in aggregated_metrics}
        log_dict.update({"epoch": self.epoch})
        if log_flag:
            # stdout logging
            bm_logging.info(f"[{split}] " + parse_logs(log_dict))
            
            # save evaluation metrics by the logger.
            if self.logger:
                self.logger.log(log_dict, step=self.step, split=split)

        if self.ema:
            self.ema.restore()

        return aggregated_metrics

    @abstractmethod
    def _forward(self, batch_list):
        """Derived classes should implement this function."""

    @abstractmethod
    def _compute_loss(self, out, batch_list):
        """Derived classes should implement this function."""

    @abstractmethod
    def _compute_metrics(self, out, batch_list, evaluator, metrics={}):
        """Derived classes should implement this function."""

    def _backward(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        # Scale down the gradients of shared parameters
        if hasattr(self.model.module, "shared_parameters"):
            for p, factor in self.model.module.shared_parameters:
                if hasattr(p, "grad") and p.grad is not None:
                    p.grad.detach().div_(factor)
                else:
                    if not hasattr(self, "warned_shared_param_no_grad"):
                        self.warned_shared_param_no_grad = True
                        logging.warning(
                            "Some shared parameters do not have a gradient. "
                            "Please check if all shared parameters are used "
                            "and point to PyTorch parameters."
                        )

        if self.clip_grad_norm:
            if self.scaler:
                self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.clip_grad_norm,
            )

        if self.scaler:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        if self.ema:
            self.ema.update()

    def create_metric_table(self, display_meV=True, dataloaders=None):
        table = PrettyTable()
        field_names = ["dataset"]
        for metric_name in self.evaluator.metric_fn:
            if "mse" in metric_name:
                # mse -> rmse for printing
                field_names.append(metric_name.replace("mse", "rmse"))
            else:
                field_names.append(metric_name)
        table.field_names = field_names
        
        if dataloaders is None:
            dataloaders = {"train": self.train_loader}
            if self.val_loader:
                dataloaders["val"] = self.val_loader
            if self.test_loader:
                dataloaders["test"] = self.test_loader

        for dataname, dataloader in dataloaders.items():
            bm_logging.info(f"Evaluating on {dataname} ...")
            metrics = self.validate(split=dataname, loader=dataloader, log_flag=False)
            table_row_metrics = [dataname]
            for metric_name in self.evaluator.metric_fn:
                if display_meV and "mae" in metric_name:
                    table_row_metrics.append(f"{metrics[metric_name]['metric'] * 1000:.1f}")
                elif display_meV and "mse" in metric_name:
                    # mse
                    # table_row_metrics.append(f"{metrics[metric_name]['metric'] * 1000000:.1f}") 
                    # rmse
                    table_row_metrics.append(f"{math.sqrt(metrics[metric_name]['metric']) * 1000:.1f}")
                else:
                    table_row_metrics.append(f"{metrics[metric_name]['metric']:.1f}")
            table.add_row(table_row_metrics)

        return table

    def _end_train(self):
        if self.train_dataset:
            self.train_dataset.close_db()
        if self.val_dataset:
            self.val_dataset.close_db()
        if self.test_dataset:
            self.test_dataset.close_db()