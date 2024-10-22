"""
Copyright (C) 2023 Samsung Electronics Co. LTD

This software is a property of Samsung Electronics.
No part of this software, either material or conceptual may be copied or distributed, transmitted,
transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
electronic, mechanical, manual or otherwise, or disclosed
to third parties without the express written permission of Samsung Electronics.
"""

import os
import logging
import copy
import datetime
import time
import importlib
import yaml
import numpy as np
import torch

from argparse import Namespace
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Optional
from pathlib import Path
from math import log10, floor, isnan

from ocpmodels.common import distutils, gp_utils
from ocpmodels.common.utils import load_config
from ocpmodels.common.registry import registry 

from src.common import distutils as benchmark_distutils
from src.common.config import check_config
from src.common.registry import md_evaluate_registry


def get_device(config):
    assert "local_rank" in config
    if torch.cuda.is_available() and (config.get("gpus", 1) > 0):
        return torch.device(f"cuda:{config['local_rank']}")
    else:
        return torch.device("cpu")


# In benchmark, we mainly use the logger named bm_logging
bm_logging = logging.getLogger("BenchmarkLogging")


# benchmark logger setting considering distributed environment 
# (we can omit 'if is_master()' when using logging)
def setup_benchmark_logging(config):
    root = logging.getLogger()
    bm_logging = logging.getLogger("BenchmarkLogging")
    if distutils.is_master():
        # The initial logging setup is performed by setup_logging() of ocpmodels.common.utils at main.py.
        # We'll follow the logging format.
        log_formatter = root.handlers[0].formatter

        # setup for benchmark logging
        # inherit root logging and remove it
        for handler in root.handlers:
            bm_logging.addHandler(handler)
            root.removeHandler(handler)
        
        if config.get("logger", None) == "files" and config["mode"] == "train":
            # send INFO to a file
            logger_name = config["logger"] if isinstance(config["logger"], str) else config["logger"]["name"]
            logdir = os.path.join(config["run_dir"], "logs", logger_name, config["timestamp_id"])
            os.makedirs(logdir, exist_ok=True)
            log_path = os.path.join(logdir, "experiment.log")
            file_handler = logging.FileHandler(log_path)
            file_handler.setFormatter(log_formatter)
            bm_logging.addHandler(file_handler)
    else:
        # disable logging by other ranks
        for handler in root.handlers:
            root.removeHandler(handler)


# reference : setup_imports() in ocp/ocpmodels/common/utils.py
def setup_benchmark_imports(config=None):
    # First, check if imports are already setup
    has_already_setup = registry.get("imports_benchmark_setup", no_warning=True)
    if has_already_setup:
        return

    has_already_setup = md_evaluate_registry.get("imports_benchmark_setup", no_warning=True)
    if has_already_setup:
        return

    try:
        this_utils_filepath = Path(__file__).resolve().absolute()
        benchmark_root = this_utils_filepath.parent.parent.parent
        logging.info(f"Project root: {benchmark_root}")

        # OCP
        importlib.import_module("ocpmodels.common.logger")
        for key in ["trainers", "datasets", "models", "tasks"]:
            for path in (benchmark_root / "codebases" / "ocp" / "ocpmodels" / key).rglob("*.py"):
                module_name = ".".join(
                    path.absolute()
                    .relative_to(benchmark_root.absolute())
                    .with_suffix("")
                    .parts
                )
                importlib.import_module(module_name)

        # SAIT-MLFF-Framework
        # : re-define classes of trainers and tasks
        importlib.import_module("src.common.logger")
        for key in ["trainers", "datasets", "models", "tasks", "md_evaluate"]:
            for path in (benchmark_root / "src" / key).rglob("*.py"):
                module_name = ".".join(
                    path.absolute()
                    .relative_to(benchmark_root.absolute())
                    .with_suffix("")
                    .parts
                )
                importlib.import_module(module_name)
    finally:
        registry.register("imports_benchmark_setup", True)
        md_evaluate_registry.register("imports_benchmark_setup", True)


# reference : new_trainer_context() in ocp/ocpmodels/common/utils.py
@contextmanager
def new_trainer_context(*, config: Dict[str, Any], args: Namespace):
    @dataclass
    class _TrainingContext:
        config: Dict[str, Any]
        task: "BaseTask"
        trainer: "BaseTrainer"

    def _set_timestamp_id(config):
        # merging timestamp and expr ID when timestamp_id is empty
        timestamp = torch.tensor(int(datetime.datetime.now().timestamp())).to(get_device(config))
        distutils.broadcast(timestamp, 0)
        timestamp = datetime.datetime.fromtimestamp(timestamp).strftime("%Y%m%d_%H%M%S")
        if config["identifier"] in ["", None]:
            return timestamp
        else:
            return f"{config['identifier']}-{timestamp}"
   
    original_config = copy.deepcopy(config)
    if args.distributed:
        benchmark_distutils.setup(config)
        if config["gp_gpus"] is not None:
            gp_utils.setup_gp(config)
    
    if config["timestamp_id"] is None:
        if config["checkpoint"] is None:
            # make timestamp_id not empty
            config["timestamp_id"] = _set_timestamp_id(config)
        else:
            if args.resume:
                # resume the training of the checkpoint
                config["timestamp_id"] = Path(config["checkpoint"]).parent.name
            else:
                # finetune the model from the given checkpoint
                config["timestamp_id"] = _set_timestamp_id(config)

    # check whether arguments which are required to initiate a Trainer class exist in a configuration
    config = check_config(config)
    # config 출력
    # {'dataset': [{'src': 'datasets/HfO/atom_graph_rmax6.0_maxneighbor50/train.lmdb', 'normalize_labels': True, 'per_atom': True, 'normalize_labels_json': 'datasets/HfO/atom_graph_rmax6.0_maxneighbor50/normalize_stats.json'}, {'src': 'datasets/HfO/atom_graph_rmax6.0_maxneighbor50/valid.lmdb'}, {'src': 'datasets/HfO/atom_graph_rmax6.0_maxneighbor50/test.lmdb'}], 'logger': 'files', 'task': {'dataset': 'lmdb', 'description': 'Regressing to energies and forces for DFT trajectories of HfO', 'metrics': ['energy_per_atom_mae', 'energy_per_atom_mse', 'forces_mae', 'forces_mse'], 'primary_metric': 'forces_mse', 'train_on_free_atoms': False, 'eval_on_free_atoms': False}, 'trainer': 'forces_mace', 'model': {'name': 'mace', 'hidden_irreps': '32x0e + 32x1o + 32x2e', 'MLP_irreps': '16x0e', 'num_interactions': 2, 'max_ell': 3, 'correlation': 3, 'num_radial_basis': 8, 'num_cutoff_basis': 5, 'gate': 'silu', 'regress_forces': True, 'cutoff': 6.0, 'max_neighbors': 50, 'otf_graph': False, 'use_pbc': True, 'E0s': 'average', 'scaling': 'rms_forces_scaling', 'shifting': True, 'chemical_symbols': ['O', 'Hf']}, 'optim': {'batch_size': 16, 'eval_batch_size': 16, 'num_workers': 2, 'max_epochs': 200, 'optimizer': 'Adam', 'optimizer_params': {'amsgrad': False}, 'weight_decay': 5e-07, 'scheduler': 'LinearLR', 'lr_initial': 0.01, 'ema_decay': 0.99, 'energy_coefficient': 1, 'force_coefficient': 1, 'loss_energy': 'energy_per_atom_mse', 'loss_force': 'force_per_dim_mse'}, 'mode': 'train', 'identifier': 'train', 'timestamp_id': 'train-20240822_143952', 'seed': 0, 'is_debug': False, 'run_dir': 'train_results/HfO/MACE', 'print_every': 100, 'amp': False, 'checkpoint': None, 'cpu': False, 'submit': False, 'summit': False, 'local_rank': 0, 'distributed_port': 13356, 'world_size': 1, 'distributed_backend': 'nccl', 'noddp': False, 'gp_gpus': None, 'save_ckpt_every_epoch': 20, 'resume': False, 'slurm': {}}

    start_time = time.time()
    try:
        # setup benchmark logging with a file handler
        setup_benchmark_logging(config)
        setup_benchmark_imports(config)

        # construct a trainer instance
        trainer_class = registry.get_trainer_class(config.get("trainer", "forces"))
        # print("trainer",trainer_class)
        # triner 출력
        # <class 'src.trainers.forces_trainer_mace.MACEForcesTrainer'>
        assert trainer_class is not None, "Trainer class is not found"
        trainer = trainer_class(config = config)

        if config["mode"] == "train" and (not config["resume"]):
            # save a training configuration yaml file into checkpoint_dir
            with open(os.path.join(trainer.config["cmd"]["checkpoint_dir"], "config_train.yml"), 'w') as f:
                input_config, _, _ = load_config(args.config_yml)
                yaml.dump(input_config, f)

        # construct a task instance (given a trainer)
        task_cls = registry.get_task_class(config["mode"])
        assert task_cls is not None, "Task is not found"
        task = task_cls(config=original_config)
        print("task", task)
        ctx = _TrainingContext(config=original_config, task=task, trainer=trainer)
        yield ctx
        distutils.synchronize()
    finally:
        total_time = time.time()-start_time
        bm_logging.info(f"Total time taken: {total_time:.1f} sec ({total_time/3600:.1f} h)")
        if args.distributed:
            distutils.cleanup()


@contextmanager
def new_evaluator_context(*, config: Dict[str, Any], args: Namespace):
    @dataclass
    class _EvaluationContext:
        config: Dict[str, Any]
        task: "BaseTask"
        evaluator: "BaseEvaluator"
   
    original_config = copy.deepcopy(config)
    start_time = time.time()
    try:
        # setup benchmark logging with a file handler
        setup_benchmark_logging(config)
        setup_benchmark_imports(config)

        # construct an evaluator or a simulator
        evaluator_class = md_evaluate_registry.get_md_evaluate_class(config["evaluation_metric"])
        assert evaluator_class is not None, f"Evaluator class is not found"
        evaluator = evaluator_class(config = config)

        # construct a task instance
        task_cls = registry.get_task_class(config["mode"])
        assert task_cls is not None, "Task class is not found"
        task = task_cls(config=original_config)
        ctx = _EvaluationContext(config=original_config, task=task, evaluator=evaluator)
        # print("evaluate_task/evaluator", ctx.task, ctx.evaluator)
        yield ctx
    finally:
        total_time = time.time()-start_time
        bm_logging.info(f"Total time taken: {total_time:.1f} sec ({total_time/3600:.1f} h)")
    