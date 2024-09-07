"""
Copyright (C) 2023 Samsung Electronics Co. LTD

This software is a property of Samsung Electronics.
No part of this software, either material or conceptual may be copied or distributed, transmitted,
transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
electronic, mechanical, manual or otherwise, or disclosed
to third parties without the express written permission of Samsung Electronics.
"""

import os

import time
from collections import OrderedDict

import torch
from torch_geometric.data import Data, Batch

import numpy as np

from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.singlepoint import SinglePointCalculator

from ocpmodels.common.registry import registry
from ocpmodels.common.utils import load_state_dict
from ocpmodels.modules.normalizer import Normalizer

from mace import data as mace_data
from mace.tools.torch_geometric.batch import Batch as BatchMACE

from src.common.collaters.parallel_collater_mace import convert_ocp_Data_into_mace_AtomicData
from src.common.utils import bm_logging
from src.modules.normalizer import NormalizerPerAtom, log_and_check_normalizers
from src.preprocessing.atoms_to_graphs import AtomsToGraphsWithTolerance


class BenchmarkCalculator(Calculator):
    
    implemented_properties = ["energy", "forces", "energy_uncertainty"]
    
    def __init__(self, ckpt_list=None, device=torch.device("cpu"), **kwargs):
        """
        ckpt_list: 여러 체크포인트 리스트를 받아서 앙상블 예측을 수행할 수 있도록 수정
        """
        Calculator.__init__(self, **kwargs)
        self.device = device

        assert ckpt_list is not None and isinstance(ckpt_list, list)
        self.models = []

        # 여러 개의 체크포인트로부터 모델을 로드
        for ckpt_path in ckpt_list:
            # 체크포인트가 경로일 경우 파일로부터 로드
            if isinstance(ckpt_path, str):
                if not os.path.isfile(ckpt_path):
                    raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")
                ckpt = torch.load(ckpt_path, map_location="cpu")
            else:
                ckpt = ckpt_path

            # 체크포인트 로드 후 모델 구성
            ckpt_config = ckpt["config"]
            self.model_name = ckpt_config["model_name"]

            # 모델 생성
            model_class = registry.get_model_class(self.model_name)
            model = model_class(
                num_atoms=None,  # not used
                bond_feat_dim=None,  # not used
                num_targets=1,  # always 1 (for energy)
                **ckpt_config["model_attributes"],
            )

            # 모델 파라미터 로드
            model_state_dict = OrderedDict()
            for key, val in ckpt["state_dict"].items():
                k = key
                while k.startswith("module."):
                    k = k[7:]
                model_state_dict[k] = val
            load_state_dict(module=model, state_dict=model_state_dict, strict=True)

            # 모델을 GPU로 이동
            model = model.to(self.device)
            model.eval()  # 평가 모드 설정
            self.models.append(model)

        # 일반적인 설정 (첫 번째 모델 기준으로 설정)
        self.normalizers = {}
        if ckpt_config.get("data_config_style", "OCP") == "OCP":
            normalizer = ckpt_config["dataset"]
        else:
            assert "normalizer" in ckpt_config.keys()
            normalizer = ckpt_config["normalizer"]

        if normalizer.get("normalize_labels", False):
            self.normalization_per_atom = normalizer.get("per_atom", False)
            if self.normalization_per_atom:
                self.normalizers["target"] = NormalizerPerAtom(mean=0.0, std=1.0, device=self.device)
            else:
                self.normalizers["target"] = Normalizer(mean=0.0, std=1.0, device=self.device)
            self.normalizers["target"].load_state_dict(ckpt["normalizers"]["target"])
            self.normalizers["grad_target"] = Normalizer(mean=0.0, std=1.0, device=self.device)
            self.normalizers["grad_target"].load_state_dict(ckpt["normalizers"]["grad_target"])
            log_and_check_normalizers(self.normalizers["target"], self.normalizers["grad_target"], loaded=True)

        # ASE -> PyG 변환 준비
        self.cutoff = self.models[0].cutoff  # 첫 번째 모델의 cutoff 사용
        self.max_neighbors = self.models[0].max_neighbors
        self.pbc = torch.tensor([self.models[0].use_pbc] * 3)

        self.atoms_to_pyg_data = AtomsToGraphsWithTolerance(
            max_neigh=self.max_neighbors,
            radius=self.cutoff,
            r_energy=False,
            r_forces=False,
            r_fixed=True,
            r_distances=False,
            r_pbc=self.models[0].use_pbc,
            r_edges=(not self.models[0].otf_graph),
            tolerance=1e-8,
        )

    def convert_atoms_to_ocp_batch(self, atoms):
        # convert ase.Atoms into pytorch_geometric data
        data = self.atoms_to_pyg_data.convert(atoms)
        batch = Batch.from_data_list([data]) # batch size = 1
        if not self.model.otf_graph:
            batch.neighbors = torch.tensor([data.edge_index.shape[1]])
        return batch


    def convert_atoms_to_mace_batch(self, atoms):
        # convert ase.Atoms into AtomicData of MACE
        data = self.atoms_to_pyg_data.convert(atoms)
        data = convert_ocp_Data_into_mace_AtomicData(data, self.models[0].z_table)
        batch = BatchMACE.from_data_list([data]) # batch size = 1
        return batch
    
    def convert_atoms_to_batch(self, atoms):
        # ASE Atoms -> PyG 데이터 변환 (기존 로직 유지)
        data = self.atoms_to_pyg_data.convert(atoms)
        batch = Batch.from_data_list([data])  # batch size = 1
        if not self.models[0].otf_graph:
            batch.neighbors = torch.tensor([data.edge_index.shape[1]])
        return batch

    def denormalization(self, energy, forces):
        if self.model_name in ["nequip", "allegro", "mace"]:
            # the model output is already in real unit in evaluation mode
            return energy, forces
        else:
            if len(self.normalizers) > 0:
                # if normalization is used in the trainer, model output values should be de-normalized.
                # (because ground truth trained by a model was normalized)
                if self.normalization_per_atom:
                    # Inference is performed on a single snapshot
                    # Thus, for N in denorm(), we use forces.shape[0] which is num of atoms of this snapshot.
                    energy = self.normalizers["target"].denorm(energy, forces.shape[0])
                else:
                    energy = self.normalizers["target"].denorm(energy)
                forces = self.normalizers["grad_target"].denorm(forces)
            return energy, forces
    
    # All Ensemble
    def calculate(self, atoms=None, properties=["energy", "forces"], system_changes=all_changes, measure_time=False):
        """
        atoms: ase.Atoms
        """
        # set atoms attribute
        Calculator.calculate(self, atoms=atoms, properties=properties, system_changes=system_changes)

        # ASE -> PyG 변환
        t1 = time.time()
        batch = self.convert_atoms_to_mace_batch(atoms)
        batch = batch.to(self.device)

        # 앙상블 예측 수행
        energy_preds = []
        force_preds = []

        for model in self.models:
            energy, forces = model(batch)
            energy_preds.append(energy.detach().cpu().item())
            force_preds.append(forces.detach().cpu().numpy())

        # 앙상블 평균 및 불확실성 계산
        mean_energy = np.mean(energy_preds)
        energy_uncertainty = np.std(energy_preds)  # 불확실성(표준 편차)
        mean_forces = np.mean(force_preds, axis=0)

        t2 = time.time()

        # 결과 저장
        self.results = {
            "energy": mean_energy,
            "forces": mean_forces,
            "energy_uncertainty": energy_uncertainty,
        }

        # print("results", self.results)

        if measure_time:
            self.time_data_preparation = t2 - t1
