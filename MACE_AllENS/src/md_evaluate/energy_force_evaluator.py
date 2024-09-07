"""
Copyright (C) 2023 Samsung Electronics Co. LTD

This software is a property of Samsung Electronics.
No part of this software, either material or conceptual may be copied or distributed, transmitted,
transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
electronic, mechanical, manual or otherwise, or disclosed
to third parties without the express written permission of Samsung Electronics.
"""

import torch
import numpy as np
import pandas as pd
import math
import time

from prettytable import PrettyTable
from pathlib import Path
from tqdm import tqdm

import ase
import ase.io
from ase.calculators.singlepoint import SinglePointCalculator

from src.common.utils import bm_logging
from src.common.registry import md_evaluate_registry
from src.modules.metric_evaluator import MetricEvaluator
from src.md_evaluate.base_evaluator import BaseEvaluator


@md_evaluate_registry.register_md_evaluate("ef")
@md_evaluate_registry.register_md_evaluate("energy_force")
class EnergyForceEvaluator(BaseEvaluator):
    def __init__(self, config):
        super().__init__(config)
        
        self.ref_traj_path = Path(self.config["reference_trajectory"])
        self.display_meV = True

        self.save_ef = self.config["save_ef"]
        if self.save_ef:
            self.save_traj_path = self.ref_traj_path.parent / f"pred_by_{self.calculator.model_name}_{self.ref_traj_path.name}"
            bm_logging.info(f"The output trajectory filling with the predicted energy and forces is saved at {self.save_traj_path}")
        
        # CSV 파일 경로 설정
        self.csv_path = self.ref_traj_path.parent / f"pred_by_{self.calculator.model_name}.csv"
        bm_logging.info(f"Prediction results will be saved as CSV at {self.csv_path}")
        self.csv_data = []
        
        self.measure_time = self.config["measure_time"]

        self.metric_evaluator = MetricEvaluator(
            task="s2ef", 
            task_metrics=["energy_per_atom_mae", "energy_per_atom_mse", "forces_mae", "forces_mse"],
            device=self.device,
        )

    def evaluate(self):
        ref_traj = ase.io.read(self.ref_traj_path, index=":", format="extxyz")

        if self.measure_time:
            # for warm start using 10 samples
            for atoms in ref_traj[:10]:
                self.calculator.calculate(atoms=atoms)
            time_per_sample = []
            time_per_sample_data_preparation = []
            time_per_sample_model_inference = []
            time_per_atom = []

        bm_logging.info("Start evaluation to predict energy and forces")
        metrics = {}
        # print("calculator :", self.calculator)
        for i, atoms in enumerate(tqdm(ref_traj)):
            target = {
                "energy": torch.tensor(atoms.get_total_energy(), dtype=torch.float32),
                "forces": torch.tensor(atoms.get_forces()),
                "natoms": torch.tensor(atoms.get_forces().shape[0], dtype=torch.long),
            }

            start_calc = time.time()
            self.calculator.calculate(atoms=atoms, measure_time=self.measure_time)
            end_calc = time.time()

            pred = {
                "energy": torch.tensor(self.calculator.results["energy"], dtype=torch.float32),
                "forces": torch.tensor(self.calculator.results["forces"]),
                "energy_uncertainty": torch.tensor(self.calculator.results["energy_uncertainty"], dtype=torch.float32),
                "natoms": target["natoms"],
            }
            metrics = self.metric_evaluator.eval(pred, target, prev_metrics=metrics)

            if self.save_ef:
                pred_for_save = {
                    "energy": pred["energy"].cpu().item(),
                    "forces": pred["forces"].cpu().numpy(),
                }
                atoms.calc = SinglePointCalculator(atoms, **pred_for_save)
                ase.io.write(self.save_traj_path, images=atoms, format="extxyz", append=True)

            # CSV 데이터 저장
            self.csv_data.append({
                'ID': f'TEST_{i:04d}',
                'energy': pred["energy"].cpu().item(),
                'energy_uncertainty': pred["energy_uncertainty"].cpu().item(),
                'forces': str(pred["forces"].cpu().numpy().tolist())  # forces를 문자열로 변환하여 저장
            })

            if self.measure_time:
                time_per_sample.append(end_calc - start_calc)
                time_per_atom.append((end_calc - start_calc)/target["natoms"])
                time_per_sample_data_preparation.append(self.calculator.time_data_preparation)
                time_per_sample_model_inference.append(self.calculator.time_model_inference)

        # CSV 파일 저장
        df = pd.DataFrame(self.csv_data)
        df.to_csv(self.csv_path, index=False)

        # 평가 메트릭 출력
        table = PrettyTable()
        field_names = ["dataset"]
        for metric_name in self.metric_evaluator.metric_fn:
            if "mse" in metric_name:
                # mse -> rmse for printing
                field_names.append(metric_name.replace("mse", "rmse"))
            else:
                field_names.append(metric_name)
        table.field_names = field_names
        table_row_metrics = [self.ref_traj_path.name]
        for metric_name in self.metric_evaluator.metric_fn:
            if self.display_meV:
                if "mae" in metric_name:
                    table_row_metrics.append(f"{metrics[metric_name]['metric'] * 1000:.1f}")
                elif "mse" in metric_name:
                    # mse는 rmse로 출력
                    table_row_metrics.append(f"{math.sqrt(metrics[metric_name]['metric']) * 1000:.1f}")
            else:
                table_row_metrics.append(f"{metrics[metric_name]['metric']:.1f}")
        table.add_row(table_row_metrics)
        bm_logging.info(f"\n{table}")

        # 시간 측정 출력
        if self.measure_time:
            def print_log(time_type, avg_time):
                if avg_time > 1.0:
                    bm_logging.info(f"{time_type}: {avg_time:.2f} s")
                elif avg_time > 0.001:
                    bm_logging.info(f"{time_type}: {avg_time*1000:.2f} ms")
                else:
                    bm_logging.info(f"{time_type}: {avg_time*1000000:.2f} us")

            print_log("Time per sample", np.mean(time_per_sample))
            print_log("-- data preparation", np.mean(time_per_sample_data_preparation))
            print_log("-- model inference", np.mean(time_per_sample_model_inference))
            print_log("Time per atom", np.mean(time_per_atom))