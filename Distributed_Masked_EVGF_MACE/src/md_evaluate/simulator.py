"""
Copyright (C) 2023 Samsung Electronics Co. LTD

This software is a property of Samsung Electronics.
No part of this software, either material or conceptual may be copied or distributed, transmitted,
transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
electronic, mechanical, manual or otherwise, or disclosed
to third parties without the express written permission of Samsung Electronics.
"""

import os
import json
import yaml
import time
import random
import numpy as np
import torch

from pathlib import Path
from tqdm import tqdm

from ase import units, md, io
from ase.build.supercells import make_supercell

from src.common.registry import md_evaluate_registry
from src.md_evaluate.base_evaluator import BaseEvaluator


@md_evaluate_registry.register_md_evaluate("simulator")
class Simulator(BaseEvaluator):

    def __init__(self, config):
        super().__init__(config)
        Simulator.seed_everywhere(self.config.get("seed"))

    @staticmethod
    def seed_everywhere(seed):
        if seed is None:
            return

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _get_simulator(self, atoms, save_dir):
        if self.config["thermostat"].lower() == "langevin":
            simulator = md.langevin.Langevin(
                atoms,
                self.config["timestep_fs"] * units.fs,
                temperature_K=self.config["temperature_K"],
                friction=self.config.get("langevin_thermostat_coeff", 0.001)
            )
        elif self.config["thermostat"].lower() == "nosehoover":
            self.logger.info(
                "NVT sim. using N-H with Q: {}".format(self.config["nh_thermostat_q"]))
            simulator = NoseHoover(
                atoms=atoms,
                timestep=self.config["timestep_fs"] * units.fs,
                temperature=self.config["temperature_K"],
                nvt_q=self.config["nh_thermostat_q"]
            )
        else:
            raise Exception(
                "Please use a supported thermostat, either 'NoseHoover' or 'Langevin'!!")

        traj_obj = io.trajectory.Trajectory(save_dir / 'atoms.traj', mode='w', atoms=atoms)
        simulator.attach(traj_obj.write, interval=self.config["save_freq"])

        logger_obj = md.MDLogger(simulator, atoms, save_dir / 'thermo.log', mode='w')
        simulator.attach(logger_obj, interval=1)

        return simulator

    def get_initial_snapshot(self):
        atoms = io.read(self.config["initial_structure"]["path"],
                        format=self.config["initial_structure"].get("format"))

        if self.config.get("n_super"):
            # make supercell
            atoms.wrap()
            super_atoms = make_supercell(
                atoms,
                [[self.config["n_super"][0], 0, 0],
                 [0, self.config["n_super"][1], 0],
                 [0, 0, self.config["n_super"][2]]],
                wrap=True
            )
            atoms = super_atoms

        # initialize velocities
        md.velocitydistribution.MaxwellBoltzmannDistribution(
            atoms, temperature_K=self.config["temperature_K"])
        md.velocitydistribution.Stationary(atoms)  # zero linear momentum
        md.velocitydistribution.ZeroRotation(atoms)  # zero angular momentum

        # for debugging purposes
        traj_temp = atoms.get_temperature()
        self.logger.debug("temp after initialization: {}".format(traj_temp))

        return atoms

    def get_output_dir(self):
        save_name = self.config["identifier"]
        if self.config.get("n_super"):
            save_name = save_name + "_n_super_" + str(self.config["n_super"])
        save_name = save_name + "_" + str(self.config["seed"])
        self.config['save_name'] = save_name

        out_dir_full = Path(self.config["out_dir"]) / save_name
        out_dir_full.mkdir(parents=True, exist_ok=True)
        return out_dir_full

    def simulate(self):
        atoms = self.get_initial_snapshot()
        n_atoms = atoms.get_global_number_of_atoms()
        atoms.calc = self.calculator

        # reference : https://www2.mpip-mainz.mpg.de/~andrienk/journal_club/thermostats.pdf
        self.config["nh_thermostat_q"] = 3.0 * n_atoms * units.kB * self.config["temperature_K"] \
            * (self.config["nh_relax_timesteps"] * self.config["timestep_fs"] * units.fs)**2

        self.logger.info(
            "Starting MD simulations for {} atoms".format(n_atoms))
        self.logger.info(f"Thermostat coeffs. are  nh_relax_timesteps: {self.config.get('nh_relax_timesteps')},  nh_thermostat_q: {self.config.get('nh_thermostat_q')}")

        out_dir = self.get_output_dir()
        simulator = self._get_simulator(atoms, out_dir)

        start_time = time.time()
        n_steps = int(self.config["simulation_time_ps"] * 1000 / self.config["timestep_fs"])
        for i_step in tqdm(range(n_steps)):
            simulator.run(1)
        elapsed = time.time() - start_time

        test_metrics = {}
        test_metrics['n_steps'] = n_steps
        test_metrics['running_time'] = elapsed
        self.logger.debug(test_metrics)

        with open(out_dir / 'run_time.json', 'a') as f:
            json.dump(test_metrics, f)
        
        with open(out_dir / 'config.yml', 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
