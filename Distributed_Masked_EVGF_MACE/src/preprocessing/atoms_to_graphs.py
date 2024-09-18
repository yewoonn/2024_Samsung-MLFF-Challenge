"""
Copyright (C) 2023 Samsung Electronics Co. LTD

This software is a property of Samsung Electronics.
No part of this software, either material or conceptual may be copied or distributed, transmitted,
transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
electronic, mechanical, manual or otherwise, or disclosed
to third parties without the express written permission of Samsung Electronics.
"""

import numpy as np
from pymatgen.io.ase import AseAtomsAdaptor

from ocpmodels.preprocessing.atoms_to_graphs import AtomsToGraphs

import ase.db.sqlite
import ase.io.trajectory
import numpy as np
import torch
from torch_geometric.data import Data
from ocpmodels.common.utils import collate
from ase.stress import voigt_6_to_full_3x3_stress, full_3x3_to_voigt_6_stress

class AtomsToGraphsWithTolerance(AtomsToGraphs):
    def __init__(
        self,
        max_neigh=200,
        radius=6,
        r_energy=False,
        r_forces=False,
        r_stress=False,
        r_distances=False,
        r_edges=True,
        r_fixed=True,
        r_pbc=False,
        tolerance=1e-8,
        favorable_w=0.7,   # w 값: Hf-O 또는 O-Hf 비율
        distance_interval=0.5,  # 거리 구간을 0.5 Å 단위로 설정
        max_distance=15.0       # 최대 거리 설정 (rmax)
    ):
        super().__init__(
            max_neigh=max_neigh,
            radius=radius,
            r_energy=r_energy,
            r_forces=r_forces,
            r_distances=r_distances,
            r_edges=r_edges,
            r_fixed=r_fixed,
            r_pbc=r_pbc,
        )
        self.r_stress = r_stress
        self.tolerance = tolerance
        self.favorable_w = favorable_w  # Hf-O 비율 (w)
        self.distance_interval = distance_interval
        self.max_distance = max_distance
    
    def _get_neighbors_pymatgen(self, atoms):
        """Preforms nearest neighbor search and returns edge index, distances, and cell offsets"""
        struct = AseAtomsAdaptor.get_structure(atoms)
        _c_index, _n_index, _offsets, n_distance = struct.get_neighbor_list(
            r=self.radius, numerical_tol=self.tolerance, exclude_self=True
        )

        _nonmax_idx = []
        for i in range(len(atoms)):
            idx_i = (_c_index == i).nonzero()[0]
            # sort neighbors by distance, remove edges larger than max_neighbors
            idx_sorted = np.argsort(n_distance[idx_i])[: self.max_neigh]
            _nonmax_idx.append(idx_i[idx_sorted])
        _nonmax_idx = np.concatenate(_nonmax_idx)

        _c_index = _c_index[_nonmax_idx]
        _n_index = _n_index[_nonmax_idx]
        n_distance = n_distance[_nonmax_idx]
        _offsets = _offsets[_nonmax_idx]

        return _c_index, _n_index, n_distance, _offsets
    
    def allocate_neighbors_by_distance(self, atoms, _c_index, _n_index, n_distance):
        """거리에 따라 Hf-Hf, Hf-O/O-Hf, O-O 이웃을 할당하여 분포를 계산"""
        atomic_numbers = atoms.get_atomic_numbers()
        Hf_atomic_number = 72
        O_atomic_number = 8

        # 거리 구간 생성: 0.5Å 단위로 최대 15Å까지
        distance_bins = np.arange(0, self.max_distance + self.distance_interval, self.distance_interval)
        num_bins = len(distance_bins) - 1

        # 이웃 분포 저장을 위한 리스트
        distance_binned_neighbors = {
            "Hf-Hf": np.zeros(num_bins),
            "O-O": np.zeros(num_bins),
            "Hf-O/O-Hf": np.zeros(num_bins)
        }

        # 각 거리 구간별로 이웃 분포 계산
        for i in range(len(atoms)):
            idx_i = (_c_index == i).nonzero()[0]
            for neighbor_idx in idx_i:
                dist = n_distance[neighbor_idx]
                bin_idx = np.digitize(dist, distance_bins) - 1

                if bin_idx >= 0 and bin_idx < num_bins:  # 구간 내에 있으면 처리
                    n_atom = _n_index[neighbor_idx]

                    if atomic_numbers[i] == Hf_atomic_number and atomic_numbers[n_atom] == Hf_atomic_number:
                        distance_binned_neighbors["Hf-Hf"][bin_idx] += 1
                    elif atomic_numbers[i] == O_atomic_number and atomic_numbers[n_atom] == O_atomic_number:
                        distance_binned_neighbors["O-O"][bin_idx] += 1
                    else:  # Hf-O or O-Hf
                        distance_binned_neighbors["Hf-O/O-Hf"][bin_idx] += 1

        return distance_binned_neighbors, distance_bins


    def allocate_neighbors(self, atoms, _c_index, _n_index, n_distance):
        """전체 거리 분포에 맞추어 Hf-Hf, O-O, Hf-O/O-Hf 이웃을 배분"""
        
        atomic_numbers = atoms.get_atomic_numbers()
        Hf_atomic_number = 72
        O_atomic_number = 8

        # 거리 구간 생성: 0.5Å 단위로 최대 15Å까지
        distance_bins = np.arange(0, self.max_distance + self.distance_interval, self.distance_interval)
        num_bins = len(distance_bins) - 1

        # 이웃 분포 저장을 위한 리스트
        distance_binned_neighbors = {
            "Hf-Hf": np.zeros(num_bins),
            "O-O": np.zeros(num_bins),
            "Hf-O/O-Hf": np.zeros(num_bins)
        }

        # 모든 이웃을 연결한 후 거리 분포 계산
        for i in range(len(atoms)):
            idx_i = (_c_index == i).nonzero()[0]
            for neighbor_idx in idx_i:
                dist = n_distance[neighbor_idx]
                bin_idx = np.digitize(dist, distance_bins) - 1

                if bin_idx >= 0 and bin_idx < num_bins:  # 구간 내에 있으면 처리
                    n_atom = _n_index[neighbor_idx]

                    if atomic_numbers[i] == Hf_atomic_number and atomic_numbers[n_atom] == Hf_atomic_number:
                        distance_binned_neighbors["Hf-Hf"][bin_idx] += 1
                    elif atomic_numbers[i] == O_atomic_number and atomic_numbers[n_atom] == O_atomic_number:
                        distance_binned_neighbors["O-O"][bin_idx] += 1
                    else:  # Hf-O or O-Hf
                        distance_binned_neighbors["Hf-O/O-Hf"][bin_idx] += 1

        selected_edges = []  # 최종 이웃 선택 저장소

        # 거리 분포를 바탕으로 최대 50개의 이웃을 각 노드에 배분
        for i in range(len(atoms)):
            idx_i = (_c_index == i).nonzero()[0]
            sorted_idx = np.argsort(n_distance[idx_i])  # 거리순으로 정렬
            idx_i = idx_i[sorted_idx]  # 정렬된 이웃들
            
            hf_hf_edges, o_o_edges, hf_o_edges, o_hf_edges = [], [], [], []

            # 각 이웃을 Hf-Hf, O-O, Hf-O/O-Hf로 분류
            for neighbor_idx in idx_i:
                n_atom = _n_index[neighbor_idx]
                if atomic_numbers[i] == Hf_atomic_number and atomic_numbers[n_atom] == Hf_atomic_number:
                    hf_hf_edges.append(neighbor_idx)
                elif atomic_numbers[i] == O_atomic_number and atomic_numbers[n_atom] == O_atomic_number:
                    o_o_edges.append(neighbor_idx)
                elif atomic_numbers[i] == Hf_atomic_number and atomic_numbers[n_atom] == O_atomic_number:
                    hf_o_edges.append(neighbor_idx)
                elif atomic_numbers[i] == O_atomic_number and atomic_numbers[n_atom] == Hf_atomic_number:
                    o_hf_edges.append(neighbor_idx)

            # 이웃을 최대 50개까지 거리 분포에 맞게 배정
            total_neighbors = min(self.max_neigh, len(idx_i))  # 최대 50개까지 이웃 배정
            selected_edges_i = []

            for bin_idx in range(num_bins):
                total_neighbors_in_bin = int(np.sum([  # 정수로 변환
                    distance_binned_neighbors["Hf-Hf"][bin_idx],
                    distance_binned_neighbors["O-O"][bin_idx],
                    distance_binned_neighbors["Hf-O/O-Hf"][bin_idx]
                ]))

                if total_neighbors_in_bin > 0:
                    # 비율 계산 (총 이웃 중 Hf-Hf, Hf-O/O-Hf, O-O 비율 적용)
                    num_hf_hf = int(total_neighbors_in_bin * (1 - self.favorable_w))
                    num_hf_o = int(total_neighbors_in_bin - num_hf_hf)  # 정수 변환

                    # Hf 원자일 경우 Hf-Hf와 Hf-O의 비율에 따라 이웃 선택
                    if atomic_numbers[i] == Hf_atomic_number:
                        selected_hf_hf = hf_hf_edges[:num_hf_hf]
                        selected_hf_o = hf_o_edges[:num_hf_o]
                        selected_edges_i = np.concatenate([selected_hf_hf, selected_hf_o])

                    # O 원자일 경우 O-O와 O-Hf의 비율에 따라 이웃 선택
                    elif atomic_numbers[i] == O_atomic_number:
                        num_o_o = int(total_neighbors_in_bin * (1 - self.favorable_w))
                        num_o_hf = int(total_neighbors_in_bin - num_o_o)  # 정수 변환
                        selected_o_o = o_o_edges[:num_o_o]
                        selected_o_hf = o_hf_edges[:num_o_hf]
                        selected_edges_i = np.concatenate([selected_o_o, selected_o_hf])

            # 선택된 이웃을 저장
            if len(selected_edges_i) > total_neighbors:
                selected_edges_i = selected_edges_i[:total_neighbors]

            selected_edges.append(selected_edges_i)

        # 선택된 이웃을 모두 정수형으로 변환하여 결합
        selected_edges = np.concatenate(selected_edges).astype(int)
        return _c_index[selected_edges], _n_index[selected_edges], n_distance[selected_edges]


    def convert(self, atoms):
        """ASE 원자 데이터를 변환하고 이웃을 거리 구간별로 Hf-Hf, Hf-O, O-O, O-Hf 비율에 맞게 배분"""
        # ASE 데이터를 그래프 데이터로 변환
        data = super().convert(atoms)
        
        if self.r_stress:
            stress = torch.Tensor(voigt_6_to_full_3x3_stress(atoms.get_stress(apply_constraint=False))).unsqueeze(0)
            data.stress = stress

        _c_index, _n_index, n_distance, _offsets = self._get_neighbors_pymatgen(atoms)
        _c_index, _n_index, n_distance = self.allocate_neighbors(atoms, _c_index, _n_index, n_distance)

        data.edge_index = torch.tensor(np.vstack([_c_index, _n_index]), dtype=torch.long)
        data.edge_attr = torch.tensor(n_distance, dtype=torch.float)
        
        return data
