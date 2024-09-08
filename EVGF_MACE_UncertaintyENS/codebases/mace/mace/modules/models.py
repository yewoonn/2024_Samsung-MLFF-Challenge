###########################################################################################
# Implementation of MACE models and other models based E(3)-Equivariant MPNNs
# Authors: Ilyes Batatia, Gregor Simm
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

from typing import Any, Callable, Dict, List, Optional, Type

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from e3nn import o3
from e3nn.util.jit import compile_mode

from mace.data import AtomicData
from mace.tools.scatter import scatter_sum

from .blocks import (
    AtomicEnergiesBlock,
    EquivariantProductBasisBlock,
    InteractionBlock,
    LinearNodeEmbeddingBlock,
    LinearReadoutBlock,
    NonLinearReadoutBlock,
    RadialEmbeddingBlock,
    ScaleShiftBlock,
    EdgeVariantGF
)
from .utils import (
    get_edge_vectors_and_lengths,
    get_outputs,
    get_symmetric_displacement,
)


@compile_mode("script")
class MACE(torch.nn.Module):
    def __init__(
        self,
        r_max: float,
        num_bessel: int,
        num_polynomial_cutoff: int,
        max_ell: int,
        interaction_cls: Type[InteractionBlock],
        interaction_cls_first: Type[InteractionBlock],
        num_interactions: int,
        num_elements: int,
        hidden_irreps: o3.Irreps,
        MLP_irreps: o3.Irreps,
        atomic_energies: np.ndarray,
        avg_num_neighbors: float,
        atomic_numbers: List[int],
        correlation: int,
        gate: Optional[Callable],
    ):
        super().__init__()
        self.register_buffer(
            "atomic_numbers", torch.tensor(atomic_numbers, dtype=torch.int64)
        )
        self.register_buffer("r_max", torch.tensor(r_max, dtype=torch.float64))
        self.register_buffer(
            "num_interactions", torch.tensor(num_interactions, dtype=torch.int64)
        )
        # Embedding
        node_attr_irreps = o3.Irreps([(num_elements, (0, 1))])
        node_feats_irreps = o3.Irreps([(hidden_irreps.count(o3.Irrep(0, 1)), (0, 1))])
        self.node_embedding = LinearNodeEmbeddingBlock(
            irreps_in=node_attr_irreps, irreps_out=node_feats_irreps
        )
        # print(self.node_embedding.irreps_in)
        self.radial_embedding = RadialEmbeddingBlock(
            r_max=r_max,
            num_bessel=num_bessel,
            num_polynomial_cutoff=num_polynomial_cutoff,
        )
        edge_feats_irreps = o3.Irreps(f"{self.radial_embedding.out_dim}x0e")

        sh_irreps = o3.Irreps.spherical_harmonics(max_ell)
        num_features = hidden_irreps.count(o3.Irrep(0, 1))
        interaction_irreps = (sh_irreps * num_features).sort()[0].simplify()
        self.spherical_harmonics = o3.SphericalHarmonics(
            sh_irreps, normalize=True, normalization="component"
        )

        # Interactions and readout
        self.atomic_energies_fn = AtomicEnergiesBlock(atomic_energies)

        inter = interaction_cls_first(
            node_attrs_irreps=node_attr_irreps,
            node_feats_irreps=node_feats_irreps,
            edge_attrs_irreps=sh_irreps,
            edge_feats_irreps=edge_feats_irreps,
            target_irreps=interaction_irreps,
            hidden_irreps=hidden_irreps,
            avg_num_neighbors=avg_num_neighbors,
        )
        self.interactions = torch.nn.ModuleList([inter])

        # Use the appropriate self connection at the first layer for proper E0
        use_sc_first = False
        if "Residual" in str(interaction_cls_first):
            use_sc_first = True

        node_feats_irreps_out = inter.target_irreps
        prod = EquivariantProductBasisBlock(
            node_feats_irreps=node_feats_irreps_out,
            target_irreps=hidden_irreps,
            correlation=correlation,
            num_elements=num_elements,
            use_sc=use_sc_first,
        )
        self.products = torch.nn.ModuleList([prod])

        self.readouts = torch.nn.ModuleList()
        self.readouts.append(LinearReadoutBlock(hidden_irreps))

        for i in range(num_interactions - 1):
            if i == num_interactions - 2:
                hidden_irreps_out = str(
                    hidden_irreps[0]
                )  # Select only scalars for last layer
            else:
                hidden_irreps_out = hidden_irreps
            inter = interaction_cls(
                node_attrs_irreps=node_attr_irreps,
                node_feats_irreps=hidden_irreps,
                edge_attrs_irreps=sh_irreps,
                edge_feats_irreps=edge_feats_irreps,
                target_irreps=interaction_irreps,
                hidden_irreps=hidden_irreps_out,
                avg_num_neighbors=avg_num_neighbors,
            )
            self.interactions.append(inter)
            prod = EquivariantProductBasisBlock(
                node_feats_irreps=interaction_irreps,
                target_irreps=hidden_irreps_out,
                correlation=correlation,
                num_elements=num_elements,
                use_sc=True,
            )
            self.products.append(prod)
            if i == num_interactions - 2:
                self.readouts.append(
                    NonLinearReadoutBlock(hidden_irreps_out, MLP_irreps, gate)
                )
            else:
                self.readouts.append(LinearReadoutBlock(hidden_irreps))

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        training: bool = False,
        compute_force: bool = True,
        compute_virials: bool = False,
        compute_stress: bool = False,
        compute_displacement: bool = False,
    ) -> Dict[str, Optional[torch.Tensor]]:
        # Setup
        data["positions"].requires_grad_(True)
        num_graphs = data["ptr"].numel() - 1
        displacement = torch.zeros(
            (num_graphs, 3, 3),
            dtype=data["positions"].dtype,
            device=data["positions"].device,
        )
        if compute_virials or compute_stress or compute_displacement:
            (
                data["positions"],
                data["shifts"],
                displacement,
            ) = get_symmetric_displacement(
                positions=data["positions"],
                unit_shifts=data["unit_shifts"],
                cell=data["cell"],
                edge_index=data["edge_index"],
                num_graphs=num_graphs,
                batch=data["batch"],
            )

        # Atomic energies
        node_e0 = self.atomic_energies_fn(data["node_attrs"])
        e0 = scatter_sum(
            src=node_e0, index=data["batch"], dim=-1, dim_size=num_graphs
        )  # [n_graphs,]

        # Embeddings
        node_feats = self.node_embedding(data["node_attrs"])
        vectors, lengths = get_edge_vectors_and_lengths(
            positions=data["positions"],
            edge_index=data["edge_index"],
            shifts=data["shifts"],
        )
        edge_attrs = self.spherical_harmonics(vectors)
        edge_feats = self.radial_embedding(lengths)

        # Interactions
        energies = [e0]
        node_energies_list = [node_e0]
        for interaction, product, readout in zip(
            self.interactions, self.products, self.readouts
        ):
            node_feats, sc = interaction(
                node_attrs=data["node_attrs"],
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=data["edge_index"],
            )
            node_feats = product(
                node_feats=node_feats,
                sc=sc,
                node_attrs=data["node_attrs"],
            )
            node_energies = readout(node_feats).squeeze(-1)  # [n_nodes, ]
            energy = scatter_sum(
                src=node_energies, index=data["batch"], dim=-1, dim_size=num_graphs
            )  # [n_graphs,]
            energies.append(energy)
            node_energies_list.append(node_energies)

        # Sum over energy contributions
        contributions = torch.stack(energies, dim=-1)
        total_energy = torch.sum(contributions, dim=-1)  # [n_graphs, ]
        node_energy_contributions = torch.stack(node_energies_list, dim=-1)
        node_energy = torch.sum(node_energy_contributions, dim=-1)  # [n_nodes, ]
        # Outputs
        forces, virials, stress = get_outputs(
            energy=total_energy,
            positions=data["positions"],
            displacement=displacement,
            cell=data["cell"],
            training=training,
            compute_force=compute_force,
            compute_virials=compute_virials,
            compute_stress=compute_stress,
        )

        return {
            "energy": total_energy,
            "node_energy": node_energy,
            "contributions": contributions,
            "forces": forces,
            "virials": virials,
            "stress": stress,
            "displacement": displacement,
        }


# @compile_mode("script")
# class ScaleShiftMACE(MACE):
#     def __init__(
#         self,
#         atomic_inter_scale: float,
#         atomic_inter_shift: float,
#         **kwargs,
#     ):
#         super().__init__(**kwargs)
#         self.scale_shift = ScaleShiftBlock(
#             scale=atomic_inter_scale, shift=atomic_inter_shift
#         )

#     def forward(
#         self,
#         data: Dict[str, torch.Tensor],
#         training: bool = False,
#         compute_force: bool = True,
#         compute_virials: bool = False,
#         compute_stress: bool = False,
#         compute_displacement: bool = False,
#     ) -> Dict[str, Optional[torch.Tensor]]:
#         # Setup
#         data["positions"].requires_grad_(True)
#         # print("====================================================")
#         # print("Initial data positions :", data["positions"])
#         print(data['num_nodes'])
#         # print("Initial data positions shape:", data["positions"].shape)

#         num_graphs = data["ptr"].numel() - 1
#         displacement = torch.zeros(
#             (num_graphs, 3, 3),
#             dtype=data["positions"].dtype,
#             device=data["positions"].device,
#         )
#         # print("Displacement tensor shape:", displacement.shape)

#         if compute_virials or compute_stress or compute_displacement:
#             (
#                 data["positions"],
#                 data["shifts"],
#                 displacement,
#             ) = get_symmetric_displacement(
#                 positions=data["positions"],
#                 unit_shifts=data["unit_shifts"],
#                 cell=data["cell"],
#                 edge_index=data["edge_index"],
#                 num_graphs=num_graphs,
#                 batch=data["batch"],
#             )
#         # print("Updated positions shape:", data["positions"].shape)
#         # print("Shifts shape:", data["shifts"].shape)   
#         # print("Displacement after symmetric displacement:", displacement.shape)
        
#         # Atomic energies
#         node_e0 = self.atomic_energies_fn(data["node_attrs"])
#         # print("Node energies shape:", node_e0.shape)

#         e0 = scatter_sum(
#             src=node_e0, index=data["batch"], dim=-1, dim_size=num_graphs
#         )  # [n_graphs,]
#         # print("Summed energies e0 shape:", e0.shape)


#         # Embeddings
#         # print("Node attributes shape before embedding:", data["node_attrs"].shape)
#         node_feats = self.node_embedding(data["node_attrs"])
#         # print("Node features shape after embedding:", node_feats.shape)
#         vectors, lengths = get_edge_vectors_and_lengths(
#             positions=data["positions"],
#             edge_index=data["edge_index"],
#             shifts=data["shifts"],
#         )
#         # print("Vectors shape:", vectors.shape)
#         # print("Lengths shape:", lengths.shape)
        
#         edge_attrs = self.spherical_harmonics(vectors)
#         # print("Edge attributes after spherical harmonics shape:", edge_attrs.shape)
#         edge_feats = self.radial_embedding(lengths)
#         # print("Edge features after radial embedding shape:", edge_feats.shape)
        
#         # Interactions
#         node_es_list = []
#         for interaction, product, readout in zip(
#             self.interactions, self.products, self.readouts
#         ):  
#             node_feats, sc = interaction(
#                 node_attrs=data["node_attrs"],
#                 node_feats=node_feats,
#                 edge_attrs=edge_attrs,
#                 edge_feats=edge_feats,
#                 edge_index=data["edge_index"],
#             )

#             node_feats = product(
#                 node_feats=node_feats, sc=sc, node_attrs=data["node_attrs"]
#             )
#             node_es_list.append(readout(node_feats).squeeze(-1))  # {[n_nodes, ], }
#             # print(f"Readout block output node_es_list:", node_es_list)

#         # Sum over interactions
#         # node_inter_es : 각 노드의 최종 에너지
#         node_inter_es = torch.sum(
#             torch.stack(node_es_list, dim=0), dim=0
#         )  # [n_nodes, ]
#         node_inter_es = self.scale_shift(node_inter_es)


#         # Sum over nodes in graph
#         # inter_e : 그래프 별 상호작용 에너지
#         inter_e = scatter_sum(
#             src=node_inter_es, index=data["batch"], dim=-1, dim_size=num_graphs
#         )  # [n_graphs,]

#         # Add E_0 and (scaled) interaction energy
#         # total_energy : 최종 그래프 에너지, 그래프의 기본 원자 에너지 + 그래프 별 상호작용 에너지
#         total_energy = e0 + inter_e

#         # node_energy : 최종 노드 에너지, 각 노드의 기본 에너지 + 각 노드의 최종 에너지
#         node_energy = node_e0 + node_inter_es
#         # print("Final Node energy shape:", node_energy.shape)


#         forces, virials, stress = get_outputs(
#             energy=inter_e,
#             positions=data["positions"],
#             displacement=displacement,
#             cell=data["cell"],
#             training=training,
#             compute_force=compute_force,
#             compute_virials=compute_virials,
#             compute_stress=compute_stress,
#         )
#         # print("Forces shape:", forces.shape if forces is not None else "None")
#         # print("Virials shape:", virials.shape if virials is not None else "None")
#         # print("Stress shape:", stress.shape if stress is not None else "None")

#         output = {
#             "energy": total_energy,
#             "node_energy": node_energy,
#             "interaction_energy": inter_e,
#             "forces": forces,
#             "virials": virials,
#             "stress": stress,
#             "displacement": displacement,
#         }

#         # print("output==================================", output)
#         return output
    

@compile_mode("script")
class ScaleShiftMACE(MACE):
    def __init__(
        self,
        atomic_inter_scale: float,
        atomic_inter_shift: float,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.scale_shift = ScaleShiftBlock(
            scale=atomic_inter_scale, shift=atomic_inter_shift
        )

    def create_gso(self, edge_index, num_nodes):
        # 인접 행렬을 초기화 (num_nodes x num_nodes 크기의 영행렬)
        GSO = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)

        # edge_index에서 각 엣지를 따라 인접 행렬을 업데이트
        for i in range(edge_index.size(1)):
            node_from = edge_index[0, i]  # 시작 노드
            node_to = edge_index[1, i]    # 끝 노드
            GSO[node_from, node_to] = 1   # 인접 행렬에서 연결된 부분을 1로 설정

        # 추가적으로, 만약 GSO를 정규화하고 싶다면 정규화 수행
        # degree = GSO.sum(dim=1)  # 각 노드의 차수 계산
        # GSO = GSO / degree[:, None]  # 노드 차수로 나누어 정규화
        return GSO

    def add_gso(self, GSO):
        # 모든 상호작용 블록에 GSO 추가 (EVGF에도 추가)
        for interaction in self.interactions:
            if hasattr(interaction, 'addGSO'):
                interaction.addGSO(GSO)

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        training: bool = False,
        compute_force: bool = True,
        compute_virials: bool = False,
        compute_stress: bool = False,
        compute_displacement: bool = False,
    ) -> Dict[str, Optional[torch.Tensor]]:
        # Setup
        data["positions"].requires_grad_(True)
        num_graphs = data["ptr"].numel() - 1
        displacement = torch.zeros(
            (num_graphs, 3, 3),
            dtype=data["positions"].dtype,
            device=data["positions"].device,
        )

        # GSO를 생성
        edge_index = data["edge_index"]  # 예시로 edge_index 사용
        num_nodes = data["positions"].size(0)
        GSO = self.create_gso(edge_index, num_nodes)  # GSO 생성 함수 호출
        # print("num_nodes", num_nodes)

        # GSO를 각 상호작용 블록과 EVGF 필터에 전달
        self.add_gso(GSO)

        if compute_virials or compute_stress or compute_displacement:
            (
                data["positions"],
                data["shifts"],
                displacement,
            ) = get_symmetric_displacement(
                positions=data["positions"],
                unit_shifts=data["unit_shifts"],
                cell=data["cell"],
                edge_index=data["edge_index"],
                num_graphs=num_graphs,
                batch=data["batch"],
            )

        # Atomic energies
        node_e0 = self.atomic_energies_fn(data["node_attrs"])
        e0 = scatter_sum(
            src=node_e0, index=data["batch"], dim=-1, dim_size=num_graphs
        )  # [n_graphs,]

        # Embeddings
        node_feats = self.node_embedding(data["node_attrs"])
        vectors, lengths = get_edge_vectors_and_lengths(
            positions=data["positions"],
            edge_index=data["edge_index"],
            shifts=data["shifts"],
        )
        edge_attrs = self.spherical_harmonics(vectors)
        edge_feats = self.radial_embedding(lengths)

        # Interactions
        # ScaleShiftMACE에서는 마지막에 e0, node_eo 더해서 최종 에너지 출력
        interaction_energy_list = [] # 그래프 에너지 리스트
        node_es_list = [node_e0] # 노드 에너지 리스트

        # 여기서 오류 시작
        for interaction, product, readout in zip(
            self.interactions, self.products, self.readouts
        ):
            node_feats, sc = interaction(
                node_attrs=data["node_attrs"],
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=data["edge_index"],
            )

            # 필터링된 노드 피처를 Product Block에 전달
            node_feats = product(
                node_feats=node_feats,
                sc=sc,
                node_attrs=data["node_attrs"],
            )

            node_energies = readout(node_feats).squeeze(-1)  # [n_nodes, ]
            energy = scatter_sum(
                src=node_energies, index=data["batch"], dim=-1, dim_size=num_graphs
            )  # [n_graphs,]
            interaction_energy_list.append(energy)
            node_es_list.append(node_energies)

        # Sum over interaction energy 
        # interaction_energy : 총 상호작용 에너지 (각 상호작용 블럭 에너지의 합)
        interaction_energy = torch.sum(torch.stack(interaction_energy_list, dim=-1), dim=-1)

        # total_energy : 최종 그래프 에너지 (e0 앞에서 더함)
        total_energy = e0 + interaction_energy

        # node_energy : 최종 노드 에너지
        node_energy_contributions = torch.stack(node_es_list, dim=-1)
        node_energy = torch.sum(node_energy_contributions, dim=-1)  # [n_nodes, ]

        # Scale-shift된 노드 에너지를 계산 (한번에 scale-shift 수행)
        node_inter_es = self.scale_shift(node_energy)

        # Outputs
        forces, virials, stress = get_outputs(
            energy=total_energy,
            positions=data["positions"],
            displacement=displacement,
            cell=data["cell"],
            training=training,
            compute_force=compute_force,
            compute_virials=compute_virials,
            compute_stress=compute_stress,
        )

        output = {
            "energy": total_energy,
            "node_energy": node_inter_es,
            "interaction_energy": interaction_energy,
            "forces": forces,
            "virials": virials,
            "stress": stress,
            "displacement": displacement,
        }

        return output

    

    
