from __future__ import annotations
from collections import defaultdict
from types import SimpleNamespace
import pytest
import torch
from comfy_research.engine.trainer.observable_viz import observable_viz_metric_updates
from comfy_research.engine.trainer.spectral_norm import author_figure1_power_estimate, singular_norm_power_estimate
from comfy_research.nodes.definitions.observables.layer_spectral_norm import record
from comfy_research.schemas.graph import Edge, Node, NodeKind
class Model(torch.nn.Module):
 def __init__(self): super().__init__(); self.square=torch.nn.Linear(3,3,bias=False); self.rect=torch.nn.Linear(3,2,bias=False)
def node(**data): return Node(id="norm",type=NodeKind.observable_layer_spectral_norm,data={"estimator":"singular_value","powerIterations":8,**data})
def rec(model): return SimpleNamespace(model=model,observable_metric_histories=defaultdict(list))
def test_estimators_are_rng_neutral_and_cover_square_and_rectangular_weights():
 matrix=torch.arange(1.,13.).reshape(3,4); torch.manual_seed(123); before=torch.get_rng_state().clone()
 assert singular_norm_power_estimate(matrix,iterations=12).item()==pytest.approx(torch.linalg.matrix_norm(matrix,ord=2).item(),rel=1e-3)
 assert torch.equal(before,torch.get_rng_state()); assert author_figure1_power_estimate(torch.eye(3),iterations=4).isfinite()
 with pytest.raises(ValueError): author_figure1_power_estimate(matrix)
def test_recorder_routes_payload_and_degrades_author_rectangular_and_no_linear():
 r=rec(Model()); record(r,node()); assert len(r.observable_metric_histories["norm::layer::1"])==1; assert len(r.observable_metric_histories["norm::layer::2"])==1
 record(r,node(estimator="author_figure1")); assert torch.isnan(torch.tensor(r.observable_metric_histories["norm::layer::2"][-1]))
 trainer=Node(id="trainer",type=NodeKind.trainer,data={}); viz=Node(id="viz",type=NodeKind.observable_viz,data={"pairedObservableId":"norm","vizVariant":"layer_spectral_norm"})
 edges=[Edge(id="o",source="norm",target="trainer",sourceHandle="observable",targetHandle="observables"),Edge(id="v",source="trainer",target="viz",sourceHandle="observable_results",targetHandle="tensor")]
 updates=observable_viz_metric_updates(edges,{n.id:n for n in (trainer,node(),viz)},"trainer",r.observable_metric_histories,{})
 assert updates[0]["series_labels"]==["Linear layer 1","Linear layer 2"] and len(updates[0]["value_histories"])==2
 empty=rec(torch.nn.ReLU()); record(empty,node()); assert torch.isnan(torch.tensor(empty.observable_metric_histories["norm"][0]))
