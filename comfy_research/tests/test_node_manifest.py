from comfy_research.generated.node_manifest import load_node_manifest


def test_generated_node_manifest_has_registry_entries() -> None:
    manifest = load_node_manifest()
    assert len(manifest) >= 180
    by_type = {entry.get("type"): entry for entry in manifest}
    assert len(by_type) == len(manifest)
    assert "mlp_model" in by_type
    assert "observable_weight_l2" in by_type
    assert "graph_assist_failure_overlay" in by_type
    assert all(isinstance(entry.get("label"), str) and entry["label"] for entry in manifest)
    assert all(isinstance(entry.get("category"), str) and entry["category"] for entry in manifest)

