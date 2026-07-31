from comfy_research.api.node_catalog import _HIDDEN_CATALOG_NODE_TYPES, list_node_categories
from comfy_research.generated.node_manifest import load_node_manifest


def test_node_catalog_contains_all_visible_manifest_nodes() -> None:
    manifest_visible = {
        str(entry["type"])
        for entry in load_node_manifest()
        if entry.get("category") != "internal"
    } - _HIDDEN_CATALOG_NODE_TYPES
    catalog = list_node_categories()
    catalog_ids = {
        str(child["id"])
        for category in catalog["categories"]
        for child in category["children"]
    }
    assert manifest_visible <= catalog_ids
    # Intentionally hidden from the sidebar palette (reachable via dedicated flows only).
    assert not (_HIDDEN_CATALOG_NODE_TYPES & catalog_ids)
    assert "graph_assist_failure_overlay" not in catalog_ids
