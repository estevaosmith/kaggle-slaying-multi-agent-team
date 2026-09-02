from kaggle_slaying.graph import build_bootstrap_graph


def test_bootstrap_graph_nodes() -> None:
    graph = build_bootstrap_graph().get_graph()

    assert "check_environment" in graph.nodes
    assert "download_data" in graph.nodes
