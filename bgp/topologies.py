'''
抓取zoo网络拓扑数据集，保存在本地（gml格式）
'''

import xml.etree.ElementTree as ET
import networkx as nx
import numpy as np
import os

def read_topology(filename):
    G = nx.DiGraph()
    tree = ET.parse(filename)
    
    def add_edge(src, dst): 
        G.add_edge(src, dst)
        G.add_edge(dst, src)

    graphs = [e for e in tree.getroot() if e.tag == "{http://graphml.graphdrawing.org/xmlns}graph"]
    assert len(graphs) == 1, "file contains more than one graph"
    graph = list(graphs)[0]

    node_elements = [e for e in graph if e.tag == "{http://graphml.graphdrawing.org/xmlns}node"]
    edge_elements = [e for e in graph if e.tag == "{http://graphml.graphdrawing.org/xmlns}edge"]

    remapping = {}
    has_edges = {}
    
    for n in node_elements:
        assert "id" in n.attrib.keys(), "node is missing id attribute"
        original_nid = int(n.attrib["id"])
        mapped_id = len(remapping)
        remapping[original_nid] = mapped_id
        has_edges[original_nid] = False
        G.add_node(mapped_id)

    for e in edge_elements:
        assert "source" in e.attrib.keys(), "edge element is missing source attribute"
        assert "target" in e.attrib.keys(), "edge element is missing target attribute"
        src, tgt = int(e.attrib["source"]), int(e.attrib["target"])
        add_edge(remapping[src],remapping[tgt])
        has_edges[src] = True
        has_edges[tgt] = True
    
    for nid,has_at_least_one_edge in has_edges.items():
        if not has_at_least_one_edge:
            G.remove_node(nid)
    
    # final remapping
    G_clean = nx.DiGraph()
    final_remapping = {}
    for i,n in enumerate(G.nodes()):
        final_remapping[n] = i
        G_clean.add_node(i)
    for src,dst in G.edges():
        G_clean.add_edge(final_remapping[src], final_remapping[dst])
    
    return G_clean
    
def generate_graph_with_topology(filename, seed, NUM_NETWORKS=None, P_STATIC_ROUTE_USE=0.5, P_STATIC_ROUTE=0.3):
    if NUM_NETWORKS is None: NUM_NETWORKS = 4
    MAX_WEIGHT = 32

    s = np.random.RandomState(seed=seed)
    graph = read_topology(filename)
    NUM_NODES = len(graph.nodes())

    # set node types
    nx.set_node_attributes(graph, "router", name="type")

    # initialise link weights
    for src, tgt in graph.edges(): 
        weight = s.randint(1, MAX_WEIGHT)
        graph[src][tgt]["weight"] = weight
        graph[tgt][src]["weight"] = weight

    # add network nodes
    network_nodes = set()
    for n in range(NUM_NETWORKS):
        node_id = len(graph.nodes())
        gateway_node = s.randint(0, NUM_NODES)
        network_nodes.add(node_id)
        
        graph.add_edge(node_id, gateway_node, weight=1)
        graph.add_edge(gateway_node, node_id, weight=1)
        graph.nodes[node_id]["type"] = "network"

    use_static_routes = (s.random() < P_STATIC_ROUTE_USE)
    # initialise static routes
    static_routes_per_node = [set() for n in range(NUM_NODES)]
    for src, tgt in graph.edges():
        if src in network_nodes or tgt in network_nodes: continue
        graph[src][tgt]["static"] = set()
        for net in network_nodes:
            # static src -> tgt to net
            if s.random() - P_STATIC_ROUTE < 0 and use_static_routes:
                if not net in static_routes_per_node[src]:
                    static_routes_per_node[src].add(net)
                    graph[src][tgt]["static"].add(net)

    return graph


all_topology_files = [f"{os.path.dirname(__file__)}/topologies/{f}" for f in os.listdir(os.path.join(os.path.dirname(__file__), "topologies")) if f.endswith("graphml")]

if __name__ == "__main__":
    from bs4 import BeautifulSoup
    import requests
    import re
    from tqdm import tqdm
    import sys

    if len(sys.argv) < 2 or sys.argv[1] != "--download":
        sys.exit(0)

    html_page = requests.get("http://www.topology-zoo.org/dataset.html").text

    files = []
    soup = BeautifulSoup(html_page)
    for link in soup.findAll('a'):
        if link.get("href").startswith("files/") and link.get("href").endswith(".graphml"):
            files.append(link.get("href"))

    for file in tqdm(files):
        r = requests.get(f"http://www.topology-zoo.org/{file}", allow_redirects=True)
        filename = file.split("/")[1]
        with open(f"topologies/{filename}", 'wb') as f:
            f.write(r.content)