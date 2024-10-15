import string

import networkx as nx
import torch
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGInNode, DAGOpNode, DAGOutNode
from qiskit.transpiler.passes import RemoveBarriers
from torch_geometric.utils.convert import from_networkx

GATE_DICT = {"rx": 0, "ry": 1, "rz": 2, "cx": 3}
NUM_NODE_TYPE = 2 + len(GATE_DICT)


def get_global_features(circ):
    data = torch.zeros((1, 6))
    data[0][0] = circ.depth()
    data[0][1] = circ.width()
    for key in GATE_DICT:
        if key in circ.count_ops():
            data[0][2 + GATE_DICT[key]] = circ.count_ops()[key]

    return data


def circ_to_dag_with_data(circ, n_qubit=10, noise=0.0):
    circ = circ.copy()
    circ = RemoveBarriers()(circ)

    dag = circuit_to_dag(circ)
    dag = dag.to_networkx()
    dag_list = list(dag.nodes())
    used_qubit_idx_list = {}
    used_qubit_idx = 0
    for node in dag_list:
        node_type, qubit_idxs = data_generator(node)
        if node_type == "in":
            succnodes = dag.succ[node]
            for succnode in succnodes:
                succnode_type, _ = data_generator(succnode)
                if succnode_type == "out":
                    dag.remove_node(node)
                    dag.remove_node(succnode)
    dag_list = list(dag.nodes())
    for node_idx, node in enumerate(dag_list):
        node_type, qubit_idxs = data_generator(node)
        for qubit_idx in qubit_idxs:
            if not qubit_idx in used_qubit_idx_list:
                used_qubit_idx_list[qubit_idx] = used_qubit_idx
                used_qubit_idx += 1
        data = torch.zeros(NUM_NODE_TYPE + n_qubit + 1 + 1)
        if node_type == "in":
            data[0] = 1
            data[NUM_NODE_TYPE + used_qubit_idx_list[qubit_idxs[0]]] = 1
        elif node_type == "out":
            data[1] = 1
            data[NUM_NODE_TYPE + used_qubit_idx_list[qubit_idxs[0]]] = 1
        else:
            data[2 + GATE_DICT[node_type]] = 1
            for i in range(len(qubit_idxs)):
                data[NUM_NODE_TYPE + used_qubit_idx_list[qubit_idxs[i]]] = 1
        data[-1] = node_idx
        if node_type == 'cx':
            data[-2] = noise
        else:
            data[-2] = 0.0
        if node in dag.nodes():
            dag.nodes[node]["x"] = data
    mapping = dict(zip(dag, string.ascii_lowercase))
    dag = nx.relabel_nodes(dag, mapping)
    return networkx_torch_convert(dag)

def networkx_torch_convert(dag):
    myedge = []
    for item in dag.edges:
        myedge.append((item[0], item[1]))
    G = nx.DiGraph()
    G.add_nodes_from(dag._node)
    G.add_edges_from(myedge)
    d_feat_x = 0
    for idx, node in enumerate(G.nodes()):
        d_feat_x = len(dag.nodes[node]["x"])
    x = torch.zeros((len(G.nodes()), d_feat_x))
    for idx, node in enumerate(G.nodes()):
        x[idx] = dag.nodes[node]["x"]
    G = from_networkx(G)
    G.x = x
    return G


def data_generator(node):
    if isinstance(node, DAGInNode):
        qubit_idx = int(node.wire._index)
        return "in", [qubit_idx]
    elif isinstance(node, DAGOutNode):
        qubit_idx = int(node.wire._index)
        return "out", [qubit_idx]
    elif isinstance(node, DAGOpNode):
        name = node.name
        qargs = node.qargs
        qubit_list = []
        for qubit in qargs:
            qubit_list.append(qubit._index)
        return (name, qubit_list)
    else:
        raise NotImplementedError("Unknown node type")
    
def raw_pyg_converter(dataset, noise=0.0):
    circ = QuantumCircuit()
    circ = circ.from_qasm_str(dataset)
    circ.remove_final_measurements()
    dag = circ_to_dag_with_data(circ, circ.num_qubits, noise)
    return dag
