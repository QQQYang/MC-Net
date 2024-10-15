import string

import networkx as nx
import torch
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGInNode, DAGOpNode, DAGOutNode
from qiskit.transpiler.passes import RemoveBarriers
from torch_geometric.utils.convert import from_networkx

GATE_DICT = {"rz": 0, "x": 1, "sx": 2, "cx": 3, "u1": 4, "u2": 5, "u3": 6}
NUM_ERROR_DATA = 7
NUM_NODE_TYPE = 2 + len(GATE_DICT)


def circ_to_dag_with_data(circ, mydict, n_qubit=10):
    circ = circ.copy()
    circ = RemoveBarriers()(circ)

    dag = circuit_to_dag(circ)
    dag = dag.to_networkx()
    dag_list = list(dag.nodes())
    used_qubit_idx_list = {}
    used_qubit_idx = 0
    for node in dag_list:
        node_type, qubit_idxs, noise_info = data_generator(node, mydict)
        if node_type == "in":
            succnodes = dag.succ[node]
            for succnode in succnodes:
                succnode_type, _, _ = data_generator(succnode, mydict)
                if succnode_type == "out":
                    dag.remove_node(node)
                    dag.remove_node(succnode)
    dag_list = list(dag.nodes())
    for node_idx, node in enumerate(dag_list):
        node_type, qubit_idxs, noise_info = data_generator(node, mydict)
        for qubit_idx in qubit_idxs:
            if not qubit_idx in used_qubit_idx_list:
                used_qubit_idx_list[qubit_idx] = used_qubit_idx
                used_qubit_idx += 1
        data = torch.zeros(NUM_NODE_TYPE + n_qubit + NUM_ERROR_DATA + 1)
        if node_type == "in":
            data[0] = 1
            data[NUM_NODE_TYPE + used_qubit_idx_list[qubit_idxs[0]]] = 1
            data[NUM_NODE_TYPE + n_qubit] = noise_info[0]["T1"]
            data[NUM_NODE_TYPE + n_qubit + 1] = noise_info[0]["T2"]
        elif node_type == "out":
            data[1] = 1
            data[NUM_NODE_TYPE + used_qubit_idx_list[qubit_idxs[0]]] = 1
            data[NUM_NODE_TYPE + n_qubit] = noise_info[0]["T1"]
            data[NUM_NODE_TYPE + n_qubit + 1] = noise_info[0]["T2"]
            data[NUM_NODE_TYPE + n_qubit + 5] = noise_info[0]["prob_meas0_prep1"]
            data[NUM_NODE_TYPE + n_qubit + 6] = noise_info[0]["prob_meas1_prep0"]
        else:
            data[2 + GATE_DICT[node_type]] = 1
            for i in range(len(qubit_idxs)):
                data[NUM_NODE_TYPE + used_qubit_idx_list[qubit_idxs[i]]] = 1
                data[NUM_NODE_TYPE + n_qubit + 2 * i] = noise_info[i]["T1"]
                data[NUM_NODE_TYPE + n_qubit + 2 * i + 1] = noise_info[i]["T2"]
            data[NUM_NODE_TYPE + n_qubit + 4] = noise_info[-1]
        data[-1] = node_idx
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
        break
    x = torch.zeros((len(G.nodes()), d_feat_x))
    for idx, node in enumerate(G.nodes()):
        x[idx] = dag.nodes[node]["x"]
    G = from_networkx(G)
    G.x = x
    return G


def data_generator(node, mydict):
    if isinstance(node, DAGInNode):
        qubit_idx = int(node.wire._index)
        return "in", [qubit_idx], [mydict["qubit"][qubit_idx]]
    elif isinstance(node, DAGOutNode):
        qubit_idx = int(node.wire._index)
        return "out", [qubit_idx], [mydict["qubit"][qubit_idx]]
    elif isinstance(node, DAGOpNode):
        name = node.name
        qargs = node.qargs
        qubit_list = []
        for qubit in qargs:
            qubit_list.append(qubit._index)
        mylist = [mydict["qubit"][qubit_idx] for qubit_idx in qubit_list]
        mylist.append(mydict["gate"][tuple(qubit_list)][name])
        return (name, qubit_list, mylist)
    else:
        raise NotImplementedError("Unknown node type")
    
def raw_pyg_converter(qasm, device_info):
    circ = QuantumCircuit()
    circ = circ.from_qasm_str(qasm)
    circ.remove_final_measurements()
    dag = circ_to_dag_with_data(circ, device_info, circ.num_qubits)
    return dag
