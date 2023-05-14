from nasbench import api
from model_src.comp_graph.tf_comp_graph import ComputeGraph, RegularNode, WeightedNode, OP2I
from torchview.computation_node import TensorNode, ModuleNode, FunctionNode
from tqdm import tqdm
from torchview import draw_graph
from torchview.computation_graph import ComputationGraph
from nasbench import api
import nasbench1
import nasbench1_spec
import pickle
import numpy as np
import random
from ptflops import get_model_complexity_info



def run():
    i = 0
    j = 0
    data = []
    flops_data = []
    
    dataset_path = '/home/ec2-user/nasbench_full.tfrecord'
    # Use nasbench_full.tfrecord for full dataset (run download command above).
    nasbench = api.NASBench(dataset_path)
    print("loaded")
    
    op2i = OP2I().build_from_file("/home/ec2-user/nas-rec-engine/data/alternative_primitives.txt")
    hash_list = [hash for hash in tqdm(nasbench.hash_iterator())]
    all_indices = range(len(hash_list))  # Example range of indices from 1 to 5000

    # Randomly select 2100 indices
    random_indices = random.sample(all_indices, 2500)

    for idx in tqdm(random_indices):
        hash = hash_list[idx]
        info = nasbench.get_metrics_from_hash(hash)
        matrix = info[0]['module_adjacency']
        ops = info[0]['module_operations']
        acc = info[1][108][-1]['final_test_accuracy']
        params = info[0]['trainable_parameters']
        
        # create pytorch network
        spec = nasbench1_spec._ToModelSpec(matrix, ops)
        net = nasbench1.Network(spec, stem_out=128, num_stacks=3, num_mods=3, num_classes=10)
        macs, _ = get_model_complexity_info(net, (3, 32, 32), as_strings=False)
        flops_data.append(macs)
        # create dot graph
        model_graph = draw_graph(
            net, input_size=(1,3,32,32),
            graph_name=f'MLP{i}'
        )
        
        i += 1
        
        new_cg = process_graph(model_graph, op2i, i)
        #print("finished ", i, new_cg)
        data.append((new_cg, (acc, macs, hash)))

    
    with open(f"nas101graphs_tiny.pkl", 'wb') as f:
        pickle.dump(data, f)
    print("flops mean: ", np.mean(flops_data), " std: ", np.std(flops_data))
    
def process_graph(model_graph: ComputationGraph, op2i, i):
    regular_nodes, weighted_nodes = [], []
    seen_nodes = {}
    MAX_H, MAX_W, MAX_C, MAX_K = 0, 0, 0, 0


    for edge in model_graph.edge_list:
        input_node, output_node = edge
        for node in [input_node, output_node]:
            if type(node) == TensorNode:
                #print(node.name, node.tensor_shape, node.main_node, node.tensor_id, node.node_id, model_graph.id_dict[node.node_id])
                node_id = node.node_id
                idx_id = model_graph.id_dict[node.node_id]
                if idx_id in seen_nodes:
                    continue
                str_id = f"{idx_id}|_{node.main_node.name}"
                #print("found input? ", node.main_node.name.split("-")[0])
                op_type_idx = op2i[node.main_node.name.split("-")[0]]
                # [Hin, Hout, Win, Wout, Cin, Cout] for every node
                rg_node = RegularNode(str_id=str_id, label=node.main_node.name, op_type_idx=op_type_idx)
                if len(node.tensor_shape) == 4:
                    rg_node.resolution = [node.tensor_shape[2], node.tensor_shape[2], node.tensor_shape[3], node.tensor_shape[3], node.tensor_shape[1], node.tensor_shape[1]]
                    MAX_H = max(node.tensor_shape[2], node.tensor_shape[2], MAX_H)
                    MAX_W = max(node.tensor_shape[3], node.tensor_shape[3], MAX_W)
                    MAX_C = max(node.tensor_shape[1], node.tensor_shape[1], MAX_C)          
                else:
                    rg_node.resolution = (0, 0, 0, 0, node.tensor_shape[1], node.tensor_shape[1])
                    MAX_C = max(node.tensor_shape[1], node.tensor_shape[1], MAX_C)
                seen_nodes[idx_id] = (rg_node, len(regular_nodes))
                #print(rg_node)
                regular_nodes.append(rg_node)
                
            elif type(node) == ModuleNode:
                #print(node.name, node.node_id, node.input_shape, node.output_shape, node.compute_unit_id, node.is_activation, model_graph.id_dict[node.node_id], node.output_nodes)
                if node.name in ["Sequential", "Conv2d", "ConvBnRelu", "Linear"]:
                    node_id = node.node_id
                    idx_id = model_graph.id_dict[node.node_id]
                    if idx_id in seen_nodes:
                        continue
                    str_id = f"{idx_id}|_{node.name}"
                    op_type_idx = op2i[node.name]
                    if "Conv" not in node.name:
                        shape = [node.input_shape[0][1], node.output_shape[0][1], 0, 0]
                    else:
                        shape = [node.input_shape[0][1], node.output_shape[0][1], 1, 1]
                        MAX_K = max(1, MAX_K)
                        if node.input_shape[0][1] != node.output_shape[0][1]:
                            val = kernel_size = node.output_shape[0][1] / (node.input_shape[0][2]) - 1
                            shape = [node.input_shape[0][1], node.output_shape[0][1], val, val]
                            MAX_K = max(MAX_K, val)
                            
                    # [Hin, Hout, Win, Wout, Cin, Cout] for every node
                    w_node = WeightedNode(str_id=str_id, label=node.name, op_type_idx=op_type_idx, shape=shape)
                    if len(node.input_shape[0]) == 4:
                        w_node.resolution = (node.input_shape[0][2], node.output_shape[0][2], node.input_shape[0][3], node.output_shape[0][3], node.input_shape[0][1], node.output_shape[0][1])
                        MAX_H = max(node.input_shape[0][2], node.output_shape[0][2], MAX_H)
                        MAX_W = max(node.input_shape[0][3], node.output_shape[0][3], MAX_W)
                        MAX_C = max(node.input_shape[0][1], node.output_shape[0][1], MAX_C)
                    else:
                        w_node.resolution = (0, 0, 0, 0, node.input_shape[0][1], node.output_shape[0][1])
                        MAX_C = max(node.input_shape[0][1], node.output_shape[0][1], MAX_C)
                    w_node.strides = 1
                    seen_nodes[idx_id] = (w_node, len(weighted_nodes))
                    #print(w_node)
                    weighted_nodes.append(w_node)
                else:
                    node_id = node.node_id
                    idx_id = model_graph.id_dict[node.node_id]
                    if idx_id in seen_nodes:
                        continue
                    str_id = f"{idx_id}|_{node.name}"
                    op_type_idx = op2i[node.name]
                    # [Hin, Hout, Win, Wout, Cin, Cout] for every node
                    rg_node = RegularNode(str_id=str_id, label=node.name, op_type_idx=op_type_idx)
                    if len(node.input_shape[0]) == 4:
                        rg_node.resolution = (node.input_shape[0][2], node.output_shape[0][2], node.input_shape[0][3], node.output_shape[0][3], node.input_shape[0][1], node.output_shape[0][1])
                        MAX_H = max(node.input_shape[0][2], node.output_shape[0][2], MAX_H)
                        MAX_W = max(node.input_shape[0][3], node.output_shape[0][3], MAX_W)
                        MAX_C = max(node.input_shape[0][1], node.output_shape[0][1], MAX_C)
                    else:
                        rg_node.resolution = (0, 0, 0, 0, node.input_shape[0][1], node.output_shape[0][1])
                        MAX_C = max(node.input_shape[0][1], node.output_shape[0][1], MAX_C)
                    seen_nodes[idx_id] = (rg_node, len(regular_nodes))
                    #print(rg_node)
                    regular_nodes.append(rg_node)
                
            elif type(node) == FunctionNode:
                #print(node.name, node.node_id, node.input_shape, node.output_shape, node.compute_unit_id, model_graph.id_dict[node.node_id], node.output_nodes)
                node_id = node.node_id
                idx_id = model_graph.id_dict[node.node_id]
                if idx_id in seen_nodes:
                    continue
                str_id = f"{idx_id}|_{node.name}"
                op_type_idx = op2i[node.name]
                # [Hin, Hout, Win, Wout, Cin, Cout] for every node
                #print(node.input_shape, node.output_shape, node.name)
                rg_node = RegularNode(str_id=str_id, label=node.name, op_type_idx=op_type_idx)
                if len(node.input_shape) == 4 and len(node.output_shape) == 4:
                    rg_node.resolution = (node.input_shape[0][2], node.output_shape[0][2], node.input_shape[0][3], node.output_shape[0][3], node.input_shape[0][1], node.output_shape[0][1])
                    MAX_H = max(node.input_shape[0][2], node.output_shape[0][2], MAX_H)
                    MAX_W = max(node.input_shape[0][3], node.output_shape[0][3], MAX_W)
                    MAX_C = max(node.input_shape[0][1], node.output_shape[0][1], MAX_C)
                
                elif len(node.input_shape) == 4 and len(node.output_shape) == 2:
                    rg_node.resolution = (node.input_shape[0][2], 0, node.input_shape[0][3], 0, node.input_shape[0][1], node.output_shape[0][1])
                    MAX_H = max(node.input_shape[0][2], 0, MAX_H)
                    MAX_W = max(node.input_shape[0][3], 0, MAX_W)
                    MAX_C = max(node.input_shape[0][1], node.output_shape[0][1], MAX_C)
                    
                else:
                    rg_node.resolution = (0, 0, 0, 0, node.input_shape[0][1], node.output_shape[0][1])
                    MAX_C = max(node.input_shape[0][1], node.output_shape[0][1], MAX_C)
                    
                seen_nodes[idx_id] = (rg_node, len(regular_nodes))
                #print(rg_node)
                regular_nodes.append(rg_node)
    
    cg_node_ls = weighted_nodes + regular_nodes
    new_edge_list = []
    for edge in model_graph.edge_list:
        in_node, out_node = edge
        cg_in_node, cg_in_pos = seen_nodes[model_graph.id_dict[in_node.node_id]]
        if isinstance(cg_in_node, RegularNode):
            cg_in_pos += len(weighted_nodes) 
        cg_out_node, cg_out_pos = seen_nodes[model_graph.id_dict[out_node.node_id]]
        if isinstance(cg_out_node, RegularNode):
            cg_out_pos += len(weighted_nodes) 
        new_edge_list.append((cg_in_pos, cg_out_pos))
        
    new_cg = ComputeGraph(C_in=3, H=32, W=32, name=f"test{i}", 
                      max_hidden_size=MAX_C, max_derived_H=MAX_H, 
                      max_derived_W=MAX_W, max_kernel_size=MAX_K)

    new_cg.edge_pairs = new_edge_list
    new_cg.regular_nodes = regular_nodes
    new_cg.weighted_nodes = weighted_nodes
    new_cg.n_regular_nodes = len(regular_nodes)
    new_cg.n_weighted_nodes = len(weighted_nodes)
    
    return new_cg

if __name__ == "__main__":
    run()