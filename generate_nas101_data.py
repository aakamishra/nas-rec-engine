import tensorflow as tf
from nasbench import api
import nasbench_keras
from nasbench_keras import ModelSpec, build_keras_model, build_module
import nasbench1
import nasbench1_spec
import tqdm
from model_src.comp_graph.tf_comp_graph_dataloaders import CGRegressDataLoader
from tqdm import tqdm
from model_src.comp_graph.tf_comp_graph_models import make_cg_regressor, make_embedding_model, make_embedding_regressor_model
import torch_geometric
from utils.model_utils import set_random_seed, device, add_weight_decay, get_activ_by_name
from model_src.comp_graph.tf_comp_graph import ComputeGraph, OP2I
from tensorflow.keras import backend as K
import pickle
import logging
logging.getLogger().setLevel(logging.ERROR)


dataset_path = '/home/ec2-user/nasbench_full.tfrecord'
# Use nasbench_full.tfrecord for full dataset (run download command above).
nasbench = api.NASBench(dataset_path)
print("loaded")

data = []

i = 0
for hash in tqdm(nasbench.hash_iterator()):
    info = nasbench.get_metrics_from_hash(hash)
    try:
        matrix = info[0]['module_adjacency']
        ops = info[0]['module_operations']
        acc = info[1][108][-1]['final_test_accuracy']
        params = info[0]['trainable_parameters']
        model_spec = api.ModelSpec(matrix=matrix, ops=ops)
        K.clear_session()
        # Create module
        with tf.compat.v1.keras.backend.get_session() as sess:
            inputs = tf.keras.layers.Input((3,224,224), 1)
            outputs = build_module(spec=model_spec, inputs=inputs, channels=128, is_training=True)
            module = tf.keras.Model(inputs=inputs, outputs=outputs)
            
            graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(
                sess,
                sess.graph.as_graph_def(),
                [node.op.name for node in module.outputs])
            op2i = OP2I().build_from_file()
            new_cg = ComputeGraph(name="test", H=224, W=224, C_in=3)

            new_cg.build_from_graph_def(graph_def, op2i, 5, False)
            data.append((new_cg,(acc, params)))
            print(i)
            i += 1

    except:
        print("failure")
        
with open('nas101graphs1.pkl', 'wb') as f:
    pickle.dump(data, f)