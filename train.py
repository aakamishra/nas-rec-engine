import time
import copy
import random
from params import *
import torch_geometric
import utils.model_utils as m_util
from model_src.demo_functions import *
from utils.misc_utils import RunningStatMeter
from model_src.model_helpers import BookKeeper
from model_src.comp_graph.tf_comp_graph import OP2I
from model_src.comp_graph.tf_comp_graph_models import make_cg_regressor, make_embedding_model
from model_src.predictor.gpi_family_data_manager import FamilyDataManager
from model_src.comp_graph.tf_comp_graph_dataloaders import CGRegressDataLoader
from utils.model_utils import set_random_seed, device, add_weight_decay, get_activ_by_name
from model_src.predictor.model_perf_predictor import train_predictor, run_predictor_demo, train_embedding_model


"""
Naive accuracy predictor training routine
For building a generalizable predictor interface
"""


def prepare_local_params(parser, ext_args=None):
    parser.add_argument("-model_name", required=False, type=str,
                        default="CL_dropout_encoder_decoder_model")
    parser.add_argument("-family_train", required=False, type=str,
                        default="nb101+nb201c10+ofa_resnet"
                        )
    parser.add_argument('-family_test', required=False, type=str,
                        default="nb201c10#50"
                                "+nb301#50"
                                "+ofa_resnet#50")
    parser.add_argument("-dev_ratio", required=False, type=float,
                        default=0.1)
    parser.add_argument("-test_ratio", required=False, type=float,
                        default=0.1)
    parser.add_argument("-epochs", required=False, type=int,
                        default=100)
    parser.add_argument("-fine_tune_epochs", required=False, type=int,
                        default=0)
    parser.add_argument("-batch_size", required=False, type=int,
                        default=32)
    parser.add_argument("-initial_lr", required=False, type=float,
                        default=0.001)
    parser.add_argument("-in_channels", help="", type=int,
                        default=128, required=False)
    parser.add_argument("-hidden_size", help="", type=int,
                        default=128, required=False)
    parser.add_argument("-out_channels", help="", type=int,
                        default=128, required=False)
    parser.add_argument("-num_layers", help="", type=int,
                        default=6, required=False)
    parser.add_argument("-dropout_prob", help="", type=float,
                        default=0.3, required=False)
    parser.add_argument("-aggr_method", required=False, type=str,
                        default="mean")
    parser.add_argument("-gnn_activ", required=False, type=str,
                        default="relu")
    parser.add_argument("-reg_activ", required=False, type=str,
                        default=None)
    parser.add_argument('-gnn_type', required=False,
                        default="GINConv")
    parser.add_argument("-normalize_HW_per_family", required=False, action="store_true",
                        default=False)
    parser.add_argument('-e_chk', type=str, default=None, required=False)
    return parser.parse_args(ext_args)


def get_family_train_size_dict(args):
    if args is None:
        return {}
    rv = {}
    for arg in args:
        if "#" in arg:
            fam, size = arg.split("#")
        else:
            fam = arg
            size = 0
        rv[fam] = int(float(size))
    return rv


def main(params):
    params.model_name = "gpi_acc_predictor_{}_seed{}".format(params.model_name, params.seed)
    book_keeper = BookKeeper(log_file_name=params.model_name + ".txt",
                             model_name=params.model_name,
                             saved_models_dir=params.saved_models_dir,
                             init_eval_perf=float("inf"), eval_perf_comp_func=lambda old, new: new < old,
                             saved_model_file=params.saved_model_file,
                             logs_dir=params.logs_dir)

    if type(params.family_test) is str:
        families_train = list(v for v in set(params.family_train.split("+")) if len(v) > 0)
        families_train.sort()
        families_test = params.family_test.split("+")
    else:
        families_train = params.family_train
        families_test = params.family_test

    book_keeper.log("Params: {}".format(params), verbose=False)
    set_random_seed(params.seed, log_f=book_keeper.log)
    book_keeper.log("Train Families: {}".format(families_train))
    book_keeper.log("Test Families: {}".format(families_test))

    families_test = get_family_train_size_dict(families_test)

    data_manager = FamilyDataManager(families_train, log_f=book_keeper.log)
    family2sets = \
        data_manager.get_regress_train_dev_test_sets(params.dev_ratio, params.test_ratio,
                                                     normalize_HW_per_family=params.normalize_HW_per_family,
                                                     normalize_target=False, group_by_family=True)

    train_data, dev_data, test_data = [], [], []
    for f, (fam_train, fam_dev, fam_test) in family2sets.items():
        train_data.extend(fam_train)
        dev_data.extend(fam_dev)
        test_data.extend(fam_test)

    random.shuffle(train_data)
    random.shuffle(dev_data)
    random.shuffle(test_data)
    book_keeper.log("Train size: {}".format(len(train_data)))
    book_keeper.log("Dev size: {}".format(len(dev_data)))
    book_keeper.log("Test size: {}".format(len(test_data)))

    b_node_size_meter = RunningStatMeter()
    for g, _ in train_data + dev_data + test_data:
        b_node_size_meter.update(len(g))
    book_keeper.log("Max num nodes: {}".format(b_node_size_meter.max))
    book_keeper.log("Min num nodes: {}".format(b_node_size_meter.min))
    book_keeper.log("Avg num nodes: {}".format(b_node_size_meter.avg))

    train_loader = CGRegressDataLoader(params.batch_size, train_data)
    dev_loader = CGRegressDataLoader(params.batch_size, dev_data)
    test_loader = CGRegressDataLoader(params.batch_size, test_data)

    book_keeper.log(
        "{} overlap(s) between train/dev loaders".format(train_loader.get_overlapping_data_count(dev_loader)))
    book_keeper.log(
        "{} overlap(s) between train/test loaders".format(train_loader.get_overlapping_data_count(test_loader)))
    book_keeper.log(
        "{} overlap(s) between dev/test loaders".format(dev_loader.get_overlapping_data_count(test_loader)))

    book_keeper.log("Initializing {}".format(params.model_name))

    if "GINConv" in params.gnn_type:
        def gnn_constructor(in_channels, out_channels):
            nn = torch.nn.Sequential(torch.nn.Linear(in_channels, in_channels),
                                     torch.nn.Linear(in_channels, out_channels),
                                     )
            return torch_geometric.nn.GINConv(nn=nn)
    else:
        def gnn_constructor(in_channels, out_channels):
            return eval("torch_geometric.nn.%s(%d, %d)"
                        % (params.gnn_type, in_channels, out_channels))

    print("checkpoint here")
    # change this lines here
    model = make_embedding_model(n_unique_labels=len(OP2I().build_from_file()), out_embed_size=params.in_channels,
                              shape_embed_size=8, kernel_embed_size=8, n_unique_kernels=8, n_shape_vals=6,
                              hidden_size=params.hidden_size, out_channels=params.out_channels,
                              gnn_constructor=gnn_constructor,
                              gnn_activ=get_activ_by_name(params.gnn_activ), n_gnn_layers=params.num_layers,
                              dropout_prob=params.dropout_prob, aggr_method=params.aggr_method,
                              regressor_activ=get_activ_by_name(params.reg_activ)).to(device())
    
    print("made model")

    if params.e_chk is not None:
        book_keeper.load_model_checkpoint(model, allow_silent_fail=False, skip_eval_perfs=True,
                                          checkpoint_file=params.e_chk)
        book_keeper.log("Loaded checkpoint: {}".format(params.e_chk))

    perf_criterion = torch.nn.MSELoss()
    model_params = add_weight_decay(model, weight_decay=0.)
    optimizer = torch.optim.Adam(model_params, lr=params.initial_lr)

    book_keeper.log(model)
    book_keeper.log("Model name: {}".format(params.model_name))
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    book_keeper.log("Number of trainable parameters: {}".format(n_params))


    def _batch_fwd_func(_model, _batch):
        # Define how a batch is handled by the model
        regular_node_inds = _batch[DK_BATCH_CG_REGULAR_IDX]
        regular_node_shapes = _batch[DK_BATCH_CG_REGULAR_SHAPES]
        weighted_node_inds = _batch[DK_BATCH_CG_WEIGHTED_IDX]
        weighted_node_shapes = _batch[DK_BATCH_CG_WEIGHTED_SHAPES]
        weighted_node_kernels = _batch[DK_BATCH_CG_WEIGHTED_KERNELS]
        weighted_node_bias = _batch[DK_BATCH_CG_WEIGHTED_BIAS]
        edge_tsr_list = _batch[DK_BATCH_EDGE_TSR_LIST]
        batch_last_node_idx_list = _batch[DK_BATCH_LAST_NODE_IDX_LIST]
        return _model(regular_node_inds, regular_node_shapes,
                      weighted_node_inds, weighted_node_shapes, weighted_node_kernels, weighted_node_bias,
                      edge_tsr_list, batch_last_node_idx_list)

    book_keeper.log("Training for {} epochs".format(params.epochs))
    start = time.time()
    try:
        train_embedding_model(_batch_fwd_func, model, train_loader, perf_criterion, optimizer, book_keeper,
                        num_epochs=params.epochs, max_gradient_norm=params.max_gradient_norm, dev_loader=dev_loader)
    except KeyboardInterrupt:
        book_keeper.log("Training interrupted")
        

if __name__ == "__main__":
    _parser = prepare_global_params()
    _args = prepare_local_params(_parser)
    m_util.DEVICE_STR_OVERRIDE = _args.device_str
    main(_args)
    print("done")
