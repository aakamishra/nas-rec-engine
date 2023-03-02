import time
import torch
import collections
from tqdm import tqdm
from constants import *
from utils.model_utils import device
from utils.eval_utils import get_regression_metrics, get_regression_rank_metrics
from torch_geometric.utils import get_laplacian
from torch_geometric.utils import to_dense_adj


def train_embedding_model(batch_fwd_func, model, train_loader, criterion, optimizer, book_keeper, num_epochs,
                    max_gradient_norm=5.0, eval_start_epoch=1, eval_every_epoch=1,
                    rv_metric_name="mean_absolute_percent_error", completed_epochs=0,
                    dev_loader=None, checkpoint=True):
    model = model.to(device())
    criterion = criterion.to(device())
    for epoch in range(num_epochs):
        report_epoch = epoch + completed_epochs + 1
        model.train()
        train_score = run_embedding_epoch(batch_fwd_func, model, train_loader, criterion, optimizer, book_keeper,
                                          rv_metric_name=rv_metric_name, max_grad_norm=max_gradient_norm,
                                          curr_epoch=report_epoch, num_epochs=num_epochs)
        book_keeper.log("Train score at epoch {}: {}".format(report_epoch, train_score))
        if checkpoint:
            book_keeper.checkpoint_model("_latest.pt", report_epoch, model, optimizer)

        if dev_loader is not None:
            with torch.no_grad():
                model.eval()
                if report_epoch >= eval_start_epoch and report_epoch % eval_every_epoch == 0:
                    dev_score = run_embedding_epoch(batch_fwd_func, model, dev_loader, criterion, None, book_keeper,
                                                    rv_metric_name=rv_metric_name, desc="Dev",
                                                    max_grad_norm=max_gradient_norm,
                                                    curr_epoch=report_epoch)
                    book_keeper.log("Dev score at epoch {}: {}".format(report_epoch, dev_score))
                    if checkpoint:
                        book_keeper.checkpoint_model("_best.pt", report_epoch, model, optimizer, eval_perf=dev_score)
                        book_keeper.report_curr_best()
        book_keeper.log("")
        
def original_cosine_similarity(a, b):
    dot = torch.sum(a * b, dim=-1)
    norm_a = torch.norm(a, dim=-1)
    norm_b = torch.norm(b, dim=-1)
    return dot / (norm_a * norm_b)

def cosine_similarity(a, b, t=0.05):
    dot = torch.sum(a * b, dim=-1)
    return dot / t

# adj = edge_list_to_edge_matrix(batch[DK_BATCH_EDGE_TSR_LIST][0], batch[DK_BATCH_EDGE_TSR_LIST][0].shape[1])
# np.sort(np.linalg.eig(normalize(laplacian(adj)))[0])[:-21]
#from sklearn.preprocessing import normalize
# from scipy.sparse.csgraph import laplacian

class SimCLRLoss(torch.nn.Module):
    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature
        self.q = 21
        
    def graph_to_eigen(self, e, q=21):
        laplace_matrix = get_laplacian(e)
        adj = to_dense_adj(laplace_matrix[0], edge_attr=laplace_matrix[1])
        norm_adj = torch.nn.functional.normalize(adj)
        eigenvalues = torch.linalg.eigvals(norm_adj)
        values, indices = torch.sort(eigenvalues.to(torch.float64))
        return values.view(-1)[:q]
    
    def forward(self, x, edges):
        """
        x1: anchor, shape = (batch_size, feature_size)
        x2: positive or negative, shape = (batch_size, feature_size)
        y: binary label indicating whether x1 and x2 are positive or negative, shape = (batch_size,)
        """
        l = len(edges)
        n = x.shape[0]
        g2e = torch.zeros(2*l, self.q).cuda()
        similarity_sum = 0.0
        for i in range(2*l):
            g2e[i] = self.graph_to_eigen(edges[i%l], q=self.q)
        
        # distance calculations
        distances = torch.linalg.norm(g2e[:, None, :] - g2e[None, :, :], axis=-1)
        graph_distance_values = distances.flatten()[1:].view(n-1, n+1)[:,:-1].flatten()
        graph_distance_values = torch.nn.functional.softmax(graph_distance_values)
        
        # similarity focus
        x_flat = torch.flatten(x, start_dim=1)
        similarity = torch.matmul(x_flat, x_flat.T) / self.temperature
        flat_similarity = similarity.flatten()[1:].view(n-1, n+1)[:,:-1].flatten()
        log_sum_exp = torch.logsumexp(flat_similarity, dim=-1)
        normalized_similarity_sum = flat_similarity / log_sum_exp
        alpha_similarity_product = graph_distance_values * normalized_similarity_sum
        return -torch.sum(alpha_similarity_product)
        
        

def loss_criterion(a, b):
    cos_sim = original_cosine_similarity(a, b)
    log_softmax = torch.nn.LogSoftmax(dim=-1)(cos_sim)
    
    # differential
    return torch.mean(log_softmax)
   
def run_embedding_epoch(batch_fwd_func, model, loader, criterion, optimizer, book_keeper,
                        desc="Train", curr_epoch=0, max_grad_norm=5.0, report_metrics=True,
                        rv_metric_name="mean_absolute_percent_error", num_epochs=40):
    """
    Compatible with a predictor/loader that batches same-sized graphs
    """
    start = time.time()
    total_loss, n_instances = 0., 0
    metrics_dict = collections.defaultdict(float)
    preds, targets = [], []
    temperature = 0.3*(1.05 - (curr_epoch/num_epochs))
    if desc != "Train":
        temperature = 0.05
    for batch in tqdm(loader, desc=desc, ascii=True):
        embed, node_embed, decoder_node_embed = batch_fwd_func(model, batch)
        cos_loss = (1 - torch.nn.functional.cosine_similarity(node_embed, decoder_node_embed)).mean()
        loss = SimCLRLoss(temperature)(embed, batch[DK_BATCH_EDGE_TSR_LIST]) + cos_loss
        total_loss += loss.item() * batch[DK_BATCH_SIZE]
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
        n_instances += batch[DK_BATCH_SIZE]
    elapsed = time.time() - start
    rv_loss = total_loss / n_instances
    msg = desc + " epoch: {}, loss: {}, elapsed time: {}, temperature: {}".format(curr_epoch, rv_loss, elapsed, temperature)
    book_keeper.log(msg)
    return rv_loss


#########################################

def train_predictor(batch_fwd_func, model, train_loader, criterion, optimizer, book_keeper, num_epochs,
                    max_gradient_norm=5.0, eval_start_epoch=1, eval_every_epoch=1,
                    rv_metric_name="mean_absolute_percent_error", completed_epochs=0,
                    dev_loader=None, checkpoint=True):
    model = model.to(device())
    criterion = criterion.to(device())
    for epoch in range(num_epochs):
        report_epoch = epoch + completed_epochs + 1
        model.train()
        train_score = run_predictor_epoch(batch_fwd_func, model, train_loader, criterion, optimizer, book_keeper,
                                          rv_metric_name=rv_metric_name, max_grad_norm=max_gradient_norm,
                                          curr_epoch=report_epoch)
        book_keeper.log("Train score at epoch {}: {}".format(report_epoch, train_score))
        if checkpoint:
            book_keeper.checkpoint_model("_latest.pt", report_epoch, model, optimizer)

        if dev_loader is not None:
            with torch.no_grad():
                model.eval()
                if report_epoch >= eval_start_epoch and report_epoch % eval_every_epoch == 0:
                    dev_score = run_predictor_epoch(batch_fwd_func, model, dev_loader, criterion, None, book_keeper,
                                                    rv_metric_name=rv_metric_name, desc="Dev",
                                                    max_grad_norm=max_gradient_norm,
                                                    curr_epoch=report_epoch)
                    book_keeper.log("Dev score at epoch {}: {}".format(report_epoch, dev_score))
                    if checkpoint:
                        book_keeper.checkpoint_model("_best.pt", report_epoch, model, optimizer, eval_perf=dev_score)
                        book_keeper.report_curr_best()
        book_keeper.log("")


def run_predictor_epoch(batch_fwd_func, model, loader, criterion, optimizer, book_keeper,
                        desc="Train", curr_epoch=0, max_grad_norm=5.0, report_metrics=True,
                        rv_metric_name="mean_absolute_percent_error"):
    """
    Compatible with a predictor/loader that batches same-sized graphs
    """
    start = time.time()
    total_loss, n_instances = 0., 0
    metrics_dict = collections.defaultdict(float)
    preds, targets = [], []
    for batch in tqdm(loader, desc=desc, ascii=True):
        batch_vals = batch_fwd_func(model, batch)
        truth = batch[DK_BATCH_TARGET_TSR].to(device())
        pred = batch_vals.squeeze(1)
        loss = criterion(pred, truth)
        total_loss += loss.item() * batch[DK_BATCH_SIZE]
        preds.extend(pred.detach().tolist())
        targets.extend(truth.detach().tolist())
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
        n_instances += batch[DK_BATCH_SIZE]
    elapsed = time.time() - start
    rv_loss = total_loss / n_instances
    msg = desc + " epoch: {}, loss: {}, elapsed time: {}".format(curr_epoch, rv_loss, elapsed)
    book_keeper.log(msg)
    if report_metrics:
        metrics_dict = get_regression_metrics(preds, targets)
        rank_metrics = get_regression_rank_metrics(preds, targets,
                                                   top_overlap_k_list=(5, 10, 25, 50),
                                                   verbose=True)
        metrics_dict["spearman_rho"] = rank_metrics["spearman_rho"]
        metrics_dict["spearman_p"] = rank_metrics["spearman_p"]
        metrics_dict["top-5 overlaps"] = rank_metrics["top-5 overlaps"]
        metrics_dict["top-10 overlaps"] = rank_metrics["top-10 overlaps"]
        metrics_dict["top-25 overlaps"] = rank_metrics["top-25 overlaps"]
        metrics_dict["top-50 overlaps"] = rank_metrics["top-50 overlaps"]
        book_keeper.log("{} performance: {}".format(desc, str(metrics_dict)))
    return rv_loss if not report_metrics else metrics_dict[rv_metric_name]


def run_predictor_demo(batch_fwd_func, model, loader, log_f=print,
                       n_batches=1, normalize_constant=None,
                       input_str_key=None):
    n_visited = 0
    input_str_list = []
    preds, targets = [], []
    for batch in loader:
        if n_visited == n_batches:
            break
        batch_vals = batch_fwd_func(model, batch)
        truth = batch[DK_BATCH_TARGET_TSR].to(device())
        pred = batch_vals.squeeze(1)
        pred_list = pred.detach().tolist()
        target_list = truth.detach().tolist()
        preds.extend(pred_list)
        targets.extend(target_list)
        n_visited += 1
        if input_str_key is not None:
            batch_input_str = batch[input_str_key]
            for bi in range(len(pred_list)):
                input_str_list.append(batch_input_str[bi])
    for i, pred in enumerate(preds):
        input_str = input_str_list[i] if len(input_str_list) == len(preds) else None
        if input_str is not None:
            log_f("Input: {}".format(input_str))
        log_f("Pred raw: {}".format(pred))
        log_f("Truth raw: {}".format(targets[i]))
        if normalize_constant is not None:
            log_f("Normalize constant: {}".format(normalize_constant))
            log_f("Pred un-normalized: {:.3f}".format(pred * normalize_constant))
            log_f("Truth un-normalized: {:.3f}".format(targets[i] * normalize_constant))
        log_f("")
