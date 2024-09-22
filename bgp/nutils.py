from typing import List
import torch
import numpy as np
class Aggregator:
    def __init__(self):
        self.metrics = {}
        self.metrics_samples = {}

    def add_sample(self, tag, value, skip_nan=False):
        if skip_nan and torch.isnan(value): return
        if tag in self.metrics.keys():
            self.metrics[tag] = self.metrics[tag] + value
            self.metrics_samples[tag] = self.metrics_samples[tag] + 1
        else:
            self.metrics[tag] = value
            self.metrics_samples[tag] = 1

    def add_to_writer(self, writer):
        for m in self.metrics.keys():
            writer.add_scalar(m, self.metrics[m] / (self.metrics_samples[m]))
    
    def get_metric(self, tag):
        return self.metrics[tag] / (self.metrics_samples[tag])

class StepWriter:
    def __init__(self, writer, step):
        self.writer = writer
        self.step = step

    def add_scalar(self, tag, scalar):
        self.writer.add_scalar(tag, scalar, global_step=self.step)

class ModelCheckpointer:
    def __init__(self, num_checkpoints=10):
        self.checkpoint_ids = [i for i in range(num_checkpoints)]
        self.checkpoint_idx = 0

        self.last_best_metric = None
    
    def next_checkpoint_id(self):
        id = self.checkpoint_ids[self.checkpoint_idx]    
        self.checkpoint_idx = (self.checkpoint_idx + 1) % len(self.checkpoint_ids)
        return id

    def save(self, model, metric):
        if self.last_best_metric is None: 
            self._save(model, metric)
        elif self.last_best_metric < metric: 
            self._save(model, metric)

    def _save(self, model, metric):
        from datetime import datetime
        
        self.last_best_metric = metric
        if not os.path.exists("models"): os.mkdir("models")
        id = str(self.next_checkpoint_id())
        existing_checkpoint = None
        for f in os.listdir("models/"):
            if f.startswith(f"model-save{id}"): os.remove("models/" + f)
        metric = metric.item() if torch.is_tensor(metric) else metric
        timestamp = datetime.now().strftime("%d-%m-%Y-%H_%M_%S")
        filename = f"models/model-save{id}-{metric}-{timestamp}.pt"
        torch.save(model.state_dict(), filename)

def mask_like(x, p=0.5):
    return ((torch.rand_like(x, dtype=torch.float) - (1-p)) >= 0)


class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def zero_grad(self):
        self.optimizer.zero_grad()
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))
    

def get_std_opt(model, factor=1):
    return NoamOpt(model.hidden_dim, factor, 4000, 
        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

def bidirectional(edge_index):
    return torch.cat([
        torch.cat([edge_index[0], edge_index[1]]).unsqueeze(0),
        torch.cat([edge_index[1], edge_index[0]]).unsqueeze(0)
    ])
def reflexive(edge_index, num_nodes):
    return torch.cat([
        torch.cat([edge_index[0], torch.arange(num_nodes, dtype=torch.long, device=edge_index.device)]).unsqueeze(0),
        torch.cat([edge_index[1], torch.arange(num_nodes, dtype=torch.long, device=edge_index.device)]).unsqueeze(0)
    ])

def reflexive_bidirectional_edge_type(edge_type, num_nodes):
    return torch.cat([
        torch.cat([edge_type, edge_type, torch.zeros([num_nodes], device=edge_type.device, dtype=torch.long)])
    ])

def split(dataset, p=0.75):
    print("Validation dataset size", int(len(dataset)*(1-p)))
    return dataset[:int(len(dataset)*p)], dataset[int(len(dataset)*p):]

def mask_node_features(x, mask):
    masked = torch.ones_like(x) * -1
    masked[mask.logical_not()] = x[mask.logical_not()]
    return masked

def categorical(values: List[torch.Tensor], probs: List[float]):
    assert len(values) == len(probs)
    num_nodes = int(torch.tensor([v.size(0) for v in values]).max().item())
    batch_size = int(torch.tensor([v.size(1) for v in values]).max().item())
    hidden_dim = int(torch.tensor([v.size(2) for v in values]).max().item())
    res = torch.zeros([num_nodes, batch_size, hidden_dim], device=values[0].device)
    z = torch.rand([num_nodes, batch_size], device=values[0].device).unsqueeze(-1)
    lower = 0.0
    upper = 0.0
    for i, value in enumerate(values): 
        upper += probs[i]
        res += (z >= lower) * (z < upper) * value
        lower = upper
    return res

def prop(e,prop):
    if prop not in e.keys(): return None
    return e[prop]

def choose_random(arr, s = None):
    s = np.random if s is None else s
    idx = s.randint(0, len(arr))
    return arr[idx]

def convert_old_gat_conv_state_dict(state_dict):
    """
    Converts a state_dict which contains GATConv parmeters from an 
    older version of PyG to the current version (parameter renaming).
    """
    res = dict()
    for name, param in state_dict.items():
        if "layer_edge_type_" in name:
            # convert name
            if "lin_l" in name:
                name = name.replace("lin_l", "lin_src")
            elif "att_l" in name:
                name = name.replace("att_l", "att_src")
            elif "lin_r" in name:
                name = name.replace("lin_r", "lin_dst")
            elif "att_r" in name:
                name = name.replace("att_r", "att_dst")
        res[name] = param
    return res
        