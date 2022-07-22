import logging
import os
import torch
import dgl

def set_logger(cfg):
    """
    setting logging
    e.g. logging path, basic Config

    Parameter
    ----------
    cfg

    """
    if not os.path.exists(os.path.join(cfg.model.save_dir,cfg.data.name)):
        os.makedirs(os.path.join(os.getcwd(),cfg.model.save_dir, cfg.data.name))

    log_file = os.path.join(cfg.model.save_dir, cfg.data.name, 'log.txt')

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.DEBUG,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )

    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

def read_id(path):
    """
    loading graph from dataset
    
    Parameters
    ----------
    path: string

    Returns
    -------
    tmp: dict (key=entity, value=type)
    """
    tmp = dict()
    with open(path, encoding='utf-8') as r:
        for line in r:
            e, t = line.strip().split('\t')
            tmp[e] = int(t)
    return tmp

def read_entity(path):
    """
    loading graph from dataset
    
    Parameters
    ----------
    path: string

    Returns
    -------
    tmp: dict (key=id, value=(type,entity,relation))
    """
    tmp = dict()
    with open(path, encoding='utf-8') as r:
        for line in r:
            e, t = line.strip().split('\t')
            tmp[int(t)] = e
    return tmp

def read_name(path):
    tmp = dict()
    with open(path, encoding='utf-8') as r:
        for line in r:
            mid, name = (line.strip().split('\t') + [None])[:2]
            tmp[mid] = name
    return tmp

def load_labels(paths, e2id, t2id):
    labels = torch.zeros(len(e2id), len(t2id))
    for path in paths:
        with open(path, encoding='utf-8') as r:
            for line in r:
                e, t = line.strip().split('\t')
                e_id, t_id = e2id[e], t2id[t]
                labels[e_id, t_id] = 1
    return labels


def load_id(path, e2id):
    ret = set()
    with open(path, encoding='utf-8') as r:
        for line in r:
            e, t = line.strip().split('\t')
            ret.add(e2id[e])
    return list(ret)

def load_triple(path, e2id, r2id):
    head = []
    e_type = []
    tail = []
    with open(path, encoding='utf-8') as r:
        for line in r:
            h, r, t = line.strip().split('\t')
            h, r, t = e2id[h], r2id[r], e2id[t]
            head.append(h)
            e_type.append(r)
            tail.append(t)
    return head, e_type, tail

def load_ET(path, e2id, t2id, r2id):
    head = []
    e_type = []
    tail = []
    with open(path, encoding='utf-8') as r:
        for line in r:
            e, t = line.strip().split('\t')
            e, t = e2id[e], t2id[t] + len(e2id)
            head.append(e)
            tail.append(t)
            e_type.append(r2id['type'])
    return head, e_type, tail


def load_graph(data_dir, e2id, r2id, t2id, loadET=True, loadKG=True, data="ET_train.txt"):
    """
    loading graph from dataset
    
    Parameters
    ----------
    data_dir: string
    e2id: dict
    r2id: dict
    loadET, loadKG: boolean
    train_data: string e.g. ET_train.txt or ET_1_1_train.txt or ET_1_n_train.txt
    valid_data: string e.g. ET_valid.txt or ET_1_1_valid.txt or ET_1_n_valid.txt or ET_unobserved_valid.txt
    test_data: string e.g. ET_test.txt or ET_1_1_test.txt or ET_1_n_test.txt or ET_unobserved_test.txt

    Returns
    -------
    g: Graph
    label: 2 dim tensor for representing relation between entity and entity type
    id_list: 1 dim tensor for representing entity id having entity type
    """
    # load graph with input features, labels and edge type
    label = load_labels([os.path.join(data_dir, data)], e2id, t2id)
    id_list = load_id(os.path.join(data_dir, data), e2id)
    if loadKG:
        head1, e_type1, tail1 = load_triple(os.path.join(data_dir, 'train.txt'), e2id, r2id)
    else:
        head1, e_type1, tail1 = [], [], []
    if loadET:
        head2, e_type2, tail2 = load_ET(os.path.join(data_dir, 'ET_train.txt'), e2id, t2id, r2id)
    else:
        head2, e_type2, tail2 = [], [], []

    head = torch.LongTensor(head1 + head2)
    tail = torch.LongTensor(tail1 + tail2)
    g = dgl.graph((head, tail))

    e_type1 = torch.LongTensor(e_type1)
    e_type2 = torch.LongTensor(e_type2)
    e_type = torch.cat([e_type1, e_type2], dim=0)
    g.edata['etype'] = e_type
    if loadET:
        g.ndata['id'] = torch.arange(len(e2id) + len(t2id))
    else:
        g.ndata['id'] = torch.arange(len(e2id))

    return g, label, id_list