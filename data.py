from logging import raiseExceptions
import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader, RandomSampler, SequentialSampler

from utils import *
import copy

class EntityTypingJointGTDataset(Dataset):
  def __init__(self, cfg, data_path, tokenizer, mode):
    """
    initialize Entity Typing Dataset

    Parameter
    cfg: Dict
    data_path: String
    tokenizer: transformers.tokenizer
    mode: string (train, valid, test)
    """
    self.cfg = cfg
    self.data_path = data_path
    self.tokenizer = tokenizer
    assert mode in ["train", "valid", "test"]
    self.mode = mode
    # preprocess graph
    self.e2id = read_id(os.path.join(data_path, 'entities.tsv'))
    self.id2e = read_entity(os.path.join(data_path, 'entities.tsv'))
    self.r2id = read_id(os.path.join(data_path, 'relations.tsv'))
    self.id2r = read_entity(os.path.join(data_path, 'relations.tsv'))
    length_r2id = len(self.r2id)
    self.r2id['type'] = length_r2id
    self.id2r[length_r2id] = 'type'
    self.t2id = read_id(os.path.join(data_path, 'types.tsv'))
    self.id2t = read_entity(os.path.join(data_path, 'types.tsv'))
    if cfg.data.name=="FB15kET" and cfg.data.change_mid_to_name: self.mid2name = read_name(os.path.join(data_path, 'mid2name.tsv'))
    self.num_entity = len(self.e2id)
    self.num_rels = len(self.r2id)
    self.num_types = len(self.t2id)
    self.num_nodes = self.num_entity + self.num_types
    self.g, self.train_label, self.valid_label, self.test_label, self.all_true, self.train_id, self.valid_id, self.test_id = load_graph(data_path, self.e2id, self.r2id, self.t2id,
                                                                       cfg.preprocess.load_ET, cfg.preprocess.load_KG,
                                                                       cfg.model.test.test_dataset)

    self.head_ids = self.tokenizer.encode(' [head]', add_special_tokens=False)
    self.rel_ids = self.tokenizer.encode(' [relation]', add_special_tokens=False)
    self.tail_ids = self.tokenizer.encode(' [tail]', add_special_tokens=False)
    self.graph_ids = self.tokenizer.encode(' [graph]', add_special_tokens=False)
    self.text_ids =  self.tokenizer.encode(' [text]', add_special_tokens=False)

    if self.cfg.model.pretrained_model == "Bart":
      self.mask_token = self.tokenizer.mask_token
      self.mask_token_id = self.tokenizer.mask_token_id
    else:
        self.mask_token = self.tokenizer.additional_special_tokens[0]
        self.mask_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.additional_special_tokens[0])
    if self.cfg.model.pretrained_model == "Bart":
      if self.cfg.model.append_another_bos:
          self.add_bos_id = [self.tokenizer.bos_token_id] * 2
      else:
          self.add_bos_id = [self.tokenizer.bos_token_id]
    else:
        self.add_bos_id = []

  def get_change_per_sample(self, triples, entities_set, relations_set):
    """
    Parameter
    triples: dict (key=head, value=[relation, tail])

    Return
    entity_tokenized_ids_dict: dict key=entity_name, value=[tokenized_entity_id, numbering]
    relation_tokenized_ids_dict: dict key=relation_name, value=[tokenized_relation_id, numbering]
    """
    entity_tokenized_ids_dict = dict()
    relation_tokenized_ids_dict = dict()

    for ent_numbering, entity in enumerate(list(entities_set)):
      entity_toks = self.tokenizer.encode(" {}".format(entity), add_special_tokens=False)
      entity_tokenized_ids_dict[entity] = [entity_toks, ent_numbering]

    for rel_numbering, rel in enumerate(list(relations_set)):
      relation_toks = self.tokenizer.encode(" {}".format(rel), add_special_tokens=False)
      relation_tokenized_ids_dict[rel] = [relation_toks, rel_numbering]
    
    return entity_tokenized_ids_dict, relation_tokenized_ids_dict

  def linearize(self, triples, entity_tokenized_ids_dict, relation_tokenized_ids_dict, head_ids, rel_ids, tail_ids, cnt_edge, adj_matrix):
    """
    
    Parameter
    triples: list
    entity_tokenized_ids_dict: dict
    relation_tokenized_ids_dict: dict
    head_ids: list (e.g. [646, 3628, 742])
    rel_ids: list (e.g. [646, 47114, 742])
    tail_ids: list (e.g. [646, 17624, 742])
    cnt_edge: int
    adj_matrix: list

    Return
    input_ids: list (e.g. [646, 3628, 742, 957, 5369, 5399, 646, 47114, 742, 9553, 7067, 8564, 646, 17624, ...])
    input_text: list (e.g. ' [head] James Craig Watson [relation] discoverer [tail] 101 Helena')
    nodes: list ids for representing node ids (e.g. [-1, -1, -1, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1, ...])
    edges: list ids for representing edge ids (e.g. [-1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, -1, -1, ...])
    cnt_edge: int (e.g. 1)
    adj_matrix: 2d list (e.g. [[-1, -1, -1, 0, -1, -1, -1, -1, -1, ...], [-1, -1, -1, -1, -1, -1, -1, -1, -1, ...], [-1, -1, -1, -1, -1, -1, -1, -1, -1, ...], [-1, -1, -1, -1, -1, -1, -1, -1, -1, ...], [-1, -1, -1, -1, -1, -1, -1, -1, -1, ...], [-1, -1, -1, -1, -1, -1, -1, -1, -1, ...], [-1, -1, -1, -1, -1, -1, -1, -1, -1, ...], [-1, -1, -1, -1, -1, -1, -1, -1, -1, ...], [-1, -1, -1, -1, -1, -1, -1, -1, -1, ...], [-1, -1, -1, -1, -1, -1, -1, -1, -1, ...], [-1, -1, -1, -1, -1, -1, -1, -1, -1, ...], [-1, -1, -1, -1, -1, -1, -1, -1, -1, ...], [-1, -1, -1, -1, -1, -1, -1, -1, -1, ...], [-1, -1, -1, -1, -1, -1, -1, -1, -1, ...], ...])
    """
    nodes, edges = [], []
    input_ids = []

    # head, value=[relation1, tail2,..]
    for head, value in triples.items():
      # Head
      input_ids += copy.deepcopy(head_ids)
      input_text = ' [head]'
      nodes.extend([-1] * len(head_ids))
      edges.extend([-1] * len(head_ids))
      input_ids += entity_tokenized_ids_dict[head][0]
      input_text += ' {}'.format(head)
      nodes.extend([entity_tokenized_ids_dict[head][1]] * len(entity_tokenized_ids_dict[head][0]))
      edges.extend([-1] * len(entity_tokenized_ids_dict[head][0]))

      assert len(value)%2==0
      for i in range(0, len(value), 2):
        rel, tail = value[i:i+2]

        # Relation
        input_ids += copy.deepcopy(rel_ids)
        input_text += ' [relation]'
        nodes.extend([-1] * len(rel_ids))
        edges.extend([-1] * len(rel_ids))
        input_ids += relation_tokenized_ids_dict[rel][0]
        input_text += ' {}'.format(rel)
        nodes.extend([-1] * len(relation_tokenized_ids_dict[rel][0]))
        edges.extend([relation_tokenized_ids_dict[rel][1]] * len(relation_tokenized_ids_dict[rel][0]))

        # Tail
        input_ids += copy.deepcopy(tail_ids)
        input_text += ' [tail]'
        nodes.extend([-1] * len(tail_ids))
        edges.extend([-1] * len(tail_ids))
        input_ids += entity_tokenized_ids_dict[tail][0]
        input_text += ' {}'.format(tail)
        nodes.extend([entity_tokenized_ids_dict[tail][1]] * len(entity_tokenized_ids_dict[tail][0]))
        edges.extend([-1] * len(entity_tokenized_ids_dict[tail][0]))

        if entity_tokenized_ids_dict[head][1] < len(adj_matrix) and entity_tokenized_ids_dict[tail][1] < len(adj_matrix):
                    adj_matrix[entity_tokenized_ids_dict[head][1]][entity_tokenized_ids_dict[tail][1]] = cnt_edge

        cnt_edge += 1
    assert len(input_ids) == len(nodes) == len(edges)
    return input_ids, input_text, nodes, edges, cnt_edge, adj_matrix

  def truncate_pair(self, a, add_bos_id, graph_ids, text_ids, node_ids, edge_ids):
    """
    Parameter

    a: list (e.g. [646, 3628, 742, 957, 5369, 5399, 646, 47114, 742, 9553, 7067, 8564, 646, 17624, ...])
    add_bos_id: list (e.g. [0, 0])
    graph_ids: list (e.g. [646, 44143, 742])
    text_ids: list (e.g. [646, 29015, 742])
    node_ids: list (e.g. [-1, -1, -1, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1, ...])
    edge_ids: list (e.g. [-1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, -1, -1, ...])

    Return
    input_ids: list (e.g. [0, 0, 646, 44143, 742, 646, 3628, 742, 957, 5369, 5399, 646, 47114, 742, ...])
    input_attn_mask: list (e.g. [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...])
    input_node_ids: list (e.g. [-1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, -1, -1, -1, ...])
    input_edge_ids: list (e.g. [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, ...])

    """
     # add_bos_id + graph_ids + a + text_ids + b + eos_token_id
    length_a_b = self.cfg.model.max_input_length - len(add_bos_id) - len(graph_ids) - len(text_ids) - 1
    if len(a) > length_a_b:
        a = a[:length_a_b]
        node_ids = node_ids[:length_a_b]
        edge_ids = edge_ids[:length_a_b]
    input_ids = add_bos_id + graph_ids + a + text_ids + [self.tokenizer.eos_token_id]
    input_node_ids = [-1] * (len(add_bos_id) + len(graph_ids)) + node_ids + [-1] * (len(text_ids) + 1)
    input_edge_ids = [-1] * (len(add_bos_id) + len(graph_ids)) + edge_ids + [-1] * (len(text_ids) + 1)
    attn_mask = [1] * len(input_ids) + [0] * (self.cfg.model.max_input_length - len(input_ids))
    input_ids += [self.tokenizer.pad_token_id] * (self.cfg.model.max_input_length - len(input_ids))
    input_node_ids += [-1] * (self.cfg.model.max_input_length - len(input_node_ids))
    input_edge_ids += [-1] * (self.cfg.model.max_input_length - len(input_edge_ids))
    assert len(input_ids) == len(attn_mask) == self.cfg.model.max_input_length == len(input_node_ids) == len(
        input_edge_ids)
    return input_ids, attn_mask, input_node_ids, input_edge_ids

  def prep_data(self, target_ids, input_ids, add_bos_id, graph_ids, text_ids, node_ids, edge_ids):
    """
    preprocess data for model's input

    Parameter

    target_ids: list (e.g. [646, 3628, 742, 957, 5369, 5399, 646, 47114, 742, 9553, 7067, 8564, 646, 17624, ...])
    input_ids: list (e.g. [6560, 25239, 34, 41, 6256, 139, 7527, 354, 9, 3550, 698, 6617, 151, 4, ...])
    add_bos_id: list ids for [self.tokenizer.bos_token_id](*2) (e.g. [0, 0])
    graph_ids: list ids for encoding " [GRAPH] "(e.g. [646, 44143, 742])
    text_ids: list ids for encoding " [TEXT]" (e.g. [646, 29015, 742])
    node_ids: list (e.g. [-1, -1, -1, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1, ...])
    edge_ids: list (e.g. [-1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, -1, -1, ...])


    Return
    input_ids: list (e.g. [0, 0, 646, 44143, 742, 646, 3628, 742, 957, 5369, 5399, 646, 47114, 742, ...])
    iput_attn_mask: list (e.g. [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...])
    target_ids: list (e.g. [0, 0, 6560, 25239, 34, 41, 6256, 139, 7527, 354, 9, 3550, 698, 6617, ...])
    target_attn_mask: list (e.g. [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...]) 
    input_node_ids: list (e.g. [-1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, -1, -1, -1, ...])
    input_edge_ids: list (e.g. [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, ...])
    """

    # add bos and eos
    target_ids = copy.deepcopy(target_ids)
    if len(target_ids) > self.cfg.model.max_output_length - len(add_bos_id) - 1:
            target_ids = target_ids[:(self.cfg.model.max_output_length - len(add_bos_id) - 1)]
    target_ids = add_bos_id + target_ids + [self.tokenizer.eos_token_id]
    target_attn_mask = [1] * len(target_ids) + [0] * (self.cfg.model.max_output_length - len(target_ids))
    target_ids += [self.tokenizer.pad_token_id] * (self.cfg.model.max_output_length - len(target_ids))
    assert len(target_ids) == self.cfg.model.max_output_length == len(target_attn_mask)

    input_ids, input_attn_mask, input_node_ids, input_edge_ids = self.truncate_pair(input_ids, add_bos_id, graph_ids, text_ids, node_ids, edge_ids)


    return input_ids, input_attn_mask, target_ids, target_attn_mask, input_node_ids, input_edge_ids

  def __len__(self):
    if self.mode=="train":
      return len(self.train_id)
    elif self.mode=="valid":
      return len(self.valid_id)
    elif self.mode=="test":
      return len(self.test_id)
    else:
      raise ValueError("mode is train or valid or test")
    
  def __getitem__(self, idx):
    """
    Parameter
    idx: int

    Return
    Batch: list
    [0]: input_ids_ar
    [1]: attn_mask_ar
    [2]: target_ids
    [3]: target_attn_mask
    [4]: input_node_ids_ar
    [5]: input_edge_ids_ar
    [6]: node_length_ar
    [7]: edge_length_ar
    [8]: adj_matrix_ar
    """

    if self.mode=="train":
      idx = self.train_id[idx]
    elif self.mode=="valid":
      idx = self.valid_id[idx]
    elif self.mode=="test":
      idx = self.test_id[idx]
    else:
      raise ValueError("mode is train or valid or test")

    # change tensor to int
    if torch.is_tensor(idx):
      idx = idx.item()
    
    in_edges = self.g.in_edges(idx)
    out_edges = self.g.out_edges(idx)

    input_ids = []
    node_ids = []
    edge_ids = []
    target_ids = []
    input_text = ""
    target_text = ""

    tokenized_node = []
    cnt_edge = 0

    adj_matrix = adj_matrix = [[-1] * (self.cfg.model.max_node_length + 1) for _ in range(self.cfg.model.max_node_length + 1)]

    # dict key=head, value=[relation, tail]
    triples = dict()
    # set entities_set and relations_set
    entities_set = set()
    relations_set = set()

    # pick up incoming_edges
    if self.cfg.model.is_in_edge:
      for entity_id in in_edges[0]:
        entity_id = entity_id.item()
        try:
          head = self.mid2name[self.id2e[entity_id]] if self.mid2name is not None else self.id2e[entity_id]
        except KeyError:
          head =  self.id2e[entity_id]
        rel = self.id2r[self.g.edata["etype"][self.g.edge_ids(entity_id,idx)].item()]
        try:
          tail = self.mid2name[self.id2e[idx]] if self.mid2name is not None else self.id2e[idx]
        except KeyError:
          tail = self.id2e[idx]
        entities_set.add(head)
        entities_set.add(tail)
        relations_set.add(rel)
        if triples.get(head):
          triples[head].extend([rel, tail]) 
        else:
          triples[head] = [rel, tail]
    # pick up outgoing_edges
    if self.cfg.model.is_out_edge:
      for entity_id in out_edges[1]:
        entity_id = entity_id.item()
        try:
          tail = self.mid2name[self.id2e[entity_id]] if self.mid2name is not None else self.id2e[entity_id] 
        except KeyError:
          tail = self.id2e[entity_id]
        rel = self.id2r[self.g.edata["etype"][self.g.edge_ids(idx,entity_id)].item()]
        try:
          head = self.mid2name[self.id2e[idx]] if self.mid2name is not None else self.id2e[idx] 
        except KeyError:
          head = self.id2e[idx]
        entities_set.add(head)
        entities_set.add(tail)
        relations_set.add(rel)
        if triples.get(head):
          triples[head].extend([rel, tail]) 
        else:
          triples[head] = [rel, tail]
    
    # get entity_tokenized_ids relation_tokenized_ids and numbering
    entity_tokenized_ids_dict, relation_tokenized_ids_dict = self.get_change_per_sample(triples, entities_set, relations_set)

    input_ids, input_text, node_ids, edge_ids, cnt_edges, adj_matrix = self.linearize(triples, entity_tokenized_ids_dict,  relation_tokenized_ids_dict, self.head_ids, self.rel_ids, self.tail_ids, cnt_edge, adj_matrix)

    # create target_ids
    if self.mode=="train":
      type_list = self.train_label[idx].nonzero().flatten()
    elif self.mode=="valid":
      type_list = self.valid_label[idx].nonzero().flatten()
    elif self.mode=="test":
      type_list = self.test_label[idx].nonzero().flatten()
    else:
      raise ValueError("mode sholed be train or valid or test")
    for type_id in type_list:
      type_id = type_id.item() if torch.is_tensor(type_id) else type_id
      type_name = self.id2t[type_id]
      target_ids += self.tokenizer.encode(" {}".format(type_name), add_special_tokens=False)
      target_text += ' ' + copy.deepcopy(type_name)

    input_ids_ar, attn_mask_ar, target_ids, target_attn_mask, input_node_ids_ar, input_edge_ids_ar = \
            self.prep_data(target_ids, input_ids, self.add_bos_id, self.graph_ids,
                              self.text_ids, node_ids, edge_ids)

    node_length_ar = max(input_node_ids_ar) + 1
    edge_length_ar = max(input_edge_ids_ar) + 1

    def masked_fill(src, masked_value, fill_value):
        """
        ids within src will be replaced by fill_value conditioned on id==masked_value

        Parameter

        src: list (e.g. [-1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, -1, -1, -1, ...])
        masked_value: int (e.g. -1)
        fill_value: int (e.g. 50)

        Return

        list (e.g. [50, 50, 50, 50, 50, 50, 50, 50, 0, 0, 0, 50, 50, 50, ...])
        """
        return [src[src_id] if src[src_id] != masked_value and src[src_id] < fill_value else fill_value for src_id
                in range(len(src))]

    input_node_ids_ar, input_edge_ids_ar = masked_fill(input_node_ids_ar, -1, self.cfg.model.max_node_length), \
                                            masked_fill(input_edge_ids_ar, -1, self.cfg.model.max_edge_length)

    def masked_fill_matrix(adj_matrix_input, masked_value, fill_value):
        """
        ids within src will be replaced by fill_value conditioned on id==masked_value

        Parameter

        src: list (e.g. [[-1, 0, -1, -1, -1, -1, -1, -1, -1, ...], [-1, -1, -1, -1, -1, -1, -1, -1, -1, ...], [-1, -1, -1, -1, -1, -1, -1, -1, -1, ...], [-1, -1, -1, -1, -1, -1, -1, -1, -1, ...], [-1, -1, -1, -1, -1, -1, -1, -1, -1, ...], [-1, -1, -1, -1, -1, -1, -1, -1, -1, ...], [-1, -1, -1, -1, -1, -1, -1, -1, -1, ...], [-1, -1, -1, -1, -1, -1, -1, -1, -1, ...], [-1, -1, -1, -1, -1, -1, -1, -1, -1, ...], [-1, -1, -1, -1, -1, -1, -1, -1, -1, ...], [-1, -1, -1, -1, -1, -1, -1, -1, -1, ...], [-1, -1, -1, -1, -1, -1, -1, -1, -1, ...], [-1, -1, -1, -1, -1, -1, -1, -1, -1, ...], [-1, -1, -1, -1, -1, -1, -1, -1, -1, ...], ...])
        masked_value: int (e.g. -1)
        fill_value: int (e.g. 60)

        Return

        list (e.g. [[60, 0, 60, 60, 60, 60, 60, 60, 60, ...], [60, 60, 60, 60, 60, 60, 60, 60, 60, ...], [60, 60, 60, 60, 60, 60, 60, 60, 60, ...], [60, 60, 60, 60, 60, 60, 60, 60, 60, ...], [60, 60, 60, 60, 60, 60, 60, 60, 60, ...], [60, 60, 60, 60, 60, 60, 60, 60, 60, ...], [60, 60, 60, 60, 60, 60, 60, 60, 60, ...], [60, 60, 60, 60, 60, 60, 60, 60, 60, ...], [60, 60, 60, 60, 60, 60, 60, 60, 60, ...], [60, 60, 60, 60, 60, 60, 60, 60, 60, ...], [60, 60, 60, 60, 60, 60, 60, 60, 60, ...], [60, 60, 60, 60, 60, 60, 60, 60, 60, ...], [60, 60, 60, 60, 60, 60, 60, 60, 60, ...], [60, 60, 60, 60, 60, 60, 60, 60, 60, ...], ...])
        """
        adj_matrix_tmp = copy.deepcopy(adj_matrix_input)
        for a_id in range(len(adj_matrix_tmp)):
            for b_id in range(len(adj_matrix_tmp)):
                if adj_matrix_tmp[a_id][b_id] == masked_value or adj_matrix_tmp[a_id][b_id] > fill_value:
                    adj_matrix_tmp[a_id][b_id] = fill_value
        return adj_matrix_tmp

    adj_matrix_ar = masked_fill_matrix(adj_matrix, -1, self.cfg.model.max_edge_length)

    assert len(input_ids_ar) == len(attn_mask_ar) == self.cfg.model.max_input_length == len(input_node_ids_ar) == len(
        input_edge_ids_ar)
    assert len(target_ids) == len(target_attn_mask) == self.cfg.model.max_output_length

    input_ids_ar = torch.LongTensor(input_ids_ar)
    attn_mask_ar = torch.LongTensor(attn_mask_ar)
    target_ids = torch.LongTensor(target_ids)
    target_attn_mask = torch.LongTensor(target_attn_mask)
    input_node_ids_ar = torch.LongTensor(input_node_ids_ar)
    input_edge_ids_ar = torch.LongTensor(input_edge_ids_ar)
    node_length_ar = torch.LongTensor([node_length_ar])
    edge_length_ar = torch.LongTensor([edge_length_ar])
    adj_matrix_ar = torch.LongTensor(adj_matrix_ar)

    return input_ids_ar, attn_mask_ar, target_ids, target_attn_mask, \
            input_node_ids_ar, input_edge_ids_ar, node_length_ar, edge_length_ar, adj_matrix_ar

class EntityTypingJointGTDataLoader(DataLoader):

  def __init__(self, cfg, dataset, mode):
    if mode=="train":
      sampler = RandomSampler(dataset)
      batch_size = cfg.model.train_batch_size
    elif mode=="valid":
      sampler = SequentialSampler(dataset)
      batch_size = cfg.model.valid_batch_size
    elif mode=="test":
      sampler = SequentialSampler(dataset)
      batch_size = cfg.model.test_batch_size

    super(EntityTypingJointGTDataLoader, self).__init__(dataset, sampler=sampler, batch_size=batch_size, num_workers=cfg.model.num_workers)