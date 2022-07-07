import hydra
from dgl.dataloading import NodeDataLoader, MultiLayerFullNeighborSampler, MultiLayerNeighborSampler
from omegaconf import DictConfig, OmegaConf
from utils import *
import torch
import JointGT


@hydra.main(config_path="config", config_name='config')
def main(cfg : DictConfig) -> None:
    set_logger(cfg)
    use_cuda = cfg.model.cuda and torch.cuda.is_available()
    data_path = os.path.join(cfg.data.data_dir, cfg.data.name, 'clean')
    save_path = os.path.join(cfg.model.save_dir, cfg.data.name)

    # graph
    e2id = read_id(os.path.join(data_path, 'entities.tsv'))
    id2e = read_entity(os.path.join(data_path, 'entities.tsv'))
    r2id = read_id(os.path.join(data_path, 'relations.tsv'))
    id2r = read_entity(os.path.join(data_path, 'relations.tsv'))
    length_r2id = len(r2id)
    r2id['type'] = length_r2id
    id2r[length_r2id] = 'type'
    t2id = read_id(os.path.join(data_path, 'types.tsv'))
    id2t = read_entity(os.path.join(data_path, 'types.tsv'))
    if cfg.data.name=="FB15kET": mid2name = read_name(os.path.join(data_path, 'mid2name.tsv'))
    num_entity = len(e2id)
    num_rels = len(r2id)
    num_types = len(t2id)
    num_nodes = num_entity + num_types
    g, train_label, all_true, train_id, valid_id, test_id = load_graph(data_path, e2id, r2id, t2id,
                                                                       cfg.preprocess.load_ET, cfg.preprocess.load_KG)

    if cfg.preprocess.neighbor_sampling:
        train_sampler = MultiLayerNeighborSampler([cfg.preprocess.neighbor_num] * cfg.preprocess.num_layers)

    else:
        train_sampler = MultiLayerFullNeighborSampler(cfg.preprocess.num_layers)
    test_sampler = MultiLayerFullNeighborSampler(cfg.preprocess.num_layers)
    train_dataloader = NodeDataLoader(
        g, train_id, train_sampler,
        batch_size=cfg.model.train_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=1
    )
    valid_dataloader = NodeDataLoader(
        g, valid_id, test_sampler,
        batch_size=cfg.model.test_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=6
    )

    # preprocess train and valid for language model's input
    if not os.path.exists(os.path.join(data_path, 'train_input.txt')):
        # make tuple (subject, relation, object)
        text_input = []
        for input_nodes, output_nodes, blocks in train_dataloader:
            # get name from mid: mid2name[id2e[?]]
            # get name rels: id2r[?%num_rels]
            text_tuple = ""
            for index, entity_id in enumerate(blocks[0].srcdata['_ID']):
                if index==0:
                    continue
                # change id to name
                # check type or entity
                if entity_id.item()>=len(e2id):
                    # if entity_id is type
                     # check incoming or outcoming graph
                    if blocks[0].edata["etype"][index-1].item()>=num_rels:
                        # if outgoing graph
                        obj = mid2name[id2e[blocks[0].dstdata['_ID'].item()]]
                        rel = id2r[blocks[0].edata["etype"][index-1].item()%num_rels]
                        sub = id2t[entity_id.item()%len(e2id)]
                    else:
                        # if incoming graph
                        sub = mid2name[id2e[blocks[0].dstdata['_ID'].item()]]
                        rel = id2r[blocks[0].edata["etype"][index-1].item()]
                        obj = id2t[entity_id.item()%len(e2id)]
                else:
                    # if entity_id is entity
                    # check incoming or outcoming graph
                    if blocks[0].edata["etype"][index-1].item()>=num_rels:
                        # if outcoming graph
                        sub = mid2name[id2e[blocks[0].dstdata['_ID'].item()]]
                        rel = id2r[blocks[0].edata["etype"][index-1].item()%num_rels]
                        obj = mid2name[id2e[entity_id.item()]]
                    else:
                        # if incoming graph
                        sub = mid2name[id2e[blocks[0].dstdata['_ID'].item()]]
                        rel = id2r[blocks[0].edata["etype"][index-1].item()]
                        obj = mid2name[id2e[entity_id.item()]]
                text_tuple += " <H> "+ sub + " <R> " + rel + " <T> " + obj
            text_input.append(text_tuple.strip())
            with open(os.path.join(data_path, 'train_input.txt'), 'w') as f:
                for item in text_input:
                    f.write("%s\n" % item)

            
            break
    # model
    if cfg.model.name == 'JointGT':
        model = JointGT(cfg, num_nodes, num_rels, num_types)
    elif cfg.model.name == 'T5':
        model = T5()
    else:
        raise ValueError('No such model')

    if use_cuda:
        model = model.to('cuda')
    for name, param in model.named_parameters():
        logging.debug('Parameter %s: %s, require_grad=%s' % (name, str(param.size()), str(param.requires_grad)))

    # optimizer
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.optimizer.learning_rate,
    )

    # loss
    criterion = torch.nn.BCELoss()

    # training
    # max_valid_mrr = 0
    # model.train()
    # for epoch in range(cfg.model.max_epoch):
    #     log = []
    #     for input_nodes, output_nodes, blocks in train_dataloader:
    #         label = train_label[output_nodes, :]
    #         if use_cuda:
    #             blocks = [b.to(torch.device('cuda')) for b in blocks]
    #             label = label.cuda()
    #         loss = model(blocks)

    #         log.append({
    #             "loss": loss.item(),
    #         })

    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()

    #     avg_loss = sum([_['loss'] for _ in log]) / len(log)
    #     logging.debug('epoch %d: loss: %f' %
    #                   (epoch, avg_loss))

        # if epoch != 0 and epoch % cfg.model.valid.max_epoch == 0:
        #     torch.save(model.state_dict(), os.path.join(save_path, 'model.pkl'))
        #     model.eval()
        #     with torch.no_grad():
        #         predict = torch.zeros(num_entity, num_types, dtype=torch.half)
        #         for input_nodes, output_nodes, blocks in valid_dataloader:
        #             if use_cuda:
        #                 blocks = [b.to(torch.device('cuda')) for b in blocks]
        #             # predict[output_nodes] = model(blocks).cpu().half()
        #         valid_mrr = evaluate(os.path.join(data_path, 'ET_valid.txt'), predict, all_true, e2id, t2id)
        #     model.train()
        #     if valid_mrr < max_valid_mrr:
        #         logging.debug('early stop')
        #         break
        #     else:
        #         torch.save(model.state_dict(), os.path.join(save_path, 'best_model.pkl'))
        #         max_valid_mrr = valid_mrr


if __name__=='__main__':
  main()
