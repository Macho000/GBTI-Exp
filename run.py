import hydra
import logging
from dgl.dataloading import NodeDataLoader, MultiLayerFullNeighborSampler, MultiLayerNeighborSampler
from omegaconf import DictConfig, OmegaConf
from utils import *
import torch
from JointGT import JointGT

from transformers import BartTokenizer, T5Tokenizer
from data import EntityTypingJointGTDataset, EntityTypingJointGTDataLoader

from tqdm import tqdm
from tqdm import trange

@hydra.main(config_path="config", config_name='config')
def main(cfg : DictConfig) -> None:
    set_logger(cfg)
    use_cuda = cfg.model.cuda and torch.cuda.is_available()
    data_path = os.path.join(cfg.data.data_dir, cfg.data.name, 'clean')
    save_path = os.path.join(cfg.model.save_dir, cfg.data.name)

    # Initialize tokenizer
    if cfg.model.pretrained_model == "Bart":
        tokenizer = BartTokenizer.from_pretrained(cfg.model.tokenizer_path)
    elif cfg.model.pretrained_model == "T5":
        tokenizer = T5Tokenizer.from_pretrained(cfg.model.tokenizer_path)

    data_path = os.path.join(cfg.data.data_dir, cfg.data.name, 'clean')
    train_dataset = EntityTypingJointGTDataset(cfg, data_path, tokenizer, "train")
    valid_dataset = EntityTypingJointGTDataset(cfg, data_path, tokenizer, "valid")

    # dataloader
    train_dataloader = EntityTypingJointGTDataLoader(cfg, train_dataset, "train")
    valid_dataloader = EntityTypingJointGTDataLoader(cfg, valid_dataset, "valid")


    # model
    if cfg.model.name == 'JointGT':
        model = JointGT(cfg)
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
        lr=cfg.model.optimizer.learning_rate,
    )

    # loss
    criterion = torch.nn.BCELoss()

    # training
    max_valid_loss = 0
    model.train()
    train_iterator = trange(int(cfg.model.max_epoch), desc="Epoch")
    logging.debug('Starting training!')
    for count_epoch in train_iterator:
        for batch_index, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            if torch.cuda.is_available():
                batch = [b.to(torch.device("cuda")) for b in batch]
            if count_epoch==0 and batch_index==0:
                for tmp_id in range(9):
                    logging.debug('batch %s: %s ' % (str(tmp_id), str(batch[tmp_id])))

            if cfg.model.name == 'JointGT':
                loss = model(batch, is_training=True)
                if cfg.model.n_gpus > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if torch.isnan(loss).data:
                    logging.debug("Stop training because loss=%s" % (loss.data))
                    stop_training = True
                    break

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Gradient accumulation
                if count_epoch % cfg.model.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.model.max_grad_norm)
                    optimizer.step()  # We have accumulated enough gradients
                    # scheduler.step()
                    model.zero_grad()

        # validation
        if ( count_epoch != 0 and count_epoch % cfg.model.valid_epoch==0 ) or cfg.model.debug:
            #torch.save(model.state_dict(), os.path.join(save_path, 'model.pkl'))
            model.eval()
            with torch.no_grad():
                for batch in valid_dataloader:
                    sources = []
                    predictions = []
                    if torch.cuda.is_available():
                        batch = [b.to(torch.device("cuda")) for b in batch]

                    outputs = model.generate(batch)
                    # Convert ids to tokens
                    for input_, output in zip(batch[0], outputs):
                        source = tokenizer.decode(input_, skip_special_tokens=True, clean_up_tokenization_spaces=cfg.model.valid.clean_up_spaces)
                        sources.append(source.strip())
                        pred = tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=cfg.model.valid.clean_up_spaces)
                        predictions.append(pred.strip())

                    valid_loss = model(batch, is_training=False)
                    if valid_loss < max_valid_loss and cfg.model.early_stopping:
                        logging.debug('early stop')
                        break
                    else:
                        torch.save(model.state_dict(), os.path.join(save_path, 'best_model.pkl'))
                        max_valid_loss = valid_loss

def evaluate(path, predict, all_true, e2id, t2id):
    logs = []
    f = open('./rank.txt', 'w', encoding='utf-8')
    with open(path, 'r', encoding='utf-8') as r:
        for line in r:
            e, t = line.strip().split('\t')
            e, t = e2id[e], t2id[t]
            tmp = predict[e] - all_true[e]
            tmp[t] = predict[e, t]
            argsort = torch.argsort(tmp, descending=True)
            ranking = (argsort == t).nonzero()
            assert ranking.size(0) == 1
            ranking = ranking.item() + 1
            print(line.strip(), ranking, file=f)
            logs.append({
                'MRR': 1.0 / ranking,
                'MR': float(ranking),
                'HIT@1': 1.0 if ranking <= 1 else 0.0,
                'HIT@3': 1.0 if ranking <= 3 else 0.0,
                'HIT@10': 1.0 if ranking <= 10 else 0.0
            })
    MRR = 0
    for metric in logs[0]:
        tmp = sum([_[metric] for _ in logs]) / len(logs)
        if metric == 'MRR':
            MRR = tmp
        logging.debug('%s: %f' % (metric, tmp))
    return MRR
                    
if __name__=='__main__':
  main()
