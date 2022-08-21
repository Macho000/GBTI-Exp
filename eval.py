import hydra
import logging
from dgl.dataloading import NodeDataLoader, MultiLayerFullNeighborSampler, MultiLayerNeighborSampler
from omegaconf import DictConfig, OmegaConf
from utils import *
import torch
from JointGT import JointGT
from T5 import T5
from Bart import Bart
from bert_score import BERTScorer

from transformers import BartTokenizer, T5Tokenizer
from data import EntityTypingJointGTDataset, EntityTypingJointGTDataLoader, EntityTypingT5Dataset, EntityTypingT5DataLoader, EntityTypingBartDataset, EntityTypingBartDataLoader

from tqdm import tqdm
from tqdm import trange

import nltk
import nltk.translate.bleu_score as bleu

@hydra.main(config_path="config", config_name='config')
def main(cfg : DictConfig) -> None:
  set_logger(cfg)
  use_cuda = cfg.model.cuda and torch.cuda.is_available()
  data_path = os.path.join(cfg.data.data_dir, cfg.data.name, 'clean')
  save_path = os.path.join(cfg.model.save_dir, cfg.data.name)

  # Initialize tokenizer
  if cfg.model.name == 'JointGT': 
    if cfg.model.pretrained_model == "Bart":
        tokenizer = BartTokenizer.from_pretrained(cfg.model.tokenizer_path)
    elif cfg.model.pretrained_model == "T5":
        tokenizer = T5Tokenizer.from_pretrained(cfg.model.tokenizer_path)
  elif cfg.model.name == "T5":
      tokenizer = T5Tokenizer.from_pretrained(cfg.model.pretrained_model)
  elif cfg.model.name == "Bart":
      tokenizer = BartTokenizer.from_pretrained(cfg.model.pretrained_model)
  else:
      raise ValueError("No such model!")

  data_path = os.path.join(cfg.data.data_dir, cfg.data.name, 'clean')
  if cfg.model.name == 'JointGT':
    test_dataset = EntityTypingJointGTDataset(cfg, data_path, cfg.model.test.test_dataset, tokenizer, "test")
    # dataloader
    test_dataloader = EntityTypingJointGTDataLoader(cfg, test_dataset, "test")

    # unobserved dataset
    test_unobserved_dataset = EntityTypingJointGTDataset(cfg, data_path, cfg.model.test.unobserved_test_dataset, tokenizer, "test")
    # dataloader
    test_unobserved_dataloader = EntityTypingJointGTDataLoader(cfg, test_unobserved_dataset, "test")
  elif cfg.model.name == 'T5':
    test_dataset = EntityTypingT5Dataset(cfg, data_path, cfg.model.test.test_dataset, tokenizer, "test")
    # dataloader
    test_dataloader = EntityTypingT5DataLoader(cfg, test_dataset, "test")

    # unobserved dataset
    test_unobserved_dataset = EntityTypingT5Dataset(cfg, data_path, cfg.model.test.unobserved_test_dataset, tokenizer, "test")
    # dataloader
    test_unobserved_dataloader = EntityTypingT5DataLoader(cfg, test_unobserved_dataset, "test")
  elif cfg.model.name == 'Bart':
    test_dataset = EntityTypingBartDataset(cfg, data_path, cfg.model.test.test_dataset, tokenizer, "test")
    # dataloader
    test_dataloader = EntityTypingBartDataLoader(cfg, test_dataset, "test")

    # unobserved dataset
    test_unobserved_dataset = EntityTypingBartDataset(cfg, data_path, cfg.model.test.unobserved_test_dataset, tokenizer, "test")
    # dataloader
    test_unobserved_dataloader = EntityTypingBartDataLoader(cfg, test_unobserved_dataset, "test")

  # model
  if cfg.model.name == 'JointGT':
      model = JointGT(cfg)
  elif cfg.model.name == 'T5':
      model = T5(cfg)
  elif cfg.model.name == 'Bart':
      model = Bart(cfg)
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

  # test
  if cfg.model.test.get_from_model:
    with torch.no_grad():
        model.load_state_dict(torch.load(os.path.join(save_path, 'best_model.pkl')))
        model.eval()
        sources = []
        predictions = []
        targets = []
        # for batch in tqdm(test_dataloader, desc="Iteration"):
        #     if torch.cuda.is_available():
        #         batch = [b.to(torch.device("cuda")) for b in batch]
        #     if use_cuda:
        #         batch = [b.to(torch.device('cuda')) for b in batch]
        #     outputs = model.generate(batch)
        #     # Convert ids to tokens
        #     for input_, output, target in zip(batch[0], outputs, batch[2]):
        #         source = tokenizer.decode(input_, skip_special_tokens=True, clean_up_tokenization_spaces=cfg.model.valid.clean_up_spaces)
        #         sources.append(source.strip())
        #         pred = tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=cfg.model.valid.clean_up_spaces)
        #         predictions.append(pred.strip())
        #         target = tokenizer.decode(target, skip_special_tokens=True, clean_up_tokenization_spaces=cfg.model.valid.clean_up_spaces)
        #         targets.append(target.strip())
        # if cfg.model.test.save_outputs:
        #     with open(os.path.join(save_path, f'{cfg.model.test.test_dataset.split(".")[0]}_inputs_test.txt'), 'w', encoding='utf-8') as f:
        #         for src_txt in sources:
        #             f.write(src_txt+"\n")
        #     with open(os.path.join(save_path, f'{cfg.model.test.test_dataset.split(".")[0]}_outputs_test.txt'), 'w', encoding='utf-8') as f:
        #         for pred_txt in predictions:
        #             f.write(pred_txt+"\n")
        #     with open(os.path.join(save_path, f'{cfg.model.test.test_dataset.split(".")[0]}_targets_test.txt'), 'w', encoding='utf-8') as f:
        #         for target_txt in targets:
        #             f.write(target_txt+"\n")
        # evaluation(cfg, targets, predictions)
        
        # unobserved test data
        sources = []
        predictions = []
        targets = []
        for batch in tqdm(test_unobserved_dataloader, desc="Iteration"):
            if torch.cuda.is_available():
                batch = [b.to(torch.device("cuda")) for b in batch]
            if use_cuda:
                batch = [b.to(torch.device('cuda')) for b in batch]
            outputs = model.generate(batch)
            # Convert ids to tokens
            for input_, output, target in zip(batch[0], outputs, batch[2]):
                source = tokenizer.decode(input_, skip_special_tokens=True, clean_up_tokenization_spaces=cfg.model.valid.clean_up_spaces)
                sources.append(source.strip())
                pred = tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=cfg.model.valid.clean_up_spaces)
                predictions.append(pred.strip())
                target = tokenizer.decode(target, skip_special_tokens=True, clean_up_tokenization_spaces=cfg.model.valid.clean_up_spaces)
                targets.append(target.strip())
        evaluation(cfg, targets, predictions)
        if cfg.model.test.save_outputs:
            with open(os.path.join(save_path, f'{cfg.model.test.unobserved_test_dataset.split(".")[0]}_inputs_test.txt'), 'w', encoding='utf-8') as f:
                for src_txt in sources:
                    f.write(src_txt+"\n")
            with open(os.path.join(save_path, f'{cfg.model.test.unobserved_test_dataset.split(".")[0]}_outputs_test.txt'), 'w', encoding='utf-8') as f:
                for pred_txt in predictions:
                    f.write(pred_txt+"\n")
            with open(os.path.join(save_path, f'{cfg.model.test.unobserved_test_dataset.split(".")[0]}_targets_test.txt'), 'w', encoding='utf-8') as f:
                for target_txt in targets:
                    f.write(target_txt+"\n")
            evaluation(cfg, targets, predictions)


  elif cfg.model.test.get_from_file:
    with open(os.path.join(f'{cfg.model.test.file_path}_inputs_test.txt'), 'r', encoding='utf-8') as f:
        inputs = f.readlines()
    with open(os.path.join(f'{cfg.model.test.file_path}_outputs_test.txt'), 'r', encoding='utf-8') as f:
        predictions = f.readlines()
    with open(os.path.join(f'{cfg.model.test.file_path}_targets_test.txt'), 'r', encoding='utf-8') as f:
        targets = f.readlines()
    evaluation(cfg, targets, predictions)

    # unobserved test data
    with open(os.path.join(save_path, f'{cfg.model.test.unobserved_test_dataset.split(".")[0]}_inputs_test.txt'), 'r', encoding='utf-8') as f:
        inputs = f.readlines()
    with open(os.path.join(save_path, f'{cfg.model.test.unobserved_test_dataset.split(".")[0]}_outputs_test.txt'), 'r', encoding='utf-8') as f:
        predictions = f.readlines()
    with open(os.path.join(save_path, f'{cfg.model.test.unobserved_test_dataset.split(".")[0]}_targets_test.txt'), 'r', encoding='utf-8') as f:
        targets = f.readlines()
    evaluation(cfg, targets, predictions)

def evaluation(cfg, targets, predictions):
  assert len(targets)==len(predictions)
  score_1gram, score_2gram, score_3gram, score_4gram = calc_bleu_score(targets, predictions)

  logging.info("test data set : {}".format(cfg.model.test.test_dataset))
  logging.info("N-grams: 1-{}, 2-{}, 3-{}, 4-{}".format(score_1gram, score_2gram, score_3gram, score_4gram))

#   P, R, F1 = calc_bert_score(targets, predictions)
#   logging.info("BERT-P:%f, BERT-R:%f, BERT-F1:%f" %(P, R, F1))

def calc_bleu_score(targets, predictions):
    """ BLEUスコアの算出

    Args:
        targets ([List[str]]): [比較対象の文]
        predictions ([List[str]]): [比較元の文]

    Returns:
        score_1gram, score_2gram, score_3gram, score_4gram
    """
    total_score_1gram = 0
    total_score_2gram = 0
    total_score_3gram = 0
    total_score_4gram = 0

    for tgt, pred in zip(targets, predictions):
        hyp = pred.split()
        ref = tgt.split()
        total_score_1gram += bleu.sentence_bleu([ref], hyp, weights=(1,0,0,0))
        total_score_2gram += bleu.sentence_bleu([ref], hyp, weights=(0,1,0,0))
        total_score_3gram += bleu.sentence_bleu([ref], hyp, weights=(0,0,1,0))
        total_score_4gram += bleu.sentence_bleu([ref], hyp, weights=(0,0,0,1))
    score_1gram = total_score_1gram/len(targets)
    score_2gram = total_score_2gram/len(targets)
    score_3gram = total_score_3gram/len(targets)
    score_4gram = total_score_4gram/len(targets)

    return score_1gram, score_2gram, score_3gram, score_4gram

def calc_bert_score(targets, predictions):
    """ BERTスコアの算出

    Args:
        targets ([List[str]]): [比較対象の文]
        predictions ([List[str]]): [比較元の文]

    Returns:
        Precision, Recall, F1スコア
    """
    total_score_p = 0
    total_score_r = 0
    total_score_f1 = 0
    scorer = BERTScorer(lang="en", verbose=True)
    Precision, Recall, F1 = scorer.score(predictions, targets)
    score_p = Precision.mean()
    score_r = Recall.mean()
    score_f1 = F1.mean()
    return score_p, score_r, score_f1



if __name__=='__main__':
  main()