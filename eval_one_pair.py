import hydra
import logging
# from dgl.dataloading import NodeDataLoader, MultiLayerFullNeighborSampler, MultiLayerNeighborSampler
# from dgl.dataloading import NodeDataLoader, MultiLayerFullNeighborSampler
from omegaconf import DictConfig
# from utils import *
from utils import set_logger
import os
import torch
from JointGT import JointGT
from T5 import T5
from Bart import Bart
from bert_score import score
from bert_score import BERTScorer

from transformers import BartTokenizer, T5Tokenizer
from data import EntityTypingJointGTDataset, EntityTypingJointGTDataLoader, EntityTypingT5Dataset, EntityTypingT5DataLoader
from data import EntityTypingBartDataset, EntityTypingBartDataLoader

from tqdm import tqdm
import nltk.translate.bleu_score as bleu
@hydra.main(config_path="config", config_name='config')
def main(cfg: DictConfig) -> None:
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
        test_unobserved_dataset = EntityTypingBartDataset(cfg, data_path, cfg.model.test.unobserved_test_dataset, tokenizer, "test")
        # dataloader
        test_unobserved_dataloader = EntityTypingBartDataLoader(cfg, test_unobserved_dataset, "test")
    elif cfg.model.name == 'T5':
        test_dataset = EntityTypingT5Dataset(cfg, data_path, cfg.model.test.test_dataset, tokenizer, "test")
        # dataloader
        test_dataloader = EntityTypingT5DataLoader(cfg, test_dataset, "test")

        # unobserved dataset
        test_unobserved_dataset = EntityTypingBartDataset(cfg, data_path, cfg.model.test.unobserved_test_dataset, tokenizer, "test")
        # dataloader
        test_unobserved_dataloader = EntityTypingBartDataLoader(cfg, test_unobserved_dataset, "test")
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
    # optimizer = torch.optim.Adam(
    #     filter(lambda p: p.requires_grad, model.parameters()),
    #     lr=cfg.model.optimizer.learning_rate,
    # )

    # test
    if cfg.model.test.get_from_model:
        with torch.no_grad():
            model.load_state_dict(torch.load(os.path.join(save_path, 'best_model.pkl')))
            model.eval()
            sources = []
            predictions = []
            targets = []
            for batch in tqdm(test_dataloader, desc="Iteration"):
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
                break
            if cfg.model.test.save_outputs:
                with open(os.path.join(save_path, f'{cfg.model.test.test_dataset.split(".")[0]}_inputs_test.txt'), 'w', encoding='utf-8') as f:
                    for src_txt in sources:
                        f.write(src_txt+"\n")
                with open(os.path.join(save_path, f'{cfg.model.test.test_dataset.split(".")[0]}_outputs_test.txt'), 'w', encoding='utf-8') as f:
                    for pred_txt in predictions:
                        f.write(pred_txt+"\n")
                with open(os.path.join(save_path, f'{cfg.model.test.test_dataset.split(".")[0]}_targets_test.txt'), 'w', encoding='utf-8') as f:
                    for target_txt in targets:
                        f.write(target_txt+"\n")
            evaluation(cfg, targets, predictions)
            
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
                break
            if cfg.model.test.save_outputs:
                input_test_path = f'{cfg.model.test.unobserved_test_dataset.split(".")[0]}_inputs_test.txt'
                with open(os.path.join(save_path, input_test_path), 'w', encoding='utf-8') as f:
                    for src_txt in sources:
                        f.write(src_txt+"\n")
                output_test_path = f'{cfg.model.test.unobserved_test_dataset.split(".")[0]}_outputs_test.txt'
                with open(os.path.join(save_path, output_test_path), 'w', encoding='utf-8') as f:
                    for pred_txt in predictions:
                        f.write(pred_txt+"\n")
                target_test_path = f'{cfg.model.test.unobserved_test_dataset.split(".")[0]}_targets_test.txt'
                with open(os.path.join(save_path, target_test_path), 'w', encoding='utf-8') as f:
                    for target_txt in targets:
                        f.write(target_txt+"\n")
            evaluation(cfg, targets, predictions)
      
    elif cfg.model.test.get_from_file:
        # with open(os.path.join(f'{cfg.model.test.file_path}_inputs_test.txt'), 'r', encoding='utf-8') as f:
        #     inputs = f.readlines()
        with open(os.path.join(f'{cfg.model.test.file_path}_outputs_test.txt'), 'r', encoding='utf-8') as f:
            predictions = f.readlines()
        with open(os.path.join(f'{cfg.model.test.file_path}_targets_test.txt'), 'r', encoding='utf-8') as f:
            targets = f.readlines()
        logging.info("test")
        # evaluation(cfg, targets, predictions)

        predictions_tokens = []
        targets_tokens = []
        for i in range(len(predictions)):
            predictions_tokens.append([predictions[i].strip().split("[sep]")[0].strip()])
        for i in range(len(targets)):
            targets_tokens.append([item.strip() for item in targets[i].strip().split("[sep]") if item != "\n" and item != ''])

        prdcts, tgts = bert_classification(predictions_tokens, targets_tokens, all_comb=False)
        logging.info("test for classified token")
        evaluation(cfg, tgts, prdcts)


def evaluation(cfg, targets, predictions):
    assert len(targets) == len(predictions)
    score_1gram, score_2gram, score_3gram, score_4gram = calc_bleu_score(targets, predictions)
    
    logging.info("test data set : {}".format(cfg.model.test.test_dataset))
    logging.info("N-grams: 1-{}, 2-{}, 3-{}, 4-{}".format(score_1gram, score_2gram, score_3gram, score_4gram))

    P, R, F1 = calc_bert_score(targets, predictions)
    logging.info("BERT-P:%f, BERT-R:%f, BERT-F1:%f" % (P, R, F1))


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
        total_score_1gram += bleu.sentence_bleu([ref], hyp, weights=(1, 0, 0, 0))
        total_score_2gram += bleu.sentence_bleu([ref], hyp, weights=(0, 1, 0, 0))
        total_score_3gram += bleu.sentence_bleu([ref], hyp, weights=(0, 0, 1, 0))
        total_score_4gram += bleu.sentence_bleu([ref], hyp, weights=(0, 0, 0, 1))
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
    Precision, Recall, F1 = score(predictions, targets, lang="en", verbose=True)
    for p, r, f1 in zip(Precision.numpy().tolist(), Recall.numpy().tolist(), F1.numpy().tolist()):
        total_score_p += p
        total_score_r += r
        total_score_f1 += f1
    score_p = total_score_p/len(Precision)
    score_r = total_score_r/len(Recall)
    score_f1 = total_score_f1/len(F1)
    return score_p, score_r, score_f1


def bert_classification(predictions_tokens, targets_tokens, all_comb=True):
    """
    input:
    predictions_tokens: 2dim list (num_predictions x num_separated_tokens)
    targets_tokens: 2dim list (num_targets x num_separated_tokens)

    output:
    predicts: 1dim list
    targets: 1dim list

    e.g.
    predictions_tokens
    [['wikicat Living people',
    'wikicat Hartlepool United F.C. players',
    'wikicat Clyde F.C. players',
    'wikicat Dundee F.C. players',
    'wikicat St. Mirren F.C. players',
    'wikicat Scottish Football League managers',
    'wikicat Scotland international footballers',
    'wikicat Forfar Athletic F.C. players',
    'wikicat Dumbarton F.C.'],
    ['wordnet person 100007846',
    'wikicat Screenwriters',
    'wikicat American people of English descent',
    'wikicat People from Kansas City, Missouri',
    'wikicat American film directors',
    'wikicat Film producers',
    'wikicat American screenwriters',
    'wikicat American people',
    'wikicat American film directors',
    'wikicat American screenwriters',
    'wikicat American people'],
    ['wikicat Football Conference players',
    'wikicat Living people',
    'wikicat The Football League players',
    'wikicat Hartlepool United F.C. players',
    'wikicat Peterborough United F.C. players',
    ...
    ['wikicat Former countries in Europe',
    'wikicat Former countries in the British Isles',
    'wikicat Former countries in Europe'],
    ['wikicat Provinces of Flanders',
    'wordnet administrative district 108491826']]

    targets_tokens
    [['wikicat St. Mirren F.C. players'],
    ['wikicat Film directors from Missouri'],
    ['wikicat Notts County F.C. players'],
    ['wikicat States and territories established in 1649'],
    ['wikicat Provinces of Belgium']]


    output:
    predicts
    ['wikicat St. Mirren F.C. players', 'wikicat People from Kansas City, Missouri', 'wikicat Notts County F.C. players', 
    'wikicat Former countries in the British Isles', 'wikicat Provinces of Flanders']

    targets:
    ['wikicat St. Mirren F.C. players', 'wikicat Film directors from Missouri', 'wikicat Notts County F.C. players',
    'wikicat States and territories established in 1649', 'wikicat Provinces of Belgium']
    """
    # 全ての組み合わせを格納
    list_all_combination_target = []
    list_all_combination_prediction = []
    list_combination_count = []
    for i in range(len(targets_tokens)):
        for j in range(len(targets_tokens[i])):
            for k in range(len(predictions_tokens[i])):
                list_all_combination_target.append(targets_tokens[i][j])
                list_all_combination_prediction.append(predictions_tokens[i][k])
        list_combination_count.append(len(targets_tokens[i]) * len(predictions_tokens[i]))

    # 全ての組み合わせのBERTスコアを取得
    scorer = BERTScorer(lang="en")
    Precision, Recall, F1 = scorer.score(list_all_combination_prediction, list_all_combination_target)

    # list_combination_countの累積和を取得
    def cumsum(comb_list: list):
        sum_comb_list = []
        sum = 0
        for item in comb_list:
            sum_comb_list.append(sum+item)
            sum += item
        return sum_comb_list
    list_combination_count_with_zero = [0] + list_combination_count
    sum_combination_count = cumsum(list_combination_count_with_zero)
        
    list_comb = []
    for i in range(1, len(list_combination_count_with_zero)):
        tmp_combination_targets = list_all_combination_target[sum_combination_count[i-1]:sum_combination_count[i]]
        tmp_combination_predictions = list_all_combination_prediction[sum_combination_count[i-1]:sum_combination_count[i]]
        tmp_F1 = F1[sum_combination_count[i-1]:sum_combination_count[i]]
        assert len(tmp_combination_targets) == len(tmp_combination_predictions)
        assert len(tmp_combination_predictions) == len(tmp_F1)

        set_picked_prediction_type = set()
        set_picked_target_type = set()

        # loop until tmp_F1 is null
        while len(tmp_F1) != 0:
            # get max index
            max_index = torch.argmax(tmp_F1)

            # get combination　（尤もらしい組を格納しておく）
            list_comb.append(tuple((tmp_combination_predictions[max_index], tmp_combination_targets[max_index])))
            set_picked_prediction_type.add(tmp_combination_predictions[max_index])
            set_picked_target_type.add(tmp_combination_targets[max_index])

            # remove entity from tmp_combination_targets and tmp_combination_predictions
            list_delete_target_index = []
            list_delete_prediction_index = []
            for index, item in enumerate(tmp_combination_targets):
                if item == tmp_combination_targets[max_index]:
                    list_delete_target_index.append(index)
            for index, item in enumerate(tmp_combination_predictions):
                if item == tmp_combination_predictions[max_index]:
                    list_delete_prediction_index.append(index)
            list_select_index = []
            for i in range(len(tmp_F1)):
                if (
                        len(list_delete_prediction_index) > 0
                        and i == list_delete_prediction_index[0]
                        and len(list_delete_target_index) > 0
                        and i == list_delete_target_index[0]
                   ):
                    list_delete_prediction_index.pop(0)
                    list_delete_target_index.pop(0)
                elif len(list_delete_prediction_index) > 0 and i == list_delete_prediction_index[0]:
                    list_delete_prediction_index.pop(0)
                elif len(list_delete_target_index) > 0 and i == list_delete_target_index[0]:
                    list_delete_target_index.pop(0)
                else:
                    # tmp_F1から削除しないインデックスの保存
                    list_select_index.append(i)
            if len(list_select_index) > 0:
                tmp_F1 = torch.index_select(tmp_F1, 0, torch.tensor(list_select_index))
                tmp_combination_targets = [item for i, item in enumerate(tmp_combination_targets) if i in list_select_index]
                tmp_combination_predictions = [item for i, item in enumerate(tmp_combination_predictions) if i in list_select_index]
            else:
                if all_comb:
                    # target, predictのいずれかが多い場合はその組み合わせも取り出す
                    for index, item in enumerate(tmp_combination_targets):
                        if item not in set_picked_target_type:
                            list_comb.append(tuple(("", item)))
                    for index, item in enumerate(tmp_combination_predictions):
                        if item not in set_picked_prediction_type:
                            list_comb.append(tuple((item, "")))
                tmp_F1 = torch.Tensor()
                tmp_combination_targets = []
                tmp_combination_predictions = []
            assert len(tmp_combination_targets) == len(tmp_combination_predictions)
            assert len(tmp_combination_predictions) == len(tmp_F1)

    def get_target_predict(list_comb):
        predicts = []
        targets = []
        for index_list_comb in range(len(list_comb)):
            predict, target = list_comb[index_list_comb]
            predicts.append(predict)
            targets.append(target)
        return predicts, targets
    predicts, targets = get_target_predict(list_comb)
    return predicts, targets



if __name__ == '__main__':
    main()
