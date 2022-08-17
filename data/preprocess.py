import argparse
import os


def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='FB15kET')
    args, _ = parser.parse_known_args()
    return args

def remove_unobserved_entity_type(entity_set, type_set, src, dst, dst_unobserved_entity=None, dst_unobserved_type=None):
    '''
    Param
    entity_set
    type_set
    src
    dst
    dst_unobserved_entity
    dst_unobserved_type
    '''
    unobserved_entity_line = []
    unobserved_type_line = []
    with open(src, encoding='utf-8') as r:
        lines = r.readlines()
    with open(dst, 'w', encoding='utf-8') as w:
        for line in lines:
            e, t = line.strip().split()
            if e not in entity_set:
                unobserved_entity_line.append(line)
                continue
            if t not in type_set:
                unobserved_type_line.append(line)
                continue
            w.write(line)
    if dst_unobserved_entity is not None:
        with open(dst_unobserved_entity, 'w', encoding='utf-8') as w:
            for line in unobserved_entity_line:
                w.write(line)
    if dst_unobserved_type is not None:
        with open(dst_unobserved_type, 'w', encoding='utf-8') as w:
            for line in unobserved_type_line:
                w.write(line)

def filter_for_1_to_1_and_n_entity_type(entity_set, type_set, src, dst_1_1, dst_1_n, dst_unobserved_entity=None, dst_1_1_unobserved_entity=None, dst_1_n_unobserved_entity=None, dst_unobserved_type=None, dst_1_1_unobserved_type=None, dst_1_n_unobserved_type=None):
    tmp_entity_counter = {}
    one_to_one_data = []
    one_to_n_data = []
    unobserved_type = []
    unobserved_entity = []
    tmp_unobserved_entity_counter = {}
    one_to_one_unobserved_entity = []
    one_to_n_unobserved_entity = []
    tmp_unobserved_type_counter = {}
    one_to_one_unobserved_type = []
    one_to_n_unobserved_type = []
    with open(src, encoding='utf-8') as r:
        lines = r.readlines()
    for line in lines:
        e, t = line.strip().split("\t")
        if e not in entity_set:
            unobserved_entity.append(line)
            # count test unobserved entity
            if not tmp_unobserved_entity_counter.get(e):
                tmp_unobserved_entity_counter[e] = 1
            else:
                tmp_unobserved_entity_counter[e] += 1
            continue
        # store unobserved type
        if t not in type_set:
            unobserved_type.append(line)
            # count test unobserved entity
            if not tmp_unobserved_type_counter.get(e):
                tmp_unobserved_type_counter[e] = 1
            else:
                tmp_unobserved_type_counter[e] += 1
            continue
        # count test entity
        if not tmp_entity_counter.get(e):
            tmp_entity_counter[e] = 1
        else:
            tmp_entity_counter[e] += 1
    for line in lines:
        e, t = line.strip().split("\t")
        if e not in entity_set:
            # store 1 to 1 between test entities and types for unobserved entity
            if tmp_unobserved_entity_counter.get(e)==1:
                one_to_one_unobserved_entity.append(line)
            # store 1 to N between test entities and types for unobserved entity
            else:
                one_to_n_unobserved_entity.append(line)
            continue
        if t not in type_set:
            # store 1 to 1 between test entities and types for unobserved type
            if tmp_unobserved_type_counter.get(e)==1:
                one_to_one_unobserved_type.append(line)
            # store 1 to N between test entities and types for unobserved type
            else:
                one_to_n_unobserved_type.append(line)
            continue
        # store 1 to 1 between test entities and types
        if tmp_entity_counter.get(e)==1:
            one_to_one_data.append(line)
        # store 1 to N between test entities and types
        else:
            one_to_n_data.append(line)
    # save 1 to 1 data
    with open(dst_1_1, 'w', encoding='utf-8') as w:
        for line in one_to_one_data:
            w.write(line)
    # save 1 to n data
    with open(dst_1_n, 'w', encoding='utf-8') as w:
        for line in one_to_n_data:
            w.write(line)
    # save unobserved type entity
    if dst_unobserved_entity is not None:
        with open(dst_unobserved_entity, 'w', encoding='utf-8') as w:
            for line in unobserved_entity:
                w.write(line)
    # save 1 to 1 data for unobserved entity
    if dst_1_1_unobserved_entity is not None:
        with open(dst_1_1_unobserved_type, 'w', encoding='utf-8') as w:
            for line in one_to_one_unobserved_entity:
                w.write(line)
    # save 1 to n data for unobserved entity
    if dst_1_n_unobserved_entity is not None:
        with open(dst_1_n_unobserved_entity, 'w', encoding='utf-8') as w:
            for line in one_to_n_unobserved_entity:
                w.write(line)
    # save unobserved type
    if dst_unobserved_type is not None:
        with open(dst_unobserved_type, 'w', encoding='utf-8') as w:
            for line in unobserved_type:
                w.write(line)
    # save 1 to 1 data for unobserved type
    if dst_1_1_unobserved_type is not None:
        with open(dst_1_1_unobserved_type, 'w', encoding='utf-8') as w:
            for line in one_to_one_unobserved_type:
                w.write(line)
    # save 1 to n data for unobserved type
    if dst_1_n_unobserved_type is not None:
        with open(dst_1_n_unobserved_type, 'w', encoding='utf-8') as w:
            for line in one_to_n_unobserved_type:
                w.write(line)


def convert_ET2triples(src, dst):
    with open(src, encoding='utf-8') as f:
        data = f.readlines()
    with open(dst, 'w', encoding='utf-8') as f:
        for line in data:
            e, t = line.strip().split('\t')
            f.write(f'{e}\thastype\t{t}\n')


def save_id(src, dst):
    with open(dst, 'w', encoding='utf-8') as w:
        for idx, item in enumerate(src):
            w.write(f'{item}\t{idx}\n')


def main(args):
    # collect entity set
    entity = set()
    relation = set()
    train_entity = set()
    train_relation = set()
    valid_entity = set()
    valid_relation = set()
    test_entity = set()
    test_relation = set()
    with open(os.path.join(args.dataset, 'original/train.txt'), encoding='utf-8') as r:
        for line in r:
            h, r, t = line.strip().split('\t')
            entity.add(h)
            entity.add(t)
            relation.add(r)
            train_entity.add(h)
            train_entity.add(t)
            train_relation.add(r)
    with open(os.path.join(args.dataset, 'original/valid.txt'), encoding='utf-8') as r:
        for line in r:
            h, r, t = line.strip().split('\t')
            entity.add(h)
            entity.add(t)
            relation.add(r)
            valid_entity.add(h)
            valid_entity.add(t)
            valid_relation.add(r)
    with open(os.path.join(args.dataset, 'original/test.txt'), encoding='utf-8') as r:
        for line in r:
            h, r, t = line.strip().split('\t')
            entity.add(h)
            entity.add(t)
            relation.add(r)
            test_entity.add(h)
            test_entity.add(t)
            test_relation.add(r)

    # remove unobserved entities and collect types
    lines = []
    train_lines = []
    valid_lines = []
    test_lines = []
    types = set()
    train_types = set()
    valid_types = set()
    test_types = set()
    with open(os.path.join(args.dataset, 'original/Entity_Type_train.txt'), encoding='utf-8') as r:
        for line in r:
            e, t = line.strip().split('\t')
            if e in entity:
                lines.append(line)
                train_lines.append(line)
                types.add(t)
                train_types.add(t)
    with open(os.path.join(args.dataset, 'original/Entity_Type_valid.txt'), encoding='utf-8') as r:
        for line in r:
            e, t = line.strip().split('\t')
            if e in entity:
                lines.append(line)
                valid_lines.append(line)
                types.add(t)
                valid_types.add(t)
    with open(os.path.join(args.dataset, 'original/Entity_Type_test.txt'), encoding='utf-8') as r:
        for line in r:
            e, t = line.strip().split('\t')
            if e in entity:
                lines.append(line)
                test_lines.append(line)
                types.add(t)
                test_types.add(t)

    with open(os.path.join(args.dataset, 'clean/ET_train.txt'), 'w', encoding='utf-8') as w:
        w.writelines(train_lines)

    # create clean entity_type_valid/test set
    remove_unobserved_entity_type(train_entity, train_types,
                                  src=os.path.join(args.dataset, 'original/Entity_Type_valid.txt'),
                                  dst=os.path.join(args.dataset, 'clean/ET_valid.txt')
                                  )
    remove_unobserved_entity_type(train_entity, train_types,
                                  src=os.path.join(args.dataset, 'original/Entity_Type_test.txt'),
                                  dst=os.path.join(args.dataset, 'clean/ET_test.txt'),dst_unobserved_entity=os.path.join(args.dataset, 'clean/ET_unobserved_entity_test.txt'),
                                  dst_unobserved_type=os.path.join(args.dataset, 'clean/ET_unobserved_type_test.txt')
                                  )

    # copy train.txt to dir clean
    os.system(f'cp {args.dataset}/original/train.txt {args.dataset}/clean/train.txt')

    # save entity, relation, type
    save_id(entity, os.path.join(args.dataset, 'clean/entities.tsv'))
    save_id(relation, os.path.join(args.dataset, 'clean/relations.tsv'))
    save_id(types, os.path.join(args.dataset, 'clean/types.tsv'))

    # create data files for KGE methods
    convert_ET2triples(f'{args.dataset}/clean/ET_train.txt', f'{args.dataset}/merge/train.txt')
    with open(f'{args.dataset}/clean/train.txt', encoding='utf-8') as f:
        data = f.readlines()
    with open(f'{args.dataset}/merge/train.txt', 'a', encoding='utf-8') as f:
        f.writelines(data)
    convert_ET2triples(f'{args.dataset}/clean/ET_valid.txt', f'{args.dataset}/merge/valid.txt')
    convert_ET2triples(f'{args.dataset}/clean/ET_test.txt', f'{args.dataset}/merge/test.txt')

    with open(f'{args.dataset}/merge/types.txt', 'w', encoding='utf-8') as f:
        for t in types:
            f.write(t + '\n')

    # save 1 to 1 data for train data
    filter_for_1_to_1_and_n_entity_type(train_entity, train_types,
                                  src=os.path.join(args.dataset, 'original/Entity_Type_train.txt'),
                                  dst_1_1=os.path.join(args.dataset, 'clean/ET_1_1_train.txt'),
                                  dst_1_n=os.path.join(args.dataset, 'clean/ET_1_n_train.txt'))

    # save 1 to 1 data for valid data
    filter_for_1_to_1_and_n_entity_type(train_entity, train_types,
                                  src=os.path.join(args.dataset, 'original/Entity_Type_valid.txt'),
                                  dst_1_1=os.path.join(args.dataset, 'clean/ET_1_1_valid.txt'),
                                  dst_1_n=os.path.join(args.dataset, 'clean/ET_1_n_valid.txt'))
    
    # save 1 to 1 data for test data
    filter_for_1_to_1_and_n_entity_type(train_entity, train_types,
                                  src=os.path.join(args.dataset, 'original/Entity_Type_test.txt'),
                                  dst_1_1=os.path.join(args.dataset, 'clean/ET_1_1_test.txt'),
                                  dst_1_n=os.path.join(args.dataset, 'clean/ET_1_n_test.txt'),
                                  dst_unobserved_entity=os.path.join(args.dataset, 'clean/ET_unobserved_entity_test.txt'),
                                  dst_1_1_unobserved_entity=os.path.join(args.dataset, 'clean/ET_1_1_unobserved_entity_test.txt'),
                                  dst_1_n_unobserved_entity=os.path.join(args.dataset, 'clean/ET_1_n_unobserved_entity_test.txt'),
                                  dst_unobserved_type=os.path.join(args.dataset, 'clean/ET_unobserved_type_test.txt'),
                                  dst_1_1_unobserved_type=os.path.join(args.dataset, 'clean/ET_1_1_unobserved_type_test.txt'),
                                  dst_1_n_unobserved_type=os.path.join(args.dataset, 'clean/ET_1_n_unobserved_type_test.txt'))

if __name__ == '__main__':
    args = get_params()
    if args.dataset not in ['FB15kET', 'YAGO43kET']:
        raise ValueError(f'Dataset {args.dataset} is not exist')
    main(args)
