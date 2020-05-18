import sys

sys.path.append("/home/shansixioing/tools/")
sys.path.append("/home/shansixioing/cloak/")

import argparse
import gen_utils
from tensorflow import set_random_seed
from encode_utils import *
import random
import pickle
import re
import locale

loc = locale.getlocale()
locale.setlocale(locale.LC_ALL, loc)

SEEDS = [12345, 23451, 34512, 45123, 51234, 54321, 43215, 32154, 21543, 15432]
IMG_SHAPE = [224, 224, 3]

MODEL = {
    'vggface1_inception': "0",
    'vggface1_dense': "1",
    "vggface2_inception": "2",
    "vggface2_dense": "3",
    "webface_dense": "4",
    "webface_inception": "5",
}

RES_DIR = '/home/shansixioing/cloak/results/'


def select_samples(data_dir):
    all_data_path = []
    for cls in os.listdir(data_dir):
        cls_dir = os.path.join(data_dir, cls)
        for data_path in os.listdir(cls_dir):
            all_data_path.append(os.path.join(cls_dir, data_path))

    return all_data_path


def generator_wrap(cloak_data, n_uncloaked, n_classes, test=False, validation_split=0.1):
    if test:
        # all_data_path = cloak_data.all_test_path
        all_data_path = select_samples(cloak_data.test_data_dir)
    else:
        # all_data_path = cloak_data.all_training_path
        all_data_path = select_samples(cloak_data.train_data_dir)
    split = int(len(cloak_data.cloaked_protect_train_X) * (1 - validation_split))
    cloaked_train_X = cloak_data.cloaked_protect_train_X[:split]
    if cloak_data.cloaked_sybil_train_X is not None:
        cloaked_sybil_X = cloak_data.cloaked_sybil_train_X #[:args.number_sybil * 131]
        #
        # for _ in range(len(cloaked_sybil_X) - 131):
        #     all_data_path.append(cloak_data.sybil_class_path[0])

    # random seed for selecting uncloaked pictures
    np.random.seed(12345)
    uncloaked_path = np.random.choice(cloak_data.protect_class_path, n_uncloaked).tolist()

    while True:
        batch_X = []
        batch_Y = []
        cur_batch_path = np.random.choice(all_data_path, args.batch_size)
        for p in cur_batch_path:
            cur_y = cloak_data.path2idx[p]
            # protect class and sybil class do not need to appear in test dataset
            if test and (re.search(cloak_data.protect_class, p) or re.search(cloak_data.sybil_class, p)):
                continue
            # protect class images in train dataset
            elif p in cloak_data.protect_class_path and p not in uncloaked_path:
                cur_x = random.choice(cloaked_train_X)
            # sybil class in train dataset
            elif p in cloak_data.sybil_class_path and cloak_data.cloaked_sybil_train_X is not None:
                cur_x = random.choice(cloaked_sybil_X)
            else:
                im = image.load_img(p, target_size=cloak_data.img_shape)
                im = image.img_to_array(im)
                cur_x = preprocess_input(im)
            batch_X.append(cur_x)
            batch_Y.append(cur_y)

        batch_X = np.array(batch_X)
        batch_Y = to_categorical(np.array(batch_Y), num_classes=n_classes)

        yield batch_X, batch_Y


def eval_uncloaked_test_data(cloak_data, n_classes):
    original_label = cloak_data.path2idx[list(cloak_data.protect_class_path)[0]]
    protect_test_X = cloak_data.protect_test_X
    original_Y = [original_label] * len(protect_test_X)
    original_Y = to_categorical(original_Y, n_classes)
    return protect_test_X, original_Y


def eval_cloaked_test_data(cloak_data, n_classes, validation_split=0.1):
    split = int(len(cloak_data.cloaked_protect_train_X) * (1 - validation_split))
    cloaked_test_X = cloak_data.cloaked_protect_train_X[split:]
    original_label = cloak_data.path2idx[list(cloak_data.protect_class_path)[0]]
    original_Y = [original_label] * len(cloaked_test_X)
    original_Y = to_categorical(original_Y, n_classes)
    return cloaked_test_X, original_Y


def main():
    SEED = SEEDS[args.seed_idx]
    random.seed(SEED)
    set_random_seed(SEED)
    gen_utils.init_gpu(args.gpu)

    if args.dataset == 'pubfig':
        N_CLASSES = 65
        CLOAK_DIR = "{}_tm{}_tgt57_r1.0_th{}".format(args.dataset, args.model_idx, args.th)
    elif args.dataset == 'scrub':
        N_CLASSES = 530
        CLOAK_DIR = "{}_tm{}_tgtPatrick_Dempsey_r1.0_th{}_joint".format(args.dataset, args.model_idx, args.th)
    elif args.dataset == 'webface':
        N_CLASSES = 10575
        CLOAK_DIR = "{}_tm{}_tgt1640351_r1.0_th0.01/".format(args.dataset, args.model_idx)
    else:
        raise ValueError
    print(CLOAK_DIR)

    CLOAK_DIR = os.path.join(RES_DIR, CLOAK_DIR)
    RES = pickle.load(open(os.path.join(CLOAK_DIR, "cloak_data.p"), 'rb'))

    print("Build attacker's model")
    cloak_data = RES['cloak_data']
    EVAL_RES = {}
    train_generator = generator_wrap(cloak_data, n_uncloaked=args.n_uncloaked, n_classes=N_CLASSES,
                                     validation_split=args.validation_split)
    test_generator = generator_wrap(cloak_data, test=True, n_uncloaked=args.n_uncloaked, n_classes=N_CLASSES,
                                    validation_split=args.validation_split)
    EVAL_RES['transfer_model'] = args.transfer_model
    if args.end2end:
        model = load_end2end_model("dense", N_CLASSES)
    else:
        base_model = load_extractor(args.transfer_model)
        model = load_victim_model(teacher_model=base_model, number_classes=N_CLASSES)

    original_X, original_Y = eval_uncloaked_test_data(cloak_data, N_CLASSES)
    cloaked_test_X, cloaked_test_Y = eval_cloaked_test_data(cloak_data, N_CLASSES,
                                                            validation_split=args.validation_split)

    model.fit_generator(train_generator, steps_per_epoch=cloak_data.number_samples // 32,
                        validation_data=(original_X, original_Y), epochs=args.n_epochs, verbose=2,
                        use_multiprocessing=True, workers=3)

    _, acc_original = model.evaluate(original_X, original_Y, verbose=0)
    print("Accuracy on uncloaked/original images TEST: {:.4f}".format(acc_original))
    EVAL_RES['acc_original'] = acc_original

    _, acc_cloaked = model.evaluate(cloaked_test_X, cloaked_test_Y, verbose=0)
    print("Accuracy on cloaked images TEST: {:.4f}".format(acc_cloaked))
    EVAL_RES['acc_cloaked'] = acc_cloaked

    # pred = model.predict_generator(test_generator, verbose=0, steps=10)
    # pred = np.argmax(pred, axis=1)
    # print(pred)
    _, other_acc = model.evaluate_generator(test_generator, verbose=0, steps=50)
    print("Accuracy on other classes {:.4f}".format(other_acc))
    EVAL_RES['other_acc'] = other_acc
    gen_utils.dump_dictionary_as_json(EVAL_RES,
                                      os.path.join(CLOAK_DIR, "{}_eval_sybil_uncloaked{}_seed{}_th{}.json".format(
                                          args.transfer_model, args.end2end, args.seed_idx, args.th)))


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', type=str,
                        help='GPU id', default='1')
    parser.add_argument('--n_uncloaked', type=int,
                        help='number of uncloaked images', default=0)
    parser.add_argument('--seed_idx', type=int,
                        help='random seed index', default=0)
    parser.add_argument('--dataset', type=str,
                        help='name of dataset', default='pubfig')
    parser.add_argument('--model_idx', type=str,
                        help='teacher model index', default="2")
    parser.add_argument('--transfer_model', type=str,
                        help='student model', default='vggface2_inception')
    parser.add_argument('--end2end', type=int,
                        help='whether use end2end', default=0)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--validation_split', type=float, default=0.1)
    parser.add_argument('--use_sybil', type=int,
                        help='whether use sybil class', default=0)
    parser.add_argument('--number_sybil', type=int,
                        help='whether use sybil class', default=1)
    parser.add_argument('--n_epochs', type=int, default=3)
    parser.add_argument('--th', type=float, default=0.01)
    parser.add_argument('--limit', type=int, default=0)
    return parser.parse_args(argv)


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main()
# python3 eval_cloak.py --gpu 2 --n_uncloaked 0 --dataset pubfig --model_idx 5 --transfer_model webface_inception