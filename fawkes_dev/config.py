import glob
import json
import os

DATASETS = {
    "pubfig": "../data/pubfig",
    "scrub": "/home/shansixioing/fawkes/data/scrub/",
    "vggface2": "/mnt/data/sixiongshan/data/vggface2/",
    "webface": "/mnt/data/sixiongshan/data/webface/",
    "youtubeface": "/mnt/data/sixiongshan/data/youtubeface/keras_flow_data/",
}


def main():
    config = {}
    for dataset in DATASETS.keys():
        path = DATASETS[dataset]
        if not os.path.exists(path):
            print("Dataset path for {} does not exist, skipped".format(dataset))
            continue
        train_dir = os.path.join(path, "train")
        test_dir = os.path.join(path, "test")
        if not os.path.exists(train_dir):
            print("Training dataset path for {} does not exist, skipped".format(dataset))
            continue
        num_classes = len(os.listdir(train_dir))
        num_images = len(glob.glob(os.path.join(train_dir, "*/*")))
        if num_images == 0 or num_classes == 0 or num_images == num_classes:
            raise Exception("Dataset {} is not setup as detailed in README.".format(dataset))

        config[dataset] = {"train_dir": train_dir, "test_dir": test_dir, "num_classes": num_classes,
                           "num_images": num_images}
        print("Successfully config {}".format(dataset))
    j = json.dumps(config)
    model_dir = os.path.join(os.path.expanduser('~'), '.fawkes')
    with open(os.path.join(model_dir, "config.json"), "wb") as f:
        f.write(j.encode())


if __name__ == '__main__':
    main()
