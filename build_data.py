import argparse
from preprocessing.clean_split_scenes import clean_split_scenes
from preprocessing.train_test_val_videos import folders_to_process
from preprocessing.tag_videos import tag_scenes_with_labels


# Construct the argument parser.
parser = argparse.ArgumentParser()
parser.add_argument(
    '-g', '--games',
    type=str,
    default='DET_DAL',
    help='Game Abbreviation'
)

args = parser.parse_args()

if __name__ == '__main__':
    # Create a directory with the model name for outputs.
    clean_split_scenes(args.game)
    tag_scenes_with_labels(args.game)
    folders_to_process(args.game)

    ## Data Loading.
    train_crop_size = tuple(args.crop_size)
    train_resize_size = tuple(args.imgsz)
