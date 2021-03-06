# Standard Modules
from __future__ import print_function
from os.path import join
from os import makedirs
from os import environ
import argparse
from time import gmtime, strftime
# image processing module
import SimpleITK as sitk
# ML Modules
from keras.utils import print_summary
# User Modules
from load_3D_data import load_data, split_data
from model_helper import create_model

time = strftime("%Y-%m-%d-%H:%M:%S", gmtime())


def main(args):
    # Ensure training, testing, and manip are not all turned off
    assert (args.train or args.test or args.manip),\
        'Cannot have train, test, and manip all set to 0, Nothing to do.'

    # Load the training, validation, and testing data
    try:
        train_list, val_list, test_list = load_data(args.data_root_dir,
                                                    args.split_num)
    except IndexError:
        # Create the training and test splits if not found
        split_data(args.data_root_dir, num_splits=4)
        train_list, val_list, test_list = load_data(args.data_root_dir,
                                                    args.split_num)

    # Get image properties from first image. Assume they are all the same.
    img_shape = sitk.GetArrayFromImage(sitk.ReadImage(join(
                                       args.data_root_dir, 'imgs',
                                       train_list[0][0]))).shape
    net_input_shape = (img_shape[0], img_shape[1], args.slices)
    print(net_input_shape)
    # Create the model for training/testing/manipulation
    model_list = create_model(args=args, input_shape=net_input_shape)
    print_summary(model=model_list[0], positions=[.38, .65, .75, 1.])

    args.output_name = args.save_prefix +\
        'split-' + str(args.split_num) + \
        '-batch-' + str(args.batch_size) + \
        '_shuff-' + str(args.shuffle_data) + \
        '_aug-' + str(args.aug_data) + \
        '_loss-' + str(args.loss) + \
        '_strid-' + str(args.stride) + \
        '_lr-' + str(args.initial_lr) + \
        '_recon-' + str(args.recon_wei)

    args.time = time

    args.check_dir = join(args.data_root_dir, 'saved_models', args.net)
    try:
        makedirs(args.check_dir)
    except FileExistsError:
        pass

    args.log_dir = join(args.data_root_dir, 'logs', args.net)
    try:
        makedirs(args.log_dir)
    except FileExistsError:
        pass

    args.tf_log_dir = join(args.log_dir, 'tf_logs')
    try:
        makedirs(args.tf_log_dir)
    except FileExistsError:
        pass

    args.output_dir = join(args.data_root_dir, 'plots', args.net)
    try:
        makedirs(args.output_dir)
    except FileExistsError:
        pass

    if args.train:
        from train import train
        # Run training
        train(args, train_list, val_list, model_list[0], net_input_shape)

    if args.test:
        from test import test
        # Run testing
        test(args, test_list, model_list, net_input_shape)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train on Medical Data')
    # Setting Data Directory
    parser.add_argument('--data_root_dir', type=str, required=True,
                        help='The root directory for your data.')
    # weights_path
    parser.add_argument('--weights_path', type=str, default='',
                        help='/path/to/trained_model.hdf5 '
                        'from root. Set to "" for none.')
    # Split Num
    parser.add_argument('--split_num', type=int, default=0,
                        help='Which training split to train/test on.')
    # Network Model
    parser.add_argument('--net', type=str.lower, default='segcapsr3',
                        choices=['segcapsr3', 'tiramisu'],
                        help='Choose your network.')
    # Training
    parser.add_argument('--train', type=int, default=1, choices=[0, 1],
                        help='Set to 1 to enable training.')
    # Test
    parser.add_argument('--test', type=int, default=1, choices=[0, 1],
                        help='Set to 1 to enable testing.')
    # Shuffle
    parser.add_argument('--shuffle_data', type=int, default=1, choices=[0, 1],
                        help='Whether or not to shuffle the training data '
                        '(both per epoch and in slice order.')
    # Data Augmentation
    parser.add_argument('--aug_data', type=int, default=1, choices=[0, 1],
                        help='Whether or not to use data '
                        'augmentation during training.')
    # Loss
    parser.add_argument('--loss', type=str.lower, default='w_bce',
                        choices=['bce', 'w_bce', 'dice', 'mar', 'w_mar'],
                        help='Which loss to use. "bce" and'
                        '"w_bce": unweighted and weighted binary cross entropy'
                        '"dice": soft dice coefficient, "mar" and "w_mar":'
                        ' unweighted and weighted margin loss.')
    # Batch Size
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for training/testing.')
    # Adam Learning rate
    parser.add_argument('--initial_lr', type=float, default=0.0001,
                        help='Initial learning rate for Adam.')
    # Reconstruction Weight
    parser.add_argument('--recon_wei', type=float, default=131.072,
                        help="If using capsnet: "
                        "The coefficient (weighting) for the loss of decoder")
    # Slices
    parser.add_argument('--slices', type=int, default=1,
                        help='Number of slices to include '
                        'for training/testing.')
    # ??
    parser.add_argument('--subsamp', type=int, default=-1,
                        help='Number of slices to skip when forming '
                        '3D samples for training. Enter -1 for random '
                        'subsampling up to 5% of total slices.')
    # STRIDE
    parser.add_argument('--stride', type=int, default=1,
                        help='Number of slices to move '
                        'when generating the next sample.')
    # Learning Notification
    parser.add_argument('--verbose', type=int, default=1, choices=[0, 1, 2],
                        help='Set the verbose value for training.'
                        ' 0: Silent, 1: per iteration, 2: per epoch.')
    # Raw Data Saving
    parser.add_argument('--save_raw', type=int, default=1, choices=[0, 1],
                        help='Enter 0 to not save, 1 to save.')
    # Segmentation Saving
    parser.add_argument('--save_seg', type=int, default=1, choices=[0, 1],
                        help='Enter 0 to not save, 1 to save.')
    # Additional Name
    parser.add_argument('--save_prefix', type=str, default='',
                        help='Prefix to append to saved CSV.')
    # Threshhold(Shikiichi)
    parser.add_argument('--thresh_level', type=float, default=0.,
                        help='Enter 0.0 for otsu thresholding, else set value')
    # Dice Loss
    parser.add_argument('--compute_dice', type=int, default=1,
                        help='0 or 1')
    # jaccard(Ruijido)
    parser.add_argument('--compute_jaccard', type=int, default=1,
                        help='0 or 1')
    # ????
    parser.add_argument('--compute_assd', type=int, default=0,
                        help='0 or 1')
    # GPU option
    parser.add_argument('--which_gpus', type=str, default="0",
                        help='Enter "-2" for CPU only,'
                        ' "-1" for all GPUs available, '
                        'or a comma separated list of '
                        'GPU id numbers ex: "0,1,4".')

    parser.add_argument('--gpus', type=int, default=-1,
                        help='Number of GPUs you have available for training. '
                             'If entering specific GPU ids under the '
                             '--which_gpus arg or if using CPU, '
                             'then this number will be inferred, '
                             'else this argument must be included.')
    # training steps
    parser.add_argument('--train_step', type=int, default=1000,
                        help='number of iteration in 1 epoch')

    parser.add_argument('--epoch_num', type=int, default=5,
                        help='number of epoch')

    parser.add_argument('--aug_option', type=int, default=0,
                        choices=[0, 1], help='augmentetation_option')

    parser.add_argument('--seg_class', type=int, default=1,
                        choices=[1, 2, 3, 4],
                        help='choose class for segmentation. '
                        '1: 0, 2: 63, 3: 127, 4: 91')
    parser.add_argument('--val_step', type=int, default=30,
                        help='validation step')

    arguments = parser.parse_args()
    # GPU Options
    if arguments.which_gpus == -2:
        environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        environ["CUDA_VISIBLE_DEVICES"] = ""
    elif arguments.which_gpus == '-1':
        assert (arguments.gpus != -1),\
            'Use all GPUs option selected under --which_gpus,'\
            'with this option the user MUST ' \
            'specify the number of GPUs available with the --gpus option.'
    else:
        arguments.gpus = len(arguments.which_gpus.split(','))
        environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        environ["CUDA_VISIBLE_DEVICES"] = str(arguments.which_gpus)

    if arguments.gpus > 1:
        assert arguments.batch_size >= arguments.gpus,\
            'Error: Must have at least as many items per batch as GPUs ' \
            'for multi-GPU training. For model parallelism instead of ' \
            'data parallelism, modifications must be made to the code.'

    main(arguments)
