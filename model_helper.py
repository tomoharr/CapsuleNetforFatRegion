import tensorflow as tf


def create_model(args, input_shape):
    # If using CPU or single GPU
    if args.gpus <= 1:
        if args.net == 'tiramisu':
            from densenets import DenseNetFCN
            model = DenseNetFCN(input_shape)
            return [model]
        elif args.net == 'segcapsr3':
            from capsnet import CapsNetR3
            model_list = CapsNetR3(input_shape)
            return model_list
        else:
            raise Exception('Unknown network type specified: {}'.format(args.net))
    # If using multiple GPUs
    else:
        with tf.device("/cpu:0"):
            if args.net == 'tiramisu':
                from densenets import DenseNetFCN
                model = DenseNetFCN(input_shape)
                return [model]
            elif args.net == 'segcapsr3':
                from capsnet import CapsNetR3
                model_list = CapsNetR3(input_shape)
                return model_list
            else:
                raise Exception('Unknown network type specified: {}'.format(args.net))
