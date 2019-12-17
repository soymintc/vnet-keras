if __name__ == '__main__':
    import glob
    import os
    import numpy as np
    import argparse
    import time, re
    import tensorflow as tf
    from keras.optimizers import Adam, SGD
    from utils import DataGenerator, dice_loss, dice_coefficient, ModelAndWeightsCheckpoint
    from vnet3d import VNet
    from keras.callbacks import LearningRateScheduler, Callback, TensorBoard, EarlyStopping
    
    parser = argparse.ArgumentParser(description="Script to run UNet3D")
    parser.add_argument('--core_tag', '-ct', required=True)
    parser.add_argument('--nii_dir', '-I', required=True)
    parser.add_argument('--batch_size', '-bs', required=True, type=int)
    parser.add_argument('--image_size', '-is', required=True, type=int)
    parser.add_argument('--learning_rate', '-lr', required=True, type=float)
    parser.add_argument('--group_size', '-gs', required=True, type=int)
    parser.add_argument('--f_root', '-fr', required=True, type=int)
    parser.add_argument('--n_validation', required=True, type=int)
    parser.add_argument('--n_test', required=True, type=int)
    parser.add_argument('--optimizer', '-op', required=True, default='adam')
    parser.add_argument('--print_summary_only', action='store_true')
    parser.set_defaults(print_summary_only=False)

    args = parser.parse_args()
    if args.optimizer == 'adam':
        args.learning_rate /= 20 # reduce lr for adam
    elif args.optimizer == 'sgd':
        pass
    else:
        raise Exception('[ERROR] optimizer = {}'.format(args.optimizer))

    # Cloud settings
    home_dir = os.path.expanduser("~")
    hostname = os.uname()[1]
    cloud_dir = '{}/gdrive/cloud/{}'.format(home_dir, hostname)
    try:
        os.system('mkdir -p ' + cloud_dir)

    # Get data
    # [IDs] Get sample IDs from src_dir
    src_dir = args.nii_dir #'data/data_with_augmentation/'
    assert os.path.exists(src_dir), "[ERROR] {} does not exist".format(src_dir)
    fpaths = glob.glob(src_dir + '/*.nii.gz')
    sids = sorted(set([os.path.split(x)[-1].rsplit('_', 1)[0] for x in fpaths]))
    
    seed = 0
    np.random.seed(seed)
    shuffle = True
    if shuffle:
        np.random.shuffle(sids)
    
    # Modules
    def lr_schedule_wrapper(learning_rate):
        learning_rate = learning_rate
        def lr_schedule(epoch):
            #learning_rate = 1e-4
            if epoch > 10:
                learning_rate /= 2
            if epoch > 20:
                learning_rate /= 2
            if epoch > 50:
                learning_rate /= 2
            tf.summary.scalar('learning_rate', learning_rate)
            #tf.compat.v1.summary.scalar('learning_rate', learning_rate)
            return learning_rate
        return lr_schedule
    
    # Set params and callbacks
    n_val, n_test = args.n_validation, args.n_test
    n_train = len(sids) - n_val - n_test
    if n_train < 0:
        raise Exception("n_train({}) < n_validation({})+n_test({})".format(n_train, n_val, n_test))
    elif n_train < n_val + n_test:
        raise Exception("n_train({}) <  n_validation({})+n_test({})".format(n_train, n_val, n_test))

    train_ids = sids[:n_train]
    valid_ids = sids[n_train : n_train+n_val]
    test_ids = sids[n_train+n_val : n_train+n_val+n_test]
    print("IDs", len(sids), len(train_ids), len(valid_ids), len(test_ids), n_train)
    epochs = 100
    h5_dir = os.path.join(cloud_dir, 'models')
    if not os.path.exists(h5_dir):
        os.system('mkdir {}'.format(h5_dir))
    prefix = os.path.join(h5_dir, args.core_tag + 
        "_b{}".format(args.batch_size))
        #"_s{}_b{}".format(args.image_size, args.batch_size))
    pattern = re.compile(prefix + '_vl([\d\.-]+)')
    existing_models = glob.glob(prefix + '_vl*.h5')
    existing_models.sort(key = lambda x: float(pattern.search(x).groups()[0][:-1]))
    
    model_weights = os.path.join(h5_dir, args.core_tag + '.h5')
    model_architecture = os.path.join(h5_dir, args.core_tag + '.json')
    checkpoint_cb = ModelAndWeightsCheckpoint(model_weights, model_architecture, 
        monitor='val_dice_coefficient', verbose=1, save_best_only=True, mode='max')
    lr_cb = LearningRateScheduler(lr_schedule_wrapper(args.learning_rate))
    earlystopping_cb = EarlyStopping(monitor='val_dice_coefficient', min_delta=0.001, 
        patience=15, verbose=1, mode='max', baseline=None, 
        restore_best_weights=True)
    time_tag = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    tf_log_dir = '{}/{}/logs/vnet'.format(cloud_dir, hostname)
    try:
        os.system('mkdir -p ' + tf_log_dir)
    if not os.path.exists(tf_log_dir):
        raise Exception("{} does not exist".format(tf_log_dir))
    log_dir = os.path.join(tf_log_dir, args.core_tag + '_' + time_tag)
    tensorboard_cb = TensorBoard(log_dir=log_dir)
    
    callbacks_list = [checkpoint_cb, 
                      # lr_cb,
                      earlystopping_cb,
                      tensorboard_cb]
    
    # Generate data
    image_shape = (args.image_size,)*3
    #FAIL: (144,144,144) #(160,160,144) #(192,192,144) #(208,208,144) #(240,240,144) 
    gen_factor = 3
    train_gen = DataGenerator(train_ids, src_dir, n_samples=n_train*gen_factor,
        rotation_range=0.4,
        batch_size=args.batch_size, image_shape=image_shape)
    valid_gen = DataGenerator(valid_ids, src_dir, n_samples=n_val*gen_factor,
        rotation_range=0.4,
        batch_size=args.batch_size, image_shape=image_shape)
    test_gen = DataGenerator(test_ids, src_dir, n_samples=n_test*gen_factor,
        rotation_range=0.4,
        batch_size=args.batch_size, image_shape=image_shape)
    train_steps = len(train_ids*gen_factor) // args.batch_size
    valid_steps = len(valid_ids*gen_factor) // args.batch_size
    
    # [V-Net 3D] # Fix #6: n_in=2 --> 4
    model = VNet(image_shape=image_shape, n_in=4, n_out=3, 
        strides=1, padding='same', kernel_size=5,
        groups=args.group_size, data_format='channels_first',
        filters=args.f_root)
    if args.optimizer == 'adam':
        optimizer = Adam(lr = args.learning_rate) # FIX #2
    elif args.optimizer == 'sgd':
        optimizer = SGD(lr=args.learning_rate, decay=1e-6, momentum=0.99)
    else:
        raise Exception('[ERROR] args.optimizer = {}'.format(args.optimizer))
    
    if len(existing_models) > 0: # if saved model exists
        print(existing_models)
        best_model = existing_models[0] # sorted ix 0 has lowest vl
    #    model.load_weights(best_model)
        print(best_model)
    model.compile(optimizer=optimizer, loss=dice_loss, metrics=[dice_coefficient])

    if args.print_summary_only:
        model.summary(line_length=150)
        raise Exception("args.print_summary_only = True")
    
    
    # Run model
    history = model.fit_generator(train_gen, 
                                  validation_data=valid_gen, 
                                  steps_per_epoch=train_steps,
                                  validation_steps=valid_steps,
                                  verbose=1,
                                  callbacks = callbacks_list,
                                  epochs=epochs)
    
