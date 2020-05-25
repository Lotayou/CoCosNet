INF = 99999999

class BaseOptions(object):
    # Data options
    dataroot='datasets/fashion'
    dataset_mode='fashion'
    name='fashion_cocosnet'
    checkpoints_dir='checkpoints'
    results_dir='results'
    num_workers=0
    batch_size=1
    serial_batches=False
    max_dataset_size=INF
    gpu_ids = [2]

    # Model options
    image_size=256
    padding=40   # For deep fashion dataset, the input image maybe cropped
    model='cocos'
    ncA=3
    ncB=3
    seg_dim=3
    ngf=16
    ndf=16
    numD=2
    nd_layers=3

    # Training options
    niter=30
    niter_decay=20
    epoch_count=0
    continue_train=False
    which_epoch='latest'

    # Logging options
    verbose=True
    print_every=10
    visual_every=1000
    save_every=5


class TrainOptions(BaseOptions):
    phase='train'
    isTrain=True

    # Training Options
    lr=0.0002
    beta1=0.5
    gan_mode='hinge'
    lr_policy='linear'
    init_type='xavier'
    init_gain=0.02

    lambda_perc = 1.0
    lambda_domain = 5.0
    lambda_feat = 10.0
    lambda_context = 10.0
    lambda_reg = 1.0
    lambda_adv = 1.0

    # To resume training, uncomment the following lines
    # continue_train=True
    # which_epoch='latest'  # or a certain number (e.g. '10' or '20200525-112233')

class DebugOptions(TrainOptions):
    max_dataset_size=4
    num_workers=0
    print_every=1
    visual_every=1
    save_every=1
    niter=2
    niter_decay=1
    verbose=False

class TestOptions(BaseOptions):
    phase='test'
    isTrain=False
    serial_batches=True
    num_workers=0
    batch_size=1
    which_epoch='latest'
