from .config_node import ConfigNode

config = ConfigNode()

config.device = 'cuda'
# cuDNN
config.cudnn = ConfigNode()
config.cudnn.benchmark = True
config.cudnn.deterministic = False

config.dataset = ConfigNode()
config.dataset.name = 'CIFAR10'
config.dataset.dataset_dir = ''
config.dataset.image_size = 32
config.dataset.n_channels = 3
config.dataset.n_classes = 10

config.model = ConfigNode()
# options: 'cifar', 'imagenet'
# Use 'cifar' for small input images
config.model.type = 'cifar'
config.model.name = 'resnet_preact'
config.model.init_mode = 'kaiming_fan_out'

# hw_add: lifting scheme takes places of conv2d(vanilla and strided)
config.model.mobilenetv2_one_fusion_HL = ConfigNode()

### hw add model:
config.model.resnet = ConfigNode()
config.model.resnet.depth = 50  # for cifar type model
config.model.resnet.n_blocks = [3, 4, 6, 3]  # for imagenet type model
config.model.resnet.block_type = 'bottleneck'
config.model.resnet.initial_channels = 64

config.model.resnet_ls_fix_max = ConfigNode()
config.model.resnet_ls_fix_max.depth = 50  # for mini-imagenet type model
config.model.resnet_ls_fix_max.n_blocks = [3, 4, 6, 3]  # for imagenet type model
config.model.resnet_ls_fix_max.block_type = 'bottleneck'
config.model.resnet_ls_fix_max.initial_channels = 64

config.model.resnet_ls_fix_min = ConfigNode()
config.model.resnet_ls_fix_min.depth = 50  # for mini-imagenet type model
config.model.resnet_ls_fix_min.n_blocks = [3, 4, 6, 3]  # for imagenet type model
config.model.resnet_ls_fix_min.block_type = 'bottleneck'
config.model.resnet_ls_fix_min.initial_channels = 64

config.model.resnet_ls_learn_max = ConfigNode()
config.model.resnet_ls_learn_max.depth = 50  # for mini-imagenet type model
config.model.resnet_ls_learn_max.n_blocks = [3, 4, 6, 3]  # for imagenet type model
config.model.resnet_ls_learn_max.block_type = 'bottleneck'
config.model.resnet_ls_learn_max.initial_channels = 64

config.model.resnet_ls_learn_min = ConfigNode()
config.model.resnet_ls_learn_min.depth = 50  # for mini-imagenet type model
config.model.resnet_ls_learn_min.n_blocks = [3, 4, 6, 3]  # for imagenet type model
config.model.resnet_ls_learn_min.block_type = 'bottleneck'
config.model.resnet_ls_learn_min.initial_channels = 64

###
### hw:新增红黑小波实现：smooth、attention
config.model.resnet_ls_fix_smooth = ConfigNode()
config.model.resnet_ls_fix_smooth.depth = 50  # for mini-imagenet type model
config.model.resnet_ls_fix_smooth.n_blocks = [3, 4, 6, 3]  # for imagenet type model
config.model.resnet_ls_fix_smooth.block_type = 'bottleneck'
config.model.resnet_ls_fix_smooth.initial_channels = 64

config.model.resnet_ls_fix_attention = ConfigNode()
config.model.resnet_ls_fix_attention.depth = 50  # for mini-imagenet type model
config.model.resnet_ls_fix_attention.n_blocks = [3, 4, 6, 3]  # for imagenet type model
config.model.resnet_ls_fix_attention.block_type = 'bottleneck'
config.model.resnet_ls_fix_attention.initial_channels = 64

config.model.resnet_ls_learn_smooth = ConfigNode()
config.model.resnet_ls_learn_smooth.depth = 50  # for mini-imagenet type model
config.model.resnet_ls_learn_smooth.n_blocks = [3, 4, 6, 3]  # for imagenet type model
config.model.resnet_ls_learn_smooth.block_type = 'bottleneck'
config.model.resnet_ls_learn_smooth.initial_channels = 64

config.model.resnet_ls_learn_attention = ConfigNode()
config.model.resnet_ls_learn_attention.depth = 50  # for mini-imagenet type model
config.model.resnet_ls_learn_attention.n_blocks = [3, 4, 6, 3]  # for imagenet type model
config.model.resnet_ls_learn_attention.block_type = 'bottleneck'
config.model.resnet_ls_learn_attention.initial_channels = 64
###

### hw: 新增 红黑小波外部选择策略
config.model.resnet_one_fusion_LH = ConfigNode()
config.model.resnet_one_fusion_LH.depth = 50  # for mini-imagenet type model
config.model.resnet_one_fusion_LH.n_blocks = [3, 4, 6, 3]  # for imagenet type model
config.model.resnet_one_fusion_LH.block_type = 'bottleneck'
config.model.resnet_one_fusion_LH.initial_channels = 64

config.model.resnet_one_fusion_HL = ConfigNode()
config.model.resnet_one_fusion_HL.depth = 50  # for mini-imagenet type model
config.model.resnet_one_fusion_HL.n_blocks = [3, 4, 6, 3]  # for imagenet type model
config.model.resnet_one_fusion_HL.block_type = 'bottleneck'
config.model.resnet_one_fusion_HL.initial_channels = 64

config.model.resnet_one_fusion_HH = ConfigNode()
config.model.resnet_one_fusion_HH.depth = 50  # for mini-imagenet type model
config.model.resnet_one_fusion_HH.n_blocks = [3, 4, 6, 3]  # for imagenet type model
config.model.resnet_one_fusion_HH.block_type = 'bottleneck'
config.model.resnet_one_fusion_HH.initial_channels = 64

config.model.resnet_one_fusion_all = ConfigNode()
config.model.resnet_one_fusion_all.depth = 50  # for mini-imagenet type model
config.model.resnet_one_fusion_all.n_blocks = [3, 4, 6, 3]  # for imagenet type model
config.model.resnet_one_fusion_all.block_type = 'bottleneck'
config.model.resnet_one_fusion_all.initial_channels = 64

config.model.resnet_two_tree_all = ConfigNode()
config.model.resnet_two_tree_all.depth = 50  # for mini-imagenet type model
config.model.resnet_two_tree_all.n_blocks = [3, 4, 6, 3]  # for imagenet type model
config.model.resnet_two_tree_all.block_type = 'bottleneck'
config.model.resnet_two_tree_all.initial_channels = 64

config.model.resnet_two_tree_LH = ConfigNode()
config.model.resnet_two_tree_LH.depth = 50  # for mini-imagenet type model
config.model.resnet_two_tree_LH.n_blocks = [3, 4, 6, 3]  # for imagenet type model
config.model.resnet_two_tree_LH.block_type = 'bottleneck'
config.model.resnet_two_tree_LH.initial_channels = 64

config.model.resnet_two_tree_HL = ConfigNode()
config.model.resnet_two_tree_HL.depth = 50  # for mini-imagenet type model
config.model.resnet_two_tree_HL.n_blocks = [3, 4, 6, 3]  # for imagenet type model
config.model.resnet_two_tree_HL.block_type = 'bottleneck'
config.model.resnet_two_tree_HL.initial_channels = 64

config.model.resnet_two_tree_HH = ConfigNode()
config.model.resnet_two_tree_HH.depth = 50  # for mini-imagenet type model
config.model.resnet_two_tree_HH.n_blocks = [3, 4, 6, 3]  # for imagenet type model
config.model.resnet_two_tree_HH.block_type = 'bottleneck'
config.model.resnet_two_tree_HH.initial_channels = 64
###

### hw: 新增resnet50 comparision：各种池化对比方法
config.model.resnet_dpp = ConfigNode()
config.model.resnet_dpp.depth = 50  # for mini-imagenet type model
config.model.resnet_dpp.n_blocks = [3, 4, 6, 3]  # for imagenet type model
config.model.resnet_dpp.block_type = 'bottleneck'
config.model.resnet_dpp.initial_channels = 64

config.model.resnet_lip = ConfigNode()
config.model.resnet_lip.depth = 50  # for mini-imagenet type model
config.model.resnet_lip.n_blocks = [3, 4, 6, 3]  # for imagenet type model
config.model.resnet_lip.block_type = 'bottleneck'
config.model.resnet_lip.initial_channels = 64

config.model.resnet_maxpool = ConfigNode()
config.model.resnet_maxpool.depth = 50  # for mini-imagenet type model
config.model.resnet_maxpool.n_blocks = [3, 4, 6, 3]  # for imagenet type model
config.model.resnet_maxpool.block_type = 'bottleneck'
config.model.resnet_maxpool.initial_channels = 64

config.model.resnet_avgpool = ConfigNode()
config.model.resnet_avgpool.depth = 50  # for mini-imagenet type model
config.model.resnet_avgpool.n_blocks = [3, 4, 6, 3]  # for imagenet type model
config.model.resnet_avgpool.block_type = 'bottleneck'
config.model.resnet_avgpool.initial_channels = 64

config.model.resnet_mixpool = ConfigNode()
config.model.resnet_mixpool.depth = 50  # for mini-imagenet type model
config.model.resnet_mixpool.n_blocks = [3, 4, 6, 3]  # for imagenet type model
config.model.resnet_mixpool.block_type = 'bottleneck'
config.model.resnet_mixpool.initial_channels = 64

config.model.resnet_gatedpool = ConfigNode()
config.model.resnet_gatedpool.depth = 50  # for mini-imagenet type model
config.model.resnet_gatedpool.n_blocks = [3, 4, 6, 3]  # for imagenet type model
config.model.resnet_gatedpool.block_type = 'bottleneck'
config.model.resnet_gatedpool.initial_channels = 64

config.model.resnet_dwtpool = ConfigNode()
config.model.resnet_dwtpool.depth = 50  # for mini-imagenet type model
config.model.resnet_dwtpool.n_blocks = [3, 4, 6, 3]  # for imagenet type model
config.model.resnet_dwtpool.block_type = 'bottleneck'
config.model.resnet_dwtpool.initial_channels = 64

##
config.train = ConfigNode()
config.train.checkpoint = ''
config.train.resume = False
config.train.use_apex = True
# optimization level for NVIDIA apex
# O0 = fp32
# O1 = mixed precision
# O2 = almost fp16
# O3 = fp16
config.train.precision = 'O0'
config.train.batch_size = 128
config.train.subdivision = 1
# optimizer (options: sgd, adam, lars, adabound, adaboundw)
config.train.optimizer = 'sgd'
config.train.base_lr = 0.1
config.train.momentum = 0.9
config.train.nesterov = True
config.train.weight_decay = 1e-4
config.train.no_weight_decay_on_bn = False
config.train.gradient_clip = 0.0
config.train.start_epoch = 0
config.train.seed = 0
config.train.val_first = True
config.train.val_period = 1
config.train.val_ratio = 0.0
config.train.use_test_as_val = True

config.train.output_dir = 'experiments/exp00'
config.train.log_period = 100
config.train.checkpoint_period = 10

config.train.use_tensorboard = True
config.tensorboard = ConfigNode()
config.tensorboard.train_images = False
config.tensorboard.val_images = False
config.tensorboard.model_params = False

# optimizer
config.optim = ConfigNode()
# Adam
config.optim.adam = ConfigNode()
config.optim.adam.betas = (0.9, 0.999)
# LARS
config.optim.lars = ConfigNode()
config.optim.lars.eps = 1e-9
config.optim.lars.threshold = 1e-2
# AdaBound
config.optim.adabound = ConfigNode()
config.optim.adabound.betas = (0.9, 0.999)
config.optim.adabound.final_lr = 0.1
config.optim.adabound.gamma = 1e-3

# scheduler
config.scheduler = ConfigNode()
config.scheduler.epochs = 160
# warm up (options: none, linear, exponential)
config.scheduler.warmup = ConfigNode()
config.scheduler.warmup.type = 'none'
config.scheduler.warmup.epochs = 0
config.scheduler.warmup.start_factor = 1e-3
config.scheduler.warmup.exponent = 4
# main scheduler (options: constant, linear, multistep, cosine, sgdr)
config.scheduler.type = 'multistep'
config.scheduler.milestones = [80, 120]
config.scheduler.lr_decay = 0.1
config.scheduler.lr_min_factor = 0.001
config.scheduler.T0 = 10
config.scheduler.T_mul = 1.

# train data loader
config.train.dataloader = ConfigNode()
config.train.dataloader.num_workers = 2
config.train.dataloader.drop_last = True
config.train.dataloader.pin_memory = False
config.train.dataloader.non_blocking = False

# validation data loader
config.validation = ConfigNode()
config.validation.batch_size = 256
config.validation.dataloader = ConfigNode()
config.validation.dataloader.num_workers = 2
config.validation.dataloader.drop_last = False
config.validation.dataloader.pin_memory = False
config.validation.dataloader.non_blocking = False

# distributed
config.train.distributed = False
config.train.dist = ConfigNode()
config.train.dist.backend = 'nccl'
config.train.dist.init_method = 'env://'
config.train.dist.world_size = -1
config.train.dist.node_rank = -1
config.train.dist.local_rank = 0
config.train.dist.use_sync_bn = False

config.augmentation = ConfigNode()
config.augmentation.use_random_crop = True
config.augmentation.use_random_horizontal_flip = True
config.augmentation.use_cutout = False
config.augmentation.use_random_erasing = False
config.augmentation.use_dual_cutout = False
config.augmentation.use_mixup = False
config.augmentation.use_ricap = False
config.augmentation.use_cutmix = False
config.augmentation.use_label_smoothing = False

config.augmentation.random_crop = ConfigNode()
config.augmentation.random_crop.padding = 4
config.augmentation.random_crop.fill = 0
config.augmentation.random_crop.padding_mode = 'constant'

config.augmentation.random_horizontal_flip = ConfigNode()
config.augmentation.random_horizontal_flip.prob = 0.5

config.augmentation.cutout = ConfigNode()
config.augmentation.cutout.prob = 1.0
config.augmentation.cutout.mask_size = 16
config.augmentation.cutout.cut_inside = False
config.augmentation.cutout.mask_color = 0
config.augmentation.cutout.dual_cutout_alpha = 0.1

config.augmentation.random_erasing = ConfigNode()
config.augmentation.random_erasing.prob = 0.5
config.augmentation.random_erasing.area_ratio_range = [0.02, 0.4]
config.augmentation.random_erasing.min_aspect_ratio = 0.3
config.augmentation.random_erasing.max_attempt = 20

config.augmentation.mixup = ConfigNode()
config.augmentation.mixup.alpha = 1.0

config.augmentation.ricap = ConfigNode()
config.augmentation.ricap.beta = 0.3

config.augmentation.cutmix = ConfigNode()
config.augmentation.cutmix.alpha = 1.0

config.augmentation.label_smoothing = ConfigNode()
config.augmentation.label_smoothing.epsilon = 0.1

config.tta = ConfigNode()
config.tta.use_resize = False
config.tta.use_center_crop = False
config.tta.resize = 256

# test config
config.test = ConfigNode()
config.test.checkpoint = ''
config.test.output_dir = ''
config.test.batch_size = 256
# test data loader
config.test.dataloader = ConfigNode()
config.test.dataloader.num_workers = 2
config.test.dataloader.pin_memory = False


def get_default_config():
    return config.clone()
