from easydict import EasyDict as edict

# make training faster
# our RAM is 256G
# mount -t tmpfs -o size=140G  tmpfs /train_tmp

config = edict()
#config.loss = "combined"
config.loss = "curricularface"
#config.loss = "cosface"
config.margin_list = (1.0, 0.0, 0.4)
config.network = "mbf"
config.resume = False
config.output = None
config.embedding_size = 512
#config.sample_rate = 1.0
config.sample_rate = 0.2
config.fp16 = True
config.momentum = 0.9
#config.weight_decay = 1e-4
config.weight_decay = 5e-4
#config.batch_size = 128
config.batch_size = 512
#config.lr = 0.4
config.lr = 0.1
#config.lr = 0.01
#config.verbose = 10000
config.verbose = 10912
#config.verbose = 6676
#config.verbose = 16691
config.dali = False

#config.rec = "/home/src/train_tmp/glint360k"
#config.num_classes = 360232
#config.num_image = 17091657
#config.num_epoch = 20
#config.warmup_epoch = 0
#config.val_targets = ['lfw', 'cfp_fp', "agedb_30"]

config.rec = "/train_tmp/"
config.num_classes = 2429074
config.num_image = 44695596
config.num_epoch = 20
#config.warmup_epoch = 2
config.warmup_epoch = 0
#config.val_targets = ['lfw', 'cfp_fp', 'agedb_30']
config.val_targets = ['lfw', 'cplfw', 'cfp_fp', 'agedb_30']

config.save_all_states = True
