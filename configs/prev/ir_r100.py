from easydict import EasyDict as edict

# make training faster
# our RAM is 256G
# mount -t tmpfs -o size=140G  tmpfs /train_tmp

config = edict()
config.margin_list = (1.0, 0.0, 0.4)
config.network = "r100"
config.resume = False
config.output = None
config.embedding_size = 512
config.sample_rate = 1.0
config.fp16 = True
config.momentum = 0.9
config.weight_decay = 1e-4
config.batch_size = 128
config.lr = 0.1
config.verbose = 2000
config.dali = False

#config.rec = "/train_tmp/glint360k"
config.rec = "/home/ir-images/trainIR"
config.num_classes = 169
config.num_image = 6637 
config.num_epoch = 20
config.warmup_epoch = 0
#config.val_targets = ['lfw', 'cfp_fp', "agedb_30"]
config.val_targets = []

config.save_all_states = True
