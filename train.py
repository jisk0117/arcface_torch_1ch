import argparse
import logging
import os

import numpy as np
import torch
from torch import distributed
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from backbones import get_model
from dataset import get_dataloader
#from losses import CombinedMarginLoss
from losses import CombinedMarginLoss, CurricularFace, CosFace
from lr_scheduler import PolyScheduler
from partial_fc import PartialFC, PartialFCAdamW
from utils.utils_callbacks import CallBackLogging, CallBackVerification
from utils.utils_config import get_config, get_config_rank
from utils.utils_logging import AverageMeter, init_logging
from utils.utils_distributed_sampler import setup_seed

# added by skji @ 2022-10-11 for RegNet
from backbones_anynet import get_model as get_model_anynet
from backbones_anynet.config_params import ConfigParams

assert torch.__version__ >= "1.9.0", "In order to enjoy the features of the new torch, \
we have upgraded the torch to 1.9.0. torch before than 1.9.0 may not work in the future."

try:
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    distributed.init_process_group("nccl")
except KeyError:
    world_size = 1
    rank = 0
    distributed.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:12584",
        rank=rank,
        world_size=world_size,
    )

# added by skji @ 2022-10-11 for ID using time =====================================================
import time
ID = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
# ==================================================================================================


def main(args):

    # get config
    cfg = get_config(args.config)
    # global control random seed
    setup_seed(seed=cfg.seed, cuda_deterministic=False)

    # added by skji @ 2022-10-11 for ID ============================================================
    #if 'anynet' in cfg.network and args.output is not None:
    if args.output is not None:
        cfg.output = args.output

    cfg.resume_rank = False
    if 'None' not in args.config_rank:
        cfg.resume_rank = True
        cfg.config_rank = get_config_rank(args.config_rank)

    cfg.resume = False
    if 'None' not in  args.config_resume:
        cfg.resume = True
        #cfg.config_resume = os.path.join(args.config_resume, "model.pt")
        cfg.config_resume = args.config_resume

    # ==============================================================================================

    torch.cuda.set_device(args.local_rank)

    os.makedirs(cfg.output, exist_ok=True)
    init_logging(rank, cfg.output, ID)

    # added by skji @ 2022-10-19 // for showing all arguments in args for log ======================
    logging.info("="*100)
    logging.info(": Arguments in command line")
    for key, value in vars(args).items():
        num_space = 25 - len(key)
        logging.info(": " + key + " " * num_space + str(value))
    logging.info("="*100)
    # ==============================================================================================


    
    summary_writer = (
        #SummaryWriter(log_dir=os.path.join(cfg.output, "tensorboard"))
        SummaryWriter(log_dir=os.path.join("work_dirs", "tensorboard", os.path.basename(cfg.output)))
        if rank == 0
        else None
    )

    train_loader = get_dataloader(
        cfg.rec,
        args.local_rank,
        cfg.batch_size,
        cfg.dali,
        cfg.seed,
        cfg.num_workers
    )

    # added by skji @ 2022-10-11 for RegNet ========================================================
    if 'anynet' in cfg.network and 'None' not in args.params:
        anynet_params = ConfigParams(path=args.params)
        param_idx = os.path.basename(args.params).split('.')[0]

        #anynet_params.se_ratio = None  # added by skji for NAS @ 2023-01-12 <==================================================== should be deleted later
    else:
        param_idx = '0000'
    # ==============================================================================================

    # added by skji @ 2022-10-11 for RegNet ========================================================
    if 'anynet' in cfg.network:
        backbone = get_model_anynet(anynet_params, mode=cfg.anynet_type).cuda()
        
        logging.info("="*100)
        logging.info(": Parameters for " + cfg.network + " " + str(param_idx))
        for key, value in anynet_params.items():
            num_space = 25 - len(key)
            logging.info(": " + key + " "*num_space + str(value))
        logging.info("="*100)
    else:
        backbone = get_model(
            cfg.network, dropout=0.0, fp16=cfg.fp16, num_features=cfg.embedding_size).cuda()
    # ==============================================================================================

    backbone = torch.nn.parallel.DistributedDataParallel(
        module=backbone, broadcast_buffers=False, device_ids=[args.local_rank], bucket_cap_mb=16,
        find_unused_parameters=True)

    backbone.train()
    # FIXME using gradient checkpoint if there are some unused parameters will cause error
    backbone._set_static_graph()

    # added by skji @ 2022-09-26 // for adding loss function =======================================
    if cfg.loss == "combined":
        margin_loss = CombinedMarginLoss(
            64,
            cfg.margin_list[0],
            cfg.margin_list[1],
            cfg.margin_list[2],
            cfg.interclass_filtering_threshold
        )
    elif cfg.loss == "curricularface":
        margin_loss = CurricularFace()
    elif cfg.loss == "cosface":
        margin_loss = CosFace()
    else:
        raise Exception("Wrong loss configuration")
    # ==============================================================================================

    if cfg.optimizer == "sgd":
        module_partial_fc = PartialFC(
            margin_loss, cfg.embedding_size, cfg.num_classes,
            cfg.sample_rate, cfg.fp16)
        module_partial_fc.train().cuda()
        # TODO the params of partial fc must be last in the params list
        opt = torch.optim.SGD(
            params=[{"params": backbone.parameters()}, {"params": module_partial_fc.parameters()}],
            lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay)

    elif cfg.optimizer == "adamw":
        module_partial_fc = PartialFCAdamW(
            margin_loss, cfg.embedding_size, cfg.num_classes,
            cfg.sample_rate, cfg.fp16)
        module_partial_fc.train().cuda()
        opt = torch.optim.AdamW(
            params=[{"params": backbone.parameters()}, {"params": module_partial_fc.parameters()}],
            lr=cfg.lr, weight_decay=cfg.weight_decay)
    else:
        raise

    cfg.total_batch_size = cfg.batch_size * world_size
    cfg.warmup_step = cfg.num_image // cfg.total_batch_size * cfg.warmup_epoch
    cfg.total_step = cfg.num_image // cfg.total_batch_size * cfg.num_epoch

    lr_scheduler = PolyScheduler(
        optimizer=opt,
        base_lr=cfg.lr,
        max_steps=cfg.total_step,
        warmup_steps=cfg.warmup_step,
        last_epoch=-1
    )

    start_epoch = 0
    global_step = 0
    # modified by skji @ 2022-10-17 // for resuming backbone  ======================================
    logging.info("="*100)
    key = "resume backbone=" + str(cfg.resume)
    num_space = 25-len(key)
    if cfg.resume:
        #dict_checkpoint = torch.load(os.path.join(cfg.output, f"checkpoint_gpu_{rank}.pt"))
        #start_epoch = dict_checkpoint["epoch"]
        #global_step = dict_checkpoint["global_step"]
        #backbone.module.load_state_dict(dict_checkpoint["state_dict_backbone"])
        #module_partial_fc.load_state_dict(dict_checkpoint["state_dict_softmax_fc"])
        #opt.load_state_dict(dict_checkpoint["state_optimizer"])
        #lr_scheduler.load_state_dict(dict_checkpoint["state_lr_scheduler"])
        dict_checkpoint = torch.load(os.path.join(cfg.config_resume, "model.pt"))
        backbone.module.load_state_dict(dict_checkpoint)
        del dict_checkpoint

        logging.info(": " + key + " " * num_space + "[SUCCESS] " + cfg.config_resume)
    else:
        logging.info(": " + key + " " * num_space + "[FAIL]    ")
    
    # added by skji @ 2022-10-17 // for transfer learning using classifiers ========================
    key = "resume classifier=" + str(cfg.resume_rank)
    num_space = 25-len(key)
    if cfg.resume_rank:
        dict_checkpoint = torch.load(os.path.join(cfg.config_rank, f"checkpoint_gpu_{rank}.pt"))
        module_partial_fc.load_state_dict(dict_checkpoint["state_dict_softmax_fc"])
        del dict_checkpoint

        logging.info(": " + key + " " * num_space + "[SUCCESS] " + args.config_rank + " : " + cfg.config_rank)
    else:
        logging.info(": " + key + " " * num_space + "[FAIL]    " + args.config_rank)
    logging.info("="*100) 
    # ==============================================================================================

    for key, value in cfg.items():
        num_space = 25 - len(key)
        logging.info(": " + key + " " * num_space + str(value))

    callback_verification = CallBackVerification(
        val_targets=cfg.val_targets, rec_prefix=cfg.rec, summary_writer=summary_writer
    )
    callback_logging = CallBackLogging(
        frequent=cfg.frequent,
        total_step=cfg.total_step,
        batch_size=cfg.batch_size,
        start_step = global_step,
        writer=summary_writer
    )

    loss_am = AverageMeter()
    amp = torch.cuda.amp.grad_scaler.GradScaler(growth_interval=100)

    for epoch in range(start_epoch, cfg.num_epoch):
    #for epoch in range(start_epoch, 5):

        if isinstance(train_loader, DataLoader):
            train_loader.sampler.set_epoch(epoch)
        for _, (img, local_labels) in enumerate(train_loader):
            global_step += 1
            local_embeddings = backbone(img)
            loss: torch.Tensor = module_partial_fc(local_embeddings, local_labels, opt)

            if cfg.fp16:
                amp.scale(loss).backward()
                amp.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(backbone.parameters(), 5)
                amp.step(opt)
                amp.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(backbone.parameters(), 5)
                opt.step()

            opt.zero_grad()
            lr_scheduler.step()

            with torch.no_grad():
                loss_am.update(loss.item(), 1)
                callback_logging(global_step, loss_am, epoch, cfg.fp16, lr_scheduler.get_last_lr()[0], amp)

                if global_step % cfg.verbose == 0 and global_step > 0:
                    callback_verification(global_step, backbone)

        if cfg.save_all_states:
            checkpoint = {
                "epoch": epoch + 1,
                "global_step": global_step,
                "state_dict_backbone": backbone.module.state_dict(),
                "state_dict_softmax_fc": module_partial_fc.state_dict(),
                "state_optimizer": opt.state_dict(),
                "state_lr_scheduler": lr_scheduler.state_dict()
            }
            torch.save(checkpoint, os.path.join(cfg.output, f"checkpoint_gpu_{rank}.pt"))

        if rank == 0:
            path_module = os.path.join(cfg.output, "model_{:02d}.pt".format(epoch))
            torch.save(backbone.module.state_dict(), path_module)

        if cfg.dali:
            train_loader.reset()

    if rank == 0:
        path_module = os.path.join(cfg.output, "model.pt")
        torch.save(backbone.module.state_dict(), path_module)

        from torch2onnx import convert_onnx
        #convert_onnx(backbone.module.cpu().eval(), path_module, os.path.join(cfg.output, "model.onnx"))
        convert_onnx(backbone.module.cpu().eval(), path_module, os.path.join(cfg.output, "model.onnx"), simplify=True) # added by skji @ 2022-09-26 // for simplify

    distributed.destroy_process_group()


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser(
        description="Distributed Arcface Training in Pytorch")
    parser.add_argument("config", type=str, help="py config file")
    parser.add_argument("--local_rank", type=int, default=0, help="local_rank")

    # added by skji @ 2022-10-11 for RegNet ========================================================
    parser.add_argument("--params", type=str, default='None', help="the path of a parameter for RegNet/AnyNet")
    parser.add_argument("--output", type=str, default=None, help="the path of an output folder")
    parser.add_argument("--config_rank", type=str, default='None', help="the path of an output folder")
    parser.add_argument("--config_resume", type=str, default='None', help="the path of an output folder")
    # ==============================================================================================

    main(parser.parse_args())

