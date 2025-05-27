import os
import sys
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

from models import build_model
from utils.validations import validate
from opts import arg_parser
from dataloaders import build_dataset
from utils.build_cfg import setup_cfg
from dassl.optim import build_lr_scheduler
from utils.trainers import train_coop
from utils.helper import save_checkpoint

# --------------- INICIO DE LAS LÍNEAS AGREGADAS PARA WANDB ---------------
import wandb
# --------------- FIN DE LAS LÍNEAS AGREGADAS PARA WANDB ------------------

def main():
    global args
    parser = arg_parser()
    args = parser.parse_args()
    cfg = setup_cfg(args)

    # building the train and val dataloaders
    train_split = cfg.DATASET.TRAIN_SPLIT
    val_split = cfg.DATASET.VAL_SPLIT
    test_split = cfg.DATASET.TEST_SPLIT

    train_dataset = build_dataset(cfg, train_split)
    val_dataset = build_dataset(cfg, val_split)
    test_dataset = build_dataset(cfg, test_split)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
        shuffle=cfg.DATALOADER.TRAIN_X.SHUFFLE,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg.DATALOADER.VAL.BATCH_SIZE,
        shuffle=cfg.DATALOADER.VAL.SHUFFLE,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        pin_memory=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
        shuffle=cfg.DATALOADER.TEST.SHUFFLE,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        pin_memory=True
    )

    classnames = val_dataset.classnames

    # build the model
    model, arch_name = build_model(cfg, args, classnames)

    # build the optimizer and lr_scheduler
    try:
        prompt_params = model.prompt_params()
    except:
        prompt_params = model.module.prompt_params()

    prompt_group = {'params': prompt_params}
    print('num of params in prompt learner: ', len(prompt_params))

    sgd_polices = [prompt_group]

    if cfg.TRAINER.FINETUNE_BACKBONE:
        try:
            backbone_params = model.backbone_params()
        except:
            backbone_params = model.module.backbone_params()

        print('num of params in backbone: ', len(backbone_params))

        base_group = {
            'params': backbone_params,
            'lr': cfg.OPTIM.LR * cfg.OPTIM.BACKBONE_LR_MULT
        }
        sgd_polices.append(base_group)

    if cfg.TRAINER.FINETUNE_ATTN:
        try:
            attn_params = model.attn_params()
        except:
            attn_params = model.module.attn_params()

        print('num of params in attn layer: ', len(attn_params))

        attn_group = {
            'params': attn_params,
            'lr': cfg.OPTIM.LR * cfg.OPTIM.ATTN_LR_MULT
        }
        sgd_polices.append(attn_group)

    
    # ============ [Daniel] ============
    try:
        dish_fc_params = model.fc_dishes.parameters()
    except:
        dish_fc_params = model.module.fc_dishes.parameters()

    dish_group = {
        'params': dish_fc_params,
        'lr': cfg.OPTIM.LR * 10
    }
    sgd_polices.append(dish_group)
    # =======================================        
    
    optim = torch.optim.SGD(
        sgd_polices,
        lr=cfg.OPTIM.LR,
        momentum=cfg.OPTIM.MOMENTUM,
        weight_decay=cfg.OPTIM.WEIGHT_DECAY,
        dampening=cfg.OPTIM.SGD_DAMPNING,
        nesterov=cfg.OPTIM.SGD_NESTEROV
    )

    sched = build_lr_scheduler(optim, cfg.OPTIM)

    log_folder = os.path.join(cfg.OUTPUT_DIR, arch_name)
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)

    logfile_path = os.path.join(log_folder, 'log.log')
    if os.path.exists(logfile_path):
        logfile = open(logfile_path, 'a')
    else:
        logfile = open(logfile_path, 'w')

    command = " ".join(sys.argv)
    print(command, flush=True)
    print(args, flush=True)
    print(cfg, flush=True)

    print(command, file=logfile, flush=True)
    print(args, file=logfile, flush=True)
    print(cfg, file=logfile, flush=True)

    if not args.auto_resume:
        print(model, file=logfile, flush=True)

    if args.auto_resume:
        args.resume = os.path.join(log_folder, 'checkpoint.pth.tar')

    best_mAP = 0
    args.start_epoch = 0

    if args.resume is not None:
        if os.path.exists(args.resume):
            print('... loading pretrained weights from %s' % args.resume)
            print('... loading pretrained weights from %s' % args.resume, file=logfile, flush=True)

            checkpoint = torch.load(args.resume, map_location='cpu')
            args.start_epoch = checkpoint['epoch']
            best_mAP = checkpoint['best_mAP']
            model.load_state_dict(checkpoint['state_dict'])
            optim.load_state_dict(checkpoint['optimizer'])

    # --------------- INICIO DE LAS LINEAS AGREGADAS PARA WANDB ---------------
    # Para iniciar sesion en W&B con tu clave:
    wandb.login(key="858f209197a8e3da7829200a7b517cbabf2ff86b")

    # Inicializamos el proyecto en wandb
    wandb.init(project="mlrgcn-e150_mafood_0.5_0.5_cutout_0.1_MNM", name=arch_name)

    # Opcional: puedes subir la configuracion para tenerla disponible en W&B
    wandb.config.update(cfg)
    # --------------- FIN DE LAS LINEAS AGREGADAS PARA WANDB ------------------

    for epoch in range(args.start_epoch, cfg.OPTIM.MAX_EPOCH):
        # NUEVO ➔ train_coop ahora devuelve dish_acc_meter tambien
        batch_time, losses, mAP_batches, dish_acc_meter, ing_loss_meter, dish_loss_meter = train_coop(
            train_loader,
            [val_loader],
            model,
            optim,
            sched,
            args,
            cfg,
            epoch
        )

        print('Train: [{}/{}]\t'
              'Time {:.3f}\t'
              'Loss {:.2f}\t'
              'Ing_Loss {:.2f}\t'
              'Dish_Loss {:.2f}\t'
              'mAP {:.2f}\t'
              'Dish_Acc {:.2f}'.format(
            epoch + 1,
            cfg.OPTIM.MAX_EPOCH,
            batch_time.avg,
            losses.avg,
            ing_loss_meter.avg,
            dish_loss_meter.avg,
            mAP_batches.avg,
            dish_acc_meter.avg
        ), flush=True)

        print('Train: [{}/{}]\t'
              'Time {:.3f}\t'
              'Loss {:.2f}\t'
              'Ing_Loss {:.2f}\t'
              'Dish_Loss {:.2f}\t'
              'mAP {:.2f}\t'
              'Dish_Acc {:.2f}'.format(
            epoch + 1,
            cfg.OPTIM.MAX_EPOCH,
            batch_time.avg,
            losses.avg,
            ing_loss_meter.avg,
            dish_loss_meter.avg,
            mAP_batches.avg,
            dish_acc_meter.avg
        ), file=logfile, flush=True)

        # --------------- INICIO DE LAS LINEAS AGREGADAS PARA WANDB ---------------
        # Logueamos metricas de entrenamiento
        wandb.log({
            "train/loss": losses.avg,
            "train/ing_loss_meter": ing_loss_meter.avg,
            "train/dish_loss_meter": dish_loss_meter.avg,
            "train/mAP": mAP_batches.avg,
            "train/acc": dish_acc_meter.avg,
            "epoch": epoch + 1
        })
        # --------------- FIN DE LAS LINEAS AGREGADAS PARA WANDB ------------------

        if (epoch + 1) % args.val_every_n_epochs == 0 or epoch == args.stop_epochs - 1:
            # NUEVO ➔ validate ahora devuelve dish_acc
            p_c, r_c, f_c, p_o, r_o, f_o, j_c, j_o, mAP_score, dish_acc = validate(val_loader, model, args)

            print('test: [{}/{}]\t '
                  'P_C {:.2f} \t R_C {:.2f} \t F_C {:.2f} \t'
                  'P_O {:.2f} \t R_O {:.2f} \t F_O {:.2f} \t'
                  'J_C {:.2f} \t J_O {:.2f} \t' 
                  'mAP {:.2f} \t Dish_Acc {:.2f}'
                  .format(
                epoch + 1,
                cfg.OPTIM.MAX_EPOCH,
                p_c, r_c, f_c,
                p_o, r_o, f_o,
                j_c, j_o,
                mAP_score, dish_acc
            ), flush=True)

            print('test: [{}/{}]\t '
                  'P_C {:.2f} \t R_C {:.2f} \t F_C {:.2f} \t'
                  'P_O {:.2f} \t R_O {:.2f} \t F_O {:.2f} \t'
                  'J_C {:.2f} \t J_O {:.2f} \t'
                  'mAP {:.2f} \t Dish_Acc {:.2f}'
                  .format(
                epoch + 1,
                cfg.OPTIM.MAX_EPOCH,
                p_c, r_c, f_c,
                p_o, r_o, f_o,
                j_c, j_o,
                mAP_score, dish_acc
            ), file=logfile, flush=True)

            # --------------- INICIO DE LAS LÃNEAS AGREGADAS PARA WANDB ---------------
            # Logueamos metricas de validacion
            wandb.log({
                "val/P_C": p_c,
                "val/R_C": r_c,
                "val/F_C": f_c,
                "val/J_C": j_c,
                "val/P_O": p_o,
                "val/R_O": r_o,
                "val/F_O": f_o,
                "val/J_O": j_o,
                "val/mAP": mAP_score,
                "val/Acc": dish_acc,
                "epoch": epoch + 1
            })
            # --------------- FIN DE LAS LÃNEAS AGREGADAS PARA WANDB ------------------

            is_best = mAP_score > best_mAP
            if is_best:
                best_mAP = mAP_score

            save_dict = {
                'epoch': epoch + 1,
                'arch': arch_name,
                'state_dict': model.state_dict(),
                'best_mAP': best_mAP,
                'optimizer': optim.state_dict(),
                'scheduler': sched.state_dict()
            }

            save_checkpoint(save_dict, is_best, log_folder)

    print('Evaluating the best model', flush=True)
    print('Evaluating the best model', file=logfile, flush=True)

    print('Evaluate with threshold %.2f' % args.thre, flush=True)
    print('Evaluate with threshold %.2f' % args.thre, file=logfile, flush=True)

    best_checkpoints = os.path.join(log_folder, 'model_best.pth.tar')
    print('... loading pretrained weights from %s' % best_checkpoints, flush=True)
    print('... loading pretrained weights from %s' % best_checkpoints, file=logfile, flush=True)

    checkpoint = torch.load(best_checkpoints, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    best_epoch = checkpoint['epoch']

    # NUEVO ➔ validate ahora devuelve dish_acc también
    p_c, r_c, f_c, p_o, r_o, f_o, j_c, j_o, mAP_score, dish_acc = validate(test_loader, model, args)

    print('Test: [{}/{}]\t '
          'P_C {:.2f} \t R_C {:.2f} \t F_C {:.2f} '
          'P_O {:.2f} \t R_O {:.2f} \t F_O {:.2f} '
          'J_C {:.2f} \t J_O {:.2f} \t'
          'mAP {:.2f} \t Dish_Acc {:.2f}'.format(
        best_epoch,
        cfg.OPTIM.MAX_EPOCH,
        p_c, r_c, f_c,
        p_o, r_o, f_o,
        j_c, j_o,
        mAP_score, dish_acc
    ), flush=True)

    print('Test: [{}/{}]\t '
          'P_C {:.2f} \t R_C {:.2f} \t F_C {:.2f} '
          'P_O {:.2f} \t R_O {:.2f} \t F_O {:.2f} '
          'J_C {:.2f} \t J_O {:.2f} \t'
          'mAP {:.2f} \t Dish_Acc {:.2f}'.format(
        best_epoch,
        cfg.OPTIM.MAX_EPOCH,
        p_c, r_c, f_c,
        p_o, r_o, f_o,
        j_c, j_o,
        mAP_score, dish_acc
    ), file=logfile, flush=True)

    # --------------- INICIO DE LAS LINEAS AGREGADAS PARA WANDB ---------------
    # Log final de metricas del test (despues de cargar el mejor modelo)
    wandb.log({
        "test/P_C": p_c,
        "test/R_C": r_c,
        "test/F_C": f_c,
        "test/J_C": j_c,
        "test/P_O": p_o,
        "test/R_O": r_o,
        "test/F_O": f_o,
        "test/J_O": j_o,
        "test/mAP": mAP_score,
        "test/Acc": dish_acc,
        "best_epoch": best_epoch
    })
    # --------------- FIN DE LAS LINEAS AGREGADAS PARA WANDB ------------------

if __name__ == '__main__':
    main()