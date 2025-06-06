import sys
sys.path.insert(0, '../')
import torch
import time
from utils.helper import AverageMeter, mAP, calc_F1
from torch.cuda.amp import autocast
from pdb import set_trace as breakpoint


def validate(data_loader, model, args):
    batch_time = AverageMeter()
    prec = AverageMeter()
    rec = AverageMeter()
    dish_acc_meter = AverageMeter()  # NUEVO ➔ Accuracy de platos

    Softmax = torch.nn.Softmax(dim=1)
    Sig = torch.nn.Sigmoid()

    model.eval()

    tp, fp, fn, tn, count = 0, 0, 0, 0, 0
    preds = []
    targets = []

    with torch.no_grad():
        end = time.time()
        
        # Acumuladores para la métrica de platos
        total_dishes = 0
        correct_dishes = 0

        for i, (images, target_ingredients, target_dishes) in enumerate(data_loader):
            target_ingredients = target_ingredients.max(dim=1)[0]
            
            if torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")

            images = images.to(device)
            target_ingredients = target_ingredients.to(device)
            target_dishes = target_dishes.to(device)

            # compute output
            with autocast():
                ingredient_logits, dish_logits = model(images)

            # NUEVO ➔ Cálculo de Accuracy para platos
            _, dish_preds = torch.max(dish_logits, dim=1)
            dish_correct = (dish_preds == target_dishes).sum().item()
            dish_acc = 100.0 * dish_correct / images.size(0)

            # Actualizar acumuladores
            correct_dishes += dish_correct
            total_dishes += images.size(0)
            dish_acc_meter.update(dish_acc, images.size(0))

            # Procesamiento de los ingredientes (multi-label)
            if ingredient_logits.dim() == 3:
                output = Softmax(ingredient_logits).cpu()[:, 1]
            else:
                output = Sig(ingredient_logits).cpu()

            preds.append(output.cpu())
            targets.append(target_ingredients.cpu())

            pred = output.data.gt(args.thre).long()

            tp += (pred + target_ingredients.cpu()).eq(2).sum(dim=0)
            fp += (pred - target_ingredients.cpu()).eq(1).sum(dim=0)
            fn += (pred - target_ingredients.cpu()).eq(-1).sum(dim=0)
            tn += (pred + target_ingredients.cpu()).eq(0).sum(dim=0)
            count += images.size(0)

            this_tp = (pred + target_ingredients.cpu()).eq(2).sum()
            this_fp = (pred - target_ingredients.cpu()).eq(1).sum()
            this_fn = (pred - target_ingredients.cpu()).eq(-1).sum()

            this_prec = this_tp.float() / (this_tp + this_fp).float() * 100.0 if this_tp + this_fp != 0 else 0.0
            this_rec = this_tp.float() / (this_tp + this_fn).float() * 100.0 if this_tp + this_fn != 0 else 0.0

            prec.update(float(this_prec), images.size(0))
            rec.update(float(this_rec), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            p_c = [float(tp[i].float() / (tp[i] + fp[i]).float()) * 100.0 if tp[i] > 0 else 0.0 for i in range(len(tp))]
            r_c = [float(tp[i].float() / (tp[i] + fn[i]).float()) * 100.0 if tp[i] > 0 else 0.0 for i in range(len(tp))]
            f_c = [2 * p_c[i] * r_c[i] / (p_c[i] + r_c[i]) if tp[i] > 0 else 0.0 for i in range(len(tp))]

            mean_p_c = sum(p_c) / len(p_c)
            mean_r_c = sum(r_c) / len(r_c)
            mean_f_c = sum(f_c) / len(f_c)

            p_o = tp.sum().float() / (tp + fp).sum().float() * 100.0
            r_o = tp.sum().float() / (tp + fn).sum().float() * 100.0
            f_o = 2 * p_o * r_o / (p_o + r_o)

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Precision {prec.val:.2f} ({prec.avg:.2f})\t'
                    'Recall {rec.val:.2f} ({rec.avg:.2f}) \t '
                    'P_C {P_C:.2f} \t R_C {R_C:.2f} \t F_C {F_C:.2f} \t '
                    'P_O {P_O:.2f} \t R_O {R_O:.2f} \t F_O {F_O:.2f}'.format(
                    i, len(data_loader), batch_time=batch_time,
                    prec=prec, rec=rec,
                    P_C=mean_p_c, R_C=mean_r_c, F_C=mean_f_c,
                    P_O=p_o, R_O=r_o, F_O=f_o
                ), flush=True)


        mAP_score = mAP(torch.cat(targets).numpy(), torch.cat(preds).numpy())

        # === NUEVO: Cálculo de la métrica Jaccard (IoU) por clase y global ===
        ji_per_class = []
        tp_fp_fn = (tp + fp + fn)

        for i_cls in range(len(tp)):
            denom = tp_fp_fn[i_cls]
            if denom > 0:
                ji = float(tp[i_cls]) / float(denom)  # tp / (tp+fp+fn)
            else:
                ji = 0.0
            ji_per_class.append(ji * 100.0)  # en %

        mean_jaccard_c = sum(ji_per_class) / len(ji_per_class)  # Promedio IoU en todas las clases

        # "Overall" Jaccard (IoU considerando todas las clases juntas)
        denom_o = (tp + fp + fn).sum().float()
        if denom_o > 0:
            jaccard_o = tp.sum().float() / denom_o
        else:
            jaccard_o = 0.0
        jaccard_o = jaccard_o * 100.0
        # === FIN NUEVO ===

    torch.cuda.empty_cache()

    # NUEVO ➔ Ahora devuelve también Dish Accuracy global
    return mean_p_c, mean_r_c, mean_f_c, p_o, r_o, f_o, mean_jaccard_c, jaccard_o, mAP_score, dish_acc_meter.avg

def get_object_names(classnames, target):
    objects = []
    for idx, t in enumerate(target):
        if t == 1:
            objects.append(classnames[idx])
    return objects


def validate_zsl(data_loader, model, args, cls_id):
    batch_time = AverageMeter()

    Softmax = torch.nn.Softmax(dim=1)
    Sig = torch.nn.Sigmoid()

    model.eval()

    preds = []
    targets = []
    output_idxs = []

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(data_loader):
            target = target.max(dim=1)[0]

            if torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")

            images = images.to(device)

            # compute output
            with autocast():
                output = model(images, cls_id)

            target = target[:, cls_id]
            if output.dim() == 3:
                output = Softmax(output).cpu()[:, 1]
            else:
                output = Sig(output).cpu()

            preds.append(output.cpu())
            targets.append(target.cpu())

            output_idx = output.argsort(dim=-1, descending=True)
            output_idxs.append(output_idx)

            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'.format(
                    i, len(data_loader), batch_time=batch_time
                ), flush=True)

        precision_3, recall_3, F1_3 = calc_F1(
            torch.cat(targets, dim=0).cpu().numpy(),
            torch.cat(output_idxs, dim=0).cpu().numpy(),
            args.top_k,
            num_classes=len(cls_id)
        )

        if F1_3 != F1_3:  # check NaN
            F1_3 = 0

        mAP_score = mAP(torch.cat(targets).numpy(), torch.cat(preds).numpy())

    torch.cuda.empty_cache()

    return precision_3, recall_3, F1_3, mAP_score