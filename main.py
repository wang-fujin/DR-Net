import os
import torch
import numpy as np
import random
import argparse
from utils import create_logger,AverageMeter,mkdir
import matplotlib.pyplot as plt
from model import TransferNet
from load_data import load_data
import time
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

def save_to_txt(save_name,string):
    f = open(save_name,mode='a')
    f.write(string)
    f.write('\n')
    f.close()
def get_args():
    parser = argparse.ArgumentParser(description='Battery Capacity Estimation')
    parser.add_argument('--input_channel',default=24)
    parser.add_argument('--embedding_length',default=256)
    parser.add_argument('--use_dsbn',type=bool,default=False)
    parser.add_argument('--predict_scheme',type=str,default='share', choices=['share','all'])
    parser.add_argument('--recon_scheme',type=str,default='all',choices=['share','all'])

    parser.add_argument('--source_dataset',type=str,default='D1')
    parser.add_argument('--target_dataset',type=str,default='D3')
    parser.add_argument('--target_test_num',type=int,default=1)

    parser.add_argument('--window_len',type=int,default=8,help='use [window_len] cycles to predict [one] capacity')
    parser.add_argument('--merge_direction',type=str,default='feature',choices=['time','feature'])

    parser.add_argument('--D1',default={'x':['x_MIT2_11','x_MIT2_12','x_MIT2_27','x_MIT2_38','x_MIT2_29'],
                                            'y':['y_MIT2_11','y_MIT2_12','y_MIT2_27','y_MIT2_38','y_MIT2_29']})
    parser.add_argument('--D2',default={'x':['x_MIT2_1','x_MIT2_2','x_MIT2_10','x_MIT2_20','x_MIT2_42'],
                                            'y':['y_MIT2_1','y_MIT2_2','y_MIT2_10','y_MIT2_20','y_MIT2_42']})
    parser.add_argument('--D3',default={'x':['x_MIT2_21','x_MIT2_22','x_MIT2_31','x_MIT2_36','x_MIT2_9'],
                                            'y':['y_MIT2_21','y_MIT2_22','y_MIT2_31','y_MIT2_36','y_MIT2_9']})
    parser.add_argument('--D4',default={'x':['x_MIT2_13','x_MIT2_16','x_MIT2_23','x_MIT2_24','x_MIT2_47'],
                                            'y':['y_MIT2_13','y_MIT2_16','y_MIT2_23','y_MIT2_24','y_MIT2_47']})
    parser.add_argument('--D5', default={'x': ['x_MIT2_19', 'x_MIT2_33', 'x_MIT2_34', 'x_MIT2_40', 'x_MIT2_43'],
                                             'y': ['y_MIT2_19', 'y_MIT2_33', 'y_MIT2_34', 'y_MIT2_40', 'y_MIT2_43']})


    parser.add_argument('--normalize_type',default='minmax',choices=['minmax','standerd'])
    parser.add_argument('--batch_size',default=64)
    parser.add_argument('--seed',default=2021)

    # L = L_mse + alpha*L_info + beta*L_diff + gamma*L_simi
    parser.add_argument('--alpha',type=float,default=0.05)
    parser.add_argument('--beta',type=float,default=0.001)
    parser.add_argument('--gamma',type=float,default=80)

    # optimizer
    parser.add_argument('--lr',type=float, default=2e-3)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)

    # learning rate scheduler
    parser.add_argument('--lr_gamma', type=float, default=0.0003)
    parser.add_argument('--lr_decay', type=float, default=0.75)
    parser.add_argument('--lr_scheduler', type=bool, default=True)

    parser.add_argument('--device',default='cuda')
    parser.add_argument('--n_epoch',default=100)
    parser.add_argument('--early_stop', default=0)

    # reulsts
    parser.add_argument('--is_plot_test_results',default=True)
    parser.add_argument('--is_save_plot',default=False)
    parser.add_argument('--is_save_logging',default=True)
    parser.add_argument('--is_save_best_model',default=True)
    parser.add_argument('--is_save_to_txt',default=True)


    parser.add_argument('--save_root',default='results')

    args = parser.parse_args()
    return args

def set_random_seed(seed=2021):
    random.seed(seed)
    np.random.seed(seed)



def get_optimizer(model,args):
    initial_lr = args.lr if not args.lr_scheduler else 1.0
    params = model.get_parameters(initial_lr=initial_lr)
    optimizer = torch.optim.SGD(params,lr=initial_lr,momentum=args.momentum,weight_decay=args.weight_decay,nesterov=False)
    return optimizer

def get_scheduler(optimizer,args):
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda x: args.lr * (1. + args.lr_gamma * float(x)) ** (
        -args.lr_decay))
    return scheduler

def test(model,target_test_loader,args):
    model.eval()
    test_loss = AverageMeter()
    criterion = torch.nn.MSELoss()
    with torch.no_grad():
        ground_true = []
        predict_label = []
        for data, target in target_test_loader:
            data, target = data.to(args.device), target.to(args.device)
            output = model.predict(data)
            loss = criterion(output,target)
            test_loss.update(loss.item())
            ground_true.append(target.cpu().detach().numpy())
            predict_label.append(output.cpu().detach().numpy())
    return test_loss.avg, np.concatenate(ground_true), np.concatenate(predict_label)


def train(source_loader,target_train_loader,target_test_loader,model,optimizer,lr_scheduler,args):
    if args.is_save_logging:
        mkdir(args.save_root)
        log_name = args.save_root + '/train info.log'
        log, consoleHander, fileHander= create_logger(filename=log_name)
        log.critical(args)
        log.info('source data:'+args.source_dataset)
        log.info('target data:'+args.target_dataset)
        log.info('target test num:'+str(args.target_test_num))
    else:
        log, consoleHander = create_logger()
    len_source_loader = len(source_loader)
    len_target_loader = len(target_train_loader)
    n_batch = min(len_source_loader,len_target_loader)

    iter_source, iter_target = iter(source_loader), iter(target_train_loader)

    stop = 0
    min_test_loss = 10
    last_best_model = None
    for e in range(1,args.n_epoch+1):
        model.train()
        train_diff_loss, train_simi_loss, train_info_loss, train_pred_loss = AverageMeter(),AverageMeter(),AverageMeter(),AverageMeter()
        train_total_loss = AverageMeter()
        if max(len_target_loader, len_source_loader) != 0:
            iter_source, iter_target = iter(source_loader), iter(target_train_loader)

        for _ in range(n_batch):
            data_source, label_source = next(iter_source)
            data_target, _ = next(iter_target)
            data_source = data_source.to(args.device)
            label_source = label_source.to(args.device)
            data_target = data_target.to(args.device)

            diff_loss, simi_loss, info_loss, pred_loss = model(data_source,data_target,label_source)

            if args.recon_scheme == 'share' and args.predict_scheme == 'share':

                loss = pred_loss + args.alpha*info_loss + args.gamma*simi_loss
            else:
                loss = pred_loss + args.alpha*info_loss + args.beta*diff_loss + args.gamma*simi_loss

            if args.alpha == 0 and args.beta==0 and args.gamma == 0:
                loss = pred_loss


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if lr_scheduler:
                lr_scheduler.step()

            train_diff_loss.update(diff_loss.item())
            train_info_loss.update(info_loss.item())
            train_simi_loss.update(simi_loss.item())
            train_pred_loss.update(pred_loss.item())
            train_total_loss.update(loss.item())
        train_info = f'Epoch:[{e}/{args.n_epoch}], pred_loss:{train_pred_loss.avg:.4f}, ' \
                     f'info_loss:{train_info_loss.avg:.4f}->{args.alpha*train_info_loss.avg:.4f}, ' \
                     f'simi_loss:{train_simi_loss.avg:.4f}->{args.gamma*train_simi_loss.avg:.4f}, ' \
                     f'diff_loss:{train_diff_loss.avg:.4f}->{args.beta*train_diff_loss.avg:.4f}.  ' \
                     f'total_loss:{train_total_loss.avg:.4f}'
        log.info(train_info)

        ##################### test #######################
        stop += 1
        test_loss, true_label, pred_label = test(model, target_test_loader, args)
        test_info = f"test_loss:{test_loss:.4f}  lr:{optimizer.state_dict()['param_groups'][0]['lr']}"
        log.warning(test_info)

        if min_test_loss > test_loss:
            min_test_loss = test_loss
            stop = 0
            #######plot test results#########
            if args.is_plot_test_results:
                plt.plot(true_label, label='true')
                plt.plot(pred_label, label='pred')
                plt.title(f"Epoch:{e}, test loss:{test_loss:.4f}")
                plt.legend()
                plt.show()
            ####### save model ########
            if args.is_save_best_model:
                if last_best_model is not None:
                    os.remove(last_best_model)  # delete last best model

                save_folder = args.save_root + '/pth'
                mkdir(save_folder)
                best_model = os.path.join(save_folder, f'Epoch{e}.pth')
                torch.save(model.state_dict(), best_model)
                last_best_model = best_model
            #########save test info to txt #####
            if args.is_save_to_txt:
                txt_path = args.save_root + '/test_info.txt'
                time_now = time.strftime("%Y-%m-%d", time.localtime())
                info = time_now + f' {args.alpha}-{args.beta}-{args.gamma}, epoch = {e}, test_loss:{test_loss:.6f}'
                save_to_txt(txt_path,info)

        if args.early_stop > 0 and stop > args.early_stop:
            print(' Early Stop !')
            if args.is_save_logging:
                log.removeHandler(consoleHander)
                log.removeHandler(fileHander)
            else:
                log.removeHandler(consoleHander)
            break
    if args.is_save_logging:
        log.removeHandler(consoleHander)
        log.removeHandler(fileHander)
    else:
        log.removeHandler(consoleHander)

def main():
    args = get_args()
    set_random_seed(args.seed)
    source_loader, target_train_loader, target_test_loader = load_data(args)
    model = TransferNet(args).to(args.device)
    optimizer = get_optimizer(model, args)
    if args.lr_scheduler:
        scheduler = get_scheduler(optimizer, args)
    else:
        scheduler = None
    train(source_loader, target_train_loader, target_test_loader, model, optimizer, scheduler, args)



if __name__ == '__main__':
    main()



