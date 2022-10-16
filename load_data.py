from torch.utils.data import TensorDataset,DataLoader
import numpy as np
from utils import Scaler
import torch



def load_data(args):
    window_len = args.window_len
    if args.merge_direction == 'feature':
        merge_direction = 1
    else:
        merge_direction = 0
    source_dataset = args.source_dataset
    target_dataset = args.target_dataset

    ############ source data ###########
    source_x = []
    source_y = []
    for i in range(len(eval(f"args.{source_dataset}")['x'])):
        x_path = f'data/{source_dataset}/' + eval(f"args.{source_dataset}")['x'][i] + '.npy'
        y_path = f'data/{source_dataset}/' + eval(f"args.{source_dataset}")['y'][i] + '.npy'
        x_i = np.load(x_path)
        y_i = np.load(y_path)
        for j in range(x_i.shape[0]-window_len+1): # sliding window
            source_x.append(np.concatenate(x_i[j:j+window_len],axis=merge_direction))
            source_y.append(y_i[j+window_len-1])
    source_x = np.array(source_x,dtype=np.float32)
    source_y = np.array(source_y,dtype=np.float32)

    ########### target data ###########
    target_num = len(eval(f"args.{target_dataset}")['x'])
    target_train_num = target_num - args.target_test_num
    target_train_x = []
    target_train_y = []
    target_test_x = []
    target_test_y = []
    count = 0
    for i in range(len(eval(f"args.{target_dataset}")['x'])):
        count += 1
        x_path = f'data/{target_dataset}/' + eval(f"args.{target_dataset}")['x'][i] + '.npy'
        y_path = f'data/{target_dataset}/' + eval(f"args.{target_dataset}")['y'][i] + '.npy'
        x_i = np.load(x_path)
        y_i = np.load(y_path)
        for j in range(x_i.shape[0] - window_len + 1):  # sliding window
            if count <= target_train_num:
                target_train_x.append(np.concatenate(x_i[j:j + window_len], axis=merge_direction))
                target_train_y.append(y_i[j + window_len - 1])
            else:
                target_test_x.append(np.concatenate(x_i[j:j + window_len], axis=merge_direction))
                target_test_y.append(y_i[j + window_len - 1])
    target_train_x = np.array(target_train_x, dtype=np.float32)
    target_train_y = np.array(target_train_y, dtype=np.float32)
    target_test_x = np.array(target_test_x, dtype=np.float32)
    target_test_y = np.array(target_test_y, dtype=np.float32)

    print('source ：', source_x.shape, source_y.shape)
    print('target train ：', target_train_x.shape,target_train_y.shape)
    print('target test ：',target_test_x.shape, target_test_y.shape)


    ############ Normalize ##############
    print('-' * 30)
    print('normalized data !')
    if args.normalize_type == 'minmax':
        target_train_x, target_test_x = Scaler(target_train_x,target_test_x).minmax()
        target_train_y, target_test_y = Scaler(target_train_y, target_test_y).minmax()
        source_x = Scaler(source_x).minmax()
        source_y = Scaler(source_y).minmax()
    elif args.normalize_type == 'standerd':
        target_train_x, target_test_x = Scaler(target_train_x,target_test_x).standerd()
        target_train_y, target_test_y = Scaler(target_train_y, target_test_y).standerd()
        source_x = Scaler(source_x).standerd()
        source_y = Scaler(source_y).standerd()

    ########## dataloader #############
    target_train_x = torch.from_numpy(np.transpose(target_train_x,(0,2,1)))
    target_train_y = torch.from_numpy(target_train_y).view(-1,1)
    target_test_x = torch.from_numpy(np.transpose(target_test_x,(0,2,1)))
    target_test_y = torch.from_numpy(target_test_y).view(-1,1)
    source_x = torch.from_numpy(np.transpose(source_x,(0,2,1)))
    source_y = torch.from_numpy(source_y).view(-1,1)

    source_loader = DataLoader(TensorDataset(source_x, source_y), batch_size=args.batch_size, shuffle=True)
    target_train_loader = DataLoader(TensorDataset(target_train_x, target_train_y), batch_size=args.batch_size,
                                     shuffle=True)
    target_test_loader = DataLoader(TensorDataset(target_test_x, target_test_y), batch_size=args.batch_size,
                                     shuffle=False)
    return source_loader, target_train_loader, target_test_loader


if __name__ == '__main__':
    from main import get_args
    args = get_args()
    load_data(args)