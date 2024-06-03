from util import weighted_sum_rate, prepare_channel_tx_ris, compute_complete_channel_continuous
from core import RISNet, RISNetPI, RTChannelsWMMSE

import torch.optim as optim
import torch
from torch.utils.data import DataLoader
import numpy as np
from params import params

import argparse
import datetime
from pathlib import Path
tb = True
try:
    from tensorboardX import SummaryWriter
except:
    tb = False
#record = False and tb
#record=False or tb

# FILE MAIN
if __name__ == '__main__':
    ''' tạo đối tượng ArgumentParser'''
    parser = argparse.ArgumentParser()
    # thêm các đối số tùy chọn
    parser.add_argument("--tsnr") #1e11 or 5e11 or 1e12
    parser.add_argument("--ris_shape")  # 32,32
    parser.add_argument("--weights")    # 0.25, 0.25, 0.25, 0.25
    parser.add_argument("--lr")         # 8e-4
    parser.add_argument("--record") # luôn True # biến ko trong dict params
    parser.add_argument("--device") # cuda or cpu # biến ko trong dict params
    parser.add_argument("--model")  # permutation_invariant(RISNetPI) or permutation_variant (RISNet)
    parser.add_argument("--iter_wmmse") # ban đầu là 100
    #Phân tích các đối số dòng lệnh và lưu trữ trong 'args'
    args = parser.parse_args()

    if args.tsnr is not None:               # nếu nhập tsnr từ command line 
        params["tsnr"] = float(args.tsnr)   # thì gán lại vào từ điển params
    if args.ris_shape is not None:
        ris_shape = args.ris_shape.split(',')
        params["ris_shape"] = tuple([int(s) for s in ris_shape]) #tuple
    if args.lr is not None:             
        params["lr"] = float(args.lr)
    if args.weights is not None:
        weights = args.weights.split(',')
        params["alphas"] = np.array([float(w) for w in weights]) # mảng

    if args.record is not None:
        record = args.record =="True" # luôn True
    tb = tb and record

    if args.device is not None:
        device = args.device
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device: {device}')
    # Nếu record=True thì tạo ra một thư mục mới để lưu trữ kết quả
    if record:
        #lấy thời gian hiện tại
        now = datetime.datetime.now()
        # định dạng thời gian
        dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")
        # tạo file trong thư mục results
        Path(params["results_path"] + dt_string).mkdir(parents=True, exist_ok=True)
        #Cập nhật đường dẫn kết quả
        params["results_path"] = params["results_path"] + dt_string + "/"
    
    # ĐOẠN NÀY THAY ĐỔI 1 SỐ PARAMS
    params["discrete_phases"] = params["discrete_phases"].to(device) #tensor([[[0.0000, 3.1416]]])
    if args.iter_wmmse is not None:
        params["iter_wmmse"] = int(args.iter_wmmse)
    

    if args.model is not None:
        if args.model=="permutation_invariant":
            params["permutation_invariant"]=True
            model = RISNetPI(params).to(device)
        elif args.model=="permutation_variant":
            params["permutation_invariant"]=False
            model = RISNet(params).to(device)

#=========================================LOAD DATASET==========================================================
    
    print("==================START LOAD DATASET===================")
    # Tạo tên cho file lưu checkpoint
    result_name = "ris_" + str(params['tsnr']) + "_" + str(params['ris_shape']) + '_' + str(args.model) + "_"

    #tính kênh channel_tx_ris (H) và nghịch đảo giả của H (H+)
    channel_tx_ris, channel_tx_ris_pinv = prepare_channel_tx_ris(params, device)
    # tạo dataset
    data_set = RTChannelsWMMSE(params, channel_tx_ris_pinv, device)
    #print(len(data_set)) #5120
    # tải dữ liệu từ data_set với kích thước lô 512
    train_loader = DataLoader(dataset=data_set, batch_size=params['batch_size'], shuffle=True)
    
    #tạo list để lưu lost
    losses = list()

    #Tensorboard
    if tb:
        writer = SummaryWriter(logdir=params["results_path"])
        tb_counter = 1 #đếm số lần ghi dữ liệu vào TensorBoard.

#=========================================TRAINING=====================================================#
    start_time = datetime.datetime.now()
    model.train()

    optimizer_wmmse = optim.Adam(model.parameters(), params['lr'])
 
    # Training with WMMSE precoder
    print("==================START TRAINING=======================")
    
    iter_WSR=[]
    WSR_batch=[]
    num_iter_wmmse=params['iter_wmmse']
    for wmmse_iter in range(params['iter_wmmse']+1): # 100
        #print(f'Start WMMSE round: {wmmse_iter}/{num_iter_wmmse}')
        
        data_set.wmmse_precode(model, channel_tx_ris, device) # tính v rồi append vô dataset
        
        epoch_per_iter_wmmse=params['epoch_per_iter_wmmse']
        for epoch in range(params['epoch_per_iter_wmmse']): #=1
            for batch in train_loader:
                item, channels_ris_rx_features_array, channels_ris_rx, channels_direct, location, precoding = batch

                optimizer_wmmse.zero_grad()
                #print(channels_ris_rx_features_array.shape)             #[512, 16, 1024]
                nn_raw_output = model(channels_ris_rx_features_array) # ma trận pha [512,1,1024]
                #print(nn_raw_output.shape)

                # model trả về phi, từ đó tính được kênh kết hợp
                complete_channel = compute_complete_channel_continuous(channel_tx_ris, nn_raw_output,
                                                                       channels_ris_rx, channels_direct, params)
                # Tính WSR
                wsr = weighted_sum_rate(complete_channel, precoding, params)

                #loss=- (trung bình các giá trị wsr /batch)
                loss = -torch.mean(wsr)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                
                for name, param in model.named_parameters():
                    if torch.isnan(param.grad).any():
                        print("nan gradient found")
                optimizer_wmmse.step()

                #print(f'WMMSE round: {wmmse_iter}/{num_iter_wmmse}, Epoch: {epoch}/{epoch_per_iter_wmmse}, WSR = {-loss}')
                WSR_batch.append(-loss.item())
                
                if tb and record:
                    writer.add_scalar("Training/WSR", -loss.item(), tb_counter)
                    tb_counter += 1

                
                #print('Checkpoint end batch')
            
            print(f'WMMSE round: {wmmse_iter}/{num_iter_wmmse}, Epoch: {epoch}/{epoch_per_iter_wmmse}, WSR = {-loss}')
        
        # Lưu WSR theo iter WSR
        iter_WSR.append(-loss.item())

        if record and wmmse_iter % params['wmmse_saving_frequency'] == 0:
            torch.save(model.state_dict(), params['results_path'] + result_name +'WMMSE_{iter}'.format(iter=wmmse_iter)+'.pt')
            if tb:
                writer.flush()
    torch.save(WSR_batch,params['results_path'] +'WSR_batch.pt')
    torch.save(iter_WSR, params['results_path'] +'iter_WSR.pt')
    
    # END TIME
    end_time = datetime.datetime.now()
    elapsed_time = end_time - start_time

    # Tính giờ, phút, giây
    hours, remainder = divmod(elapsed_time.total_seconds(), 3600)
    minutes, seconds = divmod(remainder, 60)

    print(f"Thời gian chạy: {int(hours)} giờ {int(minutes)} phút {seconds:.2f} giây")
#python train.py --tsnr 1e11 --lr 8e-4 --ris_shape 32,32 --weights 0.25,0.25,0.25,0.25 --record True --device cpu
