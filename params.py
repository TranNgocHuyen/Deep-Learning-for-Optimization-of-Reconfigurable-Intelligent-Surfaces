import numpy as np
import torch
import copy


# RT channel model

# Khởi tạo 1 dict, với các giá trị kênh cố định CHO 2 USERS
params4users = {'lr': 8e-4,             # learning rate
                'epoch': 50,            # kỷ nguyên
                'num_users': 4,         #4 USERS 
                'iter_wmmse': 10,     # số loop của mã hóa wmmse
                'epoch_per_iter_wmmse': 1,
                'entropy_history_length': 5,
                'alphas': [0.25, 0.25, 0.25, 0.25],   # hệ số của 4 người dùng
                'saving_frequency': 10, 
                'wmmse_saving_frequency': 10,
                'batch_size': 512,       
                'permutation_invariant': True,  # mode HOÁN VỊ BẤT BIẾN
                'results_path': 'results/',     # path of results
                'tsnr': 1e11,                   # tsnr
                "frequencies": np.linspace(1e6, 1e6 + 100, 10), #một mảng 10 giá trị đều nhau trong một khoảng xác định [1e6, 1e6 + 100]
                'quantile2keep': 0.6,
                "phase_shift": "continuous",    # dịch pha liên tục , not rời rạc
                "discrete_phases": torch.tensor([0, np.pi])[None, None, :], #tensor([[[0.0000, 3.1416]]])

                # Các thông số để chuẩn hóa
                'mean_ris': 6.2378e-5,          #fix channels_ris_rx
                'std_ris': 5.0614e-5,           #fix
                'mean_direct': 1.4374e-4,       #fix channel_direct @ channel_tx_ris_pinv
                'std_direct': 3.714e-4,         #fix
                
                'ris_shape': (32, 32),          # kích thước của IRS là 32x32=1024
                'n_tx_antennas': 9,
                'channel_tx_ris_original_shape': (32, 32, 9),  #(NxM)=(1024,9)=(32,32,9) # width, height of RIS and Tx antennas.DO NOT CHANGE THIS !
                'channel_ris_rx_original_shape': (10240, 32, 32),  #  samples,width, height of RIS and users ????????????
                
                
                
                'channel_direct_path': 'data/channels_direct_training_s.pt',  #torch.Size([10240, 1, 9]) ok
                'channel_tx_ris_path': 'data/channel_tx_ris_s.pt',            # #torch.Size([1024, 9]) (NxM) ok
                'channel_ris_rx_path': 'data/channels_ris_rx_training_s.pt',  # torch.Size([10240, 1, 1024])    ok
                'location_path': 'data/locations_training.pt',                  # torch.Size([16000, 3]) tọa độ
                'group_definition_path': 'data/group_definition_4users_training_s.npy', #(5120, 4)

                'los': True,
                'precoding': 'wmmse',

                'angle_diff_threshold': 0.5,  # ngưỡng lệch góc là 0.5
                'user_distance_threshold': 20,  # ngưỡng khoảng cách của user=20
                'ris_loc': torch.tensor([278.42, 576.97, 2]),
                'trained_mmse_model': None,
                # 'trained_mmse_model': 'results/RISNetPIDiscrete_MMSE_16-05-2022_13-46-01/ris_100000000000.0_(32, 32)_[0.5, 0.5]_4000',
                'channel_estimate_error': 0,
                'discount_long': 0.95,
                'discount_short': 0.4,
                'delta_support': 0.0001,
                }

# If statistical channel (thống kê)
if True:  
    params4users['mean_ris'] = 7.979e-4
    params4users['std_ris'] = 6.028e-4
    params4users['mean_direct'] = 1.0074e-4
    params4users['std_direct'] = 6.361e-5

params = params4users