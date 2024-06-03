import numpy as np
from util import test_model, prepare_channel_tx_ris, \
    compute_complete_channel_continuous, weighted_sum_rate
#from train_oma import mmse_precoding
import torch
from core import RISNet, RISNetPI, RTChannelsWMMSE
from torch.utils.data import DataLoader
from params import params
import matplotlib.pyplot as plt
tb = True
try:
    from tensorboardX import SummaryWriter
except:
    tb = False


def ecdf(data):
    x = np.sort(data)
    n = x.size
    y = np.arange(1, n+1) / n
    return x, y


if __name__ == '__main__':
    params["channel_direct_path"] = 'data/channels_direct_testing_s.pt'
    params["channel_tx_ris_path"] = 'data/channel_tx_ris_s.pt'
    params["channel_ris_rx_path"] = 'data/channels_ris_rx_testing_s.pt'
    params["group_definition_path"] = 'data/group_definition_4users_testing_s.npy'
    params['channel_ris_rx_original_shape'] = (1024, 32, 32)  # samples, width, height of RIS and users
    
    params["permutation_invariant"] = True

    if params["permutation_invariant"]==True:
            model = RISNetPI(params)
    else:
            model = RISNet(params)

    device = 'cpu'
    wsr_array=[]
    channel_tx_ris, channel_tx_ris_pinv = prepare_channel_tx_ris(params, device)

    if params["permutation_invariant"]==True:
      for i in [1e11,5e11,1e12]:
        params["tsnr"] = i

        model.load_state_dict(torch.load('/home/tranngochuyen/Do_an/Deep-Learning-for-Optimization-of-Reconfigurable-Intelligent-Surfaces/results_server/26-05-2024_21-00-02/ris_1000000000000.0_(32, 32)_permutation_invariant_WMMSE_400.pt',map_location=torch.device('cpu')))
        model.eval()
        print("DONE LOAD MODEL")

        data_set = RTChannelsWMMSE(params, channel_tx_ris_pinv, device, test=True)
        test_loader = DataLoader(dataset=data_set, batch_size=1024, shuffle=True)
        print("DONE LOAD DATA")
        
        data_set.wmmse_precode(model, channel_tx_ris, device, 500)
        print("DONE PRECODE")

        for batch in test_loader:
            sample_indices, channels_ris_rx_features_array, channels_ris_rx, channels_direct, location, precoding = batch

            entropy_current_epoch = list()
            fcn_raw_output = model(channels_ris_rx_features_array)
            complete_channel = compute_complete_channel_continuous(channel_tx_ris, fcn_raw_output, channels_ris_rx,
                                                             channels_direct, params)

            wsr = weighted_sum_rate(complete_channel, precoding, params)
            wsr_array.append(wsr)
            print('average = {ave}'.format(ave=wsr.mean()))
    
      torch.save(wsr_array,'test/'+'wsr_PI')

    elif params["permutation_invariant"]== False:
     for i in [1e11,5e11,1e12]:
        params["tsnr"] = i

        model.load_state_dict(torch.load('/home/tranngochuyen/Do_an/Deep-Learning-for-Optimization-of-Reconfigurable-Intelligent-Surfaces/results_server/26-05-2024_21-00-02/ris_1000000000000.0_(32, 32)_permutation_invariant_WMMSE_400.pt',map_location=torch.device('cpu')))
        model.eval()
        print("DONE LOAD MODEL")

        data_set = RTChannelsWMMSE(params, channel_tx_ris_pinv, device, test=True)
        test_loader = DataLoader(dataset=data_set, batch_size=1024, shuffle=True)
        print("DONE LOAD DATA")
        
        data_set.wmmse_precode(model, channel_tx_ris, device, 500)
        print("DONE PRECODE")

        for batch in test_loader:
            sample_indices, channels_ris_rx_features_array, channels_ris_rx, channels_direct, location, precoding = batch

            entropy_current_epoch = list()
            fcn_raw_output = model(channels_ris_rx_features_array)
            complete_channel = compute_complete_channel_continuous(channel_tx_ris, fcn_raw_output, channels_ris_rx,
                                                             channels_direct, params)

            wsr = weighted_sum_rate(complete_channel, precoding, params)
            wsr_array.append(wsr)
            print('average = {ave}'.format(ave=wsr.mean()))
    
     torch.save(wsr_array,'test/'+'wsr_PI')
