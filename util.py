import numpy as np
import torch
from scipy.linalg import eigh
import copy
from joblib import Parallel, delayed

solver = True
try:
    from scipy.optimize import fsolve
except:
    solver = False

from params import params
device='cpu'

'''Hàm thực hiện tiền mã hóa mmse precoding trả về V'''
def mmse_precoding(complete_channel, params, device='cpu'):
    if type(complete_channel) is np.ndarray:
        complete_channel = torch.from_numpy(complete_channel).to(device)
    eye = torch.eye(complete_channel.shape[1]).repeat((complete_channel.shape[0], 1, 1)).to(device)
    
    #torch.eye =>Trả về một tenxơ 2-D với các số 1 trên đường chéo và các số 0 ở nơi khác.
    p = complete_channel.transpose(1, 2).conj() @ torch.linalg.inv(complete_channel @ complete_channel.transpose(1, 2).conj() +
                                                          1 / params['tsnr'] * eye)
    trace = torch.sum(torch.diagonal((p @ p.transpose(1, 2).conj()), dim1=1, dim2=2).real, dim=1, keepdim=True)
    p = p / torch.unsqueeze(torch.sqrt(trace), dim=2)
    return p


'''CHuẩn hóa và trả về the features of RIS antenna n, và x1024 phần tử'''
# Tính G và chuẩn hóa và trả về features (độ lớn, pha) của G (irs-> rx)
def cp2array_risnet(cp, factor=1, mean=0, device="cpu"):
    # Input: (number of data samples,1,N antenna)
    # Output: (number of data samples,2 feature,N antenna)) 
    # features là độ lớn và pha của mỗi phần tử phức
    
    array = torch.cat([(cp.abs() - mean) * factor, cp.angle() * 0.55], dim=1) # cat là ghép 2 ma trận lại (ko phải cộng từng phần tử)
    return array.to(device)


# Tính toán J=DH+ và chuẩn hóa và trả về features (độ lớn,pha) của J 
def prepare_channel_direct_features(channel_direct, channel_tx_ris_pinv, params, device='cpu'):
    equivalent_los_channel = channel_direct @ channel_tx_ris_pinv
    return cp2array_risnet(equivalent_los_channel, 1 / params['std_direct'], params["mean_direct"], device)

'''Hàm tính WSR'''
def weighted_sum_rate(complete_channel, precoding, params):
    channel_precoding = complete_channel @ precoding 
    channel_precoding = torch.square(channel_precoding.abs())
    wsr = 0
    num_users = channel_precoding.shape[1]
    for user_idx in range(num_users):
        wsr += params["alphas"][user_idx] * torch.log2(1 + channel_precoding[:, user_idx, user_idx] /
                                                       (torch.sum(channel_precoding[:, user_idx, :], dim=1)
                                                        - channel_precoding[:, user_idx, user_idx]
                                                        + 1 / params["tsnr"]))
    return wsr


def test_model(complete_channel, precoding, params):
    if type(complete_channel) is np.ndarray:
        complete_channel = torch.from_numpy(complete_channel).cfloat()

    if type(precoding) is np.ndarray:
        precoding = torch.from_numpy(precoding).cfloat()

    channel_precoding = complete_channel @ precoding
    channel_precoding = torch.square(channel_precoding.abs())
    data_rates = list()
    num_users = channel_precoding.shape[1]
    for user_idx in range(num_users):
        data_rates.append(torch.log2(1 + channel_precoding[:, user_idx, user_idx] /
                                     (torch.sum(channel_precoding[:, user_idx, :], dim=1)
                                      - channel_precoding[:, user_idx, user_idx]
                                      + 1 / params["tsnr"])).cpu().detach().numpy())
    return channel_precoding.cpu().detach().numpy(), data_rates


def array2phase_shifts(phase_shifts):
    # Input: (batch, 1, width, height)
    # Output: (batch, antenna, antenna)
    p = torch.flatten(phase_shifts[:, 0, :, :], start_dim=1, end_dim=2)
    p = torch.diag_embed(torch.exp(1j * p))
    return p


def compute_wmmse_v_v2(h_as_array, init_v, tx_power, noise_power, params, num_iters=500):
    num_users, num_tx_antennas = h_as_array.shape
    h_list = [h_as_array[user_idx: (user_idx + 1), :] for user_idx in range(num_users)]
    v_list = [init_v[:, user_idx: (user_idx + 1)] for user_idx in range(num_users)]
    w_list = [1 for _ in range(num_users)]
    for iter in range(num_iters):
        w_list_old = copy.deepcopy(w_list)

        # Step 2
        u_list = list()
        for user_idx in range(num_users):
            inv_hvvhi = (1 / (np.sum([np.real(h_list[user_idx] @ v
                                              @ v.transpose().conj() @ h_list[user_idx].transpose().conj())
                                      for v in v_list]) + noise_power))
            u_list.append(inv_hvvhi * h_list[user_idx] @ v_list[user_idx])

        # Step 3
        for user_idx in range(num_users):
            w_list[user_idx] = 1 / np.real(1 - u_list[user_idx].transpose().conj()
                                           @ h_list[user_idx] @ v_list[user_idx])

        # Step 4
        mmu = sum([alpha * h.transpose().conj() @ u @ w @ u.transpose().conj() @ h for alpha, h, u, w, in
                   zip(params["alphas"], h_list, u_list, w_list)])
        mphi = sum([alpha ** 2 * h.transpose().conj() @ u @ w ** 2 @ u.transpose().conj() @ h for alpha, h, u, w in
                    zip(params["alphas"], h_list, u_list, w_list)])

        try:
            lambbda, d = eigh(mmu)
        except:
            break
        lambbda = np.real(lambbda)
        phi = d.transpose().conj() @ mphi @ d
        phi = np.real(np.diag(phi))
        if solver:
            mu = fsolve(solve_mu, 0, args=(phi, lambbda, tx_power))
        else:
            raise ImportError('scipy.optimize.fsolve cannot be imported.')
        mv = np.linalg.inv(mmu + mu * np.eye(num_tx_antennas))

        v_list = [alpha * mv @ h.transpose().conj() @ u @ w for alpha, h, u, w in
                  zip(params["alphas"], h_list, u_list, w_list)]

        if np.sum([np.abs(w - w_old) for w, w_old in zip(w_list, w_list_old)]) < np.abs(w_list[0]) / 20:
            break

    precoding = np.hstack(v_list)
    power = np.sum(np.abs(precoding) ** 2)
    return precoding / np.sqrt(power)


def wmmse_precoding(h, tx_power, noise_power, num_tx_antennas, params, num_cpus=1):
    num_samples = h.shape[0]
    res = Parallel(n_jobs=num_cpus)(delayed(compute_wmmse_v_v2)(h[sample_id, :, :], tx_power, noise_power,
                                                                params)
                                    for sample_id in range(num_samples))
    v = np.stack(res, axis=0)
    return v


def solve_mu(mu, *args):
    phi = args[0]
    lambbda = args[1]
    p = args[2]
    return np.sum(phi / (lambbda + mu + 1e-3) ** 2) - p

'''Từ phi, tính kênh đầy đủ '''
def compute_complete_channel_continuous(channel_tx_ris, fcn_output, channel_ris_rx, channel_direct, params): #fcn_output: features
    
    # phi(phần tử trong ma trận pha) có kích thước [1,1024]
    phi = torch.exp(1j * fcn_output)

    # compute complete channel G.phi@H or G.phi@H+D 
    if channel_direct is None:
        complete_channel = (channel_ris_rx * phi) @ channel_tx_ris
    else:
        complete_channel = (channel_ris_rx * phi) @ channel_tx_ris + channel_direct
    return complete_channel


'''Hàm tính channel_tx_ris H và nghịch đảo giả H+'''
def prepare_channel_tx_ris(params, device):
    
    channel_tx_ris = torch.load(params['channel_tx_ris_path'], map_location=torch.device(device)).to(torch.complex64)#torch.Size([1024, 9])
    
    # lấy ra thông tin kênh tương ứng số phần tử IRS=> có thể tinh chỉnh số phần tử
    channel_tx_ris = channel_tx_ris[:params["ris_shape"][0] * params["ris_shape"][1], :] #torch.Size([N*N, 9])
    
    channel_tx_ris_pinv = torch.linalg.pinv(channel_tx_ris)
    return channel_tx_ris, channel_tx_ris_pinv

