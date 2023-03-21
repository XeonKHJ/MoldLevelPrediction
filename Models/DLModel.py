import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as torchrnn
import torch.optim as optim

class DLModel(nn.Module):
    """
        Parameters:
        - feature_nums: list of feature num for every detector.
        - hidden_size: number of hidden units
        - output_size: number of output
        - num_layers: layers of LSTM to stack
    """
    def __init__(self):
        super().__init__()
        hidden_size = 4
        num_layers = 2


        self.fc1 = nn.Linear(1,3)
        self.relu1 = nn.LeakyReLU()
        self.fc2 = nn.Linear(3,4)
        self.relu2 = nn.LeakyReLU()
        self.fc3 = nn.Linear(4,2)
        
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.noiseLstm = nn.LSTM(input_size=1,hidden_size=1, num_layers=num_layers, batch_first=True)
        self.forwardCalculation = nn.Linear(hidden_size, 2)

        self.noiseFc = nn.Linear(1, 1)

        self.B = 1250  # 连铸坯宽度
        self.W = 230  # 连铸坯厚度
        self.L = 1  # 结晶器内液面高度
        self.c2h = 1  # c2(h)：流量系数
        self.A = 11313  # 下水口侧孔面积
        self.Ht = 10  # 计算水头高
        self.H1t = 1  # 中间包液面高度
        self.H2 = 1300  # 下水口水头高度
        self.H3 = 2  # 下侧孔淹没高度，需要计算
        self.h = 1  # 塞棒高度
        self.startLv = 350


    def forward(self, x, context, lengths, preLv):
        hs = x
        x = self.fc1(x)  # _x is input, size (seq_len, batch, input_size)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)

        ts = torch.zeros(x.shape[1], device=torch.device('cuda'))
        ts[:] = 0.5
        pre_lv_act = torch.zeros(x.shape[0], 1, 1, device=torch.device('cuda'))
        pre_lv_act[:] = preLv
        deltaLvs, totalLvs = self.calculate_lv_acts_tensor(hs, ts, x, x.shape[0], 0, pre_lv_act)

        # noise_out, (h, c) = self.noiseLstm(x)
        # # firstNoiseOutput, (h,c) = self.noiseLstm(h[-1,:,:].reshape([-1,1,1]), (h, c))
        # noiseOutput = torch.zeros([x.shape[0],0,1])
        # for i in range(x.shape[1]):
        #     singleNoiseOutput, (h, c) = self.noiseLstm(h[-1,:,:].reshape([-1,1,1]), (h, c))
        #     noiseOutput = torch.cat((noiseOutput, singleNoiseOutput), 1)
        # noiseOutput = self.noiseFc(noiseOutput)
        
        # finalOutput = noiseOutput + deltaLvs
        lengths = (lengths-1).reshape(-1).tolist()
        totalLvs = totalLvs - self.startLv
        deltaLvs = torchrnn.pack_padded_sequence(deltaLvs, lengths, batch_first=True)
        deltaLvs, _ = torchrnn.pad_packed_sequence(deltaLvs, batch_first=True)
        totalLvs = torchrnn.pack_padded_sequence(totalLvs, lengths, batch_first=True)
        totalLvs, _ = torchrnn.pad_packed_sequence(totalLvs, batch_first=True)

        return deltaLvs, totalLvs
    

    def stp_pos_flow_tensor(self, h_act, lv_act, t, dt=0.5, params=[0,0,0,0]):
        H1t = params[:,0] # H1：中间包液位高度，t的函数，由LSTM计算
        g = 9.8                 # 重力
        # c2h = lpm(torch.tensor(h_act).reshape(-1))  # C2：和钢种有关的系数，由全网络计算
        c2h = params[:,1]

        # 引锭头顶部距离结晶器底部高度350+结晶器液位高度（距离引锭头）283
        # if lv_act < 63.3:
        #     H3 = 0
        # else:
        #     H3 = lv_act-63.3  # H3下侧出口淹没高度
        H3 = 0
        Ht = H1t+self.H2-H3
        dL = (pow(2 * g * Ht, 0.5) * c2h * self.A * dt) / (self.B * self.W)
        return dL

    def calculate_lv_acts_tensor(self, hs, ts, params, batch_first = True, previousTime = 0, pre_lv_act = 0):
        sampleRate = 2  # 采样率是2Hz。
        # 维度为（时间，数据集数量，特征数）
        deltaLvs = torch.zeros(hs.shape, device=torch.device('cuda'))
        totalLvs = torch.zeros(hs.shape, device=torch.device('cuda'))
        batchSize = hs.shape[0]
        lv = pre_lv_act
        sample_count = 0
        for stage in range(params.shape[1]):
            # stopTimeSpan = ts[stage]
            if stage > 0:
                previousTime += ts[stage-1]
            time = stage
            current_lv = self.stp_pos_flow_tensor(hs[:,stage,:], lv, previousTime + time / 2, 1 / sampleRate, params[:, stage, :])
            # print(current_lv.reshape([-1]).item())
            current_lv = current_lv.reshape(batchSize, 1, 1)
            lv += current_lv
            deltaLvs[:,time:time+1, 0:1] = current_lv
            if time == 0:
                curTotalLv = pre_lv_act
            else:
                curTotalLv = totalLvs[:,time-1:time, 0:1]
            totalLvs[:,time:time+1, 0:1] = (curTotalLv + current_lv)
            sample_count += 1
        # if batch_first:
        #     tlvs = tlvs.reshape([tlvs.shape[1], tlvs.shape[0], -1])
        

        return deltaLvs, totalLvs