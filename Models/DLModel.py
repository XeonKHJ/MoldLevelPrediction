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


    def forward(self, x, ts, phs, pre_lv_act):
        lstm_out, (h,c) = self.lstm(x)  # _x is input, size (seq_len, batch, input_size)
        # reshape_lstm_out = h.reshape([-1])
        foward_out = self.forwardCalculation(lstm_out)
        foward_out = self.calculate_lv_acts_tensor(x[0], ts, foward_out, x.shape[0], phs.__len__()*0.5, pre_lv_act=pre_lv_act)

        noise_out, (h, c) = self.noiseLstm(x)
        # firstNoiseOutput, (h,c) = self.noiseLstm(h[-1,:,:].reshape([-1,1,1]), (h, c))
        noiseOutput = torch.zeros([x.shape[0],0,1])
        for i in range(x.shape[1]):
            singleNoiseOutput, (h, c) = self.noiseLstm(h[-1,:,:].reshape([-1,1,1]), (h, c))
            noiseOutput = torch.cat((noiseOutput, singleNoiseOutput), 1)
        noiseOutput = self.noiseFc(noiseOutput)
        
        finalOutput = noiseOutput + foward_out

        return finalOutput, foward_out, noise_out
    

    def stp_pos_flow_tensor(self, h_act, lv_act, t, dt=0.5, params=[0,0,0,0]):
        H1t = params[0,0] # H1：中间包液位高度，t的函数，由LSTM计算
        g = 9.8                 # 重力
        # c2h = lpm(torch.tensor(h_act).reshape(-1))  # C2：和钢种有关的系数，由全网络计算
        c2h = params[0,1]

        # 引锭头顶部距离结晶器底部高度350+结晶器液位高度（距离引锭头）283
        if lv_act < 633:
            H3 = 0
        else:
            H3 = lv_act-633  # H3下侧出口淹没高度
        Ht = H1t+self.H2-H3
        dL = (pow(2 * g * Ht, 0.5) * c2h * self.A * dt) / (self.B * self.W)
        return dL

    def calculate_lv_acts_tensor(self, hs, ts, params, batch_size, batch_first = True, previousTime = 0, pre_lv_act = 0):
        sampleRate = 2  # 采样率是2Hz。
        # 维度为（时间，数据集数量，特征数）
        tlvs = torch.zeros([ ts.__len__(), batch_size, 1])
        lv = pre_lv_act
        sample_count = 0
        for stage in range(ts.__len__()):
            stopTimeSpan = ts[stage]
            if stage > 0:
                previousTime += ts[stage-1]
            for time in range(int(stopTimeSpan / 0.5)):
                current_lv = self.stp_pos_flow_tensor(hs[stage], lv, previousTime + time / 2, 1 / sampleRate, params[:, stage, :])
                # print(current_lv.reshape([-1]).item())
                lv += current_lv
                tlvs[sample_count] = lv
                sample_count += 1
        if batch_first:
            tlvs = tlvs.reshape([tlvs.shape[1], tlvs.shape[0], -1])
        return tlvs