from os import path
from pickletools import optimize

from numpy import reshape
from Dataset.AnomalyDetectDataset import AnomalyDetectDataset
from DataProcessor.DeviationProcessor import DeviationProcessor
from DataProcessor.ExtractValidProcessor import ExtractValidProcessor
from DatasetReader.CastingDataReader import CastingDataReader
from DatasetReader.CompleteCastingDataReader import CompleteCastingDataReader
from DatasetReader.ValidCastingDataReader import ValidCastingDataReader
from Models.DLModel import DLModel
# from liner_param_model import LinerParamModel
# from param_model import ParamModel
import torch
import torch.nn as nn
import torch.optim

from Logger.PlotLogger import PlotLogger
from globalConfig import globalConfig
from loss.dilate_loss import dilate_loss

B = 1250  # 连铸坯宽度
W = 230  # 连铸坯厚度
L = 1  # 结晶器内液面高度
c2h = 1  # c2(h)：流量系数
A = 11313  # 下水口侧孔面积
Ht = 10  # 计算水头高
H1t = 1  # 中间包液面高度
H2 = 1300  # 下水口水头高度
H3 = 2  # 下侧孔淹没高度，需要计算
h = 1  # 塞棒高度

datasetReader = ValidCastingDataReader("./datasets", 'ML202205270218-1')
ogDatasetReader = CompleteCastingDataReader("./datasets", 'ML202205270218-1')

def convert_result_to_csv_str(hs, lv_acts, lv_preds):
    print("STP_POS,LV_ACT,LV_PRED")
    for i in range(hs.__len__()):
        print(str(hs[i])+","+str(lv_acts[i])+","+str(lv_preds[i]))


def saveCsv(data, name):
    datastr = ""
    for d in data:
        datastr += (str(d) + '\n')
    with open(name+'.csv', 'w') as output:
        output.write(datastr)

def saveCsvMuti(datas, names, filename):
    datastr = ""
    for idx, name in enumerate(names):
        datastr += name
        if idx != len(names) - 1:
            datastr += ','
        else:
            datastr += '\n'
    maxLen = 0
    for data in datas:
        maxLen = max(len(data), maxLen)
    for idx in range(maxLen):
        for dataIdx, data in enumerate(datas):
            datastr += str(data[idx])
            if dataIdx < len(datas) - 1:
                datastr += ','
            else:
                datastr += '\n'
    with open(filename+'.csv', 'w') as output:
        output.write(datastr)
            


if __name__ == '__main__':
    predictModel = DLModel()
    # predictModel.load_state_dict(torch.load(path.join('dlmodel.pt')))
    predictModel.cuda()
    ogDataTensor, ogDataLenghts, ogDataLabels, context, fileList = datasetReader.read()
    completeDataTensor, completeDataLength, _,completeContext,_ = ogDatasetReader.read()
    
    startIdx, endIdx= datasetReader.getStartIdx()

    logger = PlotLogger(False)
    dataProcessors = [DeviationProcessor()]
    for dp in dataProcessors:
        fullDataTensor, fullDataLenghts, context = dp.process(ogDataTensor, ogDataLenghts, context)

    for idx, dlv in enumerate(fullDataTensor):
        curList = dlv[0:fullDataLenghts[idx].int().item(),1].reshape(-1).tolist()
        logger.logResults([curList], ['lv_act'], 'lv-' + path.splitext(path.basename(fileList[idx]))[0], globalConfig.getOriginalPicturePath())
        stoper = dlv[0:fullDataLenghts[idx].int().item(),0].reshape(-1).tolist()
        logger.logResults([stoper], ['stoper'], 'stp-' + path.splitext(path.basename(fileList[idx]))[0], globalConfig.getOriginalPicturePath())

    epoch = 0
    hs = fullDataTensor[:,:,0:1]
    ols = ogDataTensor[:,:,1:2]
    ls = fullDataTensor[:,:,1:2]
    loss_function = nn.MSELoss()
    preLv = ls[:,0:1,:] + 0
    optimizer = torch.optim.Adam(predictModel.parameters(), 1e-2)
    offset = 6
    zeros = torch.zeros( ols[:,offset:ls.shape[1],:].shape, device=torch.device('cuda'))
    losses = []
    while True:
        predictModel.train()
        epoch += 1
        totalLv, theoryLv, noise_out = predictModel(hs, context, fullDataLenghts, preLv)
        mseloss2 = loss_function(totalLv, ls[:,offset:ls.shape[1],:])
        noiseLoss = loss_function(noise_out,zeros)
        loss = mseloss2 + noiseLoss
        losses.append(loss.item())
        loss.backward()
        print('\tdilate', loss.item())
        optimizer.step()
        optimizer.zero_grad()

        if epoch % 100 == 0:
            predictModel.eval()
            saveCsvMuti([losses], ['loss'], 'predictLosses.csv')
            # newHs = torch.zeros(hs.shape, device=torch.device('cuda'))
            # newHs[0:1,:,:] = hs
            # newHs[0:1,40:42,:] = 20
            # newTotalLv, newTheoryLv, newNoiseOut = predictModel(newHs, context, fullDataLenghts, preLv)
            for idx, dlv in enumerate(totalLv):
                torch.save(predictModel.state_dict(), path.join('dlmodel.pt'))
                outputlist = [
                    totalLv[idx].reshape([-1]).tolist(),
                    theoryLv.reshape([-1]).tolist(),
                    noise_out[idx].reshape([-1]).tolist(),
                    ogDataTensor[idx, offset:ls.shape[1], 1].reshape([-1]).tolist(),
                    ls[idx, offset:ls.shape[1]].reshape([-1]).tolist(),
                    hs[idx, 0:hs.shape[1] - offset].reshape([-1]).tolist()
                ]
                nameList = [
                    'total',
                    'grow',
                    'noise',
                    'truth',
                    'dl',
                    'hs'
                ]
                saveCsvMuti(outputlist, nameList, path.join('Saved', 'PredictResult', 'compare-' + str(idx)))

            # for idx, dlv in enumerate(totalLv):
            #     torch.save(predictModel.state_dict(), path.join('dlmodel.pt'))
            #     outputlist = [
            #         newTotalLv[idx].reshape([-1]).tolist(),
            #         newTheoryLv.reshape([-1]).tolist(),
            #         newNoiseOut[idx].reshape([-1]).tolist(),
            #         ogDataTensor[idx, offset:ls.shape[1], 1].reshape([-1]).tolist(),
            #         ls[idx, offset:ls.shape[1]].reshape([-1]).tolist(),
            #         newHs[idx, 0:hs.shape[1] - offset].reshape([-1]).tolist()
            #     ]
            #     nameList = [
            #         'total',
            #         'grow',
            #         'noise',
            #         'truth',
            #         'dl',
            #         'hs'
            #     ]
            #     saveCsvMuti(outputlist, nameList, path.join('Saved', 'PredictResult', 'compare-fake-' + str(idx)))


