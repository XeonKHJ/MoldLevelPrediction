import pandas
import torch
import os.path as path

# Only reads data that has valid mold level info.
class ValidCastingDataReader():
    def __init__(self, datasetPath) -> None:
        self.datasetPath = datasetPath
        self.contextName = "context.csv"

    def read(self):
        contexts = pandas.read_csv(path.join(self.datasetPath, self.contextName))
        datas = list()
        stoperPosDatas = list()
        filenames = contexts['filename'].tolist()
        for fileIdx, filename in enumerate(filenames):
            startIdx = 0
            endIdx = 0

            fileData = pandas.read_csv(path.join(self.datasetPath, filename))
            lv_acts = fileData['ActualLevel'].tolist()
            targetLv = float(fileData['TargetLevel'].tolist()[0])
            stoperPos = fileData['StoperPo'].tolist()
            extractSpeed = fileData['CSTSpeed'].tolist()
            stoperPosDatas.append(stoperPos)
            
            for idx, lv in enumerate(lv_acts):
                if lv > 1:
                    startIdx = idx
                    break

            for idx, es in enumerate(extractSpeed):
                if es > 0.01 and idx > startIdx:
                    endIdx = idx
                    break


            datas.append({'level':lv_acts[startIdx:endIdx], 'stoperPos': stoperPos[startIdx:endIdx], 'CSTSpeed':extractSpeed[startIdx:endIdx], 'thickness': contexts['thickness'][fileIdx], 'width': contexts['width'][fileIdx], 'steelType': contexts['steeltype'][fileIdx], 'targetLevel':targetLv})        

        datas.sort(key=(lambda elem:len(elem['level'])), reverse=True)
        
        lengths = torch.zeros([len(datas)])
        for dataIdx, data in enumerate(datas):
            lengths[dataIdx] = len(data['level'])
        maxLength = int(torch.max(lengths).item()) 

        dataTensor = torch.zeros([len(datas), maxLength, 3])
        labelTensor = torch.zeros([len(datas), maxLength, 1])
        
        # ignore steel type for now.
        contextTensor = torch.zeros([len(datas), len(datas[0]) - 2 - 1])
        for dataIdx, data in enumerate(datas):
            dataTensor[dataIdx, 0:len(data['stoperPos']), 0] = torch.tensor(data['stoperPos']).reshape([-1])
            dataTensor[dataIdx, 0:len(data['level']), 1] = torch.tensor(data['level']).reshape([-1])
            # dataTensor[dataIdx, 0:len(data['level']), 2] = torch.tensor(data['CSTSpeed']).reshape([-1])
            contextTensor[dataIdx, 0] = data['thickness']
            contextTensor[dataIdx, 1] = data['width']
            contextTensor[dataIdx, 2] = data['targetLevel']
            # contextTensor[dataIdx, 2] = data['thickness']

        if torch.cuda.is_available():
            return dataTensor.cuda(), lengths.cuda(), labelTensor.cuda(), contextTensor.cuda(), filenames
        else:
            return dataTensor, lengths, labelTensor, contextTensor, filenames
        
    def postProcess():
        # 液位能被检测之后的数据
        hs = list()
        ts = list()
        ls = list()

        # 液位能被检测之前的数据
        phs = list()
        pts = list()

        # completed data
        chs = list()
        cts = list()

        is_header_passed = False
        is_lv_detected = False  # 在结晶器中的钢液是否能被检测到
        ready_to_start = False

        sensor_to_dummy_bar_height = 350

        for line in lines:
            if is_header_passed:
                nums = line.split(',')
                current_l = float(nums[1])
                if is_lv_detected:
                    hs.append(float(nums[0]))
                    ls.append((float(nums[1]) + sensor_to_dummy_bar_height))
                    ts.append(0.5)
                    chs.append(float(nums[0]))
                    cts.append(0.5)
                if ready_to_start and not is_lv_detected:
                    pre_lv_act = float(nums[1]) + sensor_to_dummy_bar_height
                    is_lv_detected = True
                if current_l > 2:
                    ready_to_start = True
                else:
                    chs.append(float(nums[0]))
                    cts.append(0.5)
                    phs.append(float(nums[0]))
                    pts.append(0.5)
            else:
                is_header_passed = True
        return (hs, ls, ts, phs, pts, pre_lv_act, chs, cts)