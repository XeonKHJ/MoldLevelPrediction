import pandas
import torch
import os.path as path

# Only reads data that has valid mold level info.
class ValidCastingDataReader():
    def __init__(self, datasetPath, name = '') -> None:
        self.datasetPath = datasetPath
        self.contextName = "context.csv"
        self.includeName = name

    def read(self):
        contexts = pandas.read_csv(path.join(self.datasetPath, self.contextName))
        datas = list()
        stoperPosDatas = list()
        filenames = contexts['filename'].tolist()
        newFilenames = list()
        for filename in filenames:
            if filename.__contains__(self.includeName):
                newFilenames.append(filename)
        filenames = newFilenames
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
                    self.startIdx = startIdx
                    break

            for idx, es in enumerate(extractSpeed):
                if es > 0.01 and idx > startIdx:
                    endIdx = idx
                    self.endIdx = endIdx
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
        
    def getStartIdx(self):
        return self.startIdx, self.endIdx