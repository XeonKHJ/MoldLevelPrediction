import matplotlib.pyplot
import os.path

import pandas as pd

class PlotCSVLogger():
    def __init__(self, isPlotEnable, picFolder, csvFolder):
        self.isPlotEnable = isPlotEnable
        self.picFolder = picFolder
        self.csvFolder = csvFolder

    def logSingleResult(self, data, label):
        _, ax = matplotlib.pyplot.subplots()
        ax.plot(data, label=label)
        ax.legend()
        if self.isPlotEnable:
            matplotlib.pyplot.show()
        matplotlib.pyplot.close()

    def logResults(self, datas, labels, picname=None, folderName = None):
        _, ax = matplotlib.pyplot.subplots()
        for i in range(len(datas)):
            ax.plot(datas[i], label = labels[i])
        ax.legend()
        if self.isPlotEnable:
            matplotlib.pyplot.show()
        if picname != None:
            if folderName == None:
                matplotlib.pyplot.savefig(os.path.join('SavedPics', picname))
                self.saveCsv(datas, labels, picname)
            else:
                savePath = os.path.join(folderName, picname)
                matplotlib.pyplot.savefig(savePath)
                self.saveCsv(datas, labels, picname)
            
            print(picname, " saved.")
        matplotlib.pyplot.close()

    def logResult(self, ogData, predictData):
        fig, ax = matplotlib.pyplot.subplots()
        ax.plot(ogData, label="dataset")
        ax.plot(predictData, label="predict")
        ax.legend()
        if self.isPlotEnable:
            matplotlib.pyplot.show()
        matplotlib.pyplot.close()

    def saveCsv(self, datas, titles, name):
        toSaveData = dict()
        for idx, title in enumerate(titles):
            toSaveData[title] = datas[idx]
        df = pd.DataFrame(toSaveData)
        df.to_csv(os.path.join(self.csvFolder, name+".csv"), index=False)
        