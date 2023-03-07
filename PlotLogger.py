import matplotlib.pyplot
import os.path

class PlotLogger():
    def __init__(self, isPlotEnable):
        self.isPlotEnable = isPlotEnable
        pass

    def logInfo(self, folder):
        pass

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
            else:
                savePath = os.path.join(folderName, picname)
                matplotlib.pyplot.savefig(savePath)
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