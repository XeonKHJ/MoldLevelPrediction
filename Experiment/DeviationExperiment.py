from DataProcessor.DeviationProcessor import DeviationProcessor
from DatasetReader.CastingDataReader import CastingDataReader
from TaskConfig.FEGAETaskConfig import FEGAETaskConfig

from globalConfig import globalConfig

from DataProcessor.ShuffleDataProcessor import ShuffleDataProcessor
from DataProcessor.SlidingWindowStepDataProcessor import SlidingWindowStepDataProcessor

class DeviationExperiment(object):
    def __init__(self, logger):
        self.logger = logger

    def getName(self):
        return "DeviationExperiment"

    def getExperimentConfig(self):
        normalDataReader = CastingDataReader("datasets")
        config = FEGAETaskConfig(self.logger, self.getName(), showTrainingInfo=True, windowSize=20)
        trainer = config.getConfig()
        windowSize = 20
        processers = [
            DeviationProcessor(),
            SlidingWindowStepDataProcessor(windowSize=windowSize, step=1),
        ]
        return trainer, normalDataReader, normalDataReader, processers