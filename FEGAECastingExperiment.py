from DatasetReader.CastingDataReader import CastingDataReader
from FEGAETaskConfig import FEGAETaskConfig

from globalConfig import globalConfig

from DataProcessor.ShuffleDataProcessor import ShuffleDataProcessor
from DataProcessor.SlidingWindowStepDataProcessor import SlidingWindowStepDataProcessor

class FEGAECastingExperiment(object):
    def __init__(self, logger):
        self.logger = logger

    def getName(self):
        return "FEGAECastingExperiment"

    def getExperimentConfig(self):
        normalDataReader = CastingDataReader("datasets")
        config = FEGAETaskConfig(self.logger, self.getName(), showTrainingInfo=True)
        trainer = config.getConfig()
        windowSize = 20
        processers = [
            # LabelOffsetDataProcessor(windowSize),
            # PartitionDataProcessor(0.5),
            SlidingWindowStepDataProcessor(windowSize=windowSize, step=1),
            # ShuffleDataProcessor()
        ]
        return trainer, normalDataReader, normalDataReader, processers