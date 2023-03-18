from AETaskConfig import AETaskConfig
from DataProcessor.DeviationProcessor import DeviationProcessor
from DatasetReader.CastingDataReader import CastingDataReader
from FEGAETaskConfig import FEGAETaskConfig

from globalConfig import globalConfig

from DataProcessor.ShuffleDataProcessor import ShuffleDataProcessor
from DataProcessor.SlidingWindowStepDataProcessor import SlidingWindowStepDataProcessor

class AECastingExperiment(object):
    def __init__(self, logger):
        self.logger = logger

    def getName(self):
        return "AECastingExperiment"

    def getExperimentConfig(self):
        normalDataReader = CastingDataReader("datasets")
        config = AETaskConfig(self.logger, self.getName(), showTrainingInfo=True)
        trainer = config.getConfig()
        windowSize = 100
        processers = [
            # LabelOffsetDataProcessor(windowSize),
            # PartitionDataProcessor(0.5),
            DeviationProcessor(),
            SlidingWindowStepDataProcessor(windowSize=windowSize, step=1),
            # ShuffleDataProcessor()
        ]
        return trainer, normalDataReader, normalDataReader, processers