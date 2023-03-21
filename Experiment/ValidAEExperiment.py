from DatasetReader.ValidCastingDataReader import ValidCastingDataReader
from TaskConfig.AETaskConfig import AETaskConfig
from DataProcessor.DeviationProcessor import DeviationProcessor
from DatasetReader.CastingDataReader import CastingDataReader

from globalConfig import globalConfig

from DataProcessor.ShuffleDataProcessor import ShuffleDataProcessor
from DataProcessor.SlidingWindowStepDataProcessor import SlidingWindowStepDataProcessor

class ValidAEExperiment(object):
    def __init__(self, logger):
        self.logger = logger

    def getName(self):
        return "ValidAEExperiment"

    def getExperimentConfig(self):
        normalDataReader = ValidCastingDataReader("datasets")
        windowSize = 10
        config = AETaskConfig(self.logger, self.getName(), showTrainingInfo=True, windowSize = windowSize)
        trainer = config.getConfig()
        processers = [
            # LabelOffsetDataProcessor(windowSize),
            # PartitionDataProcessor(0.5),
            DeviationProcessor(),
            SlidingWindowStepDataProcessor(windowSize=windowSize, step=1),
            # ShuffleDataProcessor()
        ]
        return trainer, normalDataReader, normalDataReader, processers