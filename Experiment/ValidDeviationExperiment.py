from DataProcessor.DeviationProcessor import DeviationProcessor
from DatasetReader.CastingDataReader import CastingDataReader
from DatasetReader.ValidCastingDataReader import ValidCastingDataReader
from TaskConfig.FEGAETaskConfig import FEGAETaskConfig

from globalConfig import globalConfig

from DataProcessor.ShuffleDataProcessor import ShuffleDataProcessor
from DataProcessor.SlidingWindowStepDataProcessor import SlidingWindowStepDataProcessor

class ValidDeviationExperiment(object):
    def __init__(self, logger):
        self.logger = logger

    def getName(self):
        return "ValidDeviationExperiment"

    def getExperimentConfig(self):
        normalDataReader = ValidCastingDataReader("datasets")
        windowSize = 6
        config = FEGAETaskConfig(self.logger, self.getName(), showTrainingInfo=True, windowSize=windowSize)
        trainer = config.getConfig()
        
        processers = [
            DeviationProcessor(),
            SlidingWindowStepDataProcessor(windowSize=windowSize, step=1),
        ]
        return trainer, normalDataReader, normalDataReader, processers