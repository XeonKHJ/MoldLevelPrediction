import os.path as path

class globalConfig():

    def __init__(self) -> None:
        self.savedPath = 'Saveds'

    @staticmethod
    def getModelPath():
        return path.join('Saved', "SavedModels") 

    @staticmethod
    def getSavedPicturePath():
        return path.join('Saved', "SavedPics")

    @staticmethod
    def getOriginalPicturePath():
        return path.join('Saved', "OgPics")
    
    @staticmethod
    def getCsvPath():
        return path.join('Saved', "SavedCsvs")