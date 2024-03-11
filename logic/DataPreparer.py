from abc import ABC, abstractmethod

class DataPreparer(ABC):
    @abstractmethod
    def prepare(self, data):
        pass

    @abstractmethod
    def get_all(self):
        pass

    @abstractmethod
    def flush(self):
        pass
