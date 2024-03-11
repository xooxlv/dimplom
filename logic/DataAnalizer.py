from abc import ABC, abstractmethod


# класс для анализа данных в нейросети
class DataAnalizer(ABC):

    @abstractmethod
    def train(self, input, y):
        pass

    @abstractmethod
    def analize(self, input):
        pass

    @abstractmethod
    def store(self):
        pass

    @abstractmethod
    def load(self):
        pass