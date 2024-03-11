from abc import ABC, abstractmethod

class DataParser(ABC):
    @abstractmethod
    def parse_data(self, data) -> []:
        """
        Метод для парсинга данных.
        
        :param data: Данные, которые требуется распарсить.
        :return: Результат парсинга.
        """
        pass
