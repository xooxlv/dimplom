from queue import Queue
from logic.DataReader import InputSource
from log import log

# класс-посредник для чтения файла
class ApacheLogFile(InputSource):
    def __init__(self, input_path):
        super().__init__()
        self.__input = input_path
        self.last_position = 0

    def read_from_file(self):
        #log('Read from file:', 'position=', self.last_position, lvl='debug')
        with open(self.__input, 'r') as file:
            file.seek(self.last_position)
            raw_data = file.read()
            #log('Readen from file:', raw_data, lvl='debug')
            current_position = self.last_position
            self.last_position = file.tell()
            if current_position == self.last_position:
                return None
            return raw_data

    def read_data(self):
        raw_data = self.read_from_file()
        #log('Readen from file:', raw_data, lvl='debug')
        return raw_data
