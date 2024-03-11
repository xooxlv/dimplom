from logic.DataReader import *
import threading
from queue import Queue
from abc import ABC, abstractmethod
import time
from log import *


# тут содержится InputSource класс, который позволяет читать 
# данные откуда-то (переопределить read_data метод)

# DataReaderThread - класс для чтения InputSource объекта
# в отдельном потоке (не изменять)

# DataReader - класс для управлением DataReaderThread и для 
# получения данных прочитанных DataReaderThread из InputSource


# это интерфейс источника данных
# он должен возвращать новые сырые данные
# из реального источника данных (файла)
class InputSource(ABC):
    @abstractmethod
    def read_data(self): ...

# класс для чтения данных из InputSource и 
# помещения их в Queue в отдельном потоке
class DataReaderThread(threading.Thread):
    def __init__(self, input_source: InputSource, output_queue: Queue, reading_delay=1.0):
        super().__init__()
        self.input_source = input_source
        self.output_queue = output_queue
        self.reading_delay = reading_delay
        self._stop_event = threading.Event()

    def run(self):
        while not self._stop_event.is_set():
            data = self.input_source.read_data()
            if data is not None:
                #log('Run dataReaderThread method', 'new data len: ', len(data), lvl='debug')
                self.output_queue.put(data)
            time.sleep(self.reading_delay)

    def stop(self):
        self._stop_event.set()

# класс, для чтения данных из InputSource с использованием DataReaderThread
# для получения новых данных вызови get_new_records
class DataReader():
    def __init__(self, input_source: InputSource, reading_delay=1.0):
        self.output_queue = Queue()
        self.data_reader_thread = DataReaderThread(input_source, self.output_queue, reading_delay)

    def start(self):
        self.data_reader_thread.start()

    def stop(self):
        self.data_reader_thread.stop()
        self.data_reader_thread.join() 

    def get_new_records(self, count):
        to_return = []
        for _ in range(count):
            to_return.append(self.output_queue.get())
        return to_return
    