from logic.DataReader import DataReader, InputSource
from logic.DataParser import DataParser
from logic.DataPreparer import DataPreparer
from logic.DataAnalizer import DataAnalizer
from logic.ReportBuilder import ReportBuilder
from logic.ReportHandler import ReportHandler
from control.Client import ControlCenterClient


from log import log

reader = None
parser = None
preparer = None
neuronet = None
reporter = None
handler = None
client = None

def init_module(data_source: InputSource,
                data_parser: DataParser,
                data_preparer: DataPreparer,
                neuron_network: DataAnalizer,
                report_builder: ReportBuilder,
                report_handler: ReportHandler,
                control_client: ControlCenterClient):
    
    global reader, parser, preparer, neuronet, reporter, handler, client
    
    reader = DataReader(data_source) # проверить, что все ок
    parser = data_parser
    preparer = data_preparer
    neuronet = neuron_network
    reporter = report_builder
    handler = report_handler
    client = control_client

def run_module():
    if client.connect() == 'DENY':
        log('Отказано в подключении', lvl='fatal')
        exit()

    reader.start()
    while client.is_work() == 'True':
        log('Модуль работает нормально', lvl='info')
        raw_data = reader.get_new_records(1)
        for raw in raw_data:
            parsed_data = parser.parse_data(raw)
            for record in parsed_data:
                preparer.prepare(record)
        
        report = None
        mode = client.get_mode()
        log('Установленный режим работы:', mode, lvl='warn')
        if mode == 'train':
            y = client.get_y()
            if y != '':
                report = neuronet.train(preparer.get_all(), client.get_y())
                preparer.flush()
            else:
                log('Не установлена метка класса в модуле управления', lvl='error')
        elif mode == 'analize':
            report = neuronet.analize(preparer.get_all())
            preparer.flush()
        else:
            log('Неизвестный режим работы:', mode, lvl='error')

        report = reporter.make_report(report)
        handler.handle_report(report)
