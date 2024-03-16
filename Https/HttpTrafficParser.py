from logic.DataParser import DataParser
import json

class HttpTrafficParser(DataParser):
    def parse_data(self, data) :
        re = []
        for line in data.split('\n'):
            if line != '':
                record = eval(line)
                del record['cookies']
                re.append(record)
        return re