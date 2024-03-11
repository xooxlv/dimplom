from logic.DataParser import DataParser
import json

class HttpTrafficParser(DataParser):
    def parse_data(self, data) :
        re = []
        for line in data.split('\n'):
            if line != '':
                re.append({'path' : eval(line)['path']})
        return re