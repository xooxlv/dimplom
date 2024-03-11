from logic.DataPreparer import DataPreparer

class HttpRecordsPreparer(DataPreparer):
    buff = []

    def get_all(self):
        return self.buff.copy()
    
    def prepare(self, data):
        self.buff.append(data)
    
    def flush(self):
        self.buff.clear()
    