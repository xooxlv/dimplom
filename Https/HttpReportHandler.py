from logic.ReportHandler import ReportHandler

class HttpReportHandler(ReportHandler):
    def handle_report(self, report):
        self.cl.send_report(report)
    
    def set_client(self, clietn):
        self.cl = clietn