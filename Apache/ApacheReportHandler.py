from logic.ReportHandler import ReportHandler

class ApacheReportHandler(ReportHandler):

    instrument = None

    def handle_report(self, report):
        self.instrument.send_report(report)
    
    def set_handle_instrument(self, instrument):
        self.instrument = instrument