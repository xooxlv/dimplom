from abc import ABC, abstractmethod

class ReportHandler():
    @abstractmethod
    def handle_report(self, report):
        pass