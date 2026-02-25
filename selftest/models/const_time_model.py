import sys
from time import sleep

from class_model import Model


class Model(Model):
    """Const time model for testing"""

    def __init__(self):
        super().__init__()
        self.model = None
        self.delays = {
            'init': 0.1,
            'prepare-batch': 0.1,
            'read': 0.1,
            'read1st': 0.1,
            'prepare': 0.1,
            'inference': 0.1,
            'shutdown': 0.1,
        }
        self.set_delays(sys.argv)

    def set_delays(self, args):
        if '--delay_all' in args:
            for key in self.delays:
                self.delays[key] = float(args[args.index('--delay_all') + 1])
        for arg in args:
            if arg.startswith('--delay-'):
                try:
                    print(arg[len('--delay-') :])
                    self.delays[arg[len('--delay-') :]] = float(args[args.index(arg) + 1])
                except Exception as e:
                    print(f'Error parsing delay {arg}: {e}, using default 0.1')
                    self.delays[arg[len('--delay-') :]] = 0.1

    def prepare_batch(self, batch):
        sleep(self.delays['prepare-batch'])

    def read1st(self):
        sleep(self.delays['read1st'])

    def read(self):
        sleep(self.delays['read'])

    def prepare(self):
        sleep(self.delays['prepare'])

    def inference(self):
        sleep(self.delays['inference'])

    def shutdown(self):
        sleep(self.delays['shutdown'])
