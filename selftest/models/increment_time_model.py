import sys
from class_model import Model
from time import sleep

class Model(Model):
    """Increment time model for testing"""
    def __init__(self):
        super().__init__()
        self.model = None
        self.default_delays = {'init': 0.1, 'prepare-batch': 0.1, 'read': 0.1, 'read1st': 0.1, 'prepare': 0.1, 'inference': 0.1, 'shutdown': 0.1}
        self.delays = self.default_delays.copy()
        self.increment = 0.01
        self.set_increment(sys.argv)
        self.set_delays(sys.argv)
    def set_delays(self, args):
        if '--delay_all' in args:
            for key in self.delays:
                self.delays[key] = float(args[args.index('--delay_all') + 1])
        for arg in args:
            if arg.startswith('--delay-'):
                try:
                    self.delays[arg[len('--delay-'):]] = float(args[args.index(arg) + 1])
                except Exception as e:
                    print(f"Error parsing delay {arg}: {e}, using default {self.default_delays[arg[len('--delay-'):]]}")
                    self.delays[arg[len('--delay-'):]] = self.default_delays[arg[len('--delay-'):]]
        self.default_delays = self.delays.copy()
    def set_increment(self, args):
        if '--increment' in args:
            self.increment = float(args[args.index('--increment') + 1])
    def reset_inference_run(self):
        super().reset_inference_run()
        self.delays = self.default_delays.copy()
    def prepare_batch(self, batch_size):
        sleep(self.delays['prepare-batch'])
        self.delays['prepare-batch'] += self.increment
    def read1st(self):
        sleep(self.delays['read1st'])
        self.delays['read1st'] += self.increment
    def read(self):
        sleep(self.delays['read'])
        self.delays['read'] += self.increment
    def prepare(self):
        sleep(self.delays['prepare'])
        self.delays['prepare'] += self.increment
    def inference(self):
        sleep(self.delays['inference'])
        self.delays['inference'] += self.increment
    def shutdown(self):
        sleep(self.delays['shutdown'])
        self.delays['shutdown'] += self.increment
