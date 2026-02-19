from class_model import Model

class Model(Model):
    """Noisy model for testing"""
    def __init__(self):
        super().__init__()
        self.model = None
        print('{ "Noisy model" : "__init__()" },')
    def prepare_batch(self, batch_size):
        print(f'{{ "Noisy model" : "prepare_batch({batch_size})" }},')
    def read(self):
        print('{ "Noisy model" : "read()" },')
    def prepare(self):
        print('{ "Noisy model" : "prepare()" },')
    def inference(self):
        print('{ "Noisy model" : "inference()" },')
    def shutdown(self):
        print('{ "Noisy model" : "shutdown()" },')
