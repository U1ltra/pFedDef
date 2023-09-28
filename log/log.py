
import os

class log:
    def __init__(self, base_path):
        self.base_path = base_path

        # path_log file to pass the model chkpts to the evaluation script
        self.path_log = base_path + '/path_log'
        self.path_log_file = open(self.path_log, 'w')
    
    def __del__(self):
        self.path_log_file.close()

    def write_path_log(self, message):
        self.path_log_file.write(message + '\n')

    def save_model(self, agg, save_root):
        os.makedirs(save_root, exist_ok=True)
        agg.save_state(save_root)

