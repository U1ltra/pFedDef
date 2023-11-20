
import torch
import captum
import time
import os

class evalInternalInfluence:

    def __init__(self, model, data_iterator) -> None:
        self.model = model
        self.data_iterator = data_iterator # TODO: extend to multiple data iterators
        self.layer_names = []
        self.layers = []
        self.layer_influence = {}

        for name, module in self.model.named_modules():
            self.layer_names.append(name)
            self.layers.append(module)
            self.layer_influence[name] = 0

    def reset_influence(self):
        for name in self.layer_names:
            self.layer_influence[name] = None

    def eval_influence(self):
        
        for layer in self.model_layers():
            layer_int_infl = captum.attr.InternalInfluence(self.model, layer=layer[1])
            # num of batch = Total number of data points / batch size
            for batch_id, (x, y, indices) in enumerate(self.data_iterator): 
                # send to GPU
                x = x.cuda()
                y = y.cuda()

                layer_int_infl_vals = layer_int_infl.attribute(x, target=y)
                if self.layer_influence[layer[0]] is None:
                    self.layer_influence[layer[0]] = layer_int_infl_vals
                self.layer_influence[layer[0]] += layer_int_infl_vals # TODO: do I need .abs()?
                print(f'Layer [{layer[0]}] influence: {self.layer_influence[layer[0]]}')

    def model_layers(self):
        return zip(self.layer_names, self.layers)

    def print_influence(self, layer):
        print(f'Layer [{layer}] influence:')
        print(self.layer_influence[layer])
 
    def print_layer_names(self):
        print("Layer names:")
        for layer in self.layer_names:
            print(layer)

    def save_influence(self, save_path,):
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        for layer in self.layer_names:
            if os.path.exists(f'{save_path}/influence_{layer}.pt'):
                print(f'File {save_path}/influence_{layer}.pt already exists. Skipping.')
                continue

            torch.save(self.layer_influence[layer], f'{save_path}/influence_{layer}.pt')