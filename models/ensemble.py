import numpy as np

import torch

class Ensemble():

    def __len__(self, inputs,labels,type="Mean"):
        """
        :param inputs: param inputs: [[model A], [model B] ...], output lists of models
        :param labels: the same shape with model A
        :param type: type of ensemble, eg. Mean\ weight_mean\ vote...
        :return:
        """
        super(Ensemble, self).__init__()
        self.inputs  = inputs
        self.nums = len(inputs)
        self.feature_len = len(inputs[0][0])
        self.labels += labels

        self.ensemble_output = torch.zeros(self.inputs[0].shape)
        self.acc = 0


    def mean(self):

        for input in self.inputs:
            self.ensemble_output += input

        return self.ensemble_output/self.nums

    def weight_mean(self, weights):

        for input,weight in zip(self.inputs,weights):
            self.ensemble_output += input * weight

        return self.ensemble_output

    def eval(self):
        True_Predict = 0
        _, predicted = torch.max(self.ensemble_output, 1)
        # bbox = Decode(torch.Tensor(Mean_Box), outs[:, 2:])
        True_Predict += (predicted == self.labels).sum().item()
        self.acc = 1.0 * True_Predict/len(self.inputs[0])

    def vote(self):
        pass


if __name__ == "__main__":
    A = [0.1667, 0.4459, 0.14209, 0.1531]
    B = [0, 1, 2, 3]

    for a,b in zip(A,B):
        print(a,b)
