# Copyright 2024 Ezzat Esam
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from copy import deepcopy

import torch as T 

class EmaUpdater :
    def __init__(self, model, beta , start_after : int = 10):
        self.beta = beta
        self.model = deepcopy(model).eval().requires_grad_(False)
        self.steps = 0
        self.start_after = start_after

    def update(self, current_model : T.nn.Module):
        """
        Updates the exponential moving average (EMA) of the model parameters.

        Args:
            current_model (nn.Module): The current model whose parameters will be used to update the EMA.

        Returns:
            None
        """
        if self.steps < self.start_after:
            self.steps += 1
        else :
            for current_param, old_param in zip(current_model.parameters(), self.model.parameters()):
                current_param.data = current_param.data * (1-self.beta) + old_param.data * self.beta
                
        self.model.load_state_dict(current_model.state_dict()) # update the old model to the current model
