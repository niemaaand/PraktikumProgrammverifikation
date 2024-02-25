import math
import os.path

import torch

from src.vnnlib.vnnlib import read_vnnlib_simple


class Properties:
    def __init__(self, vnnlibs_path, num_inputs, num_outputs):
        self.vnnlibs_path = vnnlibs_path
        self._lower_input_bounds, self._upper_input_bounds = self._calc_input_bounds(num_inputs, num_outputs)

    def _calc_input_bounds(self, num_inputs, num_outputs):

        if not self.vnnlibs_path or not os.path.exists(self.vnnlibs_path):
            return None, None

        if not isinstance(num_inputs, int):
            t = 1
            for i in num_inputs:
                t *= i

            num_inputs = t

        if not isinstance(num_outputs, int):
            t = 1
            for i in num_outputs:
                t *= i

            num_outputs = t

        # read vnnlibs
        vnnlib_specs = []
        if os.path.isfile(self.vnnlibs_path):
            vnnlib_specs.append(read_vnnlib_simple(self.vnnlibs_path, num_inputs, num_outputs))
        elif os.path.isdir(self.vnnlibs_path):
            for vnnlib_file in os.listdir(self.vnnlibs_path):
                if os.path.splitext(vnnlib_file)[-1] == ".vnnlib":
                    vnnlib_specs.append(read_vnnlib_simple(os.path.join(self.vnnlibs_path, vnnlib_file), num_inputs, num_outputs))

        # concat vnnlibs
        lower_concat_spec = [0.0 for i in range(num_inputs)]
        upper_concat_spec = [0.0 for i in range(num_inputs)]

        first_run = True
        for spec in vnnlib_specs:
            for input_box, spec_list in spec:

                assert len(input_box) == num_inputs, "Length of inputs does not match."

                neuron_idx = 0
                for input_neuron in input_box:

                    if input_neuron[0] < lower_concat_spec[neuron_idx] or first_run:
                        lower_concat_spec[neuron_idx] = input_neuron[0]

                    if input_neuron[1] > upper_concat_spec[neuron_idx] or first_run:
                        upper_concat_spec[neuron_idx] = input_neuron[1]

                    neuron_idx += 1
            first_run = False

        lower_concat_spec = torch.flatten(torch.FloatTensor(lower_concat_spec))
        upper_concat_spec = torch.flatten(torch.FloatTensor(upper_concat_spec))

        assert (lower_concat_spec <= upper_concat_spec).sum() == len(lower_concat_spec)

        return lower_concat_spec, upper_concat_spec

    def calc_random_input_vector(self, size):

        rand_vec = (torch.rand(size))

        if self._lower_input_bounds is None or self._upper_input_bounds is None:
            return rand_vec

        assert len(self._lower_input_bounds) == len(self._upper_input_bounds), "Sizes of upper und lower bounds for neurons are not the same."

        for batch_idx in range(size[0]):
            rand_vec_flattened = torch.flatten(rand_vec[batch_idx])

            if not rand_vec_flattened.size()[0] == len(self._lower_input_bounds):
                print("Input sizes not matching.")
                return rand_vec

            rand_vec_flattened = rand_vec_flattened * (self._upper_input_bounds - self._lower_input_bounds) + self._lower_input_bounds

            # assert all values are between lower/upper input bounds
            assert (self._lower_input_bounds <= rand_vec_flattened).sum() == len(self._upper_input_bounds) and (rand_vec_flattened <= self._upper_input_bounds).sum() == len(self._upper_input_bounds), "Wrong formula in source code"

            rand_vec[batch_idx] = torch.unflatten(rand_vec_flattened, dim=0, sizes=size[1:])

        return rand_vec



