from dataclasses import dataclass
from typing import List

import numpy as np

from src.core.converters.string_converters import json_to_dictionary


@dataclass
class EMGPayload:
    ch1: float
    ch2: float
    ch3: float
    ch4: float

    @staticmethod
    def from_json(json):
        data = json_to_dictionary(json)
        if data and data['type'] == 'emg':
            ch1, ch2, ch3, ch4 = data['data']
            return EMGPayload(ch1, ch2, ch3, ch4)
        raise ValueError(f'EMGPayload: Failed to parse. Reason: Invalid Json')

    @staticmethod
    def avg(data_list):
        if len(data_list) == 0:
            return EMGPayload(0, 0, 0, 0)
        ch1_sum = 0
        ch2_sum = 0
        ch3_sum = 0
        ch4_sum = 0
        for data in data_list:
            ch1_sum += data.ch1
            ch2_sum += data.ch2
            ch3_sum += data.ch3
            ch4_sum += data.ch4
        return EMGPayload(ch1_sum / len(data_list), ch2_sum / len(data_list), ch3_sum / len(data_list),
                          ch4_sum / len(data_list))

    @staticmethod
    def flatten(data_list):
        if len(data_list) == 0:
            return np.array([[0, 0, 0, 0]])
        input = []
        for data in data_list:
            input.append([data.ch1, data.ch2, data.ch3,data.ch4])

        return np.array(input)
