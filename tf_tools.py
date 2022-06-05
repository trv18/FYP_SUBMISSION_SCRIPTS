import numpy


def CountParams(input_size, Layer_units, output_size, BatchNorm=True):
    Params = input_size*Layer_units[0] + Layer_units[0]
    if BatchNorm:
        Params += 4*Layer_units[0]
    for i in range(len(Layer_units)-1):
        Params += Layer_units[i]*Layer_units[i+1] + Layer_units[i+1]
        if BatchNorm:
            Params += 4*Layer_units[i+1]

    Params += output_size*Layer_units[-1] + output_size

    return Params