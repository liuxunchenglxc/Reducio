import os
import subprocess


def gene_to_code(gene_units_params):
    gene_units_params = optim_gene_code(gene_units_params)
    gene_length = 0
    space_8 = "        "
    # Code class head
    class_head = '''class GEN_SR_MODEL(Model):
    def __init__(self):
        super(GEN_SR_MODEL, self).__init__()'''
    # Gene scan
    # Name the pos=p CNN as conv_{p}
    conv_layers = ""
    # Name the pos=p output as x_{p}
    forward_code = ""
    pos = 0
    for unit_params in gene_units_params:
        gene_length = gene_length + 1
        if unit_params[0] == '-1' or unit_params[0] == '-2':  # Useless Unit
            pos = pos + 1
            gene_length = gene_length - 1
            continue
        if unit_params[0] == '0':  # Start Unit
            pos = pos + 1
            continue
        if unit_params[0] == '255':  # End Unit
            forward_code = forward_code + space_8 + "x = tf.nn.depth_to_space(x_{}, 4)\n".format(pos - 1)
            forward_code = forward_code + space_8 + "return x\n"
            break
        input_num = int(unit_params[0])
        if input_num == 1:
            input_name = 'x_' + str(max(0, pos - int(unit_params[1])))
            unit_type = int(unit_params[2])
            if unit_type == 1:  # CNN
                activation_num = int(unit_params[6])
                if activation_num == 1:
                    activation = "\"relu\""
                elif activation_num == 2:
                    activation = "\"sigmoid\""
                else:
                    activation = "None"
                conv_layers = conv_layers + space_8 + "self.conv_{} = Conv2D({}, {}, padding=\"same\", activation={})\n".format(
                    pos, unit_params[3], unit_params[4], activation)
                forward_code = forward_code + space_8 + "x_{} = self.conv_{}({})\n".format(pos, pos, input_name)
        elif input_num == 2:
            input_name_a = 'x_' + str(max(0, pos - int(unit_params[1])))
            input_name_b = 'x_' + str(max(0, pos - int(unit_params[2])))
            unit_type = int(unit_params[3])
            if unit_type == 2:  # +
                forward_code = forward_code + space_8 + "x_{} = {} + {}\n".format(pos, input_name_a, input_name_b)
            elif unit_type == 3:  # *
                forward_code = forward_code + space_8 + "x_{} = {} * {}\n".format(pos, input_name_a, input_name_b)
        pos = pos + 1
    # Link code
    code = "{}\n{}\n    @tf.function\n    def call(self, x_0):\n{}".format(class_head, conv_layers, forward_code)
    return code, gene_length


def optim_gene_code(gene_units_params):
    id_pos_mark = [[0, 1]]
    id = 0
    # scan for pos
    for unit_params in gene_units_params:
        if unit_params[0] == '0':
            id = id + 1
            continue
        if unit_params[0] == '255':
            id_pos_mark[-1][1] = 1
            break
        id_pos_mark.append([id, 0])
        id = id + 1
    # mark input pos and mark useless unit
    offset_base = 0
    for i in range(len(gene_units_params) - 1, -1, -1):
        if gene_units_params[i][0] == '0':
            break
        if gene_units_params[i][0] == '255':
            continue
        try:
            offset_base = offset_base + 1
            if gene_units_params[i][int(gene_units_params[i][0]) + 1] == '5':
                offset_base = offset_base + int(gene_units_params[i][3]) - 1
            id_pos_mark.index([i, 1])  # marked unit has the quality to mark other units
            for j in range(int(gene_units_params[i][0])):
                offset = int(gene_units_params[i][1 + j])
                id_pos_mark[max(-offset - offset_base, -len(id_pos_mark))][1] = 1
        except ValueError:  # Unit output only use by after unit, so this unit is useless
            if gene_units_params[i][0] == '2':
                gene_units_params[i][0] = '-2'
            else:
                gene_units_params[i][0] = '-1'
    return gene_units_params


def code_to_file(id: int, code: str):
    file_head = '''import tensorflow as tf

from tensorflow.keras.layers import Conv2D
from tensorflow.keras import Model'''
    file_content = file_head + "\n\n\n" + code
    if not os.path.exists("./models/"):
        os.mkdir("./models/")
        os.mknod("./models/__init__.py")
    with open('./models/model_{}.py'.format(id), 'w', encoding='utf-8') as f:
        f.write(file_content)
    return 'model_{}.py'.format(id), str(id)
