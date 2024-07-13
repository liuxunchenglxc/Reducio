import hashlib
import random
import time
from sqlalchemy.orm import Session

import ea_code_tf
import init_ea
import runtime_evaluation
from init_ea import BadCode, Code, CodeFile, Generation, GeneExpressed, Hash, Length, Main, Number, ResFile, Runtime
from init_ea import Status, Score, SR, Task, TaskEnd, TaskGet, WorkerRegister
from differentiable_search import differentiable_search


import score
from graph_seq import gene_graph_seq_with_info


def ea_select_generate(engine, iteration: int):
    # update iteration
    iteration = iteration + 1
    # select at least one couple
    while select_generate_gene(engine, iteration) == 0:
        pass
    # update status
    with Session(engine) as session:
        s = session.query(init_ea.Status).order_by(init_ea.Status.id.desc()).first()
        id = s.id + 1
        r = init_ea.Status(id=id, status=0, iteration=iteration)
        session.merge(r)
        session.commit()
    return iteration


def select_generate_gene(engine, iteration):
    genes = []
    couples = []
    # read fitness order
    with Session(engine) as session:
        s: init_ea.Number = session.query(init_ea.Number).order_by(init_ea.Number.id.desc()).first()
        number = s.number
        avg_length = s.avg_length
        # select last generation
        for row in session.execute("select id from score where id in (select id from generation where iteration={})"
                                   " order by score desc limit {}".format(iteration - 1, number - 1)).all():
            genes.append(row[0])
        session.commit()
    pos_list = [i for i in range(len(genes))]
    for _ in range(number - 1):
        father = genes[min(random.sample(pos_list, 2))]
        mother = genes[min(random.sample(pos_list, 2))]
        couples.append((father, mother))
    # generate
    for couple in couples:
        generate_gene(engine, couple, iteration, avg_length)
    # keep the best one
    with Session(engine) as session:
        row = session.execute("select * from main order by id desc limit 1").first()
        id = row[0] + 1
        row = session.execute("select * from main where id={}".format(genes[0])).first()
        gene = row[1]
        row1 = Generation(id=id, father=genes[0], mother=genes[0], iteration=iteration)
        session.merge(row1)
        session.commit()
        row2 = Main(id=id, gene=gene)
        session.merge(row2)
        session.commit()
    return len(couples)


def generate_gene(engine, couple, iteration, avg_length):
    # get genes of couple
    with Session(engine) as session:
        row_father = session.execute("select * from main where id={}".format(couple[0],)).first()
        row_mother = session.execute("select * from main where id={}".format(couple[1],)).first()
        session.commit()
    gene_f = row_father[1]
    gene_m = row_mother[1]
    # crossover gene of couple
    gene = crossover_gene(gene_f, gene_m, iteration)
    # mutant gene of couple
    gene = mutant_gene(gene, iteration, avg_length)
    # save gene as child
    # get id of child
    with Session(engine) as session:
        row = session.execute("select id from main order by id desc limit 1").first()
        id = row[0] + 1
        # get generation of child in generation
        row1 = Generation(id=id, father=couple[0], mother=couple[1], iteration=iteration)
        session.merge(row1)
        session.commit()
        row2 = Main(id=id, gene=gene)
        session.merge(row2)
        session.commit()


def crossover_gene(gene_a: str, gene_b: str, iteration: int):
    # copy the genes
    gene_a_c = gene_a
    gene_b_c = gene_b
    # prepare the genes without 0 and 255
    gene_a_units = gene_a.split('-')[1:-1]
    gene_b_units = gene_b.split('-')[1:-1]
    gene_a_len = len(gene_a_units)
    gene_b_len = len(gene_b_units)
    # crossover
    new_gene_a_units = []
    new_gene_b_units = []
    for i in range(max(gene_a_len, gene_b_len)):
        if i >= gene_a_len:
            temp_a = None
        else:
            temp_a = gene_a_units[i]
        if i >= gene_b_len:
            temp_b = None
        else:
            temp_b = gene_b_units[i]
        if random.randint(0, 1) == 0:
            if temp_a is not None:
                new_gene_a_units.append(temp_a)
            if temp_b is not None:
                new_gene_b_units.append(temp_b)
        else:
            if temp_a is not None:
                new_gene_b_units.append(temp_a)
            if temp_b is not None:
                new_gene_a_units.append(temp_b)
    # add 0 and 255
    new_gene_a_units = ['0'] + new_gene_a_units + ['255']
    new_gene_b_units = ['0'] + new_gene_b_units + ['255']
    gene_a = '-'.join(new_gene_a_units)
    gene_b = '-'.join(new_gene_b_units)
    # select one gene for next generation
    if iteration < 1:
        iteration = 1
    p_crossover = 1 / float(iteration * iteration)
    which_gene = random.randint(0, 1)
    if random.random() < p_crossover:
        if which_gene == 0:
            gene = gene_a
        else:
            gene = gene_b
    else:
        if which_gene == 0:
            gene = gene_a_c
        else:
            gene = gene_b_c
    return gene


def mutant_gene(gene: str, iteration: int, avg_length: int):
    # mutant?
    if iteration < 1:
        iteration = 1
    if avg_length < 1:
        avg_length = 1
    if random.random() >= 1 / (iteration ** (1 / float(avg_length))):
        return gene
    # prepare the gene
    gene_units = gene.split('-')
    mutant_type = random.randint(0, 2)
    # if there is no real units, only add one unit.
    if len(gene_units) == 2:
        mutant_type = 1
    if mutant_type == 0:
        # delete one unit
        if len(gene_units) > 3:
            unit_index = random.randint(1, len(gene_units) - 2)
            gene_units = gene_units[:unit_index] + gene_units[unit_index + 1:]
        gene = '-'.join(gene_units)
        return gene
    elif mutant_type == 1:
        # add one unit
        unit_index = random.randint(1, len(gene_units) - 1)
        unit_type = random.randint(1, 3)
        if unit_type == 1:
            unit_params = ['1', str(random.randint(1, unit_index)), '1', str(random.randint(4, 32)),
                           str(random.choice([1, 3, 5])), str(random.randint(1, 16)), str(random.randint(0, 2))]
        else:
            unit_params = ['2', str(random.randint(1, unit_index)), str(random.randint(1, unit_index)),
                           str(unit_type)]
        gene_units.insert(unit_index, ','.join(unit_params))
        gene = '-'.join(gene_units)
        return gene
    # modify one unit
    # choice one unit
    unit_index = random.randint(1, len(gene_units) - 2)
    # prepare the unit
    unit_params = gene_units[unit_index].split(',')
    unit_input_num = int(unit_params[0])
    unit_type = int(unit_params[unit_input_num + 1])
    # choice part of params
    if 1 < unit_type < 5:
        part_mutant = random.randint(0, 1)
    else:
        part_mutant = random.randint(0, 2)
    # mutant
    if part_mutant == 0:  # inputs part
        input_mutant = random.randint(1, unit_input_num)
        base = int(unit_params[input_mutant])
        unit_params[input_mutant] = str(random.randint(base - int(base / 2), int((base + 1) * 1.5)))
    elif part_mutant == 1:  # unit type
        type_mutant = random.randint(1, 3)
        if type_mutant == 1:  # CNN
            unit_params = ['1', unit_params[1], '1', str(random.randint(4, 32)), str(random.choice([1, 3, 5])),
                           str(random.randint(1, 16)), str(random.randint(0, 2))]
        elif type_mutant == 2 or type_mutant == 3:
            if unit_input_num == 2:
                unit_params = ['2', unit_params[1], unit_params[2], str(type_mutant)]
            else:
                if unit_params[1] == '1':
                    unit_params = ['2', '1', str(random.randint(2, 8)), str(type_mutant)]
                else:
                    unit_params = ['2', '1', unit_params[1], str(type_mutant)]
    elif part_mutant == 2:  # net params
        if unit_type == 1:  # CNN
            mutant_pos = random.randint(3, 5)
            if mutant_pos == 3:  # channel
                base = int(unit_params[3])
                unit_params[3] = str(random.randint(max(base - int(base / 2), 3), int((base + 1) * 1.5)))
            elif mutant_pos == 4:  # filter_size
                unit_params[4] = str(random.choice([1, 3, 5]))
            elif mutant_pos == 5:  # activation
                unit_params[6] = str(random.randint(0, 2))
    gene_units[unit_index] = ','.join(unit_params)
    gene = '-'.join(gene_units)
    return gene


def ea_train_score(engine, iteration: int):
    # find genes that need to express to code, and express them
    with Session(engine) as session:
        for row in session.execute("select * from main where id not in (select id from code)").all():
            express_gene(engine, row[0], row[1])
        # find codes of genes that need to train, then train and score them
        for row in session.execute("select * from code where id not in (select id from score) "
                                   "and id not in (select id from bad_code)").all():
            train_score_code(engine, row[0], row[1])
        session.commit()
    # waiting for finish
    wait_train_workers(engine)
    # update status
    with Session(engine) as session:
        row = session.execute("select * from status order by id desc limit 1").first()
        id = row[0] + 1
        row = Status(id=id, status=1, iteration=iteration)
        session.merge(row)
        session.commit()
    return iteration


def wait_train_workers(engine):
    with Session(engine) as session:
        row = session.execute("select * from task where id not in (select id from task_end)").first()
        while row is not None:
            time.sleep(60)
            row = session.execute("select * from task where id not in (select id from task_end)").first()
        session.commit()


def express_gene(engine, id: int, gene: str):
    # decode genes
    gene_units = gene.split('-')
    gene_units_params = [unit.split(',') for unit in gene_units]
    # fix gene params
    gene_units_params_fixed = fix_gene_params(gene_units_params)
    # record gene_fixed for debug
    with Session(engine) as session:
        row = GeneExpressed(id=id, gene_expressed='-'.join(
            [','.join(unit_params) for unit_params in gene_units_params_fixed]))
        session.merge(row)
        session.commit()
    # gene hash
    gene_units = [','.join(unit_params) for unit_params in gene_units_params_fixed][1:-1]
    seq, info = gene_graph_seq_with_info(gene_units)
    info_s = '-'.join(info)
    h = hashlib.sha1((seq + '|' + info_s).encode(encoding='utf-8')).hexdigest()
    with Session(engine) as session:
        row = Hash(id=id, hash=h, seq_info=seq + '|' + info_s)
        session.merge(row)
        session.commit()
    # translate to code
    code, gene_code_length = ea_code_tf.gene_to_code(gene_units_params_fixed)
    gene_length = len(gene_units_params) - 2
    with Session(engine) as session:
        row1 = Code(id=id, code=code)
        session.merge(row1)
        session.commit()
        row2 = Length(id=id, length=gene_length, gene_code_length=gene_code_length)
        session.merge(row2)
        session.commit()


def fix_gene_params(gene_units_params):
    # record gene before fixing
    gene_before = '-'.join([','.join(unit_params) for unit_params in gene_units_params])
    # bounds error fix
    gene_units_params = fix_gene_input_bounds(gene_units_params)
    # input shape error
    gene_units_params = fix_gene_input_shape(gene_units_params)
    # record gene after fixing
    gene_after = '-'.join([','.join(unit_params) for unit_params in gene_units_params])
    # compare before and after gene
    while gene_before != gene_after:
        # record gene before fixing
        gene_before = gene_after
        # bounds error fix
        gene_units_params= fix_gene_input_bounds(gene_units_params)
        # input shape error
        gene_units_params = fix_gene_input_shape(gene_units_params)
        # record gene after fixing
        gene_after = '-'.join([','.join(unit_params) for unit_params in gene_units_params])
    return gene_units_params


def fix_gene_input_bounds(gene_units_params):
    pos = 0
    for i in range(len(gene_units_params)):
        if gene_units_params[i][0] == '0':
            continue
        if gene_units_params[i][0] == '255':
            break
        pos = pos + 1
        for j in range(int(gene_units_params[i][0])):
            if int(gene_units_params[i][j + 1]) > pos:
                gene_units_params[i][j + 1] = str(pos)
        # if gene_units_params[i][int(gene_units_params[i][0]) + 1] == '5':
        #     pos = pos + 1
    return gene_units_params


def fix_gene_input_shape(gene_units_params):
    channel_record = []
    unit_insert = []
    for i in range(len(gene_units_params)):
        if gene_units_params[i][0] == '0':
            channel_record.append((i, 3))
            continue
        if gene_units_params[i][0] == '255':
            if channel_record[-1][1] != 48:
                type = int(
                    gene_units_params[channel_record[-1][0]][int(gene_units_params[channel_record[-1][0]][0]) + 1])
                if type == 1:  # CNN directly change filters
                    gene_units_params[channel_record[-1][0]][3] = '48'
                    group = int(gene_units_params[channel_record[-1][0]][5])
                    offset_a = int(gene_units_params[channel_record[-1][0]][1])
                    channel_a = channel_record[-offset_a - 1][1]
                    group = find_near_divide(group, 48, channel_a)
                    gene_units_params[channel_record[-1][0]][5] = str(group)
                else:
                    unit_insert.append((i, ['1', '1', '1', '48', '3', '1', '0']))
                fixed = 1
            break
        type = int(gene_units_params[i][int(gene_units_params[i][0]) + 1])
        if type == 1:  # CNN
            group = int(gene_units_params[i][5])
            channel = int(gene_units_params[i][3])
            offset_a = int(gene_units_params[i][1])
            channel_a = channel_record[-offset_a][1]
            group_new = find_near_divide(group, channel_a, channel)
            gene_units_params[i][5] = str(group_new)
            channel_record.append((i, channel))
        if type == 2 or type == 3:  # + *
            offset_a = int(gene_units_params[i][1])
            offset_b = int(gene_units_params[i][2])
            channel_a = channel_record[-offset_a][1]
            channel_b = channel_record[-offset_b][1]
            if channel_a == channel_b:
                channel_record.append((i, channel_a))
            else:
                if (channel_a > channel_b and gene_units_params[channel_record[-offset_a][0]][0] != '0') \
                        or gene_units_params[channel_record[-offset_b][0]][0] == '0':
                    # Fix a
                    type_a = int(gene_units_params[channel_record[-offset_a][0]][
                                     int(gene_units_params[channel_record[-offset_a][0]][0]) + 1])
                    if type_a == 1:  # CNN directly change filters
                        group = int(gene_units_params[channel_record[-offset_a][0]][5])
                        offset_a_a = int(gene_units_params[channel_record[-offset_a][0]][1])
                        channel_a_a = channel_record[-offset_a - offset_a_a][1]
                        group = find_near_divide(group, channel_b, channel_a_a)
                        gene_units_params[channel_record[-offset_a][0]][5] = str(group)
                        gene_units_params[channel_record[-offset_a][0]][3] = str(channel_b)
                        channel_record[-offset_a] = (channel_record[-offset_a][0], channel_b)
                        channel_record.append((i, channel_b))
                    else:
                        unit_insert.append((i, ['1', gene_units_params[i][1], '1', str(channel_b), '3', '1', '1']))
                        gene_units_params[i][1] = '1'
                        gene_units_params[i][2] = str(int(gene_units_params[i][2]) + 1)
                        channel_record.append((i, channel_b))
                else:
                    # Fix b
                    type_b = int(gene_units_params[channel_record[-offset_b][0]][
                                     int(gene_units_params[channel_record[-offset_b][0]][0]) + 1])
                    if type_b == 1:  # CNN directly change filters
                        group = int(gene_units_params[channel_record[-offset_b][0]][5])
                        offset_b_b = int(gene_units_params[channel_record[-offset_b][0]][1])
                        channel_b_b = channel_record[-offset_b - offset_b_b][1]
                        group = find_near_divide(group, channel_a, channel_b_b)
                        gene_units_params[channel_record[-offset_b][0]][5] = str(group)
                        gene_units_params[channel_record[-offset_b][0]][3] = str(channel_a)
                        channel_record[-offset_b] = (channel_record[-offset_b][0], channel_a)
                        channel_record.append((i, channel_a))
                    else:
                        unit_insert.append((i, ['1', gene_units_params[i][2], '1', str(channel_a), '3', '1', '1']))
                        gene_units_params[i][1] = '1'
                        gene_units_params[i][2] = str(int(gene_units_params[i][1]) + 1)
                        channel_record.append((i, channel_a))
    gene_units_params = gene_insert_units(gene_units_params, unit_insert)
    return gene_units_params


def find_near_divide(k, s, t):
    if k < 1:
        return 1
    if s != t:
        if s > t:
            p = t
            t = s
            s = p
        r = t % s
        while r > 0:
            t = s
            s = r
            r = t % s
    if k > s:
        return s
    if s % k == 0:
        return k
    d = 1
    while d < s / 2 + 1:
        if s % (k - d) == 0:
            return k - d
        if s % (k + d) == 0:
            return k + d
        d = d + 1
    return 1


def gene_insert_units(gene_units_params, unit_insert):
    insert_offset = 0
    for insert in unit_insert:
        insert_pos = insert[0] + insert_offset
        gene_units_params.insert(insert_pos, insert[1])
        pos = 1
        next_input_num = int(gene_units_params[insert_pos + 1][0])
        if next_input_num == 0 or next_input_num == 255:
            continue
        if gene_units_params[insert_pos + 1][next_input_num + 1] == '5':
            pos = pos + int(gene_units_params[insert_pos + 1][3]) - 1
        for i in range(insert_pos + 2, len(gene_units_params) - 1):
            input_num = int(gene_units_params[i][0])
            for j in range(input_num):
                offset = int(gene_units_params[i][j + 1])
                if offset > pos:
                    gene_units_params[i][j + 1] = str(offset + 1)
            if gene_units_params[i][input_num + 1] == '5':
                pos = pos + int(gene_units_params[i][3])
            else:
                pos = pos + 1
        insert_offset = insert_offset + 1
    return gene_units_params


def train_score_code(engine, id: int, code: str):
    with Session(engine) as session:
        row = session.execute("select * from code_file where id={}".format(id,)).first()
        if row is None:
            # Insert code to train and score framework and write it to file
            code_filename, code_name = ea_code_tf.code_to_file(id, code)
            # Set code file info
            ckpt_dir = './ckpt/' + code_name
            save_dir = './save/' + code_name
            log_dir = './logs/' + code_name
            row = CodeFile(id=id, file=code_filename, ckpt_dir=ckpt_dir, save_dir=save_dir, log_dir=log_dir)
            session.merge(row)
            session.commit()

        # Check whether trained
        sql = 'select psnr, ssim from sr where id in (select id from hash where hash=(select hash from hash where id={})) ' \
              'order by psnr+ssim desc limit 1'.format(id)
        row = session.execute(sql).first()
        if row is not None:
            psnr = row[0]
            ssim = row[1]
            row = session.execute("select * from sr where id={}".format(id,)).first()
            if row is None:
                row = SR(id=id, psnr=psnr, ssim=ssim)
                session.merge(row)
            session.commit()
            return
        # Submit train task
        if session.query(Task).where(Task.id==id).first() is None:
            row = Task(id=id)
            session.merge(row)
            session.commit()
        runtime_evaluation.request_evaluation([id])


def ea_number(engine, iteration: int):
    # Score all
    with Session(engine) as session:
        rows = session.execute("select sr.id from generation inner join sr on "
                               "generation.id=sr.id where iteration={}".format(iteration,)).all()
        session.commit()
    id_list = []
    for row in rows:
        id_list.append(row[0])
    k = True
    while k:
        res = runtime_evaluation.look_evaluation(id_list)
        k = False
        for r, id in zip(res, id_list):
            if r[0] is None:
                k = True
                runtime_evaluation.request_evaluation([id])
                time.sleep(60)
                break
    res = runtime_evaluation.look_evaluation(id_list)
    with Session(engine) as session:
        for i in range(len(res)):
            r = res[i][0]
            if r > 0 and r < 33.33:
                id = id_list[i]
                row = Runtime(id=id, runtime=r)
                session.merge(row)
                session.commit()
                row = session.execute("select * from sr where id={}".format(id)).first()
                if row is not None:
                    psnr = row[1]
                    ssim = row[2]
                    s = score.score_sr(psnr, ssim, r)
                    row = Score(id=id, score=s)
                    session.merge(row)
        session.commit()
    # differentiable search
    differentiable_search(engine, iteration)
    # avg_length
    with Session(engine) as session:
        row = session.execute("select AVG(length) from length").first()
        avg_length = int(row[0])
        number = avg_length * 2
        row = session.execute("select id from number order by id desc limit 1").first()
        if row is None:
            id = 0
        else:
            id = row[0] + 1
        row = Number(id=id, number=number, avg_length=avg_length)
        session.merge(row)
        session.commit()
        row = session.execute("select * from status order by id desc limit 1").first()
        id = row[0] + 1
        row = Status(id=id, status=2, iteration=iteration)
        session.merge(row)
        session.commit()
    return iteration


def ea_loop(engine, status_dict):
    iteration = status_dict['iteration']
    if status_dict['status'] == 0:
        iteration = ea_train_score(engine, iteration)
        iteration = ea_number(engine, iteration)
    elif status_dict['status'] == 1:
        iteration = ea_number(engine, iteration)
    while True:
        iteration = ea_select_generate(engine, iteration)
        iteration = ea_train_score(engine, iteration)
        iteration = ea_number(engine, iteration)
