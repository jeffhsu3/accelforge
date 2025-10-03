import pytimeloop.timeloopfe.v4 as tl
import time
import csv

from paths import DATA_DIR

NUM_THREADS = 12

def evaluate(result_fname, config_fname, evaluated_sizes):
    result_fname = DATA_DIR / result_fname
    spec = tl.Specification.from_yaml_files(
        config_fname,
        "ert/tpu_like.yaml"
    )
    spec.mapper.num_threads = NUM_THREADS
    with open(result_fname, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['evaluations', 'threads', 'mapper_time', 'energy', 'latency'])
        for evaluated_size in evaluated_sizes:
            start = time.time() 
            spec.mapper.evaluated_size = evaluated_size
            output = tl.call_mapper(spec, output_dir="outputs/timeloop")
            end = time.time()
            mapper_time = end - start
            writer.writerow([evaluated_size, NUM_THREADS, mapper_time, output.energy, output.latency])

# Without hint, finding valid mappings is easier and takes faster so search size should be larger here
evaluate('timeloop_gemm_16k.csv',
         'configs/timeloop_gemm_tpu.yaml',
         [50000, 100000, 150000, 200000])

evaluate('timeloop_gemm_16k_hint.csv',
         'configs/timeloop_gemm_tpu_hint.yaml',
         [50000, 100000, 150000, 200000])
