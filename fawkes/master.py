import socket
import subprocess
import sys
import time

print(socket.gethostname())


def assign_gpu(args, gpu_idx):
    for i, arg in enumerate(args):
        if arg == "GPUID":
            args[i] = str(gpu_idx)
    return args


def produce_present():
    process_ls = []
    gpu_ls = list(sys.argv[1])
    max_num = int(sys.argv[2])

    available_gpus = []
    i = 0
    while len(available_gpus) < max_num:
        if i > len(gpu_ls) - 1:
            i = 0
        available_gpus.append(gpu_ls[i])
        i += 1

    process_dict = {}
    all_queries_to_run = []

    for m in ['mid', 'low', 'min']:
        for directory in ['KimKardashian', 'Liuyifei', 'Obama', 'TaylorSwift', 'TomHolland']:
            args = ['python3', 'protection.py', '--gpu', 'GPUID', '-d',
                    '/home/shansixioing/fawkes/data/test/{}/'.format(directory),
                    '--batch-size', '30', '-m', m,
                    '--debug']
            args = [str(x) for x in args]
            all_queries_to_run.append(args)

    for args in all_queries_to_run:
        cur_gpu = available_gpus.pop(0)
        args = assign_gpu(args, cur_gpu)
        print(" ".join(args))
        p = subprocess.Popen(args)
        process_ls.append(p)
        process_dict[p] = cur_gpu

        gpu_ls.append(cur_gpu)
        time.sleep(5)
        while not available_gpus:
            for p in process_ls:
                poll = p.poll()
                if poll is not None:
                    process_ls.remove(p)
                    available_gpus.append(process_dict[p])

            time.sleep(20)


def main():
    produce_present()


if __name__ == '__main__':
    main()
