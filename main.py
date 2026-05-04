import argparse
import csv
import os
from src.system import *
from src.type import *
from src.config import *
from src.ramulator_wrapper import *

RAMULATOR = False


def write_csv(logfile, perfs):
    if logfile is not None:
        firstrow = False
        if not os.path.exists(logfile):
            firstrow = True

        with open(logfile, 'a', newline='') as f:
            wrt = csv.writer(f)
            if firstrow:
                col_name = [
                    'model', 'dtype', 'xpu', 'cap', 'bw', 'sys_opb', 'hw',
                    'cores', 'pipe_level', 'is parallel', 'power constraint',
                    'gqa_size', 'Lin', 'Lout', 'bs', 'required_cap_per_gpu',
                    's_flops', 'g_flops', 's_time', 's_matmul', 's_fc',
                    's_comm', 's_x2g', 's_softmax', 's_act', 's_lnorm',
                    'g_time (ms)', 'g_matmul', 'g_fc', 'g_comm', 'g_etc',
                    'g_qkv_time', 'g_prj_time', 'g_ff_time', 'g2g_comm',
                    'c2g_comm', 'g_softmax', 'g_act', 'g_lnorm',
                    'g_energy (nJ)', 'g_dram_energy', 'g_l2_energy',
                    'g_l1_energy', 'g_reg_energy', 'g_alu_energy',
                    'g_fc_mem_energy', 'g_fc_comp_energy',
                    'g_attn_mem_energy', 'g_attn_comp_energy',
                    'g_etc_mem_energy', 'g_etc_comp_energy', 'g_comm_energy'
                ]
                wrt.writerow(col_name)

            for perf in perfs:
                tag, config, time, energy = perf
                info = tag + config + time + energy
                wrt.writerow(info)


def run(system: System,
        batch,
        lin,
        lout,
        power_constraint=False,
        pipe=0,
        parallel=False,
        output_file=None):
    print("---Run simple mode Batch {} Lin {} Lout {} pipe {} parall {}---".
          format(batch, lin, lout, pipe, parallel))
    assert system.model_set, "Need to SetModel"
    perfs = []
    system.simulate(batch,
                    lin,
                    lout,
                    perfs=perfs,
                    pipe=pipe,
                    parallel_ff=parallel,
                    power_constraint=power_constraint)
    if output_file is not None:
        write_csv(output_file, perfs)


def main():
    parser = argparse.ArgumentParser(
        description="Model configuration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    ## set system configuration
    parser.add_argument(
        "--system",
        type=str,
        default="dgx",
        help="dgx (each GPU has 80GB HBM), \
              dgx-cpu (In dgx, offloading the attention layer to cpu), \
              dgx-attacc (dgx + attacc)")
    parser.add_argument(
        "--gpu",
        type=str,
        default='A100a',
        help="GPU type (A100a and H100), A100a is A100 with HBM3")
    parser.add_argument("--ngpu",
                        type=int,
                        default=8,
                        help="number of GPUs in DGX system. default=8")
    parser.add_argument("--tp",
                        type=int,
                        default=None,
                        help="tensor parallel degree. default=ngpu")
    parser.add_argument("--gmemcap",
                        type=int,
                        default=80,
                        help="memory capacity per GPU (GB). default=80")



    ## set attacc configuration
    parser.add_argument("--pim",
                        type=str,
                        default='bank',
                        help="pim mode. list: bank, bg, buffer")
    parser.add_argument("--num_attacc",
                        type=int,
                        default=None,
                        help="number of AttAcc devices. default=ngpu")
    parser.add_argument("--num_hbm",
                        type=int,
                        default=5,
                        help="HBM stacks per AttAcc")
    parser.add_argument("--interface",
                        type=str,
                        default='NVLINK3',
                        choices=['NVLINK3', 'NVLINK4', 'PCIE4', 'PCIE5'],
                        help="GPU-accelerator interface")
    parser.add_argument("--powerlimit",
                        action='store_true',
                        help="power constraint for PIM ")
    parser.add_argument("--ffopt",
                        action='store_true',
                        help="apply feedforward parallel optimization")
    parser.add_argument("--pipeopt",
                        action='store_true',
                        help="apply pipeline optimization ")
    parser.add_argument("--routing",
                        type=str,
                        default='default',
                        choices=['default', 'conservative', 'optimistic',
                                 'list'],
                        help="decoder attention routing mode")
    parser.add_argument("--pim_layers",
                        type=str,
                        default='',
                        help="comma-separated decoder layer indices for routing")

    ## set model and service environment
    parser.add_argument(
        "--model",
        type=str,
        default='GPT-175B',
        help="model list: GPT-175B, LLAMA-65B, MT-530B, OPT-66B")
    parser.add_argument("--word",
                        type=int,
                        default=2,
                        help="word size (precision): 1(INT8), 2(FP16)")
    parser.add_argument("--lin",
                        type=int,
                        default=2048,
                        help="input sequence length")
    parser.add_argument("--lout",
                        type=int,
                        default=128,
                        help="number of generated tokens")
    parser.add_argument("--max_L",
                        type=int,
                        default=2048,
                        help="maximum sequence length for PIM trace layout")
    parser.add_argument("--prefill_chunk",
                        type=int,
                        default=512,
                        help="chunk size for sampled PIM prefill")
    parser.add_argument("--prefill_samples",
                        type=int,
                        default=8,
                        help="number of sampled chunks for PIM prefill")
    parser.add_argument("--image_size",
                        type=int,
                        default=None,
                        help="square image size for VLM vision graph")
    parser.add_argument("--image_width",
                        type=int,
                        default=None,
                        help="image width for VLM vision graph")
    parser.add_argument("--image_height",
                        type=int,
                        default=None,
                        help="image height for VLM vision graph")
    parser.add_argument(
        "--batch",
        type=int,
        default=1,
        help=
        "batch size, default = 1"
    )

    args = parser.parse_args()

    def parse_layer_list(value):
        if value == '':
            return None
        return [int(item.strip()) for item in value.split(',') if item.strip()]

    def parse_image_size():
        if args.image_width is not None or args.image_height is not None:
            fallback = args.image_size
            if fallback is None:
                fallback = args.image_width if args.image_width is not None else args.image_height
            width = args.image_width if args.image_width is not None else fallback
            height = args.image_height if args.image_height is not None else fallback
            return (width, height)
        return args.image_size

    global RAMULATOR
    if RAMULATOR:
        print("The Ramulator {}".format(RAMULATOR))

    if args.gpu == 'H100':
        gpu_device = GPUType.H100
    elif args.gpu == 'A100a':
        gpu_device = GPUType.A100a
    else:
        assert 0

    num_gpu = args.ngpu
    tp = args.tp if args.tp is not None else num_gpu
    num_attacc = args.num_attacc if args.num_attacc is not None else num_gpu
    assert num_attacc == tp == num_gpu, \
        "Deployment supports only num_attacc == tp == ngpu. " \
        "Got num_attacc={}, tp={}, ngpu={}.".format(num_attacc, tp, num_gpu)

    if args.system == 'dgx-attacc':
        print("{}: ({} x {}), PIM:{}, interface:{}, [Lin, Lout, batch]: {}".
              format(args.system, args.gpu, args.ngpu, args.pim,
                     args.interface, [args.lin, args.lout, args.batch]))
    else:
        print("{}: ({} x {}), [Lin, Lout, batch]: {}".format(
            args.system, args.gpu, args.ngpu,
            [args.lin, args.lout, args.batch]))
    gmem_cap = args.gmemcap * 1024 * 1024 * 1024
    output_path = "output.csv"
    if os.path.exists(output_path):
        os.remove(output_path)

    # set system
    dtype = DataType.W16A16 if args.word == 2 else DataType.W8A8
    modelinfos = make_model_config(args.model, dtype)
    xpu_config = make_xpu_config(gpu_device, num_gpu=num_gpu, mem_cap=gmem_cap)
    system = System(xpu_config['GPU'], modelinfos, max_L=args.max_L)
    if args.system in ['dgx-attacc']:
        if args.pim == "bg":
            pim_type = PIMType.BG
        elif args.pim == "buffer":
            pim_type = PIMType.BUFFER
        else:
            pim_type = PIMType.BA
        interface_type = InterfaceType[args.interface]
        pim_config = make_pim_config(pim_type,
                                     interface_type,
                                     num_attacc=num_attacc,
                                     num_hbm=args.num_hbm,
                                     power_constraint=args.powerlimit)
        system.set_accelerator(modelinfos, DeviceType.PIM, pim_config)

    elif args.system in ['dgx-cpu']:
        xpu_config = make_xpu_config(gpu_device)
        system.set_xpu(xpu_config['GPU'])
        system.set_accelerator(modelinfos, DeviceType.CPU, xpu_config['CPU'])

    if args.routing != 'default':
        system.set_routing(args.routing, parse_layer_list(args.pim_layers))
    system.set_prefill_config(args.prefill_chunk, args.prefill_samples)
    system.set_image_size(parse_image_size())

    run(system,
        args.batch,
        args.lin,
        args.lout,
        pipe=args.pipeopt,
        parallel=args.ffopt,
        output_file=output_path,
        power_constraint=args.powerlimit)


if __name__ == "__main__":
    main()
