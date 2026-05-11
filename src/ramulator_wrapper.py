import subprocess
import math
import os
import pathlib
import sys
from src.config import *
from src.model import *
from src.type import *

try:
    import pandas as pd
except ImportError:
    pd = None


class Ramulator:

    def __init__(self,
                 modelinfos,
                 ramulator_dir,
                 output_log='',
                 fast_mode=False,
                 num_hbm=5,
                 max_L=2048):
        self.df = pd.DataFrame() if pd is not None else None
        self.ramulator_dir = ramulator_dir
        self.output_log = output_log
        self.max_L = max_L
        if pd is None:
            self.df = None
        elif os.path.exists(output_log):
            self.df = pd.read_csv(output_log)
            if 'max_L' not in self.df.columns:
                self.df.insert(1, 'max_L', self.max_L)
        self.tCK = 0.769  # ns
        self.num_hbm = num_hbm
        self.nhead = modelinfos.get('num_kv_heads', modelinfos.get('num_heads'))
        self.dhead = modelinfos['dhead']
        self.fast_mode = fast_mode

    def make_yaml_file(self, yaml_file, file_name, power_constraint):
        trace_path = os.path.join(self.ramulator_dir, file_name + ".trace")
        line = ""
        line += "Frontend:\n"
        line += "  impl: PIMLoadStoreTrace\n"
        line += "  path: {}\n".format(trace_path)
        line += "  clock_ratio: 1\n"
        line += "\n"
        line += "  Translation:\n"
        line += "    impl: NoTranslation\n"
        line += "    max_addr: 2147483648\n"
        line += "              \n"
        line += "\n"
        line += "MemorySystem:\n"
        line += "  impl: PIMDRAM\n"
        line += "  clock_ratio: 1\n"
        line += "  DRAM:\n"
        line += "    impl: HBM3-PIM\n"
        line += "    org:\n"
        line += "      preset: HBM3_8Gb_2R\n"
        line += "      channel: 16\n"
        line += "    timing:\n"
        if power_constraint:
            line += "      preset: HBM3_5.2Gbps\n"
        else:
            line += "      preset: HBM3_5.2Gbps_NPC\n"
        line += "\n"
        line += "  Controller:\n"
        line += "    impl: HBM3-PIM\n"
        line += "    Scheduler:\n"
        line += "      impl: PIM\n"
        line += "    RefreshManager:\n"
        line += "      impl: AllBankHBM3\n"
        line += "      #impl: No\n"
        line += "    plugins:\n"
        line += "\n"
        line += "  AddrMapper:\n"
        line += "    impl: HBM3-PIM\n"
        with open(yaml_file, 'w') as f:
            f.write(line)

    def update_log_file(self, log):
        assert pd is not None, "Ramulator logging requires pandas"
        if self.df.empty:
            if os.path.exists(self.output_log):
                df = pd.read_csv(self.output_log)
            else:
                columns = [
                    'L', 'max_L', 'nhead', 'dhead', 'dbyte', 'pim_type',
                    'power_constraint', 'cycle', 'mac', 'softmax', 'mvgb',
                    'mvsb', 'wrgb'
                ]
                df = pd.DataFrame(columns=columns)
        else:
            df = self.df
        if 'max_L' not in df.columns:
            df.insert(1, 'max_L', self.max_L)
        new_df = pd.DataFrame(columns=df.columns)
        new_df.loc[0] = log
        df = pd.concat([df, new_df]).drop_duplicates()
        self.df = df
        self.df.to_csv(self.output_log, index=False)

    #def run_ramulator(self):
    def run_ramulator(self, pim_type: PIMType, l, num_ops_per_hbm, dbyte,
                      yaml_file, file_name):
        pim_type_name = pim_type.name.lower(
        ) if not pim_type == PIMType.BA else "bank"
        trace_file = os.path.join(self.ramulator_dir, file_name + '.trace')

        root = pathlib.Path(__file__).resolve().parents[1]
        trace_candidates = [
            pathlib.Path(self.ramulator_dir) / "trace_gen" /
            "gen_trace_attacc_{}.py".format(pim_type_name),
            root / "pim_ramulator_src" / "trace_gen" /
            "gen_trace_attacc_{}.py".format(pim_type_name),
        ]
        trace_exc = next((p for p in trace_candidates if p.exists()), None)
        if trace_exc is None:
            raise FileNotFoundError(
                "Missing trace generator for PIM mode {}".format(
                    pim_type_name))

        # generate trace
        subprocess.run([
            sys.executable, str(trace_exc),
            "--dhead", str(self.dhead),
            "--nhead", str(num_ops_per_hbm),
            "--seqlen", str(l),
            "--maxlen", str(self.max_L),
            "--dbyte", str(dbyte),
            "--output", trace_file,
        ], check=True, capture_output=True, text=True)

        # run ramulator
        ramulator_candidates = [
            pathlib.Path(self.ramulator_dir) / "ramulator2",
            pathlib.Path(self.ramulator_dir) / "ramulator2.exe",
            pathlib.Path(self.ramulator_dir) / "build" / "ramulator2",
            pathlib.Path(self.ramulator_dir) / "build" / "ramulator2.exe",
        ]
        ramulator_file = next((p for p in ramulator_candidates if p.exists()),
                              None)
        if ramulator_file is None:
            raise FileNotFoundError(
                "Missing ramulator2 executable under {}".format(
                    self.ramulator_dir))
        try:
            result = subprocess.run([str(ramulator_file), "-f", yaml_file],
                                    check=True,
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE,
                                    text=True)
            output_lines = result.stdout.strip().split('\n')
            output_list = [
                line.strip() for line in output_lines if line.strip()
            ]
        finally:
            if os.path.exists(trace_file):
                os.remove(trace_file)

        # parsing output
        n_cmds = {"mac": 0, "sfm": 0, "mvgb": 0, "mvsb": 0, "wrgb": 0}
        cycle = 0
        for line in output_list:
            if "mac" in line:
                n_cmds["mac"] += int(line.split()[-1])
            elif "softmax_requests" in line:
                n_cmds["sfm"] += int(line.split()[-1])
            elif "move_to_gemv_buffer" in line:
                n_cmds["mvgb"] += int(line.split()[-1])
            elif "move_to_softmax_buffer" in line:
                n_cmds["mvsb"] += int(line.split()[-1])
            elif "write_to_gemv_buffer" in line:
                n_cmds["wrgb"] += int(line.split()[-1])
            elif "memory_system_cycles" in line:
                cycle += int(line.split()[-1])

        if cycle == 0:
            raise RuntimeError("Ramulator produced no memory_system_cycles")

        out = [
            cycle, n_cmds["mac"], n_cmds["sfm"], n_cmds["mvgb"], n_cmds["mvsb"],
            n_cmds["wrgb"]
        ]
        return out

    def run(self, pim_type: PIMType, layer: Layer, power_constraint=True):
        if os.path.exists(self.ramulator_dir):
            l = layer.n
            dhead = self.dhead
            dbyte = layer.dbyte
            num_ops_per_attacc = getattr(layer, 'pim_numOp', layer.numOp)
            num_ops_per_hbm = math.ceil(num_ops_per_attacc / self.num_hbm)
            num_ops_group = 1
            if self.fast_mode:
                minimum_heads = 64
                num_ops_group = math.ceil(num_ops_per_hbm / minimum_heads)
                num_ops_per_hbm = minimum_heads

            file_name = "attacc_l{}_maxl{}_nattn{}_dhead{}_dbyte{}_pc{}".format(
                l, self.max_L, num_ops_per_hbm, dhead, layer.dbyte,
                int(power_constraint))
            yaml_file = os.path.join(self.ramulator_dir, file_name + '.yaml')
            self.make_yaml_file(yaml_file, file_name, power_constraint)

            try:
                result = self.run_ramulator(pim_type, l, num_ops_per_hbm,
                                            layer.dbyte, yaml_file, file_name)
            finally:
                if os.path.exists(yaml_file):
                    os.remove(yaml_file)

            # post processing
            # 32: read granularity
            cycle, mac, sfm, mvgb, mvsb, wrgb = result
            si_io = wrgb * 32  # 256 bit
            tsv_io = (wrgb + mvsb + mvgb) * 32
            giomux_io = (wrgb + mvsb + mvgb) * 32
            bgmux_io = (wrgb + mvsb + mvgb) * 32
            mem_acc = mac * 32
            if pim_type == PIMType.BA:
                # pCH * Rank * bank group * bank
                mem_acc *= 2 * 2 * 4 * 4
            elif pim_type == PIMType.BG:
                # pCH * Rank * bank group
                mem_acc *= 2 * 2 * 4
            else:
                mem_acc *= 1

            ## update log file

            log = [
                l, self.max_L, num_ops_per_hbm, dhead, dbyte, pim_type.name,
                power_constraint
            ] + result
            self.update_log_file(log)

            ## si, tsv, giomux to bgmux, bgmux to column decoder, bank RD
            traffic = [si_io, tsv_io, giomux_io, bgmux_io, mem_acc]
            traffic = [i * self.num_hbm for i in traffic]
            traffic = [i * num_ops_group for i in traffic]
            exec_time = self.tCK * cycle / 1000 / 1000 / 1000  # ns -> s
            return exec_time, traffic

        else:
            assert 0, "Need to install ramulator"

    def output(self, pim_type: PIMType, layer: Layer, power_constraint=True):
        assert pd is not None, "Ramulator execution requires pandas"
        if self.df.empty:
            self.run(pim_type, layer, power_constraint)

        num_ops_per_attacc = getattr(layer, 'pim_numOp', layer.numOp)
        num_ops_per_hbm = math.ceil(num_ops_per_attacc / self.num_hbm)
        num_ops_group = 1
        if self.fast_mode:
            minimum_heads = 64
            num_ops_group = math.ceil(num_ops_per_hbm / minimum_heads)
            num_ops_per_hbm = minimum_heads

        l = layer.n
        dhead = layer.k
        dbyte = layer.dbyte
        row = self.df[(self.df['L'] == l) & (self.df['max_L'] == self.max_L) & \
                      (self.df['nhead'] == num_ops_per_hbm) & \
                      (self.df['dbyte'] == dbyte) & (self.df['dhead'] == dhead) & \
                      (self.df['power_constraint'] == power_constraint) &  \
                      (self.df['pim_type'] == pim_type.name)]
        if row.empty:
            return self.run(pim_type, layer, power_constraint)

        else:
            cycle = int(row.iloc[0]['cycle'])
            mac = int(row.iloc[0]['mac'])
            softmax = int(row.iloc[0]['softmax'])
            mvgb = int(row.iloc[0]['mvgb'])
            mvsb = int(row.iloc[0]['mvsb'])
            wrgb = int(row.iloc[0]['wrgb'])
            si_io = wrgb * 32  # 256 bit
            tsv_io = (wrgb + mvsb + mvgb) * 32
            giomux_io = (wrgb + mvsb + mvgb) * 32
            bgmux_io = (wrgb + mvsb + mvgb) * 32
            mem_acc = mac * 32
            if pim_type == PIMType.BA:
                # pCH * Rank * bank group * bank
                mem_acc *= 2 * 2 * 4 * 4
            elif pim_type == PIMType.BG:
                # pCH * Rank * bank group
                mem_acc *= 2 * 2 * 4
            else:
                mem_acc *= 2

            ## si, tsv, giomux to bgmux, bgmux to column decoder, bank RD
            traffic = [si_io, tsv_io, giomux_io, bgmux_io, mem_acc]
            traffic = [i * self.num_hbm for i in traffic]
            traffic = [i * num_ops_group for i in traffic]
            exec_time = self.tCK * cycle / 1000 / 1000 / 1000  # ns -> s
            exec_time *= num_ops_group
            return exec_time, traffic
