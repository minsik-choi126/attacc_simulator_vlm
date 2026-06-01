from .type import *
from .model import *
from .devices import *
from .config import *
import math
RAMPATH = "./ramulator2"
RAMLOG = "./ramulator.out"

OPB_PRINT = False


class System:

    def __init__(self,
                 gpu_config,
                 modelinfos=None,
                 hetero_name: DeviceType = DeviceType.NONE,
                 hetero_config=None,
                 max_L=2048):
        scaling_factor = SCALING_FACTOR
        self.hetero_name = hetero_name
        self.max_L = max_L
        self.GPU = xPU(DeviceType.GPU, gpu_config, scaling_factor)
        self.AttDevice = self.GPU
        if self.hetero_name == DeviceType.PIM:
            self.AttDevice = PIM(hetero_config, scaling_factor)

        elif self.hetero_name == DeviceType.CPU:
            self.AttDevice = xPU(DeviceType.CPU, hetero_config, scaling_factor)

        self.devices = {'GPU': self.GPU, 'Acc': self.AttDevice}

        self.model_set = 0
        self.routing_mode = None
        self.routing_layer_list = None
        self.prefill_chunk_size = 512
        self.prefill_sample_count = 8
        self.image_size = None
        if modelinfos is not None:
            self.model = Transformer(modelinfos,
                                     tensor_parallel=self.GPU.num_xpu)
            self.model_set = 1

        self.scaling_factor = scaling_factor

    def set_model(self, modelinfos):
        self.model = Transformer(modelinfos, tensor_parallel=self.GPU.num_xpu)
        self.model_set = 1

    def set_routing(self, mode='conservative', layer_list=None):
        self.routing_mode = mode
        self.routing_layer_list = layer_list

    def set_prefill_config(self, chunk_size=512, sample_count=8):
        self.prefill_chunk_size = chunk_size
        self.prefill_sample_count = sample_count

    def set_image_size(self, image_size=None):
        self.image_size = image_size

    def _accelerator_routing_name(self):
        if self.hetero_name == DeviceType.PIM:
            return 'pim'
        elif self.hetero_name == DeviceType.CPU:
            return 'cpu'
        return 'gpu'

    @staticmethod
    def get_pipelining_efficiency_latency(n_kv_per_gpu,
                                          num_hbm=5,
                                          batch_size=1,
                                          softmax_to_gemv=0.4):
        if batch_size >= 2:
            return 1.0
        n_kv_per_gpu = int(n_kv_per_gpu)
        if n_kv_per_gpu <= 0:
            return 1.0

        max_heads_per_hbm = math.ceil(n_kv_per_gpu / num_hbm)
        if max_heads_per_hbm <= 1:
            denom = num_hbm * (1 + softmax_to_gemv)
        else:
            denom = num_hbm * max_heads_per_hbm
        return min(1.0, n_kv_per_gpu / denom)

    def set_accelerator(self, modelinfos, name: DeviceType, config):
        self.hetero_name = name
        if self.hetero_name == DeviceType.PIM:
            ramulator = Ramulator(modelinfos,
                                  "ramulator2",
                                  "ramulator.out",
                                  num_hbm=config['NUM_HBM'],
                                  max_L=self.max_L)
            self.devices['Acc'] = PIM(config,
                                       self.scaling_factor,
                                       ramulator)

        elif self.hetero_name == DeviceType.CPU:
            self.devices['Acc'] = xPU(DeviceType.CPU, config,
                                      self.scaling_factor)

    # Set all device to GPU
    def set_xpu(self, config):
        self.hetero_name = DeviceType.NONE
        self.GPU = xPU(DeviceType.GPU, config, self.scaling_factor)
        self.devices['GPU'] = self.GPU
        self.devices['Acc'] = self.GPU
        self.model.tp = self.GPU.num_xpu

    def simulate(self,
                 batch_size,
                 lin,
                 lout,
                 perfs=None,
                 pipe=False,
                 parallel_ff=False,
                 power_constraint=False,
                 num_reqs=0):

        def add_infos(name, infos, time, energy, bound):
            new_name = name
            if new_name in infos.keys():
                infos[new_name]["time"] += time
                infos[new_name]["energy"] = [
                    eng + energy[i]
                    for i, eng in enumerate(infos[new_name]["energy"])
                ]
            else:
                infos[new_name] = {
                    "time": time,
                    "energy": energy,
                    "bound": bound
                }

        def acc_time(type, exec_times, exec_time):
            if type in exec_times.keys():
                exec_times[type] += exec_time
            else:
                exec_times[type] = exec_time

        def acc_energy(type, energies, energy):
            if type in energies.keys():
                energy_ = energies[type]
                energies[type] = [
                    energy_[i] + energy[i] for i in range(len(energy_))
                ]
            else:
                energies[type] = energy

        def _opb_print(layer, stage_name):
            if OPB_PRINT and layer.off_traffic != 0:
                opb = layer.get_flops() / layer.off_traffic
                tflops = layer.get_flops(
                ) / exec_time / 1000 / 1000 / 1000 / 1000
                print("{},{},{},{},{},{}".format(stage_name, batch_size, lin,
                                                 layer.name, opb, tflops))

        def _pipeline(layers, level=False):
            qkv_time, prj_time, score_time, context_time, x2g_time, softmax_time = 0, 0, 0, 0, 0, 0
            for layer in layers:
                if layer.name in ["qkv"]:
                    qkv_time += layer.exec_time
                elif layer.name in ["proj"]:
                    prj_time += layer.exec_time
                elif layer.name.startswith("comm_x2g"):
                    x2g_time += layer.exec_time
                elif layer.name in ["score"]:
                    score_time += layer.exec_time
                elif layer.name in ["context"]:
                    context_time += layer.exec_time
                elif layer.name in ["softmax"]:
                    softmax_time += layer.exec_time

            heads_per_xpu = max(1, self.model.num_kv_heads / self.GPU.num_xpu)
            minimum_ratio = 1 / heads_per_xpu
            if level == False:
                #softmax_time = 0
                attn_time = score_time + context_time + softmax_time
                if attn_time > x2g_time:
                    x2g_time *= minimum_ratio
                else:
                    x2g_time -= attn_time * (1 - minimum_ratio)

            else:
                #softmax_time = 0
                fc_time = qkv_time + prj_time
                attn_time = score_time + context_time + softmax_time
                if attn_time > fc_time:
                    qkv_time *= minimum_ratio
                    prj_time *= minimum_ratio

                    if attn_time > x2g_time:
                        x2g_time *= minimum_ratio
                    else:
                        x2g_time -= attn_time * (1 - minimum_ratio)
                else:
                    if fc_time > x2g_time:
                        x2g_time *= minimum_ratio
                        qkv_time -= attn_time * (1 - minimum_ratio) * (3 / 4)
                        prj_time -= attn_time * (1 - minimum_ratio) * (1 / 4)
                    else:
                        x2g_time -= attn_time * (1 - minimum_ratio)
                        qkv_time *= minimum_ratio
                        prj_time *= minimum_ratio
            softmax_time = 0

            for layer in layers:
                if layer.name in ["qkv"]:
                    layer.exec_time = qkv_time
                elif layer.name in ["proj"]:
                    layer.exec_time = prj_time
                elif layer.name.startswith("comm_x2g"):
                    # for 2 comm_x2g layers
                    x2g_layers = sum(1 for item in layers
                                     if item.name.startswith("comm_x2g"))
                    layer.exec_time = x2g_time / max(1, x2g_layers)
                elif layer.name in ["softmax"]:
                    layer.exec_time = softmax_time

        def _ff_parallel(layers):
            bw_scale = self.devices['Acc'].peak_memory_bandwidth / self.devices[
                'GPU'].peak_memory_bandwidth
            for layer in layers:
                if "ff" in layer.name:
                    if layer.bound == "compute":
                        attn_flops = self.devices[
                            'GPU'].peak_memory_bandwidth / layer.dbyte * 2 * bw_scale
                        ratio = self.devices['GPU'].peak_flops / (
                            self.devices['GPU'].peak_flops + attn_flops)
                        layer.exec_time *= ratio

                    elif layer.bound == "memory":
                        attn_eff_bw = self.devices[
                            'GPU'].peak_memory_bandwidth * bw_scale / bs
                        ratio = self.devices['GPU'].peak_memory_bandwidth / (
                            self.devices['GPU'].peak_memory_bandwidth +
                            attn_eff_bw)
                        layer.exec_time *= ratio

        s_perf_keys = [
            'all', 'matmul', 'fc', 'comm', 'x2g', 'softmax', 'act', 'norm'
        ]
        g_perf_keys = [
            'all', 'matmul', 'fc', 'comm', 'etc', 'qkv', 'prj', 'ff', 'g2g',
            'x2g', 'softmax', 'act', 'norm'
        ]

        def _zero(keys):
            return {key: 0 for key in keys}

        def _zero_unit_energy():
            return {
                'g_all': 0,
                'g_offmem': 0,
                'g_l2': 0,
                'g_l1': 0,
                'g_reg': 0,
                'g_alu': 0,
                'g_comm': 0
            }

        def _accumulate_dict(dst, src, scale=1):
            for key, value in src.items():
                dst[key] += value * scale

        def _accumulate_gen_energy(gen_energies, layer):
            if layer.type not in gen_energies:
                gen_energies[layer.type] = {'mem': 0, 'comp': 0, 'comm': 0}
            gen_energies[layer.type]['mem'] += layer.energy[0]
            gen_energies[layer.type]['comp'] += sum(layer.energy[1:5])
            gen_energies[layer.type]['comm'] += layer.energy[5]

        def _energy_component(gen_energies, layer_type, component):
            return gen_energies.get(layer_type, {}).get(component, 0)

        def _select_acc_device(layer, group_device):
            if group_device != 'gpu' and layer.type in [
                    LayerType.MATMUL, LayerType.SOFTMAX, LayerType.X2G
            ]:
                return self.devices['Acc']
            return self.devices['GPU']

        def _apply_eff_lat(decoder_block, group_device):
            if group_device != 'pim':
                return
            score_layer = next(
                (layer for layer in decoder_block if layer.name == 'score'),
                None)
            if score_layer is None:
                return
            eff_lat = self.get_pipelining_efficiency_latency(
                getattr(score_layer, 'pim_numOp', score_layer.numOp),
                num_hbm=getattr(self.devices['Acc'], 'num_hbm', 5),
                batch_size=batch_size)
            for layer in decoder_block:
                if layer.name in ['score', 'softmax', 'context']:
                    layer.exec_time /= eff_lat
                    layer.eff_lat = eff_lat

        def _sample_indices(n_items, sample_count):
            if n_items <= sample_count:
                return list(range(n_items))
            indices = {0, n_items - 1}
            denom = max(1, (2**(sample_count - 1) - 1))
            for pos in range(sample_count):
                idx = int(round((n_items - 1) * ((2**pos - 1) / denom)))
                indices.add(idx)
            return sorted(indices)

        def _interp_scalar(points, x):
            points = sorted(points)
            if x <= points[0][0]:
                return points[0][1]
            if x >= points[-1][0]:
                return points[-1][1]
            for idx in range(1, len(points)):
                x0, y0 = points[idx - 1]
                x1, y1 = points[idx]
                if x <= x1:
                    ratio = (x - x0) / max(1, x1 - x0)
                    return y0 + (y1 - y0) * ratio
            return points[-1][1]

        def _interp_vector(points, x):
            return [
                _interp_scalar([(px, py[idx]) for px, py in points], x)
                for idx in range(len(points[0][1]))
            ]

        def _simulate_pim_prefill_score(layer):
            total_l = layer.n
            chunk_size = max(1, min(self.prefill_chunk_size, total_l))
            n_chunks = math.ceil(total_l / chunk_size)
            sample_indices = _sample_indices(n_chunks,
                                             self.prefill_sample_count)
            sampled_time = []
            sampled_energy = []
            for chunk_idx in sample_indices:
                chunk_start = chunk_idx * chunk_size
                chunk_tokens = min(chunk_size, total_l - chunk_start)
                accumulated_l = min(total_l, (chunk_idx + 1) * chunk_size)
                sub_layer = Layer(layer.stage,
                                  layer.name,
                                  layer.type,
                                  layer.has_weight,
                                  layer.dtype,
                                  1,
                                  accumulated_l,
                                  layer.k,
                                  layer.numOp,
                                  pim_numOp=getattr(layer, 'pim_numOp',
                                                    layer.numOp))
                time_per_query, energy_per_query = self.devices[
                    'Acc'].get_time_and_energy(sub_layer)
                sampled_time.append(
                    (chunk_idx, time_per_query * chunk_tokens))
                sampled_energy.append(
                    (chunk_idx,
                     [energy * chunk_tokens for energy in energy_per_query]))

            exec_time = 0
            energy = [0, 0, 0, 0, 0, 0]
            exact = len(sample_indices) == n_chunks
            for chunk_idx in range(n_chunks):
                if exact:
                    idx = sample_indices.index(chunk_idx)
                    chunk_time = sampled_time[idx][1]
                    chunk_energy = sampled_energy[idx][1]
                else:
                    chunk_time = _interp_scalar(sampled_time, chunk_idx)
                    chunk_energy = _interp_vector(sampled_energy, chunk_idx)
                exec_time += chunk_time
                energy = [
                    value + chunk_energy[idx]
                    for idx, value in enumerate(energy)
                ]
            layer.prefill_chunked = True
            layer.prefill_chunk_size = chunk_size
            layer.prefill_sample_count = len(sample_indices)
            return exec_time, energy

        def _simulate_sum_group(s_decoder, group_device):
            time = 0
            wrt_io_busy = 0
            group_s_flops = 0
            for layer in s_decoder:
                device = _select_acc_device(layer, group_device)
                if group_device == 'pim' and layer.name == 'score':
                    exec_time, energy = _simulate_pim_prefill_score(layer)
                else:
                    exec_time, energy = device.get_time_and_energy(layer)
                if layer.type == LayerType.X2G:
                    exec_time += max(wrt_io_busy - time, 0)
                    wrt_io_busy = time + exec_time
                layer.exec_time = exec_time
                layer.energy = energy
                group_s_flops += layer.get_flops() * self.devices[
                    'GPU'].num_xpu
                time += exec_time
                _opb_print(layer, 'sum')
            _apply_eff_lat(s_decoder, group_device)
            return group_s_flops

        def _simulate_gen_group(g_decoder, group_device):
            group_g_flops = 0
            gen_energies = {}
            unit_energy = _zero_unit_energy()
            for gen_stage, decoder_block in enumerate(g_decoder):
                for layer in decoder_block:
                    device = _select_acc_device(layer, group_device)
                    exec_time, energy = device.get_time_and_energy(layer)
                    layer.exec_time = exec_time
                    layer.energy = energy
                    group_g_flops += layer.get_flops() * self.devices[
                        'GPU'].num_xpu
                    if gen_stage == 0:
                        _opb_print(layer, 'gen')

                    _accumulate_gen_energy(gen_energies, layer)
                    unit_energy['g_all'] += sum(layer.energy)
                    unit_energy['g_offmem'] += layer.energy[0]
                    unit_energy['g_l2'] += layer.energy[1]
                    unit_energy['g_l1'] += layer.energy[2]
                    unit_energy['g_reg'] += layer.energy[3]
                    unit_energy['g_alu'] += layer.energy[4]
                    unit_energy['g_comm'] += layer.energy[5]

                if group_device == 'pim':
                    _apply_eff_lat(decoder_block, group_device)
                    _pipeline(decoder_block, pipe)
                    if parallel_ff:
                        _ff_parallel(decoder_block)
            return group_g_flops, gen_energies, unit_energy

        def _collect_s_perf(s_decoder):
            s_perf = _zero(s_perf_keys)
            for layer in s_decoder:
                exec_time = layer.exec_time
                if layer.type == LayerType.FC:
                    s_perf['all'] += exec_time
                    s_perf['fc'] += exec_time
                elif layer.type == LayerType.MATMUL:
                    s_perf['all'] += exec_time
                    s_perf['matmul'] += exec_time
                elif layer.type == LayerType.G2G:
                    s_perf['all'] += exec_time
                    s_perf['comm'] += exec_time
                elif layer.type == LayerType.X2G:
                    s_perf['all'] += exec_time
                    s_perf['comm'] += exec_time
                    s_perf['x2g'] += exec_time
                elif layer.type == LayerType.SOFTMAX:
                    s_perf['all'] += exec_time
                    s_perf['softmax'] += exec_time
                elif layer.type == LayerType.ACT:
                    s_perf['all'] += exec_time
                    s_perf['act'] += exec_time
                elif layer.type == LayerType.NORM:
                    s_perf['all'] += exec_time
                    s_perf['norm'] += exec_time
            return s_perf

        def _collect_g_perf(g_decoder):
            g_perf = _zero(g_perf_keys)
            for decoder_block in g_decoder:
                for layer in decoder_block:
                    exec_time = layer.exec_time
                    g_perf['all'] += exec_time
                    if layer.type == LayerType.FC:
                        g_perf['fc'] += exec_time
                        if 'ff' in layer.name:
                            g_perf['ff'] += exec_time
                        elif 'qkv' in layer.name:
                            g_perf['qkv'] += exec_time
                        elif 'proj' in layer.name:
                            g_perf['prj'] += exec_time
                    elif layer.type == LayerType.MATMUL:
                        g_perf['matmul'] += exec_time
                    elif layer.type in [LayerType.G2G, LayerType.X2G]:
                        g_perf['comm'] += exec_time
                        if 'x2g' in layer.name:
                            g_perf['x2g'] += exec_time
                        elif 'g2g' in layer.name:
                            g_perf['g2g'] += exec_time
                    elif layer.type in [LayerType.ACT, LayerType.NORM]:
                        g_perf['etc'] += exec_time
                        if layer.type == LayerType.ACT:
                            g_perf['act'] += exec_time
                        elif layer.type == LayerType.NORM:
                            g_perf['norm'] += exec_time
                    elif layer.type == LayerType.SOFTMAX:
                        g_perf['softmax'] += exec_time
            return g_perf

        def _collect_energies(gen_energies, unit_energy, gen_steps):
            energies = [
                unit_energy['g_all'], unit_energy['g_offmem'],
                unit_energy['g_l2'], unit_energy['g_l1'], unit_energy['g_reg'],
                unit_energy['g_alu'],
                _energy_component(gen_energies, LayerType.FC, 'mem'),
                _energy_component(gen_energies, LayerType.FC, 'comp'),
                _energy_component(gen_energies, LayerType.MATMUL, 'mem') +
                _energy_component(gen_energies, LayerType.SOFTMAX, 'mem'),
                _energy_component(gen_energies, LayerType.MATMUL, 'comp') +
                _energy_component(gen_energies, LayerType.SOFTMAX, 'comp'),
                _energy_component(gen_energies, LayerType.ACT, 'mem') +
                _energy_component(gen_energies, LayerType.NORM, 'mem'),
                _energy_component(gen_energies, LayerType.ACT, 'comp') +
                _energy_component(gen_energies, LayerType.NORM, 'comp')
            ]
            comm_energy = sum(
                energy['comm'] for energy in gen_energies.values())
            energies.append(comm_energy)
            return [energy / gen_steps for energy in energies]

        assert self.model_set, "Need to set_model"
        routing = None
        if self.routing_mode is not None:
            routing = Routing(self.model,
                              self.routing_mode,
                              self.routing_layer_list,
                              acc_device=self._accelerator_routing_name()
                              ).to_groups(self.model.ndec)
        elif self.hetero_name in [DeviceType.CPU, DeviceType.PIM]:
            if getattr(self.model, 'has_deepstack', False):
                print("WARNING: DeepStack model on hetero path requires "
                      "layer-index preservation. Auto-forcing list routing.")
                layer_list = self.routing_layer_list or list(
                    range(self.model.ndec))
                routing = Routing(self.model,
                                  'list',
                                  layer_list,
                                  acc_device=self._accelerator_routing_name()
                                  ).to_groups(self.model.ndec)
            else:
                routing = [('all', self._accelerator_routing_name(),
                            self.model.ndec, None)]
        self.model.build(batch_size,
                         lin,
                         lout,
                         self.hetero_name in [DeviceType.CPU, DeviceType.PIM],
                         routing=routing,
                         image_size=self.image_size)
        second_batch_size = num_reqs % batch_size
        num_batches = 1
        target_bs = [batch_size]
        if num_reqs > 0:
            num_batches = int(num_reqs / batch_size)
            if second_batch_size > 0:
                target_bs = [batch_size, second_batch_size]

        perf_all = []
        energy_all = []
        s_flops = 0
        g_flops = 0
        gen_steps = max(1, lout - 1)
        cap_usage_per_gpu = 0
        for itr, bs in enumerate(target_bs):
            s_perf_total = _zero(s_perf_keys)
            g_perf_total = _zero(g_perf_keys)
            energy_total = [0 for _ in range(13)]
            s_flops_total = 0
            g_flops_total = 0

            if self.model.vision_decoder:
                vision_flops = _simulate_sum_group(self.model.vision_decoder,
                                                   'gpu')
                vision_perf = _collect_s_perf(self.model.vision_decoder)
                _accumulate_dict(s_perf_total, vision_perf, 1)
                s_flops_total += vision_flops
                # Fix B: VLM floor overhead absorbing image preprocessing,
                # RoPE, CUDA kernel launches, scheduler -- added once per
                # prefill, only when a vision graph exists.  Default 0.0
                # for LLMs leaves LLM path bit-identical.
                floor_ms = getattr(self.model,
                                    'vlm_floor_overhead_ms', 0.0)
                if floor_ms > 0:
                    s_perf_total['all'] += floor_ms / 1000.0

            for group_name, group in self.model.sum_decoder_groups.items():
                s_decoder, count, group_device, _ = group
                g_decoder = self.model.gen_decoder_groups[group_name][0]

                group_s_flops = _simulate_sum_group(s_decoder, group_device)
                group_g_flops, gen_energies, unit_energy = _simulate_gen_group(
                    g_decoder, group_device)
                s_perf = _collect_s_perf(s_decoder)
                g_perf = _collect_g_perf(g_decoder)
                g_perf = {key: value / gen_steps for key, value in g_perf.items()}
                energies = _collect_energies(gen_energies, unit_energy,
                                             gen_steps)

                _accumulate_dict(s_perf_total, s_perf, count)
                _accumulate_dict(g_perf_total, g_perf, count)
                energy_total = [
                    value + energies[idx] * count
                    for idx, value in enumerate(energy_total)
                ]
                s_flops_total += group_s_flops * count
                g_flops_total += group_g_flops * count

            perf = [s_perf_total[key] for key in s_perf_keys] + [
                g_perf_total[key] for key in g_perf_keys
            ]

            cap_usage_per_gpu = sum(
                self.get_required_mem_capacity(bs, lin, lout))

            ## Perf: ms, energy: nJ
            perf = [t * 1000 for t in perf]
            energies = [t / 1000 for t in energy_total]
            s_flops = s_flops_total / gen_steps
            g_flops = g_flops_total / gen_steps

            if itr == 0:
                if len(perf_all) > 0:
                    perf_all = [
                        v + perf[i] * num_batches
                        for i, v in enumerate(perf_all)
                    ]
                    energy_all = [
                        v + energies[i] * num_batches
                        for i, v in enumerate(energy_all)
                    ]
                else:
                    perf_all = copy.deepcopy(perf)
                    energy_all = copy.deepcopy(energies)
            else:
                perf_all = [v + perf[i] for i, v in enumerate(perf_all)]
                energy_all = [
                    v + energies[i] for i, v in enumerate(energy_all)
                ]

        ## Concat tag
        cap = self.devices['GPU'].aggregate_memory_capacity
        if self.hetero_name in [DeviceType.CPU, DeviceType.PIM]:
            cap += self.devices['Acc'].aggregate_memory_capacity
        cap = int(cap / (1024 * 1024 * 1024))
        bw_scale = self.devices['Acc'].peak_memory_bandwidth / self.devices[
            'GPU'].peak_memory_bandwidth

        opb = self.devices['GPU'].peak_flops / self.devices[
            'GPU'].peak_memory_bandwidth
        if self.model.dtype == DataType.W8A8:
            opb *= 2

        tag = [
            self.model.name, self.model.dtype.name,
            self.devices['GPU'].name.name, cap, bw_scale, opb
        ]
        config = [
            self.hetero_name.name, self.devices['GPU'].num_xpu, pipe,
            parallel_ff, power_constraint, self.model.gqa_size, lin, lout,
            batch_size, cap_usage_per_gpu, s_flops, g_flops
        ]
        if self.hetero_name == DeviceType.PIM:
            config[0] = self.devices['Acc'].pim_type.name

        output = [tag, config, perf_all, energy_all]
        print(
            "    Batch: {}, Throughput: {:.2f} tokens/s Latency: {:.2f}ms, pipe/ff_parallel: {}/{}, powerlimit: {}"
            .format(batch_size,
                    batch_size / ((perf_all[len(s_perf_keys)]) / 1000),
                    perf_all[len(s_perf_keys)], pipe, parallel_ff,
                    power_constraint))

        if perfs is not None:
            perfs.append(output)
        else:
            perfs = [output]

    def _compute_mem_components(self, batch_size, lin, lout):
        """Compute weight/temp/kv memory components.  All values are per
        TP shard, in bytes.  KV is the TOTAL across batch (per-shard).
        System-aware splitting (which device holds KV) is applied by
        callers via the kv_on_attacc() helper below.
        """
        ndec = self.model.ndec
        hdim = self.model.hdim
        nhead = self.model.num_q_heads
        n_kv = self.model.num_kv_heads
        dhead = self.model.dhead
        ff_intermediate = self.model.ff_intermediate
        w_byte = 2 if self.model.dtype in [DataType.W16A16, DataType.W16A8
                                          ] else 1
        a_byte = 2 if self.model.dtype in [DataType.W16A16, DataType.W8A16
                                          ] else 1
        l = lin + lout - 1

        qkv_weight = hdim * self.model.qkv_proj_out_total
        attn_out_weight = self.model.q_proj_out * hdim
        if self.model.ffn_type == 'gated':
            ff_weight = hdim * ff_intermediate * 3
        else:
            ff_weight = hdim * ff_intermediate * 2
        weight_memory = ndec * (qkv_weight + attn_out_weight +
                                ff_weight) * w_byte

        temp_memory = max((hdim + l * nhead) * a_byte, hdim * 2 * a_byte,
                          l * nhead * 2 * a_byte,
                          (ff_intermediate + hdim) * a_byte) + l * nhead * a_byte
        kv_memory = ndec * 2 * l * n_kv * dhead * a_byte

        weight_memory = weight_memory / self.model.fc_tp
        kv_memory = kv_memory / self.model.attn_tp

        return weight_memory, kv_memory * batch_size, temp_memory * batch_size

    def kv_on_attacc(self):
        """Fix C: dgx-attacc places KV cache on the AttAcc HBM side, freeing
        GPU capacity for weights only.  This matches paper Sec.7.1's
        DGX+AttAcc system assumption.
        """
        return self.hetero_name == DeviceType.PIM

    def get_required_mem_capacity(self, batch_size, lin, lout):
        """Per-GPU memory pressure (bytes).  Backward-compat 3-tuple.

        Returns (weight_per_gpu, kv_on_gpu, temp_per_gpu).  Under dgx-attacc
        KV is on the AttAcc side, so kv_on_gpu becomes 0.  Use
        get_attacc_kv_capacity() for the AttAcc-side KV total.
        """
        weight_memory, kv_memory, temp_memory = self._compute_mem_components(
            batch_size, lin, lout)
        kv_on_gpu = 0.0 if self.kv_on_attacc() else kv_memory
        return weight_memory, kv_on_gpu, temp_memory

    def get_attacc_kv_capacity(self, batch_size, lin, lout):
        """KV cache bytes residing on AttAcc HBM (0 if not dgx-attacc)."""
        if not self.kv_on_attacc():
            return 0.0
        _, kv_memory, _ = self._compute_mem_components(
            batch_size, lin, lout)
        return kv_memory

    def get_capacity_breakdown(self, batch_size, lin, lout):
        """Per-device capacity breakdown.

        Key semantics:
          weight_per_gpu, kv_per_gpu, temp_per_gpu, available_kv
            -- always GPU-side bytes.  Under dgx-attacc, kv_per_gpu = 0
               and available_kv reflects gpu_cap - weight - temp.
          kv_per_attacc, available_kv_attacc, attacc_capacity_total,
          max_batch_at_default_L_attacc
            -- AttAcc-side bytes, populated only when kv_on_attacc().
          max_batch_at_default_L
            -- SYSTEM limiting batch (not GPU-only).  Under dgx this is
               the GPU-side ceiling (same as before Fix C).  Under
               dgx-attacc this is overwritten to the AttAcc-side ceiling
               because the GPU side has 0 KV bytes residing on it.
               Callers like capacity_regime.py rely on this for
               unchanged interpretation across systems.
        """
        weight_memory, kv_per_gpu, temp_memory = self.get_required_mem_capacity(
            batch_size, lin, lout)
        kv_per_attacc = self.get_attacc_kv_capacity(batch_size, lin, lout)

        gpu_cap_per_device = (self.devices['GPU'].aggregate_memory_capacity /
                              max(1, self.devices['GPU'].num_xpu))
        available_gpu = gpu_cap_per_device - weight_memory - temp_memory - kv_per_gpu
        # When KV is on AttAcc, "available_kv" on GPU is essentially
        # whatever the GPU has left after weight+temp.
        available_kv_gpu = gpu_cap_per_device - weight_memory - temp_memory

        kv_per_req_gpu = (kv_per_gpu / max(1, batch_size)
                          if kv_per_gpu > 0 else 0)

        # Backward-compat: max_batch_at_default_L is the SYSTEM limiting
        # max batch -- whichever device holds the KV cache.  Pre-Fix-C
        # callers (e.g. capacity_regime.py) read this key directly; they
        # expect a system-wide ceiling, not GPU-only zero.
        gpu_max_batch = (int(available_kv_gpu / kv_per_req_gpu)
                          if kv_per_req_gpu > 0 else 0)

        result = {
            'weight_per_gpu': weight_memory,
            'kv_per_gpu': kv_per_gpu,
            'temp_per_gpu': temp_memory,
            'available_kv': available_kv_gpu,
            'max_batch_at_default_L': gpu_max_batch,  # may be overwritten below
        }
        if self.kv_on_attacc():
            acc_dev = self.devices.get('Acc')
            acc_cap = (acc_dev.aggregate_memory_capacity
                       if acc_dev is not None else 0)
            kv_per_req_attacc = (kv_per_attacc / max(1, batch_size)
                                 if kv_per_attacc > 0 else 0)
            available_kv_attacc = acc_cap - kv_per_attacc
            attacc_max_batch = (int(acc_cap / kv_per_req_attacc)
                                if kv_per_req_attacc > 0 else 0)
            result.update({
                'kv_per_attacc': kv_per_attacc,
                'available_kv_attacc': available_kv_attacc,
                'attacc_capacity_total': acc_cap,
                'max_batch_at_default_L_attacc': attacc_max_batch,
                # Limiting batch = min(GPU side, AttAcc side).  GPU side is
                # bounded by weight+temp (we model 0 KV bytes on GPU); we
                # ignore that floor here since it's typically much higher
                # than the AttAcc-side KV ceiling for VLM workloads.
                'max_batch_at_default_L': attacc_max_batch,
            })
        return result

