## Define models and layer.
## Generate models
from .type import *
import copy
import math


class Layer:

    def __init__(self,
                 stage,
                 name,
                 type,
                 has_weight,
                 dtype,
                 m,
                 n,
                 k,
                 numOp,
                 pim_numOp=None):
        self.stage = stage
        self.name = name
        self.type = type
        self.has_weight = has_weight
        self.m = int(m)
        self.n = int(n)
        self.k = int(k)
        self.numOp = int(numOp)
        self.pim_numOp = int(pim_numOp) if pim_numOp is not None else int(numOp)
        self.dtype = dtype
        self.dbyte = 2
        if dtype in [DataType.W16A16]:
            self.dbyte = 2
        elif dtype in [DataType.W8A8]:
            self.dbyte = 1
        else:
            assert 0, "Only support W16A16, W8A8"
        self.bound = 'compute'  # 'memory'
        self.exec_time = 0
        self.energy = 0

        assert isinstance(type, LayerType), "Not support layer type"
        assert isinstance(dtype, DataType), "Not support data type"

    def get_infos(self):
        return self.m, self.n, self.k, self.numOp, self.dbyte

    def get_flops(self):
        if self.type == LayerType.SOFTMAX:
            return 5 * self.m * self.n * self.numOp

        elif self.type == LayerType.ACT:
            if 'relu' in self.name:
                return 1 * self.m * self.n * self.numOp
            elif 'glu' in self.name:
                return (8 + 1) * self.m * self.n * self.numOp
            else:
                return 8 * self.m * self.n * self.numOp

        elif self.type == LayerType.NORM:
            return 5 * self.m * self.n * self.numOp

        elif self.type in [LayerType.FC, LayerType.MATMUL]:
            return 2 * self.m * self.n * self.k * self.numOp

        elif self.type in [LayerType.G2G, LayerType.X2G]:
            return 0

        else:
            assert 0, "In Function \"get_flops\": Not support layer type"

    def get_size(self):
        in1 = self.numOp * self.m * self.k * self.dbyte
        in2 = self.numOp * self.n * self.k * self.dbyte
        out = self.numOp * self.m * self.n * self.dbyte

        if self.type in [
                LayerType.SOFTMAX, LayerType.ACT, LayerType.G2G, LayerType.X2G
        ]:
            in1 = self.numOp * self.m * self.n * self.dbyte
            in2 = 0
            out = in1

            # For SwiGLU and GeGLU
            if 'glu' in self.name:
                in2 = in1

        elif self.type == LayerType.NORM:
            in1 = self.numOp * self.m * self.n * self.dbyte
            in2 = in1
            out = in1

        return in1, in2, out


class Routing:
    DEFAULT_LAYER_LIST = [
        0, 8, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
        27, 28, 29, 31, 33
    ]

    def __init__(self,
                 model,
                 mode='conservative',
                 layer_list=None,
                 acc_device='pim'):
        self.mode = mode
        self.layer_list = layer_list or self.DEFAULT_LAYER_LIST
        self.acc_device = acc_device
        self.has_deepstack = getattr(model, 'has_deepstack', False)
        if self.has_deepstack and self.mode != 'list':
            print("WARNING: DeepStack model requires layer indices. "
                  "Forcing routing mode to list.")
            self.mode = 'list'

    def to_groups(self, ndec):
        layer_set = {idx for idx in self.layer_list if 0 <= idx < ndec}
        if self.mode == 'conservative':
            return [('all', self.acc_device, ndec, None)]

        elif self.mode == 'optimistic':
            acc_count = len(layer_set)
            groups = []
            if acc_count > 0:
                groups.append(('acc', self.acc_device, acc_count, None))
            if ndec - acc_count > 0:
                groups.append(('gpu', 'gpu', ndec - acc_count, None))
            return groups

        elif self.mode == 'list':
            return [('l{}'.format(idx),
                     self.acc_device if idx in layer_set else 'gpu', 1,
                     [idx]) for idx in range(ndec)]

        assert 0, "Invalid routing mode: {}".format(self.mode)


class Transformer:

    def __init__(self, modelinfos, tensor_parallel=8):
        self.sum_decoder = []
        self.gen_decoder = []
        self.vision_decoder = []
        self.sum_decoder_groups = {}
        self.gen_decoder_groups = {}
        self.routing_meta = []
        if isinstance(modelinfos, list):
            modelinfos = self._list_to_dict(modelinfos)
        self.name = modelinfos['name']
        self.ndec = modelinfos['ndec']
        self.hdim = modelinfos['hdim']
        self.num_q_heads = modelinfos.get('num_q_heads',
                                          modelinfos.get('num_heads'))
        self.num_kv_heads = modelinfos.get('num_kv_heads', self.num_q_heads)
        self.dhead = modelinfos.get('dhead',
                                    int(self.hdim / self.num_q_heads))
        self.q_proj_out = modelinfos.get('q_proj_out',
                                         self.num_q_heads * self.dhead)
        self.kv_proj_out = modelinfos.get('kv_proj_out',
                                          self.num_kv_heads * self.dhead)
        self.qkv_proj_out_total = modelinfos.get(
            'qkv_proj_out_total', self.q_proj_out + 2 * self.kv_proj_out)
        self.gqa_size = modelinfos.get('gqa_size',
                                       int(self.num_q_heads /
                                           self.num_kv_heads))
        self.ff_scale = modelinfos['ff_scale']
        self.ff_intermediate = int(
            modelinfos.get('ff_intermediate', int(self.ff_scale * self.hdim)))
        self.ffn_type = modelinfos.get('ffn_type', 'standard')
        self.activation = modelinfos.get('activation', 'gelu')
        self.dtype = modelinfos['dtype']
        self.tp_arg = tensor_parallel
        self.fc_tp = self.tp_arg
        self.attn_tp = min(self.tp_arg, self.num_kv_heads)
        self.ff_tp = self.tp_arg
        if self.attn_tp != self.tp_arg:
            print("WARNING: attn_tp clamped to num_kv_heads={} (ngpu={}). "
                  "qkv FC sharded {}-way but attn only uses {}-way. "
                  "Edge-case (R5): KV repartition between FC output and "
                  "attention input is analytical fallback only.".format(
                      self.num_kv_heads, self.tp_arg, self.fc_tp,
                      self.attn_tp))
        self.num_heads = self.num_q_heads  # back-compat alias
        self.tp = self.fc_tp
        self.has_deepstack = modelinfos.get('has_deepstack', False)
        self.deepstack_layers = modelinfos.get('deepstack_layers', [])
        self.vit_layers = modelinfos.get('vit_layers', 0)
        self.vit_hidden = modelinfos.get('vit_hidden', 0)
        self.vit_num_heads = modelinfos.get('vit_num_heads', 0)
        self.vit_intermediate = modelinfos.get('vit_intermediate', 0)
        self.vit_out_hidden = modelinfos.get('vit_out_hidden', self.hdim)
        self.vit_activation = modelinfos.get('vit_activation', 'gelu')
        self.projector_type = modelinfos.get('projector_type', None)
        self.patch_size = modelinfos.get('patch_size', 0)
        self.image_size_default = modelinfos.get('image_size_default', 0)
        self.spatial_merge_size = modelinfos.get('spatial_merge_size', 1)
        self.num_vis_tokens_per_image = modelinfos.get(
            'num_vis_tokens_per_image', 0)
        self.is_anyres = modelinfos.get('is_anyres', False)
        self.image_grid_pinpoints = modelinfos.get('image_grid_pinpoints', [])
        self.use_image_newline_parameter = modelinfos.get(
            'use_image_newline_parameter', False)
        # VLM-side floor overhead absorbing image preprocessing, RoPE,
        # CUDA kernel launches, vLLM scheduler overhead.  Calibrated
        # from LLaVA-1.5 (measured 41.2ms - simulated 18.7ms = 22.5ms);
        # treat as HW-independent first-order constant.  LLM-only models
        # leave this 0 (default), so LLM path is bit-identical.
        self.vlm_floor_overhead_ms = modelinfos.get(
            'vlm_floor_overhead_ms', 0.0)
        self.modelinfos = modelinfos

    @staticmethod
    def _list_to_dict(modelinfos):
        ndec, hdim, nheads, dhead, ff_scale, gqa_size = modelinfos
        return {
            'name': 'LEGACY',
            'ndec': ndec,
            'hdim': hdim,
            'num_heads': nheads,
            'num_q_heads': nheads,
            'num_kv_heads': int(nheads / gqa_size),
            'dhead': dhead,
            'ff_scale': ff_scale,
            'gqa_size': gqa_size,
            'dtype': DataType.W16A16,
        }

    @staticmethod
    def _split(value, parts, field):
        assert value % parts == 0, "{}={} is not divisible by tp={}".format(
            field, value, parts)
        return int(value / parts)

    def _heads_per_attn_shard(self, batch):
        q_heads = self._split(self.num_q_heads, self.attn_tp, 'num_q_heads')
        kv_heads = self._split(self.num_kv_heads, self.attn_tp,
                               'num_kv_heads')
        return q_heads * batch, kv_heads * batch

    def _activation_name(self):
        if self.activation in ['relu', 'gelu']:
            return self.activation
        return 'glu'

    def _vit_activation_name(self):
        if self.vit_activation in ['relu', 'gelu', 'silu']:
            return 'vit_{}'.format(self.vit_activation)
        return 'vit_gelu'

    def _append_ffn(self, decoder, stage, tokens):
        ff_per_tp = self._split(self.ff_intermediate, self.ff_tp,
                                'ff_intermediate')
        if self.ffn_type == 'gated':
            decoder.append(
                Layer(stage, 'ff1', LayerType.FC, True, self.dtype, tokens,
                      ff_per_tp, self.hdim, 1))
            decoder.append(
                Layer(stage, 'ff2', LayerType.FC, True, self.dtype, tokens,
                      ff_per_tp, self.hdim, 1))
            decoder.append(
                Layer(stage, 'glu', LayerType.ACT, False, self.dtype, tokens,
                      ff_per_tp, 1, 1))
            decoder.append(
                Layer(stage, 'ff3', LayerType.FC, True, self.dtype, tokens,
                      self.hdim, ff_per_tp, 1))
        else:
            decoder.append(
                Layer(stage, 'ff1', LayerType.FC, True, self.dtype, tokens,
                      ff_per_tp, self.hdim, 1))
            decoder.append(
                Layer(stage, self._activation_name(), LayerType.ACT, False,
                      self.dtype, tokens, ff_per_tp, 1, 1))
            decoder.append(
                Layer(stage, 'ff2', LayerType.FC, True, self.dtype, tokens,
                      self.hdim, ff_per_tp, 1))

    @staticmethod
    def _normalize_image_size(image_size):
        if image_size is None:
            return None
        if isinstance(image_size, (list, tuple)):
            return int(image_size[0]), int(image_size[1])
        return int(image_size), int(image_size)

    @staticmethod
    def select_best_resolution(image_size, possible_resolutions):
        original_w, original_h = image_size
        best_fit = possible_resolutions[0]
        max_effective = -1
        min_wasted = None
        for width, height in possible_resolutions:
            scale = min(width / original_w, height / original_h)
            down_w = min(int(original_w * scale), width)
            down_h = min(int(original_h * scale), height)
            effective = min(down_w * down_h, original_w * original_h)
            wasted = width * height - effective
            if effective > max_effective or (effective == max_effective and
                                             (min_wasted is None or
                                              wasted < min_wasted)):
                max_effective = effective
                min_wasted = wasted
                best_fit = [width, height]
        return best_fit

    def compute_visual_tokens(self, image_size=None):
        if self.num_vis_tokens_per_image == 0:
            return 0
        image_size = self._normalize_image_size(image_size)
        if not self.is_anyres or not self.image_grid_pinpoints:
            if image_size is not None and self.patch_size > 0:
                width, height = image_size
                raw = (width // self.patch_size) * (height // self.patch_size)
                merge = self.spatial_merge_size * self.spatial_merge_size
                return max(1, int(raw / merge))
            return self.num_vis_tokens_per_image

        if image_size is None:
            image_size = (self.image_size_default, self.image_size_default)
        grid_w, grid_h = self.select_best_resolution(
            image_size, self.image_grid_pinpoints)
        n_patches = (grid_w // self.patch_size) * (grid_h // self.patch_size)
        base = (self.image_size_default // self.patch_size)**2
        if self.use_image_newline_parameter:
            n_patches += grid_h // self.patch_size
        return n_patches + base

    def compute_vit_tokens(self, image_size=None):
        if self.num_vis_tokens_per_image == 0 or self.patch_size == 0:
            return 0
        image_size = self._normalize_image_size(image_size)
        if self.is_anyres and self.image_grid_pinpoints:
            return (self.image_size_default // self.patch_size)**2
        if image_size is None:
            if self.spatial_merge_size > 1:
                return self.num_vis_tokens_per_image * (
                    self.spatial_merge_size * self.spatial_merge_size)
            image_size = (self.image_size_default, self.image_size_default)
        width, height = image_size
        return (width // self.patch_size) * (height // self.patch_size)

    def compute_vit_attention_tokens(self, image_size=None):
        # ViT runs attention at full patch-token count. Spatial merging
        # (2x2 -> 1, etc.) happens at the projector boundary, modeled by
        # the merger FC in _build_projector (in_dim = vit_hidden * merge^2).
        # Dividing here by spatial_merge^2 was a bug that under-counted
        # attention compute by spatial_merge^4 (e.g. 16x for Qwen2.5-VL,
        # which has spatial_merge_size=2). See calibration/ for s_corr
        # before/after this fix.
        return self.compute_vit_tokens(image_size)

    def compute_vit_num_images(self, image_size=None):
        if self.num_vis_tokens_per_image == 0:
            return 0
        image_size = self._normalize_image_size(image_size)
        if not self.is_anyres or not self.image_grid_pinpoints:
            return 1
        if image_size is None:
            image_size = (self.image_size_default, self.image_size_default)
        grid_w, grid_h = self.select_best_resolution(
            image_size, self.image_grid_pinpoints)
        base = (self.image_size_default // self.patch_size)**2
        grid_patches = (grid_w // self.patch_size) * (
            grid_h // self.patch_size)
        return 1 + int(math.ceil(grid_patches / max(1, base)))

    def _build_vit(self, batch, image_size=None):
        decoder = []
        tokens = self.compute_vit_tokens(image_size)
        if tokens == 0 or self.vit_layers == 0:
            return decoder

        attn_tokens = self.compute_vit_attention_tokens(image_size)
        num_images = self.compute_vit_num_images(image_size)
        block_ops = self.vit_layers * max(1, num_images)
        attn_ops = self.vit_num_heads * batch * block_ops
        vit_dhead = int(self.vit_hidden / self.vit_num_heads)
        decoder.append(
            Layer('vit', 'vit_qkv', LayerType.FC, True, self.dtype,
                  batch * tokens, 3 * self.vit_hidden, self.vit_hidden,
                  block_ops))
        decoder.append(
            Layer('vit', 'vit_score', LayerType.MATMUL, False, self.dtype,
                  attn_tokens, attn_tokens, vit_dhead, attn_ops))
        decoder.append(
            Layer('vit', 'vit_softmax', LayerType.SOFTMAX, False, self.dtype,
                  attn_tokens, attn_tokens, 1, attn_ops))
        decoder.append(
            Layer('vit', 'vit_context', LayerType.MATMUL, False, self.dtype,
                  attn_tokens, vit_dhead, attn_tokens, attn_ops))
        decoder.append(
            Layer('vit', 'vit_proj', LayerType.FC, True, self.dtype,
                  batch * tokens, self.vit_hidden, self.vit_hidden,
                  block_ops))
        decoder.append(
            Layer('vit', 'vit_norm1', LayerType.NORM, False, self.dtype,
                  batch * tokens, self.vit_hidden, 1, block_ops))
        decoder.append(
            Layer('vit', 'vit_ff1', LayerType.FC, True, self.dtype,
                  batch * tokens, self.vit_intermediate, self.vit_hidden,
                  block_ops))
        decoder.append(
            Layer('vit', self._vit_activation_name(), LayerType.ACT, False,
                  self.dtype, batch * tokens, self.vit_intermediate, 1,
                  block_ops))
        decoder.append(
            Layer('vit', 'vit_ff2', LayerType.FC, True, self.dtype,
                  batch * tokens, self.vit_hidden, self.vit_intermediate,
                  block_ops))
        decoder.append(
            Layer('vit', 'vit_norm2', LayerType.NORM, False, self.dtype,
                  batch * tokens, self.vit_hidden, 1, block_ops))
        return decoder

    def _build_projector(self, batch, image_size=None):
        decoder = []
        tokens = self.compute_visual_tokens(image_size)
        if tokens == 0 or self.projector_type is None:
            return decoder

        if self.projector_type == 'mlp_with_merger':
            in_dim = self.vit_hidden * self.spatial_merge_size * self.spatial_merge_size
            decoder.append(
                Layer('projector', 'proj_merger_fc1', LayerType.FC, True,
                      self.dtype, batch * tokens, self.hdim, in_dim, 1))
            decoder.append(
                Layer('projector', 'proj_merger_act', LayerType.ACT, False,
                      self.dtype, batch * tokens, self.hdim, 1, 1))
            decoder.append(
                Layer('projector', 'proj_merger_fc2', LayerType.FC, True,
                      self.dtype, batch * tokens, self.hdim, self.hdim, 1))
        elif self.projector_type == 'pixel_shuffle_mlp':
            in_dim = self.vit_hidden * self.spatial_merge_size * self.spatial_merge_size
            decoder.append(
                Layer('projector', 'proj_pixel_fc1', LayerType.FC, True,
                      self.dtype, batch * tokens, self.hdim, in_dim, 1))
            decoder.append(
                Layer('projector', 'proj_pixel_act', LayerType.ACT, False,
                      self.dtype, batch * tokens, self.hdim, 1, 1))
            decoder.append(
                Layer('projector', 'proj_pixel_fc2', LayerType.FC, True,
                      self.dtype, batch * tokens, self.hdim, self.hdim, 1))
        else:
            decoder.append(
                Layer('projector', 'proj_mlp_fc1', LayerType.FC, True,
                      self.dtype, batch * tokens, self.hdim, self.vit_hidden,
                      1))
            decoder.append(
                Layer('projector', 'proj_mlp_act', LayerType.ACT, False,
                      self.dtype, batch * tokens, self.hdim, 1, 1))
            decoder.append(
                Layer('projector', 'proj_mlp_fc2', LayerType.FC, True,
                      self.dtype, batch * tokens, self.hdim, self.hdim, 1))

        if self.fc_tp > 1:
            decoder.append(
                Layer('projector', 'vit_broadcast', LayerType.G2G, False,
                      self.dtype, batch * tokens, self.hdim, 1, 1))
        return decoder

    def _build_vision(self, batch, image_size=None):
        decoder = []
        decoder.extend(self._build_vit(batch, image_size))
        decoder.extend(self._build_projector(batch, image_size))
        return decoder

    def _build_sum(self, batch, lin, attn_on_hetero, layer_idx=None):
        decoder = []
        q_heads, kv_heads = self._heads_per_attn_shard(batch)
        qkv_per_tp = self._split(self.qkv_proj_out_total, self.fc_tp,
                                 'qkv_proj_out_total')
        q_per_fc_tp = self._split(self.q_proj_out, self.fc_tp, 'q_proj_out')
        kv_per_fc_tp = self._split(self.kv_proj_out, self.fc_tp,
                                   'kv_proj_out')
        q_per_attn_tp = self._split(self.q_proj_out, self.attn_tp,
                                    'q_proj_out')

        decoder.append(
            Layer('sum', 'qkv', LayerType.FC, True, self.dtype, batch * lin,
                  qkv_per_tp, self.hdim, 1))
        if attn_on_hetero:
            decoder.append(
                Layer('sum', 'comm_x2g_kv', LayerType.X2G, False, self.dtype,
                      batch * lin, 2 * kv_per_fc_tp, 1, 1))
            decoder.append(
                Layer('sum', 'comm_x2g_q', LayerType.X2G, False, self.dtype,
                      batch * lin, q_per_fc_tp, 1, 1))

        # Layer shape contract for ramulator_wrapper:
        #   layer.n = accumulated KV length (L); layer.k = dhead
        #   layer.numOp = Q heads for GPU/FLOPs
        #   layer.pim_numOp = KV heads for PIM trace
        #   layer.m = query count (externally scaled for chunked prefill)
        decoder.append(
            Layer('sum', 'score', LayerType.MATMUL, False, self.dtype, lin,
                  lin, self.dhead, q_heads, pim_numOp=kv_heads))
        decoder.append(
            Layer('sum', 'softmax', LayerType.SOFTMAX, False, self.dtype, lin,
                  lin, 1, q_heads, pim_numOp=kv_heads))
        decoder.append(
            Layer('sum', 'context', LayerType.MATMUL, False, self.dtype, lin,
                  self.dhead, lin, q_heads, pim_numOp=kv_heads))
        if layer_idx in self.deepstack_layers:
            decoder.append(
                Layer('sum', 'deepstack_add', LayerType.ACT, False,
                      self.dtype, batch * lin, self.hdim, 1, 1))
        if attn_on_hetero:
            decoder.append(
                Layer('sum', 'comm_x2g_return', LayerType.X2G, False,
                      self.dtype, batch * lin, q_per_attn_tp, 1, 1))

        decoder.append(
            Layer('sum', 'proj', LayerType.FC, True, self.dtype, batch * lin,
                  self.hdim, self._split(self.hdim, self.fc_tp, 'hdim'), 1))
        decoder.append(
            Layer('sum', 'comm_g2g', LayerType.G2G, False, self.dtype,
                  batch * lin, self.hdim, 1, 1))
        decoder.append(
            Layer('sum', 'norm1', LayerType.NORM, False, self.dtype,
                  batch * lin, self.hdim, 1, 1))
        self._append_ffn(decoder, 'sum', batch * lin)
        decoder.append(
            Layer('sum', 'comm_g2g', LayerType.G2G, False, self.dtype,
                  batch * lin, self.hdim, 1, 1))
        decoder.append(
            Layer('sum', 'norm2', LayerType.NORM, False, self.dtype,
                  batch * lin, self.hdim, 1, 1))
        return decoder

    def _build_gen_stage(self, batch, lin, stage, attn_on_hetero):
        decoder = []
        q_heads, kv_heads = self._heads_per_attn_shard(batch)
        qkv_per_tp = self._split(self.qkv_proj_out_total, self.fc_tp,
                                 'qkv_proj_out_total')
        q_per_attn_tp = self._split(self.q_proj_out, self.attn_tp,
                                    'q_proj_out')
        kv_len = lin + stage

        decoder.append(
            Layer('gen', 'qkv', LayerType.FC, True, self.dtype, batch,
                  qkv_per_tp, self.hdim, 1))
        if attn_on_hetero:
            decoder.append(
                Layer('gen', 'comm_x2g_qkv', LayerType.X2G, False,
                      self.dtype, batch, qkv_per_tp, 1, 1))
        decoder.append(
            Layer('gen', 'score', LayerType.MATMUL, False, self.dtype, 1,
                  kv_len, self.dhead, q_heads, pim_numOp=kv_heads))
        decoder.append(
            Layer('gen', 'softmax', LayerType.SOFTMAX, False, self.dtype, 1,
                  kv_len, 1, q_heads, pim_numOp=kv_heads))
        decoder.append(
            Layer('gen', 'context', LayerType.MATMUL, False, self.dtype, 1,
                  self.dhead, kv_len, q_heads, pim_numOp=kv_heads))
        if attn_on_hetero:
            decoder.append(
                Layer('gen', 'comm_x2g_return', LayerType.X2G, False,
                      self.dtype, batch, q_per_attn_tp, 1, 1))
        decoder.append(
            Layer('gen', 'proj', LayerType.FC, True, self.dtype, batch,
                  self.hdim, self._split(self.hdim, self.fc_tp, 'hdim'), 1))
        decoder.append(
            Layer('gen', 'comm_g2g', LayerType.G2G, False, self.dtype, batch,
                  self.hdim, 1, 1))
        decoder.append(
            Layer('gen', 'norm1', LayerType.NORM, False, self.dtype, batch,
                  self.hdim, 1, 1))
        self._append_ffn(decoder, 'gen', batch)
        decoder.append(
            Layer('gen', 'comm_g2g', LayerType.G2G, False, self.dtype, batch,
                  self.hdim, 1, 1))
        decoder.append(
            Layer('gen', 'norm2', LayerType.NORM, False, self.dtype, batch,
                  self.hdim, 1, 1))
        return decoder

    @staticmethod
    def _parse_routing_entry(entry):
        if len(entry) == 3:
            group_name, device, count = entry
            indices = None
        else:
            group_name, device, count, indices = entry
        return group_name, device, int(count), indices

    def build(self,
              batch,
              lin,
              lout,
              attn_on_hetero=False,
              routing=None,
              image_size=None):
        self.sum_decoder = []
        self.gen_decoder = []
        self.vision_decoder = self._build_vision(batch, image_size)
        self.sum_decoder_groups = {}
        self.gen_decoder_groups = {}
        self.routing_meta = []

        if routing is None:
            if self.has_deepstack:
                device = 'pim' if attn_on_hetero else 'gpu'
                routing = [('l{}'.format(idx), device, 1, [idx])
                           for idx in range(self.ndec)]
            else:
                device = 'pim' if attn_on_hetero else 'gpu'
                routing = [('all', device, self.ndec, None)]

        for entry in routing:
            group_name, device, count, indices = self._parse_routing_entry(
                entry)
            if count <= 0:
                continue
            group_on_hetero = device != 'gpu'
            layer_idx = indices[0] if indices else None
            sum_decoder = self._build_sum(batch,
                                          lin,
                                          group_on_hetero,
                                          layer_idx=layer_idx)
            gen_decoder = []
            for stage in range(1, lout, 1):
                decoder = self._build_gen_stage(batch, lin, stage,
                                                group_on_hetero)
                gen_decoder.append(copy.deepcopy(decoder))
            self.sum_decoder_groups[group_name] = (sum_decoder, count, device,
                                                   indices)
            self.gen_decoder_groups[group_name] = (gen_decoder, count, device,
                                                   indices)
            self.routing_meta.append((group_name, device, count, indices))

        if self.routing_meta:
            first_group = self.routing_meta[0][0]
            # Back-compat alias only. Use *_decoder_groups for full graph.
            self.sum_decoder = self.sum_decoder_groups[first_group][0]
            self.gen_decoder = self.gen_decoder_groups[first_group][0]
