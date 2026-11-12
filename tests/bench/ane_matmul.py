"""How fast is the Neural Engine at this model's matmuls? A chain of 27B-sized MLP projections
(5120 -> 17408 -> 5120), weights 4-bit (palettized, per-grouped-channel) and fp16, run with
compute units CPU_AND_NE at T = 1, 2, 4, 8 input rows. Reports ms and effective weight GB/s, and
where Core ML actually placed the ops."""
import os, sys, time, tempfile
import numpy as np
import coremltools as ct
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types

H, F, L = 5120, 17408, int(os.environ.get("PAIRS", 4))
rng = np.random.default_rng(0)
W = [((rng.standard_normal((F, H)) * 0.02).astype(np.float16), (rng.standard_normal((H, F)) * 0.02).astype(np.float16))
     for _ in range(L)]


def build(T):
    @mb.program(input_specs=[mb.TensorSpec(shape=(T, H), dtype=types.fp16)],
                opset_version=ct.target.macOS15)
    def prog(x):
        h = x
        for u, d in W:
            h = mb.linear(x=h, weight=u)
            h = mb.mul(x=h, y=np.float16(0.01))
            h = mb.linear(x=h, weight=d)
            h = mb.mul(x=h, y=np.float16(0.01))
        return h
    return ct.convert(prog, convert_to="mlprogram", minimum_deployment_target=ct.target.macOS15,
                      compute_precision=ct.precision.FLOAT16)


def placement(m):
    try:
        from coremltools.models.compute_plan import MLComputePlan
        path = m.get_compiled_model_path()
        plan = MLComputePlan.load_from_path(path=path, compute_units=ct.ComputeUnit.CPU_AND_NE)
        fn = plan.model_structure.program.functions["main"]
        dev = {}
        for op in fn.block.operations:
            if op.operator_name in ("ios18.linear", "ios17.linear", "linear"):
                u = plan.get_compute_device_usage_for_mlprogram_operation(op)
                name = type(u.preferred_compute_device).__name__ if u else "?"
                dev[name] = dev.get(name, 0) + 1
        return dev
    except Exception as e:
        return f"n/a ({e.__class__.__name__})"


for bits in tuple(int(b) for b in os.environ.get("BITS", "4,16").split(",")):
    for T in tuple(int(t) for t in os.environ.get("TS", "1,2,4,8").split(",")):
        m = build(T)
        if bits == 4:
            cfg = ct.optimize.coreml.OptimizationConfig(global_config=ct.optimize.coreml.OpPalettizerConfig(
                nbits=4, mode="uniform", granularity="per_grouped_channel", group_size=16))
            m = ct.optimize.coreml.palettize_weights(m, cfg)
        d = tempfile.mkdtemp()
        p = os.path.join(d, "m.mlpackage"); m.save(p)
        m = ct.models.MLModel(p, compute_units=ct.ComputeUnit.CPU_AND_NE)
        x = {"x": rng.standard_normal((T, H)).astype(np.float16)}
        for _ in range(3):
            m.predict(x)
        best = 1e9
        for _ in range(8):
            t = time.perf_counter(); m.predict(x); best = min(best, time.perf_counter() - t)
        nbytes = 2 * L * F * H * bits / 8
        print(f"{bits:2d}-bit T={T}  {best*1000:7.2f} ms  {nbytes/best/1e9:6.1f} GB/s  {2*2*L*F*H*T/best/1e12:5.2f} TFLOPS   linear ops on: {placement(m)}", flush=True)
