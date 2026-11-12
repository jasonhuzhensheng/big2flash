"""Wait until the Mac is quiet enough to benchmark: background GPU users (Photos analysis, Spotlight)
otherwise cut measured speeds several-fold. Import and call wait_quiet(), or run as a script."""
import re, subprocess, time

NOISY = ("mediaanalysisd", "corespotlightd", "mds_stores", "photoanalysisd", "mdworker", "kavd", "knowledgeconstructiond")


def gpu_busy():
    out = subprocess.run(["ioreg", "-r", "-d", "1", "-w", "0", "-c", "IOAccelerator"],
                         capture_output=True, text=True).stdout
    m = re.search(r'"Device Utilization %"=(\d+)', out)
    return int(m.group(1)) if m else 0


def noisy():
    out = subprocess.run(["ps", "-Ao", "pcpu,comm"], capture_output=True, text=True).stdout
    return [(float(l.split()[0]), l.split("/")[-1]) for l in out.splitlines()[1:]
            if any(n in l for n in NOISY) and float(l.split()[0]) > 15]


def wait_quiet(timeout=1800, samples=6):
    """Quiet = no noisy background process and GPU use averaging under 25% over ~3 s (single samples
    spike: something on this Mac touches the GPU every few seconds)."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        if not noisy():
            use = []
            for _ in range(samples):
                use.append(gpu_busy())
                time.sleep(0.5)
            if sum(use) / samples < 25:
                return True
        else:
            time.sleep(2)
    print(f"warning: still noisy after {timeout}s: {noisy()} gpu {gpu_busy()}%", flush=True)
    return False


if __name__ == "__main__":
    print("quiet" if wait_quiet() else "noisy")
