r"""
Hypnosis scene-finder benchmark.   Usage:   python bench_scenes.py  "C:\path\to\video.mp4"  [more videos...]
Races every way of finding scene cuts that is worth trying on THIS PC and prints a table + saves bench_results.txt.
Needs: opencv-python, numpy.   Optional: ffmpeg on PATH (or `pip install imageio-ffmpeg`) for the ffmpeg modes.
Send me bench_results.txt and I will wire the winner into the app.
"""
import os, sys, time, math, shutil, subprocess, threading
import numpy as np
import cv2

CUT, MIN_SCENE = 0.40, 1.2


def hist(f):
    f = f.reshape(-1, 3) >> 5
    idx = (f[:, 0].astype(np.int32) << 6) | (f[:, 1].astype(np.int32) << 3) | f[:, 2].astype(np.int32)
    h = np.bincount(idx, minlength=512).astype(np.float32)
    return h / max(1.0, float(h.sum()))


def dist(a, b):
    return float(math.sqrt(max(0.0, 1.0 - float(np.sum(np.sqrt(a * b))))))


def finish(cands):
    out, last = [0.0], 0.0
    for t in sorted(cands):
        if t - last >= MIN_SCENE:
            out.append(t)
            last = t
    return out


def probe(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return fps, n, w, h


def open_cap(path, hw):
    if hw and hasattr(cv2, "CAP_PROP_HW_ACCELERATION"):
        try:
            cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG, [cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY])
            if cap.isOpened() and cap.get(cv2.CAP_PROP_HW_ACCELERATION) > 0:
                return cap, True
            cap.release()
        except Exception:
            pass
    return cv2.VideoCapture(path), False


# ---------------------------------------------------------------- mode 1/2/3: OpenCV, N chunks
def cv_scan(path, fps, n, workers, hw):
    step = max(1, int(round(fps / 4.0)))
    bounds = [int(n * k / workers) // step * step for k in range(workers + 1)]
    bounds[-1] = n
    cands = [set() for _ in range(workers)]
    active = [False] * workers

    def work(wi):
        f0, f1 = bounds[wi], bounds[wi + 1]
        cap, active[wi] = open_cap(path, hw)
        start = max(0, f0 - step)
        if start > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        idx, prev = start, None
        while idx < f1 and cap.grab():
            if (idx - start) % step == 0:
                ok, fr = cap.retrieve()
                if ok:
                    h = hist(cv2.resize(fr, (48, 27), interpolation=cv2.INTER_AREA))
                    if prev is not None and idx >= f0 and dist(prev, h) > CUT:
                        cands[wi].add(idx / fps)
                    prev = h
            idx += 1
        cap.release()

    ts = [threading.Thread(target=work, args=(i,)) for i in range(workers)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    if hw and not all(active):
        raise RuntimeError("GPU decoding not available in this OpenCV build / driver")
    return finish(set().union(*cands))


# ---------------------------------------------------------------- mode 4/5/6: ffmpeg
def find_ffmpeg():
    p = shutil.which("ffmpeg")
    if p:
        return p
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def ff_scan(ffmpeg, path, mode, fps):
    pre, vf = [], "fps=4,scale=48:27"
    if mode == "gpu":
        pre = ["-hwaccel", "auto"]
    elif mode == "keys":
        pre = ["-skip_frame", "nokey"]
        vf = "scale=48:27"
    cmd = [ffmpeg, "-v", "error", "-nostdin", *pre, "-i", path, "-an", "-sn", "-vf", vf + ",showinfo" if mode == "keys" else vf,
           "-vsync", "0" if mode == "keys" else "cfr", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    if mode == "keys":                                           # need real timestamps: use ffprobe-free trick, showinfo on stderr
        cmd = [ffmpeg, "-v", "info", "-nostdin", "-skip_frame", "nokey", "-i", path, "-an", "-sn",
               "-vf", "scale=48:27,showinfo", "-vsync", "0", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE if mode == "keys" else subprocess.DEVNULL)
    times = []
    if mode == "keys":
        def rd():
            import re
            for line in proc.stderr:
                m = re.search(rb"pts_time:([0-9.]+)", line)
                if m:
                    times.append(float(m.group(1)))
        th = threading.Thread(target=rd, daemon=True)
        th.start()
    size = 48 * 27 * 3
    cands, prev, i = set(), None, 0
    while True:
        buf = proc.stdout.read(size)
        if len(buf) < size:
            break
        h = hist(np.frombuffer(buf, np.uint8).reshape(27, 48, 3))
        if mode == "keys":
            time.sleep(0)                                        # let the stderr reader catch up
            t = times[i] if i < len(times) else i * 2.0
        else:
            t = i / 4.0
        if prev is not None and dist(prev, h) > CUT:
            cands.add(t)
        prev = h
        i += 1
    proc.wait()
    if proc.returncode not in (0, None) and i == 0:
        raise RuntimeError("ffmpeg could not decode (GPU decoder missing?)")
    return finish(cands)


# ---------------------------------------------------------------- mode 7: PyAV two-stage (keyframes first, then refine)
def pyav_two_stage(path, workers):
    import av
    from concurrent.futures import ThreadPoolExecutor
    c = av.open(path)
    st = c.streams.video[0]
    st.thread_type = "AUTO"
    st.codec_context.skip_frame = "NONKEY"
    keys = []
    for fr in c.decode(st):                                      # stage 1: keyframes only (a handful of frames per minute)
        t = float(fr.time) if fr.time is not None else (keys[-1][0] + 2.0 if keys else 0.0)
        keys.append((t, hist(fr.reformat(width=48, height=27, format="rgb24").to_ndarray())))
    c.close()
    jobs = [(keys[i], keys[i + 1]) for i in range(len(keys) - 1) if dist(keys[i][1], keys[i + 1][1]) > 0.18]

    def refine(job):                                             # stage 2: decode only the stretches where something changed
        (t0, h0), (t1, h1) = job
        out = []
        cc = av.open(path)
        s2 = cc.streams.video[0]
        s2.thread_type = "AUTO"
        s2.codec_context.skip_frame = "NONREF"
        cc.seek(int(t0 / float(s2.time_base)), stream=s2, backward=True, any_frame=False)
        prev, last_t = h0, t0
        for fr in cc.decode(s2):
            t = float(fr.time) if fr.time is not None else last_t + 0.25
            if t <= t0 + 1e-3:
                continue
            if t >= t1 - 1e-3:
                break
            if t - last_t >= 0.24:
                h = hist(fr.reformat(width=48, height=27, format="rgb24").to_ndarray())
                if dist(prev, h) > CUT:
                    out.append(t)
                prev, last_t = h, t
        if dist(prev, h1) > CUT and not out:                     # the change sits right at the next keyframe
            out.append(t1)
        cc.close()
        return out

    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        res = list(ex.map(refine, jobs))
    return finish({t for r in res for t in r}), len(keys), len(jobs)


def agree(ref, got, tol=0.6):
    """How many reference scene starts have a found scene within tol seconds (and how many extras)."""
    hit = sum(1 for t in ref if any(abs(t - g) <= tol for g in got))
    extra = sum(1 for g in got if not any(abs(t - g) <= tol for t in ref))
    return hit, extra


def main():
    paths = [p for p in sys.argv[1:] if os.path.isfile(p)]
    if not paths:
        print(__doc__)
        return
    ncpu = os.cpu_count() or 1
    ff = find_ffmpeg()
    lines = [f"cores: {ncpu}   opencv: {cv2.__version__}   ffmpeg: {ff or 'not found'}"]
    for path in paths:
        fps, n, w, h = probe(path)
        lines.append(f"\n=== {os.path.basename(path)}   {w}x{h}  {fps:.1f}fps  {n} frames  ({n / fps / 60:.1f} min)")
        modes = [("OpenCV CPU, 1 thread (old app)", lambda: cv_scan(path, fps, n, 1, False)),
                 (f"OpenCV CPU, {ncpu} chunks (current app)", lambda: cv_scan(path, fps, n, ncpu, False)),
                 ("OpenCV GPU decode, 1 chunk", lambda: cv_scan(path, fps, n, 1, True)),
                 (f"OpenCV GPU decode, {min(ncpu, 4)} chunks", lambda: cv_scan(path, fps, n, min(ncpu, 4), True))]
        if ff:
            modes += [("ffmpeg CPU (all cores, 4 samples/s)", lambda: ff_scan(ff, path, "cpu", fps)),
                      ("ffmpeg GPU decode (-hwaccel auto)", lambda: ff_scan(ff, path, "gpu", fps)),
                      ("ffmpeg keyframes only (fastest, coarser)", lambda: ff_scan(ff, path, "keys", fps))]
        try:
            import av
            modes.append((f"PyAV two-stage (keyframes + refine, {ncpu} threads)", lambda: pyav_two_stage(path, ncpu)[0]))
        except ImportError:
            lines.append("  (pip install av  to test the PyAV two-stage mode)")
        ref, t_ref = None, None
        rows = []
        for name, fn in modes:
            try:
                t0 = time.perf_counter()
                sc = fn()
                dt = time.perf_counter() - t0
            except Exception as e:
                rows.append((name, None, None, f"unavailable ({e})"))
                continue
            if ref is None:
                ref, t_ref = sc, dt
            hit, extra = agree(ref, sc)
            rows.append((name, dt, len(sc), f"{hit}/{len(ref)} of baseline scenes matched, {extra} extra   speed x{t_ref / dt:.1f}"))
        for name, dt, ns, note in rows:
            lines.append(f"  {name:<44} " + (f"{dt:7.1f}s  {ns:4d} scenes  {note}" if dt is not None else note))
    lines.append("\nNote: a 'GPU' row can silently fall back to the CPU. To be sure the GPU really decoded, watch Task Manager > Performance > GPU"
                 " > 'Video Decode' while that row runs.")
    out = "\n".join(lines)
    print(out)
    with open("bench_results.txt", "w", encoding="utf-8") as f:
        f.write(out + "\n")
    print("\nSaved bench_results.txt")


if __name__ == "__main__":
    main()
