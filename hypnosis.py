"""
HYPNOSIS - a music-reactive spiral visualizer for Windows (also runs on Linux/macOS).

  * Drag & drop audio files (or folders) onto the window to start.
  * SPACE  play / pause
  * The spiral spins with the rhythm; bass adds a subtle zoom + chromatic aberration.

Keys
  Esc          glass menu (play queue, settings, Domination Mode)
  Space        play / pause
  F / F11      fullscreen          D     toggle Domination Mode
  M / Tab      cycle visual mode   (Classic, Prism, Neon, Kaleidoscope)
  Left/Right   seek -/+ 5 s        Up/Down or wheel   volume
  N / P        next / previous     H     show / hide help
  S            Spotify Mode (hears only Spotify.exe, via WASAPI process loopback)
  V            jump to the next scene of the media layer
  X / Delete   hide / delete the scene on screen (Z undoes)
  Esc > Scenes click a scene to preview it, drag the trim handles to shorten / lengthen it
  Esc > Visuals: collapsible sections (style, live preview, motion, media); wheel scrolls (Ctrl+wheel nudges a slider)
  Drop a video / GIF / image to blend it into the visuals (Esc > Visuals > Media & stack)
  Q            quit
"""
import os
import sys
import math
import time
import ctypes
import shutil
import json
import random
import threading
import subprocess

import numpy as np
import pygame
import moderngl
import sounddevice as sd

try:
    import miniaudio
except Exception:  # ffmpeg fallback still works without it
    miniaudio = None

SR = 44100
AUDIO_EXT = {".mp3", ".wav", ".flac", ".ogg", ".oga", ".m4a", ".aac", ".opus", ".wma", ".mka"}


# --------------------------------------------------------------------------- audio
def decode_audio(path):
    """Return float32 array (n, 2) at SR Hz."""
    err = None
    if miniaudio is not None:
        try:
            d = miniaudio.decode_file(
                path, output_format=miniaudio.SampleFormat.FLOAT32, nchannels=2, sample_rate=SR
            )
            return np.frombuffer(d.samples, dtype=np.float32).reshape(-1, 2).copy()
        except Exception as e:  # unsupported container/codec -> try ffmpeg
            err = e
    ff = shutil.which("ffmpeg")
    if ff:
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        p = subprocess.run(
            [ff, "-v", "quiet", "-i", path, "-vn", "-f", "f32le", "-ac", "2", "-ar", str(SR), "-"],
            capture_output=True, creationflags=flags,
        )
        if p.returncode == 0 and p.stdout:
            return np.frombuffer(p.stdout, dtype=np.float32).reshape(-1, 2).copy()
    raise RuntimeError(f"Can't decode this file ({err or 'install ffmpeg for m4a/opus/wma'})")


class AudioEngine:
    def __init__(self):
        self.data = None
        self.pos = 0
        self.playing = False
        self.finished = False
        self.volume = 0.85
        self.gain = 0.0            # de-click ramp
        self.loading = False
        self.error = None
        self._pending = None
        self._token = 0
        self.muffle = 0.0          # 0..1, set from the menu animation
        self.stream = None
        self.latency = 0.05
        self._open_stream()

    def _open_stream(self):
        try:
            self.stream = sd.OutputStream(
                samplerate=SR, channels=2, dtype="float32", latency=0.06, callback=self._callback
            )
            self.stream.start()
            self.latency = float(self.stream.latency)
        except Exception as e:
            self.error = f"Audio device error: {e}"

    def _callback(self, out, frames, tinfo, status):
        data = self.data
        if data is None:
            out.fill(0)
            self.gain = 0.0
            return
        target = 1.0 if self.playing else 0.0
        g0 = self.gain
        if target == 0.0 and g0 == 0.0:
            out.fill(0)
            return
        step = frames / (SR * 0.025)
        g1 = g0 + max(-step, min(step, target - g0))
        ramp = np.linspace(g0, g1, frames, endpoint=False, dtype=np.float32)[:, None]
        self.gain = g1
        pos = self.pos
        end = min(pos + frames, len(data))
        n = end - pos
        m = self.muffle
        if m > 0.01 and n > 0:
            # causal windowed-sinc low-pass, swept 20 kHz -> ~650 Hz; history comes straight from the track
            fc = min(20000.0 * (650.0 / 20000.0) ** m, 0.49 * SR) / SR
            T = 257
            idx = np.arange(T) - (T - 1) / 2.0
            k = (2 * fc * np.sinc(2 * fc * idx) * np.blackman(T)).astype(np.float32)
            k /= k.sum()
            lo = pos - (T - 1)
            seg = data[lo:end] if lo >= 0 else np.concatenate((np.zeros((-lo, 2), np.float32), data[:end]))
            block = np.stack((np.convolve(seg[:, 0], k, "valid"), np.convolve(seg[:, 1], k, "valid")), axis=1)
            block *= 1.0 - 0.22 * m
        else:
            block = data[pos:end]
        out[:n] = block * ramp[:n] * self.volume
        if n < frames:
            out[n:] = 0
            self.playing = False
            self.finished = True
            self.gain = 0.0
        self.pos = end

    # -- loading
    def load(self, path):
        self._token += 1
        token = self._token
        self.loading = True
        self.error = None

        def work():
            try:
                arr = decode_audio(path)
                if token == self._token:
                    self._pending = (path, arr)
            except Exception as e:
                if token == self._token:
                    self.error = str(e)
            finally:
                if token == self._token:
                    self.loading = False

        threading.Thread(target=work, daemon=True).start()

    def poll(self):
        """Call from main thread; returns path when a new track just started."""
        if self._pending:
            path, arr = self._pending
            self._pending = None
            self.playing = False
            self.pos = 0
            self.gain = 0.0
            self.finished = False
            self.data = arr
            self.playing = True
            return path
        return None

    def stop(self):
        """Unload the current track (silences output, drops pending loads)."""
        self._token += 1
        self._pending = None
        self.loading = False
        self.playing = False
        self.finished = False
        self.gain = 0.0
        self.pos = 0
        self.data = None

    # -- transport
    def toggle(self):
        if self.data is None:
            return
        if self.finished or self.pos >= len(self.data):
            self.pos = 0
            self.finished = False
        self.playing = not self.playing

    def seek(self, seconds):
        if self.data is not None:
            self.pos = int(max(0, min(len(self.data) - 1, self.pos + seconds * SR)))
            self.finished = False

    def set_volume(self, v):
        self.volume = float(max(0.0, min(1.0, v)))

    def window(self, n):
        """Mono samples that are audible *right now* (compensating output latency)."""
        if self.data is None:
            return None
        end = int(self.pos - self.latency * SR)
        if end < n:
            return None
        seg = self.data[end - n:end]
        return seg.mean(axis=1)


# --------------------------------------------------------------------------- analysis
class Analyzer:
    """Turns audio windows into smooth, punchy control signals."""

    def __init__(self, n=2048, sr=SR):
        self.n = n
        self.sr = sr
        self.win = np.hanning(n).astype(np.float32)
        f = np.fft.rfftfreq(n, 1.0 / sr)
        self.ib = np.where((f >= 30) & (f < 150))[0]
        self.im = np.where((f >= 150) & (f < 2500))[0]
        self.ih = np.where((f >= 2500) & (f < 14000))[0]
        self.iflux = np.where((f >= 40) & (f < 9000))[0]
        self.isn_c = np.where((f >= 1500) & (f < 8000))[0]     # snare crack (hats have no body, so they don't count)
        self.sn_prev = None
        self.sn_pk = np.full(3, 1e-4)
        self.last_kick = 0.0
        self.last_snare = 0.0
        self.kpk, self.avg_k = 2e-4, 0.0
        self.k_sens = self.s_sens = 1.0
        self.rng = None
        self.want_vis = False
        self.vis_edges = None
        self.vis_env = np.zeros(48)
        self.vpk = 1e-4
        self.vis_bars = (0,) * 48
        self.set_ranges(30, 150, 170, 420, 1.0, 1.0)
        self.kick = 0.0      # >0 on the frame a kick is detected (strength)
        self.snare = 0.0     # >0 on the frame a snare / clap is detected (strength)
        self.prev = None
        self.peaks = np.full(3, 1e-4)
        self.avg_bass = 0.0
        self.flux_peak = 1e-4
        self.flux_avg = 0.0
        self.last_beat = 0.0
        self.bass = 0.0      # 0..1 punchy kick envelope
        self.mid = 0.0
        self.high = 0.0
        self.energy = 0.0
        self.onset = 0.0
        self.beat = 0.0      # >0 on the frame a beat is detected (strength)

    def set_ranges(self, kl, kh, sl, sh, ks, ss):
        """Kick / snare detection bands (Hz) and sensitivities - user tunable."""
        key = (round(kl), round(kh), round(sl), round(sh), round(ks, 3), round(ss, 3))
        if key == self.rng:
            return
        self.rng = key
        f = np.fft.rfftfreq(self.n, 1.0 / self.sr)
        kh, sh = max(kh, kl + 10), max(sh, sl + 20)
        self.ik = np.where((f >= kl) & (f < kh))[0]
        self.isn_b = np.where((f >= sl) & (f < sh))[0]
        if len(self.ik) == 0:
            self.ik = np.array([max(1, int(np.argmin(np.abs(f - kl))))])
        if len(self.isn_b) == 0:
            self.isn_b = np.array([max(1, int(np.argmin(np.abs(f - sl))))])
        self.k_sens, self.s_sens = max(0.2, ks), max(0.2, ss)
        self.vis_edges = np.searchsorted(f, 20.0 * (600.0 ** (np.arange(49) / 48.0)))

    @staticmethod
    def _env(cur, target, dt, release):
        if target > cur:
            return cur + (target - cur) * (1 - math.exp(-dt * 70))
        return cur * math.exp(-dt * release)

    def step(self, chunk, dt, now):
        dt = max(1e-4, min(dt, 0.1))
        if chunk is None:
            spec = np.zeros(self.n // 2 + 1, dtype=np.float32)
        else:
            spec = np.abs(np.fft.rfft(chunk * self.win)).astype(np.float32) / self.n

        raw = np.array([
            math.sqrt(float(np.mean(spec[self.ib] ** 2))),
            math.sqrt(float(np.mean(spec[self.im] ** 2))),
            math.sqrt(float(np.mean(spec[self.ih] ** 2))),
        ])
        # auto-gain: slowly-decaying peak per band
        self.peaks = np.maximum(raw, np.maximum(self.peaks * math.exp(-dt * 0.30), 2e-4))
        nb, nm, nh = raw / self.peaks

        # bass: transient vs. sustained bed
        self.avg_bass += (nb - self.avg_bass) * (1 - math.exp(-dt / 0.5))
        trans = max(0.0, nb - self.avg_bass * 0.9)
        bass_t = min(1.0, trans * 2.2 + 0.45 * nb ** 3)
        self.bass = self._env(self.bass, bass_t, dt, 7.5)
        self.mid = self._env(self.mid, nm ** 1.5, dt, 5.0)
        self.high = self._env(self.high, nh ** 1.5, dt, 8.0)
        self.energy = self._env(self.energy, min(1.0, (nb + nm + nh) / 2.2), dt, 3.0)

        # spectral flux -> onsets -> beats
        mag = np.sqrt(spec[self.iflux])
        flux = 0.0
        if self.prev is not None:
            flux = float(np.sum(np.maximum(0.0, mag - self.prev)))
        self.prev = mag
        self.flux_peak = max(flux, self.flux_peak * math.exp(-dt * 0.4), 1e-4)
        onset = flux / self.flux_peak
        self.onset = self._env(self.onset, onset, dt, 6.0)
        self.beat = 0.0
        self.flux_avg += (flux - self.flux_avg) * (1 - math.exp(-dt / 0.6))
        is_kick = trans > 0.28 and nb > 0.45
        is_onset = onset > 0.55 and flux > self.flux_avg * 1.4
        self.kick = 0.0
        rk = math.sqrt(float(np.mean(spec[self.ik] ** 2)))
        self.kpk = max(rk, self.kpk * math.exp(-dt * 0.30), 2e-4)
        nk = rk / self.kpk
        self.avg_k += (nk - self.avg_k) * (1 - math.exp(-dt / 0.5))
        tk = max(0.0, nk - self.avg_k * 0.9)
        if tk * self.k_sens > 0.28 and nk * min(1.0, self.k_sens) > 0.45 and now - self.last_kick > 0.14:
            self.last_kick = now
            self.kick = min(1.0, 0.55 + 0.9 * tk)
        self.snare = 0.0
        sm = np.sqrt(spec)
        cur = np.array([np.mean(sm[self.isn_b]), np.mean(sm[self.isn_c]), np.mean(sm[self.ik])])
        if self.sn_prev is not None:
            fl = np.maximum(0.0, cur - self.sn_prev)
            self.sn_pk = np.maximum(fl, np.maximum(self.sn_pk * math.exp(-dt * 0.4), 1e-4))
            nf = fl / self.sn_pk
            score = float(min(nf[0], nf[1]))
            if score * self.s_sens > 0.38 and nf[2] < 0.3 and now - self.last_snare > 0.14 and now - self.last_kick > 0.05:
                self.last_snare = now
                self.snare = min(1.0, 0.5 + 0.6 * score)
        self.sn_prev = cur
        if self.want_vis and self.vis_edges is not None:
            e = self.vis_edges
            v = np.array([math.sqrt(float(np.mean(spec[e[i]:max(e[i + 1], e[i] + 1)] ** 2))) for i in range(48)])
            self.vpk = max(float(v.max()), self.vpk * math.exp(-dt * 0.3), 1e-4)
            tgt = (v / self.vpk) ** 0.6
            pad = np.concatenate(([tgt[0]], tgt, [tgt[-1]]))
            tgt = 0.25 * pad[:-2] + 0.5 * pad[1:-1] + 0.25 * pad[2:]            # blend neighbouring bars: no single-bar spikes
            k = np.where(tgt > self.vis_env, 1 - math.exp(-dt * 26.0), 1 - math.exp(-dt * 5.0))   # quick rise, gentle fall (frame-rate independent)
            self.vis_env = self.vis_env + (tgt - self.vis_env) * k
            self.vis_bars = tuple(round(float(x) * 24.0, 2) for x in self.vis_env)
        if (is_kick or is_onset) and now - self.last_beat > 0.16:
            self.last_beat = now
            self.beat = min(1.0, 0.5 + 0.9 * max(trans, onset * 0.7))


# --------------------------------------------------------------------------- shaders
VERT = """
#version 330
in vec2 in_pos;
out vec2 vUV;
void main(){ vUV = in_pos * 0.5 + 0.5; gl_Position = vec4(in_pos, 0.0, 1.0); }
"""

FRAG = """
#version 330
uniform vec2  uRes;
uniform float uRot, uFlow, uFlowK, uZoom, uAb, uBass, uMid, uHigh, uEnergy, uHue, uTime, uWarp, uDim;
uniform int   uMode;
uniform sampler2D uMediaA, uMediaB;
uniform float uMOn, uMFade, uMAmt, uMAsp, uMAspB, uMPulse, uMBlur, uMZ, uMGain;
uniform sampler2D uPrevA, uFlowA, uPrevB, uFlowB;   // previous frame + motion field of each playback head
uniform vec3 uMI;       // x = frame interpolation on, y / z = how far we are between the two frames (head A / head B)
uniform float uPsyT;   // psychedelic flow clock: only ever moves forward, speeds up on the beat
uniform vec2 uHueWave;   // x = wave strength, y = wave phase
uniform vec4 uMSway;   // xy = uv shift, z = roll (rad), w = extra zoom
uniform int   uMBlend;
out vec4 fragColor;

const float TAU   = 6.28318530718;
const float ARMS  = 2.0;    // must be an integer -> seamless
const float PITCH = 2.7;    // radial band density (log-spiral)

vec3 pal(float t){ return 0.5 + 0.5 * cos(TAU * (t + vec3(0.0, 0.33, 0.67))); }
float hash(vec2 p){ return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453); }

// ---------------------------------------------------------------- KALEIDOSCOPE
// Conformal log-polar triangle lattice, mirror-folded into wedges. Every motion is a continuous
// function of time / audio, and all time frequencies are multiples of 0.005 so the shader time
// wraps without a jump.
const float PI = 3.14159265359;
const float KSCALE = 5.0;        // triangles per radian (density)
const float KFOLD  = 3.0;        // mirror wedges (x2 sectors)

float par(float f, float e){ return 0.5 - 0.5 * clamp(sin(PI * f) / e, -1.0, 1.0); }
float xor2(float a, float b){ return a + b - 2.0 * a * b; }

float lattice(vec2 q, float e){                       // smooth alternating up/down triangles, 0..1
    float f1 = q.y;
    float f2 = 0.8660254 * q.x + 0.5 * q.y;
    float f3 = 0.8660254 * q.x - 0.5 * q.y;
    return xor2(xor2(par(f1, e), par(f2, e)), par(f3, e));
}

vec3 grad(float t){                                   // yellow -> orange -> pink -> purple -> blue
    const vec3 c0 = vec3(1.00, 0.88, 0.04);
    const vec3 c1 = vec3(1.00, 0.50, 0.27);
    const vec3 c2 = vec3(1.00, 0.14, 0.62);
    const vec3 c3 = vec3(0.60, 0.08, 0.93);
    const vec3 c4 = vec3(0.18, 0.12, 1.00);
    t = clamp(t, 0.0, 1.0) * 4.0;
    vec3 c = mix(c0, c1, smoothstep(0.0, 1.0, t));
    c = mix(c, c2, smoothstep(1.0, 2.0, t));
    c = mix(c, c3, smoothstep(2.0, 3.0, t));
    return mix(c, c4, smoothstep(3.0, 4.0, t));
}
float pingpong(float x){ return 1.0 - abs(fract(x * 0.5) * 2.0 - 1.0); }   // 0..1..0, period 2

vec3 kaleido(vec2 p, float px)
{
    float r  = length(p) + 1e-5;
    float th = atan(p.y, p.x);
    float wedge = PI / KFOLD;
    float a1 = abs(mod(th + 0.3333333 * uRot, 2.0 * wedge) - wedge);  // mirror fold (continuous); 1/3 spin = exactly 1 wedge-period per turn
    float a2 = abs(mod(th - 0.6666667 * uRot + 0.35, 2.0 * wedge) - wedge); // counter-rotating inner layer (2/3 = 2 wedge-periods per turn)

    // slow, smooth "breathing" ripple travelling outward
    float lr = log(r) + 0.05 * sin(uTime * 0.35 + r * 3.2);

    // swirl: the lattice slowly twists into / out of spirals (mid-range drives it)
    float phi = 0.16 * sin(uTime * 0.21) + 0.10 * sin(uTime * 0.13 + 1.7) + 0.12 * uMid;
    float c = cos(phi), s = sin(phi);
    const float U0 = -3.5;

    float fw = KSCALE / r * px * (1.35 + 5.0 * uWarp);
    float e  = PI * fw + 0.012;

    // ---- layer 1: main triangles
    vec2 q = vec2(lr * KSCALE - U0, a1 * KSCALE);
    q.y += uWarp * KSCALE * 0.6 * sin(q.x * 0.85 - uTime * 1.1);       // audio-driven ripple
    q = mat2(c, -s, s, c) * q;
    q.x += U0 - uFlowK;                                                // zoom IN: triangles grow outward, endlessly
    float m1 = lattice(q, e);

    // ---- layer 2: bigger, dimmer lattice seen through the dark triangles
    const float S2 = 3.1;
    float fw2 = S2 / r * px * (1.35 + 5.0 * uWarp);
    vec2 q2 = vec2(lr * S2 - U0, a2 * S2);
    float c2 = cos(-0.8 * phi + 0.6), s2 = sin(-0.8 * phi + 0.6);
    q2 = mat2(c2, -s2, s2, c2) * q2;
    q2.x += U0 - uFlowK;                                               // same direction (inner layer is bigger -> parallax)
    float m2 = lattice(q2, PI * fw2 + 0.012);

    // ---- colour: radial gradient that drifts with the music
    float g  = pow(r / 1.9, 0.85);
    float gp = g * 0.95 + uHue * 0.5 + 0.07 * uHigh;
    vec3 tri = grad(pingpong(gp));
    vec3 under = grad(pingpong(gp + 0.55)) * 0.34 * m2;
    vec3 dark = vec3(0.010) + under + tri * 0.035 * uEnergy;
    vec3 col = mix(dark, tri, m1);
    col *= 0.40 + 0.60 * smoothstep(0.02, 0.85, r);                    // tunnel depth: the far end sinks into shadow, the walls rush past bright
    col += exp(-r * 9.0) * vec3(1.0, 0.84, 0.20) * 0.30;               // light at the end of the tunnel
    return col;
}

// ---------------------------------------------------------------- PSYCHEDELIC
// Liquid marble: a domain-warped flow field. Every time frequency is a multiple of 0.005 rad/s (seamless wrap),
// the rotation is the full-turn uRot, and the music pushes the warp (mid), the ripple (bass) and the colour (hue).
float psyField(vec2 p)
{
    vec2 q = p * 2.0;
    float t = uPsyT;
    float A = 0.95;                                                   // fixed fold strength -> the shape never bounces back
    for (int i = 0; i < 5; i++) {
        float fi = float(i);
        vec2 w = vec2(sin(1.55 * q.y + t * (0.15 + 0.075 * fi) + fi * 1.7),
                      sin(1.35 * q.x - t * (0.20 + 0.050 * fi) + fi * 2.3));
        q += A * w / (1.0 + 0.35 * fi);
        q = mat2(0.8, -0.6, 0.6, 0.8) * q;                          // fixed twist between folds breaks the symmetry
    }
    float v = 0.55 * q.x + 0.45 * sin(1.2 * q.y + 0.1 * t) + 0.35 * length(q);
    return v;
}

vec3 psyche(vec2 p, float px)
{
    const float E = 0.004;
    float v  = psyField(p);
    float vx = psyField(p + vec2(E, 0.0));
    float vy = psyField(p + vec2(0.0, E));
    vec2 g = vec2(vx - v, vy - v) / E;                               // field gradient -> lighting + anti-aliasing width
    float gl = length(g);

    vec3 col = pal(uHue + 0.30 * v + 0.10 * sin(1.7 * v));   // saturated rainbow flowing along the field
    float shade = 0.5 + 0.5 * dot(g, vec2(0.6, 0.8)) / (1.0 + gl);
    col *= 0.72 + 0.5 * shade;                                       // glossy highlights on one side of every fold

    float BF = 3.8;                                    // contour density
    float band = v * BF;
    float d = abs(fract(band + 0.5) - 0.5);                          // distance to the nearest contour (band units)
    float aa = clamp(gl * BF * px * 1.5, 0.002, 0.45);
    float line = 1.0 - smoothstep(0.03, 0.03 + aa + 0.02, d);
    float par = mod(floor(band + 0.5), 2.0);
    vec3 lc = mix(vec3(0.02, 0.01, 0.16), vec3(1.0, 0.98, 0.9), par);       // alternating navy / white filaments
    col = mix(col, lc, line * 0.9);
    float d2 = abs(fract(band * 2.7 + 0.5) - 0.5);                    // second, finer family of hair-thin filaments
    float aa2 = clamp(gl * BF * 2.7 * px * 1.5, 0.002, 0.45);
    col = mix(col, pal(uHue + 0.5 + 0.3 * v) * 1.1, (1.0 - smoothstep(0.02, 0.02 + aa2 + 0.015, d2)) * 0.55);
    float soft = 1.0 - smoothstep(0.0, 0.5, d);                      // faint halo beside each contour
    col += 0.08 * soft * pal(uHue + 0.5 + 0.2 * v);
    col = mix(col, vec3(dot(col, vec3(0.333))), smoothstep(0.35, 1.2, gl * BF * px * 3.0));  // far-too-dense areas relax to grey (no moire)
    return col;
}

vec3 pattern(vec2 p, float px)
{
    if (uMode == 4) return psyche(p, px);
    if (uMode == 3) return kaleido(p, px);
    float r  = length(p) + 1e-5;
    float a  = atan(p.y, p.x) + uRot;
    float lr = log(r);

    // organic swirl wobble (integer multiples of the angle -> no seam)
    float wob = uWarp * sin(3.0 * a + 4.0 * lr - uTime * 1.3)
              + uWarp * 0.5 * sin(5.0 * a - 7.0 * lr + uTime * 0.9);
    float u = ARMS * a / TAU + PITCH * lr + uFlow + wob;

    // analytic anti-aliasing width (no fwidth => no seam artefacts)
    float fw = sqrt((ARMS / TAU) * (ARMS / TAU) + PITCH * PITCH) / r * px * (1.0 + 3.0 * uWarp);

    float f = fract(u);
    float t = abs(f - 0.5) * 2.0;                 // triangle wave 0..1
    float duty = 0.5 + 0.05 * sin(uTime * 0.7);
    float aa = clamp(fw * 2.0, 0.0005, 0.6);
    float m = smoothstep(duty - aa, duty + aa, t);
    m = mix(m, 0.5, smoothstep(0.18, 0.5, fw));    // dissolve into the vanishing point

    vec3 col;
    if (uMode == 0) {                              // CLASSIC black & white
        col = mix(vec3(0.012), vec3(0.965), m);
    } else if (uMode == 1) {                       // PRISM two-tone colour bands
        float idx = floor(u);
        vec3 c = pal(uHue + idx * 0.5 + lr * 0.11);
        col = mix(c * 0.035, c, m);
        col = mix(col, vec3(0.5), smoothstep(0.18, 0.5, fw) * 0.6);
    } else {                                       // NEON iridescent tunnel
        vec3 c1 = pal(uHue + lr * 0.16 + t * 0.35);
        vec3 c2 = pal(uHue + 0.5 + lr * 0.16 - t * 0.35 + 0.1 * sin(u * 3.14159265));
        col = mix(c1, c2, smoothstep(0.0, 1.0, t));
        float ridge = exp(-pow((t - duty) / (0.07 + aa), 2.0));
        col = col * (0.55 + 0.45 * m) + ridge * vec3(0.55, 0.75, 0.85);
        col = mix(col, vec3(0.6), smoothstep(0.18, 0.5, fw) * 0.5);
    }
    return col;
}

vec2 aberrate(vec2 p, float k)
{
    float r = length(p);
    float s = 1.0 + uAb * k * (0.25 + r);
    float ang = uAb * k * 1.6 * r;
    float c = cos(ang), sn = sin(ang);
    return mat2(c, -sn, sn, c) * p * s;
}

vec2 fitUV(vec2 p, float asp)
{
    float sa = uRes.x / uRes.y;
    vec2 s = vec2(p.x / (2.0 * sa), p.y * 0.5) + 0.5;
    if (sa > asp) s.y = 0.5 + (s.y - 0.5) * (asp / sa);              // cover-fit
    else          s.x = 0.5 + (s.x - 0.5) * (sa / asp);
    return s;
}

vec2 mirrorFlip(vec2 u)
{
    u = 1.0 - abs(1.0 - mod(u, 2.0));                                // mirror at the edges (zoom / aberration overshoot)
    u.y = 1.0 - u.y;
    return u;
}

vec2 swayUV(vec2 u)
{
    if (uMSway.w == 0.0 && uMSway.z == 0.0 && uMSway.x == 0.0 && uMSway.y == 0.0) return u;
    float a = uRes.x / uRes.y;
    vec2 c = (u - 0.5) * vec2(a, 1.0);
    float cs = cos(uMSway.z), sn = sin(uMSway.z);
    c = mat2(cs, -sn, sn, cs) * c;                                   // handheld roll
    c /= (1.0 + uMSway.w);                                           // push-in on the hit
    return c / vec2(a, 1.0) + 0.5 + uMSway.xy;
}

vec3 texI(sampler2D cur, sampler2D prv, sampler2D flw, vec2 u, float al)
{
    vec2 t = mirrorFlip(u);
    if (uMI.x < 0.5 || al >= 0.995) return texture(cur, t).rgb;
    vec2 f = texture(flw, t).rg;                                     // motion prev -> cur, in texture units
    return mix(texture(prv, t - al * f).rgb, texture(cur, t + (1.0 - al) * f).rgb, al);   // warp both frames to the in-between moment
}

vec3 sampleMedia(vec2 p)
{
    vec2 ua = swayUV(fitUV(p, uMAsp)), ub = swayUV(fitUV(p, uMAspB));
    if (uMZ >= 0.0) {                                                // zoom-through: scene A rushes in, scene B settles from a push-in
        float sA = 1.0 + 1.7 * uMZ, sB = 1.0 + 0.9 * (1.0 - uMZ);
        return mix(texI(uMediaA, uPrevA, uFlowA, 0.5 + (ua - 0.5) / sA, uMI.y), texI(uMediaB, uPrevB, uFlowB, 0.5 + (ub - 0.5) / sB, uMI.z), smoothstep(0.3, 0.75, uMZ));
    }
    if (uMBlur > 0.0005) {                                           // blur-through transition: golden-angle disc of taps
        vec3 a = vec3(0.0), b = vec3(0.0);
        float jit = hash(gl_FragCoord.xy) * 6.2831853;
        for (int i = 0; i < 14; i++) {
            float fi = float(i) + 0.5;
            float r = sqrt(fi / 14.0) * uMBlur;
            float an = fi * 2.399963 + jit;
            vec2 o = vec2(cos(an), sin(an) * uRes.x / uRes.y) * r;
            a += texture(uMediaA, mirrorFlip(ua + o)).rgb;
            b += texture(uMediaB, mirrorFlip(ub + o)).rgb;
        }
        return mix(a, b, uMFade) / 14.0;
    }
    vec3 ca = texI(uMediaA, uPrevA, uFlowA, ua, uMI.y);
    if (uMFade <= 0.001) return ca;
    return mix(ca, texI(uMediaB, uPrevB, uFlowB, ub, uMI.z), uMFade);
}

vec3 blendMedia(vec3 pat, vec3 m)
{
    float lum = dot(pat, vec3(0.299, 0.587, 0.114));
    float mx  = max(pat.r, max(pat.g, pat.b));
    vec3 theme = pat / (mx + 0.02);
    if (uMBlend == 3)                                                // Video style: no spiral, keeps zoom / aberration / flash
        return m * (0.55 + 0.45 * uMAmt) * (1.0 + 0.45 * uMPulse);
    vec3 mt = m * mix(vec3(1.0), theme, 0.5);                        // tint the footage with the theme colours
    mt *= 1.0 + 0.45 * uMPulse;                                      // flash on the bass
    vec3 res;
    if (uMBlend == 0) {                                              // spiral window: footage shows through the bright bands
        float mask = smoothstep(0.05, 0.45, lum);
        res = mix(pat, mt, mask * uMAmt);
    } else if (uMBlend == 1) {                                       // soft overlay
        vec3 sl = (1.0 - 2.0 * pat) * mt * mt + 2.0 * pat * mt;
        res = mix(pat, mix(sl, mt, 0.35), uMAmt);
    } else {                                                         // glow (screen)
        res = 1.0 - (1.0 - pat) * (1.0 - mt * uMAmt);
    }
    return res;
}

vec3 hueWave(vec3 c, vec2 p)
{
    if (uHueWave.x <= 0.001) return c;
    // smooth travelling waves in screen space (no polar maths -> no seam / pinch at the centre)
    float w1 = sin(dot(p, vec2(0.80, 0.60)) * 1.9 - uHueWave.y);
    float w2 = sin(dot(p, vec2(-0.55, 0.83)) * 1.3 + uHueWave.y + 1.7 * w1 * 0.35);
    float h = uHueWave.x * 2.4 * (0.6 * w1 + 0.4 * w2);
    float cs = cos(h), sn = sin(h);
    const mat3 toYIQ = mat3(0.299, 0.596, 0.211, 0.587, -0.274, -0.523, 0.114, -0.322, 0.312);
    const mat3 toRGB = mat3(1.0, 1.0, 1.0, 0.956, -0.272, -1.106, 0.621, -0.647, 1.703);
    vec3 y = toYIQ * c;
    y.yz = mat2(cs, sn, -sn, cs) * y.yz;
    return max(toRGB * y, 0.0);
}

vec3 scene(vec2 p, float px)
{
    vec3 pat = pattern(p, px);
    if (uMOn < 0.5) return hueWave(pat, p);
    return hueWave(blendMedia(pat, sampleMedia(p) * uMGain), p);
}

void main()
{
    vec2 p0 = (gl_FragCoord.xy - 0.5 * uRes) / (0.5 * uRes.y);
    vec2 p  = p0 / (1.0 + uZoom);
    float px = 2.0 / uRes.y / (1.0 + uZoom);

    vec3 col;
    col.r = scene(aberrate(p,  1.0), px).r;
    col.g = scene(aberrate(p,  0.0), px).g;
    col.b = scene(aberrate(p, -1.0), px).b;

    float r0 = length(p0);
    col *= 1.0 + 0.30 * uBass + 0.08 * uEnergy;                        // gentle bass lift
    col += uBass * 0.10 * exp(-r0 * 3.5) * pal(uHue + 0.1);            // core bloom
    col *= 1.0 - 0.42 * smoothstep(0.55, 2.0, r0);                     // vignette
    col *= uDim;                                                       // idle / pause dim
    col += (hash(gl_FragCoord.xy + fract(uTime) * 100.0) - 0.5) / 255.0; // anti-banding dither
    fragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}
"""

OVL_FRAG = """
#version 330
uniform sampler2D uTex;
uniform float uAlpha;
in vec2 vUV;
out vec4 fragColor;
void main(){ vec4 c = texture(uTex, vUV); fragColor = vec4(c.rgb, c.a * uAlpha); }
"""



PBLUR_FRAG = """
#version 330
uniform sampler2D uTex;
uniform vec2  uPx;       // blur radius in uv units
uniform float uAlpha;
in vec2 vUV;
out vec4 fragColor;
void main(){
    vec4 acc = vec4(0.0);
    const int N = 32;
    for (int i = 0; i < N; i++) {                      // golden-angle disc; alpha-weighted so the edges don't darken
        float r = sqrt((float(i) + 0.5) / float(N));
        float a = float(i) * 2.39996323;
        vec4 c = texture(uTex, vUV + vec2(cos(a), sin(a)) * r * uPx);
        acc += vec4(c.rgb * c.a, c.a);
    }
    acc /= float(N);
    fragColor = vec4(acc.rgb / max(acc.a, 1e-4), acc.a * uAlpha);
}
"""

PREV_FRAG = """
#version 330
uniform sampler2D uTex;
uniform vec2  uSize;     // preview size, px
uniform float uRad;      // corner radius, px
uniform float uAlpha;
uniform int   uFlip;     // 1: the texture is a decoded frame (row 0 = top)
uniform float uBlur;     // blur radius in px (tab switches)
in vec2 vUV;
out vec4 fragColor;
void main(){
    vec2 p = (vUV - 0.5) * uSize;
    vec2 q = abs(p) - uSize * 0.5 + uRad;
    float d = length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - uRad;
    float m = 1.0 - smoothstep(-0.8, 0.8, d);
    vec2 uv = uFlip == 1 ? vec2(vUV.x, 1.0 - vUV.y) : vUV;
    vec3 c;
    if (uBlur > 0.3) {
        c = vec3(0.0);
        for (int i = 0; i < 24; i++) {
            float r = sqrt((float(i) + 0.5) / 24.0);
            float a = float(i) * 2.39996323;
            c += texture(uTex, uv + vec2(cos(a), sin(a)) * r * uBlur / uSize).rgb;
        }
        c /= 24.0;
    } else {
        c = texture(uTex, uv).rgb;
    }
    float rim = 1.0 - smoothstep(0.0, 1.6, abs(d + 1.0));
    c = mix(c, vec3(1.0), rim * 0.35);
    fragColor = vec4(c, m * uAlpha);
}
"""

RECT_VERT = """
#version 330
uniform vec2 uCenter;   // NDC centre
uniform vec2 uHalf;     // NDC half extents
in vec2 in_pos;
out vec2 vUV;
void main(){ vUV = in_pos * 0.5 + 0.5; gl_Position = vec4(uCenter + in_pos * uHalf, 0.0, 1.0); }
"""

GLITCH_FRAG = """
#version 330
uniform sampler2D uTex;
uniform float uAge;      // seconds since spawn
uniform float uGlitch;   // 0..1
uniform float uSeed;
uniform float uAlpha;
in vec2 vUV;
out vec4 fragColor;

float hash(float n){ return fract(sin(n * 127.1 + uSeed * 17.3) * 43758.5453); }

void main()
{
    vec2 uv = vUV;
    float tstep = floor(uAge * 34.0);

    // horizontal slice tearing
    float rows = 18.0;
    float row  = floor(uv.y * rows);
    float h    = hash(row + tstep * 7.0);
    float tear = step(1.0 - 0.38 * uGlitch, h);
    uv.x += tear * (hash(row * 3.1 + tstep) - 0.5) * 0.30 * uGlitch;

    // fine jitter
    uv.y += (hash(tstep + 9.0) - 0.5) * 0.02 * uGlitch;

    // chromatic split (strong when glitching)
    float ca = 0.003 + 0.022 * uGlitch * (0.3 + 0.7 * hash(tstep + 1.0));
    vec4 sr = texture(uTex, uv + vec2( ca, 0.0));
    vec4 sg = texture(uTex, uv);
    vec4 sb = texture(uTex, uv - vec2( ca, 0.0));
    vec3 col = vec3(sr.r, sg.g, sb.b);
    float a  = max(max(sr.a, sg.a), sb.a);

    // occasional inverted tear bars
    float inv = tear * step(0.72, hash(row * 5.3 + tstep * 2.0));
    col = mix(col, 1.0 - col, inv * uGlitch);

    // scanlines
    col *= 0.93 + 0.07 * sin(vUV.y * 500.0);
    fragColor = vec4(col, a * uAlpha);
}
"""

DOWN_FRAG = """
#version 330
uniform sampler2D uTex;
uniform vec2 uTexel;
in vec2 vUV;
out vec4 fragColor;
void main(){
    vec3 c = texture(uTex, vUV + uTexel * vec2(-1.0, -1.0)).rgb
           + texture(uTex, vUV + uTexel * vec2( 1.0, -1.0)).rgb
           + texture(uTex, vUV + uTexel * vec2(-1.0,  1.0)).rgb
           + texture(uTex, vUV + uTexel * vec2( 1.0,  1.0)).rgb;
    fragColor = vec4(c * 0.25, 1.0);
}
"""

BLUR_FRAG = """
#version 330
uniform sampler2D uTex;
uniform vec2 uStep;
in vec2 vUV;
out vec4 fragColor;
void main(){
    vec3 c = texture(uTex, vUV).rgb * 0.2270270270;
    c += (texture(uTex, vUV + uStep * 1.3846153846).rgb + texture(uTex, vUV - uStep * 1.3846153846).rgb) * 0.3162162162;
    c += (texture(uTex, vUV + uStep * 3.2307692308).rgb + texture(uTex, vUV - uStep * 3.2307692308).rgb) * 0.0702702703;
    fragColor = vec4(c, 1.0);
}
"""

# final pass while the menu is visible: blurred backdrop + frosted "liquid glass" panel
COMP_FRAG = """
#version 330
uniform sampler2D uScene;
uniform sampler2D uA;      // backdrop blur
uniform sampler2D uC;      // heavier blur for the glass
uniform vec2  uRes;
uniform float uT;          // eased menu progress 0..1
uniform vec2  uRectC;      // panel centre, px (top-left origin)
uniform vec2  uRectH;      // panel half size, px
uniform float uRad;        // corner radius, px
in vec2 vUV;
out vec4 fragColor;

float sdRound(vec2 p, vec2 c, vec2 h, float r){
    vec2 q = abs(p - c) - h + r;
    return length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - r;
}
float hash(vec2 p){ return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453); }

void main(){
    vec2 p = vec2(gl_FragCoord.x, uRes.y - gl_FragCoord.y);
    vec3 scene = texture(uScene, vUV).rgb;
    vec3 blur  = texture(uA, vUV).rgb;
    vec3 col = mix(scene, blur, smoothstep(0.0, 0.3, uT));
    col *= 1.0 - 0.26 * uT;

    float d  = sdRound(p, uRectC, uRectH, uRad);
    float ds = sdRound(p - vec2(0.0, uRad * 0.5), uRectC, uRectH, uRad);
    col *= 1.0 - 0.40 * uT * (1.0 - smoothstep(-uRad * 0.3, uRad * 1.8, ds));   // soft drop shadow

    vec3 g = texture(uC, vUV).rgb;
    float l = dot(g, vec3(0.299, 0.587, 0.114));
    g = mix(vec3(l), g, 1.4);                     // vibrancy
    g = g * 0.50 + 0.09;                          // frosted tint
    float ty = clamp((p.y - (uRectC.y - uRectH.y)) / (2.0 * uRectH.y), 0.0, 1.0);
    g += 0.075 * (1.0 - ty) * (1.0 - ty);         // top sheen
    float tx = clamp((p.x - (uRectC.x - uRectH.x)) / (2.0 * uRectH.x), 0.0, 1.0);
    float rim = 1.0 - smoothstep(0.0, 2.2, abs(d + 1.0));
    g += rim * (0.10 + 0.40 * pow(clamp(1.0 - (tx * 0.6 + ty * 0.8), 0.0, 1.0), 1.5));  // specular edge
    g += (hash(gl_FragCoord.xy) - 0.5) / 120.0;   // grain, hides banding

    float mask = 1.0 - smoothstep(-0.8, 0.8, d);
    col = mix(col, g, mask * uT);
    fragColor = vec4(col, 1.0);
}
"""

# blur-fade between visual modes: the old look blurs out while the new one blurs in
TRANS_FRAG = """
#version 330
uniform sampler2D uNew, uOld, uNewB, uOldB;
uniform float uK;
in vec2 vUV;
out vec4 fragColor;
void main(){
    float bo = smoothstep(0.0, 0.5, uK);            // old image: sharp -> blurred
    float bn = 1.0 - smoothstep(0.5, 1.0, uK);      // new image: blurred -> sharp
    float wn = smoothstep(0.22, 0.78, uK);          // cross-fade weight
    vec3 o = mix(texture(uOld, vUV).rgb, texture(uOldB, vUV).rgb, smoothstep(0.0, 0.3, bo));
    vec3 n = mix(texture(uNew, vUV).rgb, texture(uNewB, vUV).rgb, smoothstep(0.0, 0.3, bn));
    fragColor = vec4(mix(o, n, wn), 1.0);
}
"""

COPY_FRAG = """
#version 330
uniform sampler2D uTex;
in vec2 vUV;
out vec4 fragColor;
void main(){ fragColor = vec4(texture(uTex, vUV).rgb, 1.0); }
"""

def smoothstep(a, b, x):
    t = max(0.0, min(1.0, (x - a) / (b - a)))
    return t * t * (3.0 - 2.0 * t)


def lerp_col(a, b, t):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(4))


DEFAULT_PHRASES = [
    "Submit", "Don't Resist", "Temptations", "Good Puppy", "Look Closer", "Eyes Forward",
    "Obey", "Go Deeper", "Let Go", "Empty Your Mind", "Keep Staring", "Give In",
    "Sink Deeper", "Focus", "Surrender", "Stay Still", "Don't Look Away", "Good Pet",
]

DEFAULTS = dict(
    volume=0.85, mode=0, fs_kind="borderless",
    spin=1.0, zoom=1.0, ab=1.0, color=1.0,
    dom_size=1.0, dom_rate=1.0,
    loop=False, help=True, domination=False,
    spotify=False, spot_delay=40.0, kzoom=1.0,
    media_path="", media_paths="", media_remember=False, media_on=True, media_blend=0, media_calm=0.40, media_peak=1.0, media_auto=True, media_speed=1.0, media_rate=1.0, media_pulse=1.0,
    beat_smooth=0.35, media_flash=0.0, media_smooth=True, media_sway=False, media_sway_amt=1.0, media_hit=False, media_style=0, hit_cool=1.0,
    kick_lo=30.0, kick_hi=150.0, kick_sens=1.0, snare_lo=170.0, snare_hi=420.0, snare_sens=1.0,
    vis_open="style,preview,motion",
)

SLIDERS = [  # key, label, min, max   (grouped: dividers are drawn after rows 0, 5 and 7)
    ("volume", "Volume", 0.0, 1.0),
    ("spin", "Spin response", 0.0, 2.0),
    ("zoom", "Bass zoom", 0.0, 2.0),
    ("ab", "Chromatic aberration", 0.0, 2.0),
    ("color", "Colour drift & wave", 0.0, 3.0),
    ("kzoom", "Kaleidoscope zoom speed", 0.0, 3.0),
    ("dom_size", "Domination: text size", 0.5, 2.0),
    ("dom_rate", "Domination: text rate", 0.3, 3.0),
    ("spot_delay", "Spotify sync delay", 0.0, 400.0),
    ("media_calm", "Opacity \u2013 calm", 0.0, 1.0),
    ("media_peak", "Opacity \u2013 at max", 0.0, 1.0),
    ("media_speed", "Kick/snare burst", 0.0, 2.0),
    ("media_flash", "Kick/snare flash", 0.0, 1.0),
    ("media_sway_amt", "Camera sway intensity", 0.0, 2.0),
    ("media_rate", "Scene change rate", 0.3, 3.0),
    ("media_pulse", "Beat pulse", 0.0, 2.0),
    ("beat_smooth", "Effect smoothing", 0.0, 1.0),
    ("hit_cool", "Hit-cut cooldown", 0.2, 4.0),
    ("kick_lo", "Kick range \u2013 low", 20.0, 120.0),
    ("kick_hi", "Kick range \u2013 high", 60.0, 300.0),
    ("kick_sens", "Kick sensitivity", 0.4, 2.5),
    ("snare_lo", "Snare range \u2013 low", 100.0, 600.0),
    ("snare_hi", "Snare range \u2013 high", 200.0, 1200.0),
    ("snare_sens", "Snare sensitivity", 0.4, 2.5),
]
SETTINGS_KEYS = ("spot_delay",)
VISUAL_KEYS = ("spin", "zoom", "ab", "beat_smooth", "color", "kzoom", "dom_size", "dom_rate")
BEAT_KEYS = ("hit_cool", "kick_lo", "kick_hi", "kick_sens", "snare_lo", "snare_hi", "snare_sens")
MEDIA_KEYS = ("media_calm", "media_peak", "media_speed", "media_flash", "media_sway_amt", "media_rate", "media_pulse")
BLENDS = ["Spiral window", "Soft overlay", "Glow"]


def slider_norm(key, v):
    lo, hi = next((a, b) for k, _, a, b in SLIDERS if k == key)
    return 0.0 if hi == lo else max(0.0, min(1.0, (v - lo) / (hi - lo)))


def spring_step(x, v, target, k, c, dt):
    """Damped spring, sub-stepped so it stays stable at any frame time."""
    n = max(1, int(math.ceil(dt * 240.0)))
    h = dt / n
    for _ in range(n):
        v += (-k * (x - target) - c * v) * h
        x += v * h
    return x, v


# values that wrap seamlessly (every shader frequency is a multiple of 0.005 rad/s)
TAU_F = 6.283185307179586
TIME_WRAP = TAU_F / 0.005
FLOWK_PERIOD = 2.0 / 0.8660254037844386
MODES = ["Classic", "Prism", "Neon", "Kaleidoscope", "Psychedelic", "Video"]
VIDEO_MODE = 5            # shows only the media layer (no spiral); keeps playing at its last speed when the audio stops


def app_dir():
    return os.path.dirname(sys.executable if getattr(sys, "frozen", False) else os.path.abspath(__file__))


def settings_path():
    base = os.environ.get("APPDATA") or os.path.join(os.path.expanduser("~"), ".config")
    return os.path.join(base, "Hypnosis", "settings.json")


def scenes_path():
    return os.path.join(os.path.dirname(settings_path()), "scenes.json")


SCENE_TOL = 0.3        # a hidden / deleted scene matches a detected scene start within this many seconds


def load_scene_prefs():
    """{video path: {"off": [scene start, ...], "del": [...]}} - which scenes the user hid or deleted."""
    try:
        with open(scenes_path(), "r", encoding="utf-8") as f:
            raw = json.load(f)
        out = {}
        for p, v in raw.items():
            off = [float(x) for x in v.get("off", []) if isinstance(x, (int, float))]
            dl = [float(x) for x in v.get("del", []) if isinstance(x, (int, float))]
            tr = []
            for x in v.get("trim", []):
                try:
                    t0, a0, b0 = (float(q) for q in x)
                    if b0 - a0 >= 0.1:
                        tr.append([t0, a0, b0])
                except Exception:
                    pass
            if off or dl or tr:
                out[str(p)] = {"off": off, "del": dl, "trim": tr}
        return out
    except Exception:
        return {}


def save_scene_prefs(prefs):
    try:
        p = scenes_path()
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump({k: v for k, v in prefs.items() if v["off"] or v["del"] or v.get("trim")}, f, indent=1)
    except Exception:
        pass


def _has(lst, t):
    return any(abs(x - t) <= SCENE_TOL for x in lst)


PRESET_SKIP = {"vis_open", "media_remember", "volume", "spotify", "media_path", "media_paths", "fs_kind", "spot_delay", "help"}


def presets_path():
    return os.path.join(os.path.dirname(settings_path()), "presets.json")


def load_presets():
    """presets.json -> list of (name, settings dict), oldest first. Bad files / values are ignored."""
    out = []
    try:
        with open(presets_path(), "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data.get("presets", []):
            name, vals = str(item.get("name", ""))[:40], item.get("values", {})
            if not name or not isinstance(vals, dict):
                continue
            clean = {}
            for k, v in vals.items():
                if k not in DEFAULTS or k in PRESET_SKIP:
                    continue
                d = DEFAULTS[k]
                if isinstance(d, bool):
                    ok = isinstance(v, bool)
                elif isinstance(d, (int, float)):
                    ok = isinstance(v, (int, float)) and not isinstance(v, bool)
                else:
                    ok = isinstance(v, str)
                if ok:
                    clean[k] = v
            out.append((name, clean))
    except Exception:
        pass
    return out


def save_presets(presets, last=""):
    try:
        p = presets_path()
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump({"last": last, "presets": [{"name": n, "values": v} for n, v in presets]}, f, indent=1)
    except Exception:
        pass


def load_last_preset_name():
    try:
        with open(presets_path(), "r", encoding="utf-8") as f:
            return str(json.load(f).get("last", ""))
    except Exception:
        return ""


def log(msg):
    """Append to %APPDATA%\\Hypnosis\\log.txt (works even in a --noconsole exe)."""
    try:
        p = os.path.join(os.path.dirname(settings_path()), "log.txt")
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "a", encoding="utf-8") as f:
            f.write(time.strftime("%Y-%m-%d %H:%M:%S  ") + msg + "\n")
    except Exception:
        pass


def load_settings():
    cfg = dict(DEFAULTS)
    try:
        with open(settings_path(), "r", encoding="utf-8") as f:
            saved = json.load(f)
        for k, v in saved.items():
            if k not in cfg:
                continue
            d = cfg[k]
            if isinstance(d, bool):
                ok = isinstance(v, bool)
            elif isinstance(d, (int, float)):
                ok = isinstance(v, (int, float)) and not isinstance(v, bool)
            else:
                ok = isinstance(v, str)
            if ok:
                cfg[k] = v
    except Exception:
        pass
    if cfg.get("media_blend", 0) not in (0, 1, 2):          # the old "Video only" blend became the Video visual style
        cfg["media_blend"] = 0
    return cfg


def save_settings(cfg):
    try:
        p = settings_path()
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            out = dict(cfg)
            if not out.get("media_remember"):                # forget the clip stack when the app closes
                out["media_paths"], out["media_path"] = "", ""
            json.dump(out, f, indent=1)
    except Exception:
        pass


def load_phrases():
    """phrases.txt (one per line) next to the app or in the settings folder overrides the defaults."""
    for folder in (app_dir(), os.path.dirname(settings_path())):
        p = os.path.join(folder, "phrases.txt")
        if os.path.isfile(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]
                if lines:
                    return lines
            except Exception:
                pass
    return list(DEFAULT_PHRASES)


def pick_files_dialog(out):
    """Runs in a thread; pushes chosen paths into `out`."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        files = filedialog.askopenfilenames(
            title="Add music to queue",
            filetypes=[("Audio", "*.mp3 *.wav *.flac *.ogg *.oga *.m4a *.aac *.opus *.wma"), ("All files", "*.*")],
        )
        root.destroy()
        out.extend(files)
    except Exception:
        pass


def collect_audio(paths):
    out = []
    for p in paths:
        if os.path.isdir(p):
            for root, _, files in os.walk(p):
                for f in sorted(files):
                    if os.path.splitext(f)[1].lower() in AUDIO_EXT:
                        out.append(os.path.join(root, f))
        elif os.path.isfile(p):
            out.append(p)
    return out


def fmt_slider(key, v):
    if key == "spot_delay":
        return f"{v:.0f} ms"
    if key == "hit_cool":
        return f"{v:.2f} s"
    if key in ("kick_lo", "kick_hi", "snare_lo", "snare_hi"):
        return f"{v:.0f} Hz"
    return f"{v * 100:.0f}%"


def send_media_key(vk):
    """Global media key (0xB3 play/pause, 0xB0 next, 0xB1 previous) - Spotify reacts to these."""
    if sys.platform == "win32":
        try:
            u = ctypes.windll.user32
            u.keybd_event(vk, 0, 0, 0)
            u.keybd_event(vk, 0, 2, 0)
        except Exception:
            pass


def fetch_album_art(title):
    """Album art for a Spotify window title 'Artist - Song' via the public iTunes search (no key). Returns a PIL image or None."""
    import urllib.request, urllib.parse, json as _json, io, re
    from PIL import Image
    if " - " not in title:
        return None
    artist, song = title.split(" - ", 1)
    clean = lambda t: re.sub(r"[\(\[].*?[\)\]]", "", t).strip().lower()
    q = urllib.parse.quote(f"{artist} {clean(song)}")
    hdr = {"User-Agent": "Hypnosis/1.0"}
    with urllib.request.urlopen(urllib.request.Request(f"https://itunes.apple.com/search?term={q}&entity=song&limit=8", headers=hdr), timeout=6) as r:
        res = _json.loads(r.read().decode("utf-8", "replace")).get("results", [])
    pick = None
    for it in res:
        if clean(it.get("trackName", "")) == clean(song) and clean(artist.split(",")[0]) in it.get("artistName", "").lower():
            pick = it
            break
    pick = pick or next((it for it in res if clean(song) in it.get("trackName", "").lower()), None) or (res[0] if res else None)
    if not pick or not pick.get("artworkUrl100"):
        return None
    url = pick["artworkUrl100"].replace("100x100bb", "300x300bb")
    with urllib.request.urlopen(urllib.request.Request(url, headers=hdr), timeout=8) as r:
        return Image.open(io.BytesIO(r.read())).convert("RGB")


_SPOT_BASE = None


def spot_icon_base():
    """(luma, mask) of the Spotify button picture: spotify_icon.png next to the program (the glassy disc is cropped out of its black
    background); without the file a plain glassy disc with a sound-wave is used. Both 128x128 float arrays 0..1."""
    global _SPOT_BASE
    if _SPOT_BASE is not None:
        return _SPOT_BASE
    from PIL import Image
    N = 128
    yy, xx = np.mgrid[0:N, 0:N].astype(np.float32)
    r = np.hypot(xx - N / 2 + 0.5, yy - N / 2 + 0.5) / (N / 2)
    try:
        im = Image.open(os.path.join(app_dir(), "spotify_icon.png")).convert("RGB")
        a = np.asarray(im).astype(np.float32)
        lum = a.mean(2)
        ys, xs = np.where(lum > 14)
        cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
        rad = max(xs.max() - xs.min(), ys.max() - ys.min()) / 2 + 2
        crop = im.crop((int(cx - rad), int(cy - rad), int(cx + rad), int(cy + rad))).resize((N, N), Image.LANCZOS)
        L = np.asarray(crop).astype(np.float32).mean(2) / 255.0
        mask = np.clip((1.0 - r) * (N / 2) / 1.5, 0.0, 1.0)
        L = np.clip(L * 1.6 + 0.10, 0.0, 1.0)
    except Exception:
        mask = np.clip((1.0 - r) * (N / 2) / 1.5, 0.0, 1.0)
        L = 0.22 + 0.55 * np.clip((r - 0.82) / 0.18, 0.0, 1.0) + 0.12 * (1.0 - yy / N)
        for k, h in enumerate((0.25, 0.45, 0.65, 0.45, 0.25)):
            x0 = N * (0.28 + k * 0.11)
            bar = (np.abs(xx - x0) < 4.0) & (np.abs(yy - N / 2) < N * h / 2)
            L = np.where(bar, 0.78, L)
        L = np.clip(L, 0.0, 1.0)
    _SPOT_BASE = (L.astype(np.float32), mask.astype(np.float32))
    return _SPOT_BASE


def spot_icon_pixels(px, to_on, prog):
    """The Spotify button as RGBA bytes: grey when off, green when on; while prog (0..1) runs a colourful wave sweeps across
    the glass from the old colour to the new one."""
    from PIL import Image
    L0, M0 = spot_icon_base()
    L = np.asarray(Image.fromarray((L0 * 255).astype(np.uint8)).resize((px, px), Image.LANCZOS)).astype(np.float32) / 255.0
    M = np.asarray(Image.fromarray((M0 * 255).astype(np.uint8)).resize((px, px), Image.LANCZOS)).astype(np.float32) / 255.0
    green, grey = np.array([30, 215, 96], np.float32), np.array([138, 140, 152], np.float32)
    lit = (0.30 + 1.15 * L)[..., None]
    hi = (255.0 * np.clip(L - 0.72, 0.0, 1.0) * 1.1)[..., None]

    def tint(c):
        return np.clip(c[None, None, :] * lit + hi, 0, 255)

    new, old = (green, grey) if to_on else (grey, green)
    if prog is None:
        rgb = tint(new if to_on else grey) if to_on else tint(grey)
        alpha = M * (1.0 if to_on else 0.82)
    else:
        v, u = np.mgrid[0:px, 0:px].astype(np.float32) / max(1, px - 1)
        d = u * 0.75 + (1.0 - v) * 0.25 + 0.10 * np.sin(v * 9.0 + prog * 14.0)
        f = -0.28 + prog * 1.56
        m = np.clip((d - (f - 0.12)) / 0.24, 0.0, 1.0)
        m = m * m * (3 - 2 * m)                                   # 1 = wave not here yet (old colour), 0 = passed (new colour)
        band = np.exp(-(((d - f) / 0.17) ** 2))
        h = (d * 2.2 + prog * 2.5) % 1.0
        rainbow = np.stack([np.clip(np.abs(h * 6 - 3) - 1, 0, 1), np.clip(2 - np.abs(h * 6 - 2), 0, 1), np.clip(2 - np.abs(h * 6 - 4), 0, 1)], -1) * 255.0
        base = tint(old) * m[..., None] + tint(new) * (1 - m[..., None])
        glow = np.clip(rainbow * lit + hi, 0, 255)
        rgb = base * (1 - 0.85 * band[..., None]) + glow * (0.85 * band[..., None])
        a0, a1 = (0.82, 1.0) if to_on else (1.0, 0.82)
        alpha = M * (a1 * (1 - m) + a0 * m)
    out = np.dstack([rgb, alpha[..., None] * 255.0]).astype(np.uint8)
    return out.tobytes()


class SmtcSpotify:
    """Spotify's playback position and seeking through Windows' media session (the same thing the volume flyout shows).
    Needs the winrt packages; without them ok stays False and the bar stays read-only."""

    def __init__(self, active, on_missing):
        self.active, self.on_missing = active, on_missing
        self.ok = None                     # None = not tried yet, True = connected, False = unavailable
        self.dur = None                    # track length in seconds (None = unknown)
        self.base, self.stamp, self.playing = 0.0, 0.0, False
        self._target = None                # queued seek (seconds)
        self.hold = 0.0                    # ignore the polled position until then (a seek is on its way)
        self.thread = None

    def start(self):
        if self.thread is None:
            self.thread = threading.Thread(target=self._thread, daemon=True)
            self.thread.start()

    def position(self):
        if self.dur is None:
            return None
        p = self.base + ((time.perf_counter() - self.stamp) if self.playing else 0.0)
        return max(0.0, min(self.dur, p))

    def seek(self, t):
        if self.dur:
            t = max(0.0, min(self.dur, t))
            self.base, self.stamp, self.hold = t, time.perf_counter(), time.perf_counter() + 1.5
            self._target = t

    def _thread(self):
        import asyncio
        try:
            asyncio.run(self._run())
        except Exception as ex:
            self.ok = False
            log(f"SMTC thread ended: {ex}")

    async def _run(self):
        import asyncio
        from datetime import datetime, timezone
        try:
            from winrt.windows.media.control import GlobalSystemMediaTransportControlsSessionManager as Mgr
            mgr = await Mgr.request_async()
        except Exception as ex:
            self.ok = False
            log(f"SMTC unavailable: {ex}")
            self.on_missing()
            return
        self.ok = True
        warned = False
        while True:
            try:
                if not self.active():
                    self.dur = None
                    await asyncio.sleep(0.6)
                    continue
                ses = None
                for s_ in mgr.get_sessions():
                    if "spotify" in (s_.source_app_user_model_id or "").lower():
                        ses = s_
                if ses is None:
                    self.dur = None
                else:
                    if self._target is not None:
                        t, self._target = self._target, None
                        await ses.try_change_playback_position_async(int(t * 10_000_000))
                    tl = ses.get_timeline_properties()
                    start, end = tl.start_time.total_seconds(), tl.end_time.total_seconds()
                    playing = int(ses.get_playback_info().playback_status) == 4          # Playing
                    pos = tl.position.total_seconds() - start
                    if playing:
                        age = (datetime.now(timezone.utc) - tl.last_updated_time).total_seconds()
                        pos += max(0.0, min(age, 30.0))
                    if end - start > 1.0:
                        self.dur = end - start
                        if time.perf_counter() >= self.hold:
                            self.base, self.stamp = pos, time.perf_counter()
                        self.playing = playing
                    else:
                        self.dur = None
            except Exception as ex:
                if not warned:
                    warned = True
                    log(f"SMTC poll failed: {ex}")
            await asyncio.sleep(0.25)




def spotify_session_volume(set_to=None):
    """Spotify's own volume in the Windows mixer (needs pycaw). Reads it (or sets it when set_to is given).
    Returns the volume 0..1, or None when Spotify has no audio session yet (it appears once Spotify plays something)."""
    if sys.platform != "win32":
        return None
    from pycaw.pycaw import AudioUtilities
    out = None
    for ses in AudioUtilities.GetAllSessions():
        try:
            p = ses.Process
            if p is None or p.name().lower() != "spotify.exe":
                continue
            vol = ses.SimpleAudioVolume
            if set_to is not None:
                vol.SetMasterVolume(float(max(0.0, min(1.0, set_to))), None)
            out = float(vol.GetMasterVolume())
        except Exception:
            continue
    return out


def find_spotify_window():
    """(is_running, window_title) for the Spotify desktop app. Title is 'Artist - Song' while playing."""
    if sys.platform != "win32":
        return False, ""
    from ctypes import wintypes
    user32, kernel32 = ctypes.windll.user32, ctypes.windll.kernel32
    user32.IsWindowVisible.argtypes = [wintypes.HWND]
    user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
    user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
    user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.QueryFullProcessImageNameW.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD)]
    titles, found = [], [False]
    path = ctypes.create_unicode_buffer(1024)

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def cb(hwnd, _):
        if not user32.IsWindowVisible(hwnd):
            return True
        pid = wintypes.DWORD(0)
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        h = kernel32.OpenProcess(0x1000, False, pid.value)      # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return True
        try:
            size = wintypes.DWORD(1024)
            if kernel32.QueryFullProcessImageNameW(h, 0, path, ctypes.byref(size)):
                if os.path.basename(path.value).lower() == "spotify.exe":
                    found[0] = True
                    n = user32.GetWindowTextLengthW(hwnd)
                    if n > 0:
                        t = ctypes.create_unicode_buffer(n + 1)
                        user32.GetWindowTextW(hwnd, t, n + 1)
                        titles.append(t.value)
        finally:
            kernel32.CloseHandle(h)
        return True

    user32.EnumWindows(cb, 0)
    return found[0], (titles[0] if titles else "")


def find_spotify_root_pid():
    """PID of the main Spotify process (the one whose parent isn't also spotify.exe), or None."""
    if sys.platform != "win32":
        return None
    from ctypes import wintypes

    class PROCESSENTRY32W(ctypes.Structure):
        _fields_ = [("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD), ("th32ProcessID", wintypes.DWORD),
                    ("th32DefaultHeapID", ctypes.c_size_t), ("th32ModuleID", wintypes.DWORD),
                    ("cntThreads", wintypes.DWORD), ("th32ParentProcessID", wintypes.DWORD),
                    ("pcPriClassBase", ctypes.c_long), ("dwFlags", wintypes.DWORD),
                    ("szExeFile", ctypes.c_wchar * 260)]

    k32 = ctypes.windll.kernel32
    k32.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    k32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    k32.Process32FirstW.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESSENTRY32W)]
    k32.Process32NextW.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESSENTRY32W)]
    k32.CloseHandle.argtypes = [wintypes.HANDLE]
    snap = k32.CreateToolhelp32Snapshot(0x2, 0)               # TH32CS_SNAPPROCESS
    if not snap or snap == wintypes.HANDLE(-1).value:
        return None
    procs = {}
    try:
        e = PROCESSENTRY32W()
        e.dwSize = ctypes.sizeof(PROCESSENTRY32W)
        ok = k32.Process32FirstW(snap, ctypes.byref(e))
        while ok:
            if e.szExeFile.lower() == "spotify.exe":
                procs[e.th32ProcessID] = e.th32ParentProcessID
            ok = k32.Process32NextW(snap, ctypes.byref(e))
    finally:
        k32.CloseHandle(snap)
    roots = [pid for pid, parent in procs.items() if parent not in procs]
    return roots[0] if roots else None


class ProcessCapture:
    """Hears ONLY one application's audio (Spotify) using WASAPI process loopback.
    Uses the native module from the `proc-tap` package; includes the app's child processes."""

    def __init__(self, pid):
        self.pid = pid
        self.native = None
        self.rate = 48000
        self.channels = 2
        self.bits = 32
        self.L = self.rate * 3
        self.buf = np.zeros(self.L, np.float32)
        self.w = 0
        self.last = 0.0
        self.last_loud = 0.0
        self.active = False
        self.lock = threading.Lock()
        self._run = False
        self._thread = None

    def start(self):
        from proctap._native import ProcessLoopback          # ImportError -> caller shows an install hint
        self.native = ProcessLoopback(int(self.pid))
        fmt = self.native.get_format()
        self.rate = int(fmt["sample_rate"])
        self.channels = max(1, int(fmt["channels"]))
        self.bits = int(fmt["bits_per_sample"])
        self.L = self.rate * 3
        self.buf = np.zeros(self.L, np.float32)
        self.w = 0
        self.native.start()
        self._run = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while self._run:
            try:
                data = self.native.read()                    # non-blocking; None when nothing is buffered
            except Exception:
                time.sleep(0.01)
                continue
            if not data:
                time.sleep(0.002)
                continue
            try:
                if self.bits == 32:
                    x = np.frombuffer(data, np.float32)
                else:                                        # 16-bit fallback format
                    x = np.frombuffer(data, np.int16).astype(np.float32) / 32768.0
                n = (len(x) // self.channels) * self.channels
                if n == 0:
                    continue
                x = x[:n]
                if self.channels > 1:
                    x = x.reshape(-1, self.channels).mean(axis=1)
                with self.lock:
                    idx = (self.w + np.arange(len(x))) % self.L
                    self.buf[idx] = x
                    self.w += len(x)
                self.last = time.perf_counter()
            except Exception:
                pass

    def window(self, n, delay_s=0.0):
        """Latest n mono samples (optionally `delay_s` in the past). None when Spotify is silent."""
        now = time.perf_counter()
        if now - self.last > 0.25:                           # no packets while the app is silent
            self.active = now - self.last_loud < 0.6
            return None
        with self.lock:
            end = self.w - int(delay_s * self.rate)
            if end < n:
                return None
            x = self.buf[(np.arange(end - n, end)) % self.L].copy()
        if float(np.sqrt(np.mean(x * x))) > 1e-4:
            self.last_loud = now
        self.active = now - self.last_loud < 0.6
        return x if self.active else None

    def stop(self):
        self._run = False
        if self._thread:
            self._thread.join(timeout=1.0)
        try:
            if self.native:
                self.native.stop()
        except Exception:
            pass
        self.native = None
        self._thread = None
        self.active = False


def fmt_time(sec):
    sec = int(max(0, sec))
    return f"{sec // 60}:{sec % 60:02d}"


def ndc_rect(x, y, w, h, W, H):
    return ((x + w / 2) / W * 2 - 1, 1 - (y + h / 2) / H * 2), (w / W, h / H)


# --------------------------------------------------------------------------- GL helpers
class TexQuad:
    """Draws an RGBA texture into a pixel rectangle with alpha blending."""

    def __init__(self, ctx, quad):
        self.ctx = ctx
        self.prog = ctx.program(vertex_shader=RECT_VERT, fragment_shader=OVL_FRAG)
        self.vao = ctx.vertex_array(self.prog, [(quad, "2f", "in_pos")])
        self.white = ctx.texture((1, 1), 4, bytes([0, 0, 0, 255]))   # black 1x1 for dimming

    def draw(self, tex, rect, screen, alpha=1.0):
        (cx, cy), (hx, hy) = ndc_rect(*rect, *screen)
        self.prog["uCenter"].value = (cx, cy)
        self.prog["uHalf"].value = (hx, hy)
        self.prog["uAlpha"].value = float(min(1.0, alpha))
        tex.use(0)
        self.prog["uTex"].value = 0
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self.vao.render(moderngl.TRIANGLE_STRIP)
        self.ctx.disable(moderngl.BLEND)

    def dim(self, screen, alpha):
        W, H = screen
        self.draw(self.white, (0, 0, W, H), screen, alpha)


def surface_to_texture(ctx, surf, old=None):
    conv = getattr(pygame.image, "tobytes", None) or pygame.image.tostring
    data = conv(surf, "RGBA", True)
    if old is not None:
        old.release()
    tex = ctx.texture(surf.get_size(), 4, data)
    tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
    tex.repeat_x = False
    tex.repeat_y = False
    return tex


class Overlay:
    """Centered toast / hint text."""

    def __init__(self, ctx, quads):
        self.ctx, self.quads = ctx, quads
        self.tex = None
        self.key = None
        self.fonts = {}

    def font(self, size):
        if size not in self.fonts:
            self.fonts[size] = pygame.font.SysFont("segoeui,helveticaneue,dejavusans,arial", size)
        return self.fonts[size]

    def build(self, lines, size):
        w, h = size
        surf = pygame.Surface((w, h), pygame.SRCALPHA)
        scale = max(0.7, min(w, h) / 720.0)
        total = sum(int(self.font(int(fs * scale)).get_height() * 1.25) for _, fs in lines)
        y = (h * 0.86 - total) if len(lines) > 1 else (h * 0.5 - total / 2)
        for text, fs in lines:
            f = self.font(int(fs * scale))
            for off, colr in (((2, 2), (0, 0, 0)), ((0, 0), (255, 255, 255))):
                s = f.render(text, True, colr)
                surf.blit(s, (w / 2 - s.get_width() / 2 + off[0], y + off[1]))
            y += int(f.get_height() * 1.25)
        self.tex = surface_to_texture(self.ctx, surf, self.tex)

    def draw(self, lines, size, alpha):
        if not lines or alpha <= 0.003:
            return
        key = (tuple(lines), size)
        if key != self.key:
            self.build(lines, size)
            self.key = key
        self.quads.draw(self.tex, (0, 0, size[0], size[1]), size, alpha)


class GlitchText:
    """Domination-mode popups: white text on a black banner, torn + chromatic-split."""

    def __init__(self, ctx, quad, quads):
        self.ctx, self.quads = ctx, quads
        self.prog = ctx.program(vertex_shader=RECT_VERT, fragment_shader=GLITCH_FRAG)
        self.vao = ctx.vertex_array(self.prog, [(quad, "2f", "in_pos")])
        self.cache = {}
        self.font = pygame.font.SysFont("impact,arialblack,segoeuiblack,arial", 150)
        self.active = []
        self.last_phrase = None

    def texture_for(self, phrase):
        if phrase not in self.cache:
            txt = self.font.render(phrase.upper(), True, (255, 255, 255))
            padx, pady, b = 54, 26, 4
            w, h = txt.get_width() + 2 * (padx + b), txt.get_height() + 2 * (pady + b)
            surf = pygame.Surface((w, h), pygame.SRCALPHA)
            pygame.draw.rect(surf, (0, 0, 0, 255), (b, b, w - 2 * b, h - 2 * b))
            surf.blit(txt, (b + padx, b + pady))
            self.cache[phrase] = (surface_to_texture(self.ctx, surf), w, h)
        return self.cache[phrase]

    def spawn(self, phrases, now, screen, size_mul):
        W, H = screen
        choices = [p for p in phrases if p != self.last_phrase] or phrases
        phrase = random.choice(choices)
        self.last_phrase = phrase
        life = random.uniform(0.38, 0.55)
        height = H * 0.21 * size_mul * random.uniform(0.85, 1.25)
        self.active.append(dict(
            phrase=phrase, born=now, life=life, height=height,
            cx=W * 0.5 + random.uniform(-0.22, 0.22) * W,
            cy=H * 0.5 + random.uniform(-0.20, 0.20) * H,
            seed=random.uniform(0, 100),
        ))
        self.active = self.active[-3:]

    def draw(self, now, screen):
        W, H = screen
        keep = []
        for p in self.active:
            age = now - p["born"]
            if age > p["life"]:
                continue
            keep.append(p)
            tex, tw, th = self.texture_for(p["phrase"])
            height = p["height"] * (1.0 + 0.14 * max(0.0, 1.0 - age / 0.09))     # punch-in
            width = height * tw / th
            if width > W * 0.94:
                height *= W * 0.94 / width
                width = W * 0.94
            tail = p["life"] - age
            glitch = 1.0 if (age < 0.10 or tail < 0.12) else 0.38
            alpha = 1.0
            if tail < 0.12 and random.random() < 0.4:
                alpha = 0.15
            jx = random.uniform(-1, 1) * glitch * 0.012 * W
            jy = random.uniform(-1, 1) * glitch * 0.006 * H
            x, y = p["cx"] - width / 2 + jx, p["cy"] - height / 2 + jy
            (cx, cy), (hx, hy) = ndc_rect(x, y, width, height, W, H)
            pr = self.prog
            pr["uCenter"].value = (cx, cy)
            pr["uHalf"].value = (hx, hy)
            pr["uAge"].value = age
            pr["uGlitch"].value = glitch
            pr["uSeed"].value = p["seed"]
            pr["uAlpha"].value = alpha
            tex.use(0)
            pr["uTex"].value = 0
            self.ctx.enable(moderngl.BLEND)
            self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
            self.vao.render(moderngl.TRIANGLE_STRIP)
            self.ctx.disable(moderngl.BLEND)
        self.active = keep


# --------------------------------------------------------------------------- menu panel
PW, PH = 840, 660
HDR_Y, FTR_Y = 64, 584                     # the Visuals page scrolls between the header line and the footer line
PREV_W = 420                                # live preview width on the Visuals tab (logical px)


SEC_HDR_H, SEC_GAP, SEC_PAD = 38, 8, 8        # Visuals page: section header height, gap between sections, space above a section's content
VIS_SECTIONS = (("style", "Visual style"), ("preview", "Live preview"), ("motion", "Motion & effects"), ("media", "Media & stack"))
MEDIA_H = 592                                   # height of the Media & stack content
MFX_H, MSEP_H = 352, 46                         # the "Media effects" block inside Motion & effects, and the divider above it
CARD_H = 101


def visuals_layout(aspect, anim=None):
    """The Visuals page: collapsible sections stacked from the top. anim = {name: 0..1} (how far each one is open).
    Each section: hy (header top), ct (content top), ch (full content height), vis (visible content height), plus the scroll range."""
    ph = int(max(170, min(270, PREV_W * aspect)))
    full = dict(style=CARD_H + 4, preview=ph + 12, motion=len(VISUAL_KEYS) * SL_PITCH + 12 + MSEP_H + MFX_H, media=MEDIA_H)
    out, y = {}, 72.0
    for name, _ in VIS_SECTIONS:
        a = 1.0 if anim is None else max(0.0, min(1.0, anim.get(name, 1.0)))
        out[name] = dict(hy=y, ct=y + SEC_HDR_H + SEC_PAD, ch=full[name], a=a, vis=a * (full[name] + SEC_PAD))
        y += SEC_HDR_H + a * (full[name] + SEC_PAD) + SEC_GAP
    out["ph"] = ph
    out["end"] = y
    out["max_scroll"] = max(0, int(math.ceil(y - FTR_Y + 4)))
    return out


PREV_MINI = 0.42                                 # size of the preview once it floats along with the scroll


def preview_geom(lay, vscroll):
    """Where the Live preview sits (logical px): in its section, or - once scrolled past - docked at the top right, never above its section.
    Returns x, y, w, h and f (0 = in place, 1 = fully docked)."""
    L = lay["preview"]
    y_n = L["ct"] + 4 - vscroll
    line = HDR_Y + 8
    d = line - y_n
    w, h = float(PREV_W), float(lay["ph"])
    if d <= 0:
        return (PW - w) / 2, y_n, w, h, 0.0
    t = min(1.0, d / (h * 0.7))
    f = t * t * (3 - 2 * t)
    w2, h2 = w * (1 - (1 - PREV_MINI) * f), h * (1 - (1 - PREV_MINI) * f)
    xc, xr = (PW - w) / 2, PW - 28 - w * PREV_MINI
    return xc + (xr - xc) * f, line, w2, h2, f


STACK_ROWS, STACK_ROW_H = 6, 46
SCN_Y, SCN_ROWS, SCN_ROW_H = 128, 8, 46
SCN_PV = (24, 128, 388, 218)               # Scenes tab: preview window (logical px)
TRK_X0, TRK_W, TRK_Y = 40, 360, 412        # trim slider track
LIST_X, LIST_W = 424, 392                  # Scenes tab: the scene list (one column)
TRIM_MIN = 0.3
TRACK_X0, TRACK_W = 262, 400           # slider track (logical px)
SL_PITCH, SL_PAD, SL_R = 33, 14, 11.0   # slider row height, thumb overhang, thumb radius
SS = 3                                  # supersampling for slider artwork
SEEK_X0, SEEK_W = 312, 216               # now-playing bar: seek track (centered on the panel)
VOL_X0, VOL_W = 622, 100                 # now-playing bar: volume track
ART = 60                                 # Spotify album art size in the bar
ROW_H, LIST_Y, LIST_ROWS = 40, 98, 10

WHITE = (255, 255, 255)
DARK = (24, 24, 34)
G_FILL, G_HOVER, G_CARD, G_LINE = 30, 62, 22, 46       # glass alphas
C_PILL = (255, 255, 255, 238)                          # selected segment / primary button
C_PILL_H = (255, 255, 255, 255)
C_DOM = (255, 59, 48, 244)
C_DOM_H = (255, 94, 84, 255)
C_SPOT = (30, 215, 96, 244)
C_SPOT_H = (72, 232, 126, 255)
DIM_AL = 150


def R_(x, y, w, h, s):
    return pygame.Rect(int(x * s), int(y * s), int(w * s), int(h * s))


class Panel:
    """Renders the Esc menu *content* (no background - the GL glass pass draws that)."""

    def __init__(self):
        self.fonts = {}
        self.hits = []
        self.sl_cache = {}
        self.sp_icons = {}
        self.art_surf = None
        self.icon_cache = {}

    @staticmethod
    def scale(W, H):
        return max(0.55, min(W / (PW + 60), H / (PH + 60), 1.6))

    def font(self, px, bold=False):
        k = (px, bold)
        if k not in self.fonts:
            self.fonts[k] = pygame.font.SysFont("segoeuivariable,segoeui,helveticaneue,dejavusans,arial", px, bold=bold)
        return self.fonts[k]

    def pick(self, lx, ly):
        for (x, y, w, h), key in reversed(self.hits):
            if x <= lx <= x + w and y <= ly <= y + h:
                return key
        return None

    def slider_surface(self, s, t, sc, grab, hov, def_t, w_log, h_log):
        """Glass slider (groove, glowing fill, default notch, springy glass thumb), supersampled for clean edges."""
        k = s * SS
        W, H = max(8, int(w_log * k)), max(8, int(h_log * k))
        g = pygame.Surface((W, H), pygame.SRCALPHA)
        g.fill((255, 255, 255, 0))                 # white, transparent: no dark fringe when downscaled
        pad, cy, th = SL_PAD, h_log / 2.0, 8.0
        tw = TRACK_W

        def rrect(x, y, w, h, col, r):
            rc = pygame.Rect(int(x * k), int(y * k), max(1, int(w * k)), max(1, int(h * k)))
            tmp = pygame.Surface(rc.size, pygame.SRCALPHA)
            pygame.draw.rect(tmp, col, tmp.get_rect(), border_radius=max(1, int(r * k)))
            g.blit(tmp, rc.topleft)

        def disc(cx_, cyy, r, col):
            rad = max(1, int(r * k))
            tmp = pygame.Surface((rad * 2 + 2, rad * 2 + 2), pygame.SRCALPHA)
            pygame.draw.circle(tmp, col, (rad + 1, rad + 1), rad)
            g.blit(tmp, (int(cx_ * k) - rad - 1, int(cyy * k) - rad - 1))

        # groove: dark inset with a light lower lip
        rrect(pad - 0.6, cy - th / 2 + 1.0, tw + 1.2, th, (255, 255, 255, 38), th / 2)
        rrect(pad, cy - th / 2, tw, th, (6, 6, 14, 112), th / 2)
        rrect(pad + 1, cy - th / 2 + 0.6, tw - 2, th * 0.38, (0, 0, 0, 40), th / 2)
        # default-value notch
        if abs(def_t - t) * tw > 12:
            rrect(pad + tw * def_t - 0.8, cy - th / 2 - 2.4, 1.6, th + 4.8, (255, 255, 255, 96), 0.8)
        # fill + glow
        cx = pad + tw * t
        fw = max(tw * t, th)
        rrect(pad - 2, cy - th / 2 - 3, fw + 4, th + 6, (255, 255, 255, int(20 + 14 * grab)), (th + 6) / 2)
        rrect(pad, cy - th / 2, fw, th, (206, 212, 232, 245), th / 2)
        rrect(pad, cy - th / 2, fw, th * 0.62, (255, 255, 255, 250), th / 2)
        # thumb
        r = SL_R * sc
        for dr, a, dy in ((5.0, 9, 3.4), (3.4, 13, 2.9), (2.0, 19, 2.3), (0.8, 27, 1.7)):
            disc(cx, cy + dy * (1.0 - 0.55 * grab), r + dr * (1.0 - 0.3 * grab), (0, 0, 0, a))
        hv = max(grab, 0.55 if hov else 0.0)
        if hv > 0:
            disc(cx, cy, r + 6.5, (255, 255, 255, int(14 * hv)))
            disc(cx, cy, r + 3.5, (255, 255, 255, int(26 * hv)))
        d = 1.0 - 0.11 * grab
        def tone(c):
            return (int(c[0] * d), int(c[1] * d), int(c[2] * d), 255)
        disc(cx, cy, r, tone((204, 208, 224)))
        disc(cx, cy - 0.35, r - 1.1, tone((234, 237, 247)))
        disc(cx, cy - 1.0, r - 2.7, tone((250, 251, 255)))
        return pygame.transform.smoothscale(g, (int(w_log * s), int(h_log * s)))

    def beat_tab(self, st, s, surf, text, box, draw_sliders, cfg):
        bt = st["beat"]
        box(24, 72, 792, 204, (255, 255, 255, G_CARD), 18)
        X0, XW, Y0, YB = 40, 760, 96, 246
        fx = lambda f: X0 + XW * math.log(max(f, 20.0) / 20.0) / math.log(600.0)
        for (a, b, col, lab) in ((cfg["kick_lo"], cfg["kick_hi"], (255, 100, 150), "KICK"),
                                 (cfg["snare_lo"], cfg["snare_hi"], (90, 200, 255), "SNARE")):
            xa, xb = fx(a), fx(min(b, 12000.0))
            box(xa, Y0 - 6, max(3.0, xb - xa), YB - Y0 + 10, (col[0], col[1], col[2], 46), 8)
            text(lab, (xa + xb) / 2, 80, 11, col, True, "c")
        bw = XW / 48.0
        for i, v in enumerate(bt["bars"]):
            fc = 20.0 * 600.0 ** ((i + 0.5) / 48.0)
            col = (255, 120, 170, 235) if cfg["kick_lo"] <= fc < cfg["kick_hi"] else \
                  (110, 210, 255, 235) if cfg["snare_lo"] <= fc < cfg["snare_hi"] else (255, 255, 255, 150)
            h = max(2.0, (YB - Y0) * v / 24.0)
            box(X0 + i * bw + 1, YB - h, bw - 2, h, col, 3)
        for f, lab in ((50, "50"), (100, "100"), (500, "500"), (1000, "1k"), (5000, "5k"), (10000, "10k")):
            text(lab + " Hz" if f == 100 else lab, fx(f), 252, 11, WHITE, anchor="c", al=DIM_AL)
        for k, (lab, col, fl) in enumerate((("KICK", (255, 100, 150), bt["kf"]), ("SNARE", (90, 200, 255), bt["sf"]))):
            x = 640 + k * 82
            box(x, 80, 74, 22, (col[0], col[1], col[2], int(40 + 200 * min(1.0, fl))), 11)
            text(lab, x + 37, 80, 12, WHITE if fl > 0.3 else (255, 255, 255), True, "c", vh=22, al=255 if fl > 0.3 else 170)
        draw_sliders(BEAT_KEYS, 284, (0, 3))
        text("Shaded bands = what the detector listens to \u2013 tune until KICK / SNARE flash only on the real drums.", 28, 536, 13, WHITE, al=DIM_AL)
        text("Lower sensitivity if hats or vocals trigger it, raise it if hits are missed.", 28, 558, 13, WHITE, al=DIM_AL)

    def visuals_tab(self, st, hover, press, s, surf, hits, text, box, button, icon_button, ic_x, draw_sliders, cfg):
        """Collapsible sections (they open / close with an eased height); everything scrolls under the header."""
        sy = -st.get("vscroll", 0.0)
        anim = st.get("sec_a") or {}
        lay = visuals_layout(st.get("aspect", 0.5625), anim)
        opened = set(filter(None, cfg.get("vis_open", "").split(",")))
        top_c, bot_c = HDR_Y, FTR_Y

        def clip_to(y0, y1):
            y0, y1 = max(top_c, y0), min(bot_c, y1)
            surf.set_clip(R_(0, y0, PW, max(0.0, y1 - y0), s))
            return y0, y1

        def keep(n0, y0, y1):
            kept = []
            for (hx, hy, hw, hh), hk in hits[n0:]:
                a, b = max(hy, y0), min(hy + hh, y1)
                if b > a:
                    kept.append(((hx, a, hw, b - a), hk))
            hits[n0:] = kept

        mi = st["media"]
        summary = {"style": MODES[cfg["mode"]] if 0 <= cfg["mode"] < len(MODES) else "",
                   "preview": "mirrors the screen", "motion": f"{len(VISUAL_KEYS) + len(MEDIA_KEYS)} settings",
                   "media": ((mi.get("name") or "")[:34] if mi.get("loaded") else "no media") + ("  \u00b7  on" if cfg["media_on"] else "  \u00b7  off")}
        for name, title in VIS_SECTIONS:
            L = lay[name]
            hy, ct, a = L["hy"] + sy, L["ct"] + sy, L["a"]
            # ---- header
            n0 = len(hits)
            y0, y1 = clip_to(HDR_Y, FTR_Y)
            key = ("sec", name)
            hov = hover == key
            box(24, hy, 792, SEC_HDR_H, (255, 255, 255, G_HOVER if hov else G_CARD), 14)
            ang = a * math.pi / 2                                           # chevron: points right when closed, down when open
            cx_, cy_ = 46.0, hy + SEC_HDR_H / 2
            pts = []
            for px_, py_ in ((-4.5, -6.0), (-4.5, 6.0), (6.0, 0.0)):
                pts.append((int((cx_ + px_ * math.cos(ang) - py_ * math.sin(ang)) * s), int((cy_ + px_ * math.sin(ang) + py_ * math.cos(ang)) * s)))
            pygame.draw.polygon(surf, (255, 255, 255, 230), pts)
            text(title.upper(), 66, hy, 13, WHITE, True, vh=SEC_HDR_H)
            text(summary[name], 800, hy, 12, WHITE, False, "r", vh=SEC_HDR_H, al=DIM_AL, maxw=420)
            hits.append(((24, hy, 792, SEC_HDR_H), key))
            keep(n0, y0, y1)
            if a <= 0.002:
                continue
            # ---- content, clipped to the part of the section that is open
            n0 = len(hits)
            y0, y1 = clip_to(hy + SEC_HDR_H, hy + SEC_HDR_H + L["vis"])
            if y1 <= y0:
                continue
            if name == "style":
                self.style_section(st, hover, press, s, surf, hits, text, box, cfg, ct)
            elif name == "preview":
                px = (PW - PREV_W) / 2
                box(px - 4, ct - 0, PREV_W + 8, lay["ph"] + 8, (0, 0, 0, int(110 * (1.0 - preview_geom(lay, st.get("vscroll", 0.0))[4]))), 16)             # frame; the GL pass draws the live picture on top
            elif name == "motion":
                draw_sliders(VISUAL_KEYS, ct + 4, (3, 5))
                sep = ct + 4 + len(VISUAL_KEYS) * SL_PITCH + 12
                text("MEDIA EFFECTS", 28, sep + 8, 13, WHITE, True, al=DIM_AL)
                pygame.draw.line(surf, (255, 255, 255, 40), (int(24 * s), int((sep + 30) * s)), (int(816 * s), int((sep + 30) * s)), max(1, int(s)))
                self.media_effects(st, hover, press, s, surf, hits, text, box, button, draw_sliders, cfg, sep + MSEP_H)
            else:
                self.media_section(st, hover, press, s, surf, hits, text, box, button, icon_button, ic_x, draw_sliders, cfg, ct)
            keep(n0, y0, y1)
        surf.set_clip(None)
        if lay["preview"]["a"] > 0.05:
            gx, gy, gw, gh, gf = preview_geom(lay, st.get("vscroll", 0.0))
            if gf > 0.01:                                                      # docked preview: frame + swallow clicks under it
                surf.set_clip(R_(0, HDR_Y, PW, FTR_Y - HDR_Y, s))
                box(gx - 4, gy - 4, gw + 8, gh + 8, (10, 10, 18, int(190 * gf)), 14)
                surf.set_clip(None)
                hits.append(((gx - 4, gy - 4, gw + 8, gh + 8), ("noop", "pvmini")))
        if lay["max_scroll"] > 0:                                              # scroll bar
            tr_y0, tr_h = HDR_Y + 8, FTR_Y - HDR_Y - 16
            th = max(30, tr_h * (FTR_Y - HDR_Y) / max(1.0, lay["end"] - HDR_Y))
            ty = tr_y0 + (tr_h - th) * (st.get("vscroll", 0.0) / lay["max_scroll"])
            box(PW - 12, tr_y0, 4, tr_h, (255, 255, 255, 14), 2)
            box(PW - 12, ty, 4, th, (255, 255, 255, 90), 2)

    def style_section(self, st, hover, press, s, surf, hits, text, box, cfg, y0):
        prev = st.get("style_prev")
        hsp = st.get("hov", {})
        nm = len(MODES)
        cw = (792 - 8 * (nm - 1)) / nm
        iw = int(cw - 12)
        ih = int(iw * 9 / 16)
        card_h = ih + 12 + 26
        for k, name in enumerate(MODES):
            x, y = 24 + k * (cw + 8), y0
            key = ("style", k)
            pp = press.get(key, 0.0)
            sc = (1.0 - 0.07 * pp) * (1.0 + 0.05 * hsp.get(key, 0.0))
            sel = cfg["mode"] == k
            cx, cy = x + cw / 2, y + card_h / 2
            w, h = cw * sc, card_h * sc
            fill = 58 if sel else (34 if hover == key else G_CARD)
            box(cx - w / 2, cy - h / 2, w, h, (255, 255, 255, fill), 14)
            mth = st["media"]
            if prev:
                if k == VIDEO_MODE and mth.get("loaded"):
                    img = self.thumb_surface(("vidprev", mth["tid"]), mth["thumb"], s, iw, ih, 10)
                else:
                    img = self.thumb_surface(("sty", k), prev[k], s, iw, ih, 10)
                if abs(sc - 1.0) > 0.002:
                    img = pygame.transform.smoothscale(img, (max(2, int(img.get_width() * sc)), max(2, int(img.get_height() * sc))))
                surf.blit(img, (int((cx - w / 2 + 6 * sc) * s), int((cy - h / 2 + 6 * sc) * s)))
            text(name, cx, cy + h / 2 - 25 * sc, 14, WHITE, sel, "c", vh=22 * sc)
            if sel:
                pygame.draw.rect(surf, (255, 255, 255, 235), R_(cx - w / 2, cy - h / 2, w, h, s),
                                 width=max(2, int(2 * s)), border_radius=max(2, int(14 * s)))
            hits.append(((x, y, cw, card_h), key))

    def thumb_surface(self, tid, thumb, s, w=240, h=135, rad=14):
        key = (tid, round(s, 3), w)
        cache = self.__dict__.setdefault("_thumbs", {})
        if key in cache:
            return cache[key]
        if len(cache) > 48:
            cache.clear()
        pw, ph = int(w * s), int(h * s)
        img = pygame.image.frombuffer(np.ascontiguousarray(thumb).tobytes(), (thumb.shape[1], thumb.shape[0]), "RGB")
        img = pygame.transform.smoothscale(img, (pw, ph)).convert_alpha()
        mask = pygame.Surface((pw, ph), pygame.SRCALPHA)
        pygame.draw.rect(mask, (255, 255, 255, 255), mask.get_rect(), border_radius=max(2, int(rad * s)))
        img.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MIN)
        cache[key] = img
        return img

    def scenes_tab(self, st, hover, press, s, surf, hits, text, box, button, icon_button, ic_x, ic_prev, ic_next):
        sc = st["scn"]
        box(24, 72, 792, 46, (255, 255, 255, G_CARD), 16)
        if sc["none"]:
            text("No videos or GIFs in the stack yet \u2013 drop some onto the window.", PW / 2, 72, 16, WHITE, anchor="c", vh=46, al=DIM_AL)
            text("Every scene the app finds in a clip will be listed here, so you can hide or delete the ones you don't want.", PW / 2, 150, 14, WHITE, anchor="c", al=DIM_AL)
            return
        multi = sc["n_clips"] > 1
        if multi:
            icon_button(30, 79, 36, 32, ("scn_clip", -1), ic_prev)
            icon_button(774, 79, 36, 32, ("scn_clip", 1), ic_next)
        text(sc["name"], PW / 2, 76, 16, WHITE, True, "c", maxw=620)
        bits = [f"{sc['total']} scene{'s' if sc['total'] != 1 else ''} listed"]
        if sc["hidden"]:
            bits.append(f"{sc['hidden']} hidden")
        if sc["deleted"]:
            bits.append(f"{sc['deleted']} deleted")
        if multi:
            bits.append(f"clip {sc['idx'] + 1} of {sc['n_clips']}")
        if sc["scanning"]:
            bits.append("still finding scenes\u2026")
        text("  \u00b7  ".join(bits), PW / 2, 97, 12, WHITE, anchor="c", al=DIM_AL)

        # ---- left: preview + trim
        px, py, pw_, ph_ = SCN_PV
        box(px - 4, py - 4, pw_ + 8, ph_ + 8, (0, 0, 0, 120), 16)
        tr = sc.get("trim")
        if tr is None or not sc["rows"]:
            text("Click a scene to preview it", px + pw_ / 2, py, 15, WHITE, anchor="c", vh=ph_, al=DIM_AL)
        else:
            lo, hi, a, b = tr["lo"], tr["hi"], tr["a"], tr["b"]
            ts, te = tr["s"], tr["e"]
            xo = lambda v: TRK_X0 + (v - lo) / max(1e-6, hi - lo) * TRK_W
            text(f"Scene {tr['no']}  \u00b7  looping {fmt_time(ts)} \u2013 {fmt_time(te)}", px, py + ph_ + 10, 14, WHITE, True)
            text(f"{te - ts:.1f}s", px + pw_, py + ph_ + 10, 14, WHITE, False, "r", al=DIM_AL)
            text("TRIM", px + 4, TRK_Y - 30, 12, WHITE, True, al=DIM_AL)
            text("drag the handles to shorten or lengthen the scene", px + pw_, TRK_Y - 30, 12, WHITE, False, "r", al=110)
            box(TRK_X0, TRK_Y - 3, TRK_W, 6, (255, 255, 255, 40), 3)
            box(xo(ts), TRK_Y - 3, max(2, xo(te) - xo(ts)), 6, (255, 255, 255, 170), 3)
            for v in (a, b):                                        # ticks: where the scene originally starts / ends
                box(xo(v) - 1, TRK_Y + 9, 2, 8, (255, 255, 255, 110), 1)
            hov_t = hover == ("trim",)
            for v, kk in ((ts, "s"), (te, "e")):
                grab = tr["grab"] == kk
                r_ = 9.5 + (1.5 if (grab or hov_t) else 0)
                pygame.draw.circle(surf, (0, 0, 0, 70), (int(xo(v) * s), int((TRK_Y + 1.5) * s)), int((r_ + 1.5) * s))
                pygame.draw.circle(surf, (255, 255, 255, 250), (int(xo(v) * s), int(TRK_Y * s)), int(r_ * s))
            if tr["pos"] is not None:
                pxp = xo(max(lo, min(hi, tr["pos"])))
                box(pxp - 1, TRK_Y - 12, 2, 24, (255, 90, 120, 230), 1)
            text(fmt_time(lo), TRK_X0, TRK_Y + 20, 11, WHITE, al=100)
            text(fmt_time(hi), TRK_X0 + TRK_W, TRK_Y + 20, 11, WHITE, False, "r", al=100)
            hits.append(((TRK_X0 - 16, TRK_Y - 18, TRK_W + 32, 36), ("trim",)))
            by_ = TRK_Y + 46
            button(24, by_, 110, 34, "Reset trim", ("scn_trim_reset",), False, 13, r=10) if tr["has"] else \
                box(24, by_, 110, 34, (255, 255, 255, 12), 10)
            if not tr["has"]:
                text("Reset trim", 24 + 55, by_, 13, WHITE, anchor="c", vh=34, al=70)
            button(142, by_, 170, 34, "Play on screen", ("scn_play",), False, 13, r=10)

        # ---- right: the scene list
        box(LIST_X - 6, SCN_Y - 4, LIST_W + 6, SCN_ROWS * SCN_ROW_H + 8, (255, 255, 255, 13), 16)
        if not sc["rows"]:
            text("Nothing listed \u2013 every scene was deleted.", LIST_X + LIST_W / 2, SCN_Y + 150, 14, WHITE, anchor="c", al=DIM_AL)
            text("Use \u201cRestore deleted\u201d below.", LIST_X + LIST_W / 2, SCN_Y + 172, 14, WHITE, anchor="c", al=DIM_AL)
        for k, r in enumerate(sc["rows"]):
            x = LIST_X
            y = SCN_Y + k * SCN_ROW_H
            w = LIST_W - 8
            hid = r["f"] == 1
            sel = sc.get("sel") is not None and abs(r["t"] - sc["sel"]) < 0.01
            key = ("scn_go", sc["id"], round(r["t"], 2))
            pp = max(0.0, press.get(key, 0.0))
            if sel:
                box(x - 2, y + 2, w + 2, SCN_ROW_H - 4, (255, 255, 255, 44), 12)
                pygame.draw.rect(surf, (255, 255, 255, 200), R_(x - 2, y + 2, w + 2, SCN_ROW_H - 4, s), width=max(1, int(1.5 * s)), border_radius=max(2, int(12 * s)))
            elif hover == key or pp > 0:
                box(x + 3 * pp, y + 2 + 2 * pp, w - 6 * pp, SCN_ROW_H - 4 - 4 * pp, lerp_col((255, 255, 255, 24), (38, 38, 48, 125), pp), 12)
            hits.append(((x, y, w - 108, SCN_ROW_H), key))
            if r["th"] is not None:
                surf.blit(self.thumb_surface((sc["tid"], round(r["t"], 2)), r["th"], s, 64, 36, 6), (int((x + 10) * s), int((y + 5) * s)))
            else:
                box(x + 10, y + 5, 64, 36, (255, 255, 255, 22), 6)
            if hid:
                box(x + 10, y + 5, 64, 36, (0, 0, 0, 150), 6)
            text(f"Scene {r['no']}", x + 84, y + 3, 14, WHITE, not hid, al=120 if hid else 255)
            sub = f"{fmt_time(r['t'])}  \u00b7  {r['ln']:.1f}s" + ("  \u00b7  trimmed" if r["tr"] else "") + ("  \u00b7  hidden" if hid else "")
            text(sub, x + 84, y + 24, 12, WHITE, al=DIM_AL)
            button(x + w - 98, y + 9, 60, 28, "Show" if hid else "Hide", ("scn_hide", sc["id"], round(r["t"], 2)), hid, 12, False, r=10)
            icon_button(x + w - 34, y + 9, 28, 28, ("scn_del", sc["id"], round(r["t"], 2)), ic_x)
        n_rows = sc["total"]
        if n_rows > SCN_ROWS:
            th = SCN_ROWS * SCN_ROW_H
            bh = max(24, th * SCN_ROWS / n_rows)
            by = SCN_Y + (th - bh) * (st["scn_scroll"] / max(1, n_rows - SCN_ROWS))
            box(810, by, 4, bh, (255, 255, 255, 120), 2)
        yb = SCN_Y + SCN_ROWS * SCN_ROW_H + 18
        button(24, yb, 150, 38, "Hide all", ("scn_all", "hide_all"), False, 14)
        button(182, yb, 190, 38, f"Show hidden ({sc['hidden']})", ("scn_all", "show_all"), False, 14)
        button(380, yb, 210, 38, f"Restore deleted ({sc['deleted']})", ("scn_all", "restore"), False, 14)
        text("Click a scene to preview it.  Hidden scenes stay listed but never play; deleted ones leave the list (restore any time).", 28, yb + 50, 13, WHITE, al=DIM_AL)
        if sc["few"]:
            text("Few cuts were found in this clip, so evenly spaced moments are used as well until you hide or delete a scene.", 28, yb + 72, 13, WHITE, al=DIM_AL)

    def media_section(self, st, hover, press, s, surf, hits, text, box, button, icon_button, ic_x, draw_sliders, cfg, top):
        o = top - 72                                              # the layout below was drawn for a tab starting at y = 72
        mi = st["media"]
        box(24, 72 + o, 792, 168, (255, 255, 255, G_CARD), 18)
        if mi.get("loaded"):
            surf.blit(self.thumb_surface(mi["tid"], mi["thumb"], s), (int(40 * s), int((88 + o) * s)))
        else:
            box(40, 88 + o, 240, 135, (255, 255, 255, 18), 14)
            text("No media", 160, 88 + o, 15, WHITE, True, "c", vh=135, al=DIM_AL)
        if mi.get("loaded"):
            kind = {"video": "Video", "gif": "GIF", "image": "Image"}.get(mi["kind"], mi["kind"].title())
            text(mi["name"], 304, 84 + o, 20, WHITE, True, maxw=490)
            bits = [kind, f"{mi['size'][0]}\u00d7{mi['size'][1]}"]
            if mi["kind"] != "image" and mi["duration"] > 0:
                bits.append(fmt_time(mi["duration"]))
            text("  \u00b7  ".join(bits), 304, 114 + o, 14, WHITE, al=DIM_AL)
            if mi["kind"] == "image":
                note = "Still image \u2013 stays put, pulses with the bass"
            elif mi["scan"] < 100:
                note = f"Finding scenes\u2026 {mi['scan']}%"
            else:
                note = f"{mi['scenes']} scenes found" if mi["scenes"] else "No hard cuts found \u2013 using evenly spaced moments"
            text(note, 304, 138 + o, 13, WHITE, al=DIM_AL)
            if mi["count"] > 1 and not mi.get("error"):
                text(f"{mi['count']} clips in the stack \u00b7 {mi['total_scenes']} scenes", 304, 158 + o, 13, WHITE, al=DIM_AL)
        elif mi.get("loading"):
            text("Loading\u2026", 304, 84 + o, 20, WHITE, True)
        else:
            text("Drop videos, GIFs or images", 304, 84 + o, 20, WHITE, True)
            text("It blends into the visuals and changes scene on the beat", 304, 114 + o, 14, WHITE, al=DIM_AL)
        if mi.get("error"):
            text(mi["error"], 304, 158 + o, 13, (255, 150, 140), maxw=490)
        button(304, 180 + o, 160, 38, "Add files\u2026", ("media_add",))
        button(474, 180 + o, 120, 38, "Scenes\u2026", ("tab", "scenes"))
        on = cfg["media_on"]
        button(604, 180 + o, 192, 38, "Media layer: " + ("ON" if on else "OFF"), ("toggle", "media_on"), on, 14)

        self.stack_list(st, hover, press, s, surf, hits, text, box, button, icon_button, ic_x, cfg, top + 180)

    def stack_list(self, st, hover, press, s, surf, hits, text, box, button, icon_button, ic_x, cfg, y0):
        mi = st["media"]
        items, sc = mi["items"], st["mscroll"]
        n = len(items)
        head = f"STACK  \u00b7  {n} clip{'s' if n != 1 else ''}"
        if n:
            head += f"  \u00b7  {mi['total_scenes']} scenes to pick from"
        text(head, 28, y0, 13, WHITE, True, al=DIM_AL)
        ly = y0 + 20
        box(24, ly, 792, STACK_ROWS * STACK_ROW_H + 8, (255, 255, 255, 13), 16)
        hits.append(((24, ly, 792, STACK_ROWS * STACK_ROW_H + 8), ("noop", "mlist")))          # lets the wheel scroll the list, not the page
        if n == 0:
            text("Drop videos, GIFs or images onto the window \u2013 they stack up here.", PW / 2, ly + 100, 16, WHITE, anchor="c", al=DIM_AL)
            text("Scene changes will pick from every clip in the stack.", PW / 2, ly + 128, 14, WHITE, anchor="c", al=DIM_AL)
        for r in range(STACK_ROWS):
            i = sc + r
            if i >= n:
                break
            it = items[i]
            y = ly + 4 + r * STACK_ROW_H
            is_cur = it["id"] == mi["cur_id"]
            key = ("mrow", it["id"])
            pp = max(0.0, press.get(key, 0.0))
            if is_cur or hover == key or pp > 0:
                rc = lerp_col((255, 255, 255, 52 if is_cur else 24), (38, 38, 48, 125), pp)
                box(30 + 3 * pp, y + 2 + 2 * pp, 780 - 6 * pp, STACK_ROW_H - 4 - 4 * pp, rc, 12)
            if it["ok"]:
                surf.blit(self.thumb_surface(it["tid"], it["thumb"], s, 64, 36, 6), (int(42 * s), int((y + 5) * s)))
            else:
                box(42, y + 5, 64, 36, (255, 255, 255, 22), 6)
            text(it["name"], 120, y + 3, 15, WHITE, is_cur, maxw=480)
            if it["ok"]:
                kind = {"video": "Video", "gif": "GIF", "image": "Image"}.get(it["kind"], it["kind"].title())
                bits = [kind, f"{it['size'][0]}\u00d7{it['size'][1]}"]
                if it["kind"] != "image" and it["duration"] > 0:
                    bits.append(fmt_time(it["duration"]))
                if it["kind"] == "video":
                    bits.append(f"finding scenes\u2026 {it['scan']}%" if it["scan"] < 100 else f"{it['scenes']} scenes")
                text("  \u00b7  ".join(bits), 120, y + 24, 12, WHITE, al=DIM_AL)
            else:
                text("Loading\u2026", 120, y + 24, 12, WHITE, al=DIM_AL)
            if is_cur:
                box(662, y + 12, 84, 22, (255, 255, 255, 235), 11)
                text("ON SCREEN", 704, y + 12, 11, DARK, True, "c", vh=22)
            hits.append(((30, y, 726, STACK_ROW_H), key))
            icon_button(766, y + 8, 32, 30, ("mrow_remove", it["id"]), ic_x)
        if n > STACK_ROWS:
            th = STACK_ROW_H * STACK_ROWS
            bh = max(24, th * STACK_ROWS / n)
            by = ly + 4 + (th - bh) * (sc / max(1, n - STACK_ROWS))
            box(810, by, 4, bh, (255, 255, 255, 120), 2)
        yb = ly + STACK_ROWS * STACK_ROW_H + 20
        button(24, yb, 150, 38, "Clear all", ("media_clear",), danger=True)
        text("Click a clip to jump to it.  Kick / snare cuts, beats, V and the timer pick scenes from the whole stack.", 28, yb + 48, 13, WHITE, al=DIM_AL)
        text("Drop more files any time to add them.  Video sound is ignored.", 28, yb + 68, 13, WHITE, al=DIM_AL)

    def media_effects(self, st, hover, press, s, surf, hits, text, box, button, draw_sliders, cfg, top):
        o = top - 248                                              # the layout below was drawn with the blend row at y = 248
        for k, name in enumerate(BLENDS):
            button(24 + k * 158, 248 + o, 150, 36, name, ("mblend", k), cfg["media_blend"] == k, 14)
        ao = cfg["media_auto"]
        button(660, 248 + o, 156, 36, "Beat: " + ("ON" if ao else "OFF"), ("toggle", "media_auto"), ao, 14)
        draw_sliders(MEDIA_KEYS, 296 + o, ())
        hit = cfg["media_hit"]
        button(24, 536 + o, 142, 34, "Cut on hit: " + ("ON" if hit else "OFF"), ("toggle", "media_hit"), hit, 13)
        sw = cfg["media_sway"]
        button(172, 536 + o, 112, 34, "Sway: " + ("ON" if sw else "OFF"), ("toggle", "media_sway"), sw, 13)
        sm_ = cfg["media_smooth"]
        button(290, 536 + o, 154, 34, "Smooth video: " + ("ON" if sm_ else "OFF"), ("toggle", "media_smooth"), sm_, 13)
        for k, name in enumerate(("Fade", "Blur", "Instant", "Zoom", "Random")):
            button(452 + k * 73, 536 + o, 69, 34, name, ("mstyle", k), cfg["media_style"] == k, 13)
        text("Smooth video: motion interpolation adds in-between frames.  Sway = handheld camera on hits.  V = next scene.", 28, 576 + o, 13, WHITE, al=DIM_AL)

    def render(self, st, W, H, hover, tab, scroll):
        s = self.scale(W, H)
        pw, ph = int(PW * s), int(PH * s)
        surf = pygame.Surface((pw, ph), pygame.SRCALPHA)
        hits = []
        press = st.get("press", {})
        hsp = st.get("hov", {})

        def R(x, y, w, h):
            return pygame.Rect(int(x * s), int(y * s), int(w * s), int(h * s))

        def text(t, x, y, sz=16, col=WHITE, bold=False, anchor="l", maxw=None, vh=None, al=255):
            f = self.font(max(8, int(sz * s)), bold)
            if maxw:
                mw = int(maxw * s)
                if f.size(t)[0] > mw:
                    while len(t) > 1 and f.size(t + "…")[0] > mw:
                        t = t[:-1]
                    t += "…"
            img = f.render(t, True, col)
            if al < 255:
                img.set_alpha(al)
            px = x * s
            if anchor == "c":
                px -= img.get_width() / 2
            elif anchor == "r":
                px -= img.get_width()
            py = y * s if vh is None else y * s + (vh * s - img.get_height()) / 2
            surf.blit(img, (int(px), int(py)))

        def box(x, y, w, h, col, r=10):
            pygame.draw.rect(surf, col, R(x, y, w, h), border_radius=max(2, int(r * s)))

        def line(y, a=G_LINE):
            pygame.draw.line(surf, (255, 255, 255, a), (int(24 * s), int(y * s)), (int((PW - 24) * s), int(y * s)), 1)

        def button(x, y, w, h, label, key, active=False, sz=15, bold=False, accent=C_PILL, accent_h=C_PILL_H, r=12, tcol_override=None, danger=False):
            hov = hover == key
            pp = press.get(key, 0.0)                               # <0 while the spring overshoots on release
            cp = max(0.0, pp)
            if active:
                col = accent_h if hov else accent
                tcol = tcol_override or (DARK if accent[:3] == WHITE else WHITE)
                if cp > 0:
                    col = lerp_col(col, (int(col[0] * 0.70), int(col[1] * 0.70), int(col[2] * 0.70), col[3]), cp)
            else:
                if danger:                                         # red glass (Exit)
                    col = (255, 59, 48, 120 if hov else 52)
                    tcol = WHITE if hov else (255, 168, 160)
                else:
                    col = (255, 255, 255, G_HOVER if hov else G_FILL)
                    tcol = WHITE
                if cp > 0:
                    col = lerp_col(col, (38, 38, 48, 120), cp)
            sc = (1.0 - 0.08 * pp) * (1.0 + 0.055 * hsp.get(key, 0.0))   # depress on click; springy swell on hover
            box(x + w * (1 - sc) / 2, y + h * (1 - sc) / 2 + 0.8 * cp, w * sc, h * sc, col, r)
            text(label, x + w / 2, y + 0.8 * cp, sz, tcol, bold, "c", maxw=w - 8, vh=h)
            hits.append(((x, y, w, h), key))

        def icon_button(x, y, w, h, key, draw_icon, active=False):
            hov = hover == key
            pp = press.get(key, 0.0)
            cp = max(0.0, pp)
            if active:
                col = C_PILL_H if hov else C_PILL
                if cp > 0:
                    col = lerp_col(col, (176, 176, 186, 255), cp)
            else:
                col = (255, 255, 255, G_HOVER if hov else G_FILL)
                if cp > 0:
                    col = lerp_col(col, (38, 38, 48, 120), cp)
            sc = (1.0 - 0.10 * pp) * (1.0 + 0.07 * hsp.get(key, 0.0))
            box(x + w * (1 - sc) / 2, y + h * (1 - sc) / 2 + 0.8 * cp, w * sc, h * sc, col, 10)
            draw_icon(x + w / 2, y + h / 2 + 0.8 * cp, DARK if active else WHITE)
            hits.append(((x, y, w, h), key))

        def P(pts):
            return [(int(px * s), int(py * s)) for px, py in pts]

        def aa(name, cx, cy, col, size=24.0):
            """Anti-aliased icon: drawn 4x on a transparent tile, shrunk smoothly, cached."""
            px = max(8, int(round(size * s)))
            key_ = (name, px, tuple(col))
            img = self.icon_cache.get(key_)
            if img is None:
                B = px * 4
                u = B / 24.0
                big = pygame.Surface((B, B), pygame.SRCALPHA)
                c = tuple(col)
                lw = max(2, int(2.1 * u))

                def poly(pts):
                    pygame.draw.polygon(big, c, [(x * u, y * u) for x, y in pts])

                def pl(pts, w=lw):
                    q = [(x * u, y * u) for x, y in pts]
                    pygame.draw.lines(big, c, False, q, w)
                    for p_ in q:
                        pygame.draw.circle(big, c, (int(p_[0]), int(p_[1])), w // 2)

                if name == "play":
                    poly([(8.0, 5.0), (8.0, 19.0), (19.5, 12.0)])
                elif name == "pause":
                    for x_ in (6.5, 13.5):
                        pygame.draw.rect(big, c, (x_ * u, 5 * u, 4 * u, 14 * u), border_radius=int(1.2 * u))
                elif name == "next":
                    poly([(5.5, 6.0), (5.5, 18.0), (16.0, 12.0)])
                    pygame.draw.rect(big, c, (16.8 * u, 6 * u, 2.6 * u, 12 * u), border_radius=int(0.8 * u))
                elif name == "prev":
                    poly([(18.5, 6.0), (18.5, 18.0), (8.0, 12.0)])
                    pygame.draw.rect(big, c, (4.6 * u, 6 * u, 2.6 * u, 12 * u), border_radius=int(0.8 * u))
                elif name == "shuffle":
                    pl([(3.5, 7.0), (7.5, 7.0), (14.0, 17.0), (17.0, 17.0)])
                    pl([(3.5, 17.0), (7.5, 17.0), (10.0, 13.2)])
                    pl([(12.0, 10.8), (14.0, 7.0), (17.0, 7.0)])
                    poly([(16.5, 3.8), (21.0, 7.0), (16.5, 10.2)])
                    poly([(16.5, 13.8), (21.0, 17.0), (16.5, 20.2)])
                elif name == "repeat":
                    pl([(7.0, 8.0), (5.0, 8.0), (3.5, 9.5), (3.5, 14.5), (5.0, 16.0), (13.0, 16.0)])
                    pl([(17.0, 16.0), (19.0, 16.0), (20.5, 14.5), (20.5, 9.5), (19.0, 8.0), (11.0, 8.0)])
                    poly([(10.5, 4.6), (15.0, 8.0), (10.5, 11.4)])
                    poly([(13.5, 12.6), (9.0, 16.0), (13.5, 19.4)])
                elif name == "speaker":
                    poly([(3.0, 9.5), (7.0, 9.5), (12.0, 5.0), (12.0, 19.0), (7.0, 14.5), (3.0, 14.5)])
                    for r_ in (4.5, 8.0):
                        rect = pygame.Rect((12.5 - r_ * 0.35) * u, (12 - r_) * u, r_ * 2 * u * 0.9, r_ * 2 * u)
                        pygame.draw.arc(big, c, rect, -0.85, 0.85, max(2, int(1.7 * u)))
                elif name == "mute":
                    poly([(3.0, 9.5), (7.0, 9.5), (12.0, 5.0), (12.0, 19.0), (7.0, 14.5), (3.0, 14.5)])
                    pl([(15.0, 9.0), (21.0, 15.0)])
                    pl([(21.0, 9.0), (15.0, 15.0)])
                img = pygame.transform.smoothscale(big, (px, px))
                if len(self.icon_cache) > 400:
                    self.icon_cache.clear()
                self.icon_cache[key_] = img
            surf.blit(img, (int(cx * s - px / 2), int(cy * s - px / 2)))

        def disc(cx, cy, d, col):
            """Anti-aliased filled circle."""
            px = max(6, int(round(d * s)))
            key_ = ("disc", px, tuple(col))
            img = self.icon_cache.get(key_)
            if img is None:
                big = pygame.Surface((px * 4, px * 4), pygame.SRCALPHA)
                pygame.draw.circle(big, tuple(col), (px * 2, px * 2), px * 2)
                img = pygame.transform.smoothscale(big, (px, px))
                self.icon_cache[key_] = img
            surf.blit(img, (int(cx * s - px / 2), int(cy * s - px / 2)))

        def ring(cx, cy, d, col, w=1.4):
            """Anti-aliased circle outline."""
            px = max(6, int(round(d * s)))
            wp = max(1, int(round(w * s * 4)))
            key_ = ("ring", px, wp, tuple(col))
            img = self.icon_cache.get(key_)
            if img is None:
                big = pygame.Surface((px * 4, px * 4), pygame.SRCALPHA)
                pygame.draw.circle(big, tuple(col), (px * 2, px * 2), px * 2, wp)
                img = pygame.transform.smoothscale(big, (px, px))
                self.icon_cache[key_] = img
            surf.blit(img, (int(cx * s - px / 2), int(cy * s - px / 2)))

        def round_btn(cx, cy, d, key, icon, filled=False, on=False, dim=False, isz=24.0):
            """Round transport button: white disc when filled (play/pause), soft glass disc on hover / when on."""
            hov = hover == key
            pp = press.get(key, 0.0)
            cp = max(0.0, pp)
            sc = (1.0 - 0.10 * pp) * (1.0 + 0.07 * hsp.get(key, 0.0))
            dd = d * sc
            if filled:
                disc(cx, cy, dd, (255, 255, 255, 96 if hov else 62))             # frosted glass disc with a bright rim
                ring(cx, cy, dd, (255, 255, 255, 170 if hov else 120), 1.5)
                aa(icon, cx + (0.6 if icon == "play" else 0.0), cy, WHITE, isz)
            else:
                if on:
                    disc(cx, cy, dd, (255, 255, 255, 62))
                elif hov and not dim:
                    disc(cx, cy, dd, (255, 255, 255, 40))
                aa(icon, cx, cy, (255, 255, 255, 80 if dim else 255), isz * sc)
            hits.append(((cx - d / 2, cy - d / 2, d, d), key))

        def ic_play(cx, cy, col):
            pygame.draw.polygon(surf, col, P([(cx - 6, cy - 9), (cx - 6, cy + 9), (cx + 9, cy)]))

        def ic_pause(cx, cy, col):
            pygame.draw.rect(surf, col, R(cx - 8, cy - 9, 5, 18), border_radius=2)
            pygame.draw.rect(surf, col, R(cx + 3, cy - 9, 5, 18), border_radius=2)

        def ic_next(cx, cy, col):
            pygame.draw.polygon(surf, col, P([(cx - 8, cy - 8), (cx - 8, cy + 8), (cx + 4, cy)]))
            pygame.draw.rect(surf, col, R(cx + 5, cy - 8, 3, 16))

        def ic_prev(cx, cy, col):
            pygame.draw.polygon(surf, col, P([(cx + 8, cy - 8), (cx + 8, cy + 8), (cx - 4, cy)]))
            pygame.draw.rect(surf, col, R(cx - 8, cy - 8, 3, 16))

        def ic_shuffle(cx, cy, col):
            w = max(2, int(2 * s))
            pygame.draw.lines(surf, col, False, P([(cx - 9, cy - 6), (cx - 4, cy - 6), (cx + 4, cy + 6), (cx + 6, cy + 6)]), w)
            pygame.draw.lines(surf, col, False, P([(cx - 9, cy + 6), (cx - 4, cy + 6), (cx + 4, cy - 6), (cx + 6, cy - 6)]), w)
            pygame.draw.polygon(surf, col, P([(cx + 5, cy - 10), (cx + 5, cy - 2), (cx + 11, cy - 6)]))
            pygame.draw.polygon(surf, col, P([(cx + 5, cy + 2), (cx + 5, cy + 10), (cx + 11, cy + 6)]))

        def ic_repeat(cx, cy, col):
            w = max(2, int(2 * s))
            pygame.draw.lines(surf, col, False, P([(cx - 5, cy - 6), (cx - 9, cy - 6), (cx - 10, cy - 2), (cx - 10, cy + 2), (cx - 9, cy + 6), (cx + 1, cy + 6)]), w)
            pygame.draw.lines(surf, col, False, P([(cx + 5, cy + 6), (cx + 9, cy + 6), (cx + 10, cy + 2), (cx + 10, cy - 2), (cx + 9, cy - 6), (cx - 1, cy - 6)]), w)
            pygame.draw.polygon(surf, col, P([(cx - 3, cy - 11), (cx - 3, cy - 1), (cx + 3, cy - 6)]))
            pygame.draw.polygon(surf, col, P([(cx + 3, cy + 1), (cx + 3, cy + 11), (cx - 3, cy + 6)]))

        def ic_speaker(cx, cy, col):
            pygame.draw.polygon(surf, col, P([(cx - 8, cy - 3), (cx - 4, cy - 3), (cx + 1, cy - 8), (cx + 1, cy + 8), (cx - 4, cy + 3), (cx - 8, cy + 3)]))
            w = max(2, int(1.6 * s))
            pygame.draw.arc(surf, col, R(cx - 2, cy - 6, 12, 12), -0.9, 0.9, w)
            pygame.draw.arc(surf, col, R(cx - 2, cy - 10, 20, 20), -0.9, 0.9, w)

        def ic_up(cx, cy, col):
            pygame.draw.polygon(surf, col, P([(cx - 6, cy + 3), (cx + 6, cy + 3), (cx, cy - 5)]))

        def ic_down(cx, cy, col):
            pygame.draw.polygon(surf, col, P([(cx - 6, cy - 3), (cx + 6, cy - 3), (cx, cy + 5)]))

        def ic_x(cx, cy, col):
            w = max(2, int(2 * s))
            pygame.draw.line(surf, col, (int((cx - 6) * s), int((cy - 6) * s)), (int((cx + 6) * s), int((cy + 6) * s)), w)
            pygame.draw.line(surf, col, (int((cx - 6) * s), int((cy + 6) * s)), (int((cx + 6) * s), int((cy - 6) * s)), w)

        def draw_sliders(keys, y0, dividers):
            rows = [r for key in keys for r in SLIDERS if r[0] == key]
            box(24, y0 - 4, 792, len(rows) * SL_PITCH + 8, (255, 255, 255, 13), 18)
            for gi in dividers:
                gy = int((y0 + (gi + 1) * SL_PITCH) * s)
                pygame.draw.line(surf, (255, 255, 255, 22), (int(46 * s), gy), (int(794 * s), gy), 1)
            slst = st.get("sliders", {})
            grab_key = st.get("drag_slider")
            for k, (key, label, lo, hi) in enumerate(rows):
                y = y0 + k * SL_PITCH
                v = cfg[key]
                t_real = slider_norm(key, v)
                t_disp, sc_disp = slst.get(key, (t_real, 1.0))
                hov = hover == ("slider", key)
                grab = 1.0 if grab_key == key else 0.0
                text(label, 46, y, 15, WHITE, vh=SL_PITCH, al=238)
                def_t = slider_norm(key, DEFAULTS[key])
                gsig = (round(s, 3), round(t_disp, 4), round(sc_disp, 3), grab, hov, def_t)
                hit = self.sl_cache.get(key)
                if hit and hit[0] == gsig:
                    gfx = hit[1]
                else:
                    gfx = self.slider_surface(s, t_disp, sc_disp, grab, hov, def_t, TRACK_W + 2 * SL_PAD, SL_PITCH)
                    self.sl_cache[key] = (gsig, gfx)
                surf.blit(gfx, (int((TRACK_X0 - SL_PAD) * s), int(y * s)))
                modified = abs(v - DEFAULTS[key]) > 1e-6
                button(700, y + 4.5, 88, 24, fmt_slider(key, v), ("sl_reset", key), modified, 13, False,
                       (255, 255, 255, 80), (255, 255, 255, 124), 12, WHITE)
                hits.append(((TRACK_X0 - SL_PAD, y, TRACK_W + 2 * SL_PAD, SL_PITCH), ("slider", key)))

        # ---- header
        text("Hypnosis", 30, 12, 26, WHITE, True, vh=40)
        for k, (nm, tk) in enumerate((("Queue", "queue"), ("Visuals", "visuals"),
                                      ("Scenes", "scenes"), ("Beat", "beat"), ("Settings", "settings"))):
            button(150 + k * 92, 13, 88, 36, nm, ("tab", tk), tab == tk, 13, True, r=18)
        button(676, 13, 94, 36, "Exit", ("exit",), False, 14, True, r=18, danger=True)
        icon_button(778, 13, 40, 36, ("close",), ic_x)
        line(62)

        cfg = st["cfg"]

        if tab == "queue":
            names, cur = st["names"], st["cur"]
            n = len(names)
            text(f"QUEUE  ·  {n} track{'s' if n != 1 else ''}", 28, 72, 13, WHITE, True, al=DIM_AL)
            box(24, LIST_Y - 2, 792, ROW_H * LIST_ROWS + 4, (255, 255, 255, 13), 16)
            if n == 0:
                text("Queue is empty — drop files onto the window or use “Add files”.", PW / 2, LIST_Y + 110, 16, WHITE, anchor="c", al=DIM_AL)
            for r in range(LIST_ROWS):
                i = scroll + r
                if i >= n:
                    break
                y = LIST_Y + r * ROW_H
                is_cur = i == cur
                play_key = ("row_play", i)
                pp = max(0.0, press.get(play_key, 0.0))
                if is_cur or hover == play_key or pp > 0:
                    rc = lerp_col((255, 255, 255, 52 if is_cur else 24), (38, 38, 48, 125), pp)
                    inset = 3.0 * pp
                    box(30 + inset, y + 2 + inset * 0.6, 780 - 2 * inset, ROW_H - 4 - inset * 1.2, rc, 12)
                text(f"{i + 1}", 46, y, 14, WHITE, is_cur, vh=ROW_H, al=255 if is_cur else DIM_AL)
                text(names[i], 86, y, 16, WHITE, is_cur, maxw=560, vh=ROW_H, al=255 if is_cur else 215)
                hits.append(((30, y, 640, ROW_H), play_key))
                icon_button(690, y + 7, 30, 26, ("row_up", i), ic_up)
                icon_button(726, y + 7, 30, 26, ("row_down", i), ic_down)
                icon_button(768, y + 7, 30, 26, ("row_remove", i), ic_x)
            if n > LIST_ROWS:
                th = ROW_H * LIST_ROWS
                bh = max(24, th * LIST_ROWS / n)
                by = LIST_Y + (th - bh) * (scroll / max(1, n - LIST_ROWS))
                box(810, by, 4, bh, (255, 255, 255, 120), 2)

            by = 528
            button(24, by, 180, 36, "Add files…", ("add_files",))
            button(214, by, 150, 36, "Shuffle upcoming", ("shuffle",), sz=14)
            button(374, by, 150, 36, "Clear upcoming", ("clear_up",), sz=14)
            button(534, by, 130, 36, "Clear all", ("clear_all",), sz=14)
            button(674, by, 142, 36, "Loop queue", ("toggle", "loop"), cfg["loop"], 14)

        elif tab == "visuals":
            self.visuals_tab(st, hover, press, s, surf, hits, text, box, button, icon_button, ic_x, draw_sliders, cfg)

        elif tab == "settings":
            text("MODES", 28, 68, 13, WHITE, True, al=DIM_AL)
            dom, spot = cfg["domination"], st["spotify"]
            button(24, 84, 280, 32, "DOMINATION MODE  ·  " + ("ON" if dom else "OFF"), ("toggle", "domination"),
                   dom, 14, True, C_DOM, C_DOM_H, r=14)
            button(316, 84, 280, 32, "SPOTIFY MODE  ·  " + ("ON" if spot else "OFF"), ("spotify",),
                   spot, 14, True, C_SPOT, C_SPOT_H, r=14, tcol_override=DARK)
            if st["spot_error"]:
                waiting = st["spot_error"].startswith("Waiting")
                text(st["spot_error"], 608, 84, 13, WHITE if waiting else (255, 150, 140), maxw=208, vh=32, al=DIM_AL if waiting else 255)
            else:
                text("Keys:  D  and  S", 608, 84, 13, WHITE, vh=32, al=DIM_AL)

            o = 46
            text("DISPLAY", 28, 74 + o, 13, WHITE, True, al=DIM_AL)
            kinds = [("windowed", "Windowed"), ("borderless", "Borderless fullscreen"), ("exclusive", "Exclusive fullscreen")]
            cur_kind = st["fs_kind"] if st["is_fs"] else "windowed"
            for k, (kind, label) in enumerate(kinds):
                button(24 + k * 268, 94 + o, 256, 38, label, ("display", kind), cur_kind == kind, 15)

            text("INTERFACE", 28, 142 + o, 13, WHITE, True, al=DIM_AL)
            button(24, 162 + o, 240, 34, "Show help hints: " + ("On" if cfg["help"] else "Off"), ("toggle", "help"), cfg["help"], 14)
            button(272, 162 + o, 290, 34, "Remember media on startup: " + ("On" if cfg["media_remember"] else "Off"), ("toggle", "media_remember"), cfg["media_remember"], 14)
            box(570, 162 + o, 246, 34, (255, 255, 255, G_CARD), 12)
            text("Domination phrases: add phrases.txt", 693, 162 + o, 13, WHITE, anchor="c", maxw=232, vh=34, al=DIM_AL)

            text("AUDIO", 28, 206 + o, 13, WHITE, True, al=DIM_AL)
            draw_sliders(SETTINGS_KEYS, 228 + o, (0,))

            text("SHORTCUTS", 28, 267 + o, 13, WHITE, True, al=DIM_AL)
            box(24, 287 + o, 792, 104, (255, 255, 255, 13), 16)
            keys = [("Space", "Play / pause"), ("Esc", "Open / close menu"), ("F / F11", "Fullscreen"),
                    ("M / Tab", "Next visual style"), ("D", "Domination mode"), ("S", "Spotify mode"),
                    ("V", "Next scene"), ("N / P", "Next / previous track"), ("Left/Right", "Seek 5 seconds"),
                    ("Up/Down", "Volume"), ("H", "Help hints"), ("Q", "Quit")]
            for i, (kk, dd) in enumerate(keys):
                col, row = divmod(i, 4)
                x, y = 38 + col * 262, 295 + o + row * 24
                box(x, y + 2, 78, 22, (255, 255, 255, 40), 8)
                text(kk, x + 39, y + 2, 12, WHITE, True, "c", vh=22)
                text(dd, x + 88, y + 2, 13, WHITE, vh=22, al=DIM_AL + 20, maxw=160)

            text("PRESETS", 28, 401 + o, 13, WHITE, True, al=DIM_AL)
            plist, psel = st["presets"], st["preset_sel"]
            pop = st["preset_open"]
            button(24, 421 + o, 300, 38, ("\u25BE  " + psel) if psel else "No presets saved yet", ("preset_dd",), pop, 14)
            button(332, 421 + o, 140, 38, "Save as new", ("preset_save",), False, 14)
            button(480, 421 + o, 110, 38, "Update", ("preset_update",), False, 14)
            button(598, 421 + o, 150, 38, "Click to confirm" if st["preset_del"] else "Delete", ("preset_delete",), False, 14, danger=True)
            text("Saves every slider, toggle and style (not your clips or queue) to presets.json.", 28, 465 + o, 13, WHITE, al=DIM_AL)

            armed = st["reset_armed"]
            button(24, 499 + o, 240, 34, "Click again to confirm" if armed else "Reset all settings", ("reset_all",),
                   False, 14, danger=True)
            text("Restores every slider and toggle to its default. Your clips and queue stay.", 280, 499 + o, 13, WHITE, vh=34, al=DIM_AL)

            if pop and plist:                                   # dropdown list: newest first, drawn last so it sits on top
                shown = list(reversed(plist))[:7]
                rh = 32
                bh = len(shown) * rh + 12
                by = 421 + o - bh - 6
                hits.append(((24, by, 300, bh + 6), ("noop",)))
                box(24, by, 300, bh, (32, 34, 52, 245), 14)
                for i, nm in enumerate(shown):
                    ry = by + 6 + i * rh
                    key = ("preset_pick", nm)
                    hov_ = hover == key
                    if hov_ or nm == psel:
                        box(30, ry, 288, rh - 2, (255, 255, 255, 70 if hov_ else 36), 10)
                    text(nm + ("   \u2713" if nm == psel else ""), 44, ry, 14, WHITE, vh=rh - 2)
                    hits.append(((30, ry, 288, rh - 2), key))

        elif tab == "scenes":
            self.scenes_tab(st, hover, press, s, surf, hits, text, box, button, icon_button, ic_x, ic_prev, ic_next)
        else:
            self.beat_tab(st, s, surf, text, box, draw_sliders, cfg)

        # ---- now-playing bar (a Spotify-style player in the same glass): art + song | shuffle prev play next repeat + seek | volume
        line(FTR_Y)
        spot = st["spotify"]
        names, cur = st["names"], st["cur"]
        by0 = FTR_Y + 8
        x0 = 24
        if spot:
            art = st.get("art")
            ak = (st["art_ver"], int(ART * s))
            if self.art_surf is None or self.art_surf[0] != ak:
                if art is not None:
                    sz = int(ART * s)
                    im = art.resize((sz, sz))
                    sf_ = pygame.image.frombuffer(im.tobytes(), im.size, "RGB").convert_alpha()
                    m_ = pygame.Surface((sz, sz), pygame.SRCALPHA)
                    pygame.draw.rect(m_, (255, 255, 255, 255), m_.get_rect(), border_radius=max(2, int(8 * s)))
                    sf_.blit(m_, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
                    self.art_surf = (ak, sf_)
                else:
                    self.art_surf = (ak, None)
            if self.art_surf[1] is not None:
                surf.blit(self.art_surf[1], (int(x0 * s), int(by0 * s)))
            else:
                box(x0, by0, ART, ART, (30, 215, 96, 46), 8)
                text("\u266A", x0 + ART / 2, by0, 28, (30, 215, 96), True, "c", vh=ART)
            t = st["spot_title"]
            if (not t or t.lower().startswith("spotify")) and st["spot_running"] and st["spot_track"]:
                t = st["spot_track"]                                   # paused: keep showing the last song instead of "paused"
            if st["spot_error"]:
                wt = st["spot_error"].startswith("Waiting")
                song, sub, subcol = ("Spotify is paused" if st["spot_running"] else "Spotify isn't running"), st["spot_error"], ((200, 200, 210) if wt else (255, 150, 140))
            elif t and not t.lower().startswith("spotify"):
                a_, _, so = t.partition(" - ")
                song, sub, subcol = (so, a_, (30, 215, 96)) if so else (t, "SPOTIFY MODE", (30, 215, 96))
            else:
                song, sub, subcol = ("Spotify is paused" if st["spot_running"] else "Spotify isn't running"), "SPOTIFY MODE  \u00b7  hearing Spotify only", (30, 215, 96)
        else:
            box(x0, by0, ART, ART, (255, 255, 255, 26), 8)
            text("\u266A", x0 + ART / 2, by0, 28, WHITE, True, "c", vh=ART, al=150)
            song = names[cur] if 0 <= cur < len(names) else "Nothing playing"
            if st["loading"]:
                song += "   (loading\u2026)"
            sub, subcol = (f"Queue  \u00b7  track {cur + 1} of {len(names)}" if 0 <= cur < len(names) else "Drop files onto the window"), WHITE
        text(song, x0 + ART + 12, by0 + 8, 17, WHITE, True, maxw=196)
        text(sub, x0 + ART + 12, by0 + 32, 13, subcol, False, maxw=196, al=235 if subcol != WHITE else DIM_AL)

        cy_ = by0 + 19
        mid = PW / 2
        round_btn(mid - 108, cy_, 34, ("noop", "sh") if spot else ("shuffle",), "shuffle", dim=spot, isz=26)
        round_btn(mid - 60, cy_, 34, ("prev",), "prev", isz=26)
        round_btn(mid, cy_, 38, ("playpause",), "pause" if st["playing"] else "play", filled=True, isz=24)
        round_btn(mid + 60, cy_, 34, ("next",), "next", isz=26)
        round_btn(mid + 108, cy_, 34, ("noop", "rp") if spot else ("toggle", "loop"), "repeat", on=(not spot and cfg["loop"]), dim=spot, isz=26)
        sy_ = by0 + 50
        if spot:
            pos, total = (st["sp_pos"] or 0.0), (st["sp_dur"] or 0.0)
        else:
            pos, total = st["pos"], st["total"]
        can_seek = total > 0
        box(SEEK_X0, sy_, SEEK_W, 5, (255, 255, 255, 55 if can_seek else 40), 3)
        if can_seek:
            frac = max(0.0, min(1.0, pos / total))
            if frac > 0:
                box(SEEK_X0, sy_, SEEK_W * frac, 5, (255, 255, 255, 240), 3)
            disc(SEEK_X0 + SEEK_W * frac, sy_ + 2.5, 11, WHITE) if (frac > 0 or spot) else None
            hits.append(((SEEK_X0 - 4, sy_ - 9, SEEK_W + 8, 24), ("seek",)))
            text(fmt_time(pos), SEEK_X0 - 10, sy_ - 8, 12, WHITE, anchor="r", vh=21, al=DIM_AL + 40)
            text(fmt_time(total), SEEK_X0 + SEEK_W + 10, sy_ - 8, 12, WHITE, vh=21, al=DIM_AL + 40)
        elif spot:
            text("-:--", SEEK_X0 - 10, sy_ - 8, 12, WHITE, anchor="r", vh=21, al=90)
            text("-:--", SEEK_X0 + SEEK_W + 10, sy_ - 8, 12, WHITE, vh=21, al=90)
        else:
            text(fmt_time(pos), SEEK_X0 - 10, sy_ - 8, 12, WHITE, anchor="r", vh=21, al=DIM_AL + 40)
            text(fmt_time(total), SEEK_X0 + SEEK_W + 10, sy_ - 8, 12, WHITE, vh=21, al=DIM_AL + 40)

        sv = st.get("spot_vol")
        vol = (sv if sv is not None else 1.0) if spot else cfg["volume"]
        vy = by0 + 30
        aa("mute" if vol < 0.005 else "speaker", VOL_X0 - 20, vy, WHITE, 22)
        hv_ = hover == ("vol",)
        box(VOL_X0, vy - 2.5, VOL_W, 5, (255, 255, 255, 55), 3)
        box(VOL_X0, vy - 2.5, max(2.0, VOL_W * vol), 5, (255, 255, 255, 240), 3)
        disc(VOL_X0 + VOL_W * vol, vy, 15 if (hv_ or st.get("drag_vol")) else 11, WHITE)
        text(f"{vol * 100:.0f}%", VOL_X0 + VOL_W + 10, vy - 10, 12, WHITE, vh=20, al=DIM_AL + 40)
        hits.append(((VOL_X0 - 34, vy - 16, VOL_W + 32, 32), ("vol",)))

        # Spotify mode switch: a glass disc, grey when off / green when on, with a colourful wave between the two
        tr = st.get("sp_tr")
        skey = ("spotify",)
        sc_ = 1.0 + 0.09 * hsp.get(skey, 0.0) - 0.08 * max(0.0, press.get(skey, 0.0))
        spx = max(16, int(round(40 * s * sc_)))
        ik = (spx, tr[0] if tr else spot, int(tr[1] * 28) if tr else -1)
        img = self.sp_icons.get(ik)
        if img is None:
            if len(self.sp_icons) > 160:
                self.sp_icons.clear()
            raw = spot_icon_pixels(spx, tr[0] if tr else spot, min(1.0, tr[1]) if tr else None)
            img = pygame.image.frombuffer(raw, (spx, spx), "RGBA").convert_alpha()
            self.sp_icons[ik] = img
        scx, scy = 798, vy
        surf.blit(img, (int(scx * s - spx / 2), int(scy * s - spx / 2)))
        hits.append(((scx - 22, scy - 22, 44, 44), skey))

        self.hits = hits
        return surf, s


# --------------------------------------------------------------------------- media layer (video / GIF / image)
VIDEO_EXT = {".mp4", ".mkv", ".webm", ".mov", ".avi", ".m4v", ".wmv", ".flv", ".mpg", ".mpeg", ".ts", ".3gp"}
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff", ".jfif"}
GIF_EXT = {".gif"}
MEDIA_EXT = VIDEO_EXT | IMAGE_EXT | GIF_EXT


def is_media_path(p):
    return os.path.isfile(p) and os.path.splitext(p)[1].lower() in MEDIA_EXT


def pick_media_dialog(out):
    """Runs in a thread; pushes the chosen media path into `out`."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        exts = " ".join("*" + e for e in sorted(MEDIA_EXT))
        f = filedialog.askopenfilenames(title="Add videos, GIFs or images to the stack",
                                        filetypes=[("Video, GIF, image", exts), ("All files", "*.*")])
        root.destroy()
        out.extend(f)
    except Exception:
        pass


def _fit_size(w, h, limit):
    sc = min(1.0, float(limit) / max(w, h, 1))
    return max(2, int(w * sc) // 2 * 2), max(2, int(h * sc) // 2 * 2)


def _thumb(frame, size=(240, 135)):
    """Letter-boxed RGB thumbnail (numpy) for the menu."""
    from PIL import Image
    im = Image.fromarray(frame)
    im.thumbnail(size, Image.LANCZOS)
    canvas = Image.new("RGB", size, (0, 0, 0))
    canvas.paste(im, ((size[0] - im.size[0]) // 2, (size[1] - im.size[1]) // 2))
    return np.asarray(canvas, dtype=np.uint8)


def _hist(frame_small):
    """512-bin colour histogram (3 bits per channel), normalised."""
    f = frame_small.reshape(-1, 3) >> 5
    idx = (f[:, 0].astype(np.int32) << 6) | (f[:, 1].astype(np.int32) << 3) | f[:, 2].astype(np.int32)
    h = np.bincount(idx, minlength=512).astype(np.float32)
    return h / max(1.0, float(h.sum()))


def _hist_dist(a, b):
    """Bhattacharyya distance: 0 = identical colour make-up, 1 = nothing in common."""
    return float(math.sqrt(max(0.0, 1.0 - float(np.sum(np.sqrt(a * b))))))


CUT_THRESHOLD = 0.40
MIN_SCENE_S = 1.2
SCENE_THUMB = (128, 72)


class StillHead:
    ready = True

    def __init__(self, frame):
        self.frame, self.serial = frame, 1

    def play_from(self, t, paused=False):
        pass

    def resume(self):
        pass

    def set_paused(self, p):
        pass

    def get(self):
        return self.serial, self.frame

    def stop(self):
        pass


class StillSource:
    kind = "image"

    def __init__(self, path):
        from PIL import Image, ImageOps
        im = Image.open(path)
        self.orig = im.size
        im = ImageOps.exif_transpose(im).convert("RGB")
        tw, th = _fit_size(im.size[0], im.size[1], 2048)
        if (tw, th) != im.size:
            im = im.resize((tw, th), Image.LANCZOS)
        self.w, self.h = im.size
        self.frame = np.ascontiguousarray(np.asarray(im, dtype=np.uint8))
        self.duration, self.scenes, self.scan, self.nframes = 0.0, [0.0], 1.0, 1
        self.thumb = _thumb(self.frame)

    def make_head(self):
        return StillHead(self.frame)

    def close(self):
        pass


class MemoryHead:
    """Plays an in-memory frame list (GIF) against the wall clock."""
    ready = True

    def __init__(self, src):
        self.src = src
        self.pos0, self.t0 = 0.0, time.perf_counter()
        self.paused, self._pause_at = False, 0.0
        self.serial, self._last = 0, -1
        self._acc, self.rate = 0.0, 1.0

    def play_from(self, t, paused=False):
        self.pos0, self.t0 = float(t), time.perf_counter()
        self.paused, self._pause_at = bool(paused), self.t0
        self._acc = 0.0
        self._last = -1

    def _elapsed(self):
        if self.paused:
            return self._acc
        return self._acc + (time.perf_counter() - self.t0) * self.rate

    def set_rate(self, r):
        self._acc = self._elapsed()
        self.t0 = time.perf_counter()
        self.rate = max(0.05, float(r))

    def resume(self):
        self.set_paused(False)

    def set_paused(self, p):
        if p and not self.paused:
            self._acc = self._elapsed()
            self.paused, self._pause_at = True, time.perf_counter()
        elif not p and self.paused:
            self.t0 = time.perf_counter()
            self.paused = False

    def _pos(self):
        return self.pos0 + self._elapsed()

    def get(self):
        src = self.src
        t = self._pos() % max(src.duration, 1e-3)
        i = min(len(src.frames) - 1, int(np.searchsorted(src.cum, t, side="right")))
        if i != self._last:
            self._last = i
            self.serial += 1
        return self.serial, src.frames[i]

    def stop(self):
        pass


class GifSource:
    """Animated GIF / WebP / APNG: all frames live in memory; scenes come from colour jumps between frames."""
    kind = "gif"

    def __init__(self, path):
        from PIL import Image
        im = Image.open(path)
        n = max(1, getattr(im, "n_frames", 1))
        self.orig = im.size
        tw, th = _fit_size(im.size[0], im.size[1], 960)
        while n * tw * th * 3 > 256 * 1024 * 1024 and tw > 96:      # keep memory sane for huge GIFs
            tw, th = max(2, int(tw * 0.85) // 2 * 2), max(2, int(th * 0.85) // 2 * 2)
        frames, durs = [], []
        for i in range(n):
            im.seek(i)
            fr = im.convert("RGB")
            if fr.size != (tw, th):
                fr = fr.resize((tw, th), Image.BILINEAR)
            frames.append(np.ascontiguousarray(np.asarray(fr, dtype=np.uint8)))
            d = im.info.get("duration", 100) or 100
            durs.append(100 if d <= 10 else d)                       # browsers treat <=10 ms as 100 ms
        self.frames = frames
        self.w, self.h = tw, th
        self.cum = np.cumsum(np.asarray(durs, dtype=np.float64) / 1000.0)
        self.duration = float(self.cum[-1])
        self.nframes = n
        # scene starts: colour-histogram jumps between consecutive frames
        scenes, last_t, prev = [0.0], 0.0, _hist(frames[0][::8, ::8])
        for i in range(1, n):
            h = _hist(frames[i][::8, ::8])
            t = float(self.cum[i - 1])
            if _hist_dist(prev, h) > CUT_THRESHOLD and t - last_t >= MIN_SCENE_S:
                scenes.append(t)
                last_t = t
            prev = h
        self.scenes, self.scan = scenes, 1.0
        self.thumb = _thumb(frames[min(n - 1, n // 5)])

    def scene_thumb(self, t):
        key = round(t, 2)
        c = self.__dict__.setdefault("_sthumbs", {})
        if key not in c:
            i = min(len(self.frames) - 1, int(np.searchsorted(self.cum, t, side="right")))
            c[key] = _thumb(self.frames[i], SCENE_THUMB)
        return c[key]

    sthumb_n = 0

    def make_head(self):
        return MemoryHead(self)

    def close(self):
        self.frames = []


class VideoHead:
    """Decodes one playback position of a video in a background thread, paced by the wall clock."""

    def __init__(self, src):
        self.src = src
        self.lock = threading.Lock()
        self.frame, self.serial, self.ready = None, 0, False
        self.extra = None
        self.interp = False
        self._cmd, self._run = None, True
        self.paused, self._pause_at = True, time.perf_counter()
        self._wall0, self._pos0 = time.perf_counter(), 0.0
        self._acc, self.rate = 0.0, 1.0
        self._clk = threading.RLock()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def play_from(self, t, paused=False):
        with self.lock:
            self.frame, self.ready = None, False
        self._cmd = (float(t), bool(paused))

    def resume(self):
        self.set_paused(False)

    def _elapsed(self):
        with self._clk:
            if self.paused:
                return self._acc
            return self._acc + (time.perf_counter() - self._wall0) * self.rate

    def set_paused(self, p):
        if p and not self.paused:
            self._acc = self._elapsed()
            self.paused, self._pause_at = True, time.perf_counter()
        elif not p and self.paused:
            self._wall0 = time.perf_counter()
            self.paused = False

    def set_rate(self, r):
        with self._clk:
            self._acc = self._elapsed()
            self._wall0 = time.perf_counter()
            self.rate = max(0.05, float(r))

    def pos_frames(self):
        """Where playback is right now, in (fractional) frames - drives the in-between frames so slow motion stays continuous."""
        return (self._pos0 + self._elapsed()) * self.src.fps

    def get(self):
        with self.lock:
            if self.frame is None:
                return None
            return self.serial, self.frame, self.extra

    def stop(self):
        self._run = False

    def _loop(self):
        import cv2
        src = self.src
        cap = cv2.VideoCapture(src.path)
        fps, tw, th = src.fps, src.tw, src.th
        cur = 0
        active = False
        skipped = 0
        prev_g = None
        dis = None
        try:
            while self._run:
                cmd, self._cmd = self._cmd, None
                if cmd:
                    t, paused = cmd
                    cur = max(0, int(t * fps))
                    cap.set(cv2.CAP_PROP_POS_FRAMES, cur)
                    self._pos0 = cur / fps
                    now = time.perf_counter()
                    self._acc = 0.0
                    self._wall0, self._pause_at, self.paused = now, now, paused
                    active, first = True, True
                if not active:
                    time.sleep(0.01)
                    continue
                if self.paused and not first:
                    time.sleep(0.005)
                    continue
                if not first:
                    want = int((self._pos0 + self._elapsed()) * fps) + (1 if self.interp else 0)   # interpolating: stay one frame ahead
                    if cur > want:
                        time.sleep(0.002)
                        continue
                    skipped = 0
                    for _ in range(max(0, want - cur - 2)):          # running late: skip frames
                        if not cap.grab():
                            break
                        cur += 1
                        skipped += 1
                ok, fr = cap.read()
                cur += 1
                fidx = cur - 1
                if not ok:                                           # end of file: loop
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    cur, self._pos0 = 0, 0.0
                    self._acc, self._wall0 = 0.0, time.perf_counter()
                    first = True
                    continue
                if fr.shape[1] != tw or fr.shape[0] != th:
                    fr = cv2.resize(fr, (tw, th), interpolation=cv2.INTER_AREA)
                rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
                extra = None
                if self.interp:                                      # motion field between this frame and the last one (small, cheap)
                    try:
                        if dis is None:
                            dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
                        sw = 192
                        sh = max(8, int(round(sw * th / float(tw))))
                        g = cv2.cvtColor(cv2.resize(fr, (sw, sh), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
                        cut = first or skipped > 2 or prev_g is None or prev_g.shape != g.shape
                        if not cut and float(cv2.absdiff(g, prev_g).mean()) > 28.0:
                            cut = True                               # hard cut: never smear across it
                        if cut:
                            extra = (None, True, fidx)
                        else:
                            fl = dis.calc(prev_g, g, None)
                            fl = fl / np.array([sw, sh], dtype=np.float32)
                            extra = (np.ascontiguousarray(fl.astype(np.float16)), False, fidx)
                        prev_g = g
                    except Exception:
                        extra, prev_g = (None, True, fidx), None
                else:
                    prev_g = None
                skipped = 0
                with self.lock:
                    self.frame = rgb
                    self.extra = extra
                    self.serial += 1
                    self.ready = True
                first = False
        finally:
            cap.release()


class VideoSource:
    kind = "video"

    def __init__(self, path):
        try:
            import cv2
        except ImportError:
            raise RuntimeError("Videos need OpenCV:  pip install opencv-python")
        self.path = path
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError("Can't open this video file")
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if not (1.0 < fps < 241.0):
            fps = 30.0
        if w <= 0 or h <= 0:
            cap.release()
            raise RuntimeError("Can't read this video's size")
        self.fps, self.orig = fps, (w, h)
        self.nframes = n
        self.duration = n / fps if n > 0 else 0.0
        self.tw, self.th = _fit_size(w, h, 1280)
        self.w, self.h = self.tw, self.th
        if n > 10:                                                   # thumbnail from ~12% in (skips black intros)
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(n * 0.12))
        ok, fr = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, fr = cap.read()
        cap.release()
        if not ok:
            raise RuntimeError("Can't decode frames from this video")
        self.thumb = _thumb(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
        self.scenes, self.scan = [0.0], 0.0
        self._stop = False
        self._sl, self._sthumbs, self._sq, self._sworker, self.sthumb_n = threading.Lock(), {}, [], None, 0
        self._scan_thread = threading.Thread(target=self._scan, daemon=True)
        self._scan_thread.start()

    def scene_thumb(self, t):
        """Thumbnail of the frame at scene start t, or None until the background worker has fetched it."""
        key = round(t, 2)
        with self._sl:
            th = self._sthumbs.get(key)
            if th is not None:
                return th
            if key not in self._sq:
                self._sq.append(key)
            if self._sworker is None:
                self._sworker = threading.Thread(target=self._thumb_work, daemon=True)
                self._sworker.start()
        return None

    def _thumb_work(self):
        try:
            import cv2
            cap = cv2.VideoCapture(self.path)
            while not self._stop:
                with self._sl:
                    if not self._sq:
                        self._sworker = None
                        break
                    key = self._sq.pop()                          # newest request first = what is on screen now
                cap.set(cv2.CAP_PROP_POS_FRAMES, int((key + 0.15) * self.fps))
                ok, fr = cap.read()
                if not ok:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(key * self.fps))
                    ok, fr = cap.read()
                th = _thumb(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB), SCENE_THUMB) if ok else np.zeros((SCENE_THUMB[1], SCENE_THUMB[0], 3), np.uint8)
                with self._sl:
                    self._sthumbs[key] = th
                    self.sthumb_n += 1
            cap.release()
        except Exception:
            with self._sl:
                self._sworker = None

    def _quick_scan(self):
        """Fast path: only decode the KEYFRAMES (encoders put them at scene cuts) - a 40 minute video takes seconds instead of
        half a minute. Needs PyAV; returns False when it isn't available or the video has too few keyframes to be useful."""
        try:
            import av
        except Exception:
            return False
        try:
            c = av.open(self.path)
            st = c.streams.video[0]
            st.thread_type = "AUTO"
            st.codec_context.skip_frame = "NONKEY"
            prev, last_t, nk = None, 0.0, 0
            for fr in c.decode(st):
                if self._stop:
                    break
                if fr.time is None:
                    continue
                t = float(fr.time)
                h = _hist(fr.reformat(width=48, height=27, format="rgb24").to_ndarray())
                nk += 1
                if prev is not None and _hist_dist(prev, h) > CUT_THRESHOLD and t - last_t >= MIN_SCENE_S:
                    self.scenes.append(t)
                    last_t = t
                prev = h
                self.scan = min(0.999, t / max(1.0, self.duration))
            c.close()
            return nk >= max(6, int(self.duration / 15.0))        # at least one keyframe per ~15 s, else too coarse
        except Exception:
            return False

    def _scan(self):
        """Find scene starts (hard colour changes). The video is cut into one chunk per CPU core and every chunk is
        decoded in parallel, so the wait shrinks roughly with the number of cores. Scenes appear as they are found."""
        try:
            if self.duration <= 0 or self.duration > 6 * 3600:        # unknown / absurdly long: skip, use even spacing
                self.scan = 1.0
                return
            if self._quick_scan():
                return
            self.scenes, self.scan = [0.0], 0.0                    # quick scan unusable: do the exact chunked scan instead
            n = max(1, self.nframes)
            workers = max(1, min(os.cpu_count() or 1, 32, n // 240))  # tiny clips: not worth splitting
            step = max(1, int(round(self.fps / 4.0)))                 # look 4 times a second
            bounds = [int(n * k / workers) // step * step for k in range(workers + 1)]
            bounds[-1] = n
            cands, done = [set() for _ in range(workers)], [0] * workers
            lock = threading.Lock()

            def merge():                                             # candidate cuts of all chunks -> scene list (min spacing rule)
                allc = sorted(t for cs in cands for t in cs)
                out, last = [0.0], 0.0
                for t in allc:
                    if t - last >= MIN_SCENE_S:
                        out.append(t)
                        last = t
                self.scenes = out

            def work(wi):
                import cv2
                try:
                    if sys.platform == "win32":                       # keep the render thread responsive
                        import ctypes
                        ctypes.windll.kernel32.SetThreadPriority(ctypes.windll.kernel32.GetCurrentThread(), -1)
                except Exception:
                    pass
                try:
                    f0, f1 = bounds[wi], bounds[wi + 1]
                    cap = cv2.VideoCapture(self.path)
                    start = max(0, f0 - step)                         # one extra sample before the chunk so a cut on the border is seen
                    if start > 0:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
                    idx, prev = start, None
                    while not self._stop and idx < f1:
                        if not cap.grab():
                            break
                        if (idx - start) % step == 0:
                            ok, fr = cap.retrieve()
                            if ok:
                                h = _hist(cv2.resize(fr, (48, 27), interpolation=cv2.INTER_AREA))
                                t = idx / self.fps
                                if prev is not None and idx >= f0 and _hist_dist(prev, h) > CUT_THRESHOLD:
                                    with lock:
                                        cands[wi].add(t)
                                        merge()
                                prev = h
                            done[wi] = max(0, idx - f0)
                            self.scan = min(0.999, sum(done) / float(n))
                        idx += 1
                    cap.release()
                except Exception:
                    pass
                finally:
                    done[wi] = f1 - f0

            ts = [threading.Thread(target=work, args=(i,), daemon=True) for i in range(workers)]
            for t in ts:
                t.start()
            for t in ts:
                t.join()
            with lock:
                merge()
        except Exception:
            pass
        finally:
            self.scan = 1.0

    def make_head(self):
        return VideoHead(self)

    def close(self):
        self._stop = True


def open_source(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in VIDEO_EXT:
        return VideoSource(path)
    from PIL import Image
    with Image.open(path) as im:
        n = getattr(im, "n_frames", 1)
    return GifSource(path) if n > 1 else StillSource(path)


class MediaLayer:
    """A stack of videos / GIFs / images blended into the visualizer.
    Scene changes pick from every clip in the stack. Two playback heads are used: the current one, and a second one
    parked (paused) on the next scene so a beat / kick / snare can release it instantly into a transition."""
    FADE_S = 0.5
    STYLE_S = (0.5, 0.65, 0.001, 0.42)  # fade, blur, instant, zoom   (style 4 = random)
    style = 0

    def __init__(self, ctx):
        self.ctx = ctx
        self.items = []                  # dicts: id, path, src, status (loading|ok), 
        self._next_id = 1
        self._pending = []
        self._plock = threading.Lock()
        self.cur = self.nxt = None
        self.cur_it = self.nxt_it = None
        self.tex = [None, None]
        self.ptex = [None, None]         # previous frame of each head (frame interpolation)
        self.ftex = [None, None]         # motion field of each head
        self.itp = [dict(arr=0.0, gap=1 / 30.0, valid=False, al=1.0), dict(arr=0.0, gap=1 / 30.0, valid=False, al=1.0)]
        self.interp = False
        self.tex_serial = [-1, -1]
        self.tex_size = [None, None]
        self.dummy = ctx.texture((1, 1), 3, bytes([0, 0, 0]), alignment=1)
        self.dummy_flow = ctx.texture((1, 1), 2, np.zeros((1, 1, 2), np.float16).tobytes(), dtype="f2")
        self.state = "idle"              # idle | static | prep | armed | fading
        self.fade = 0.0
        self.vis = 0.0
        self.last_change = 0.0
        self.prep_since = 0.0
        self.cur_start = self.nxt_start = 0.0
        self.recent = []                 # (item id, scene time)
        self.undo = []                   # (item id, scene time, previous flag) for the Z hotkey
        self._req = False
        self._focus = None
        self.error = None
        self._en = True
        self.prefs = load_scene_prefs()      # hidden / deleted scenes per video path (survives restarts)
        self.pver = 0                        # bumps whenever a hide / delete changes
        self._force_t = None
        self.speed = 1.0
        self.fade_s = self.FADE_S
        self.cur_style = 0

    # ---- the stack
    def paths(self):
        return [it["path"] for it in self.items]

    def _ok(self):
        return [it for it in self.items if it["src"] is not None]

    def add(self, paths):
        have = set(self.paths())
        for path in paths:
            if path in have:
                continue
            have.add(path)
            it = dict(id=self._next_id, path=path, src=None, status="loading")
            self._next_id += 1
            self.items.append(it)
            threading.Thread(target=self._work, args=(it["id"], path), daemon=True).start()

    def _work(self, iid, path):
        try:
            src, err = open_source(path), None
        except Exception as e:
            src, err = None, (str(e) or e.__class__.__name__)
        with self._plock:
            self._pending.append((iid, src, err))

    def poll(self, now):
        """Main thread: install freshly opened clips. Returns the name of the last one installed (or None)."""
        with self._plock:
            pend, self._pending = self._pending, []
        got = None
        for iid, src, err in pend:
            it = next((x for x in self.items if x["id"] == iid), None)
            if it is None:                                   # removed while loading
                if src is not None:
                    src.close()
                continue
            if src is None:
                self.error = f"{os.path.basename(it['path'])}: {err}"
                self.items.remove(it)
                continue
            it["src"], it["status"] = src, "ok"
            got = os.path.basename(it["path"])
        if got is not None:
            self._refresh(now)
        return got

    def remove(self, iid, now):
        it = next((x for x in self.items if x["id"] == iid), None)
        if it is None:
            return
        self.items.remove(it)
        if it["src"] is not None:
            it["src"].close()
        if it is self.cur_it:
            self._drop_heads()
        elif it is self.nxt_it:                              # only the parked scene went away: just re-park
            if self.nxt is not None:
                self.nxt.stop()
            self.nxt = self.nxt_it = None
            self.tex_serial[1] = -1
            self.state, self.fade = "static", 0.0
        self._refresh(now)

    def clear(self):
        self._drop_heads()
        for it in self.items:
            if it["src"] is not None:
                it["src"].close()
        self.items = []
        with self._plock:
            self._pending = []                               # late loaders find no item and close their source

    def _drop_heads(self):
        for h in (self.cur, self.nxt):
            if h is not None:
                h.stop()
        self.cur = self.nxt = None
        self.cur_it = self.nxt_it = None
        for i in (0, 1):
            for arr in (self.tex, self.ptex, self.ftex):
                if arr[i] is not None:
                    arr[i].release()
                    arr[i] = None
            self.itp[i].update(valid=False, al=1.0)
            self.tex_size[i], self.tex_serial[i] = None, -1
        self.state, self.fade = "idle", 0.0

    def _refresh(self, now):
        """The stack changed: (re)start playback, or switch between static and changing."""
        oks = self._ok()
        if not oks:
            if self.cur is not None:
                self._drop_heads()
            return
        if self.cur is None:
            it = random.choice(self._playable())
            t = self._pick_start(it)
            self.cur_it, self.cur_start = it, t
            self.cur = it["src"].make_head()
            self.cur.play_from(self.eff_start(it, t))
            if hasattr(self.cur, "set_rate"):
                self.cur.set_rate(self.speed)
            self.tex_serial = [-1, -1]
            self.fade, self.last_change = 0.0, now
            self.recent = [(it["id"], t)]
            self.state = "static"
        if self.state == "static" and (len(oks) > 1 or self.cur_it["src"].kind != "image"):
            self._prepare(now)
        elif self.state in ("armed", "prep") and len(oks) == 1 and oks[0]["src"].kind == "image":
            if self.nxt is not None:
                self.nxt.stop()
                self.nxt = self.nxt_it = None
            self.state = "static"

    # ---- scenes
    def _pf(self, it):
        return self.prefs.get(it["path"])

    def flag(self, it, t):
        """0 = plays, 1 = hidden, 2 = deleted."""
        p = self._pf(it)
        if not p:
            return 0
        if _has(p["del"], t):
            return 2
        return 1 if _has(p["off"], t) else 0

    def active_scenes(self, it):
        return [t for t in it["src"].scenes if self.flag(it, t) == 0]

    def _pool(self, it, force=False):
        src = it["src"]
        d = src.duration
        if not force and self._pf(it):                       # the user curated this clip: only their active scenes, no filler
            return self.active_scenes(it)
        pool = list(src.scenes)
        if d > 0 and len(pool) < 5:                          # few real cuts: add evenly spaced pseudo-scenes
            k = int(max(4, min(14, d / 4.0)))
            pool += [d * (i + 0.5) / k for i in range(k)]
        return pool or [0.0]

    def _playable(self):
        """Clips that still have at least one scene to play (if every scene of every clip is hidden, play anyway)."""
        oks = self._ok()
        pl = [it for it in oks if self._pool(it)]
        return pl or oks

    def _pick_start(self, it):
        if it["src"].kind == "image":
            return 0.0
        d = it["src"].duration
        pool = self._pool(it) or self._pool(it, True)
        pool = [t for t in pool if d <= 0 or t < d - 0.5] or pool or [0.0]
        return random.choice(pool)

    def _pick_next(self):
        oks = self._playable()
        cur_id = self.cur_it["id"] if self.cur_it else None
        it, ft = None, None
        if self._focus is not None:
            it = next((x for x in oks if x["id"] == self._focus), None) or next((x for x in self._ok() if x["id"] == self._focus), None)
            ft = self._force_t
            self._focus = self._force_t = None
        if it is None:
            others = [x for x in oks if x["id"] != cur_id]
            it = random.choice(others) if (others and random.random() < 0.75) else random.choice(oks)
        if ft is not None:
            return it, ft
        d = it["src"].duration
        pool = self._pool(it) or self._pool(it, True)
        far = [t for t in pool if all(not (r[0] == it["id"] and abs(t - r[1]) <= 2.0) for r in self.recent)
               and (d <= 0 or t < d - 0.5)]
        return it, random.choice(far or pool)

    # ---- scene trims (a scene's own start / end, set in the Scenes tab)
    def trim_of(self, it, t):
        """(start, end) the user set for the scene that begins at t, or None."""
        for t0, a, b in ((self._pf(it) or {}).get("trim") or []):
            if abs(t0 - t) <= SCENE_TOL:
                return a, b
        return None

    def natural_range(self, it, t):
        src = it["src"]
        sc = sorted(src.scenes)
        nxt = [x for x in sc if x > t + SCENE_TOL]
        end = nxt[0] if nxt else (src.duration if src.duration > t + 0.5 else t + 5.0)
        return t, end

    def scene_range(self, it, t):
        return self.trim_of(it, t) or self.natural_range(it, t)

    def trim_domain(self, it, t):
        """How far the trim handles may travel: a few seconds either side of the original scene."""
        a, b = self.natural_range(it, t)
        d = it["src"].duration
        return max(0.0, a - 5.0), (min(d, b + 5.0) if d > 0 else b + 5.0)

    def eff_start(self, it, t):
        tr = self.trim_of(it, t)
        return tr[0] if tr else t

    def set_trim(self, iid, t, a, b, now):
        it = next((x for x in self.items if x["id"] == iid and x["src"] is not None), None)
        if it is None:
            return
        p = self.prefs.setdefault(it["path"], {"off": [], "del": [], "trim": []})
        p["trim"] = [x for x in p.get("trim", []) if abs(x[0] - t) > SCENE_TOL]
        na, nb = self.natural_range(it, t)
        if abs(a - na) > 0.04 or abs(b - nb) > 0.04:
            p["trim"].append([t, round(a, 3), round(b, 3)])
        self._changed(it, now)

    def _head_pos(self, h):
        try:
            if hasattr(h, "pos_frames"):
                return h.pos_frames() / max(1e-6, h.src.fps)
            if hasattr(h, "_pos"):
                return h._pos()
        except Exception:
            pass
        return 0.0

    # ---- scene manager (hide / delete / restore)
    def scene_rows(self, it):
        """Rows for the Scenes tab: (start, length, flag, number) of every scene that isn't deleted."""
        src, out = it["src"], []
        sc = sorted(src.scenes)
        for i, t in enumerate(sc):
            f = self.flag(it, t)
            if f == 2:
                continue
            end = sc[i + 1] if i + 1 < len(sc) else (src.duration if src.duration > t else t)
            tr = self.trim_of(it, t)
            out.append((t, max(0.0, (tr[1] - tr[0]) if tr else (end - t)), f, i + 1))
        return out

    def counts(self, it):
        sc = it["src"].scenes
        return (sum(1 for t in sc if self.flag(it, t) == 1), sum(1 for t in sc if self.flag(it, t) == 2))

    def set_flag(self, iid, t, mode, now):
        """mode: hide (toggle) | delete | show"""
        it = next((x for x in self.items if x["id"] == iid and x["src"] is not None), None)
        if it is None:
            return
        p = self.prefs.setdefault(it["path"], {"off": [], "del": [], "trim": []})
        cur = self.flag(it, t)
        for lst in (p["off"], p["del"]):
            lst[:] = [x for x in lst if abs(x - t) > SCENE_TOL]
        if mode == "delete":
            p["del"].append(t)
        elif (mode == "hide" and cur == 0) or mode == "hide_force":
            p["off"].append(t)
        self._changed(it, now)

    def playing_scene(self):
        """(item, scene start) of the scene on screen right now, or None (images / clips with no scene list)."""
        use_n = self.state == "fading" and self.fade > 0.5 and self.nxt is not None and self.nxt_it is not None
        h, it = (self.nxt, self.nxt_it) if use_n else (self.cur, self.cur_it)
        start = self.nxt_start if use_n else self.cur_start
        if h is None or it is None or it.get("src") is None or it["src"].kind == "image" or not it["src"].scenes:
            return None
        pos = start
        if hasattr(h, "pos_frames") and it["src"].fps:
            try:
                pos = h.pos_frames() / it["src"].fps
            except Exception:
                pos = start
        sc = sorted(it["src"].scenes)
        if _has(sc, start):
            a, b = self.natural_range(it, start)
            tr = self.trim_of(it, start)
            if min(a, tr[0] if tr else a) - 0.3 <= pos < max(b, tr[1] if tr else b) + 0.15:
                return it, start
        before = [t for t in sc if t <= pos + 0.15]
        return it, (before[-1] if before else sc[0])

    def quick_flag(self, mode, now):
        """Hotkey: hide / delete the scene on screen and cut away. Returns a message for the toast."""
        ps = self.playing_scene()
        if ps is None:
            return "No scene to remove here"
        it, t = ps
        sc = sorted(it["src"].scenes)
        num = next((i + 1 for i, x in enumerate(sc) if abs(x - t) < 1e-6), 0)
        prev = self.flag(it, t)
        self.undo.append((it["id"], t, prev))
        del self.undo[:-30]
        self.set_flag(it["id"], t, mode, now)
        self.request_change()
        name = os.path.basename(it["path"])
        return f"Scene {num} of {name[:28]} {'deleted' if mode == 'delete' else 'hidden'}   (Z to undo)"

    def undo_flag(self, now):
        while self.undo:
            iid, t, prev = self.undo.pop()
            it = next((x for x in self.items if x["id"] == iid and x["src"] is not None), None)
            if it is None:
                continue
            self.set_flag(iid, t, {0: "show", 1: "hide", 2: "delete"}[prev] if prev != 1 else "hide_force", now)
            return "Scene restored" if prev == 0 else "Undone"
        return "Nothing to undo"

    def set_all(self, iid, mode, now):
        """mode: hide_all | show_all | restore"""
        it = next((x for x in self.items if x["id"] == iid and x["src"] is not None), None)
        if it is None:
            return
        p = self.prefs.setdefault(it["path"], {"off": [], "del": [], "trim": []})
        sc = list(it["src"].scenes)
        if mode == "hide_all":
            p["off"] = [t for t in sc if not _has(p["del"], t)]
        elif mode == "show_all":
            p["off"] = []
        elif mode == "restore":
            p["del"] = []
        self._changed(it, now)

    def _changed(self, it, now):
        self.pver += 1
        pf = self.prefs.get(it["path"]) or {}
        if not pf.get("off") and not pf.get("del") and not pf.get("trim"):
            self.prefs.pop(it["path"], None)
        save_scene_prefs(self.prefs)
        if self.state in ("armed", "prep") and self.nxt_it is it and self.flag(it, self.nxt_start) != 0:
            self._prepare(now)                               # the parked next scene was just hidden / deleted: pick another

    def _prepare(self, now):
        """Park a second head, paused, on the next scene's first frame so a beat can fade to it instantly."""
        if self.nxt is not None:
            self.nxt.stop()
        it, t = self._pick_next()
        self.nxt_it, self.nxt_start = it, t
        self.nxt = it["src"].make_head()
        self.nxt.play_from(self.eff_start(it, t), paused=True)
        if hasattr(self.nxt, "set_rate"):
            self.nxt.set_rate(self.speed)
        self.tex_serial[1] = -1
        self.state, self.prep_since = "prep", now

    def request_change(self):
        self._req = True

    def request_item(self, iid, t=None):
        """Jump to a scene of this particular clip (a specific scene start when t is given)."""
        self._focus = iid
        self._force_t = t
        self._req = True
        if self.state in ("armed", "prep"):
            self._prepare(time.perf_counter())

    def set_style(self, i):
        self.style = int(i) if int(i) in (0, 1, 2, 3, 4) else 0

    def on_hit(self, strength, now, playing, cooldown):
        """Kick / snare: optionally cut to another scene (rate-limited)."""
        if self.state == "armed" and playing and strength > 0 and now - self.last_change >= cooldown:
            self._go(now)

    def set_speed(self, r):
        self.speed = r
        for h in (self.cur, self.nxt):
            if h is not None and hasattr(h, "set_rate"):
                h.set_rate(r)

    def on_beat(self, strength, now, playing, rate):
        if self.state == "armed" and playing and strength > 0:
            if now - self.last_change >= 3.0 / max(0.3, rate):
                self._go(now)

    def _go(self, now):
        self.nxt.resume()
        eff = self.style
        if eff == 4:                                          # random: never the same look twice in a row
            eff = random.choice([k for k in (0, 1, 2, 3) if k != self.cur_style])
        self.cur_style = eff
        self.fade_s = self.STYLE_S[eff]
        self.state, self.fade, self._req = "fading", 0.0, False

    # ---- per frame
    def _new_tex(self, w, h, comps=3, dtype="f1"):
        t = self.ctx.texture((w, h), comps, alignment=1, dtype=dtype)
        t.filter = (moderngl.LINEAR, moderngl.LINEAR)
        t.repeat_x = t.repeat_y = False
        return t

    def _upload(self, slot, serial, frame, extra=None, now=0.0):
        h, w = frame.shape[0], frame.shape[1]
        fresh = False
        if self.tex[slot] is None or self.tex_size[slot] != (w, h):
            for arr in (self.tex, self.ptex):
                if arr[slot] is not None:
                    arr[slot].release()
                    arr[slot] = None
            self.tex[slot], self.tex_size[slot] = self._new_tex(w, h), (w, h)
            fresh = True
        it = self.itp[slot]
        if self.interp and extra is not None:
            if self.ptex[slot] is None:
                self.ptex[slot] = self._new_tex(w, h)
                fresh = True
            self.tex[slot], self.ptex[slot] = self.ptex[slot], self.tex[slot]      # last frame becomes the "previous" one
            flow, cut, fidx = extra
            ok = flow is not None and not cut and not fresh
            if ok:
                fh, fw = flow.shape[0], flow.shape[1]
                if self.ftex[slot] is None or self.ftex[slot].size != (fw, fh):
                    if self.ftex[slot] is not None:
                        self.ftex[slot].release()
                    self.ftex[slot] = self._new_tex(fw, fh, 2, "f2")
                self.ftex[slot].write(flow.tobytes())
            it["idx"], it["valid"] = fidx, ok
        else:
            it["valid"] = False
        self.tex[slot].write(frame, alignment=1)
        self.tex_serial[slot] = serial

    def update(self, now, dt, enabled, paused, rate):
        if self.cur is None:
            self.vis += (0.0 - self.vis) * (1 - math.exp(-dt * 4.0))
            self._en = bool(enabled)
            return
        self.vis += ((1.0 if enabled else 0.0) - self.vis) * (1 - math.exp(-dt * 3.0))
        self._en = bool(enabled)
        halt = paused or not enabled
        for h in (self.cur, self.nxt):
            if h is not None and (h is self.cur or self.state == "fading"):
                h.set_paused(halt)
        for slot, h in ((0, self.cur), (1, self.nxt)):
            if h is None:
                continue
            if hasattr(h, "interp"):
                h.interp = self.interp
            fr = h.get()
            if fr is not None and fr[0] != self.tex_serial[slot]:
                self._upload(slot, fr[0], fr[1], fr[2] if len(fr) > 2 else None, now)
        for slot, h in ((0, self.cur), (1, self.nxt)):
            it = self.itp[slot]
            if self.interp and it["valid"] and h is not None and hasattr(h, "pos_frames"):
                it["al"] = min(1.0, max(0.0, h.pos_frames() - (it["idx"] - 1)))      # playback clock between the two real frames
            else:
                it["al"] = 1.0
        if self.state == "prep":
            if self.tex_serial[1] != -1:
                self.state = "armed"
            elif now - self.prep_since > 4.0:                 # seek failed / too slow: try another scene
                self._prepare(now)
        elif self.state == "armed":
            hold = 3.0 / max(0.3, rate)
            tr = self.trim_of(self.cur_it, self.cur_start) if self.cur_it is not None else None
            end_hit = tr is not None and self._head_pos(self.cur) >= tr[1]          # a trimmed scene is over: cut away
            if not halt and (self._req or end_hit or now - self.last_change > max(9.0, 3.5 * hold)):
                self._go(now)                                 # request / no beats for a while
        elif self.state == "fading":
            self.fade += dt / self.fade_s
            if self.fade >= 1.0:
                self.cur.stop()
                self.cur, self.nxt = self.nxt, None
                self.cur_it, self.nxt_it = self.nxt_it, None
                self.tex[0], self.tex[1] = self.tex[1], self.tex[0]
                self.ptex[0], self.ptex[1] = self.ptex[1], self.ptex[0]
                self.ftex[0], self.ftex[1] = self.ftex[1], self.ftex[0]
                self.itp[0], self.itp[1] = self.itp[1], dict(arr=0.0, gap=1 / 30.0, valid=False, al=1.0)
                self.tex_size[0], self.tex_size[1] = self.tex_size[1], self.tex_size[0]
                self.tex_serial[0], self.tex_serial[1] = self.tex_serial[1], -1
                self.cur_start = self.nxt_start
                self.recent = (self.recent + [(self.cur_it["id"], self.cur_start)])[-6:]
                self.fade, self.last_change = 0.0, now
                self._prepare(now)

    def bind(self, prog, amt, pulse, blend, gain=1.0, sway=(0.0, 0.0, 0.0, 0.0), solo=False):
        """Set the shader's media uniforms and bind the two frame textures (units 5 and 6).
        solo = Video style: while there is any media the picture is always the footage (black while a frame is on its way),
        so the spiral can never flash up between scenes."""
        on = self.cur is not None and self.tex[0] is not None and self.vis > 0.003 and (amt > 0.003 or solo)
        if solo and not on and self.items and self._en:
            on, amt = True, max(amt, 0.5)
        a = self.tex[0] if self.tex[0] is not None else self.dummy
        fading = self.state == "fading" and self.tex[1] is not None
        b = self.tex[1] if fading else a
        a.use(5)
        b.use(6)
        sb_ = 1 if fading else 0
        (self.ptex[0] or self.dummy).use(7)
        (self.ftex[0] or self.dummy_flow).use(8)
        (self.ptex[sb_] or self.dummy).use(9)
        (self.ftex[sb_] or self.dummy_flow).use(10)
        use_i = self.interp and self.itp[0]["valid"] and self.ftex[0] is not None
        use_ib = self.interp and self.itp[sb_]["valid"] and self.ftex[sb_] is not None
        f = min(1.0, max(0.0, self.fade)) if self.state == "fading" else 0.0
        cs = self.cur_style
        blur = math.sin(math.pi * f) * 0.045 if (cs == 1 and self.state == "fading") else 0.0
        f = f * f * f * (f * (6 * f - 15) + 10)
        zz = f if (cs == 3 and self.state == "fading") else -1.0
        sa = self.tex_size[0] or (1, 1)
        sb = (self.tex_size[1] or sa) if fading else sa
        vals = {"uMediaA": 5, "uMediaB": 6, "uPrevA": 7, "uFlowA": 8, "uPrevB": 9, "uFlowB": 10,
                "uMI": (1.0 if self.interp else 0.0, self.itp[0]["al"] if use_i else 1.0, self.itp[sb_]["al"] if use_ib else 1.0),
                 "uMOn": 1.0 if on else 0.0, "uMFade": f,
                "uMAmt": float(amt * self.vis), "uMAsp": sa[0] / float(sa[1]), "uMAspB": sb[0] / float(sb[1]),
                "uMPulse": float(pulse), "uMBlend": int(blend), "uMBlur": float(blur), "uMZ": float(zz), "uMGain": float(gain), "uMSway": tuple(float(x) for x in sway)}
        for k, v in vals.items():
            if k in prog:
                prog[k].value = v

    def scene_state(self, iid, scroll, per):
        """Everything the Scenes tab draws. Thumbnails of the visible rows are requested here (fetched in the background)."""
        its = [it for it in self._ok() if it["src"].kind != "image"]
        it = next((x for x in its if x["id"] == iid), None) or (its[0] if its else None)
        if it is None:
            return dict(none=True, n_clips=0, sig=("none",))
        rows = self.scene_rows(it)
        top = scroll
        out = []
        for t, ln, f, no in rows[top:top + per]:
            out.append(dict(t=t, ln=ln, f=f, no=no, th=it["src"].scene_thumb(t), tr=self.trim_of(it, t) is not None))
        hid, dele = self.counts(it)
        idx = its.index(it)
        s = it["src"]
        return dict(none=False, id=it["id"], name=os.path.basename(it["path"]), rows=out, total=len(rows), hidden=hid, deleted=dele,
                    idx=idx, n_clips=len(its), tid=id(s), few=(len(s.scenes) < 5 and not self._pf(it)),
                    scanning=getattr(s, "scan", 1.0) < 1.0,
                    sig=(it["id"], self.pver, getattr(s, "sthumb_n", 0), scroll, len(s.scenes), len(rows)), it=it)

    def info(self):
        oks = self._ok()
        d = dict(loaded=False, loading=any(it["src"] is None for it in self.items), count=len(self.items), items=[],
                 cur_id=self.cur_it["id"] if self.cur_it else None,
                 total_scenes=sum(max(1, len(self.active_scenes(it))) if not self._pf(it) else len(self.active_scenes(it)) for it in oks),
                 pver=self.pver)
        for it in self.items:
            s = it["src"]
            row = dict(id=it["id"], name=os.path.basename(it["path"]), ok=s is not None)
            if s is not None:
                row.update(kind=s.kind, size=s.orig, duration=round(s.duration, 1), scenes=len(self.active_scenes(it)),
                           scan=int(s.scan * 100), tid=id(s), thumb=s.thumb)
            d["items"].append(row)
        shown = next((r for r in d["items"] if r["id"] == d["cur_id"] and r["ok"]), None) or \
            next((r for r in d["items"] if r["ok"]), None)
        if shown is not None:
            d.update(loaded=True, name=shown["name"], kind=shown["kind"], size=shown["size"], duration=shown["duration"],
                     scenes=shown["scenes"], scan=shown["scan"], tid=shown["tid"], thumb=shown["thumb"])
        d["sig"] = tuple((r["id"], r["ok"], r.get("scan"), r.get("scenes")) for r in d["items"]) + (d["cur_id"], self.pver)
        return d


# --------------------------------------------------------------------------- app
class App:
    def __init__(self):
        if sys.platform == "win32":
            try:
                ctypes.windll.shcore.SetProcessDpiAwareness(2)
            except Exception:
                try:
                    ctypes.windll.user32.SetProcessDPIAware()
                except Exception:
                    pass

        pygame.init()
        pygame.display.set_caption("Hypnosis")
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
        pygame.display.set_mode((1280, 720), pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE, vsync=1)
        try:
            from pygame._sdl2.video import Window
            self.win = Window.from_display_module()
        except Exception:
            self.win = None

        self.ctx = moderngl.create_context()
        self.init_gl()

        self.cfg = load_settings()
        self.audio = AudioEngine()
        self.audio.set_volume(self.cfg["volume"])
        self.ana = Analyzer()
        self.phrases = load_phrases()

        self.playlist, self.cur = [], -1
        self.finish_handled = False
        self.is_fs = False
        self.menu_open = False
        self.tab = "queue"
        self.scroll = 0
        self.hover = None
        self.drag = None
        self.mouse = (0, 0)
        self.picked = []          # files chosen in the file dialog thread

        self.rot = self.flow = self.hue = 0.0
        self.spin_kick = 0.0
        self.dim = 0.55
        self.dom_prev, self.dom_last = 0.0, 0.0
        self.toast, self.toast_t = "", -10.0
        self.vscroll, self.vscroll_d = 0.0, 0.0
        self.tab_seen, self.tab_old, self.old_tex, self.tab_k = self.tab, self.tab, None, 1.0
        self.card_mix, self.card_tex, self.demo = {}, {}, None          # per style card: hover mix 0..1, (texture, fbo); shared fake-beat clock
        op = set(filter(None, self.cfg.get("vis_open", "").split(",")))
        self.sec_a = {n: (1.0 if n in op else 0.0) for n, _ in VIS_SECTIONS}
        self.last_mouse = time.perf_counter()
        self.held = {}
        self.running = True
        self.menu_p = 0.0         # linear menu progress 0..1 (animated)
        self.mode_shown = self.cfg["mode"]
        self.mode_t = 1.0         # 1 = no mode transition running
        self.final_t = None
        self.press = {}           # button -> press animation state
        self.hov_sp = {}          # button -> hover spring state
        self.sl = {}              # slider -> spring state (display position, thumb scale)
        self.exit_at = None
        self.zoomk = 0.0          # Kaleidoscope zoom phase (wrapped at the lattice period)
        self.zk_speed = 1.0
        self.zk_boost = 0.0
        self.spotify_on = False
        self.cap = None
        self.cap_lock = threading.Lock()
        self.spot_wake = threading.Event()
        self.spot_vol, self.spot_vol_pend, self.spot_vol_t, self.spot_vol_warned = None, None, 0.0, False
        self.sp_play_ov = None
        self.sp_tr = None                                  # (start time, turning on?) of the Spotify button wave
        self.smtc = SmtcSpotify(lambda: self.spotify_on, lambda: self.toast_msg("Spotify seeking needs winrt: run setup_env.bat (or pip install winrt-runtime winrt-Windows.Media.Control)"))
        self.attach_failed_pid = None
        self.ana_spot = None
        self.spot_title, self.spot_running, self.spot_error, self.spot_last = "", False, "", ""
        self.art_title, self.art_img, self.art_ver, self.art_cache = None, None, 0, None
        self.spot_track = ""                               # last real "Artist - Song" seen (kept while paused)   # Spotify album art (fetched in the background)
        threading.Thread(target=self._poll_spotify, daemon=True).start()
        if self.cfg["spotify"]:
            self.set_spotify(True)

    def init_gl(self):
        ctx = self.ctx
        quad = ctx.buffer(np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype="f4").tobytes())
        self.screen_fbo = ctx.screen
        self.prog = ctx.program(vertex_shader=VERT, fragment_shader=FRAG)
        self.vao = ctx.vertex_array(self.prog, [(quad, "2f", "in_pos")])
        self.quads = TexQuad(ctx, quad)
        self.overlay = Overlay(ctx, self.quads)
        self.glitch = GlitchText(ctx, quad, self.quads)
        self.panel = Panel()
        self.pblur_prog = ctx.program(vertex_shader=RECT_VERT, fragment_shader=PBLUR_FRAG)
        self.pblur_vao = ctx.vertex_array(self.pblur_prog, [(quad, "2f", "in_pos")])
        self.prev_prog = ctx.program(vertex_shader=RECT_VERT, fragment_shader=PREV_FRAG)
        self.prev_vao = ctx.vertex_array(self.prev_prog, [(quad, "2f", "in_pos")])
        self.media = MediaLayer(ctx)
        self.media_picked = []
        self.mscroll = 0
        self.scn_id, self.scn_scroll = None, 0
        self.scn_sel, self.scn_sel_i, self.trim_drag = None, 0, None
        self.pv = dict(head=None, key=None, tex=None, ser=None, pos=None, t0=0.0, s=0.0, e=0.0, scrub=-1.0)
        self.style_prev = None
        self.reset_t = -10.0
        self.panel_tex = None
        self.presets = load_presets()
        self.preset_sel = load_last_preset_name()
        if self.preset_sel not in [n for n, _ in self.presets]:
            self.preset_sel = self.presets[-1][0] if self.presets else ""
        self.preset_open = False
        self.preset_del_t = -10.0
        self.panel_sig = None
        self.panel_rect = (0, 0, 1, 1)
        self.panel_scale = 1.0
        self.panel_base = (1, 1)
        # blur / glass pipeline
        self.down_prog = ctx.program(vertex_shader=VERT, fragment_shader=DOWN_FRAG)
        self.down_vao = ctx.vertex_array(self.down_prog, [(quad, "2f", "in_pos")])
        self.blur_prog = ctx.program(vertex_shader=VERT, fragment_shader=BLUR_FRAG)
        self.blur_vao = ctx.vertex_array(self.blur_prog, [(quad, "2f", "in_pos")])
        self.comp_prog = ctx.program(vertex_shader=VERT, fragment_shader=COMP_FRAG)
        self.comp_vao = ctx.vertex_array(self.comp_prog, [(quad, "2f", "in_pos")])
        self.trans_prog = ctx.program(vertex_shader=VERT, fragment_shader=TRANS_FRAG)
        self.trans_vao = ctx.vertex_array(self.trans_prog, [(quad, "2f", "in_pos")])
        self.copy_prog = ctx.program(vertex_shader=VERT, fragment_shader=COPY_FRAG)
        self.copy_vao = ctx.vertex_array(self.copy_prog, [(quad, "2f", "in_pos")])
        self.target_size = None
        self.rt_objs = []

    # ---------------------------------------------------------------- offscreen targets / blur
    def ensure_targets(self, W, H):
        if self.target_size == (W, H):
            return
        for o in self.rt_objs:
            o.release()

        def mk(w, h):
            t = self.ctx.texture((max(2, w), max(2, h)), 4)
            t.filter = (moderngl.LINEAR, moderngl.LINEAR)
            t.repeat_x = False
            t.repeat_y = False
            f = self.ctx.framebuffer(color_attachments=[t])
            self.rt_objs.extend([t, f])
            return t, f

        self.rt_objs = []
        self.scene_t, self.scene_f = mk(W, H)
        self.a1, self.a1f = mk(W // 4, H // 4)
        self.a2, self.a2f = mk(W // 4, H // 4)
        self.c1, self.c1f = mk(W // 8, H // 8)
        self.c2, self.c2f = mk(W // 8, H // 8)
        self.prev_t, self.prev_f = mk(W, H)            # frozen look we are fading away from
        self.trans_t, self.trans_f = mk(W, H)          # blended result while a mode change runs
        self.tba1, self.tba1f = mk(W // 4, H // 4)
        self.tba2, self.tba2f = mk(W // 4, H // 4)
        self.tbb1, self.tbb1f = mk(W // 4, H // 4)
        self.tbb2, self.tbb2f = mk(W // 4, H // 4)
        self.target_size = (W, H)
        self.mode_t = 1.0                              # (re)created targets: no transition in flight

    def pass_down(self, src, dst_f, k):
        dst_f.use()
        src.use(0)
        self.down_prog["uTex"].value = 0
        self.down_prog["uTexel"].value = (k / src.width, k / src.height)
        self.down_vao.render(moderngl.TRIANGLE_STRIP)

    def pass_blur(self, src, dst_f, dx, dy):
        dst_f.use()
        src.use(0)
        self.blur_prog["uTex"].value = 0
        self.blur_prog["uStep"].value = (dx / src.width, dy / src.height)
        self.blur_vao.render(moderngl.TRIANGLE_STRIP)

    def blur_chain(self, src, t1, f1, t2, f2, step):
        """Blur `src` into t1 (quarter resolution): downsample + two separable gaussian rounds."""
        self.pass_down(src, f1, 1.0)
        for _ in range(2):
            self.pass_blur(t1, f2, step, 0.0)
            self.pass_blur(t2, f1, 0.0, step)

    def transition_pass(self, k):
        """Mode change: blur the old look out and the new look in. Writes trans_f."""
        bo = smoothstep(0.0, 0.5, k)
        bn = 1.0 - smoothstep(0.5, 1.0, k)
        self.blur_chain(self.prev_t, self.tba1, self.tba1f, self.tba2, self.tba2f, 0.5 + 2.6 * bo)
        self.blur_chain(self.scene_t, self.tbb1, self.tbb1f, self.tbb2, self.tbb2f, 0.5 + 2.6 * bn)
        W, H = self.target_size
        self.trans_f.use()
        self.ctx.viewport = (0, 0, W, H)
        self.scene_t.use(0)
        self.prev_t.use(1)
        self.tbb1.use(2)
        self.tba1.use(3)
        tp = self.trans_prog
        tp["uNew"].value = 0
        tp["uOld"].value = 1
        tp["uNewB"].value = 2
        tp["uOldB"].value = 3
        tp["uK"].value = float(k)
        self.trans_vao.render(moderngl.TRIANGLE_STRIP)

    def present(self, tex, W, H):
        self.screen_fbo.use()
        self.ctx.viewport = (0, 0, W, H)
        self.ctx.disable(moderngl.BLEND)
        tex.use(0)
        self.copy_prog["uTex"].value = 0
        self.copy_vao.render(moderngl.TRIANGLE_STRIP)

    def blur_scene(self, e):
        """Backdrop blur (radius grows with the animation) + a heavier blur for the glass."""
        step = 0.5 + 2.6 * e
        self.pass_down(self.final_t, self.a1f, 1.0)
        for _ in range(2):
            self.pass_blur(self.a1, self.a2f, step, 0.0)
            self.pass_blur(self.a2, self.a1f, 0.0, step)
        self.pass_down(self.a1, self.c1f, 0.5)
        for _ in range(2):
            self.pass_blur(self.c1, self.c2f, 1.7, 0.0)
            self.pass_blur(self.c2, self.c1f, 0.0, 1.7)

    # ---------------------------------------------------------------- spotify mode
    def _poll_spotify(self):
        """Background: read Spotify's window title, and (re)attach the capture to Spotify's process."""
        try:
            import comtypes
            comtypes.CoInitialize()                           # COM is per thread (Spotify volume)
        except Exception:
            pass
        while self.running:
            if self.spotify_on:
                try:
                    self.spot_running, self.spot_title = find_spotify_window()
                except Exception:
                    self.spot_running, self.spot_title = False, ""
                t_ = self.spot_title
                if t_ and not t_.lower().startswith("spotify"):
                    self.spot_track = t_
                if t_ != self.art_title and t_ and not t_.lower().startswith("spotify"):
                    self.art_title = t_
                    try:
                        img = fetch_album_art(t_)
                    except Exception as ex:
                        log(f"album art lookup failed: {ex}")
                        img = None
                    if self.art_title == t_:
                        self.art_img, self.art_ver = img, self.art_ver + 1
                try:
                    pid = find_spotify_root_pid()
                except Exception:
                    pid = None
                self._sync_capture(pid)
                try:
                    if self.spot_vol_pend is not None:
                        got = spotify_session_volume(self.spot_vol_pend)
                        if got is not None:
                            self.spot_vol_pend = None
                    elif time.perf_counter() - self.spot_vol_t > 1.5 and not (self.drag and self.drag[0] == "vol"):
                        cur = spotify_session_volume()
                        if cur is not None:
                            self.spot_vol = cur
                except ImportError:
                    self.spot_vol_pend = None
                    if not self.spot_vol_warned:
                        self.spot_vol_warned = True
                        self.toast_msg("Spotify volume needs pycaw: run setup_env.bat (or pip install pycaw)")
                except Exception as ex:
                    log(f"spotify volume failed: {ex}")
            self.spot_wake.wait(1.0)
            self.spot_wake.clear()

    def _sync_capture(self, pid):
        with self.cap_lock:
            if not self.spotify_on:
                return
            cap = self.cap
            if pid is None:
                if cap:
                    cap.stop()
                    self.cap = None
                self.attach_failed_pid = None
                self.spot_error = "Waiting for Spotify"
                return
            if cap is not None and cap.pid == pid:
                return
            if cap:
                cap.stop()
                self.cap = None
            if self.attach_failed_pid == pid:
                return                                       # don't retry a failing PID every second
            new = ProcessCapture(pid)
            try:
                new.start()
            except Exception as e:
                new.stop()
                import traceback
                traceback.print_exc()
                self.attach_failed_pid = pid
                self.spot_error = f"Capture failed: {e}"
                log(f"Spotify attach to pid {pid} failed: " + traceback.format_exc())
                return
            self.attach_failed_pid = None
            self.cap = new
            self.ana_spot = Analyzer(2048, new.rate)
            self.spot_error = ""
            log(f"Spotify attached: pid {pid}, {new.rate} Hz, {new.channels} ch, {new.bits}-bit")

    def set_spotify(self, on):
        if on == self.spotify_on:
            return
        if on:
            if sys.platform != "win32":
                self.spot_error = "Windows only"
                self.toast_msg("Spotify Mode needs Windows (WASAPI process loopback)")
                self.cfg["spotify"] = False
                return
            try:
                import proctap._native  # noqa: F401
            except ImportError as e:
                import traceback
                traceback.print_exc()
                print(f"[Spotify Mode] running under: {sys.executable}")
                log(f"Spotify Mode import failed ({'exe' if getattr(sys, 'frozen', False) else 'python'}: {sys.executable}): "
                    + traceback.format_exc())
                if (getattr(e, "name", "") or "").split(".")[0] == "proctap":
                    self.spot_error = "Install proc-tap"
                else:
                    self.spot_error = "proc-tap won't load"
                self.toast_msg(f"Spotify Mode import error: {e}")
                self.cfg["spotify"] = False
                return
            self.spot_error = "Waiting for Spotify"
            self.attach_failed_pid = None
            self.audio.stop()                      # local playback gives way to Spotify
            self.spot_last = ""
            self.spotify_on = True
            self.sp_tr = (time.perf_counter(), True)
            self.smtc.start()
            self.spot_wake.set()                   # attach right away
        else:
            self.sp_tr = (time.perf_counter(), False)
            with self.cap_lock:
                self.spotify_on = False
                if self.cap:
                    self.cap.stop()
                    self.cap = None
            self.spot_error = ""
        self.cfg["spotify"] = self.spotify_on

    # ---------------------------------------------------------------- playlist
    def toast_msg(self, msg):
        self.toast, self.toast_t = msg, time.perf_counter()

    def ensure_style_previews(self):
        """Render each visual style once into a small offscreen target for the Visuals tab cards."""
        if self.style_prev is not None:
            return
        ctx = self.ctx
        w, h = 320, 180
        fbo = ctx.simple_framebuffer((w, h))
        fbo.use()
        ctx.viewport = (0, 0, w, h)
        out = []
        for m in range(len(MODES)):
            for name, val in (("uRes", (float(w), float(h))), ("uRot", 0.9), ("uFlow", 0.3), ("uFlowK", 0.8), ("uZoom", 0.0),
                              ("uAb", 0.004), ("uBass", 0.15), ("uMid", 0.3), ("uHigh", 0.2), ("uEnergy", 0.5), ("uHue", 0.1),
                              ("uTime", 7.0 + m), ("uPsyT", 7.0 + m), ("uWarp", 0.06), ("uDim", 1.0), ("uMode", m if m != VIDEO_MODE else 0), ("uMOn", 0.0)):
                if name in self.prog:
                    self.prog[name].value = val
            self.vao.render(moderngl.TRIANGLE_STRIP)
            raw = np.frombuffer(fbo.read(components=3), dtype=np.uint8).reshape(h, w, 3)
            img = np.ascontiguousarray(raw[::-1])
            if m == VIDEO_MODE:
                img = np.zeros_like(img)                 # Video card: black until a clip's thumbnail takes its place
            out.append(img)
        fbo.release()
        self.style_prev = out

    def reset_all(self):
        keep = {"media_paths", "media_path", "media_on", "spotify", "media_remember"}
        for k, v in DEFAULTS.items():
            if k not in keep:
                self.cfg[k] = v
        self.audio.set_volume(self.cfg["volume"])
        self.toast_msg("Settings reset to defaults")

    def preset_snapshot(self):
        return {k: v for k, v in self.cfg.items() if k in DEFAULTS and k not in PRESET_SKIP}

    def preset_save_new(self):
        used = {n for n, _ in self.presets}
        n = 1
        while f"Preset {n}" in used:
            n += 1
        name = f"Preset {n}"
        self.presets.append((name, self.preset_snapshot()))
        self.preset_sel = name
        save_presets(self.presets, name)
        self.toast_msg(f"Saved {name}")

    def preset_update(self):
        for i, (n, _) in enumerate(self.presets):
            if n == self.preset_sel:
                self.presets[i] = (n, self.preset_snapshot())
                save_presets(self.presets, n)
                self.toast_msg(f"Updated {n}")
                return

    def preset_load(self, name):
        for n, vals in self.presets:
            if n == name:
                for k, v in vals.items():
                    self.cfg[k] = v
                if self.cfg.get("media_blend", 0) not in (0, 1, 2):
                    self.cfg["media_blend"] = 0
                for k in ("kick_lo", "kick_hi", "snare_lo", "snare_hi"):
                    self.fix_pairs(k)
                self.preset_sel = name
                save_presets(self.presets, name)
                self.toast_msg(f"Loaded {name}")
                return

    def preset_delete(self):
        self.presets = [(n, v) for n, v in self.presets if n != self.preset_sel]
        self.preset_sel = self.presets[-1][0] if self.presets else ""
        save_presets(self.presets, self.preset_sel)
        self.toast_msg("Preset deleted")

    def scn_clips(self):
        return [it for it in self.media._ok() if it["src"].kind != "image"]

    def scn_pick_default(self):
        its = self.scn_clips()
        if any(it["id"] == self.scn_id for it in its):
            return
        cur = self.media.cur_it
        self.scn_id = cur["id"] if (cur and cur["src"].kind != "image") else (its[0]["id"] if its else None)

    def scn_step(self, d):
        its = self.scn_clips()
        if not its:
            return
        i = next((k for k, it in enumerate(its) if it["id"] == self.scn_id), 0)
        self.scn_id = its[(i + d) % len(its)]["id"]
        self.scn_scroll = 0
        self.scn_sel = None

    def scn_clamp(self):
        it = next((x for x in self.scn_clips() if x["id"] == self.scn_id), None)
        n = len(self.media.scene_rows(it)) if it else 0
        self.scn_scroll = max(0, min(max(0, n - SCN_ROWS), self.scn_scroll))

    def sync_media_cfg(self):
        self.cfg["media_paths"] = "|".join(self.media.paths())

    def add_media(self, paths):
        have = set(self.media.paths())
        new = [p for p in paths if p not in have]
        if not new:
            return
        self.media.add(new)
        self.cfg["media_on"] = True
        self.sync_media_cfg()
        self.toast_msg(("Adding " + os.path.basename(new[0])) if len(new) == 1 else f"Adding {len(new)} clips")

    def add_files(self, paths):
        vis = [p for p in paths if is_media_path(p)]
        if vis:
            self.add_media(vis)
            paths = [p for p in paths if p not in vis]
        files = collect_audio(paths)
        if not files:
            return
        if self.spotify_on:
            self.set_spotify(False)              # dropping music = back to local playback
        first_new = len(self.playlist)
        self.playlist.extend(files)
        if (self.audio.data is None and not self.audio.loading) or self.audio.finished:
            self.play_index(first_new)
        self.scroll_to_current()

    def play_index(self, i):
        if 0 <= i < len(self.playlist):
            self.cur = i
            self.finish_handled = False
            self.audio.finished = False
            self.audio.load(self.playlist[i])
            self.scroll_to_current()

    def next_index(self):
        if self.cur + 1 < len(self.playlist):
            return self.cur + 1
        if self.cfg["loop"] and self.playlist:
            return 0
        return None

    def next_track(self):
        if self.spotify_on:
            send_media_key(0xB0)
            return
        n = self.next_index()
        if n is not None:
            self.play_index(n)

    def prev_track(self):
        if self.spotify_on:
            send_media_key(0xB1)
            return
        if self.audio.pos > 3 * SR or self.cur <= 0:
            self.audio.pos = 0
            self.audio.finished = False
        else:
            self.play_index(self.cur - 1)

    def remove_index(self, i):
        if not (0 <= i < len(self.playlist)):
            return
        was_current = i == self.cur
        if was_current:
            self.audio.stop()
        self.playlist.pop(i)
        if i < self.cur:
            self.cur -= 1
        elif was_current:
            if i < len(self.playlist):
                self.play_index(i)
            elif self.cfg["loop"] and self.playlist:
                self.play_index(0)
            else:
                self.cur = -1
        self.scroll = max(0, min(self.scroll, max(0, len(self.playlist) - LIST_ROWS)))

    def move_index(self, i, d):
        j = i + d
        if not (0 <= i < len(self.playlist) and 0 <= j < len(self.playlist)):
            return
        self.playlist[i], self.playlist[j] = self.playlist[j], self.playlist[i]
        if self.cur == i:
            self.cur = j
        elif self.cur == j:
            self.cur = i

    def clear_upcoming(self):
        if self.cur >= 0:
            del self.playlist[self.cur + 1:]
        else:
            self.playlist.clear()
        self.scroll = max(0, min(self.scroll, max(0, len(self.playlist) - LIST_ROWS)))

    def clear_all(self):
        self.playlist.clear()
        self.cur = -1
        self.audio.stop()
        self.scroll = 0

    def shuffle_upcoming(self):
        tail = self.playlist[self.cur + 1:]
        random.shuffle(tail)
        self.playlist[self.cur + 1:] = tail

    def scroll_to_current(self):
        if self.cur < 0:
            return
        if self.cur < self.scroll:
            self.scroll = self.cur
        elif self.cur >= self.scroll + LIST_ROWS:
            self.scroll = self.cur - LIST_ROWS + 1

    def spot_playing(self):
        """Is Spotify playing? The click's own answer for a moment, then Windows' media session, else whether audio is flowing."""
        ov = self.sp_play_ov
        if ov is not None and time.perf_counter() < ov[1]:
            return ov[0]
        if self.smtc.dur is not None:
            return bool(self.smtc.playing)
        return bool(self.cap and self.cap.active)

    def play_pause(self):
        if self.spotify_on:
            now_playing = self.spot_playing()
            self.sp_play_ov = (not now_playing, time.perf_counter() + 1.5)       # flip the icon right away; the real state takes over after
            send_media_key(0xB3)
            return
        if self.audio.data is None:
            if self.playlist and not self.audio.loading:
                self.play_index(max(self.cur, 0))
        else:
            self.audio.toggle()

    # ---------------------------------------------------------------- display
    def set_display(self, kind):
        w = self.win
        if kind == "windowed":
            if self.is_fs:
                if w:
                    w.set_windowed()
                else:
                    pygame.display.toggle_fullscreen()
            self.is_fs = False
        else:
            self.cfg["fs_kind"] = kind
            if w:
                w.set_fullscreen(desktop=(kind == "borderless"))
            elif not self.is_fs:
                pygame.display.toggle_fullscreen()
            self.is_fs = True

    def toggle_fullscreen(self):
        self.set_display(self.cfg["fs_kind"] if not self.is_fs else "windowed")

    # ---------------------------------------------------------------- menu
    def open_menu(self, on):
        self.menu_open = on
        self.hover = None
        self.drag = None
        self.preset_open = False
        if on:
            pygame.mouse.set_visible(True)
            self.scroll_to_current()
        else:
            save_settings(self.cfg)

    def slider_value(self, key, lx):
        lo, hi = next((a, b) for k, _, a, b in SLIDERS if k == key)
        t = max(0.0, min(1.0, (lx - TRACK_X0) / TRACK_W))
        v = lo + (hi - lo) * t
        if abs(v - 1.0) < 0.03 * (hi - lo) and key not in ("volume", "spot_delay", "kick_lo", "kick_hi", "snare_lo", "snare_hi"):
            v = 1.0                                # snap to default
        self.cfg[key] = v
        self.fix_pairs(key)
        if key == "volume":
            self.audio.set_volume(v)

    def fix_pairs(self, key):
        c = self.cfg
        if key == "kick_lo":
            c["kick_hi"] = max(c["kick_hi"], c["kick_lo"] + 10)
        elif key == "kick_hi":
            c["kick_lo"] = min(c["kick_lo"], c["kick_hi"] - 10)
        elif key == "snare_lo":
            c["snare_hi"] = max(c["snare_hi"], c["snare_lo"] + 20)
        elif key == "snare_hi":
            c["snare_lo"] = min(c["snare_lo"], c["snare_hi"] - 20)

    def to_logical(self, mx, my):
        x0, y0, w, h = self.panel_rect
        return (mx - x0) * PW / max(1.0, w), (my - y0) * PH / max(1.0, h)

    def menu_press(self, mx, my):
        if self.menu_p < 0.8:
            return
        lx, ly = self.to_logical(mx, my)
        key = self.panel.pick(lx, ly)
        if key is None:
            if not (0 <= lx <= PW and 0 <= ly <= PH):
                self.open_menu(False)
            return
        kind = key[0]
        if self.preset_open and kind not in ("preset_pick", "noop"):
            self.preset_open = False
            if kind == "preset_dd":
                return
        if kind not in ("slider", "seek", "noop", "trim", "vol"):
            self.press_start(key)
        if kind == "noop":
            pass
        elif kind == "preset_dd":
            self.preset_open = bool(self.presets)
        elif kind == "preset_pick":
            self.preset_open = False
            self.preset_load(key[1])
        elif kind == "preset_save":
            self.preset_save_new()
        elif kind == "preset_update":
            self.preset_update()
        elif kind == "preset_delete":
            now = time.perf_counter()
            if now - self.preset_del_t < 3.0:
                self.preset_del_t = -10.0
                self.preset_delete()
            else:
                self.preset_del_t = now
        elif kind == "tab":
            self.tab = key[1]
            if self.tab == "scenes":
                self.scn_scroll = 0
                self.scn_pick_default()
        elif kind == "close":
            self.open_menu(False)
        elif kind == "toggle":
            name = key[1]
            self.cfg[name] = not self.cfg[name]
        elif kind == "spotify":
            self.set_spotify(not self.spotify_on)
        elif kind == "prev":
            self.prev_track()
        elif kind == "next":
            self.next_track()
        elif kind == "playpause":
            self.play_pause()
        elif kind == "seek":
            self.drag = key
            self.seek_to(lx)
        elif kind == "vol":
            self.drag = key
            self.vol_to(lx)
        elif kind == "row_play":
            self.play_index(key[1])
        elif kind == "row_up":
            self.move_index(key[1], -1)
        elif kind == "row_down":
            self.move_index(key[1], 1)
        elif kind == "row_remove":
            self.remove_index(key[1])
        elif kind == "add_files":
            threading.Thread(target=pick_files_dialog, args=(self.picked,), daemon=True).start()
        elif kind == "clear_up":
            self.clear_upcoming()
        elif kind == "clear_all":
            self.clear_all()
        elif kind == "shuffle":
            self.shuffle_upcoming()
        elif kind == "display":
            self.set_display(key[1])
        elif kind == "style":
            self.cfg["mode"] = key[1]
        elif kind == "slider":
            self.drag = key
            self.slider_value(key[1], lx)
        elif kind == "sl_reset":
            self.set_slider(key[1], DEFAULTS[key[1]])
        elif kind == "media_add":
            threading.Thread(target=pick_media_dialog, args=(self.media_picked,), daemon=True).start()
        elif kind == "media_clear":
            self.media.clear()
            self.sync_media_cfg()
            self.mscroll = 0
            self.scn_id, self.scn_scroll = None, 0
            self.scn_sel = None
        elif kind == "mrow":
            self.media.request_item(key[1])
        elif kind == "sec":
            op = [n for n, _ in VIS_SECTIONS if n in set(filter(None, self.cfg["vis_open"].split(",")))]
            if key[1] in op:
                op.remove(key[1])
            else:
                op.append(key[1])
                self.reveal = key[1]                                   # scroll the freshly opened section into view as it grows
            self.cfg["vis_open"] = ",".join(op)
        elif kind == "trim":
            self.trim_begin(lx)
        elif kind == "scn_go":
            it = next((x for x in self.scn_clips() if x["id"] == key[1]), None)
            ex = next((r[0] for r in self.media.scene_rows(it) if abs(r[0] - key[2]) < 0.01), key[2]) if it else key[2]
            self.scn_sel = (key[1], ex)
        elif kind == "scn_trim_reset":
            it = next((x for x in self.scn_clips() if x["id"] == self.scn_sel[0]), None) if self.scn_sel else None
            if it is not None:
                a, b = self.media.natural_range(it, self.scn_sel[1])
                self.media.set_trim(self.scn_sel[0], self.scn_sel[1], a, b, time.perf_counter())
        elif kind == "scn_play":
            if self.scn_sel:
                self.media.request_item(self.scn_sel[0], self.scn_sel[1])
                self.toast_msg("Playing that scene on screen")
        elif kind == "scn_hide":
            self.media.set_flag(key[1], key[2], "hide", time.perf_counter())
        elif kind == "scn_del":
            self.media.set_flag(key[1], key[2], "delete", time.perf_counter())
            self.scn_clamp()
        elif kind == "scn_all":
            if self.scn_id is not None:
                self.media.set_all(self.scn_id, key[1], time.perf_counter())
                self.scn_clamp()
        elif kind == "scn_clip":
            self.scn_step(key[1])
        elif kind == "mrow_remove":
            self.media.remove(key[1], time.perf_counter())
            self.sync_media_cfg()
        elif kind == "reset_all":
            now = time.perf_counter()
            if now - self.reset_t < 3.0:
                self.reset_t = -10.0
                self.reset_all()
            else:
                self.reset_t = now
        elif kind == "mstyle":
            self.cfg["media_style"] = key[1]
        elif kind == "mblend":
            self.cfg["media_blend"] = key[1]
        elif kind == "exit":
            self.exit_at = time.perf_counter() + 0.18         # let the press animation play, then quit

    def press_start(self, key):
        ent = self.press.get(key)
        t0 = time.perf_counter()
        if ent:
            ent["held"], ent["t0"] = True, t0
        else:
            self.press[key] = {"v": 0.0, "w": 0.0, "held": True, "t0": t0}

    def update_press(self, now, dt):
        """Every menu button is a damped spring: quick push-in, then a soft overshoot as it springs back."""
        dead = []
        for k, ent in self.press.items():
            target = 1.0 if (ent["held"] or now - ent["t0"] < 0.09) else 0.0
            ent["v"], ent["w"] = spring_step(ent["v"], ent["w"], target, 900.0, 33.0, dt)
            if target == 0.0 and abs(ent["v"]) < 0.004 and abs(ent["w"]) < 0.05:
                dead.append(k)
        for k in dead:
            del self.press[k]

    def smooth(self, key, tgt, dt, sm):
        """One-pole smoother with a fast-ish rise and slower fall, both stretched by the smoothing slider."""
        st = self.__dict__.setdefault("_sm", {})
        v = st.get(key, tgt)
        if sm <= 0.001:
            st[key] = tgt
            return tgt
        tau = (0.012 + 0.20 * sm) if tgt > v else (0.03 + 0.55 * sm)
        v += (tgt - v) * (1.0 - math.exp(-dt / tau))
        st[key] = v
        return v

    def update_hover(self, dt):
        """Every button swells with a spring when the cursor is over it (and settles back with a little overshoot)."""
        h = self.hover if (self.menu_open and self.hover and self.hover[0] not in ("slider", "seek", "row_play", "vol")) else None
        if h is not None and h not in self.hov_sp:
            self.hov_sp[h] = {"v": 0.0, "w": 0.0}
        dead = []
        for k, ent in self.hov_sp.items():
            tgt = 1.0 if k == h else 0.0
            ent["v"], ent["w"] = spring_step(ent["v"], ent["w"], tgt, 520.0, 19.0, dt)
            if tgt == 0.0 and abs(ent["v"]) < 0.003 and abs(ent["w"]) < 0.04:
                dead.append(k)
        for k in dead:
            del self.hov_sp[k]

    def update_sliders(self, dt):
        """Slider thumbs: the knob springs to its value, squishes while held, swells slightly on hover."""
        for key, _, lo, hi in SLIDERS:
            tgt = slider_norm(key, self.cfg[key])
            ent = self.sl.get(key)
            if ent is None:
                ent = self.sl[key] = dict(x=tgt, v=0.0, sc=1.0, sv=0.0)
            grabbing = self.drag == ("slider", key)
            hov = self.hover == ("slider", key)
            if grabbing:
                ent["x"], ent["v"] = spring_step(ent["x"], ent["v"], tgt, 1500.0, 80.0, dt)      # tight: follows the cursor
            else:
                ent["x"], ent["v"] = spring_step(ent["x"], ent["v"], tgt, 380.0, 26.0, dt)       # springy glide / click jumps
            if ent["x"] < 0.0 or ent["x"] > 1.0:
                ent["x"], ent["v"] = max(0.0, min(1.0, ent["x"])), 0.0
            ent["sc"], ent["sv"] = spring_step(ent["sc"], ent["sv"], 0.86 if grabbing else (1.07 if hov else 1.0), 520.0, 20.0, dt)
            if abs(ent["x"] - tgt) < 0.0004 and abs(ent["v"]) < 0.02:
                ent["x"], ent["v"] = tgt, 0.0
            tsc = 0.86 if grabbing else (1.07 if hov else 1.0)
            if abs(ent["sc"] - tsc) < 0.0008 and abs(ent["sv"]) < 0.02:
                ent["sc"], ent["sv"] = tsc, 0.0

    def set_slider(self, key, v):
        lo, hi = next((a, b) for k, _, a, b in SLIDERS if k == key)
        v = max(lo, min(hi, v))
        self.cfg[key] = v
        self.fix_pairs(key)
        if key == "volume":
            self.audio.set_volume(v)

    def set_player_volume(self, v):
        """The bar's volume slider: Spotify's own volume in Spotify mode, the normal volume otherwise."""
        v = max(0.0, min(1.0, v))
        if self.spotify_on:
            self.spot_vol, self.spot_vol_pend, self.spot_vol_t = v, v, time.perf_counter()
            self.spot_wake.set()
        else:
            self.set_slider("volume", v)

    def vol_to(self, lx):
        self.set_player_volume((lx - VOL_X0) / VOL_W)

    def seek_to(self, lx):
        if self.spotify_on:                                  # Spotify: through the Windows media session
            self.smtc.seek(max(0.0, min(1.0, (lx - SEEK_X0) / SEEK_W)) * (self.smtc.dur or 0.0))
            return
        a = self.audio
        if a.data is not None:
            t = max(0.0, min(1.0, (lx - SEEK_X0) / SEEK_W))
            a.pos = int(t * (len(a.data) - 1))
            a.finished = False

    def menu_motion(self, mx, my):
        if self.menu_p < 0.8:
            return
        lx, ly = self.to_logical(mx, my)
        if self.drag:
            if self.drag[0] == "slider":
                self.slider_value(self.drag[1], lx)
            elif self.drag[0] == "seek":
                self.seek_to(lx)
            elif self.drag[0] == "vol":
                self.vol_to(lx)
            elif self.drag[0] == "trim":
                self.trim_move(lx)
        self.hover = self.panel.pick(lx, ly)

    def scn_state(self):
        self.scn_pick_default()
        self.scn_clamp()
        sc = self.media.scene_state(self.scn_id, self.scn_scroll, SCN_ROWS)
        if sc.get("none"):
            self.scn_sel = None
            return sc
        it = sc["it"]
        rows = self.media.scene_rows(it)
        ts = [r[0] for r in rows]
        if not ts:
            self.scn_sel = None
        elif self.scn_sel is None or self.scn_sel[0] != it["id"]:
            self.scn_sel_i = 0
            self.scn_sel = (it["id"], ts[0])
        else:
            k = next((i for i, t in enumerate(ts) if abs(t - self.scn_sel[1]) < 0.01), None)
            if k is None:                                              # the selected scene was deleted: land on its neighbour
                k = max(0, min(self.scn_sel_i, len(ts) - 1))
                self.scn_sel = (it["id"], ts[k])
            self.scn_sel_i = k
        sel, tr = (self.scn_sel[1] if self.scn_sel else None), None
        if sel is not None:
            a, b = self.media.natural_range(it, sel)
            lo, hi = self.media.trim_domain(it, sel)
            td = self.trim_drag
            s_, e_ = (td["s"], td["e"]) if td else self.media.scene_range(it, sel)
            tr = dict(lo=lo, hi=hi, a=a, b=b, s=s_, e=e_, no=next((r[3] for r in rows if abs(r[0] - sel) < 0.01), 0),
                      has=self.media.trim_of(it, sel) is not None, grab=(td["which"] if td else None), pos=self.pv["pos"] if self.pv["head"] is not None else None)
        sc.update(sel=sel, trim=tr)
        sc["sig"] = sc["sig"] + (sel, (round(tr["s"], 2), round(tr["e"], 2), tr["has"], tr["grab"], round(tr["pos"] * 8) if tr["pos"] is not None else None) if tr else None)
        return sc

    def update_scene_preview(self, now):
        """Scenes tab: loop the selected scene (start to finish) in the preview; scrub while a trim handle is dragged."""
        pv = self.pv
        it = None
        if self.menu_open and self.tab == "scenes" and self.scn_sel is not None:
            it = next((x for x in self.scn_clips() if x["id"] == self.scn_sel[0]), None)
        if it is None:
            if pv["head"] is not None:
                pv["head"].stop()
                pv["head"], pv["key"], pv["pos"] = None, None, None
            return
        iid, t = self.scn_sel
        td = self.trim_drag
        s_, e_ = (td["s"], td["e"]) if td else self.media.scene_range(it, t)
        if td:
            want = td["s"] if td["which"] == "s" else max(td["s"], td["e"] - 0.05)
            if pv["head"] is None or pv["key"] is not None:
                if pv["head"] is not None:
                    pv["head"].stop()
                pv["head"], pv["key"], pv["scrub"] = it["src"].make_head(), None, -1.0
            if abs(want - pv["scrub"]) > 0.04:
                pv["head"].play_from(want, paused=True)
                pv["scrub"] = want
            pv["pos"] = want
        else:
            key = (iid, round(t, 2), round(s_, 2), round(e_, 2))
            if pv["key"] != key or pv["head"] is None:
                if pv["head"] is not None:
                    pv["head"].stop()
                h = it["src"].make_head()
                h.play_from(s_)
                pv.update(head=h, key=key, t0=now, scrub=-1.0)
            pos = s_ + (now - pv["t0"])
            if pos >= e_:
                pv["head"].play_from(s_)
                pv["t0"], pos = now, s_
            pv["pos"] = pos
        fr = pv["head"].get()
        if fr is not None and (id(pv["head"]), fr[0]) != pv["ser"]:
            f = np.ascontiguousarray(fr[1])
            hh, ww = f.shape[0], f.shape[1]
            if pv["tex"] is None or pv["tex"].size != (ww, hh):
                if pv["tex"] is not None:
                    pv["tex"].release()
                pv["tex"] = self.ctx.texture((ww, hh), 3, alignment=1)
                pv["tex"].filter = (moderngl.LINEAR, moderngl.LINEAR)
                pv["tex"].repeat_x = pv["tex"].repeat_y = False
            pv["tex"].write(f, alignment=1)
            pv["ser"] = (id(pv["head"]), fr[0])

    def trim_begin(self, lx):
        sel = self.scn_sel
        it = next((x for x in self.scn_clips() if x["id"] == sel[0]), None) if sel else None
        if it is None:
            return
        lo, hi = self.media.trim_domain(it, sel[1])
        s_, e_ = self.media.scene_range(it, sel[1])
        xs = TRK_X0 + (s_ - lo) / max(1e-6, hi - lo) * TRK_W
        xe = TRK_X0 + (e_ - lo) / max(1e-6, hi - lo) * TRK_W
        which = "s" if (abs(lx - xs) < abs(lx - xe) or (abs(abs(lx - xs) - abs(lx - xe)) < 1e-6 and lx < xs)) else "e"
        self.trim_drag = dict(iid=sel[0], t=sel[1], lo=lo, hi=hi, s=s_, e=e_, which=which)
        self.drag = ("trim",)
        self.trim_move(lx)

    def trim_move(self, lx):
        td = self.trim_drag
        it = next((x for x in self.scn_clips() if x["id"] == td["iid"]), None) if td else None
        if it is None:
            return
        lo, hi = td["lo"], td["hi"]
        v = lo + (lx - TRK_X0) / TRK_W * (hi - lo)
        for nv in self.media.natural_range(it, td["t"]):                   # snap to where the scene originally starts / ends
            if abs((nv - v) / max(1e-6, hi - lo) * TRK_W) < 5.0:
                v = nv
        v = max(lo, min(hi, v))
        if td["which"] == "s":
            td["s"] = max(lo, min(v, td["e"] - TRIM_MIN))
        else:
            td["e"] = min(hi, max(v, td["s"] + TRIM_MIN))

    def trim_commit(self, now):
        td, self.trim_drag = self.trim_drag, None
        if td:
            self.media.set_trim(td["iid"], td["t"], td["s"], td["e"], now)

    def state_for_panel(self):
        a = self.audio
        if self.tab == "visuals":
            self.ensure_style_previews()
        return dict(
            cfg=self.cfg,
            names=[os.path.splitext(os.path.basename(p))[0] for p in self.playlist],
            cur=self.cur,
            playing=(self.spot_playing() if self.spotify_on else a.playing),
            loading=a.loading,
            pos=a.pos / SR,
            total=(len(a.data) / SR) if a.data is not None else 0.0,
            is_fs=self.is_fs,
            fs_kind=self.cfg["fs_kind"],
            spotify=self.spotify_on,
            spot_title=self.spot_title,
            spot_track=self.spot_track,
            sp_tr=((self.sp_tr[1], (time.perf_counter() - self.sp_tr[0]) / 0.95) if (self.sp_tr and time.perf_counter() - self.sp_tr[0] < 0.95) else None),
            sp_pos=self.smtc.position(), sp_dur=self.smtc.dur,
            art_ver=self.art_ver, art=self.art_img, spot_vol=(round(self.spot_vol, 3) if self.spot_vol is not None else None),
            spot_running=self.spot_running,
            spot_error=self.spot_error,
            press={k: round(ent["v"], 3) for k, ent in self.press.items()},
            hov={k: round(ent["v"], 3) for k, ent in self.hov_sp.items()},
            sliders={k: (round(v["x"], 4), round(v["sc"], 3)) for k, v in self.sl.items()},
            drag_slider=(self.drag[1] if (self.drag and self.drag[0] == "slider") else None),
            drag_vol=bool(self.drag and self.drag[0] == "vol"),
            media=dict(self.media.info(), error=self.media.error),
            mscroll=self.mscroll,
            vscroll=round(self.vscroll_d, 1), aspect=round(self.vis_layout()["ph"] / PREV_W, 3),
            sec_a={k: round(v, 3) for k, v in self.sec_a.items()},
            scn=(self.scn_state() if self.tab == "scenes" else None), scn_scroll=self.scn_scroll,
            style_prev=self.style_prev,
            reset_armed=(time.perf_counter() - self.reset_t < 3.0),
            presets=[n for n, _ in self.presets], preset_sel=self.preset_sel, preset_open=self.preset_open,
            preset_del=(time.perf_counter() - self.preset_del_t < 3.0),
            beat=dict(bars=getattr(self, "vis_bars", (0,) * 48), kf=round(getattr(self, "kf", 0.0), 1), sf=round(getattr(self, "sf", 0.0), 1)),
        )

    def vis_layout(self):
        W, H = pygame.display.get_window_size()
        return visuals_layout(H / max(1, W), {k: round(v, 3) for k, v in self.sec_a.items()})

    def draw_preview(self, W, H, e, tab=None, fade=1.0, blur=0.0, tex_override=None):
        """Live picture drawn over the panel (clipped to the scrolling area): Visuals = the mirrored scene, Scenes = the looping clip."""
        x0, y0, w, h = self.panel_rect
        k = w / PW
        tab = tab or self.tab
        if tab == "visuals":
            lay = self.vis_layout()
            L = lay["preview"]
            if L["a"] < 0.01:
                return
            lx, ly, lw, lh, gf = preview_geom(lay, self.vscroll_d)
            tex, flip = (tex_override if tex_override is not None else self.final_t), 0
            if gf > 0.0:
                clip_top, clip_bot = HDR_Y, FTR_Y                              # docked: follows the scroll over the whole page
                fade = fade * min(1.0, L["a"] * 2.0)
            else:
                clip_top = L["hy"] + SEC_HDR_H - self.vscroll_d                 # only the open part of the section shows
                clip_bot = clip_top + L["vis"]
        else:
            tex, flip = self.pv["tex"], 1
            clip_top, clip_bot = HDR_Y, FTR_Y
            if tex is None:
                return
            bx, by, bw, bh = SCN_PV
            sc_ = min(bw / tex.size[0], bh / tex.size[1])
            lw, lh = tex.size[0] * sc_, tex.size[1] * sc_
            lx, ly = bx + (bw - lw) / 2, by + (bh - lh) / 2
        top, bot = y0 + max(HDR_Y, clip_top) * k, y0 + min(FTR_Y, clip_bot) * k
        rx, ry, rw, rh = x0 + lx * k, y0 + ly * k, lw * k, lh * k
        if bot <= top:
            return
        if ry + rh <= top or ry >= bot:
            return
        ctx = self.ctx
        ctx.scissor = (int(x0), int(H - bot), int(w), int(max(1, bot - top)))
        (cx, cy), (hx, hy) = ndc_rect(rx, ry, rw, rh, W, H)
        pr = self.prev_prog
        pr["uCenter"].value, pr["uHalf"].value = (cx, cy), (hx, hy)
        pr["uSize"].value = (float(rw), float(rh))
        pr["uRad"].value = 12.0 * k
        pr["uAlpha"].value = float(min(1.0, e ** 1.2) * fade)
        pr["uFlip"].value = flip
        pr["uBlur"].value = float(blur * w / max(1.0, self.panel_base[0]))
        tex.use(0)
        pr["uTex"].value = 0
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self.prev_vao.render(moderngl.TRIANGLE_STRIP)
        ctx.disable(moderngl.BLEND)
        ctx.scissor = None

    def scroll_visuals(self, dy):
        self.vscroll = max(0.0, min(float(self.vis_layout()["max_scroll"]), self.vscroll + dy))

    def draw_menu(self, W, H, e):
        """Composite pass: blurred backdrop + frosted glass panel, then the panel content on top."""
        for m_, v_ in self.card_mix.items():
            if v_ > 0.01:
                self.render_style_demo(m_)
        st = self.state_for_panel()
        sig = (self.tab, self.hover, self.scroll, tuple(self.playlist), self.cur, int(st["pos"]),
               st["playing"], st["loading"], self.is_fs, W, H, tuple(sorted(self.cfg.items())), st["spot_title"], st["spot_error"], st["spotify"], st["art_ver"], st["spot_vol"], st["spot_track"], ((st["sp_tr"][0], int(st["sp_tr"][1] * 28)) if st["sp_tr"] else None), (int((st["sp_pos"] / st["sp_dur"]) * 432) if st["sp_dur"] else None), tuple(st["press"].items()), tuple(st["hov"].items()), tuple(sorted(st["sliders"].items())), st["drag_slider"],
               (tuple(n for n, _ in self.presets), self.preset_sel, self.preset_open, time.perf_counter() - self.preset_del_t < 3.0),
               (st["media"]["sig"], st["media"]["loading"], st["media"]["error"], self.mscroll, st["reset_armed"], st["style_prev"] is not None),
               (st["beat"]["bars"], st["beat"]["kf"], st["beat"]["sf"]) if self.tab == "beat" else None,
               st["scn"]["sig"] if st["scn"] else None, (st["vscroll"], st["aspect"], tuple(st["sec_a"].values()), st["cfg"]["vis_open"]) if self.tab == "visuals" else None)
        if self.tab != self.tab_seen:                      # tab switch: keep the old picture to blur-fade out of
            if self.menu_p > 0.9 and self.panel_tex is not None and self.panel_sig is not None and self.panel_tex.size == (self.panel_base[0], self.panel_base[1]):
                if self.old_tex is not None:
                    self.old_tex.release()
                self.old_tex, self.panel_tex = self.panel_tex, None
                self.tab_old, self.tab_k = self.tab_seen, 0.0
            self.tab_seen = self.tab
        if sig != self.panel_sig:
            surf, sc = self.panel.render(st, W, H, self.hover, self.tab, self.scroll)
            self.panel_tex = surface_to_texture(self.ctx, surf, self.panel_tex)
            self.panel_scale = sc
            self.panel_base = surf.get_size()
            self.panel_sig = sig
        pw, ph = self.panel_base
        za = 0.90 + 0.10 * e                              # zoom in on open, zoom out on close
        w, h = pw * za, ph * za
        x, y = (W - w) / 2, (H - h) / 2
        self.panel_rect = (x, y, w, h)

        self.screen_fbo.use()
        self.ctx.viewport = (0, 0, W, H)
        self.final_t.use(0)
        self.a1.use(1)
        self.c1.use(2)
        cp = self.comp_prog
        cp["uScene"].value = 0
        cp["uA"].value = 1
        cp["uC"].value = 2
        cp["uRes"].value = (float(W), float(H))
        cp["uT"].value = float(e)
        cp["uRectC"].value = (x + w / 2, y + h / 2)
        cp["uRectH"].value = (w / 2, h / 2)
        cp["uRad"].value = 30.0 * self.panel_scale * za
        self.comp_vao.render(moderngl.TRIANGLE_STRIP)
        k = self.tab_k
        if k < 1.0 and self.old_tex is not None and self.old_tex.size == self.panel_tex.size:
            self.draw_tab_fade(W, H, e, k)
        else:
            if self.old_tex is not None:
                self.old_tex.release()
                self.old_tex = None
            ob = 0.0
            if e < 0.999:                                   # opening / closing: the panel sharpens in from a blur and blurs away again
                ob = 16.0 * self.panel_scale * (1.0 - e) ** 1.4
                self.ctx.enable(moderngl.BLEND)
                self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
                self.draw_blurred(self.panel_tex, e ** 1.2, ob)
                self.ctx.disable(moderngl.BLEND)
            else:
                self.quads.draw(self.panel_tex, self.panel_rect, (W, H), e ** 1.2)
            if self.tab in ("visuals", "scenes"):
                self.draw_preview(W, H, e, blur=ob)
            if self.tab == "visuals":
                for m_, v_ in self.card_mix.items():
                    if v_ > 0.01 and m_ in self.card_tex:
                        self.draw_card_demo(W, H, e, m_, v_)

    def style_demo_target(self):
        """The style card under the mouse (Visuals tab, menu fully open), else None."""
        h = self.hover
        if self.menu_open and self.tab == "visuals" and self.menu_p > 0.9 and h and h[0] == "style" and self.tab_k >= 1.0:
            return h[1]
        return None

    def update_style_demo(self, dt):
        """Hover a style card and its picture plays that style on a looping fake 120 bpm beat (each card blur-fades in / out)."""
        want = self.style_demo_target()
        if want is not None and self.demo is None:
            self.demo = dict(t=0.0, rot=0.0, flow=0.0, hue=0.0, psy=0.0, wave=0.0, zk=0.0, kenv=0.0, penv=0.0, spin=0.0, bass=0.12, beat=-1)
        if want is not None:
            self.card_mix.setdefault(want, 0.0)
        for m_ in list(self.card_mix):
            v_ = self.card_mix[m_] + ((1.0 if m_ == want else 0.0) - self.card_mix[m_]) * (1 - math.exp(-dt * 3.6))
            if m_ != want and v_ < 0.01:
                del self.card_mix[m_]
            else:
                self.card_mix[m_] = v_
        if not self.card_mix:
            self.demo = None
            return
        d, cfg = self.demo, self.cfg
        if d is None:
            return
        d["t"] += dt
        beat_i = int(d["t"] / 0.5)
        ph = (d["t"] % 0.5) / 0.5
        kick = math.exp(-ph * 7.0)
        if beat_i != d["beat"]:
            d["beat"] = beat_i
            d["spin"] += 1.5
        d["spin"] *= math.exp(-dt * 3.6)
        bass = 0.12 + 0.75 * kick
        d["bass"] += (bass - d["bass"]) * (1 - math.exp(-dt * 30.0))
        mid, high, energy = 0.30 + 0.15 * math.sin(d["t"] * 1.7), 0.20 + 0.10 * math.sin(d["t"] * 2.9 + 1.0), 0.30 + 0.35 * kick
        d["rot"] += cfg["spin"] * (0.22 + d["spin"] + 0.9 * mid) * dt
        d["flow"] += dt * (0.16 + 0.9 * mid + 0.5 * high)
        d["hue"] += dt * (0.015 + 0.10 * high) * cfg["color"]
        pe = min(1.0, d["bass"] * 1.2 + 0.8 * kick)
        d["penv"] += (pe - d["penv"]) * (1 - math.exp(-dt * (14.0 if pe > d["penv"] else 2.2)))
        d["psy"] = (d["psy"] + dt * (1.0 + 4.5 * d["penv"] + 0.8 * mid)) % TIME_WRAP
        d["wave"] = (d["wave"] + dt * (0.7 + 1.8 * energy) * (0.5 + 0.5 * min(cfg["color"], 2.0))) % TAU_F
        d["kenv"] += (pe - d["kenv"]) * (1 - math.exp(-dt * (12.0 if pe > d["kenv"] else 2.0)))
        d["zk"] = (d["zk"] + dt * cfg["kzoom"] * (0.36 + 1.5 * d["kenv"])) % FLOWK_PERIOD
        d["mid"], d["high"], d["energy"] = mid, high, energy

    def render_style_demo(self, m):
        """Draw style m into its own small target (same shader as the real scene)."""
        d, cfg, ctx = self.demo, self.cfg, self.ctx
        if d is None or "mid" not in d:
            return
        w, h = 339, 189                                          # 3x the card picture
        if m not in self.card_tex:
            t_ = ctx.texture((w, h), 4)
            t_.filter = (moderngl.LINEAR, moderngl.LINEAR)
            t_.repeat_x = t_.repeat_y = False
            self.card_tex[m] = (t_, ctx.framebuffer(color_attachments=[t_]))
        self.card_tex[m][1].use()
        ctx.viewport = (0, 0, w, h)
        vid = m == VIDEO_MODE
        mdl = self.media
        has = bool(mdl.items) and cfg["media_on"]
        for name, val in (
            ("uRes", (float(w), float(h))), ("uRot", d["rot"] % TAU_F), ("uFlow", d["flow"] % 2.0), ("uFlowK", d["zk"]),
            ("uZoom", d["bass"] * 0.075 * cfg["zoom"]), ("uAb", 0.0025 + d["bass"] * 0.040 * cfg["ab"]), ("uBass", float(d["bass"])),
            ("uMid", float(d["mid"])), ("uHigh", float(d["high"])), ("uEnergy", float(d["energy"])), ("uHue", d["hue"] % 4.0),
            ("uPsyT", float(d["psy"])), ("uHueWave", (min(1.0, cfg["color"] / 3.0), float(d["wave"]))), ("uTime", d["t"] + 7.0),
            ("uWarp", 0.02 + 0.13 * d["mid"] + 0.05 * d["high"]), ("uDim", (1.0 if has else 0.0) if vid else 1.0),
            ("uMode", 0 if vid else m), ("uMOn", 0.0),
        ):
            if name in self.prog:
                self.prog[name].value = val
        if vid and has:
            mdl.bind(self.prog, cfg["media_peak"], 0.0, 3, 1.0, (0.0, 0.0, 0.0, 0.0), solo=True)
        self.vao.render(moderngl.TRIANGLE_STRIP)
        self.screen_fbo.use()

    def draw_card_demo(self, W, H, e, m, mix):
        """Style card m's picture plays the live demo, blur-fading in / out (drawn over the panel, clipped to the open part of the section)."""
        lay = self.vis_layout()
        L = lay["style"]
        if L["a"] < 0.01:
            return
        x0, y0, w, h = self.panel_rect
        k = w / PW
        nm = len(MODES)
        cw = (792 - 8 * (nm - 1)) / nm
        iw = int(cw - 12)
        ih = int(iw * 9 / 16)
        card_h = ih + 12 + 26
        key = ("style", m)
        pp = (self.press.get(key) or {}).get("v", 0.0)
        hv = (self.hov_sp.get(key) or {}).get("v", 0.0)
        sc = (1.0 - 0.07 * pp) * (1.0 + 0.05 * hv)
        cx = 24 + m * (cw + 8) + cw / 2
        cy = L["ct"] - self.vscroll_d + card_h / 2
        lx, ly = cx - cw * sc / 2 + 6 * sc, cy - card_h * sc / 2 + 6 * sc
        lw, lh = iw * sc, ih * sc
        clip_top = max(HDR_Y, L["hy"] + SEC_HDR_H - self.vscroll_d)
        clip_bot = min(FTR_Y, L["hy"] + SEC_HDR_H - self.vscroll_d + L["vis"])
        top, bot = y0 + clip_top * k, y0 + clip_bot * k
        if bot <= top:
            return
        ctx = self.ctx
        ctx.scissor = (int(x0), int(H - bot), int(w), int(max(1, bot - top)))
        (ncx, ncy), (hx, hy) = ndc_rect(x0 + lx * k, y0 + ly * k, lw * k, lh * k, W, H)
        pr = self.prev_prog
        pr["uCenter"].value, pr["uHalf"].value = (ncx, ncy), (hx, hy)
        pr["uSize"].value = (float(lw * k), float(lh * k))
        pr["uRad"].value = 10.0 * k * sc
        me = mix * mix * (3.0 - 2.0 * mix)
        pr["uAlpha"].value = float(min(1.0, e ** 1.2) * min(1.0, me * 1.4))
        pr["uFlip"].value = 0
        pr["uBlur"].value = float((1.0 - me) * 8.0 * k)             # arrives blurred and sharpens; leaves by blurring away again
        self.card_tex[m][0].use(0)
        pr["uTex"].value = 0
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self.prev_vao.render(moderngl.TRIANGLE_STRIP)
        ctx.disable(moderngl.BLEND)
        ctx.scissor = None

    def draw_blurred(self, tex, alpha, radius):
        """The panel picture, blurred by `radius` texture pixels, at `alpha` (needs the blend state set by the caller)."""
        x, y, w, h = self.panel_rect
        W, H = pygame.display.get_window_size()          # not screen_fbo.size: that one keeps the size from startup
        (cx, cy), (hx, hy) = ndc_rect(x, y, w, h, W, H)
        pr = self.pblur_prog
        pr["uCenter"].value, pr["uHalf"].value = (cx, cy), (hx, hy)
        pr["uPx"].value = (radius / tex.size[0], radius / tex.size[1])
        pr["uAlpha"].value = float(max(0.0, min(1.0, alpha)))
        tex.use(0)
        pr["uTex"].value = 0
        self.pblur_vao.render(moderngl.TRIANGLE_STRIP)

    def draw_tab_fade(self, W, H, e, k):
        """Tab switch: the old page blurs and fades away while the new one sharpens in; header and footer stay put."""
        ctx = self.ctx
        x, y, w, h = self.panel_rect
        kk = w / PW
        yh, yf = y + HDR_Y * kk, y + FTR_Y * kk
        ss = lambda a, b, v: max(0.0, min(1.0, (v - a) / (b - a))) ** 2 * (3 - 2 * max(0.0, min(1.0, (v - a) / (b - a))))
        base = e ** 1.2
        for (sy0, sy1) in ((y, yh), (yf, y + h)):                            # header + footer: the new picture right away
            ctx.scissor = (int(x) - 1, int(H - sy1) - 1, int(w) + 3, int(sy1 - sy0) + 3)
            self.quads.draw(self.panel_tex, self.panel_rect, (W, H), base)
        ctx.scissor = (int(x) - 1, int(H - yf), int(w) + 3, int(yf - yh) + 1)
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        rmax = 11.0 * self.panel_scale
        a_out, a_in = 1.0 - ss(0.05, 0.60, k), ss(0.25, 0.80, k)
        self.draw_blurred(self.old_tex, a_out * base, rmax * ss(0.0, 0.75, k))
        self.draw_blurred(self.panel_tex, a_in * base, rmax * (1.0 - ss(0.25, 1.0, k)))
        ctx.disable(moderngl.BLEND)
        ctx.scissor = None
        if self.tab_old in ("visuals", "scenes") and a_out > 0.01:
            self.draw_preview(W, H, e, self.tab_old, a_out, rmax * ss(0.0, 0.75, k))
        if self.tab in ("visuals", "scenes") and a_in > 0.01:
            self.draw_preview(W, H, e, self.tab, a_in, rmax * (1.0 - ss(0.25, 1.0, k)))

    # ---------------------------------------------------------------- input
    def handle_key(self, k, now):
        a, cfg = self.audio, self.cfg
        if k == pygame.K_SPACE:
            self.play_pause()
        elif k == pygame.K_ESCAPE:
            self.open_menu(not self.menu_open)
        elif k in (pygame.K_f, pygame.K_F11):
            self.toggle_fullscreen()
        elif k == pygame.K_q:
            self.running = False
        elif k in (pygame.K_m, pygame.K_TAB):
            cfg["mode"] = (cfg["mode"] + 1) % len(MODES)
        elif k == pygame.K_d:
            cfg["domination"] = not cfg["domination"]
        elif k == pygame.K_h:
            cfg["help"] = not cfg["help"]
        elif k == pygame.K_v:
            self.media.request_change()
        elif k in (pygame.K_x, pygame.K_DELETE) and not self.menu_open:
            self.toast_msg(self.media.quick_flag("delete" if k == pygame.K_DELETE else "hide", now))
        elif k == pygame.K_z and not self.menu_open:
            self.toast_msg(self.media.undo_flag(now))
        elif k == pygame.K_s:
            self.set_spotify(not self.spotify_on)
        elif k == pygame.K_n:
            self.next_track()
        elif k == pygame.K_p:
            self.prev_track()
        elif k == pygame.K_RIGHT and not self.spotify_on:
            a.seek(5)
        elif k == pygame.K_LEFT and not self.spotify_on:
            a.seek(-5)
        elif k in (pygame.K_UP, pygame.K_DOWN):
            self.change_volume(0.05 if k == pygame.K_UP else -0.05)

    def change_volume(self, d):
        self.audio.set_volume(self.audio.volume + d)
        self.cfg["volume"] = self.audio.volume
        self.toast_msg(f"Volume {int(self.audio.volume * 100)}%")

    def events(self, now):
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                self.running = False
            elif e.type == pygame.DROPFILE:
                self.add_files([e.file])
            elif e.type == pygame.MOUSEMOTION:
                self.last_mouse = now
                self.mouse = e.pos
                pygame.mouse.set_visible(True)
                if self.menu_open:
                    self.menu_motion(*e.pos)
            elif e.type == pygame.MOUSEWHEEL:
                if self.menu_open:
                    if self.hover == ("vol",):
                        self.set_player_volume((self.spot_vol if (self.spotify_on and self.spot_vol is not None) else (1.0 if self.spotify_on else self.cfg["volume"])) + 0.05 * e.y)
                    elif self.tab == "queue":
                        mx_scroll = max(0, len(self.playlist) - LIST_ROWS)
                        self.scroll = max(0, min(mx_scroll, self.scroll - e.y))
                    elif self.tab == "visuals" and self.hover and (self.hover == ("noop", "mlist") or self.hover[0] in ("mrow", "mrow_remove")) \
                            and len(self.media.items) > STACK_ROWS:
                        self.mscroll = max(0, min(max(0, len(self.media.items) - STACK_ROWS), self.mscroll - e.y))
                    elif self.tab == "scenes":
                        self.scn_scroll -= e.y * 2
                        self.scn_clamp()
                    elif self.tab == "visuals" and not (pygame.key.get_mods() & pygame.KMOD_CTRL and self.hover and self.hover[0] in ("slider", "sl_reset")):
                        self.scroll_visuals(-e.y * 70.0)
                    elif self.tab in ("settings", "visuals", "media", "beat") and self.hover and self.hover[0] in ("slider", "sl_reset"):
                        key = self.hover[1]
                        lo, hi = next((a, b) for k, _, a, b in SLIDERS if k == key)
                        self.set_slider(key, self.cfg[key] + (hi - lo) * 0.02 * e.y)
                else:
                    self.change_volume(0.05 * e.y)
            elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                if self.menu_open:
                    self.menu_press(*e.pos)
                elif getattr(e, "clicks", 1) == 2:
                    self.toggle_fullscreen()
            elif e.type == pygame.MOUSEBUTTONUP and e.button == 1:
                self.drag = None
                for ent in self.press.values():
                    ent["held"] = False
            elif e.type == pygame.KEYDOWN:
                self.held[e.key] = [now, now]
                self.handle_key(e.key, now)
            elif e.type == pygame.KEYUP:
                self.held.pop(e.key, None)
        # manual key-repeat for seek / volume only (Space etc. must not repeat)
        for k, (t0, tl) in list(self.held.items()):
            if k in (pygame.K_LEFT, pygame.K_RIGHT, pygame.K_UP, pygame.K_DOWN) and now - t0 > 0.3 and now - tl > 0.07:
                self.held[k][1] = now
                self.handle_key(k, now)

    # ---------------------------------------------------------------- main loop
    def run(self, argv):
        if argv:
            self.add_files(argv)
        clock = pygame.time.Clock()
        last = time.perf_counter()
        cfg = self.cfg
        saved = [p for p in (cfg.get("media_paths") or "").split("|") if p] or ([cfg["media_path"]] if cfg.get("media_path") else [])
        saved = [p for p in saved if os.path.isfile(p)] if cfg["media_remember"] else []
        if saved:
            self.media.add(saved)
        self.sync_media_cfg()

        while self.running:
            now = time.perf_counter()
            dt = min(now - last, 0.1)
            last = now
            a = self.audio
            ana = self.ana_spot if (self.spotify_on and self.ana_spot is not None) else self.ana

            self.events(now)
            if self.exit_at is not None and now >= self.exit_at:
                self.running = False
            if self.picked:
                files, self.picked[:] = list(self.picked), []
                self.add_files(files)
            if self.media_picked:
                mp, self.media_picked[:] = list(self.media_picked), []
                self.add_media(mp)

            started = a.poll()
            if started:
                name = os.path.splitext(os.path.basename(started))[0]
                pygame.display.set_caption(f"Hypnosis  -  {name}")
                self.toast_msg(name)
                self.finish_handled = False
            if not a.finished:
                self.finish_handled = False
            if a.finished and not a.loading and not self.finish_handled:
                self.finish_handled = True
                n = self.next_index()
                if n is not None:
                    self.play_index(n)
            if a.error:
                self.toast_msg(a.error)
                a.error = None

            spot = self.spotify_on
            cap = self.cap if spot else None
            if spot:
                t = self.spot_title
                if t != self.spot_last:
                    self.spot_last = t
                    if t and not t.lower().startswith("spotify"):
                        self.toast_msg(t)
                        pygame.display.set_caption(f"Hypnosis  -  {t}")
                chunk = cap.window(ana.n, cfg["spot_delay"] / 1000.0) if cap else None
            else:
                chunk = a.window(ana.n) if a.playing or a.gain > 0.01 else None
            for an in (self.ana, self.ana_spot):
                if an is not None:
                    an.set_ranges(cfg["kick_lo"], cfg["kick_hi"], cfg["snare_lo"], cfg["snare_hi"], cfg["kick_sens"], cfg["snare_sens"])
                    an.want_vis = bool(self.menu_open and self.tab == "beat")
            ana.step(chunk, dt, now)
            self.vis_bars = ana.vis_bars
            self.kf = max(getattr(self, "kf", 0.0) * math.exp(-dt * 7.0), ana.kick)
            self.sf = max(getattr(self, "sf", 0.0) * math.exp(-dt * 7.0), ana.snare)

            # ---- media layer (video / GIF / image)
            mdl = self.media
            got = mdl.poll(now)
            if got:
                self.toast_msg(got)
            if mdl.error:
                self.toast_msg("Couldn't open media: " + mdl.error)
                mdl.error = None
                self.sync_media_cfg()

            # ---- motion ----
            idle = False if spot else a.data is None
            playing = bool(cap and cap.active) if spot else a.playing
            vid = cfg["mode"] == VIDEO_MODE                          # Video style: only the footage, never idles with the spiral
            rms = 0.0 if chunk is None else float(np.sqrt(np.mean(np.square(chunk))))
            self.quiet_t = 0.0 if rms > 0.0015 else getattr(self, "quiet_t", 0.0) + dt
            live = playing and self.quiet_t < 0.8                    # audio is actually coming in (not paused / stopped / silent)
            dom = cfg["domination"]
            sm = cfg["beat_smooth"]                                  # 0 = raw & snappy ... 1 = very soft / flowing
            s_bass = self.smooth("bass", ana.bass, dt, sm)
            s_mid = self.smooth("mid", ana.mid, dt, sm)
            s_high = self.smooth("high", ana.high, dt, sm)
            s_energy = self.smooth("energy", ana.energy, dt, sm)
            s_onset = self.smooth("onset", ana.onset, dt, sm)
            if ana.beat > 0:
                self.spin_kick += 2.6 * ana.beat * (1.0 - 0.5 * sm)
            self.spin_kick *= math.exp(-dt * 3.6 / (1.0 + 1.5 * sm))
            base = 0.30 if idle else 0.22
            omega = cfg["spin"] * (base + self.spin_kick + 0.9 * s_mid + 0.5 * s_onset)   # spin slider at 0% = no rotation at all
            if not playing and not idle:
                omega = 0.0
            omega = self.smooth("omega", omega, dt, sm * 0.8)
            self.rot += omega * dt
            self.flow += dt * ((0.10 if idle else 0.16) + (0.9 * s_mid + 0.5 * s_high) * (1 if playing else 0))
            self.hue += dt * (0.015 + 0.10 * s_high) * cfg["color"]
            # Psychedelic flow clock: always forward; kicks / snares / bass make it surge faster, then it glides back to a calm drift
            pe = min(1.0, s_bass * 1.2 + 0.7 * ana.beat + 0.9 * max(ana.kick, ana.snare)) if playing else 0.0
            self.psy_env = getattr(self, "psy_env", 0.0)
            self.psy_env += (pe - self.psy_env) * (1 - math.exp(-dt * ((14.0 / (1.0 + 3.0 * sm)) if pe > self.psy_env else 2.2 / (1.0 + 1.5 * sm))))
            self.psy_t = (getattr(self, "psy_t", 0.0) + dt * (1.0 + 4.5 * self.psy_env + 0.8 * s_mid)) % TIME_WRAP
            self.wave_ph = (getattr(self, "wave_ph", 0.0) + dt * (0.7 + 1.8 * s_energy) * (0.5 + 0.5 * min(cfg["color"], 2.0))) % TAU_F
            # Kaleidoscope: slow endless zoom. Smoothed speed -> never jerks; freezes (eased) while paused.
            self.zk_speed += ((0.0 if (not playing and not idle) else 1.0) - self.zk_speed) * (1 - math.exp(-dt * 3.0))
            boost = (0.20 * s_mid + 0.10 * s_high + 0.06 * s_energy) if playing else 0.0
            self.zk_boost += (boost - self.zk_boost) * (1 - math.exp(-dt * 2.5))
            # tunnel flight: cruising speed, surging forward on kicks / snares / bass, then easing back (never reverses)
            ke = min(1.0, s_bass * 1.2 + 0.7 * ana.beat + 0.9 * max(ana.kick, ana.snare)) if playing else 0.0
            self.k_env = getattr(self, "k_env", 0.0)
            self.k_env += (ke - self.k_env) * (1 - math.exp(-dt * ((12.0 / (1.0 + 3.0 * sm)) if ke > self.k_env else 2.0 / (1.0 + 1.5 * sm))))
            self.zoomk = (self.zoomk + dt * self.zk_speed * cfg["kzoom"] * (0.36 + 1.5 * self.k_env + self.zk_boost)) % FLOWK_PERIOD
            zoom = s_bass * 0.075 * cfg["zoom"]                 # bass zoom + chromatic aberration stay on in
            ab = 0.0025 + s_bass * 0.040 * cfg["ab"]             # Domination Mode; the popups are added on top
            ubass = s_bass
            if cfg["mode"] == 3:                                 # Kaleidoscope: the beat drives the tunnel speed, no zoom-punch
                zoom = 0.0
            if cfg["mode"] == 4:                                 # Psychedelic: no zoom-punch / bloom, the beat speeds the flow instead
                zoom, ubass = 0.0, 0.0
            mdl.on_beat(ana.beat, now, playing, cfg["media_rate"])
            mdl.set_style(cfg["media_style"])
            mdl.interp = cfg["media_smooth"]
            if cfg["media_hit"] and (ana.kick > 0 or ana.snare > 0):
                mdl.on_hit(max(ana.kick, ana.snare), now, playing, cfg["hit_cool"])
            env_t = min(1.0, s_bass * 1.15 + 0.5 * ana.beat) if playing else 0.0
            self.m_env = getattr(self, "m_env", 0.0)
            self.m_env += (env_t - self.m_env) * (1 - math.exp(-dt * ((20.0 / (1.0 + 4.0 * sm)) if env_t > self.m_env else 4.5 / (1.0 + sm))))
            hit = max(ana.kick, ana.snare)                                  # kick / snare -> short speed burst
            # ---- kick / snare brightness flash: dark between hits, swells smoothly to the chosen level on each hit
            fl = cfg["media_flash"]
            self.fl_hold = getattr(self, "fl_hold", 0.0)
            self.fl_v = getattr(self, "fl_v", 0.0)
            self.fl_pm = getattr(self, "fl_pm", 1.0)
            if hit > 0 and playing:
                self.fl_hold = 0.08 + 0.14 * sm
                self.fl_peak = min(1.0, 0.6 + 0.5 * hit)
            ft = getattr(self, "fl_peak", 1.0) if self.fl_hold > 0 else 0.0
            self.fl_hold = max(0.0, self.fl_hold - dt)
            tau = (0.03 * (1.0 + 3.0 * sm)) if ft > self.fl_v else (0.20 * (1.0 + 1.5 * sm))
            self.fl_v += (ft - self.fl_v) * (1.0 - math.exp(-dt / tau))
            self.fl_pm += ((0.0 if playing else 1.0) - self.fl_pm) * (1.0 - math.exp(-dt / 0.4))   # paused / idle: show the video normally
            flash_gain = 1.0 if fl <= 0.001 else (1.0 - self.fl_pm) * fl * self.fl_v + self.fl_pm
            # ---- camera sway: each hit/beat swings the picture to a new spot, springs there and drifts back
            sw_on = 1.0 if (cfg["media_sway"] and playing) else 0.0
            sw = getattr(self, "sw", None)
            if sw is None:
                sw = self.sw = {"x": 0.0, "y": 0.0, "r": 0.0, "z": 0.0, "vx": 0.0, "vy": 0.0, "vr": 0.0, "vz": 0.0,
                                "tx": 0.0, "ty": 0.0, "tr": 0.0, "tz": 0.0, "dir": 1.0, "cd": 0.0, "pb": 0.0, "amt": 0.0, "ph": 0.0}
            sw["cd"] = max(0.0, sw["cd"] - dt)
            beat_edge = ana.beat > 0.6 and sw["pb"] <= 0.6
            sw["pb"] = ana.beat
            trig = max(hit, 0.7 if beat_edge else 0.0)
            if trig > 0 and playing and sw["cd"] <= 0.0:
                sw["cd"] = 0.16
                sw["dir"] = -sw["dir"]
                m = 0.45 + 0.55 * min(1.0, trig)
                sw["tx"] = sw["dir"] * 0.030 * m
                sw["ty"] = random.uniform(-0.016, 0.016) * m
                sw["tr"] = -sw["dir"] * 0.022 * m
                sw["tz"] = 0.035 * m
            relax = math.exp(-dt / (0.35 * (1.0 + 1.5 * sm)))             # target eases back to centre between hits
            for k in ("tx", "ty", "tr", "tz"):
                sw[k] *= relax
            sw["ph"] += dt * (0.7 + 1.6 * self.m_env)
            sway_idle = 0.35 + 0.65 * self.m_env                              # slow handheld wander, livelier with the bass
            wx = math.sin(sw["ph"]) * 0.006 * sway_idle
            wy = math.sin(sw["ph"] * 0.73 + 1.3) * 0.005 * sway_idle
            wr = math.sin(sw["ph"] * 0.51 + 2.1) * 0.007 * sway_idle
            sw["amt"] += (sw_on - sw["amt"]) * (1.0 - math.exp(-dt / 0.25))
            for p_, v_, t_ in (("x", "vx", sw["tx"] + wx), ("y", "vy", sw["ty"] + wy), ("r", "vr", sw["tr"] + wr), ("z", "vz", sw["tz"])):
                w0 = 16.0 / (1.0 + 1.6 * sm)                              # spring: soft overshoot, smoother with the smoothing slider
                a_ = w0 * w0 * (t_ - sw[p_]) - 2.0 * 0.7 * w0 * sw[v_]
                sw[v_] += a_ * min(dt, 0.05)
                sw[p_] += sw[v_] * min(dt, 0.05)
            g_ = cfg["media_sway_amt"] * sw["amt"]
            sway_v = (sw["x"] * g_, sw["y"] * g_, sw["r"] * g_, (sw["z"] + 0.012 * sway_idle) * g_)
            self.m_burst = getattr(self, "m_burst", 0.0)
            self.m_burst = max(self.m_burst * math.exp(-dt * 6.5 / (1.0 + 2.0 * sm)), hit) if playing else self.m_burst * math.exp(-dt * 6.5 / (1.0 + 2.0 * sm))
            S = cfg["media_speed"]
            vo = vid                                                       # Video style: calm parts crawl almost to a stop
            sp_calm, sp_peak = max(0.05, 1.0 - (0.85 if vo else 0.55) * S), 1.0 + 3.2 * S
            self.m_speed = getattr(self, "m_speed", 1.0)
            self.m_avg = getattr(self, "m_avg", 1.0)                       # ~1.5 s running average of the speed while audio plays
            if vid and not live:                                           # no audio: keep moving at the last known speed until it returns
                self.m_speed += (max(0.5, self.m_avg) - self.m_speed) * (1 - math.exp(-dt * 2.0))
            else:
                self.m_speed += ((sp_calm + (sp_peak - sp_calm) * self.m_burst) - self.m_speed) * (1 - math.exp(-dt * ((45.0 / (1.0 + 8.0 * sm)) if self.m_burst > 0.05 else 9.0)))
                if live and rms > 0.0015:
                    self.m_avg += (self.m_speed - self.m_avg) * (1 - math.exp(-dt / 1.5))
            mdl.set_speed(self.m_speed)
            mdl.update(now, dt, cfg["media_on"], (not playing and not idle) and not vid, cfg["media_rate"])
            warp = 0.02 + 0.13 * s_mid + 0.05 * s_high
            vid_blank = vid and (not mdl.items or not cfg["media_on"])        # Video style with nothing to show: plain black, no spiral
            target_dim = 0.0 if vid_blank else (1.0 if vid else (0.6 if idle else (1.0 if playing else 0.5)))
            self.dim += (target_dim - self.dim) * (1 - math.exp(-dt * 5))

            # ---- menu animation (0 = closed, 1 = open) ----
            W, H = pygame.display.get_window_size()
            rate = (1.0 / 0.42) if self.menu_open else -(1.0 / 0.32)
            self.menu_p = max(0.0, min(1.0, self.menu_p + rate * dt))
            p = self.menu_p
            e = p * p * p * (p * (6 * p - 15) + 10)            # smootherstep
            # audio low-pass follows the menu curve - except on the Beat tab, where you tune detection by ear (clean audio)
            want = 0.0 if self.tab == "beat" else 1.0
            if p <= 0.0:
                self.muf = want
            else:
                self.muf = getattr(self, "muf", want)
                self.muf += (want - self.muf) * (1 - math.exp(-dt * 10.0))
            a.muffle = e * self.muf
            rv = getattr(self, "reveal", None)
            if rv is not None:
                if rv not in set(filter(None, cfg.get("vis_open", "").split(","))) or self.sec_a.get(rv, 0.0) > 0.999 or self.tab != "visuals":
                    self.reveal = None
                else:
                    L = self.vis_layout()[rv]
                    want = min(L["hy"] - HDR_Y - 8, L["hy"] + SEC_HDR_H + L["vis"] - FTR_Y + 10)
                    self.vscroll = max(self.vscroll, min(want, float(self.vis_layout()["max_scroll"])))
            self.scroll_visuals(0.0)                           # keep the target valid after a window resize
            op = set(filter(None, cfg.get("vis_open", "").split(",")))
            for n_ in self.sec_a:                              # sections open / close with an eased height
                tgt = 1.0 if n_ in op else 0.0
                v_ = self.sec_a[n_] + (tgt - self.sec_a[n_]) * (1 - math.exp(-dt * 11.0))
                self.sec_a[n_] = tgt if abs(tgt - v_) < 0.004 else v_
            self.vscroll_d += (self.vscroll - self.vscroll_d) * (1 - math.exp(-dt * 14.0))
            if abs(self.vscroll - self.vscroll_d) < 0.05:
                self.vscroll_d = self.vscroll
            if self.menu_open and self.tab == "visuals" and self.vscroll_d != getattr(self, "_vs_seen", None):
                self._vs_seen = self.vscroll_d
                self.hover = self.panel.pick(*self.to_logical(*pygame.mouse.get_pos()))   # the page moved under a still mouse
            if self.tab_k < 1.0:
                self.tab_k = min(1.0, self.tab_k + dt / 0.36)
            self.update_style_demo(dt)
            menu_visible = p > 0.0

            # ---- render scene (offscreen only while the menu is visible) ----
            self.ensure_targets(W, H)
            if cfg["mode"] != self.mode_shown:                  # visual mode changed: freeze the old look
                src_f = self.trans_f if self.mode_t < 1.0 else self.scene_f
                self.ctx.copy_framebuffer(self.prev_f, src_f)
                self.mode_shown = cfg["mode"]
                self.mode_t = 0.0
            self.scene_f.use()
            self.ctx.viewport = (0, 0, W, H)
            for name, val in (
                ("uRes", (float(W), float(H))), ("uRot", self.rot % TAU_F), ("uFlow", self.flow % 2.0), ("uFlowK", self.zoomk % FLOWK_PERIOD),
                ("uZoom", zoom),
                ("uAb", ab), ("uBass", float(ubass)), ("uMid", float(s_mid)),
                ("uHigh", float(s_high)), ("uEnergy", float(s_energy)), ("uHue", self.hue % 4.0), ("uPsyT", float(self.psy_t)), ("uHueWave", (min(1.0, cfg["color"] / 3.0), float(self.wave_ph))),
                ("uTime", now % TIME_WRAP), ("uWarp", warp), ("uDim", self.dim), ("uMode", 0 if vid else cfg["mode"]),
            ):
                if name in self.prog:            # the GLSL compiler drops unused uniforms
                    self.prog[name].value = val
            amt = (cfg["media_calm"] + (cfg["media_peak"] - cfg["media_calm"]) * self.m_env) if cfg["media_auto"] else cfg["media_peak"]
            if vid and not live:
                amt = cfg["media_peak"]                                    # no audio: the footage stays at full brightness
            mdl.bind(self.prog, amt, s_bass * cfg["media_pulse"], 3 if vid else min(2, cfg["media_blend"]), flash_gain, sway_v, solo=vid)
            self.vao.render(moderngl.TRIANGLE_STRIP)

            # ---- domination popups (bass-triggered) ----
            if dom and playing:
                cd = 0.30 / max(0.05, cfg["dom_rate"])
                if ana.bass > 0.55 and self.dom_prev <= 0.55 and now - self.dom_last > cd:
                    self.dom_last = now
                    self.glitch.spawn(self.phrases, now, (W, H), cfg["dom_size"])
            self.dom_prev = ana.bass
            if not dom:
                self.glitch.active.clear()
            self.glitch.draw(now, (W, H))

            final_t = self.scene_t
            if self.mode_t < 1.0:                               # blur-fade between visual modes
                self.mode_t = min(1.0, self.mode_t + dt / 0.8)
                kk = self.mode_t * self.mode_t * self.mode_t * (self.mode_t * (6 * self.mode_t - 15) + 10)
                self.transition_pass(kk)
                final_t = self.trans_t
            self.final_t = final_t
            self.update_press(now, dt)
            self.update_hover(dt)
            self.update_sliders(dt)
            if self.trim_drag and not (self.drag and self.drag[0] == "trim"):
                self.trim_commit(now)
            self.update_scene_preview(now)

            if menu_visible:
                self.blur_scene(e)
                self.draw_menu(W, H, e)
            else:
                self.present(final_t, W, H)
                lines, alpha = [], 0.0
                if spot and now - self.toast_t < 3.2:
                    lines = [(self.toast[:90], 30)]
                    alpha = min(1.0, (3.2 - (now - self.toast_t)) * 1.5)
                elif spot and not playing:
                    msg = ("Play something in Spotify - listening to Spotify only" if cap
                           else "Waiting for Spotify to start...")
                    lines = [(msg, 30)]
                    alpha = 0.55 + 0.35 * math.sin(now * 1.6)
                elif spot:
                    pass
                elif vid_blank and now - self.toast_t >= 3.2:
                    lines, alpha = [("Add Media To Begin" if not mdl.items else "Media layer is off", 40)], 0.55 + 0.35 * math.sin(now * 1.6)
                elif vid and mdl.items and now - self.toast_t >= 3.2:
                    pass                                                  # Video style: just the footage, no idle / paused captions
                elif idle and not a.loading and now - self.toast_t >= 3.2:
                    lines = [("Drop a music file to begin", 40)]
                    if cfg["help"]:
                        lines.append(("Space  play / pause    Esc  menu    F  fullscreen    M  mode    D  domination    V  next scene", 20))
                    alpha = 0.55 + 0.35 * math.sin(now * 1.6)
                elif a.loading and idle:
                    lines, alpha = [("Loading...", 34)], 0.9
                elif now - self.toast_t < 3.2:
                    lines = [(self.toast[:90], 30)]
                    alpha = min(1.0, (3.2 - (now - self.toast_t)) * 1.5)
                elif not playing and not a.loading:
                    lines, alpha = [("PAUSED", 34)], 0.75
                self.overlay.draw(lines, (W, H), alpha)

            pygame.display.flip()

            if self.is_fs and not self.menu_open and now - self.last_mouse > 2.0:
                pygame.mouse.set_visible(False)
            clock.tick(240)

        self.media.clear()
        self.spot_wake.set()
        with self.cap_lock:
            if self.cap:
                self.cap.stop()
        save_settings(cfg)
        pygame.quit()


def main():
    App().run(sys.argv[1:])


if __name__ == "__main__":
    main()
