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
import queue
from fractions import Fraction
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
uniform vec2  uZoomC, uShake;     // zoom focus point (centre = 0,0) and a shake offset
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

// ================================================================ MORE STYLES (ids 6..17)
// Every time term is either sin / cos of (uPsyT or uTime * a multiple of 0.005) or a fract() driven by sp(): sp() rounds a speed so the
// clock wrap (TIME_WRAP) is a whole number of cycles -> nothing ever jumps. Spin comes from uRot (a full turn), music from uBass / uMid / uHigh.
const float TW = 1256.6370614;
float sp(float s){ return floor(s * TW + 0.5) / TW; }
vec2 rot2(vec2 p, float a){ float c = cos(a), s = sin(a); return mat2(c, -s, s, c) * p; }
float vnoise(vec2 p){
    vec2 i = floor(p), f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash(i), hash(i + vec2(1.0, 0.0)), f.x), mix(hash(i + vec2(0.0, 1.0)), hash(i + vec2(1.0, 1.0)), f.x), f.y);
}
float fbm(vec2 p){
    float v = 0.0, a = 0.5;
    for (int i = 0; i < 4; i++) { v += a * vnoise(p); p = mat2(1.6, 1.2, -1.2, 1.6) * p; a *= 0.5; }
    return v;
}

vec3 sTunnel(vec2 p, float px)                                   // 6: checkerboard tunnel rushing toward you
{
    float r = length(p) + 1e-4;
    float a = atan(p.y, p.x) + uRot;
    float z = 0.5 / r;
    float s1 = sin(a * 6.0);
    float s2 = sin(TAU * (z * 0.5 - uPsyT * sp(0.40)));
    float w = 0.04 + px * 8.0 / r;
    float m = smoothstep(-w, w, s1 * s2);
    vec3 c1 = pal(uHue + 0.06 * z) * 0.18;
    vec3 c2 = pal(uHue + 0.45 + 0.04 * z);
    vec3 col = mix(c1, c2, m);
    col += pal(uHue + 0.2) * (1.0 - smoothstep(0.0, 0.25, abs(s1 * s2))) * 0.25;
    col *= smoothstep(0.0, 0.4, r) * (0.9 + 0.5 * uBass);
    col += exp(-r * 6.0) * pal(uHue + 0.7) * (0.25 + 0.5 * uBass);
    return col;
}

vec3 sVortex(vec2 p, float px)                                   // 7: spiral galaxy with a glowing core and twinkling stars
{
    float r = length(p) + 1e-4;
    float a = atan(p.y, p.x) + uRot;
    float lr = log(r + 0.03);
    float arm = 0.5 + 0.5 * sin(3.0 * a - 5.0 * lr + uPsyT * 0.30);
    arm = pow(arm, 1.6 + 1.5 * r);
    float dust = fbm(rot2(p, uRot) * 5.0 + vec2(cos(uPsyT * 0.2), sin(uPsyT * 0.2)));
    float dens = arm * exp(-r * 1.25) * (0.55 + 0.9 * dust);
    vec3 col = pal(uHue + 0.55 - 0.22 * lr + 0.3 * arm) * dens * (1.6 + uBass);
    vec2 q = rot2(p, uRot * 2.0) * 22.0;
    vec2 cell = floor(q);
    float h = hash(cell);
    float star = step(0.93, h) * smoothstep(0.35, 0.0, length(fract(q) - 0.5)) * (0.5 + 0.5 * sin(uPsyT * 2.0 + h * 60.0));
    col += star * 0.9;
    col += exp(-r * 7.0) * pal(uHue + 0.1) * (1.1 + uBass * 1.2);
    return col;
}

vec3 sPlasma(vec2 p, float px)                                   // 8: liquid plasma
{
    vec2 q = rot2(p, uRot) * 2.4;
    float t = uPsyT;
    float v = sin(q.x * 1.3 + t * 0.70) + sin(q.y * 1.7 - t * 0.50) + sin((q.x + q.y) * 1.1 + t * 0.60)
            + sin(length(q + vec2(sin(t * 0.30), cos(t * 0.40)) * 1.5) * 2.2 - t * 0.90 + uBass * 3.0);
    vec3 col = pal(uHue + v * 0.14 + 0.12 * uMid);
    col *= 0.55 + 0.45 * sin(v * 2.0 + t * 0.2);
    float line = 1.0 - smoothstep(0.0, 0.12, abs(sin(v * 3.1415)));
    col = mix(col, vec3(1.0), line * 0.20);
    return col * (0.9 + 0.4 * uEnergy);
}

vec3 sRipples(vec2 p, float px)                                  // 9: water ripples from drifting sources
{
    float t = uPsyT;
    float sum = 0.0;
    for (int i = 0; i < 3; i++) {
        float fi = float(i);
        vec2 c = rot2(vec2(0.55, 0.0), uRot + fi * 2.0944) * (0.6 + 0.4 * sin(t * (0.20 + 0.05 * fi) + fi));
        float d = length(p - c);
        sum += sin(d * (22.0 + 4.0 * fi) - t * (2.0 + 0.5 * fi) - uBass * 2.0) / (1.0 + 3.5 * d);
    }
    vec3 col = pal(uHue + sum * 0.30 + 0.12 * length(p));
    col *= 0.30 + 0.85 * smoothstep(-0.6, 0.9, sum);
    col += smoothstep(0.85, 1.0, sin(sum * 6.0)) * 0.25;
    return col;
}

vec3 sSynth(vec2 p, float px)                                    // 10: retro sun over a neon grid
{
    float hz = 0.10;
    float y = p.y - hz;
    vec3 col;
    float pulse = 1.0 + 0.5 * uBass;
    if (y > 0.0) {
        float k = y / (1.0 - hz);
        col = mix(vec3(0.95, 0.20, 0.45), vec3(0.04, 0.00, 0.20), pow(k, 0.55));
        vec2 sc = vec2(0.0 + 0.0, hz + 0.34);
        float d = length(p - sc);
        float rad = 0.30 * (1.0 + 0.10 * uBass);
        float bands = smoothstep(0.0, 0.05, sin((p.y - sc.y) * 38.0 + 1.0) + 0.55 + 1.1 * (sc.y - p.y) / rad);
        float disc = smoothstep(rad, rad - 0.012, d) * mix(1.0, bands, step(p.y, sc.y));
        col = mix(col, mix(vec3(1.0, 0.90, 0.25), vec3(1.0, 0.15, 0.55), clamp((sc.y + rad - p.y) / (2.0 * rad), 0.0, 1.0)), disc);
        col += exp(-max(d - rad, 0.0) * 5.0) * vec3(1.0, 0.25, 0.6) * 0.35 * pulse;
        vec2 q = p * 40.0 + 7.0;
        col += step(0.985, hash(floor(q))) * smoothstep(0.4, 0.0, length(fract(q) - 0.5)) * k * 0.8;
    } else {
        float z = 0.42 / (-y + 0.002);
        float x = (p.x + 0.45 * sin(uRot)) * z;
        float gx = abs(fract(x * 0.9) - 0.5);
        float gz = abs(fract(z * 1.1 - uPsyT * sp(0.55)) - 0.5);
        float wx = 0.020 + px * z * 2.0, wz = 0.030 + px * z * 1.0;
        float line = max(1.0 - smoothstep(0.0, wx, gx), 1.0 - smoothstep(0.0, wz, gz));
        vec3 floorc = mix(vec3(0.20, 0.00, 0.30), vec3(0.02, 0.00, 0.08), clamp(-y, 0.0, 1.0));
        col = floorc + line * mix(vec3(1.0, 0.15, 0.75), vec3(0.1, 0.9, 1.0), clamp(z * 0.08, 0.0, 1.0)) * pulse * smoothstep(0.0, 0.12, -y);
        col += exp(y * 28.0) * vec3(1.0, 0.3, 0.7) * 0.35;
    }
    return col;
}

vec3 sWarp(vec2 p, float px)                                     // 11: hyperspace star streaks
{
    float r = length(p) + 1e-4;
    float a = atan(p.y, p.x) + uRot;
    vec3 col = vec3(0.0);
    for (int l = 0; l < 3; l++) {
        float fl = float(l);
        float N = 36.0 + 24.0 * fl;
        float sct = a * N / TAU;
        float id = mod(floor(sct), N);
        float f = fract(sct);
        float h = hash(vec2(id, fl + 3.0));
        float z = fract(h + uPsyT * sp(0.22 + 0.08 * fl));
        float head = z * z * 1.8;
        float len = 0.03 + 0.30 * z * z * (1.0 + 0.8 * uBass);
        float width = 0.0035 + 0.007 * z;
        float lateral = abs(f - 0.5) * TAU / N * r;
        float on = smoothstep(width + 0.003, width, lateral);
        float along = smoothstep(head - len, head, r) * step(r, head);
        col += pal(h + uHue) * on * along * (0.4 + 1.2 * z);
    }
    col += exp(-r * 5.0) * pal(uHue + 0.6) * (0.22 + 0.5 * uBass);
    return col;
}

vec3 sHoney(vec2 p, float px)                                    // 12: pulsing honeycomb
{
    vec2 q = rot2(p, uRot) * 4.2;
    const vec2 s = vec2(1.0, 1.7320508);
    vec4 hC = floor(vec4(q, q - vec2(0.5, 1.0)) / s.xyxy) + 0.5;
    vec4 h = vec4(q - hC.xy * s, q - (hC.zw + 0.5) * s);
    vec2 g, id;
    if (dot(h.xy, h.xy) < dot(h.zw, h.zw)) { g = h.xy; id = hC.xy; } else { g = h.zw; id = hC.zw + 0.5; }
    vec2 ag = abs(g);
    float d = max(dot(ag, vec2(0.5, 0.8660254)), ag.x);
    float edge = 0.5 - d;
    float hh = hash(id);
    float wave = 0.5 + 0.5 * sin(length(id) * 0.9 - uPsyT * 2.0 + hh * 1.5 - uBass * 3.0);
    vec3 fill = pal(uHue + 0.05 * length(id) + 0.25 * hh) * (0.08 + 0.95 * wave * wave);
    float rim = 1.0 - smoothstep(0.0, 0.06 + px * 6.0, edge);
    vec3 col = fill + rim * pal(uHue + 0.5) * (0.5 + 0.8 * wave);
    col *= 0.8 + 0.5 * uBass;
    return col;
}

vec3 sJulia(vec2 p, float px)                                    // 13: a Julia fractal that morphs with the music
{
    vec2 z = rot2(p, uRot) * 1.35;
    float ang = uPsyT * 0.10 + 1.0;
    vec2 cis = vec2(cos(ang), sin(ang)), cis2 = vec2(cos(2.0 * ang), sin(2.0 * ang));
    vec2 c = (0.5 * cis - 0.25 * cis2) * (0.985 + 0.05 * uBass);                  // rides the edge of the Mandelbrot cardioid; bass shatters it
    float n = 0.0, trap = 1e3;
    for (int i = 0; i < 48; i++) {
        if (dot(z, z) > 16.0) break;
        z = vec2(z.x * z.x - z.y * z.y, 2.0 * z.x * z.y) + c;
        trap = min(trap, abs(dot(z, z) - 0.35));
        n += 1.0;
    }
    if (n > 47.5) return pal(uHue + 0.6 + 0.5 * trap) * (0.10 + 0.5 * exp(-trap * 5.0)) * (0.8 + 0.5 * uEnergy);
    float sm = n - log2(log2(dot(z, z))) + 4.0;
    vec3 col = pal(uHue + sm * 0.035 + 0.1 * uMid);
    col *= 0.35 + 0.9 * (0.5 + 0.5 * cos(sm * 0.45));
    return col * (1.0 + 0.4 * uBass);
}

vec3 sLava(vec2 p, float px)                                     // 14: lava lamp metaballs
{
    float t = uPsyT;
    float v = 0.0;
    for (int i = 0; i < 7; i++) {
        float fi = float(i);
        vec2 c = vec2(0.85 * sin(t * (0.30 + 0.07 * fi) + fi * 2.1), 0.62 * cos(t * (0.27 + 0.05 * fi) + fi * 1.3));
        c = rot2(c, uRot);
        float rr = 0.26 + 0.07 * uBass + 0.04 * sin(t * 0.8 + fi);
        vec2 d = p - c;
        v += rr * rr / (dot(d, d) + 1e-3);
    }
    vec3 bg = mix(vec3(0.10, 0.00, 0.12), vec3(0.00, 0.02, 0.18), p.y * 0.5 + 0.5);
    float inside = smoothstep(0.95, 1.05, v);
    float vc = min(v, 4.0);
    vec3 blob = pal(uHue + 0.12 * vc + 0.05) * (0.7 + 0.5 * smoothstep(1.0, 3.5, vc));
    vec3 col = mix(bg + vec3(0.25, 0.05, 0.2) * smoothstep(0.2, 1.0, v), blob, inside);
    col += (1.0 - smoothstep(0.0, 0.05, abs(v - 1.0))) * 0.35;
    return col;
}

vec3 sAurora(vec2 p, float px)                                   // 15: northern lights
{
    vec2 q = vec2(p.x + 0.35 * sin(uRot), p.y);
    float y = q.y * 0.5 + 0.5;
    vec3 col = mix(vec3(0.00, 0.01, 0.06), vec3(0.01, 0.04, 0.12), y);
    vec2 sq = p * 30.0;
    col += step(0.975, hash(floor(sq))) * smoothstep(0.4, 0.0, length(fract(sq) - 0.5)) * (0.4 + 0.4 * sin(uPsyT * 2.0 + hash(floor(sq)) * 40.0)) * y;
    for (int l = 0; l < 3; l++) {
        float fl = float(l);
        float w = 0.5 + 0.5 * sin(q.x * (1.7 + 0.8 * fl) + uPsyT * (0.20 + 0.08 * fl) + fl * 2.0 + 1.6 * sin(q.x * 0.9 + uPsyT * 0.15));
        float hgt = 0.30 + 0.28 * w + 0.12 * fl * 0.5 + 0.10 * uBass;
        float base = 0.18 + 0.04 * fl;
        float band = smoothstep(base, base + 0.12, y) * smoothstep(hgt + 0.25, hgt - 0.05, y);
        float rays = 0.55 + 0.45 * vnoise(vec2(q.x * 26.0 + fl * 9.0 + 3.0 * cos(uPsyT * 0.5), fl * 3.0 + 3.0 * sin(uPsyT * 0.5)));
        col += pal(0.38 + 0.18 * fl + uHue + 0.2 * (y - base)) * band * rays * (0.85 + 0.6 * uMid);
    }
    return col;
}

vec3 sBurst(vec2 p, float px)                                    // 16: sunburst equaliser (rays grow with bass / mids / highs)
{
    float r = length(p) + 1e-4;
    float a = atan(p.y, p.x) + uRot;
    const float N = 36.0;
    float sct = a * N / TAU;
    float id = mod(floor(sct), N);
    float f = fract(sct);
    float b = mod(id, 3.0);
    float e = b < 0.5 ? uBass : (b < 1.5 ? uMid : uHigh);
    float h = hash(vec2(id, 1.0));
    float len = 0.34 + 1.0 * clamp(e * 1.3 + 0.12 + 0.25 * h * (0.5 + 0.5 * sin(uPsyT * 1.5 + h * 20.0)), 0.0, 1.0);
    float across = abs(f - 0.5) * 2.0;
    float wd = 0.78 - 0.1 * h;
    float aa = 0.04 + px * 3.0 / r;
    float m = smoothstep(wd + aa, wd - aa, across) * smoothstep(len + 0.02, len - 0.02, r) * smoothstep(0.07, 0.11, r);
    vec3 ray = pal(id / N + uHue) * (0.6 + 0.6 * smoothstep(len, 0.0, r));
    vec3 bg = pal(uHue + 0.55) * 0.10 * (1.2 - r) + vec3(0.01);
    vec3 col = mix(bg, ray, m);
    col += exp(-r * 8.0) * pal(uHue + 0.1) * (0.6 + uBass);
    return col;
}

vec3 sSmoke(vec2 p, float px)                                    // 17: swirling ink / smoke
{
    vec2 q = rot2(p, uRot) * 1.7;
    vec2 o1 = 0.7 * vec2(cos(uPsyT * 0.20), sin(uPsyT * 0.20));
    vec2 o2 = 0.9 * vec2(sin(uPsyT * 0.15 + 1.0), cos(uPsyT * 0.25));
    float f1 = fbm(q + o1);
    float f2 = fbm(q + 2.2 * f1 + o2);
    float f = fbm(q + 2.4 * f2 + vec2(1.7, 9.2));
    vec3 col = mix(vec3(0.02, 0.01, 0.07), pal(uHue + 0.3 * f2 + 0.2 * f1), smoothstep(0.15, 0.85, f));
    col = mix(col, pal(uHue + 0.6 + 0.4 * f) * 1.2, smoothstep(0.55, 0.95, f * f2 * 2.0) * 0.6);
    col *= 0.85 + 0.6 * uBass * f;
    return col;
}

vec3 moreStyles(vec2 p, float px)
{
    if (uMode == 6)  return sTunnel(p, px);
    if (uMode == 7)  return sVortex(p, px);
    if (uMode == 8)  return sPlasma(p, px);
    if (uMode == 9)  return sRipples(p, px);
    if (uMode == 10) return sSynth(p, px);
    if (uMode == 11) return sWarp(p, px);
    if (uMode == 12) return sHoney(p, px);
    if (uMode == 13) return sJulia(p, px);
    if (uMode == 14) return sLava(p, px);
    if (uMode == 15) return sAurora(p, px);
    if (uMode == 16) return sBurst(p, px);
    return sSmoke(p, px);
}

vec3 pattern(vec2 p, float px)
{
    if (uMode >= 6) return moreStyles(p, px);
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
    vec2 p  = uZoomC + (p0 - uZoomC) / (1.0 + uZoom) + uShake;
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

BFX_FRAG = """
#version 330
uniform sampler2D uTex;
uniform float uI;        // intensity this frame (slider x bass envelope), 0..2
uniform int   uKind;     // 0 blur, 1 vibrate, 2 glitch
uniform float uSeed;
uniform float uTime;
uniform vec2  uRes;
in vec2 vUV;
out vec4 fragColor;

float hash(float n){ return fract(sin(n * 127.1 + uSeed * 17.3) * 43758.5453); }

void main()
{
    vec2 uv = vUV;
    float asp = uRes.x / uRes.y;
    vec3 col;
    if (uKind == 0) {                                   // blur: a soft disc that opens on the hit
        float r = 0.034 * uI;
        vec3 acc = texture(uTex, uv).rgb;
        for (int i = 0; i < 28; i++) {
            float a = float(i) * 2.39996;
            float rad = sqrt((float(i) + 0.5) / 28.0);
            acc += texture(uTex, uv + vec2(cos(a) / asp, sin(a)) * rad * r).rgb;
        }
        col = acc / 29.0;
    } else if (uKind == 1) {                            // vibrate: the whole picture shakes with a ghost trailing it
        float st = mod(floor(uTime * 60.0), 61.0);          // kept small: sin() of big numbers is garbage on a GPU
        vec2 d = (vec2(hash(st), hash(st + 3.7)) - 0.5) * 2.0;
        vec2 sh = d * 0.024 * uI * vec2(1.0 / asp, 1.0);
        col = 0.58 * texture(uTex, uv + sh).rgb + 0.42 * texture(uTex, uv - sh * 0.8).rgb;
    } else {                                            // glitch: the Domination banner's tearing, inverted bars and colour split
        float g = min(uI, 1.0);
        float extra = max(uI - 1.0, 0.0);
        float tstep = mod(floor(uTime * 34.0), 61.0);        // wrapped: a huge step count made the hash degrade after a few minutes
        float rows = 22.0;
        float row = floor(uv.y * rows);
        float h = hash(row + tstep * 7.0);
        float tear = step(1.0 - (0.38 * g + 0.30 * extra), h);
        uv.x += tear * (hash(row * 3.1 + tstep) - 0.5) * 0.30 * max(g, 0.2 * extra);
        uv.y += (hash(tstep + 9.0) - 0.5) * 0.02 * g;
        float ca = 0.003 + 0.022 * g * (0.3 + 0.7 * hash(tstep + 1.0));
        vec3 c3 = vec3(texture(uTex, uv + vec2(ca, 0.0)).r, texture(uTex, uv).g, texture(uTex, uv - vec2(ca, 0.0)).b);
        float inv = tear * step(0.72, hash(row * 5.3 + tstep * 2.0));
        col = mix(c3, 1.0 - c3, inv * g);
        col *= 1.0 - g * 0.07 * (0.5 - 0.5 * sin(vUV.y * 500.0));
    }
    fragColor = vec4(col, 1.0);
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
    fx_off="", spin=0.4, zoom=1.0, ab=1.0, color=3.0,
    dom_size=1.0, dom_rate=1.0,
    loop=False, help=True, domination=False,
    spotify=False, spot_delay=40.0, kzoom=1.0,
    media_path="", media_paths="", media_remember=False, media_on=True, media_blend=0, media_calm=0.40, media_peak=1.0, media_auto=True, media_speed=1.0, media_rate=0.3, media_pulse=1.0, scene_order=False,
    beat_smooth=0.35, media_flash=0.0, media_smooth=True, media_sway=False, media_sway_amt=1.0, media_hit=True, media_style=2, hit_cool=1.0,
    kick_lo=30.0, kick_hi=150.0, kick_sens=1.0, snare_lo=170.0, snare_hi=420.0, snare_sens=1.0,
    vis_open="style,preview,motion",
    exp_res=1, exp_fps=1, exp_q=1, exp_fmt=0, exp_from=0, exp_len=0, exp_aspect=0, exp_audio=True, exp_gpu=False, exp_dir="", bass_fx=0, bass_fx_amt=1.0, zoom_mode=0,
)

SLIDERS = [  # key, label, min, max   (grouped: dividers are drawn after rows 0, 5 and 7)
    ("volume", "Volume", 0.0, 1.0),
    ("spin", "Spin response", 0.0, 2.0),
    ("zoom", "Bass zoom", 0.0, 5.0),
    ("ab", "Chromatic aberration", 0.0, 2.0),
    ("color", "Colour drift & wave", 0.0, 3.0),
    ("kzoom", "Kaleidoscope zoom speed", 0.0, 3.0),
    ("dom_size", "Domination: text size", 0.5, 2.0),
    ("dom_rate", "Domination: text rate", 0.3, 3.0),
    ("bass_fx_amt", "Bass distortion", 0.0, 2.0),
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
VISUAL_KEYS = ("spin", "zoom", "bass_fx_amt", "ab", "beat_smooth", "color", "media_pulse", "kzoom", "dom_size", "dom_rate")


FX_NEUTRAL = dict(spin=0.0, zoom=0.0, bass_fx_amt=0.0, ab=0.0, beat_smooth=0.0, color=0.0, media_pulse=0.0, kzoom=0.0, dom_size=1.0, dom_rate=1.0)    # what a switched-off effect slider counts as


def fxv(cfg, key):
    """A Motion & effects slider's value for the engine: its neutral value while the slider's dot is switched off."""
    off = cfg.get("fx_off")
    if off and key in off.split(","):
        return FX_NEUTRAL[key]
    return cfg[key]


def vis_keys(dom, mode=0):
    """The Motion & effects sliders: the Domination ones only show while Domination mode is on, the Kaleidoscope one only with that style."""
    return tuple(k for k in VISUAL_KEYS if (dom or not k.startswith("dom_")) and (mode == 3 or k != "kzoom"))
BFX_NAMES = ("Blur", "Vibrate", "Glitch", "Random")
ZOOM_MODES = ("Centered zoom", "Shaky zoom", "Random area zoom")
BEAT_KEYS = ("hit_cool", "kick_lo", "kick_hi", "kick_sens", "snare_lo", "snare_hi", "snare_sens")
MEDIA_KEYS = ("media_calm", "media_peak", "media_speed", "media_flash", "media_sway_amt", "media_rate")
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
MODES = ["Classic", "Prism", "Neon", "Kaleidoscope", "Psychedelic", "Media", "Tunnel", "Vortex", "Plasma", "Ripples", "Synthwave", "Warp", "Honeycomb", "Julia", "Lava", "Aurora", "Sunburst", "Smoke"]
MODE_ORDER = [5, 0, 1, 2, 3, 4] + list(range(6, 18))        # the order of the style cards (Media first); a style's number never changes
VIDEO_MODE = 5            # shows only the media layer (no spiral); keeps playing at its last speed when the audio stops


def app_dir():
    return os.path.dirname(sys.executable if getattr(sys, "frozen", False) else os.path.abspath(__file__))


def settings_path():
    base = os.environ.get("APPDATA") or os.path.join(os.path.expanduser("~"), ".config")
    return os.path.join(base, "Hypnosis", "settings.json")


def scenes_path():
    return os.path.join(os.path.dirname(settings_path()), "scenes.json")


SCENE_TOL = 0.3        # a hidden / deleted scene matches a detected scene start within this many seconds


def _nums(v):
    return [float(x) for x in v if isinstance(x, (int, float)) and not isinstance(x, bool)] if isinstance(v, list) else []


def clean_scene_pref(v):
    """One clip's scene edits, validated: hidden / deleted scene starts, trims, user-made cuts and the custom order."""
    if not isinstance(v, dict):
        return None
    tr, cu = [], []
    for x in (v.get("trim") or []):
        try:
            t0, a0, b0 = (float(q) for q in x)
            if b0 - a0 >= 0.1:
                tr.append([t0, a0, b0])
        except Exception:
            pass
    for x in (v.get("custom") or []):
        try:
            k0, a0, b0 = (float(q) for q in x)
            if b0 - a0 >= 0.1:
                cu.append([k0, a0, b0])
        except Exception:
            pass
    pf = {"off": _nums(v.get("off")), "del": _nums(v.get("del")), "trim": tr, "custom": cu, "order": _nums(v.get("order"))}
    return pf if (pf["off"] or pf["del"] or tr or cu or pf["order"]) else None


def clean_scene_prefs(raw):
    out = {}
    if isinstance(raw, dict):
        for p, v in raw.items():
            pf = clean_scene_pref(v)
            if pf:
                out[str(p)] = pf
    return out


def load_scene_prefs():
    """{video path: scene edits} - which scenes the user hid, deleted, trimmed, cut out or re-ordered (survives restarts)."""
    try:
        with open(scenes_path(), "r", encoding="utf-8") as f:
            return clean_scene_prefs(json.load(f))
    except Exception:
        return {}


def save_scene_prefs(prefs):
    try:
        p = scenes_path()
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump({k: v for k, v in prefs.items() if v.get("off") or v.get("del") or v.get("trim") or v.get("custom") or v.get("order")}, f, indent=1)
    except Exception:
        pass


# ---- remembered scene scans: a video's detected cuts are stored so they are never rebuilt (also embedded in projects)
def scancache_path():
    return os.path.join(os.path.dirname(settings_path()), "scancache.json")


_SCAN = {"data": None}


def _scan_load():
    if _SCAN["data"] is None:
        try:
            with open(scancache_path(), "r", encoding="utf-8") as f:
                d = json.load(f)
            _SCAN["data"] = d if isinstance(d, dict) else {}
        except Exception:
            _SCAN["data"] = {}
    return _SCAN["data"]


def scan_sig(path, duration):
    try:
        return [os.path.getsize(path), round(float(duration), 1)]
    except OSError:
        return None


def _scan_ok(ent, sig):
    if not (isinstance(ent, dict) and isinstance(ent.get("scenes"), list) and ent.get("sig") == sig and sig is not None):
        return None
    sc = _nums(ent["scenes"])
    if len(sc) <= 1 and ent.get("v") != 2:                  # a lone scene from an older build / project may be a failed scan: look again
        return None
    return sorted(sc) if sc and abs(sorted(sc)[0]) < 1e-6 else None


def scan_cache_get(path, duration):
    return _scan_ok(_scan_load().get(path), scan_sig(path, duration))


def scan_cache_put(path, duration, scenes):
    sig = scan_sig(path, duration)
    if sig is None or not scenes:
        return
    d = _scan_load()
    d.pop(path, None)
    d[path] = {"sig": sig, "scenes": [round(float(t), 4) for t in scenes], "v": 2}
    while len(d) > 300:
        d.pop(next(iter(d)))
    try:
        os.makedirs(os.path.dirname(scancache_path()), exist_ok=True)
        tmp = scancache_path() + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(d, f)
        os.replace(tmp, scancache_path())
    except Exception:
        pass


def scan_cache_clear(keep):
    """Forget every remembered scene list except those of the files in `keep` (the clips that are loaded right now). Returns how many went."""
    d = _scan_load()
    keep = set(keep)
    gone = [p for p in d if p not in keep]
    for p in gone:
        d.pop(p, None)
    try:
        os.makedirs(os.path.dirname(scancache_path()), exist_ok=True)
        tmp = scancache_path() + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(d, f)
        os.replace(tmp, scancache_path())
    except Exception:
        pass
    return len(gone)


def scan_cache_export(paths):
    d = _scan_load()
    return {p: d[p] for p in paths if isinstance(d.get(p), dict) and d[p].get("scenes")}


def scan_cache_import(raw, remap=None):
    """Scene scans stored in a project: make them the known scans for those files (so opening the project skips scanning)."""
    if not isinstance(raw, dict):
        return
    d, changed = _scan_load(), False
    for p, ent in raw.items():
        if isinstance(ent, dict) and isinstance(ent.get("scenes"), list) and isinstance(ent.get("sig"), list) and _nums(ent["scenes"]):
            q = (remap or {}).get(p, p)
            d.pop(q, None)
            d[q] = {"sig": ent["sig"], "scenes": _nums(ent["scenes"])}
            if ent.get("v") == 2:
                d[q]["v"] = 2
            changed = True
    if changed:
        try:
            os.makedirs(os.path.dirname(scancache_path()), exist_ok=True)
            with open(scancache_path(), "w", encoding="utf-8") as f:
                json.dump(d, f)
        except Exception:
            pass


def _has(lst, t):
    return any(abs(x - t) <= SCENE_TOL for x in lst)


PRESET_SKIP = {"exp_res", "exp_fps", "exp_q", "exp_fmt", "exp_from", "exp_len", "exp_aspect", "exp_audio", "exp_gpu", "exp_dir", "vis_open", "media_remember", "volume", "spotify", "media_path", "media_paths", "fs_kind", "spot_delay", "help"}


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


# --------------------------------------------------------------------------- projects (.hypno files + the recent list)
PROJ_EXT = ".hypno"
THUMB_W, THUMB_H = 320, 180


def projects_dir():
    return os.path.join(os.path.dirname(settings_path()), "projects")


def recents_path():
    return os.path.join(os.path.dirname(settings_path()), "recents.json")


def load_recents():
    try:
        with open(recents_path(), "r", encoding="utf-8") as f:
            raw = json.load(f).get("recent", [])
        return [{"path": str(r["path"]), "used": float(r.get("used", 0))} for r in raw if isinstance(r, dict) and "path" in r]
    except Exception:
        return []


def save_recents(rec):
    try:
        os.makedirs(os.path.dirname(recents_path()), exist_ok=True)
        with open(recents_path(), "w", encoding="utf-8") as f:
            json.dump({"recent": rec[:40]}, f, indent=1)
    except Exception:
        pass


def touch_recent(path):
    path = os.path.abspath(path)
    rec = [r for r in load_recents() if os.path.normcase(os.path.abspath(r["path"])) != os.path.normcase(path)]
    rec.insert(0, {"path": path, "used": time.time()})
    save_recents(rec)


def forget_recent(path):
    n = os.path.normcase(os.path.abspath(path))
    save_recents([r for r in load_recents() if os.path.normcase(os.path.abspath(r["path"])) != n])


def read_project(path):
    """The project dict, or None if the file is missing / not a Hypnosis project."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and data.get("app") == "Hypnosis":
            return data
    except Exception:
        pass
    return None


def write_project(path, data):
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1)
    os.replace(tmp, path)


def thumb_encode(rgb):
    """HxWx3 uint8 -> base64 JPEG."""
    try:
        import base64
        import io
        from PIL import Image
        im = Image.fromarray(rgb).resize((THUMB_W, THUMB_H), Image.LANCZOS)
        b = io.BytesIO()
        im.save(b, "JPEG", quality=82)
        return base64.b64encode(b.getvalue()).decode("ascii")
    except Exception:
        return ""


def thumb_decode(s):
    try:
        import base64
        import io
        from PIL import Image
        im = Image.open(io.BytesIO(base64.b64decode(s))).convert("RGB")
        return np.asarray(im, dtype=np.uint8).copy()
    except Exception:
        return None


def _plist(v):
    return [str(p) for p in v if isinstance(p, str) and p] if isinstance(v, list) else []


def project_paths(data):
    q = data.get("queue") if isinstance(data.get("queue"), dict) else {}
    m = data.get("media") if isinstance(data.get("media"), dict) else {}
    return _plist(q.get("paths")), _plist(m.get("paths"))


def project_missing(data):
    qp, mp = project_paths(data)
    return [p for p in qp if not os.path.isfile(p)], [p for p in mp if not os.path.isfile(p)]


_psum = {}


def project_summary(path):
    """Cheap card info for the home screen (cached until the file changes)."""
    try:
        mt = os.path.getmtime(path)
    except OSError:
        return None
    hit = _psum.get(path)
    if hit and hit[0] == mt:
        return hit[1]
    data = read_project(path)
    if data is None:
        return None
    qp, mp = project_paths(data)
    mq, mm = project_missing(data)
    info = dict(path=path, name=str(data.get("name") or os.path.splitext(os.path.basename(path))[0])[:60],
                tracks=len(qp), clips=len(mp), missing=len(mq) + len(mm),
                saved=float(data.get("saved", mt) or mt), thumb=thumb_decode(data.get("thumb", "")) if data.get("thumb") else None)
    _psum[path] = (mt, info)
    return info


def list_projects(limit=60):
    """Projects from the recent list plus anything in the projects folder, most recently used first."""
    used, seen = {}, {}
    for r in load_recents():
        k = os.path.normcase(os.path.abspath(r["path"]))
        if k not in seen:
            seen[k] = r["path"]
            used[k] = r["used"]
    try:
        d = projects_dir()
        for f in os.listdir(d):
            if f.lower().endswith(PROJ_EXT):
                p = os.path.join(d, f)
                k = os.path.normcase(os.path.abspath(p))
                if k not in seen:
                    seen[k] = p
                    used[k] = os.path.getmtime(p)
    except Exception:
        pass
    out = []
    for k, p in seen.items():
        info = project_summary(p)
        if info is not None:
            out.append(dict(info, used=used[k]))
    out.sort(key=lambda r: -r["used"])
    return out[:limit]


def fmt_ago(ts):
    d = max(0.0, time.time() - ts)
    if d < 60:
        return "just now"
    if d < 3600:
        return f"{int(d // 60)} min ago"
    if d < 86400:
        h = int(d // 3600)
        return f"{h} hour{'s' if h != 1 else ''} ago"
    if d < 86400 * 7:
        n = int(d // 86400)
        return f"{n} day{'s' if n != 1 else ''} ago"
    return time.strftime("%b %d, %Y", time.localtime(ts))


def find_files(names, root, limit_s=25.0):
    """Walk `root` looking for files called any of `names`. -> {name.lower(): full path}."""
    want = {n.lower() for n in names}
    found = {}
    t_end = time.time() + limit_s
    for dp, dn, fn in os.walk(root):
        for f in fn:
            if f.lower() in want and f.lower() not in found:
                found[f.lower()] = os.path.join(dp, f)
        if len(found) == len(want) or time.time() > t_end:
            break
        dn[:] = [d for d in dn if not d.startswith(".") and d.lower() not in ("$recycle.bin", "windows", "node_modules")]
    return found


def clean_cfg_values(vals):
    """Keep only known, correctly-typed, non-skipped settings (same rules as presets)."""
    out = {}
    if not isinstance(vals, dict):
        return out
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
            out[k] = v
    return out


def _tk_dialog(fn):
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        try:
            return fn(root)
        finally:
            root.destroy()
    except Exception:
        return None


def dlg_save_project(inbox, tag, initial_name):
    from tkinter import filedialog
    os.makedirs(projects_dir(), exist_ok=True)
    p = _tk_dialog(lambda r: filedialog.asksaveasfilename(
        title="Save project", initialdir=projects_dir(), initialfile=initial_name, defaultextension=PROJ_EXT,
        filetypes=[("Hypnosis project", "*" + PROJ_EXT)]))
    inbox.append((tag, p or None))


def dlg_open_project(inbox, tag):
    from tkinter import filedialog
    os.makedirs(projects_dir(), exist_ok=True)
    p = _tk_dialog(lambda r: filedialog.askopenfilename(
        title="Open project", initialdir=projects_dir(), filetypes=[("Hypnosis project", "*" + PROJ_EXT), ("All files", "*.*")]))
    inbox.append((tag, p or None))


def dlg_find(inbox, names):
    """Pick a folder, then search it for `names`; posts ("find", {lowercase name: path})."""
    from tkinter import filedialog
    d = _tk_dialog(lambda r: filedialog.askdirectory(title="Folder to search for the missing files"))
    found = {}
    if d:
        try:
            found = find_files(names, d)
        except Exception:
            found = {}
    inbox.append((("find",), found))


def dlg_locate(inbox, tag, name):
    from tkinter import filedialog
    p = _tk_dialog(lambda r: filedialog.askopenfilename(title="Find " + name, initialfile=name))
    inbox.append((tag, p or None))


# --------------------------------------------------------------------------- export (record the scene + the track to a video file)
EXP_RES = (("720p", 720), ("1080p", 1080), ("1440p", 1440), ("4K", 2160))
EXP_FPS = (24, 30, 60)
EXP_Q = (("Draft", 28), ("Good", 23), ("High", 19), ("Best", 16))
EXP_FMT = (("MP4", "mp4"), ("MKV", "mkv"), ("WebM", "webm"))
EXP_PV = (532, 98, 268, 150)         # the live preview on the Export tab (logical panel coordinates)
EXP_LEN = (("Whole song", 0), ("30 sec", 30), ("1 min", 60), ("2 min", 120))


def exp_opt(cfg, name, table):
    """The table entry for a stored option index (a hand-edited settings file can hold anything)."""
    return table[min(max(int(cfg[name]), 0), len(table) - 1)]


def exp_default_dir():
    home = os.path.expanduser("~")
    return os.path.join(home, "Videos", "Hypnosis")


def exp_folder(cfg):
    d = (cfg.get("exp_dir") or "").strip()
    return d if d else exp_default_dir()


def exp_size(cfg, win_w, win_h):
    """Output size (even numbers) from the chosen height and aspect (16:9, or the window's own)."""
    h = exp_opt(cfg, "exp_res", EXP_RES)[1]
    ar = (win_w / float(max(1, win_h))) if int(cfg["exp_aspect"]) == 1 else 16.0 / 9.0
    ar = max(0.4, min(3.0, ar))
    w = int(round(h * ar / 2.0)) * 2
    return max(2, w), h


def exp_name(title):
    base = "".join(c for c in (title or "Hypnosis") if c not in '\\/:*?"<>|').strip().strip(".") or "Hypnosis"
    return f"{base[:60]} - Hypnosis {time.strftime('%Y-%m-%d %H%M%S')}"


def fmt_bytes(n):
    n = float(n)
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024 or u == "GB":
            return f"{n:.0f} {u}" if u == "B" else f"{n:.1f} {u}"
        n /= 1024.0


def fmt_clock(sec):
    sec = max(0, int(sec))
    return f"{sec // 60}:{sec % 60:02d}"


class Exporter:
    """Encodes in a worker thread: video frames come in (RGB, bottom-up rows) tagged with their frame number, missing numbers
    are filled with the previous picture (constant frame rate), and the track's own samples are muxed alongside."""

    def __init__(self, path, size, fps, crf, fmt, data, p0, gpu, with_audio):
        self.path, self.size, self.fps = path, size, fps
        self.fmt, self.crf, self.gpu, self.with_audio = fmt, crf, gpu, with_audio
        self.data, self.p0 = data, p0
        self.q = queue.Queue(maxsize=10)
        self.state = "running"            # running -> finalizing -> done | error
        self.error = None
        self.frames = 0                   # frames written so far
        self.dropped = 0                  # frames the encoder could not keep up with (filled by repeating the previous one)
        self.encoder = ""
        self.end_ts = None
        self.th = threading.Thread(target=self._run, daemon=True)
        self.th.start()

    # ---- main-thread side
    def push(self, idx, rgb):
        try:
            self.q.put_nowait((idx, rgb))
            return True
        except queue.Full:
            self.dropped += 1
            return False

    def finish(self, end_ts):
        self.end_ts = max(0.0, float(end_ts))
        self.state = "finalizing"
        self.q.put(None)                  # blocks only if the queue is full - the worker is draining it

    def abort(self):
        self.end_ts = -1.0
        self.state = "finalizing"
        try:
            while True:
                self.q.get_nowait()
        except queue.Empty:
            pass
        self.q.put(None)

    # ---- worker
    def _open(self, av):
        w, h = self.size
        fmt = self.fmt
        c = av.open(self.path, "w", format={"mp4": "mp4", "mkv": "matroska", "webm": "webm"}[fmt])
        opts, name = {}, "libx264"
        if fmt == "webm":
            name = "libvpx-vp9"
            opts = {"crf": str(self.crf + 6), "b": "0", "deadline": "realtime", "cpu-used": "6", "row-mt": "1"}
        elif self.gpu:
            try:
                cc = av.codec.CodecContext.create("h264_nvenc", "w")
                cc.width, cc.height, cc.pix_fmt, cc.time_base = 640, 360, "yuv420p", Fraction(1, 30)
                cc.open()
                name = "h264_nvenc"
                opts = {"preset": "p5", "rc": "vbr", "cq": str(self.crf), "b": "0"}
            except Exception:
                name = "libx264"
        if name == "libx264":
            opts = {"preset": "veryfast" if self.crf >= 23 else "faster", "crf": str(self.crf)}
            if fmt == "mp4":
                opts["movflags"] = "+faststart"
        self.encoder = {"libx264": "x264 (CPU)", "h264_nvenc": "NVENC (GPU)", "libvpx-vp9": "VP9 (CPU)"}[name]
        v = c.add_stream(name, rate=self.fps)
        v.width, v.height, v.pix_fmt = w, h, "yuv420p"
        v.codec_context.time_base = Fraction(1, self.fps)
        v.time_base = Fraction(1, self.fps)
        v.options = opts
        try:
            v.codec_context.colorspace = 1          # BT.709 tags so players show the same colours
            v.codec_context.color_primaries = 1
            v.codec_context.color_trc = 1
        except Exception:
            pass
        a = None
        ar = SR
        if self.with_audio and self.data is not None:
            if fmt == "webm":
                a = c.add_stream("libopus", rate=48000)
                ar = 48000
            else:
                a = c.add_stream("aac", rate=SR)
                a.bit_rate = 256000
            a.layout = "stereo"
        return c, v, a, ar

    def _run(self):
        try:
            import av
            c, v, a, ar = self._open(av)
            res = av.AudioResampler(format="fltp", layout="stereo", rate=ar) if a is not None else None
            w, h = self.size
            last_yuv, next_v, a_done = None, 0, 0

            def write_video(yuv, idx):
                yuv.pts = idx
                for pk in v.encode(yuv):
                    c.mux(pk)
                self.frames = idx + 1

            def write_audio(upto):
                nonlocal a_done
                if a is None:
                    return
                total = len(self.data) - self.p0
                upto = min(int(upto), total)
                while a_done < upto:
                    n = min(upto - a_done, 8192)
                    seg = np.ascontiguousarray(self.data[self.p0 + a_done:self.p0 + a_done + n], dtype=np.float32)
                    fr = av.AudioFrame.from_ndarray(seg.reshape(1, -1), format="flt", layout="stereo")
                    fr.sample_rate = SR
                    fr.pts = a_done
                    fr.time_base = Fraction(1, SR)
                    a_done += n
                    for r_ in res.resample(fr):
                        for pk in a.encode(r_):
                            c.mux(pk)

            while True:
                item = self.q.get()
                if item is None:
                    break
                idx, rgb = item
                img = np.frombuffer(rgb, np.uint8).reshape(h, w, 3)[::-1]
                fr = av.VideoFrame.from_ndarray(np.ascontiguousarray(img), format="rgb24")
                yuv = fr.reformat(format="yuv420p", dst_colorspace=av.video.reformatter.Colorspace.ITU709) \
                    if hasattr(av.video, "reformatter") and hasattr(av.video.reformatter, "Colorspace") else fr.reformat(format="yuv420p")
                while last_yuv is not None and next_v < idx:          # frames we never got: repeat the last picture
                    write_video(last_yuv, next_v)
                    self.dropped += 1
                    next_v += 1
                if idx >= next_v:
                    write_video(yuv, idx)
                    next_v = idx + 1
                last_yuv = yuv
                write_audio((next_v) * SR / self.fps)
            if self.end_ts is not None and self.end_ts >= 0:
                end_n = int(self.end_ts * self.fps)
                while last_yuv is not None and next_v < end_n:
                    write_video(last_yuv, next_v)
                    next_v += 1
                write_audio(self.end_ts * SR)
            for pk in v.encode(None):
                c.mux(pk)
            if a is not None:
                for pk in a.encode(None):
                    c.mux(pk)
            c.close()
            if self.end_ts is not None and self.end_ts < 0:
                try:
                    os.remove(self.path)
                except OSError:
                    pass
                self.state = "aborted"
            else:
                self.state = "done"
        except Exception as e:
            self.error = f"{type(e).__name__}: {e}"
            self.state = "error"
            log("export failed: " + self.error)
            try:
                while True:
                    self.q.get_nowait()
            except queue.Empty:
                pass


def dlg_pick_folder(inbox, tag, initial):
    from tkinter import filedialog
    if initial and not os.path.isdir(initial):
        initial = os.path.dirname(initial) if os.path.isdir(os.path.dirname(initial)) else None
    d = _tk_dialog(lambda r: filedialog.askdirectory(title="Export folder", initialdir=initial or None, mustexist=False))
    inbox.append((tag, d or None))


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


class LoadingOverlay:
    """Full-screen "Loading Scenes" cover shown while a freshly added video is scanned for its scenes."""
    CW, CH = 600, 275

    def __init__(self, ctx, quads):
        self.ctx, self.quads = ctx, quads
        self.tex, self.key, self.fonts = None, None, {}

    def font(self, size, bold=False):
        k = (size, bold)
        if k not in self.fonts:
            self.fonts[k] = pygame.font.SysFont("segoeui,helveticaneue,dejavusans,arial", size, bold=bold)
        return self.fonts[k]

    def build(self, sc, pct, scenes, name, dots, k=1, n=1):
        cw, ch = int(self.CW * sc), int(self.CH * sc)
        surf = pygame.Surface((cw, ch), pygame.SRCALPHA)
        ft = self.font(int(40 * sc), True)
        t = ft.render("Loading Scenes" + "." * dots, True, (255, 255, 255))
        surf.blit(t, (cw / 2 - ft.size("Loading Scenes...")[0] / 2, 8 * sc))
        by = int((84 if n <= 1 else 112) * sc)
        if n > 1:
            fv = self.font(int(24 * sc), True)
            tv = fv.render("Video %d of %d" % (k, n), True, (120, 210, 255))
            surf.blit(tv, (cw / 2 - tv.get_width() / 2, 62 * sc))
        bw, bh = int(480 * sc), int(22 * sc)
        bx = (cw - bw) // 2
        pygame.draw.rect(surf, (255, 255, 255, 40), (bx, by, bw, bh), border_radius=bh // 2)
        fw = int(bw * max(0.0, min(1.0, pct / 100.0)))
        if fw >= bh // 2:
            bar = pygame.Surface((fw, bh), pygame.SRCALPHA)
            for x in range(fw):
                f = x / float(max(1, bw - 1))
                pygame.draw.line(bar, (int(90 + 150 * f), int(210 - 110 * f), 255, 255), (x, 0), (x, bh))
            mask = pygame.Surface((fw, bh), pygame.SRCALPHA)
            pygame.draw.rect(mask, (255, 255, 255, 255), (0, 0, fw, bh), border_radius=bh // 2)
            bar.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
            surf.blit(bar, (bx, by))
        fp = self.font(int(34 * sc), True)
        p = fp.render("%d%%" % pct, True, (255, 255, 255))
        surf.blit(p, (cw / 2 - p.get_width() / 2, by + bh + 14 * sc))
        fs = self.font(int(22 * sc))
        t2 = fs.render("%d scene%s found" % (scenes, "" if scenes == 1 else "s"), True, (210, 215, 235))
        surf.blit(t2, (cw / 2 - t2.get_width() / 2, by + bh + 62 * sc))
        if name:
            fn = self.font(int(16 * sc))
            t3 = fn.render(name if len(name) < 60 else name[:57] + "...", True, (140, 145, 170))
            surf.blit(t3, (cw / 2 - t3.get_width() / 2, by + bh + 100 * sc))
        self.tex = surface_to_texture(self.ctx, surf, self.tex)
        return cw, ch

    def draw(self, size, alpha, pct, scenes, name, now, k=1, n=1, blur=0.0, pblur=None):
        if alpha <= 0.003:
            return
        W, H = size
        sc = max(0.7, min(W / 1280.0, H / 720.0))
        key = (round(sc, 2), int(pct), scenes, name, int(now * 3) % 4, k, n)
        if key != self.key:
            self.key = key
            self.cwh = self.build(sc, int(pct), scenes, name, int(now * 3) % 4, k, n)
        cw, ch = self.cwh
        if pblur is not None and blur > 0.3:                     # the card sharpens in / blurs away
            prog, vao = pblur
            (ncx, ncy), (hx, hy) = ndc_rect((W - cw) / 2, (H - ch) / 2, cw, ch, W, H)
            prog["uCenter"].value, prog["uHalf"].value = (ncx, ncy), (hx, hy)
            prog["uPx"].value = (blur / cw, blur / ch)
            prog["uAlpha"].value = float(alpha)
            self.tex.use(0)
            prog["uTex"].value = 0
            self.ctx.enable(moderngl.BLEND)
            self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
            vao.render(moderngl.TRIANGLE_STRIP)
            self.ctx.disable(moderngl.BLEND)
        else:
            self.quads.draw(self.tex, ((W - cw) / 2, (H - ch) / 2, cw, ch), size, alpha)


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


class BounceText:
    """The Video style's placeholder while no clip is on the stack: extruded 3D text that spins and bounces around the
    screen like the old DVD player's idle logo, changing colour whenever it hits an edge."""
    LAYERS = 16
    TEXT = "VIDEO EXAMPLE"
    VERT = """
#version 330
in vec4 in_pos;     // clip space, w already = camera distance ratio (perspective-correct texture lookup)
in vec2 in_uv;
out vec2 vUV;
void main(){ vUV = in_uv; gl_Position = in_pos; }
"""
    FRAG = """
#version 330
uniform sampler2D uTex;
uniform vec4 uCol;
in vec2 vUV;
out vec4 fragColor;
void main(){ fragColor = vec4(uCol.rgb, texture(uTex, vUV).a * uCol.a); }
"""

    def __init__(self, ctx):
        self.ctx = ctx
        self.prog = ctx.program(vertex_shader=self.VERT, fragment_shader=self.FRAG)
        self.vbo = ctx.buffer(reserve=4 * 6 * 4)
        self.vao = ctx.vertex_array(self.prog, [(self.vbo, "4f 2f", "in_pos", "in_uv")])
        self.tex, self.aspect = None, 4.0
        self.x = self.y = None
        self.vx = self.vy = 0.0
        self.t = random.uniform(0.0, 6.0)
        self.hue = random.random()
        self.tscale = 1.0                                      # text size factor (the small style card draws it larger)
        self.keep_alpha = False                                # True when drawing into a target whose alpha must stay opaque

    def _texture(self):
        font = pygame.font.SysFont("impact,arialblack,segoeuiblack,arial", 180)
        txt = font.render(self.TEXT, True, (255, 255, 255))
        pad = 16
        surf = pygame.Surface((txt.get_width() + 2 * pad, txt.get_height() + 2 * pad), pygame.SRCALPHA)
        for ox in range(-3, 4):                                # fatten the letters so the extruded sides read as solid 3D
            for oy in range(-3, 4):
                if ox * ox + oy * oy <= 10:
                    surf.blit(txt, (pad + ox, pad + oy))
        self.aspect = surf.get_width() / float(surf.get_height())
        self.tex = surface_to_texture(self.ctx, surf)

    @staticmethod
    def _hsv(h, s_, v):
        h = (h % 1.0) * 6.0
        i = int(h)
        f = h - i
        p_, q, t_ = v * (1 - s_), v * (1 - s_ * f), v * (1 - s_ * (1 - f))
        return [(v, t_, p_), (q, v, p_), (p_, v, t_), (p_, q, v), (t_, p_, v), (v, p_, q)][i % 6]

    def draw(self, dt, W, H):
        if self.tex is None:
            self._texture()
        dt = min(dt, 0.1)
        self.t += dt
        th = 0.17 * H * self.tscale
        tw = th * self.aspect
        if tw > 0.62 * W:
            tw = 0.62 * W
            th = tw / self.aspect
        hw, hh = tw * 0.5 * 1.04, th * 0.5 * 1.25
        if self.x is None:
            self.x, self.y = random.uniform(hw, W - hw), random.uniform(hh, H - hh)
            sp = 0.16 * W
            self.vx, self.vy = sp * random.choice((-1, 1)), sp * 0.62 * random.choice((-1, 1))
        self.x += self.vx * dt
        self.y += self.vy * dt
        hit = False
        if self.x < hw:
            self.x, self.vx, hit = hw, abs(self.vx), True
        elif self.x > W - hw:
            self.x, self.vx, hit = W - hw, -abs(self.vx), True
        if self.y < hh:
            self.y, self.vy, hit = hh, abs(self.vy), True
        elif self.y > H - hh:
            self.y, self.vy, hit = H - hh, -abs(self.vy), True
        if hit:
            self.hue += 0.17 + random.random() * 0.2          # a new colour on every bounce, like the DVD logo
        ay, ax, az = self.t * 1.15, 0.30 * math.sin(self.t * 0.7), 0.07 * math.sin(self.t * 0.45)
        cy_, sy_, cx_, sx_, cz_, sz_ = math.cos(ay), math.sin(ay), math.cos(ax), math.sin(ax), math.cos(az), math.sin(az)

        def rot(x, y, z):                                      # Y spin, then X tilt, then Z roll
            x, z = x * cy_ + z * sy_, -x * sy_ + z * cy_
            y, z = y * cx_ - z * sx_, y * sx_ + z * cx_
            return x * cz_ - y * sz_, x * sz_ + y * cz_, z

        depth = 0.34 * th
        D = 2.0 * max(W, H)
        n = self.LAYERS
        layers = []
        for k in range(n):
            z = (k / (n - 1.0) - 0.5) * depth
            pts = []
            for lx, ly, u, v in ((-1, 1, 0, 1), (-1, -1, 0, 0), (1, 1, 1, 1), (1, -1, 1, 0)):
                x2, y2, z2 = rot(lx * tw * 0.5, ly * th * 0.5, z)
                w = (D - z2) / D
                sx, sy = self.x + x2 / w, self.y - y2 / w
                pts.append(((sx / W * 2.0 - 1.0) * w, (1.0 - sy / H * 2.0) * w, 0.0, w, u, v))
            layers.append((rot(0.0, 0.0, z)[2], k, pts))
        layers.sort(key=lambda L: L[0])                        # farthest first: the cap that faces you is drawn last, on top
        ctx = self.ctx
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA, moderngl.ZERO, moderngl.ONE) if self.keep_alpha \
            else (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
        self.tex.use(0)
        self.prog["uTex"].value = 0
        for i, (_, k, pts) in enumerate(layers):
            cap = i == n - 1
            r, g, b = self._hsv(self.hue + 0.015 * k, 0.55 if cap else 0.8, 1.0 if cap else 0.30 + 0.22 * (i / (n - 1.0)))
            self.prog["uCol"].value = (r, g, b, 1.0)
            self.vbo.write(np.array(pts, dtype="f4").tobytes())
            self.vao.render(moderngl.TRIANGLE_STRIP)
        ctx.disable(moderngl.BLEND)


# --------------------------------------------------------------------------- menu panel
PW, PH = 840, 660
HDR_Y, FTR_Y = 64, 584                     # the Visuals page scrolls between the header line and the footer line
PREV_W = 420                                # live preview width on the Visuals tab (logical px)


SEC_HDR_H, SEC_GAP, SEC_PAD = 38, 8, 8        # Visuals page: section header height, gap between sections, space above a section's content
VIS_SECTIONS = (("style", "Visual style"), ("preview", "Live preview"), ("motion", "Motion & effects"), ("media", "Media & stack"))
MEDIA_H = 592                                   # height of the Media & stack content
ZM_H = 88                                     # the "Bass zoom mode" buttons inside Motion & effects
BFX_H = 88                                     # the "Bass distortion style" buttons inside Motion & effects
MFX_H, MSEP_H = 319, 46                         # the "Media effects" block inside Motion & effects, and the divider above it
CARD_H = 101
STY_CW, STY_GAP, STY_X0, STY_VW = 125, 8, 24, 792        # the style cards: one scrolls past the next in a row 792 wide
STY_PITCH = STY_CW + STY_GAP
STY_MARGIN = 10                                            # room past the row's ends while it is scrolled all the way there
STY_FZ = 56                                                # how wide the blur-fade at the row's edges is
STY_SB_H = 22                                              # the scroll bar under the row


def sty_total():
    return len(MODE_ORDER) * STY_PITCH - STY_GAP


def sty_max():
    return max(0.0, sty_total() - STY_VW)


def sty_thumb():
    return max(44.0, STY_VW * STY_VW / sty_total())


def visuals_layout(aspect, anim=None, dom=True, mode=0):
    """The Visuals page: collapsible sections stacked from the top. anim = {name: 0..1} (how far each one is open).
    Each section: hy (header top), ct (content top), ch (full content height), vis (visible content height), plus the scroll range."""
    ph = int(max(170, min(270, PREV_W * aspect)))
    full = dict(style=CARD_H + 4 + STY_SB_H, preview=ph + 12, motion=len(vis_keys(dom, mode)) * SL_PITCH + 12 + ZM_H + BFX_H + MSEP_H + MFX_H, media=MEDIA_H)
    out, y = {}, 72.0
    for name, _ in VIS_SECTIONS:
        a = 1.0 if anim is None else max(0.0, min(1.0, anim.get(name, 1.0)))
        out[name] = dict(hy=y, ct=y + SEC_HDR_H + SEC_PAD, ch=full[name], a=a, vis=a * (full[name] + SEC_PAD))
        y += SEC_HDR_H + a * (full[name] + SEC_PAD) + SEC_GAP
    out["ph"] = ph
    out["end"] = y
    out["max_scroll"] = max(0, int(math.ceil(y - FTR_Y + 4)))
    return out


DOCK_H = 60                                      # the mini live preview that takes over the player bar's art + title
DOCK_S = 0.62                                    # seconds for the preview <-> player bar hand-over


def _ss(a, b, v):
    t = max(0.0, min(1.0, (v - a) / (b - a)))
    return t * t * (3 - 2 * t)


def dock_phases(k):
    """k = 0 (preview in its section) .. 1 (preview in the player bar) -> (page preview alpha, its blur, bar preview alpha, its blur), all 0..1."""
    return 1.0 - _ss(0.0, 0.58, k), _ss(0.0, 0.72, k), _ss(0.34, 0.96, k), 1.0 - _ss(0.30, 1.0, k)


def preview_geom(lay, vscroll):
    """Where the Live preview sits (logical px): always in its section, scrolling with the page. Returns x, y, w, h and 0.0 (kept for callers)."""
    L = lay["preview"]
    w, h = float(PREV_W), float(lay["ph"])
    return (PW - w) / 2, L["ct"] + 4 - vscroll, w, h, 0.0


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
HOME_COLS, HOME_ROWS_VIS, DLG_ROWS = 3, 2, 5
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


def cfg_mode_changed(app):
    return app.cfg["mode"] != app._sx_mode


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

    def vbar(self, hits, box, hover, bid, x, ty, tr_h, bh, val, vmax, alpha=120, w=4):
        """A vertical scroll bar you can grab: faint track + thumb (thicker while hovered / dragged) and a hit area along the whole track."""
        by = ty + (tr_h - bh) * (val / vmax if vmax > 0 else 0.0)
        grab = hover == ("vsb", bid)
        box(x, ty, w, tr_h, (255, 255, 255, 14), 2)
        box(x - (1 if grab else 0), by, w + (2 if grab else 0), bh, (255, 255, 255, 190 if grab else alpha), 2)
        hits.append(((x - 3, ty, w + 10, tr_h), ("vsb", bid)))
        self.__dict__.setdefault("sb_geom", {})[bid] = (ty, tr_h, bh, vmax)

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
        lay = visuals_layout(st.get("aspect", 0.5625), anim, cfg["domination"], cfg["mode"])
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
                   "preview": "mirrors the screen", "motion": f"{len(vis_keys(cfg['domination'], cfg['mode'])) + len(MEDIA_KEYS)} settings",
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
                vk = vis_keys(cfg["domination"], cfg["mode"])
                nb = 7 + (1 if "kzoom" in vk else 0)
                draw_sliders(vk, ct + 4, (4, nb - 1) if cfg["domination"] else (4,), True)
                zm0 = ct + 4 + len(vk) * SL_PITCH + 12
                text("BASS ZOOM STYLE", 28, zm0 + 8, 13, WHITE, True, al=DIM_AL)
                pygame.draw.line(surf, (255, 255, 255, 40), (int(24 * s), int((zm0 + 30) * s)), (int(816 * s), int((zm0 + 30) * s)), max(1, int(s)))
                text("How the Bass zoom slider moves the picture", 816, zm0 + 8, 12, WHITE, anchor="r", al=DIM_AL)
                for k_, nm_ in enumerate(ZOOM_MODES):
                    button(24 + k_ * 160, zm0 + 40, 152, 34, nm_, ("zmode_sel", k_), cfg["zoom_mode"] == k_, 14)
                sep0 = zm0 + ZM_H
                text("BASS DISTORTION STYLE", 28, sep0 + 8, 13, WHITE, True, al=DIM_AL)
                pygame.draw.line(surf, (255, 255, 255, 40), (int(24 * s), int((sep0 + 30) * s)), (int(816 * s), int((sep0 + 30) * s)), max(1, int(s)))
                text("Hits on every bass / kick", 816, sep0 + 8, 12, WHITE, anchor="r", al=DIM_AL)
                for k_, nm_ in enumerate(BFX_NAMES):
                    button(24 + k_ * 120, sep0 + 40, 112, 34, nm_, ("bfx_sel", k_), cfg["bass_fx"] == k_, 14)
                sep = sep0 + BFX_H
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
            self.vbar(hits, box, hover, "vis", PW - 12, tr_y0, tr_h, th, st.get("vscroll", 0.0), lay["max_scroll"], 90)

    def edge_fade(self, surf, s, x, y, w, h, side, amt):
        """The row's left / right edge: what scrolls out (or in) blurs and fades away. amt 0..1 = how far from the end of the row we are (0 = at the end: no fade)."""
        if amt <= 0.003 or w < 2 or h < 2:
            return
        ex = 6                                                                               # a few pixels past the row's edge, so no sliver of a card is left sharp
        rx, ry, rw, rh = int(x * s), int(y * s), int(w * s), int(h * s)
        edge_px = rx if side == "l" else rx + rw                                             # where the row ends, in pixels
        rect = pygame.Rect(rx - (ex if side == "l" else 0), ry, rw + ex, rh).clip(surf.get_clip())
        if rect.w < 4 or rect.h < 4:
            return
        reg = surf.subsurface(rect).copy()
        small = pygame.transform.smoothscale(reg, (max(2, rect.w // 5), max(2, rect.h // 5)))
        blur = pygame.transform.smoothscale(small, (rect.w, rect.h))
        sh_rgb, sh_a = pygame.surfarray.array3d(reg).astype(np.float32), pygame.surfarray.array_alpha(reg).astype(np.float32)
        bl_rgb, bl_a = pygame.surfarray.array3d(blur).astype(np.float32), pygame.surfarray.array_alpha(blur).astype(np.float32)
        px = np.arange(rect.w, dtype=np.float32) + 0.5 + rect.x
        u = (px - edge_px) / max(1.0, rw) if side == "l" else (edge_px - px) / max(1.0, rw)   # 0 at the very edge (and beyond), 1 inside
        m = np.clip(u, 0.0, 1.0)
        m = m * m * (3.0 - 2.0 * m)
        m = 1.0 - (1.0 - m) * amt                                                            # 1 = untouched
        b = (1.0 - m)[:, None]
        rgb = sh_rgb * (1.0 - b[..., None]) + bl_rgb * b[..., None]
        al = (sh_a * (1.0 - b) + bl_a * b) * m[:, None]
        out = pygame.Surface(rect.size, pygame.SRCALPHA)
        pygame.surfarray.pixels3d(out)[:] = np.clip(rgb, 0, 255).astype(np.uint8)
        pygame.surfarray.pixels_alpha(out)[:] = np.clip(al, 0, 255).astype(np.uint8)
        surf.fill((0, 0, 0, 0), rect)
        surf.blit(out, rect.topleft)

    def style_section(self, st, hover, press, s, surf, hits, text, box, cfg, y0):
        prev = st.get("style_prev")
        hsp = st.get("hov", {})
        cw = STY_CW
        iw = int(cw - 12)
        ih = int(iw * 9 / 16)
        card_h = ih + 12 + 26
        sx = st.get("sx", 0.0)
        smax = sty_max()
        old_clip = surf.get_clip()
        ml, mr = STY_MARGIN * (1.0 - min(1.0, sx / 40.0)), STY_MARGIN * (1.0 - min(1.0, (smax - sx) / 40.0))     # at either end of the row the hovered (grown) card has room
        surf.set_clip(pygame.Rect(int((STY_X0 - ml) * s), int((y0 - 10) * s), int((STY_VW + ml + mr) * s), int((card_h + 20) * s)).clip(old_clip))
        row_hits = []
        for j, k in enumerate(MODE_ORDER):
            name = MODES[k]
            x, y = STY_X0 + j * STY_PITCH - sx, y0
            if x + cw < STY_X0 or x > STY_X0 + STY_VW:
                continue
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
            l, r = max(x, STY_X0), min(x + cw, STY_X0 + STY_VW)
            if r - l > 2:
                row_hits.append(((l, y, r - l, card_h), key))
        surf.set_clip(old_clip)
        fy, fh = y0 - 8, card_h + 16
        self.edge_fade(surf, s, STY_X0, fy, STY_FZ, fh, "l", min(1.0, sx / 40.0))
        self.edge_fade(surf, s, STY_X0 + STY_VW - STY_FZ, fy, STY_FZ, fh, "r", min(1.0, (smax - sx) / 40.0))
        hits.append(((STY_X0, y0, STY_VW, card_h), ("stylerow",)))
        hits.extend(row_hits)
        ty = y0 + card_h + 10                                              # the scroll bar
        th = sty_thumb()
        tx = STY_X0 + (STY_VW - th) * (sx / smax if smax > 0 else 0.0)
        grab = hover == ("stysb",) or press.get(("stysb",), 0.0) > 0.0 or st.get("drag_sty")
        box(STY_X0, ty, STY_VW, 6, (255, 255, 255, 16), 3)
        box(tx, ty - (1 if grab else 0), th, 6 + (2 if grab else 0), (255, 255, 255, 150 if grab else 90), 3)
        hits.append(((STY_X0, ty - 8, STY_VW, 22), ("stysb",)))

    def thumb_surface(self, tid, thumb, s, w=240, h=135, rad=14):
        key = (tid, round(s, 3), w)
        cache = self.__dict__.setdefault("_thumbs", {})
        if key in cache:
            return cache[key]
        if len(cache) > 80:
            cache.clear()
        pw, ph = int(w * s), int(h * s)
        img = pygame.image.frombuffer(np.ascontiguousarray(thumb).tobytes(), (thumb.shape[1], thumb.shape[0]), "RGB")
        img = pygame.transform.smoothscale(img, (pw, ph)).convert_alpha()
        mask = pygame.Surface((pw, ph), pygame.SRCALPHA)
        pygame.draw.rect(mask, (255, 255, 255, 255), mask.get_rect(), border_radius=max(2, int(rad * s)))
        img.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MIN)
        cache[key] = img
        return img

    def export_tab(self, st, hover, press, s, surf, hits, text, box, button):
        cfg, ex = st["cfg"], st["exp"]
        busy = ex["active"] or ex["final"]
        text("EXPORT  ·  record the scene with the song as a video", 28, 68, 13, WHITE, True, al=DIM_AL)
        box(24, 88, 792, 288, (255, 255, 255, 13), 18)
        n_hits = len(hits)

        def row(i, label, name, labels, w, gap=8):
            y = 98 + i * 40
            text(label, 44, y, 14, WHITE, vh=34, al=225)
            cur = min(max(int(cfg[name]), 0), len(labels) - 1)
            for k, lb in enumerate(labels):
                button(150 + k * (w + gap), y, w, 34, lb, ("exp_opt", name, k), cur == k, 14)

        row(0, "Resolution", "exp_res", [n for n, _ in EXP_RES], 84, 6)
        row(1, "Frame rate", "exp_fps", [f"{f} fps" for f in EXP_FPS], 84, 6)
        row(2, "Quality", "exp_q", [n for n, _ in EXP_Q], 84, 6)
        row(3, "Format", "exp_fmt", [n for n, _ in EXP_FMT], 84, 6)
        row(4, "Start from", "exp_from", ["The beginning", "Where it is now"], 140, 6)
        row(5, "Length", "exp_len", [n for n, _ in EXP_LEN], 84, 6)
        row(6, "Shape", "exp_aspect", ["16:9 widescreen", "Match the window"], 140, 6)
        w_, h_ = exp_size(cfg, *st["win"])
        pvx, pvy, pvw, pvh = EXP_PV
        box(pvx, pvy, pvw, pvh, (0, 0, 0, 200), 10)
        fmt_note = "H.264 · plays everywhere" if int(cfg["exp_fmt"]) == 0 else ("H.264 · safe if the app crashes" if int(cfg["exp_fmt"]) == 1 else "VP9 + Opus · web friendly")
        text(f"Live preview  ·  {w_} × {h_}", pvx, pvy + pvh + 6, 13, WHITE, maxw=pvw, vh=22, al=DIM_AL)
        text(fmt_note, pvx, pvy + pvh + 28, 13, WHITE, maxw=pvw, vh=22, al=DIM_AL)
        if busy:
            del hits[n_hits:]                                  # the options are locked while a recording runs

        by = 386
        button(24, by, 230, 34, "Include the song: " + ("On" if cfg["exp_audio"] else "Off"), ("exp_toggle", "exp_audio"), cfg["exp_audio"], 14)
        button(262, by, 290, 34, "GPU encoding (NVIDIA): " + ("On" if cfg["exp_gpu"] else "Off"), ("exp_toggle", "exp_gpu"), cfg["exp_gpu"], 14)
        text("Records in real time", 800, by, 13, WHITE, anchor="r", vh=34, al=DIM_AL)

        fy = 430
        text("SAVE TO", 28, fy - 2, 13, WHITE, True, al=DIM_AL)
        box(24, fy + 18, 466, 34, (255, 255, 255, G_CARD), 12)
        text(ex["folder"], 38, fy + 18, 14, WHITE, maxw=440, vh=34, al=235)
        button(498, fy + 18, 100, 34, "Change…", ("exp_dir",), False, 14)
        button(606, fy + 18, 90, 34, "Default", ("exp_dir_default",), False, 14)
        button(704, fy + 18, 112, 34, "Open folder", ("exp_open",), False, 14)

        ay = 494
        box(24, ay, 792, 82, (255, 255, 255, 18), 18)
        if ex["active"]:
            button(40, ay + 18, 190, 46, "Stop & save", ("exp_stop",), False, 16, True, r=16)
        elif ex["final"]:
            button(40, ay + 18, 190, 46, "Finishing…", ("noop",), False, 16, True, r=16)
        else:
            button(40, ay + 18, 190, 46, "Start export", ("exp_start",), not ex["blocker"], 16, True, r=16)
        tx = 250
        if busy:
            tot = max(1.0, ex["total"])
            k = max(0.0, min(1.0, ex["ts"] / tot))
            head = ("Finishing the video…" if ex["final"] else ("Paused" if ex["paused"] else "Recording")) + f"  ·  {fmt_clock(ex['ts'])} / {fmt_clock(ex['total'])}"
            text(head, tx, ay + 10, 15, WHITE, True, maxw=550, vh=24)
            box(tx, ay + 40, 546, 8, (255, 255, 255, 40), 4)
            if k > 0.004:
                box(tx, ay + 40, 546 * k, 8, (255, 255, 255, 235), 4)
            sub = f"{ex['res'][0]}×{ex['res'][1]} @ {ex['fps']}  ·  {ex['frames']} frames  ·  {fmt_bytes(ex['size'])}  ·  {ex['enc']}"
            if ex["dropped"]:
                sub += f"  ·  {ex['dropped']} repeated"
            text(sub, tx, ay + 52, 13, WHITE, maxw=550, vh=24, al=DIM_AL)
        elif ex["blocker"]:
            text(ex["blocker"], tx, ay, 14, (255, 190, 90), maxw=550, vh=82)
        elif ex["note"]:
            nt = ex["note"]
            if nt[0] == "ok":
                text("Saved  ·  " + fmt_bytes(nt[2]), tx, ay + 10, 15, WHITE, True, maxw=550, vh=24)
                text(nt[1], tx, ay + 38, 13, WHITE, maxw=550, vh=24, al=DIM_AL)
            else:
                text(nt[1], tx, ay, 14, (255, 150, 140), maxw=550, vh=82)
        else:
            text("Captures exactly what you see (no menu) with the song's own audio.", tx, ay + 10, 14, WHITE, maxw=550, vh=24, al=215)
            text("Pause and the recording waits for you. Changing songs ends it.", tx, ay + 38, 13, WHITE, maxw=550, vh=24, al=DIM_AL)

    def scenes_tab(self, st, hover, press, s, surf, hits, text, box, button, icon_button, ic_x, ic_prev, ic_next, ic_up, ic_down):
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
            text(f"{tr['no']}  \u00b7  looping {fmt_time(ts)} \u2013 {fmt_time(te)}", px, py + ph_ + 10, 14, WHITE, True)
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
            if tr["has"]:
                button(24, by_, 92, 34, "Reset trim", ("scn_trim_reset",), False, 13, r=10)
                button(244, by_, 168, 34, "Save as new scene", ("scn_newcut",), False, 13, r=10)
            else:
                box(24, by_, 92, 34, (255, 255, 255, 12), 10)
                text("Reset trim", 24 + 46, by_, 13, WHITE, anchor="c", vh=34, al=70)
                box(244, by_, 168, 34, (255, 255, 255, 12), 10)
                text("Save as new scene", 244 + 84, by_, 13, WHITE, anchor="c", vh=34, al=70)
            button(124, by_, 112, 34, "Play on screen", ("scn_play",), False, 13, r=10)
            text("Trim a scene, then save that range as its own scene.", 28, by_ + 42, 11, WHITE, maxw=384, al=110)

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
            hits.append(((x, y, w - 140, SCN_ROW_H), key))
            if r["th"] is not None:
                surf.blit(self.thumb_surface((sc["tid"], round(r["st"], 2)), r["th"], s, 64, 36, 6), (int((x + 10) * s), int((y + 5) * s)))
            else:
                box(x + 10, y + 5, 64, 36, (255, 255, 255, 22), 6)
            if hid:
                box(x + 10, y + 5, 64, 36, (0, 0, 0, 150), 6)
            text(f"{r['no']}", x + 84, y + 3, 14, WHITE, not hid, maxw=w - 224, al=120 if hid else 255)
            sub = f"{fmt_time(r['st'])}  \u00b7  {r['ln']:.1f}s" + ("  \u00b7  trimmed" if r["tr"] else "") + ("  \u00b7  hidden" if hid else "")
            text(sub, x + 84, y + 24, 12, WHITE, maxw=w - 224, al=DIM_AL)
            rk = round(r["t"], 2)
            if r["up"]:
                icon_button(x + w - 138, y + 4, 30, 18, ("scn_move", sc["id"], rk, -1), ic_up)
            if r["dn"]:
                icon_button(x + w - 138, y + 24, 30, 18, ("scn_move", sc["id"], rk, 1), ic_down)
            button(x + w - 98, y + 9, 60, 28, "Show" if hid else "Hide", ("scn_hide", sc["id"], round(r["t"], 2)), hid, 12, False, r=10)
            icon_button(x + w - 34, y + 9, 28, 28, ("scn_del", sc["id"], round(r["t"], 2)), ic_x)
        n_rows = sc["total"]
        if n_rows > SCN_ROWS:
            th = SCN_ROWS * SCN_ROW_H
            bh = max(24, th * SCN_ROWS / n_rows)
            self.vbar(hits, box, hover, "scn", 810, SCN_Y, th, bh, st["scn_scroll"], n_rows - SCN_ROWS)
        yb = SCN_Y + SCN_ROWS * SCN_ROW_H + 18
        button(24, yb, 150, 38, "Hide all", ("scn_all", "hide_all"), False, 14)
        button(182, yb, 190, 38, f"Show hidden ({sc['hidden']})", ("scn_all", "show_all"), False, 14)
        button(380, yb, 206, 38, f"Restore deleted ({sc['deleted']})", ("scn_all", "restore"), False, 14)
        order = st["cfg"]["scene_order"]
        button(594, yb, 118, 38, "Play in order", ("scn_order", True), order, 13)
        button(716, yb, 100, 38, "Random", ("scn_order", False), not order, 13)
        text("Click a scene to preview it.  Hidden scenes stay listed but never play; deleted ones leave the list (restore any time).  Arrows re-order the list.", 28, yb + 50, 12, WHITE, maxw=788, al=DIM_AL)
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
            self.vbar(hits, box, hover, "mstack", 810, ly + 4, th, bh, sc, n - STACK_ROWS)
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
        button(24, 503 + o, 142, 34, "Cut on hit: " + ("ON" if hit else "OFF"), ("toggle", "media_hit"), hit, 13)
        sw = cfg["media_sway"]
        button(172, 503 + o, 112, 34, "Sway: " + ("ON" if sw else "OFF"), ("toggle", "media_sway"), sw, 13)
        sm_ = cfg["media_smooth"]
        button(290, 503 + o, 154, 34, "Smooth video: " + ("ON" if sm_ else "OFF"), ("toggle", "media_smooth"), sm_, 13)
        for k, name in enumerate(("Fade", "Blur", "Instant", "Zoom", "Random")):
            button(452 + k * 73, 503 + o, 69, 34, name, ("mstyle", k), cfg["media_style"] == k, 13)
        text("Smooth video: motion interpolation adds in-between frames.  Sway = handheld camera on hits.  V = next scene.", 28, 543 + o, 13, WHITE, al=DIM_AL)

    def home_tab(self, st, hover, press, s, surf, hits, text, box, button, icon_button, ic_x, ic_trash, ic_save):
        """The start screen: new / open / resume buttons and a grid of recent-project cards with thumbnails."""
        h = st["home"]
        items, row0 = h["items"], h["row"]
        pj = st["proj"] or dict(name="")
        button(24, 76, 170, 42, "New project", ("proj_new",), True, 15, True, r=16)
        button(204, 76, 170, 42, "Open project\u2026", ("proj_open",), False, 15, True, r=16)
        if not st["in_session"]:
            text("Or drop music / video files anywhere to start right away.", 392, 76, 13, WHITE, vh=42, al=DIM_AL)
        text("RECENT PROJECTS", 28, 134, 13, WHITE, True, al=DIM_AL)
        n = len(items)
        cw, ch, gap = 256, 224, 14
        if n == 0:
            box(24, 158, 792, 440, (255, 255, 255, 13), 20)
            text("No projects yet", PW / 2, 330, 20, WHITE, True, "c")
            text("Start a new project, or drop music files onto the window.", PW / 2, 366, 14, WHITE, anchor="c", al=DIM_AL)
            text("Save one from Settings \u203a Project and it will show up here.", PW / 2, 390, 14, WHITE, anchor="c", al=DIM_AL)
        for idx in range(row0 * HOME_COLS, min(n, (row0 + HOME_ROWS_VIS) * HOME_COLS)):
            it = items[idx]
            col, r = idx % HOME_COLS, idx // HOME_COLS - row0
            x, y = 24 + col * (cw + 12), 158 + r * (ch + gap)
            key, fkey, tkey = ("proj_card", idx), ("proj_forget", idx), ("proj_trash", idx)
            hov = hover in (key, fkey, tkey, ("proj_tsave",), ("proj_ttrash",))
            pp = max(0.0, press.get(key, 0.0))
            c = lerp_col((255, 255, 255, G_HOVER - 14 if hov else G_CARD), (38, 38, 48, 125), pp)
            inset = 3.0 * pp
            box(x + inset, y + inset * 0.6, cw - 2 * inset, ch - inset * 1.2, c, 16)
            if it["thumb"] is not None:
                img = self.thumb_surface(("proj", it["path"], it["saved"]), it["thumb"], s, 240, 135, 12)
                surf.blit(img, (int((x + 8) * s), int((y + 8) * s)))
            else:
                box(x + 8, y + 8, 240, 135, (255, 255, 255, 18), 12)
                text("\u266A", x + 128, y + 8, 40, WHITE, True, "c", vh=135, al=90)
            text(it["name"], x + 14, y + 152, 16, WHITE, True, maxw=228)
            tr, cl = it["tracks"], it["clips"]
            text(f"{tr} track{'s' if tr != 1 else ''}  \u00b7  {cl} clip{'s' if cl != 1 else ''}", x + 14, y + 177, 13, WHITE, maxw=228, al=DIM_AL + 50)
            text("Not saved yet" if it.get("temp") else "Opened " + fmt_ago(it["used"]), x + 14, y + 198, 12, WHITE, maxw=228, al=DIM_AL - 20)
            if it["missing"]:
                m = it["missing"]
                box(x + 16, y + 16, 104, 22, (255, 176, 64, 238), 11)
                text(f"! {m} missing", x + 68, y + 16, 12, DARK, True, "c", vh=22)
            hits.append(((x, y, cw, ch), key))
            if it.get("temp"):
                icon_button(x + cw - 76, y + ch - 40, 30, 28, ("proj_tsave",), ic_save)         # save the unsaved session as a project
                icon_button(x + cw - 42, y + ch - 40, 30, 28, ("proj_ttrash",), ic_trash)        # discard it (after asking)
                continue
            if hov:
                icon_button(x + cw - 40, y + 14, 28, 26, fkey, ic_x)
            icon_button(x + cw - 42, y + ch - 40, 30, 28, tkey, ic_trash)          # permanently deletes this project file (after asking)
        rows = self.home_rows_for(n)
        if rows > HOME_ROWS_VIS:
            th = HOME_ROWS_VIS * ch + (HOME_ROWS_VIS - 1) * gap
            bh = max(30, th * HOME_ROWS_VIS / rows)
            self.vbar(hits, box, hover, "home", 822, 158, th, bh, row0, rows - HOME_ROWS_VIS)
        ns = sum(1 for it_ in items if not it_.get("temp"))
        if ns:
            text(f"{ns} project{'s' if ns != 1 else ''}" + ("   \u00b7   scroll for more" if rows > HOME_ROWS_VIS else ""), 28, 628, 12, WHITE, al=DIM_AL - 20)

    @staticmethod
    def home_rows_for(n):
        return (n + HOME_COLS - 1) // HOME_COLS

    def notice_pill(self, st, s, surf, box):
        """A small confirmation banner inside the menu (e.g. "Saved My project")."""
        nt = st.get("notice")
        if not nt:
            return
        msg, a, slide = nt
        f = self.font(max(8, int(14 * s)), True)
        tw = f.size(msg)[0] / s
        w = tw + 62
        x, y = (PW - w) / 2, 70 - 10 * (1.0 - slide)
        layer = pygame.Surface(surf.get_size(), pygame.SRCALPHA)
        old = surf
        pygame.draw.rect(layer, (22, 24, 38, 244), R_(x, y, w, 36, s), border_radius=int(18 * s))
        pygame.draw.rect(layer, (255, 255, 255, 70), R_(x, y, w, 36, s), max(1, int(s)), border_radius=int(18 * s))
        cx, cy = x + 20, y + 18
        pygame.draw.circle(layer, (30, 215, 96, 255), (int(cx * s), int(cy * s)), int(9 * s))
        pygame.draw.lines(layer, (14, 30, 20), False, [(int((cx - 4.2) * s), int(cy * s)), (int((cx - 1.2) * s), int((cy + 3) * s)), (int((cx + 4.4) * s), int((cy - 3.4) * s))], max(2, int(2 * s)))
        img = f.render(msg, True, WHITE)
        layer.blit(img, (int((x + 36) * s), int((y + 18) * s - img.get_height() / 2)))
        layer.set_alpha(int(255 * a))
        old.blit(layer, (0, 0))

    def dialog(self, st, hover, press, s, surf, hits, text, box, button):
        """Modal box over the panel (drawn last, so its hit areas win)."""
        d = st.get("dlg")
        if not d:
            return
        hits.append(((0, 0, PW, PH), ("noop", "dlg")))
        box(0, 0, PW, PH, (6, 6, 14, 175), 30)
        items, lines = d["items"], d["lines"]
        vis = min(len(items), DLG_ROWS)
        bw = 640
        bh = 66 + len(lines) * 22 + 10 + ((vis * 46 + 14) if items else 0) + (62 if d["buttons"] else 22)
        x, y = (PW - bw) / 2, (PH - bh) / 2
        box(x, y, bw, bh, (34, 36, 56, 250), 22)
        pygame.draw.rect(surf, (255, 255, 255, 46), R_(x, y, bw, bh, s), max(1, int(s)), border_radius=int(22 * s))
        text(d["title"], x + 28, y + 22, 20, WHITE, True)
        cy = y + 62
        for i, ln in enumerate(lines):
            text(ln, x + 28, cy, 14, WHITE, maxw=bw - 56, al=255 if i == 0 else DIM_AL + 50)
            cy += 22
        if items:
            sc = d["scroll"]
            cy += 10
            for i, (nm, folder, orig) in enumerate(items[sc:sc + DLG_ROWS]):
                ry = cy + i * 46
                box(x + 20, ry, bw - 40, 42, (255, 255, 255, 16), 12)
                text(nm, x + 34, ry + 3, 14, WHITE, True, maxw=400)
                text(folder or "(unknown folder)", x + 34, ry + 22, 11, WHITE, maxw=400, al=DIM_AL)
                button(x + bw - 138, ry + 7, 100, 28, "Locate\u2026", ("dlg_loc", sc + i), False, 13)
            if len(items) > DLG_ROWS:
                th = DLG_ROWS * 46 - 4
                bh2 = max(20, th * DLG_ROWS / len(items))
                self.vbar(hits, box, hover, "dlg", x + bw - 22, cy, th, bh2, sc, len(items) - DLG_ROWS)
        bs = d["buttons"]
        if bs:
            tot = len(bs) * 160 - 10
            bx = x + bw - 24 - tot
            for k, (lab, prim, dg) in enumerate(bs):
                button(bx + k * 160, y + bh - 56, 150, 38, lab, ("dlg", k), prim, 14, True, r=14, danger=dg and not prim)

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
                elif name == "home":
                    poly([(12.0, 2.8), (22.0, 11.6), (19.6, 11.6), (4.4, 11.6), (2.0, 11.6)])
                    pygame.draw.rect(big, c, (5.2 * u, 10.5 * u, 13.6 * u, 10.2 * u), border_radius=int(1.4 * u))
                    pygame.draw.rect(big, (0, 0, 0, 0), (10.0 * u, 14.2 * u, 4.0 * u, 6.6 * u), border_radius=int(1.0 * u))
                elif name == "save":
                    pygame.draw.rect(big, c, (3.5 * u, 3.0 * u, 17.0 * u, 18.0 * u), border_radius=int(2.2 * u))
                    pygame.draw.rect(big, (0, 0, 0, 0), (7.5 * u, 3.0 * u, 9.0 * u, 6.2 * u))
                    pygame.draw.rect(big, c, (13.4 * u, 4.0 * u, 2.2 * u, 4.2 * u), border_radius=int(0.5 * u))
                    pygame.draw.rect(big, (0, 0, 0, 0), (7.0 * u, 12.6 * u, 10.0 * u, 8.4 * u), border_radius=int(1.0 * u))
                    pygame.draw.rect(big, c, (8.6 * u, 14.6 * u, 6.8 * u, 1.5 * u))
                    pygame.draw.rect(big, c, (8.6 * u, 17.4 * u, 6.8 * u, 1.5 * u))
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

        def ic_trash(cx, cy, col):
            w = max(2, int(1.8 * s))
            pygame.draw.line(surf, col, (int((cx - 7.5) * s), int((cy - 4.5) * s)), (int((cx + 7.5) * s), int((cy - 4.5) * s)), w)
            pygame.draw.lines(surf, col, False, P([(cx - 2.5, cy - 4.5), (cx - 2.5, cy - 7.5), (cx + 2.5, cy - 7.5), (cx + 2.5, cy - 4.5)]), w)
            pygame.draw.lines(surf, col, False, P([(cx - 5.5, cy - 2.5), (cx - 4.6, cy + 7.5), (cx + 4.6, cy + 7.5), (cx + 5.5, cy - 2.5)]), w)
            pygame.draw.line(surf, col, (int((cx - 1.6) * s), int((cy - 0.5) * s)), (int((cx - 1.6) * s), int((cy + 5) * s)), max(1, int(1.4 * s)))
            pygame.draw.line(surf, col, (int((cx + 1.6) * s), int((cy - 0.5) * s)), (int((cx + 1.6) * s), int((cy + 5) * s)), max(1, int(1.4 * s)))

        def ic_save(cx, cy, col):
            w = max(2, int(1.6 * s))
            pygame.draw.rect(surf, col, (int((cx - 7.5) * s), int((cy - 7.5) * s), int(15 * s), int(15 * s)), w, border_radius=int(2.2 * s))
            pygame.draw.rect(surf, col, (int((cx - 3.8) * s), int((cy - 7.5) * s), int(7.6 * s), int(4.6 * s)), w)
            pygame.draw.rect(surf, col, (int((cx - 4.6) * s), int((cy + 0.8) * s), int(9.2 * s), int(6.7 * s)), w)

        def draw_sliders(keys, y0, dividers, toggles=False):
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
                fx_off = toggles and key in (cfg.get("fx_off") or "").split(",")
                fa = st.get("fxa", {}).get(key, 0.0 if fx_off else 1.0) if toggles else 1.0           # 0 = off .. 1 = on (eased)
                if toggles:                                                    # the on / off light for this effect
                    dcx, dcy = 36, y + SL_PITCH / 2
                    dh = hover == ("fxdot", key)
                    pygame.draw.circle(surf, (255, 255, 255, int((110 if dh else 60) * (1.0 - fa))), (int(dcx * s), int(dcy * s)), max(2, int(5 * s)), max(1, int(1.5 * s)))
                    if fa > 0.02:
                        pygame.draw.circle(surf, (255, 255, 255, int(38 * fa)), (int(dcx * s), int(dcy * s)), max(2, int((4.0 + 4.5 * fa) * s)))
                        pygame.draw.circle(surf, (255, 255, 255, int((255 if dh else 235) * fa)), (int(dcx * s), int(dcy * s)), max(1, int(4.5 * fa * s)))
                    hits.append(((24, y, 22, SL_PITCH), ("fxdot", key)))
                text(label, 46, y, 15, WHITE, vh=SL_PITCH, al=int(90 + 148 * fa))
                def_t = slider_norm(key, DEFAULTS[key])
                gsig = (round(s, 3), round(t_disp, 4), round(sc_disp, 3), grab, hov, def_t)
                hit = self.sl_cache.get(key)
                if hit and hit[0] == gsig:
                    gfx = hit[1]
                else:
                    gfx = self.slider_surface(s, t_disp, sc_disp, grab, hov, def_t, TRACK_W + 2 * SL_PAD, SL_PITCH)
                    self.sl_cache[key] = (gsig, gfx)
                gfx.set_alpha(int(60 + 195 * fa))
                surf.blit(gfx, (int((TRACK_X0 - SL_PAD) * s), int(y * s)))
                modified = abs(v - DEFAULTS[key]) > 1e-6 and not fx_off
                button(700, y + 4.5, 88, 24, fmt_slider(key, v), ("sl_reset", key), modified, 13, False,
                       (255, 255, 255, 80), (255, 255, 255, 124), 12, WHITE)
                if not fx_off:
                    hits.append(((TRACK_X0 - SL_PAD, y, TRACK_W + 2 * SL_PAD, SL_PITCH), ("slider", key)))

        # ---- header
        text("Hypnosis", 30, 12, 26, WHITE, True, vh=40)
        if tab == "home":
            text("Home", 164, 12, 15, WHITE, vh=40, al=DIM_AL)
            if st["in_session"]:
                button(722, 13, 96, 36, "Resume", ("proj_resume",), False, 14, True, r=18)
        else:
            for k, (nm, tk) in enumerate((("Queue", "queue"), ("Visuals", "visuals"),
                                          ("Scenes", "scenes"), ("Beat", "beat"), ("Export", "export"), ("Settings", "settings"))):
                button(158 + k * 83, 13, 79, 36, nm, ("tab", tk), tab == tk, 13, True, r=18)
            round_btn(702, 31, 32, ("home",), "home", isz=21)
            round_btn(738, 31, 32, ("proj_save",), "save", isz=21)
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
                self.vbar(hits, box, hover, "queue", 810, LIST_Y, th, bh, scroll, n - LIST_ROWS)

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

            text("PROJECT", 28, 267 + o, 13, WHITE, True, al=DIM_AL)
            box(24, 287 + o, 792, 84, (255, 255, 255, 13), 16)
            pj = st["proj"] or dict(name="Untitled New Project", named=False, dirty=False, missing=0)
            text(pj["name"], 40, 292 + o, 15, WHITE, True, maxw=330, vh=30)
            note, ncol = ("", WHITE)
            if pj["missing"]:
                note, ncol = f"{pj['missing']} missing file{'s' if pj['missing'] != 1 else ''} kept in the project", (255, 190, 90)
            elif pj["dirty"]:
                note, ncol = ("Unsaved changes" if pj["named"] else "Not saved yet"), (255, 214, 120)
            elif pj["named"]:
                note, ncol = "Saved", WHITE
            text(note, 800, 292 + o, 13, ncol, anchor="r", maxw=400, vh=30, al=255 if ncol != WHITE else DIM_AL)
            button(36, 326 + o, 176, 34, "Save project", ("proj_save",), pj["dirty"] and pj["named"], 14)
            button(220, 326 + o, 140, 34, "Save as\u2026", ("proj_saveas",), False, 14)
            button(368, 326 + o, 170, 34, "Load project\u2026", ("proj_load",), False, 14)
            button(546, 326 + o, 130, 34, "Home", ("home",), False, 14)
            button(686, 326 + o, 118, 34, "Exit app", ("exit",), False, 14, True, r=12, danger=True)
            text("Space play  \u00b7  Esc menu  \u00b7  F fullscreen  \u00b7  M style  \u00b7  D domination  \u00b7  S Spotify  \u00b7  V scene  \u00b7  N/P track  \u00b7  \u2190/\u2192 seek  \u00b7  \u2191/\u2193 volume  \u00b7  H hints  \u00b7  Q quit",
                 28, 377 + o, 12, WHITE, maxw=788, al=DIM_AL)

            text("PRESETS", 28, 401 + o, 13, WHITE, True, al=DIM_AL)
            plist, psel = st["presets"], st["preset_sel"]
            pop = st["preset_open"]
            button(24, 421 + o, 300, 38, ("\u25BE  " + psel) if psel else "No presets saved yet", ("preset_dd",), pop, 14)
            button(332, 421 + o, 140, 38, "Save as new", ("preset_save",), False, 14)
            button(480, 421 + o, 110, 38, "Update", ("preset_update",), False, 14)
            button(598, 421 + o, 150, 38, "Click to confirm" if st["preset_del"] else "Delete", ("preset_delete",), False, 14, danger=True)
            text("Saves every slider, toggle and style (not your clips or queue) to presets.json.", 28, 465 + o, 13, WHITE, al=DIM_AL)

            armed = st["reset_armed"]
            button(24, 499 + o, 220, 34, "Click again to confirm" if armed else "Reset all settings", ("reset_all",),
                   False, 14, danger=True)
            sarmed = st["scache_armed"]
            button(252, 499 + o, 250, 34, "Click again to confirm" if sarmed else "Clear all scene cache", ("scan_clear",),
                   False, 14, danger=True)
            text("Reset: sliders and toggles back to default.", 514, 499 + o, 12, WHITE, vh=17, al=DIM_AL)
            text("Cache: old clips get scanned again. Loaded ones stay.", 514, 499 + o + 17, 12, WHITE, vh=17, al=DIM_AL)

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
            self.scenes_tab(st, hover, press, s, surf, hits, text, box, button, icon_button, ic_x, ic_prev, ic_next, ic_up, ic_down)
        elif tab == "home":
            pass
        elif tab == "export":
            self.export_tab(st, hover, press, s, surf, hits, text, box, button)
        else:
            self.beat_tab(st, s, surf, text, box, draw_sliders, cfg)

        if tab == "home":
            self.home_tab(st, hover, press, s, surf, hits, text, box, button, icon_button, ic_x, ic_trash, ic_save)
            self.notice_pill(st, s, surf, box)
            self.dialog(st, hover, press, s, surf, hits, text, box, button)
            self.hits = hits
            return surf, s

        # ---- now-playing bar (a Spotify-style player in the same glass): art + song | shuffle prev play next repeat + seek | volume
        line(FTR_Y)
        spot = st["spotify"]
        names, cur = st["names"], st["cur"]
        by0 = FTR_Y + 8
        x0 = 24
        dk = st.get("dock", 0.0)
        main_surf = surf
        if dk > 0.004:                                         # art + title draw on their own layer so they can blur out
            surf = pygame.Surface(main_surf.get_size(), pygame.SRCALPHA)
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
        if dk > 0.004:
            a_out, b_out, a_in, b_in = dock_phases(dk)
            reg = pygame.Rect(int(14 * s), int((FTR_Y + 2) * s), int(290 * s), int(74 * s)).clip(main_surf.get_rect())

            def soft(layer, blur, alpha):
                if alpha <= 0.004 or reg.w < 4 or reg.h < 4:
                    return
                part = layer.subsurface(reg).copy()
                f_ = 1 + int(round(blur * 9))
                if f_ > 1:
                    part = pygame.transform.smoothscale(pygame.transform.smoothscale(part, (max(2, reg.w // f_), max(2, reg.h // f_))), reg.size)
                part.set_alpha(int(255 * alpha))
                main_surf.blit(part, reg.topleft)

            soft(surf, b_out, a_out)                           # the art + song name melt away ...
            surf = pygame.Surface(main_surf.get_size(), pygame.SRCALPHA)
            cx_ = x0 + max(64.0, min(150.0, DOCK_H * PREV_W / max(1.0, float(st.get("prev_ph", 236))))) + 16   # ... and a compact art + title melts in beside the mini preview
            AS = 36                                            # compact album art + song / artist beside the mini preview
            if spot and self.art_surf is not None and self.art_surf[1] is not None:
                sm_ = pygame.transform.smoothscale(self.art_surf[1], (int(AS * s), int(AS * s)))
                surf.blit(sm_, (int(cx_ * s), int((by0 + 12) * s)))
            else:
                box(cx_, by0 + 12, AS, AS, (30, 215, 96, 46) if spot else (255, 255, 255, 30), 7)
                text("\u266A", cx_ + AS / 2, by0 + 12, 18, (30, 215, 96) if spot else WHITE, True, "c", vh=AS, al=255 if spot else 150)
            tx_ = cx_ + AS + 8
            text(song, tx_, by0 + 11, 13, WHITE, True, maxw=292 - tx_)
            text(sub.replace("Queue  \u00b7  track", "Track"), tx_, by0 + 31, 11, subcol, False, maxw=292 - tx_, al=235 if subcol != WHITE else DIM_AL)
            soft(surf, b_in, a_in)
            surf = main_surf

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

        self.notice_pill(st, s, surf, box)
        self.dialog(st, hover, press, s, surf, hits, text, box, button)
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


class ExampleHead:
    """Plays the built-in example clip against the wall clock (rendered live on the GPU, 30 frames a second)."""
    ready = True
    FPS = 30.0

    def __init__(self, src):
        self.src = src
        self.pos0, self.t0 = 0.0, time.perf_counter()
        self.paused, self._acc, self.rate = False, 0.0, 1.0
        self.serial, self._last, self.frame = 0, -1, None
        self._ppos = None
        self.st = dict(x=None, y=None, vx=0.0, vy=0.0, t=random.uniform(0.0, 6.0), hue=None)

    def play_from(self, t, paused=False):
        self.pos0, self.t0 = float(t), time.perf_counter()
        self.paused, self._acc = bool(paused), 0.0
        self._last, self._ppos, self.frame = -1, None, None

    def _elapsed(self):
        return self._acc if self.paused else self._acc + (time.perf_counter() - self.t0) * self.rate

    def _pos(self):
        return (self.pos0 + self._elapsed()) % self.src.duration

    def set_rate(self, r):
        self._acc = self._elapsed()
        self.t0 = time.perf_counter()
        self.rate = max(0.05, float(r))

    def set_paused(self, p):
        if p and not self.paused:
            self._acc = self._elapsed()
            self.paused = True
        elif not p and self.paused:
            self.t0 = time.perf_counter()
            self.paused = False

    def resume(self):
        self.set_paused(False)

    def get(self):
        pos = self._pos()
        n = int(pos * self.FPS)
        if self.frame is None or (not self.paused and n != self._last):
            self._last = n
            dt = 0.0 if self._ppos is None else (pos - self._ppos) % self.src.duration
            self._ppos = pos
            self.frame = self.src.render(self.st, pos, min(dt, 0.25))
            self.serial += 1
        return self.serial, self.frame

    def stop(self):
        pass


class ExampleSource:
    """The Video style's stand-in clip while the stack is empty: 5 scenes of bouncing, spinning 3D "VIDEO EXAMPLE" text over
    different moving backgrounds, so every media effect and scene change can be seen without a real clip."""
    kind = "example"
    SCENE_S = 6.0
    # background kind, top colour, bottom colour, accent, text hue, speed
    LOOKS = (
        (0, (0.0, 0.0, 0.0), (0.02, 0.02, 0.05), (0.2, 0.2, 0.3), 0.55, 1.0),
        (1, (0.02, 0.03, 0.18), (0.0, 0.0, 0.06), (0.15, 0.45, 1.0), 0.05, 1.25),
        (2, (0.17, 0.02, 0.13), (0.04, 0.0, 0.05), (0.9, 0.2, 0.6), 0.30, 0.9),
        (3, (0.0, 0.14, 0.13), (0.0, 0.03, 0.05), (0.1, 0.8, 0.7), 0.82, 1.1),
        (4, (0.18, 0.10, 0.0), (0.05, 0.02, 0.0), (1.0, 0.65, 0.1), 0.15, 0.8),
    )
    BG_VERT = """
#version 330
in vec2 p;
out vec2 vUV;
void main(){ vUV = p * 0.5 + 0.5; gl_Position = vec4(p, 0.0, 1.0); }
"""
    BG_FRAG = """
#version 330
uniform float uT, uAsp;
uniform int uKind;
uniform vec3 uC1, uC2, uAc;
in vec2 vUV;
out vec4 fragColor;
void main(){
    vec2 q = vec2((vUV.x - 0.5) * uAsp, vUV.y - 0.5);
    vec3 col = mix(uC2, uC1, vUV.y);
    float m = 0.0;
    if (uKind == 1) { vec2 g = fract((q + vec2(uT * 0.05, uT * 0.03)) * 8.0); m = step(g.x, 0.04) + step(g.y, 0.04); }
    else if (uKind == 2) { m = 0.6 * step(0.5, fract((q.x + q.y) * 7.0 - uT * 0.25)); }
    else if (uKind == 3) { m = smoothstep(0.75, 1.0, 0.5 + 0.5 * sin(length(q) * 28.0 - uT * 2.0)); }
    else if (uKind == 4) { vec2 g = fract(q * 10.0 + vec2(uT * 0.1, 0.0)) - 0.5; m = smoothstep(0.22, 0.18, length(g)); }
    col = mix(col, uAc, clamp(m, 0.0, 1.0) * 0.5);
    fragColor = vec4(col, 1.0);
}
"""

    def __init__(self, ctx, w=960, h=540, tscale=1.0):
        self.ctx = ctx
        self.w, self.h = w, h
        self.orig = (self.w, self.h)
        self.duration = self.SCENE_S * len(self.LOOKS)
        self.scenes = [self.SCENE_S * i for i in range(len(self.LOOKS))]
        self.scan, self.nframes, self.sthumb_n = 1.0, int(self.duration * 30), 0
        self.thumb = np.zeros((135, 240, 3), np.uint8)
        self.fbo = ctx.simple_framebuffer((self.w, self.h))
        self.prog = ctx.program(vertex_shader=self.BG_VERT, fragment_shader=self.BG_FRAG)
        self.vbo = ctx.buffer(np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype="f4").tobytes())
        self.vao = ctx.vertex_array(self.prog, [(self.vbo, "2f", "p")])
        self.bt = BounceText(ctx)
        self.bt.keep_alpha = True
        self.bt.tscale = tscale

    def scene_thumb(self, t):
        return None

    def make_head(self):
        return ExampleHead(self)

    def render(self, st, pos, dt):
        ctx = self.ctx
        look = self.LOOKS[int(pos / self.SCENE_S) % len(self.LOOKS)]
        prev, vp = ctx.fbo, ctx.viewport
        self.fbo.use()
        ctx.viewport = (0, 0, self.w, self.h)
        pr = self.prog
        pr["uT"].value, pr["uAsp"].value, pr["uKind"].value = float(pos), self.w / float(self.h), look[0]
        pr["uC1"].value, pr["uC2"].value, pr["uAc"].value = look[1], look[2], look[3]
        self.vao.render(moderngl.TRIANGLE_STRIP)
        bt = self.bt
        if st["hue"] is None:
            st["hue"] = look[4]
        bt.x, bt.y, bt.vx, bt.vy, bt.t, bt.hue = st["x"], st["y"], st["vx"], st["vy"], st["t"], st["hue"]
        bt.draw(dt * look[5], self.w, self.h)
        st.update(x=bt.x, y=bt.y, vx=bt.vx, vy=bt.vy, t=bt.t, hue=bt.hue)
        raw = np.frombuffer(self.fbo.read(components=3), dtype=np.uint8).reshape(self.h, self.w, 3)
        prev.use()
        ctx.viewport = vp
        return np.ascontiguousarray(raw[::-1])

    def close(self):
        pass


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


_SCAN_Q, _SCAN_CV, _SCAN_T = [], threading.Condition(), [None]


def _scan_enqueue(src):
    """Scene scans run strictly one after another (each already uses every core), so a batch of dropped videos can't starve
    or fail each other, and the loading cover can follow them one by one."""
    with _SCAN_CV:
        _SCAN_Q.append(src)
        if _SCAN_T[0] is None or not _SCAN_T[0].is_alive():
            _SCAN_T[0] = threading.Thread(target=_scan_worker, daemon=True)
            _SCAN_T[0].start()
        _SCAN_CV.notify()


def _scan_worker():
    while True:
        with _SCAN_CV:
            while not _SCAN_Q:
                if not _SCAN_CV.wait(timeout=20.0) and not _SCAN_Q:
                    _SCAN_T[0] = None
                    return
            src = _SCAN_Q.pop(0)
        if src._stop:
            continue
        try:
            src.scanning = True
            src._scan()
        except Exception:
            src.scan = 1.0
        finally:
            src.scanning = False


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
        self.scanning, self._scan_err = False, False
        _scan_enqueue(self)                                           # one video is scanned at a time, in the order they were added

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
        """Use the remembered scene list when this exact file was scanned before, else scan (and remember the result)."""
        try:
            cached = scan_cache_get(self.path, self.duration) if 0 < self.duration <= 6 * 3600 else None
        except Exception:
            cached = None
        if cached and not (len(cached) <= 1 and self.duration > 90):      # a lone "scene 0" on a long clip is a failed scan, not a result
            self.scenes, self.scan = cached, 1.0
            return
        self._scan_impl()
        if not self._stop and not self._scan_err and 0 < self.duration <= 6 * 3600 and self.scenes:
            try:
                scan_cache_put(self.path, self.duration, self.scenes)
            except Exception:
                pass

    def _scan_impl(self):
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
                    if not cap.isOpened():
                        self._scan_err = True
                    start = max(0, f0 - step)                         # one extra sample before the chunk so a cut on the border is seen
                    if start > 0:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
                    idx, prev = start, None
                    while not self._stop and idx < f1:
                        if not cap.grab():
                            if idx < f1 - 2 * step:
                                self._scan_err = True            # ended long before its chunk did: don't remember this result
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
                    self._scan_err = True
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
        self.order_play = False                  # True: scenes play in list order instead of at random
        self.fade_s = self.FADE_S
        self.cur_style = 0
        self._lq, self._lcv, self._lthread = [], threading.Condition(), None      # loading queue (one clip at a time)
        self.want_example = False        # the app asks for the built-in example clip (Video style with an empty stack)
        self.ex_on = False
        self.ex_it = None

    def _sync_example(self, now):
        """The hidden example clip plays while the stack is empty and is dropped as soon as a real clip is added."""
        want = self.want_example and not self.items
        if want and not self.ex_on:
            if self.ex_it is None:
                self.ex_it = dict(id=-1, path="<example>", src=ExampleSource(self.ctx), status="ok", example=True)
            self.ex_on = True
            self._refresh(now)
        elif not want and self.ex_on:
            self.ex_on = False
            self._drop_heads()

    # ---- the stack
    def paths(self):
        return [it["path"] for it in self.items]

    def _real(self):
        return [it for it in self.items if it["src"] is not None]

    def _ok(self):
        real = self._real()
        return real or ([self.ex_it] if (self.ex_on and not self.items and self.ex_it is not None) else [])

    def add(self, paths):
        have = set(self.paths())
        for path in paths:
            if path in have:
                continue
            have.add(path)
            it = dict(id=self._next_id, path=path, src=None, status="loading")
            self._next_id += 1
            self.items.append(it)
            with self._lcv:
                self._lq.append((it["id"], path))
                if self._lthread is None or not self._lthread.is_alive():
                    self._lthread = threading.Thread(target=self._loader, daemon=True)
                    self._lthread.start()
                self._lcv.notify()

    def _loader(self):
        """Media is loaded strictly one at a time: a clip is opened and its scenes are found before the next one starts."""
        while True:
            with self._lcv:
                while not self._lq:
                    if not self._lcv.wait(timeout=20.0) and not self._lq:
                        self._lthread = None
                        return
                iid, path = self._lq.pop(0)
            if not any(x["id"] == iid for x in self.items):          # removed while waiting its turn
                continue
            src = self._work(iid, path)
            while (src is not None and getattr(src, "kind", "") == "video" and src.scan < 1.0 and not src._stop
                   and any(x["id"] == iid for x in self.items)):
                time.sleep(0.1)

    def _work(self, iid, path):
        try:
            src, err = open_source(path), None
        except Exception as e:
            src, err = None, (str(e) or e.__class__.__name__)
        with self._plock:
            self._pending.append((iid, src, err))
        return src

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
        with self._lcv:
            self._lq.clear()

    def restart(self, now):
        """Start playback over from scratch (after a scan has found every scene)."""
        self._drop_heads()
        self._refresh(now)

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
        if self.ex_on and self.items:
            self._sync_example(now)
        oks = self._ok()
        if not oks:
            if self.cur is not None:
                self._drop_heads()
            return
        if self.cur is None:
            seq = self._seq() if self.order_play else []
            if seq:
                it, t = seq[0]
            else:
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

    def _pfw(self, it):
        """This clip's scene edits, created (with every field) when missing."""
        p = self.prefs.setdefault(it["path"], {})
        for k_ in ("off", "del", "trim", "custom", "order"):
            p.setdefault(k_, [])
        return p

    def set_order(self, on):
        self.order_play = bool(on)

    # ---- the scene list: detected scenes + cuts the user made, in the user's order
    def custom_of(self, it, t):
        for k, a, b in ((self._pf(it) or {}).get("custom") or []):
            if abs(k - t) <= SCENE_TOL:
                return a, b
        return None

    def scene_keys(self, it):
        """Every scene's key (its detected start time, or the key of a user-made cut) in play / list order."""
        pf = self._pf(it) or {}
        default = sorted(it["src"].scenes) + [c[0] for c in (pf.get("custom") or [])]
        order = pf.get("order") or []
        if not order:
            return default
        out, left = [], list(default)
        for k in order:
            m = next((d for d in left if abs(d - k) <= SCENE_TOL), None)
            if m is not None:
                out.append(m)
                left.remove(m)
        return out + left

    def scene_label(self, it, t):
        cu = [c[0] for c in ((self._pf(it) or {}).get("custom") or [])]
        for i, k in enumerate(cu):
            if abs(k - t) <= SCENE_TOL:
                return f"Cut {i + 1}"
        sc = sorted(it["src"].scenes)
        return f"Scene {next((i + 1 for i, x in enumerate(sc) if abs(x - t) <= SCENE_TOL), 0)}"

    def active_scenes(self, it):
        return [t for t in self.scene_keys(it) if self.flag(it, t) == 0]

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

    def _seq(self):
        """Every playable scene of every clip, in play order: clips in stack order, each clip's scenes in its list order."""
        out = []
        for it in self._playable():
            if it["src"].kind == "image":
                out.append((it, 0.0))
                continue
            pool = self.active_scenes(it) if self._pf(it) else sorted(self._pool(it))
            d = it["src"].duration
            out += [(it, t) for t in pool if d <= 0 or t < d - 0.5 or self.custom_of(it, t)]
        return out

    def _pick_next(self):
        oks = self._playable()
        cur_id = self.cur_it["id"] if self.cur_it else None
        it, ft = None, None
        if self._focus is not None:
            it = next((x for x in oks if x["id"] == self._focus), None) or next((x for x in self._ok() if x["id"] == self._focus), None)
            ft = self._force_t
            self._focus = self._force_t = None
        elif self.order_play:
            seq = self._seq()
            if seq:
                i = next((j for j, (x, t) in enumerate(seq) if self.cur_it is not None and x["id"] == cur_id and abs(t - self.cur_start) <= SCENE_TOL), -1)
                return seq[(i + 1) % len(seq)]
        if it is None:
            return self._pick_fresh(oks, cur_id)
        if ft is not None:
            return it, ft
        d = it["src"].duration
        pool = self._pool(it) or self._pool(it, True)
        far = [t for t in pool if all(not (r[0] == it["id"] and abs(t - r[1]) <= 2.0) for r in self.recent)
               and (d <= 0 or t < d - 0.5)]
        return it, random.choice(far or pool)

    def _pick_fresh(self, oks, cur_id):
        """Next scene when cutting on the beat / kick: never the one on screen, and as far as possible one that has not played
        lately, so every cut looks different. When everything has played recently, one of the older half is picked."""
        cands = []
        for x in oks:
            d = x["src"].duration
            pool = self._pool(x) or self._pool(x, True)
            ts = [t for t in pool if d <= 0 or t < d - 0.5] or pool or [0.0]
            cands += [(x, t) for t in ts]
        rec = self.recent

        def age(c):                                          # how many scenes ago this one played (999 = not lately)
            ix = [i for i, r in enumerate(rec) if r[0] == c[0]["id"] and abs(c[1] - r[1]) <= 2.0]
            return (len(rec) - max(ix)) if ix else 999

        not_cur = [c for c in cands if not (c[0]["id"] == cur_id and abs(c[1] - self.cur_start) <= 2.0)]
        cands = not_cur or cands
        best = [c for c in cands if age(c) == 999]
        if not best:
            cands.sort(key=age, reverse=True)                # all played lately: one of the older half (not a fixed loop)
            best = cands[:max(1, len(cands) // 2)]
        items = {c[0]["id"] for c in best}
        if len(items) > 1 and random.random() < 0.75:        # another clip more often than not
            diff = [c for c in best if c[0]["id"] != cur_id]
            best = diff or best
        return random.choice(best)

    # ---- scene trims (a scene's own start / end, set in the Scenes tab)
    def has_trim(self, it, t):
        return any(abs(t0 - t) <= SCENE_TOL for t0, a, b in ((self._pf(it) or {}).get("trim") or []))

    def trim_of(self, it, t):
        """(start, end) the user set for the scene that begins at t (a user-made cut always has one), or None."""
        for t0, a, b in ((self._pf(it) or {}).get("trim") or []):
            if abs(t0 - t) <= SCENE_TOL:
                return a, b
        return self.custom_of(it, t)

    def natural_range(self, it, t):
        cu = self.custom_of(it, t)
        if cu:
            return cu
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
        return self.scene_range(it, t)[0]

    def set_trim(self, iid, t, a, b, now):
        it = next((x for x in self.items if x["id"] == iid and x["src"] is not None), None)
        if it is None:
            return
        p = self._pfw(it)
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
        """Rows for the Scenes tab, in list order: (key, length, flag, label, start, is a user-made cut) of every scene that isn't deleted."""
        src, out = it["src"], []
        base = sorted(src.scenes)
        for k in self.scene_keys(it):
            f = self.flag(it, k)
            if f == 2:
                continue
            cu = self.custom_of(it, k)
            tr = self.trim_of(it, k)
            if cu:
                out.append((k, max(0.0, tr[1] - tr[0]), f, self.scene_label(it, k), tr[0], True))
                continue
            i = next((j for j, x in enumerate(base) if abs(x - k) < 1e-6), 0)
            end = base[i + 1] if i + 1 < len(base) else (src.duration if src.duration > k else k)
            out.append((k, max(0.0, (tr[1] - tr[0]) if tr else (end - k)), f, f"Scene {i + 1}", k, False))
        return out

    def counts(self, it):
        ks = self.scene_keys(it)
        return (sum(1 for t in ks if self.flag(it, t) == 1), sum(1 for t in ks if self.flag(it, t) == 2))

    def add_cut(self, iid, t, now):
        """Turn the trimmed range of scene t into a scene of its own, right after it in the list (the original goes back to full length)."""
        it = next((x for x in self.items if x["id"] == iid and x["src"] is not None), None)
        if it is None or not self.has_trim(it, t):
            return None
        a, b = self.trim_of(it, t)
        keys = self.scene_keys(it)
        key = round(a, 3)
        while any(abs(key - x) <= SCENE_TOL * 1.5 for x in keys):
            key = round(key + 0.45, 3)
        p = self._pfw(it)
        p["trim"] = [x for x in p["trim"] if abs(x[0] - t) > SCENE_TOL]
        p["custom"].append([key, round(a, 3), round(b, 3)])
        i = next((j for j, x in enumerate(keys) if abs(x - t) <= SCENE_TOL), len(keys) - 1)
        keys.insert(i + 1, key)
        p["order"] = keys
        self._changed(it, now)
        return key

    def move_scene(self, iid, t, d, now):
        """Swap scene t with its neighbour in the list (d = -1 up / +1 down); deleted scenes are skipped over."""
        it = next((x for x in self.items if x["id"] == iid and x["src"] is not None), None)
        if it is None:
            return False
        keys = self.scene_keys(it)
        i = next((j for j, x in enumerate(keys) if abs(x - t) <= SCENE_TOL), None)
        if i is None:
            return False
        j = i + d
        while 0 <= j < len(keys) and self.flag(it, keys[j]) == 2:
            j += d
        if not (0 <= j < len(keys)):
            return False
        keys[i], keys[j] = keys[j], keys[i]
        self._pfw(it)["order"] = keys
        self._changed(it, now)
        return True

    def set_flag(self, iid, t, mode, now):
        """mode: hide (toggle) | delete | show"""
        it = next((x for x in self.items if x["id"] == iid and x["src"] is not None), None)
        if it is None:
            return
        p = self._pfw(it)
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
        if h is None or it is None or it.get("src") is None or it.get("example") or it["src"].kind == "image" or not it["src"].scenes:
            return None
        pos = start
        if hasattr(h, "pos_frames") and it["src"].fps:
            try:
                pos = h.pos_frames() / it["src"].fps
            except Exception:
                pos = start
        sc = sorted(it["src"].scenes)
        if self.custom_of(it, start):
            return it, start
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
        label = self.scene_label(it, t)
        prev = self.flag(it, t)
        self.undo.append((it["id"], t, prev))
        del self.undo[:-30]
        self.set_flag(it["id"], t, mode, now)
        self.request_change()
        name = os.path.basename(it["path"])
        return f"{label} of {name[:28]} {'deleted' if mode == 'delete' else 'hidden'}   (Z to undo)"

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
        p = self._pfw(it)
        sc = self.scene_keys(it)
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
        if not pf.get("off") and not pf.get("del") and not pf.get("trim") and not pf.get("custom") and not pf.get("order"):
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
        self._sync_example(now)
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
                self.recent = (self.recent + [(self.cur_it["id"], self.cur_start)])[-24:]
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
        its = [it for it in self._real() if it["src"].kind != "image"]
        it = next((x for x in its if x["id"] == iid), None) or (its[0] if its else None)
        if it is None:
            return dict(none=True, n_clips=0, sig=("none",))
        rows = self.scene_rows(it)
        top = scroll
        out = []
        for i_, (t, ln, f, no, st_, cu) in enumerate(rows[top:top + per]):
            out.append(dict(t=t, ln=ln, f=f, no=no, st=st_, cut=cu, th=it["src"].scene_thumb(st_), tr=self.has_trim(it, t),
                            up=top + i_ > 0, dn=top + i_ < len(rows) - 1))
        hid, dele = self.counts(it)
        idx = its.index(it)
        s = it["src"]
        return dict(none=False, id=it["id"], name=os.path.basename(it["path"]), rows=out, total=len(rows), hidden=hid, deleted=dele,
                    idx=idx, n_clips=len(its), tid=id(s), few=(len(s.scenes) < 5 and not self._pf(it)),
                    scanning=getattr(s, "scan", 1.0) < 1.0,
                    sig=(it["id"], self.pver, getattr(s, "sthumb_n", 0), scroll, len(s.scenes), len(rows), self.order_play), it=it)

    def info(self):
        oks = self._real()
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
        self.fxa = {}                                                    # the effect dots' light, eased 0 (off) .. 1 (on)
        self.style_x, self.style_xd, self._sx_mode = 0.0, 0.0, None     # the style row: scroll target / eased / the mode we last scrolled to
        self.dock_p, self.dock_goal = 0.0, False       # live preview hand-over to the player bar (0 = in the page, 1 = in the bar)
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
        self.zm_c, self.zm_v, self.zm_t, self.zm_hi, self.zm_ang = [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], False, 0.0
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
        self.loading = LoadingOverlay(ctx, self.quads)
        self.load_bg = None
        self.scan_watch, self.scan_vid, self.scan_a, self.scan_stat = set(), False, 0.0, (0.0, 0, "", 1, 1)
        self.glitch = GlitchText(ctx, quad, self.quads)
        self.card_ex, self.card_head = None, None            # the Video card's own copy of the example clip for its hover demo
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
        self.scache_t = -10.0
        self.panel_tex = None
        self.presets = load_presets()
        self.preset_sel = load_last_preset_name()
        if self.preset_sel not in [n for n, _ in self.presets]:
            self.preset_sel = self.presets[-1][0] if self.presets else ""
        self.preset_open = False
        self.preset_del_t = -10.0
        self.notice, self.notice_t = "", -10.0
        self.proj = None                       # the current named project: dict(path, name, sig)
        self.proj_missing = {"queue": [], "media": []}     # [(original index, path)] left out by "Load anyway" (kept for saving)
        self.in_session = False                # False until a project / files are opened from the home screen
        self.dlg, self.dlg_ver = None, 0
        self.home_items, self.home_ver, self.home_row = [], 0, 0
        self.proj_inbox = []
        self.after_save = None
        self.exp = None                        # a recording in progress (see export_start)
        self.exp_note = None                   # last result / error line shown on the Export tab
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
        self.bfx_prog = ctx.program(vertex_shader=VERT, fragment_shader=BFX_FRAG)
        self.bfx_vao = ctx.vertex_array(self.bfx_prog, [(quad, "2f", "in_pos")])
        self.bfx_env, self.bfx_prev, self.bfx_seed, self.bfx_pick = 0.0, 0.0, 1.0, 0
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
        self.fx_t, self.fx_f = mk(W, H)                # scratch for the bass-distortion pass
        self.tba1, self.tba1f = mk(W // 4, H // 4)
        self.tba2, self.tba2f = mk(W // 4, H // 4)
        self.tbb1, self.tbb1f = mk(W // 4, H // 4)
        self.tbb2, self.tbb2f = mk(W // 4, H // 4)
        self.target_size = (W, H)
        self.mode_t = 1.0                              # (re)created targets: no transition in flight

    def bass_fx_pass(self, now, W, H):
        """Whole-screen bass distortion (blur / vibrate / glitch): scene -> scratch with the shader, then copied back."""
        cfg = self.cfg
        kind = self.bfx_pick if cfg["bass_fx"] == 3 else min(2, max(0, int(cfg["bass_fx"])))
        self.fx_f.use()
        self.ctx.viewport = (0, 0, W, H)
        self.ctx.disable(moderngl.BLEND)
        self.scene_t.use(0)
        pr = self.bfx_prog
        pr["uTex"].value = 0
        pr["uI"].value = float(self.bfx_env * fxv(cfg, "bass_fx_amt"))
        pr["uKind"].value = int(kind)
        pr["uSeed"].value = float(self.bfx_seed)
        pr["uTime"].value = float(now % TIME_WRAP)
        pr["uRes"].value = (float(W), float(H))
        self.bfx_vao.render(moderngl.TRIANGLE_STRIP)
        self.ctx.copy_framebuffer(self.scene_f, self.fx_f)

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
        self.ctx.disable(moderngl.BLEND)
        vx, vy, vw, vh = 0, 0, W, H
        if abs(tex.width * H - tex.height * W) > 0.02 * tex.height * W:      # a recording at another aspect: fit it, black bars
            k = min(W / tex.width, H / tex.height)
            vw, vh = int(tex.width * k), int(tex.height * k)
            vx, vy = (W - vw) // 2, (H - vh) // 2
            self.ctx.viewport = (0, 0, W, H)
            self.ctx.clear(0.0, 0.0, 0.0, 1.0)
        self.ctx.viewport = (vx, vy, vw, vh)
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
        self.notice, self.notice_t = msg, time.perf_counter()          # also shown inside the menu (the on-screen toast hides behind it)

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
            if m == VIDEO_MODE:                          # Video card: a frame of the built-in example clip (a clip's thumbnail takes over when one is loaded)
                ex = ExampleSource(ctx, w, h, 1.5)
                st_ = dict(x=w * 0.5, y=h * 0.5, vx=0.0, vy=0.0, t=0.5, hue=None)
                img = ex.render(st_, ExampleSource.SCENE_S * 3 + 1.0, 0.0)
                ex.fbo.release()
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
        return [it for it in self.media._real() if it["src"].kind != "image"]

    def scn_pick_default(self):
        its = self.scn_clips()
        if any(it["id"] == self.scn_id for it in its):
            return
        cur = self.media.cur_it
        self.scn_id = cur["id"] if (cur and not cur.get("example") and cur["src"].kind != "image") else (its[0]["id"] if its else None)

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

    # ---------------------------------------------------------------- projects / home screen
    def proj_state_sig(self):
        mp = self.media.paths()
        return json.dumps([self.preset_snapshot(), self.playlist, mp, self.proj_missing, [self.media.prefs.get(p) for p in mp], self.spotify_on], sort_keys=True, default=str)

    def proj_dirty(self):
        if not self.in_session:
            return False
        if self.proj is None:
            return True                                           # a project that was never saved counts as unsaved data from the moment it starts
        return self.proj["sig"] is None or self.proj["sig"] != self.proj_state_sig()

    def proj_title(self):
        return self.proj["name"] if self.proj else "Untitled New Project"

    def proj_thumb(self):
        t = self.final_t
        if t is None:
            return ""
        try:
            w, h = t.size
            arr = np.frombuffer(t.read(), np.uint8).reshape(h, w, t.components)[::-1, :, :3]
            ch = min(h, int(w * 9 / 16))                    # centre-crop to 16:9
            cw = min(w, int(h * 16 / 9))
            y0, x0 = (h - ch) // 2, (w - cw) // 2
            return thumb_encode(np.ascontiguousarray(arr[y0:y0 + ch, x0:x0 + cw]))
        except Exception:
            return ""

    def proj_capture(self, name):
        qp, mp = list(self.playlist), self.media.paths()
        cur_path = self.playlist[self.cur] if 0 <= self.cur < len(self.playlist) else None
        for lst, miss in ((qp, self.proj_missing["queue"]), (mp, self.proj_missing["media"])):
            for i, p in sorted(miss):                       # files "Load anyway" left out go back where they were
                if p not in lst:
                    lst.insert(min(i, len(lst)), p)
        scenes = {p: self.media.prefs[p] for p in mp if p in self.media.prefs}
        return dict(app="Hypnosis", version=1, name=name, saved=time.time(), cfg=self.preset_snapshot(),
                    queue=dict(paths=qp, cur=(qp.index(cur_path) if cur_path in qp else -1)),
                    media=dict(paths=mp), spotify=bool(self.spotify_on), scenes=scenes, scan=scan_cache_export(mp), thumb=self.proj_thumb())

    def proj_save(self, path, name=None):
        name = (name or os.path.splitext(os.path.basename(path))[0])[:60]
        try:
            write_project(path, self.proj_capture(name))
        except Exception as ex:
            self.toast_msg(f"Couldn't save the project: {ex}")
            return False
        self.proj = dict(path=path, name=name, sig=self.proj_state_sig())
        touch_recent(path)
        self.toast_msg(f"Saved {name}")
        return True

    def proj_save_as(self, then=None):
        self.after_save = then
        first = os.path.splitext(os.path.basename(self.playlist[0]))[0] if self.playlist else ""
        init = (self.proj["name"] if self.proj else first or "My project") + PROJ_EXT
        threading.Thread(target=dlg_save_project, args=(self.proj_inbox, ("save",), init), daemon=True).start()

    def proj_save_cmd(self, then=None):
        if self.proj:
            if self.proj_save(self.proj["path"], self.proj["name"]) and then:
                then()
        else:
            self.proj_save_as(then)

    def clear_session(self):
        if self.spotify_on:
            self.set_spotify(False)
        self.audio.stop()
        self.playlist.clear()
        self.cur, self.scroll = -1, 0
        self.media.clear()
        self.proj_missing = {"queue": [], "media": []}
        self.mscroll, self.scn_id, self.scn_scroll, self.scn_sel = 0, None, 0, None
        self.sync_media_cfg()

    def request_quit(self, delay=0.0):
        """Close the app - but first offer to save when the open project / session has unsaved changes."""
        if self.dlg and self.dlg.get("kind") == "quit":
            return                                                # already asking
        def go():
            self.exit_at = time.perf_counter() + delay
        if not self.proj_dirty():
            go()
            return
        self.dlg = None
        if not self.menu_open:
            self.open_menu(True)                                  # the question is drawn on the menu panel
        self.guard(go, "Save before closing?", "quit")

    def guard(self, cont, title="Save changes?", kind="plain"):
        """Run cont() - after asking about unsaved changes when replacing a session that has some."""
        if not self.proj_dirty():
            cont()
            return
        named = self.proj is not None
        msg = [f"“{self.proj['name']}” has changes that aren’t saved." if named else "This session isn’t saved as a project yet."]
        self.dlg_open(title, msg, [], [
            ("Cancel", None, False), ("Don’t save", cont, False),
            ("Save" if named else "Save as…", lambda: self.proj_save_cmd(cont), True)], kind=kind)

    def leave_home(self, keep_menu=False, tab="visuals"):
        self.in_session = True
        self.tab = tab
        if not keep_menu:
            self.open_menu(False)

    def go_home(self):
        self.dlg = None
        if self.tab != "home":
            self.tab_before_home = self.tab
        self.refresh_home()
        self.open_menu(True)
        self.tab = "home"

    def proj_resume(self):
        """Back from the home screen to the menu (the tab you left), not out of the menus altogether."""
        self.dlg = None
        self.tab = getattr(self, "tab_before_home", None) or "visuals"

    def refresh_home(self):
        self.home_items = list_projects()
        self.home_row = 0
        self.home_ver += 1

    def home_list(self):
        """The saved projects, with the session that was started but never saved shown first (only until it is saved or closed)."""
        if self.in_session and self.proj is None:
            tmp = dict(name="Untitled New Project", path=None, temp=True, thumb=None, saved=0, used=time.time(), missing=0,
                       tracks=len(self.playlist), clips=len(self.media.items))
            return [tmp] + self.home_items
        return self.home_items

    def home_rows(self):
        return max(0, (len(self.home_list()) + HOME_COLS - 1) // HOME_COLS)

    def proj_new(self):
        def go():
            self.clear_session()
            for k, v in DEFAULTS.items():
                if k not in PRESET_SKIP:
                    self.cfg[k] = v
            self.proj = None
            self.leave_home(keep_menu=True)
            self.toast_msg("New project")
        self.guard(go)

    def proj_open_flow(self, path):
        self.guard(lambda: self.proj_load_checked(path))

    def proj_load_checked(self, path, remap=None, data=None, warned=False):
        data = data or read_project(path)
        if data is None:
            self.toast_msg("That project file is missing or isn’t a Hypnosis project")
            forget_recent(path)
            self.refresh_home()
            return
        remap = remap or {}
        qp, mp = project_paths(data)
        miss = [(p, remap.get(p, p)) for p in qp + mp if not os.path.isfile(remap.get(p, p))]
        if not miss or warned:
            self.proj_apply(data, path, remap)
            return
        self.missing_dialog(data, path, remap, miss)

    def missing_dialog(self, data, path, remap, miss, busy=False):
        seen, items = set(), []
        for orig, cur in miss:
            if orig in seen:
                continue
            seen.add(orig)
            items.append((os.path.basename(cur) or cur, os.path.dirname(cur), orig))
        n = len(items)
        name = str(data.get("name") or os.path.splitext(os.path.basename(path))[0])
        lines = [f"{n} file{'s' if n != 1 else ''} used by “{name}” couldn’t be found.",
                 "Searching…" if busy else "Find them, or load the project without them."]
        prev = self.dlg["scroll"] if self.dlg and self.dlg.get("kind") == "missing" else 0
        self.dlg_open("Missing files", lines, items, [
            ("Back to home", self.go_home if self.tab != "home" else None, False),
            ("Find files…", lambda: self.missing_find(data, path, remap, items), False),
            ("Load anyway", lambda: self.proj_load_checked(path, remap, data, True), True)], kind="missing")
        self.dlg.update(data=data, path=path, remap=remap, scroll=min(prev, max(0, n - DLG_ROWS)), busy=busy)

    def missing_find(self, data, path, remap, items):
        names = [it[0] for it in items]
        self.missing_dialog(data, path, remap, [(it[2], remap.get(it[2], it[2])) for it in items], busy=True)
        self.dlg["buttons"] = []
        threading.Thread(target=dlg_find, args=(self.proj_inbox, names), daemon=True).start()
        return True

    def proj_apply(self, data, path, remap=None):
        remap = remap or {}
        qp, mp = project_paths(data)
        qp, mp = [remap.get(p, p) for p in qp], [remap.get(p, p) for p in mp]
        self.clear_session()
        for k, v in DEFAULTS.items():
            if k not in PRESET_SKIP:
                self.cfg[k] = v
        self.cfg.update(clean_cfg_values(data.get("cfg")))
        if self.cfg.get("media_blend", 0) not in (0, 1, 2):
            self.cfg["media_blend"] = 0
        for k in ("kick_lo", "kick_hi", "snare_lo", "snare_hi"):
            self.fix_pairs(k)
        qok = [p for p in qp if os.path.isfile(p)]
        mok = [p for p in mp if os.path.isfile(p)]
        self.proj_missing = {"queue": [(i, p) for i, p in enumerate(qp) if not os.path.isfile(p)],
                             "media": [(i, p) for i, p in enumerate(mp) if not os.path.isfile(p)]}
        q = data.get("queue") if isinstance(data.get("queue"), dict) else {}
        ci = q.get("cur", -1) if isinstance(q.get("cur", -1), int) else -1
        cur_path = qp[ci] if 0 <= ci < len(qp) else None
        self.playlist.extend(qok)
        scan_cache_import(data.get("scan"), remap)             # the project's remembered scene lists: no re-scanning
        if mok:
            self.media.add(mok)
        old = {o: n for o, n in remap.items() if o != n}
        pf = clean_scene_prefs(data.get("scenes"))
        for p_ in mp:                                          # exactly the project's scene edits for its clips (none = untouched)
            self.media.prefs.pop(p_, None)
        for p_, v_ in pf.items():
            self.media.prefs[remap.get(p_, p_)] = v_
        save_scene_prefs(self.media.prefs)
        self.sync_media_cfg()
        want_spot = data.get("spotify") is True
        if self.playlist and not want_spot:
            self.play_index(qok.index(cur_path) if cur_path in qok else 0)
        if want_spot:                                          # the project was saved in Spotify Mode: switch it back on
            self.set_spotify(True)
        name = str(data.get("name") or os.path.splitext(os.path.basename(path))[0])[:60]
        self.proj = dict(path=path, name=name, sig=None if old else self.proj_state_sig())
        touch_recent(path)
        self.dlg = None
        self.leave_home(keep_menu=True)
        nm = len(self.proj_missing["queue"]) + len(self.proj_missing["media"])
        self.toast_msg(f"Opened {name}" + ("  \u00b7  Spotify Mode on" if self.spotify_on else "") + (f"  ·  {nm} missing file{'s' if nm != 1 else ''} skipped" if nm else ""))

    # ---- the little modal dialog drawn over the panel
    def dlg_open(self, title, lines, items, buttons, kind="plain"):
        self.dlg = dict(title=title, lines=lines, items=items, buttons=buttons, scroll=0, kind=kind, busy=False)
        self.dlg_ver += 1

    def dlg_press(self, i):
        d = self.dlg
        if not d or not (0 <= i < len(d["buttons"])):
            return
        cb = d["buttons"][i][1]
        self.dlg = None
        self.dlg_ver += 1
        if cb:
            cb()

    def proj_discard(self):
        """Throw the unsaved session away: everything back to defaults over the Psychedelic home screen."""
        self.clear_session()
        for k, v in DEFAULTS.items():
            if k not in PRESET_SKIP:
                self.cfg[k] = v
        self.proj = None
        self.in_session = False
        self.refresh_home()
        self.toast_msg("Project discarded")

    def proj_delete_ask(self, it):
        path, name = it["path"], it["name"]
        self.dlg_open("Delete this project?", [f"\u201c{name}\u201d will be permanently deleted.",
                                               "Only this project file is removed (your music and videos are not touched). This can't be undone."],
                      [], [("Cancel", None, True), ("Delete forever", lambda: self.proj_delete(path, name), False, True)], kind="delete")

    def proj_delete(self, path, name):
        """Remove exactly one project file (never anything that isn't a .hypno) and drop it from the recent list."""
        try:
            if not path.lower().endswith(PROJ_EXT):
                raise OSError("not a project file")
            if os.path.isfile(path):
                os.remove(path)
        except OSError as e:
            self.toast_msg("Couldn't delete: " + (e.strerror or str(e)))
            return
        forget_recent(path)
        if self.proj and os.path.normcase(os.path.abspath(self.proj.get("path") or "")) == os.path.normcase(os.path.abspath(path)):
            self.proj_discard()                                # the open project is gone: back to a fresh start over the Psychedelic home screen
        row = self.home_row
        self.refresh_home()
        self.home_row = max(0, min(row, self.home_rows() - HOME_ROWS_VIS))          # stay where you were scrolled to
        self.toast_msg(f"Deleted {name}")

    def dlg_cancel(self):
        if self.dlg and not self.dlg.get("busy"):
            self.dlg = None
            self.dlg_ver += 1

    def notice_state(self):
        age = time.perf_counter() - self.notice_t
        if not self.notice or age > 2.6 or not self.menu_open:
            return None
        a = min(1.0, age / 0.18, (2.6 - age) / 0.5)
        return (self.notice[:70], round(max(0.0, a), 1), round(min(1.0, age / 0.25), 1))

    def dlg_state(self):
        d = self.dlg
        if not d:
            return None
        return dict(title=d["title"], lines=d["lines"], items=d["items"], scroll=d["scroll"],
                    buttons=[(b[0], b[2], len(b) > 3 and bool(b[3])) for b in d["buttons"]], busy=d["busy"])

    def proj_poll(self):
        while self.proj_inbox:
            tag, val = self.proj_inbox.pop(0)
            kind = tag[0]
            if kind == "save":
                cb, self.after_save = self.after_save, None
                if val:
                    if not val.lower().endswith(PROJ_EXT):
                        val += PROJ_EXT
                    if self.proj_save(val) and cb:
                        cb()
                    if self.tab == "home":
                        self.refresh_home()
            elif kind == "open":
                if val:
                    self.proj_open_flow(val)
            elif kind == "expdir":
                if val:
                    self.cfg["exp_dir"] = os.path.normpath(val)
            elif kind in ("loc", "find") and self.dlg and self.dlg.get("kind") == "missing":
                d = self.dlg
                remap = dict(d["remap"])
                items = d["items"]
                n0 = len(items)
                if kind == "loc":
                    if val:
                        remap[tag[1]] = val
                else:
                    for nm, _, orig in items:
                        hit = (val or {}).get(nm.lower())
                        if hit:
                            remap[orig] = hit
                    left = sum(1 for _, _, o in items if not os.path.isfile(remap.get(o, o)))
                    self.toast_msg(f"Found {n0 - left} of {n0}" if val else "No matching files found there")
                self.proj_load_checked(d["path"], remap, d["data"])

    # ---------------------------------------------------------------- export
    def export_blocker(self):
        if self.spotify_on:
            return "Spotify Mode can't be exported – turn it off and load a song"
        if self.audio.loading:
            return "Loading the song…"
        if self.audio.data is None:
            return "Load a song from the Queue tab to export"
        return None

    def export_start(self, W, H):
        if self.exp:
            return
        why = self.export_blocker()
        if why:
            self.toast_msg(why)
            return
        cfg, a = self.cfg, self.audio
        folder = exp_folder(cfg)
        try:
            os.makedirs(folder, exist_ok=True)
        except OSError as e:
            self.exp_note = ("err", f"Can't use that folder: {e.strerror or e}")
            return
        title = os.path.splitext(os.path.basename(self.playlist[self.cur]))[0] if 0 <= self.cur < len(self.playlist) else "Hypnosis"
        ext = exp_opt(cfg, "exp_fmt", EXP_FMT)[1]
        base = os.path.join(folder, exp_name(title))
        path, n = f"{base}.{ext}", 2
        while os.path.exists(path):
            path, n = f"{base} ({n}).{ext}", n + 1
        size = exp_size(cfg, W, H)
        fps = EXP_FPS[min(max(int(cfg["exp_fps"]), 0), len(EXP_FPS) - 1)]
        crf = exp_opt(cfg, "exp_q", EXP_Q)[1]
        if int(cfg["exp_from"]) == 0 or a.finished or a.pos >= len(a.data) - SR // 2:
            a.pos = 0
            a.finished = False
        a.playing = True
        p0 = int(a.pos)
        max_dur = float(exp_opt(cfg, "exp_len", EXP_LEN)[1])
        remain = (len(a.data) - p0) / float(SR)
        try:
            ex = Exporter(path, size, fps, crf, ext, a.data, p0, bool(cfg["exp_gpu"]), bool(cfg["exp_audio"]))
        except Exception as e:
            self.exp_note = ("err", f"Couldn't start the encoder: {e}")
            return
        self.exp = dict(ex=ex, size=size, fps=fps, p0=p0, data=a.data, next=0, last_ts=None, cur_ts=0.0, ring=None, slot=0, pend=[],
                        max_dur=max_dur, total=min(max_dur, remain) if max_dur else remain, path=path, capturing=True,
                        bytes=0, bytes_t=0.0, folder=folder, t0=time.perf_counter())
        self.exp_note = None
        self.tab_k = 1.0

    def export_stop(self, why=None, abort=False):
        """Stop capturing; the worker then flushes and closes the file (export_tick notices when it is done)."""
        x = self.exp
        if not x or not x["capturing"]:
            return
        x["capturing"] = False
        a = self.audio
        self._exp_drain()
        ex = x["ex"]
        if abort:
            ex.abort()
            return
        end = max(0.0, x["last_ts"] or 0.0)
        if why == "end":
            end = (len(x["data"]) - x["p0"]) / float(SR)
        elif why == "len":
            end = x["max_dur"]
        if x["max_dur"]:
            end = min(end, x["max_dur"])
        ex.finish(end)

    def _exp_drain(self):
        x = self.exp
        for idx, buf in x["pend"]:
            try:
                x["ex"].push(idx, buf.read())
            except Exception:
                pass
        x["pend"] = []

    def export_free(self):
        x = self.exp
        if x and x["ring"]:
            for b in x["ring"]:
                try:
                    b.release()
                except Exception:
                    pass
            x["ring"] = None

    def export_tick(self, now):
        """Once per frame, after the clean scene has been drawn: grab the frame the audio clock says is due."""
        x = self.exp
        if not x:
            return
        ex, a = x["ex"], self.audio
        if not x["capturing"]:
            if ex.state in ("done", "error", "aborted"):
                self.export_free()
                self.exp = None
                if ex.state == "done":
                    sz = os.path.getsize(ex.path) if os.path.isfile(ex.path) else 0
                    self.exp_note = ("ok", ex.path, sz)
                    self.toast_msg("Export saved: " + os.path.basename(ex.path))
                elif ex.state == "error":
                    self.exp_note = ("err", "Export failed: " + (ex.error or "unknown error"))
                    self.toast_msg("Export failed")
                else:
                    self.exp_note = ("err", "Export cancelled")
            return
        if ex.state == "error":                               # the encoder died: stop feeding it
            x["capturing"] = False
            return
        if a.data is not x["data"]:                           # the track was changed / unloaded / skipped
            self.export_stop("track")
            return
        ts = (a.pos - a.latency * SR - x["p0"]) / float(SR)
        if x["last_ts"] is not None and abs(ts - x["last_ts"]) > 1.0:
            self.export_stop("seek")                          # a seek would break the audio / video sync: end the file here
            self.toast_msg("Export ended - the song was skipped")
            return
        if ts < 0:
            return
        x["last_ts"] = x["cur_ts"] = ts
        if now - x["bytes_t"] > 0.5:
            x["bytes_t"] = now
            try:
                x["bytes"] = os.path.getsize(x["path"])
            except OSError:
                pass
        if a.finished or a.pos >= len(x["data"]):
            self.export_stop("end")
            return
        if x["max_dur"] and ts >= x["max_dur"]:
            self.export_stop("len")
            return
        n = int(ts * x["fps"])
        if n < x["next"]:
            return
        w, h = x["size"]
        if self.target_size != (w, h):
            return
        if x["ring"] is None:
            x["ring"] = [self.ctx.buffer(reserve=w * h * 3) for _ in range(3)]
        fbo = self.trans_f if self.final_t is self.trans_t else self.scene_f
        buf = x["ring"][x["slot"]]
        fbo.read_into(buf, components=3, alignment=1)          # async copy into a pixel buffer; read back one frame later
        x["slot"] = (x["slot"] + 1) % 3
        x["pend"].append((n, buf))
        x["next"] = n + 1
        while len(x["pend"]) > 1:
            idx, b = x["pend"].pop(0)
            ex.push(idx, b.read())

    def export_shutdown(self):
        x = self.exp
        if not x:
            return
        if x["capturing"]:
            self.export_stop("track")
        x["ex"].th.join(timeout=20.0)
        self.export_free()
        self.exp = None

    def export_state(self):
        cfg, x = self.cfg, self.exp
        d = dict(blocker=self.export_blocker(), folder=exp_folder(cfg), note=self.exp_note, active=False, final=False, sig=None)
        if x:
            ex = x["ex"]
            d.update(active=x["capturing"], final=not x["capturing"], ts=x["cur_ts"], total=x["total"], frames=ex.frames, dropped=ex.dropped,
                     size=x["bytes"], enc=ex.encoder, paused=not self.audio.playing, name=os.path.basename(x["path"]), res=x["size"], fps=x["fps"])
            d["sig"] = (int(x["cur_ts"] * 4), x["capturing"], d["paused"], x["bytes"] // 65536, ex.encoder, ex.state)
        return d

    def export_open_folder(self):
        folder = exp_folder(self.cfg)
        try:
            os.makedirs(folder, exist_ok=True)
            if hasattr(os, "startfile"):
                os.startfile(folder)
            else:
                subprocess.Popen(["xdg-open", folder])
        except Exception:
            pass

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
            if self.tab == "home" and self.in_session:
                self.tab = getattr(self, "tab_before_home", None) or "visuals"          # reopening after peeking at the home screen: back to the app
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
        if self.dlg:                                           # a dialog is up: it takes every click
            if key and key[0] == "vsb" and key[1] == "dlg":
                self.drag = ("vsb", "dlg")
                self.vsb_press("dlg", ly)
            elif key and key[0] == "dlg":
                self.press_start(key)
                self.dlg_press(key[1])
            elif key and key[0] == "dlg_loc" and 0 <= key[1] < len(self.dlg["items"]):
                self.press_start(key)
                nm, _, orig = self.dlg["items"][key[1]]
                threading.Thread(target=dlg_locate, args=(self.proj_inbox, ("loc", orig), nm), daemon=True).start()
            return
        if key is None:
            if not (0 <= lx <= PW and 0 <= ly <= PH) and not (self.tab == "home" and not self.in_session):
                self.open_menu(False)
            return
        kind = key[0]
        if self.preset_open and kind not in ("preset_pick", "noop"):
            self.preset_open = False
            if kind == "preset_dd":
                return
        if kind not in ("slider", "seek", "noop", "trim", "vol", "stylerow", "stysb", "vsb"):
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
        elif kind == "home":
            self.go_home()
        elif kind == "proj_new":
            self.proj_new()
        elif kind == "proj_open":
            threading.Thread(target=dlg_open_project, args=(self.proj_inbox, ("open",)), daemon=True).start()
        elif kind == "proj_resume":
            self.proj_resume()
        elif kind == "proj_card":
            hl = self.home_list()
            if 0 <= key[1] < len(hl):
                if hl[key[1]].get("temp"):
                    self.proj_resume()                                     # the unsaved session: just resume it
                else:
                    self.proj_open_flow(hl[key[1]]["path"])
        elif kind == "proj_tsave":
            self.proj_save_cmd()
        elif kind == "proj_ttrash":
            self.dlg_open("Discard this project?", ["\u201cUntitled New Project\u201d hasn\u2019t been saved.",
                                                    "Discarding clears this session\u2019s music, videos and settings. This can\u2019t be undone."],
                          [], [("Cancel", None, True), ("Discard", self.proj_discard, False, True)], kind="delete")
        elif kind == "proj_trash":
            hl = self.home_list()
            if 0 <= key[1] < len(hl) and not hl[key[1]].get("temp"):
                self.proj_delete_ask(hl[key[1]])
        elif kind == "proj_forget":
            hl = self.home_list()
            if 0 <= key[1] < len(hl) and not hl[key[1]].get("temp"):
                forget_recent(hl[key[1]]["path"])
                self.refresh_home()
        elif kind == "proj_save":
            self.proj_save_cmd()
        elif kind == "proj_saveas":
            self.proj_save_as()
        elif kind == "proj_load":
            threading.Thread(target=dlg_open_project, args=(self.proj_inbox, ("open",)), daemon=True).start()
        elif kind == "toggle":
            name = key[1]
            self.cfg[name] = not self.cfg[name]
        elif kind == "zmode_sel":
            self.cfg["zoom_mode"] = key[1]
        elif kind == "bfx_sel":
            self.cfg["bass_fx"] = key[1]
        elif kind == "exp_opt":
            if not self.exp:
                self.cfg[key[1]] = key[2]
        elif kind == "exp_toggle":
            if not self.exp:
                self.cfg[key[1]] = not self.cfg[key[1]]
        elif kind == "exp_start":
            self.export_start(*pygame.display.get_window_size())
        elif kind == "exp_stop":
            self.export_stop("stop")
        elif kind == "exp_dir":
            if not self.exp:
                threading.Thread(target=dlg_pick_folder, args=(self.proj_inbox, ("expdir",), exp_folder(self.cfg)), daemon=True).start()
        elif kind == "exp_dir_default":
            if not self.exp:
                self.cfg["exp_dir"] = ""
        elif kind == "exp_open":
            self.export_open_folder()
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
            self.proj_missing["queue"] = []
        elif kind == "shuffle":
            self.shuffle_upcoming()
        elif kind == "display":
            self.set_display(key[1])
        elif kind == "style":
            self.cfg["mode"] = key[1]
            self._sx_mode = key[1]
        elif kind == "stysb":
            self.drag = key
            self.style_sb_press(lx)
        elif kind == "vsb":
            self.drag = ("vsb", key[1])
            self.vsb_press(key[1], ly)
        elif kind == "slider":
            self.drag = key
            self.slider_value(key[1], lx)
        elif kind == "fxdot":
            off = [k for k in (self.cfg.get("fx_off") or "").split(",") if k]
            if key[1] in off:
                off.remove(key[1])
            else:
                off.append(key[1])
            self.cfg["fx_off"] = ",".join(off)
        elif kind == "sl_reset":
            self.set_slider(key[1], DEFAULTS[key[1]])
        elif kind == "media_add":
            threading.Thread(target=pick_media_dialog, args=(self.media_picked,), daemon=True).start()
        elif kind == "media_clear":
            self.media.clear()
            self.proj_missing["media"] = []
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
        elif kind == "scn_order":
            self.cfg["scene_order"] = bool(key[1])
            self.toast_msg("Scenes play in list order" if key[1] else "Scenes play at random")
        elif kind == "scn_move":
            if self.media.move_scene(key[1], key[2], key[3], time.perf_counter()):
                self.scn_sel = (key[1], next((r[0] for r in self.media.scene_rows(next(x for x in self.scn_clips() if x["id"] == key[1])) if abs(r[0] - key[2]) < 0.01), key[2]))
        elif kind == "scn_newcut":
            if self.scn_sel:
                nk = self.media.add_cut(self.scn_sel[0], self.scn_sel[1], time.perf_counter())
                if nk is not None:
                    self.scn_sel = (self.scn_sel[0], nk)
                    it_ = next((x for x in self.scn_clips() if x["id"] == self.scn_sel[0]), None)
                    rows_ = self.media.scene_rows(it_) if it_ else []
                    i_ = next((i for i, r in enumerate(rows_) if abs(r[0] - nk) < 0.01), 0)
                    self.scn_sel_i = i_
                    self.scn_scroll = max(0, min(i_ - SCN_ROWS + 2, max(0, len(rows_) - SCN_ROWS))) if i_ >= SCN_ROWS - 1 else self.scn_scroll
                    self.toast_msg("Added the trimmed range as a new scene")
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
        elif kind == "scan_clear":
            now = time.perf_counter()
            if now - self.scache_t < 3.0:
                self.scache_t = -10.0
                n = scan_cache_clear(self.media.paths())              # the clips that are loaded keep their scenes
                self.toast_msg(f"Cleared {n} remembered scene list{'s' if n != 1 else ''}")
            else:
                self.scache_t = now
        elif kind == "mstyle":
            self.cfg["media_style"] = key[1]
        elif kind == "mblend":
            self.cfg["media_blend"] = key[1]
        elif kind == "exit":
            self.request_quit(0.18)                           # (after the press animation) - asks about unsaved changes first

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
            elif self.drag[0] == "stysb":
                self.style_sb_to(lx)
            elif self.drag[0] == "vsb":
                self.vsb_set(self.drag[1], ly)
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
            tr = dict(lo=lo, hi=hi, a=a, b=b, s=s_, e=e_, no=next((r[3] for r in rows if abs(r[0] - sel) < 0.01), "Scene"),
                      has=self.media.has_trim(it, sel), grab=(td["which"] if td else None), pos=self.pv["pos"] if self.pv["head"] is not None else None)
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
            if pv["tex"] is not None and not self.scn_clips():       # the last clip left the stack: the preview picture goes too
                pv["tex"].release()
                pv["tex"], pv["ser"] = None, None
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
            vscroll=round(self.vscroll_d, 1), sx=round(self.style_xd, 1), fxa={k: round(v, 2) for k, v in self.fxa.items()}, drag_sty=bool(self.drag and self.drag[0] == "stysb"), aspect=round(self.vis_layout()["ph"] / PREV_W, 3),
            sec_a={k: round(v, 3) for k, v in self.sec_a.items()},
            scn=(self.scn_state() if self.tab == "scenes" else None), scn_scroll=self.scn_scroll,
            style_prev=self.style_prev,
            reset_armed=(time.perf_counter() - self.reset_t < 3.0), scache_armed=(time.perf_counter() - self.scache_t < 3.0),
            home=dict(items=self.home_list(), row=self.home_row, ver=self.home_ver) if self.tab == "home" else None,
            in_session=self.in_session, dlg=self.dlg_state(),
            dock=round(self.dock_p, 2), prev_ph=self.vis_layout()["ph"],
            notice=self.notice_state(),
            exp=self.export_state() if self.tab == "export" else None, win=pygame.display.get_window_size(),
            proj=(dict(name=self.proj_title(), named=self.proj is not None, dirty=self.proj_dirty(),
                       missing=len(self.proj_missing["queue"]) + len(self.proj_missing["media"])) if self.tab in ("settings", "home") else None),
            presets=[n for n, _ in self.presets], preset_sel=self.preset_sel, preset_open=self.preset_open,
            preset_del=(time.perf_counter() - self.preset_del_t < 3.0),
            beat=dict(bars=getattr(self, "vis_bars", (0,) * 48), kf=round(getattr(self, "kf", 0.0), 1), sf=round(getattr(self, "sf", 0.0), 1)),
        )

    def vis_layout(self):
        W, H = pygame.display.get_window_size()
        return visuals_layout(H / max(1, W), {k: round(v, 3) for k, v in self.sec_a.items()}, self.cfg["domination"], self.cfg["mode"])

    def dock_wanted(self):
        """Hand the preview to the player bar once most of it has scrolled up out of sight (with hysteresis, so it doesn't flicker)."""
        if not self.menu_open or self.tab != "visuals":
            return False
        lay = self.vis_layout()
        L = lay["preview"]
        if L["a"] < 0.5:
            return False
        ly = preview_geom(lay, self.vscroll_d)[1]
        gone = max(0.0, HDR_Y - ly) / max(1.0, float(lay["ph"]))
        return gone > (0.38 if self.dock_goal else 0.58)

    def draw_dock(self, W, H, e, blur_extra=0.0):
        """The mini live preview in the player bar's left corner (over the art / song title), blurring in as the page one blurs out."""
        if self.dlg:                                          # a question is on screen: no live picture may float over it
            return
        if self.dock_p <= 0.004 or self.final_t is None:
            return
        _, _, a_in, b_in = dock_phases(self.dock_p)
        if a_in <= 0.003:
            return
        lay = self.vis_layout()
        hh = float(DOCK_H)
        ww = max(64.0, min(150.0, hh * PREV_W / max(1.0, float(lay["ph"]))))
        sc = 0.88 + 0.12 * a_in
        lx, ly = 24 + ww * (1 - sc) / 2, FTR_Y + 8 + hh * (1 - sc) / 2
        rmax = 11.0 * self.panel_scale
        self._pv_draw(self.final_t, 0, lx, ly, ww * sc, hh * sc, FTR_Y + 1, PH, e, a_in, rmax * b_in + blur_extra, 10.0)

    def _pv_draw(self, tex, flip, lx, ly, lw, lh, clip_top, clip_bot, e, fade, blur, rad=12.0):
        x0, y0, w, h = self.panel_rect
        k = w / PW
        W, H = pygame.display.get_window_size()
        top, bot = y0 + clip_top * k, y0 + clip_bot * k
        rx, ry, rw, rh = x0 + lx * k, y0 + ly * k, lw * k, lh * k
        if bot <= top or ry + rh <= top or ry >= bot:
            return
        ctx = self.ctx
        ctx.scissor = (int(x0), int(H - bot), int(w), int(max(1, bot - top)))
        (cx, cy), (hx, hy) = ndc_rect(rx, ry, rw, rh, W, H)
        pr = self.prev_prog
        pr["uCenter"].value, pr["uHalf"].value = (cx, cy), (hx, hy)
        pr["uSize"].value = (float(rw), float(rh))
        pr["uRad"].value = rad * k
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

    def draw_preview(self, W, H, e, tab=None, fade=1.0, blur=0.0, tex_override=None):
        """Live picture drawn over the panel (clipped to the scrolling area): Visuals = the mirrored scene, Scenes = the looping clip."""
        if self.dlg:                                          # a question is on screen: no live picture may float over it
            return
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
        elif tab == "export":
            tex, flip = self.final_t, 0
            if tex is None:
                return
            clip_top, clip_bot = HDR_Y, FTR_Y
            bx, by, bw, bh = EXP_PV
            sc_ = min(bw / tex.size[0], bh / tex.size[1])
            lw, lh = tex.size[0] * sc_, tex.size[1] * sc_
            lx, ly = bx + (bw - lw) / 2, by + (bh - lh) / 2
        else:
            tex, flip = self.pv["tex"], 1
            clip_top, clip_bot = HDR_Y, FTR_Y
            if tex is None or not self.scn_clips():
                return
            bx, by, bw, bh = SCN_PV
            sc_ = min(bw / tex.size[0], bh / tex.size[1])
            lw, lh = tex.size[0] * sc_, tex.size[1] * sc_
            lx, ly = bx + (bw - lw) / 2, by + (bh - lh) / 2
        if tab == "visuals" and self.dock_p > 0.0:                             # handing over to the player bar: blur + fade away
            a_out, b_out, _, _ = dock_phases(self.dock_p)
            fade, blur = fade * a_out, blur + 11.0 * self.panel_scale * b_out
            if fade <= 0.003:
                return
        self._pv_draw(tex, flip, lx, ly, lw, lh, max(HDR_Y, clip_top), min(FTR_Y, clip_bot), e, fade, blur, 12.0)

    def sb_track(self, val):
        """Remember how fast a dragged scroll bar is moving (for the glide after letting go)."""
        sb, now = self._sb, time.perf_counter()
        dt = now - sb["t"]
        if dt > 1e-3:
            sb["v"] = sb["v"] * 0.4 + (val - sb["last"]) / dt * 0.6
            sb["last"], sb["t"] = val, now

    def style_sb_press(self, lx):
        """Press on the style bar: on the thumb = grab it where you hit it; on the track = glide the row over to that spot."""
        th, smax = sty_thumb(), sty_max()
        tx = STY_X0 + (STY_VW - th) * (self.style_xd / smax if smax > 0 else 0.0)
        on = tx <= lx <= tx + th
        self._sb = dict(off=(lx - (tx + th / 2)) if on else 0.0, direct=on, v=0.0, last=self.style_x, t=time.perf_counter())
        self.style_sb_to(lx)

    def style_sb_to(self, lx):
        """Drag the scroll bar: the thumb's middle follows the mouse (a track click only moves the target, so the row eases over)."""
        sb = self._sb
        th = sty_thumb()
        f = (lx - sb["off"] - STY_X0 - th / 2) / max(1.0, STY_VW - th)
        self.style_x = max(0.0, min(sty_max(), f * sty_max()))
        if sb["direct"]:
            self.style_xd = self.style_x
            self.sb_track(self.style_x)

    def end_drag(self):
        """Mouse let go: a scroll bar thrown quickly keeps gliding and eases to a stop."""
        d, sb = self.drag, getattr(self, "_sb", None)
        self.drag = None
        if not d or not sb or not sb["direct"] or time.perf_counter() - sb["t"] > 0.09:
            return
        if d[0] == "stysb":
            self.style_x = max(0.0, min(sty_max(), self.style_x + sb["v"] * 0.14))
        elif d[0] == "vsb" and d[1] == "vis":
            self.vscroll = max(0.0, min(float(self.vis_layout()["max_scroll"]), self.vscroll + sb["v"] * 0.14))

    def style_reveal(self, mode, instant=False):
        """Scroll so the card of `mode` is fully in view."""
        j = MODE_ORDER.index(mode) if mode in MODE_ORDER else 0
        lo, hi = j * STY_PITCH, j * STY_PITCH + STY_CW
        if lo - 12 < self.style_x:
            self.style_x = max(0.0, lo - 12)
        elif hi + 12 > self.style_x + STY_VW:
            self.style_x = min(sty_max(), hi + 12 - STY_VW)
        if instant:
            self.style_xd = self.style_x

    def vsb_press(self, bid, ly):
        g = self.panel.__dict__.get("sb_geom", {}).get(bid)
        on = False
        off = 0.0
        if g and bid == "vis":
            ty, trh, bh, vmax = g
            top = ty + (trh - bh) * (self.vscroll_d / vmax if vmax > 0 else 0.0)
            on = top <= ly <= top + bh
            off = (ly - (top + bh / 2)) if on else 0.0
        self._sb = dict(off=off, direct=on or bid != "vis", v=0.0, last=self.vscroll, t=time.perf_counter())
        self.vsb_set(bid, ly)

    def vsb_set(self, bid, ly):
        """Dragging a vertical scroll bar: the thumb's middle follows the mouse (on the page bar a track click glides there)."""
        g = self.panel.__dict__.get("sb_geom", {}).get(bid)
        if not g:
            return
        ty, trh, bh, vmax = g
        off = self._sb["off"] if bid == "vis" else 0.0
        v = max(0.0, min(1.0, (ly - off - ty - bh / 2) / max(1.0, trh - bh))) * vmax
        if bid == "vis":
            self.vscroll = float(v)
            if self._sb["direct"]:
                self.vscroll_d = self.vscroll
                self.sb_track(self.vscroll)
        elif bid == "queue":
            self.scroll = int(round(v))
        elif bid == "mstack":
            self.mscroll = int(round(v))
        elif bid == "scn":
            self.scn_scroll = int(round(v))
            self.scn_clamp()
        elif bid == "home":
            self.home_row = int(round(v))
        elif bid == "dlg" and self.dlg:
            self.dlg["scroll"] = int(round(v))
            self.dlg_ver += 1

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
               (st["media"]["sig"], st["media"]["loading"], st["media"]["error"], self.mscroll, st["reset_armed"], st["scache_armed"], st["style_prev"] is not None),
               (st["beat"]["bars"], st["beat"]["kf"], st["beat"]["sf"]) if self.tab == "beat" else None,
               st["scn"]["sig"] if st["scn"] else None, (self.dlg_ver, self.dlg["scroll"] if self.dlg else 0), st["notice"], st["dock"], (self.home_ver, self.home_row, self.in_session, self.proj is None, len(self.media.items)) if self.tab == "home" else None,
               tuple(st["proj"].values()) if st["proj"] else None, (st["exp"]["sig"], st["exp"]["blocker"], st["exp"]["folder"], repr(st["exp"]["note"])) if st["exp"] else None, (st["vscroll"], st["sx"], tuple(st["fxa"].values()), st["drag_sty"], st["aspect"], tuple(st["sec_a"].values()), st["cfg"]["vis_open"]) if self.tab == "visuals" else None)
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
            if self.tab in ("visuals", "scenes", "export"):
                self.draw_preview(W, H, e, blur=ob)
            self.draw_dock(W, H, e, ob)
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
        d["rot"] += fxv(cfg, "spin") * (0.22 + d["spin"] + 0.9 * mid) * dt
        d["flow"] += dt * (0.16 + 0.9 * mid + 0.5 * high)
        d["hue"] += dt * (0.015 + 0.10 * high) * fxv(cfg, "color")
        pe = min(1.0, d["bass"] * 1.2 + 0.8 * kick)
        d["penv"] += (pe - d["penv"]) * (1 - math.exp(-dt * (14.0 if pe > d["penv"] else 2.2)))
        d["psy"] = (d["psy"] + dt * (1.0 + 4.5 * d["penv"] + 0.8 * mid)) % TIME_WRAP
        d["wave"] = (d["wave"] + dt * (0.7 + 1.8 * energy) * (0.5 + 0.5 * min(fxv(cfg, "color"), 2.0))) % TAU_F
        d["kenv"] += (pe - d["kenv"]) * (1 - math.exp(-dt * (12.0 if pe > d["kenv"] else 2.0)))
        d["zk"] = (d["zk"] + dt * fxv(cfg, "kzoom") * (0.36 + 1.5 * d["kenv"])) % FLOWK_PERIOD
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
        vid = m == VIDEO_MODE
        mdl = self.media
        if vid and not mdl.items:                                # no clip: the card plays the built-in example clip (scene changes and all)
            if self.card_ex is None:
                self.card_ex = ExampleSource(ctx, w, h, 1.5)
                self.card_head = self.card_ex.make_head()
                self.card_head.play_from(ExampleSource.SCENE_S * random.randint(0, 4))
            fr = self.card_head.get()[1]
            rgba = np.empty((h, w, 4), np.uint8)
            rgba[..., :3] = fr[::-1]
            rgba[..., 3] = 255
            self.card_tex[m][0].write(rgba.tobytes())
            self.screen_fbo.use()
            return
        self.card_tex[m][1].use()
        ctx.viewport = (0, 0, w, h)
        has = bool(mdl.items) and cfg["media_on"]
        for name, val in (
            ("uRes", (float(w), float(h))), ("uRot", d["rot"] % TAU_F), ("uFlow", d["flow"] % 2.0), ("uFlowK", d["zk"]),
            ("uZoom", d["bass"] * 0.075 * fxv(cfg, "zoom")), ("uAb", 0.0025 + d["bass"] * 0.040 * fxv(cfg, "ab")), ("uBass", float(d["bass"])),
            ("uMid", float(d["mid"])), ("uHigh", float(d["high"])), ("uEnergy", float(d["energy"])), ("uHue", d["hue"] % 4.0),
            ("uPsyT", float(d["psy"])), ("uHueWave", (min(1.0, fxv(cfg, "color") / 3.0), float(d["wave"]))), ("uTime", d["t"] + 7.0),
            ("uWarp", 0.02 + 0.13 * d["mid"] + 0.05 * d["high"]), ("uDim", (1.0 if has else 0.0) if vid else 1.0),
            ("uMode", 0 if vid else m), ("uMOn", 0.0),
        ):
            if name in self.prog:
                self.prog[name].value = val
        if vid and has:
            mdl.bind(self.prog, cfg["media_peak"], 0.0, 3, 1.0, (0.0, 0.0, 0.0, 0.0), solo=True)
        self.vao.render(moderngl.TRIANGLE_STRIP)
        self.screen_fbo.use()

    def draw_loading(self, W, H, now):
        """The "Loading Scenes" cover: a blurred Psychedelic picture behind the progress card; it blurs in when it comes up and blurs out when done."""
        ctx = self.ctx
        a = self.scan_a
        e = a * a * (3.0 - 2.0 * a)
        if self.load_bg is None:
            tx = ctx.texture((640, 360), 4)
            tx.filter = (moderngl.LINEAR, moderngl.LINEAR)
            tx.repeat_x = tx.repeat_y = False
            self.load_bg = (tx, ctx.framebuffer(color_attachments=[tx]))
        tx, fbo = self.load_bg
        fbo.use()
        ctx.viewport = (0, 0, 640, 360)
        for name, val in (("uRes", (640.0, 360.0)), ("uRot", (now * 0.35) % TAU_F), ("uFlow", (now * 0.2) % 2.0), ("uFlowK", 0.8), ("uZoom", 0.0),
                          ("uAb", 0.003), ("uBass", 0.18 + 0.10 * math.sin(now * 2.0)), ("uMid", 0.30), ("uHigh", 0.20), ("uEnergy", 0.5),
                          ("uHue", (now * 0.04) % 4.0), ("uTime", now % TIME_WRAP), ("uPsyT", (now * 1.4) % TIME_WRAP), ("uWarp", 0.06),
                          ("uDim", 1.0), ("uMode", 4), ("uMOn", 0.0)):
            if name in self.prog:
                self.prog[name].value = val
        self.vao.render(moderngl.TRIANGLE_STRIP)
        self.screen_fbo.use()
        ctx.viewport = (0, 0, W, H)
        sc = max(0.7, min(W / 1280.0, H / 720.0))
        rb = (26.0 + 70.0 * (1.0 - e)) * sc                        # screen px: soft at rest, heavier while it blurs in / out
        (ncx, ncy), (hx, hy) = ndc_rect(0, 0, W, H, W, H)
        pr = self.pblur_prog
        pr["uCenter"].value, pr["uHalf"].value = (ncx, ncy), (hx, hy)
        pr["uPx"].value = (rb / W, rb / H)
        pr["uAlpha"].value = float(e)
        tx.use(0)
        pr["uTex"].value = 0
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self.pblur_vao.render(moderngl.TRIANGLE_STRIP)
        ctx.disable(moderngl.BLEND)
        self.quads.dim((W, H), 0.40 * e)
        self.loading.draw((W, H), e, self.scan_stat[0], self.scan_stat[1], self.scan_stat[2], now, self.scan_stat[3], self.scan_stat[4],
                          blur=(1.0 - e) * 14.0 * sc, pblur=(self.pblur_prog, self.pblur_vao))

    def draw_card_demo(self, W, H, e, m, mix):
        """Style card m's picture plays the live demo, blur-fading in / out (drawn over the panel, clipped to the open part of the section)."""
        if self.dlg:                                          # a question is on screen: no live picture may float over it
            return
        lay = self.vis_layout()
        L = lay["style"]
        if L["a"] < 0.01:
            return
        x0, y0, w, h = self.panel_rect
        k = w / PW
        cw = STY_CW
        iw = int(cw - 12)
        ih = int(iw * 9 / 16)
        card_h = ih + 12 + 26
        key = ("style", m)
        pp = (self.press.get(key) or {}).get("v", 0.0)
        hv = (self.hov_sp.get(key) or {}).get("v", 0.0)
        sc = (1.0 - 0.07 * pp) * (1.0 + 0.05 * hv)
        j = MODE_ORDER.index(m) if m in MODE_ORDER else m
        cx = STY_X0 + j * STY_PITCH + cw / 2 - self.style_xd
        if cx + cw / 2 < STY_X0 or cx - cw / 2 > STY_X0 + STY_VW:
            return
        d = min(cx - STY_X0, STY_X0 + STY_VW - cx)                            # how deep the card's middle is inside the row
        edge = 1.0
        if cx - STY_X0 < STY_X0 + STY_VW - cx:
            amt = min(1.0, self.style_xd / 40.0)
        else:
            amt = min(1.0, (sty_max() - self.style_xd) / 40.0)
        edge = 1.0 - (1.0 - max(0.0, min(1.0, (d + cw / 2) / (cw * 0.75 + 1e-6)))) * amt
        cy = L["ct"] - self.vscroll_d + card_h / 2
        lx, ly = cx - cw * sc / 2 + 6 * sc, cy - card_h * sc / 2 + 6 * sc
        lw, lh = iw * sc, ih * sc
        clip_top = max(HDR_Y, L["hy"] + SEC_HDR_H - self.vscroll_d)
        clip_bot = min(FTR_Y, L["hy"] + SEC_HDR_H - self.vscroll_d + L["vis"])
        top, bot = y0 + clip_top * k, y0 + clip_bot * k
        if bot <= top:
            return
        ctx = self.ctx
        ml = STY_MARGIN * (1.0 - min(1.0, self.style_xd / 40.0))
        mr = STY_MARGIN * (1.0 - min(1.0, (sty_max() - self.style_xd) / 40.0))
        sl, sr = x0 + (STY_X0 - ml) * k, x0 + (STY_X0 + STY_VW + mr) * k
        ctx.scissor = (int(sl), int(H - bot), int(sr - sl), int(max(1, bot - top)))
        (ncx, ncy), (hx, hy) = ndc_rect(x0 + lx * k, y0 + ly * k, lw * k, lh * k, W, H)
        pr = self.prev_prog
        pr["uCenter"].value, pr["uHalf"].value = (ncx, ncy), (hx, hy)
        pr["uSize"].value = (float(lw * k), float(lh * k))
        pr["uRad"].value = 10.0 * k * sc
        me = mix * mix * (3.0 - 2.0 * mix)
        pr["uAlpha"].value = float(min(1.0, e ** 1.2) * min(1.0, me * 1.4) * edge)
        pr["uFlip"].value = 0
        pr["uBlur"].value = float(((1.0 - me) * 8.0 + (1.0 - edge) * 8.0) * k)             # arrives blurred and sharpens; leaves by blurring away again
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
        self.draw_dock(W, H, e)
        if self.tab_old in ("visuals", "scenes", "export") and a_out > 0.01:
            self.draw_preview(W, H, e, self.tab_old, a_out, rmax * ss(0.0, 0.75, k))
        if self.tab in ("visuals", "scenes", "export") and a_in > 0.01:
            self.draw_preview(W, H, e, self.tab, a_in, rmax * (1.0 - ss(0.25, 1.0, k)))

    # ---------------------------------------------------------------- input
    def handle_key(self, k, now):
        a, cfg = self.audio, self.cfg
        if self.dlg:
            if k == pygame.K_ESCAPE:
                self.dlg_cancel()
            return
        if k == pygame.K_ESCAPE and self.menu_open and self.tab == "home" and not self.in_session:
            return                                             # nothing to go back to yet
        if k == pygame.K_SPACE:
            self.play_pause()
        elif k == pygame.K_ESCAPE:
            self.open_menu(not self.menu_open)
        elif k in (pygame.K_f, pygame.K_F11):
            self.toggle_fullscreen()
        elif k == pygame.K_q:
            self.request_quit()
        elif k in (pygame.K_m, pygame.K_TAB):
            cfg["mode"] = MODE_ORDER[(MODE_ORDER.index(cfg["mode"]) + 1) % len(MODE_ORDER)] if cfg["mode"] in MODE_ORDER else 0
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
                self.request_quit()
            elif e.type == pygame.DROPFILE:
                if self.dlg:
                    continue
                if e.file.lower().endswith(PROJ_EXT):
                    self.proj_open_flow(e.file)
                else:
                    n0 = len(self.playlist) + len(self.media.items)
                    self.add_files([e.file])
                    if self.menu_open and self.tab == "home" and len(self.playlist) + len(self.media.items) > n0:
                        self.leave_home(keep_menu=True, tab="queue")
            elif e.type == pygame.MOUSEMOTION:
                self.last_mouse = now
                self.mouse = e.pos
                pygame.mouse.set_visible(True)
                if self.menu_open:
                    self.menu_motion(*e.pos)
            elif e.type == pygame.MOUSEWHEEL:
                if self.menu_open and self.dlg:
                    d = self.dlg
                    d["scroll"] = max(0, min(max(0, len(d["items"]) - DLG_ROWS), d["scroll"] - e.y))
                    self.dlg_ver += 1
                elif self.menu_open and self.tab == "home":
                    self.home_row = max(0, min(max(0, self.home_rows() - HOME_ROWS_VIS), self.home_row - e.y))
                elif self.menu_open:
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
                    elif self.tab == "visuals" and self.hover and self.hover[0] in ("style", "stylerow", "stysb"):
                        d_ = e.y if e.y else -getattr(e, "x", 0)
                        self.style_x = max(0.0, min(sty_max(), self.style_x - d_ * STY_PITCH))
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
                self.end_drag()
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
        proj_arg = next((a for a in argv if a.lower().endswith(PROJ_EXT) and os.path.isfile(a)), None)
        files = [a for a in argv if a != proj_arg]
        if files:
            self.add_files(files)
            self.in_session = True
        clock = pygame.time.Clock()
        last = time.perf_counter()
        cfg = self.cfg
        saved = [p for p in (cfg.get("media_paths") or "").split("|") if p] or ([cfg["media_path"]] if cfg.get("media_path") else [])
        saved = [p for p in saved if os.path.isfile(p)] if cfg["media_remember"] else []
        if saved:
            self.media.add(saved)
        self.sync_media_cfg()
        if saved:
            self.in_session = True
        if proj_arg:
            self.proj_load_checked(proj_arg)
        elif not files:                                        # no files given: start on the home screen
            self.refresh_home()
            self.tab = self.tab_seen = "home"
            self.open_menu(True)

        while self.running:
            now = time.perf_counter()
            dt = min(now - last, 0.1)
            last = now
            a = self.audio
            ana = self.ana_spot if (self.spotify_on and self.ana_spot is not None) else self.ana

            self.events(now)
            if self.exit_at is not None and now >= self.exit_at:
                self.running = False
            self.proj_poll()
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
            mode_now = 4 if not self.in_session else cfg["mode"]            # the home screen at launch always sits over Psychedelic
            vid = mode_now == VIDEO_MODE                          # Video style: only the footage, never idles with the spiral
            rms = 0.0 if chunk is None else float(np.sqrt(np.mean(np.square(chunk))))
            self.quiet_t = 0.0 if rms > 0.0015 else getattr(self, "quiet_t", 0.0) + dt
            live = playing and self.quiet_t < 0.8                    # audio is actually coming in (not paused / stopped / silent)
            dom = cfg["domination"]
            sm = fxv(cfg, "beat_smooth")                                  # 0 = raw & snappy ... 1 = very soft / flowing
            s_bass = self.smooth("bass", ana.bass, dt, sm)
            s_mid = self.smooth("mid", ana.mid, dt, sm)
            s_high = self.smooth("high", ana.high, dt, sm)
            s_energy = self.smooth("energy", ana.energy, dt, sm)
            s_onset = self.smooth("onset", ana.onset, dt, sm)
            if ana.beat > 0:
                self.spin_kick += 2.6 * ana.beat * (1.0 - 0.5 * sm)
            self.spin_kick *= math.exp(-dt * 3.6 / (1.0 + 1.5 * sm))
            base = 0.30 if idle else 0.22
            omega = fxv(cfg, "spin") * (base + self.spin_kick + 0.9 * s_mid + 0.5 * s_onset)   # spin slider at 0% = no rotation at all
            if not playing and not idle:
                omega = 0.0
            omega = self.smooth("omega", omega, dt, sm * 0.8)
            self.rot += omega * dt
            self.flow += dt * ((0.10 if idle else 0.16) + (0.9 * s_mid + 0.5 * s_high) * (1 if playing else 0))
            self.hue += dt * (0.015 + 0.10 * s_high) * fxv(cfg, "color")
            # Psychedelic flow clock: always forward; kicks / snares / bass make it surge faster, then it glides back to a calm drift
            pe = min(1.0, s_bass * 1.2 + 0.7 * ana.beat + 0.9 * max(ana.kick, ana.snare)) if playing else 0.0
            self.psy_env = getattr(self, "psy_env", 0.0)
            self.psy_env += (pe - self.psy_env) * (1 - math.exp(-dt * ((14.0 / (1.0 + 3.0 * sm)) if pe > self.psy_env else 2.2 / (1.0 + 1.5 * sm))))
            self.psy_t = (getattr(self, "psy_t", 0.0) + dt * (1.0 + 4.5 * self.psy_env + 0.8 * s_mid)) % TIME_WRAP
            self.wave_ph = (getattr(self, "wave_ph", 0.0) + dt * (0.7 + 1.8 * s_energy) * (0.5 + 0.5 * min(fxv(cfg, "color"), 2.0))) % TAU_F
            # Kaleidoscope: slow endless zoom. Smoothed speed -> never jerks; freezes (eased) while paused.
            self.zk_speed += ((0.0 if (not playing and not idle) else 1.0) - self.zk_speed) * (1 - math.exp(-dt * 3.0))
            boost = (0.20 * s_mid + 0.10 * s_high + 0.06 * s_energy) if playing else 0.0
            self.zk_boost += (boost - self.zk_boost) * (1 - math.exp(-dt * 2.5))
            # tunnel flight: cruising speed, surging forward on kicks / snares / bass, then easing back (never reverses)
            ke = min(1.0, s_bass * 1.2 + 0.7 * ana.beat + 0.9 * max(ana.kick, ana.snare)) if playing else 0.0
            self.k_env = getattr(self, "k_env", 0.0)
            self.k_env += (ke - self.k_env) * (1 - math.exp(-dt * ((12.0 / (1.0 + 3.0 * sm)) if ke > self.k_env else 2.0 / (1.0 + 1.5 * sm))))
            self.zoomk = (self.zoomk + dt * self.zk_speed * fxv(cfg, "kzoom") * (0.36 + 1.5 * self.k_env + self.zk_boost)) % FLOWK_PERIOD
            zoom = s_bass * 0.075 * fxv(cfg, "zoom")                 # bass zoom + chromatic aberration stay on in
            ab = 0.0025 + s_bass * 0.040 * fxv(cfg, "ab")             # Domination Mode; the popups are added on top
            ubass = s_bass
            zmode = int(cfg["zoom_mode"])
            shake = (0.0, 0.0)
            if zmode == 2:                                       # random area: every new bass swell zooms toward a fresh spot
                if s_bass > 0.38 and not self.zm_hi:
                    self.zm_hi = True
                    for _ in range(8):
                        ang = random.uniform(0.0, 2.0 * math.pi)
                        if abs((ang - self.zm_ang + math.pi) % (2.0 * math.pi) - math.pi) > 1.0:
                            break
                    rad = random.uniform(0.40, 0.90)
                    self.zm_ang, self.zm_t = ang, [math.cos(ang) * rad * 1.3, math.sin(ang) * rad * 0.85]
                elif s_bass < 0.18:
                    self.zm_hi = False
            else:
                self.zm_t = [0.0, 0.0]
                self.zm_hi = False
            for i_ in (0, 1):                                    # the focus glides to its target (critically damped, so no snapping)
                self.zm_c[i_], self.zm_v[i_] = spring_step(self.zm_c[i_], self.zm_v[i_], self.zm_t[i_], 55.0, 14.8, dt)
            if zmode == 1:                                       # shaky: the zoom trembles, harder the further it is in
                amp = min(zoom, 0.45) * 0.42
                shake = (amp * (math.sin(now * 43.0) + 0.6 * math.sin(now * 71.3 + 1.7) + 0.4 * math.sin(now * 29.1 + 4.0)) / 2.0,
                         amp * (math.sin(now * 47.0 + 2.1) + 0.6 * math.sin(now * 67.9 + 0.4) + 0.4 * math.sin(now * 31.7 + 5.2)) / 2.0)
            if mode_now == 3:                                 # Kaleidoscope: the beat drives the tunnel speed, no zoom-punch
                zoom = 0.0
            if mode_now == 4:                                 # Psychedelic: no zoom-punch / bloom, the beat speeds the flow instead
                zoom, ubass = 0.0, 0.0
            if zoom == 0.0:
                shake = (0.0, 0.0)
            trig = playing and (ana.kick > 0 or (ana.bass > 0.6 and self.bfx_prev <= 0.6))
            self.bfx_prev = ana.bass
            self.bfx_env *= math.exp(-dt * 7.0)
            if trig:                                             # bass distortion: a hit opens it fully, then it falls away
                self.bfx_env = 1.0
                self.bfx_seed = random.uniform(0.0, 100.0)
                self.bfx_pick = random.randrange(3)              # what "Random" uses for this hit
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
            mdl.set_order(cfg["scene_order"])
            # ---- "Loading Scenes" cover: new clips are scanned for scenes; when the scan is done playback restarts clean
            for it_ in mdl.items:
                if it_["id"] not in self.scan_watch and (it_["src"] is None or (it_["src"].kind == "video" and it_["src"].scan < 1.0)):
                    self.scan_watch.add(it_["id"])
            if self.scan_watch:
                alive = {x["id"]: x for x in mdl.items}
                self.scan_watch &= set(alive)
                busy, fin, cur_ = False, 0, None
                for it_ in mdl.items:                               # stack order: the first clip still scanning is "the current one"
                    if it_["id"] not in self.scan_watch:
                        continue
                    src_ = it_["src"]
                    if src_ is not None and src_.kind == "video":
                        self.scan_vid = True
                    done_ = src_ is not None and (src_.kind != "video" or src_.scan >= 1.0)
                    if done_:
                        fin += 1
                    else:
                        busy = True
                        if cur_ is None or (getattr(src_, "scanning", False) and not getattr(cur_["src"], "scanning", False)):
                            cur_ = it_
                nw = len(self.scan_watch)
                if busy:
                    sc_ = cur_["src"]
                    self.scan_stat = (100.0 * (sc_.scan if sc_ is not None else 0.0), len(sc_.scenes) if sc_ is not None else 0,
                                      os.path.basename(cur_["path"]), fin + 1, nw)
                else:
                    self.scan_stat = (100.0, self.scan_stat[1], self.scan_stat[2], nw, nw)
                    if self.scan_vid:
                        mdl.restart(now)                               # every scene is known: begin again so nothing starts half-prepared
                    self.scan_watch, self.scan_vid = set(), False
                self.scan_a = min(1.0, self.scan_a + dt * 2.4)
            else:
                self.scan_a = max(0.0, self.scan_a - dt * 2.2)
            ex_vid = vid and not mdl.items                                  # Video style, empty stack: the built-in example clip plays
            mdl.want_example = ex_vid
            mdl.update(now, dt, cfg["media_on"] or ex_vid, (not playing and not idle) and not vid, cfg["media_rate"])
            warp = 0.02 + 0.13 * s_mid + 0.05 * s_high
            vid_blank = vid and bool(mdl.items) and not cfg["media_on"]        # Video style with nothing to show: plain black, no spiral
            target_dim = 0.0 if vid_blank else (1.0 if vid else (0.6 if idle else (1.0 if playing else 0.5)))
            self.dim += (target_dim - self.dim) * (1 - math.exp(-dt * 5))

            # ---- menu animation (0 = closed, 1 = open) ----
            W, H = pygame.display.get_window_size()
            rate = (1.0 / 0.42) if self.menu_open else -(1.0 / 0.32)
            self.menu_p = max(0.0, min(1.0, self.menu_p + rate * dt))
            p = self.menu_p
            e = p * p * p * (p * (6 * p - 15) + 10)            # smootherstep
            # audio low-pass follows the menu curve - except on the Beat tab, where you tune detection by ear (clean audio)
            want = 0.0 if self.tab in ("beat", "export") else 1.0
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
            if self.tab == "visuals" and self.menu_open and cfg_mode_changed(self):
                self.style_reveal(self.cfg["mode"], instant=self._sx_mode is None)
                self._sx_mode = self.cfg["mode"]
            if self.tab == "visuals" and self.vscroll > 0:                     # the page can get shorter (Domination sliders hide): stay inside it
                self.vscroll = min(self.vscroll, float(self.vis_layout()["max_scroll"]))
            offs = (self.cfg.get("fx_off") or "").split(",")
            for k_ in VISUAL_KEYS:                                              # the effect dots fade in / out
                tg_ = 0.0 if k_ in offs else 1.0
                v_ = self.fxa.get(k_, tg_)
                v_ += (tg_ - v_) * (1 - math.exp(-dt * 16.0))
                self.fxa[k_] = tg_ if abs(tg_ - v_) < 0.01 else v_
            self.style_xd += (self.style_x - self.style_xd) * (1 - math.exp(-dt * 14.0))
            if abs(self.style_x - self.style_xd) < 0.05:
                self.style_xd = self.style_x
            self.dock_goal = self.dock_wanted()
            self.dock_p = max(0.0, min(1.0, self.dock_p + (dt / DOCK_S if self.dock_goal else -dt / DOCK_S)))
            if not self.menu_open and self.menu_p <= 0.0:
                self.dock_p, self.dock_goal = 0.0, False
            if self.menu_open and self.tab == "visuals" and self.vscroll_d != getattr(self, "_vs_seen", None):
                self._vs_seen = self.vscroll_d
                self.hover = self.panel.pick(*self.to_logical(*pygame.mouse.get_pos()))   # the page moved under a still mouse
            if self.tab_k < 1.0:
                self.tab_k = min(1.0, self.tab_k + dt / 0.36)
            self.update_style_demo(dt)
            menu_visible = p > 0.0

            # ---- render scene (offscreen only while the menu is visible) ----
            RW, RH = self.exp["size"] if (self.exp and self.exp["capturing"]) else (W, H)     # recording: the scene renders at the export size
            self.ensure_targets(RW, RH)
            if mode_now != self.mode_shown:                  # visual mode changed: freeze the old look
                src_f = self.trans_f if self.mode_t < 1.0 else self.scene_f
                self.ctx.copy_framebuffer(self.prev_f, src_f)
                self.mode_shown = mode_now
                self.mode_t = 0.0
            self.scene_f.use()
            self.ctx.viewport = (0, 0, RW, RH)
            for name, val in (
                ("uRes", (float(RW), float(RH))), ("uRot", self.rot % TAU_F), ("uFlow", self.flow % 2.0), ("uFlowK", self.zoomk % FLOWK_PERIOD),
                ("uZoom", zoom), ("uZoomC", (self.zm_c[0], self.zm_c[1])), ("uShake", shake),
                ("uAb", ab), ("uBass", float(ubass)), ("uMid", float(s_mid)),
                ("uHigh", float(s_high)), ("uEnergy", float(s_energy)), ("uHue", self.hue % 4.0), ("uPsyT", float(self.psy_t)), ("uHueWave", (min(1.0, fxv(cfg, "color") / 3.0), float(self.wave_ph))),
                ("uTime", now % TIME_WRAP), ("uWarp", warp), ("uDim", self.dim), ("uMode", 0 if vid else mode_now),
            ):
                if name in self.prog:            # the GLSL compiler drops unused uniforms
                    self.prog[name].value = val
            amt = (cfg["media_calm"] + (cfg["media_peak"] - cfg["media_calm"]) * self.m_env) if cfg["media_auto"] else cfg["media_peak"]
            if vid and not live:
                amt = cfg["media_peak"]                                    # no audio: the footage stays at full brightness
            mdl.bind(self.prog, amt, s_bass * fxv(cfg, "media_pulse"), 3 if vid else min(2, cfg["media_blend"]), flash_gain, sway_v, solo=vid)
            self.vao.render(moderngl.TRIANGLE_STRIP)

            # ---- domination popups (bass-triggered) ----
            if dom and playing:
                cd = 0.30 / max(0.05, fxv(cfg, "dom_rate"))
                if ana.bass > 0.55 and self.dom_prev <= 0.55 and now - self.dom_last > cd:
                    self.dom_last = now
                    self.glitch.spawn(self.phrases, now, (RW, RH), fxv(cfg, "dom_size"))
            self.dom_prev = ana.bass
            if not dom:
                self.glitch.active.clear()
            self.glitch.draw(now, (RW, RH))
            if fxv(cfg, "bass_fx_amt") > 0.001 and self.bfx_env > 0.01:
                self.bass_fx_pass(now, RW, RH)

            final_t = self.scene_t
            if self.mode_t < 1.0:                               # blur-fade between visual modes
                self.mode_t = min(1.0, self.mode_t + dt / 0.8)
                kk = self.mode_t * self.mode_t * self.mode_t * (self.mode_t * (6 * self.mode_t - 15) + 10)
                self.transition_pass(kk)
                final_t = self.trans_t
            self.final_t = final_t
            self.export_tick(now)
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
                if self.exp:
                    xx = self.exp
                    lines = [((("\u25CF  Recording  " + fmt_clock(xx["cur_ts"]) + " / " + fmt_clock(xx["total"])) if xx["capturing"] else "Finishing the video\u2026"), 26),
                             ("Esc  opens the menu   \u00b7   Export tab: Stop", 18)]
                    alpha = 0.55
                elif spot and now - self.toast_t < 3.2:
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
                    lines, alpha = [("Media layer is off", 40)], 0.55 + 0.35 * math.sin(now * 1.6)
                elif vid and (mdl.items or ex_vid) and now - self.toast_t >= 3.2:
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

            if self.scan_a > 0.003:
                Wc, Hc = pygame.display.get_window_size()
                self.draw_loading(Wc, Hc, now)
            pygame.display.flip()

            if self.is_fs and not self.menu_open and now - self.last_mouse > 2.0:
                pygame.mouse.set_visible(False)
            clock.tick(240)

        self.export_shutdown()
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
