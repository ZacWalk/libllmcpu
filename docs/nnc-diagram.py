#!/usr/bin/env python3
"""Draw nnc-diagram.png, the architecture figure at the top of README.md.

    python docs/nnc-diagram.py        # writes nnc-diagram.png in the repo root

Needs Python 3.8+ and Pillow. Fonts are Segoe UI and Consolas on Windows,
DejaVu Sans elsewhere. The text below describes the code, so when a file,
layer or flow in docs/design.md changes, change it here too and re-run.
Text that does not fit its box is reported, and the exit code is then 1.

Coordinates are layout units. The PNG has OUTPUT_SCALE pixels per unit, so
it stays sharp when zoomed; everything is drawn SUPERSAMPLE times larger
again and shrunk at the end, which smooths lines and curves.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

OUT = Path(__file__).resolve().parent.parent / "nnc-diagram.png"
W, H = 2400, 1672  # layout units
OUTPUT_SCALE = 2  # PNG pixels per unit: 4800 x 3344
SUPERSAMPLE = 2
SCALE = OUTPUT_SCALE * SUPERSAMPLE  # drawing pixels per unit

FONT_FILES = {
    "regular": ("segoeui.ttf", "DejaVuSans.ttf"),
    "semibold": ("seguisb.ttf", "DejaVuSans-Bold.ttf"),
    "bold": ("segoeuib.ttf", "DejaVuSans-Bold.ttf"),
    "mono": ("consola.ttf", "DejaVuSansMono.ttf"),
    "monobold": ("consolab.ttf", "DejaVuSansMono-Bold.ttf"),
}

PAPER = "#f8fafc"
WHITE = "#ffffff"
INK = "#0f172a"
TEXT = "#1e293b"
MUTED = "#64748b"
RULE = "#cbd5e1"
ARROW = "#475569"


@dataclass(frozen=True)
class Theme:
    strong: str  # card headers, borders, step titles
    tint: str  # band background
    label: str  # band label column, step background


APP = Theme("#c2410c", "#fff7ed", "#ffedd5")
API = Theme("#4338ca", "#eef2ff", "#e0e7ff")
ENGINE = Theme("#0f766e", "#f0fdfa", "#ccfbf1")
KERNEL = Theme("#6d28d9", "#f5f3ff", "#ede9fe")
OS = Theme("#334155", "#f1f5f9", "#e2e8f0")


@dataclass(frozen=True)
class Box:
    x: float
    y: float
    w: float
    h: float

    @property
    def right(self) -> float:
        return self.x + self.w

    @property
    def bottom(self) -> float:
        return self.y + self.h

    @property
    def cx(self) -> float:
        return self.x + self.w / 2

    @property
    def cy(self) -> float:
        return self.y + self.h / 2


_fonts: dict[tuple[str, float], ImageFont.FreeTypeFont] = {}


def font(style: str, size: float) -> ImageFont.FreeTypeFont:
    key = (style, size)
    if key not in _fonts:
        for name in FONT_FILES[style]:
            try:
                _fonts[key] = ImageFont.truetype(name, round(size * SCALE))
                break
            except OSError:
                continue
        else:
            sys.exit(f"nnc-diagram: needs one of these fonts: {', '.join(FONT_FILES[style])}")
    return _fonts[key]


def px(v: float) -> int:
    return round(v * SCALE)


def grid(x0: float, x1: float, n: int, gap: float) -> list[tuple[float, float]]:
    """n equal columns between x0 and x1: (left, width) of each."""
    w = (x1 - x0 - gap * (n - 1)) / n
    return [(x0 + i * (w + gap), w) for i in range(n)]


class Canvas:
    def __init__(self) -> None:
        self.img = Image.new("RGBA", (px(W), px(H)), PAPER)
        self.draw = ImageDraw.Draw(self.img)
        self.problems: list[str] = []

    # ---- primitives ---------------------------------------------------------

    def rounded(self, b: Box, r: float, fill=None, outline=None, width: float = 0, corners=None) -> None:
        self.draw.rounded_rectangle((px(b.x), px(b.y), px(b.right), px(b.bottom)), radius=px(r), fill=fill,
                                    outline=outline, width=px(width) if width else 0, corners=corners)

    def shadow(self, b: Box, r: float, blur: float = 8, dy: float = 3, alpha: int = 34) -> None:
        pad = blur * 3
        patch = Image.new("RGBA", (px(b.w + 2 * pad), px(b.h + 2 * pad)), (0, 0, 0, 0))
        ImageDraw.Draw(patch).rounded_rectangle((px(pad), px(pad), px(pad + b.w), px(pad + b.h)),
                                                radius=px(r), fill=(15, 23, 42, alpha))
        patch = patch.filter(ImageFilter.GaussianBlur(px(blur) / 2))
        self.img.alpha_composite(patch, (px(b.x - pad), px(b.y - pad + dy)))
        self.draw = ImageDraw.Draw(self.img)

    def width(self, s: str, style: str, size: float) -> float:
        return self.draw.textlength(s, font=font(style, size)) / SCALE

    def text(self, x: float, y: float, s: str, style: str = "regular", size: float = 19, fill: str = TEXT,
             anchor: str = "la") -> None:
        self.draw.text((px(x), px(y)), s, font=font(style, size), fill=fill, anchor=anchor)

    def fits(self, s: str, style: str, size: float, room: float, where: str) -> None:
        if self.width(s, style, size) > room:
            self.problems.append(f"{where}: '{s}' needs {self.width(s, style, size):.0f}px of {room:.0f}")

    def wrap(self, s: str, style: str, size: float, room: float, where: str) -> list[str]:
        lines, cur = [], ""
        for word in s.split(" "):
            trial = f"{cur} {word}" if cur else word
            if cur and self.width(trial, style, size) > room:
                lines.append(cur)
                cur = word
            else:
                cur = trial
        lines.append(cur)
        for line in lines:
            self.fits(line, style, size, room, where)
        return lines

    def line(self, pts, color: str = ARROW, width: float = 2.6) -> None:
        self.draw.line([(px(a), px(b)) for a, b in pts], fill=color, width=px(width), joint="curve")

    def arrow(self, pts, color: str = ARROW, width: float = 2.6, head: float = 13) -> None:
        (x0, y0), (x1, y1) = pts[-2], pts[-1]
        ang = math.atan2(y1 - y0, x1 - x0)
        ca, sa = math.cos(ang), math.sin(ang)
        # End the shaft inside the head so its square end cannot poke past the tip.
        self.line(list(pts[:-1]) + [(x1 - head * 0.7 * ca, y1 - head * 0.7 * sa)], color, width)
        half = head * 0.48
        corners = ((x1, y1), (x1 - head * ca + half * sa, y1 - head * sa - half * ca),
                   (x1 - head * ca - half * sa, y1 - head * sa + half * ca))
        self.draw.polygon([(px(a), px(b)) for a, b in corners], fill=color)

    def tag(self, x: float, y: float, s: str, bg: str, fill: str = MUTED, size: float = 15) -> None:
        w = self.width(s, "semibold", size)
        self.rounded(Box(x - w / 2 - 6, y - size * 0.75, w + 12, size * 1.5), 6, fill=bg)
        self.text(x, y, s, "semibold", size, fill, anchor="mm")

    def badge(self, right: float, cy: float, label: str, fg: str, bg: str) -> float:
        w = self.width(label, "bold", 12.5) + 14
        self.rounded(Box(right - w, cy - 10.5, w, 21), 10.5, fill=bg)
        self.text(right - w / 2, cy, label, "bold", 12.5, fg, anchor="mm")
        return w

    # ---- composite shapes ---------------------------------------------------

    def band(self, b: Box, theme: Theme, labels) -> None:
        """A layer: tinted background with its name in a column on the left."""
        self.rounded(b, 18, fill=theme.tint)
        self.rounded(Box(b.x, b.y, 150, b.h), 18, fill=theme.label, corners=(True, False, False, True))
        step = [size * 1.5 for _, _, size, _ in labels]
        y = b.cy - sum(step) / 2
        for (s, style, size, fill), dy in zip(labels, step):
            self.text(b.x + 75, y + dy / 2, s, style, size, fill, anchor="mm")
            y += dy

    def card(self, b: Box, title: str, lines, theme: Theme, avx2: bool, size: float = 18.5) -> None:
        """A source file: name in a coloured header, what it does below."""
        r, head = 12, 38
        self.shadow(b, r)
        self.rounded(b, r, fill=WHITE, outline=theme.strong, width=2)
        self.rounded(Box(b.x, b.y, b.w, head), r, fill=theme.strong, corners=(True, True, False, False))
        room = b.w - 28
        if avx2:
            room -= self.badge(b.right - 10, b.y + head / 2, "AVX2", theme.strong, WHITE) + 8
        self.fits(title, "monobold", 19, room, title)
        self.text(b.x + 14, b.y + head / 2, title, "monobold", 19, WHITE, anchor="lm")
        y = b.y + head + 9
        for line in lines:
            style = "regular"
            if isinstance(line, tuple):
                line, style = line
            for part in self.wrap(line, style, size, b.w - 28, title):
                self.text(b.x + 14, y, part, style, size, TEXT)
                y += size * 1.36
        if y > b.bottom - 4:
            self.problems.append(f"{title}: the text runs {y - b.bottom + 4:.0f}px past the card")

    def tab_card(self, b: Box, title: str, lines: list[str], theme: Theme, size: float = 17) -> None:
        """A one-row card: name on a coloured tab, what it does to its right."""
        r, tab = 12, 176
        self.shadow(b, r)
        self.rounded(b, r, fill=WHITE, outline=theme.strong, width=2)
        self.rounded(Box(b.x, b.y, tab, b.h), r, fill=theme.strong, corners=(True, False, False, True))
        self.text(b.x + tab / 2, b.cy, title, "monobold", 19, WHITE, anchor="mm")
        step = size * 1.36
        y = b.cy - step * len(lines) / 2
        for line in lines:
            self.fits(line, "regular", size, b.w - tab - 28, title)
            self.text(b.x + tab + 16, y + step / 2, line, "regular", size, TEXT, anchor="lm")
            y += step

    def panel(self, b: Box, title: str) -> float:
        self.shadow(b, 14)
        self.rounded(b, 14, fill=WHITE, outline=RULE, width=1.5)
        self.text(b.x + 22, b.y + 16, title, "bold", 22, INK)
        return b.y + 58

    def bullets(self, b: Box, y: float, items: list[str], size: float = 16.5) -> float:
        x, room = b.x + 22, b.w - 44 - 18
        for item in items:
            dot = y + size * 0.72
            self.draw.ellipse((px(x + 2), px(dot - 3.2), px(x + 8.4), px(dot + 3.2)), fill=MUTED)
            for line in self.wrap(item, "regular", size, room, "sidebar"):
                self.text(x + 18, y, line, "regular", size, TEXT)
                y += size * 1.38
            y += 7
        if y > b.bottom - 8:
            self.problems.append(f"sidebar panel at y={b.y:.0f}: {y - b.bottom + 8:.0f}px too much text")
        return y

    def step(self, b: Box, title: str, lines: list[str], theme: Theme) -> None:
        """One step of a flow."""
        self.rounded(b, 12, fill=theme.label, outline=theme.strong, width=1.6)
        self.fits(title, "bold", 19, b.w - 20, title)
        self.text(b.cx, b.y + 12, title, "bold", 19, theme.strong, anchor="mt")
        y = b.y + 42
        for line in lines:
            self.fits(line, "regular", 16.5, b.w - 20, title)
            self.text(b.cx, y, line, "regular", 16.5, TEXT, anchor="mt")
            y += 21.5


def flow(c: Canvas, y: float, name: str, call: str, note: str, steps, h: float = 90,
         gap: float = 38) -> list[Box]:
    """A row of steps joined by arrows, named on the left."""
    c.text(30, y + 8, name, "bold", 27, INK)
    c.text(30, y + 48, call, "mono", 16, MUTED)
    if note:
        c.text(30, y + 70, note, "mono", 15, MUTED)
    boxes = [Box(x, y, w, h) for x, w in grid(268, W - 30, len(steps), gap)]
    for b, (theme, title, lines) in zip(boxes, steps):
        c.step(b, title, lines, theme)
    for a, b in zip(boxes, boxes[1:]):
        c.arrow([(a.right + 5, a.cy), (b.x - 5, b.cy)])
    return boxes


def legend(c: Canvas, b: Box) -> None:
    c.rounded(b, 14, fill=WHITE, outline=RULE, width=1.5)
    sample = Box(b.x + 18, b.y + 16, 74, 30)
    c.rounded(sample, 8, fill=KERNEL.strong)
    c.badge(sample.right - 9, sample.cy, "AVX2", KERNEL.strong, WHITE)
    y = b.y + 11
    for line in c.wrap("compiled for AVX2 + FMA. Unmarked code (the app, baseline.cpp and sys_*.cpp) runs on "
                       "any x86-64 CPU, so the CPU check always runs first.", "regular", 16.5,
                       b.right - sample.right - 34, "legend"):
        c.text(sample.right + 16, y, line, "regular", 16.5, TEXT)
        y += 21.5
    ay = b.bottom - 19
    c.arrow([(b.x + 22, ay), (b.x + 90, ay)])
    c.text(sample.right + 16, ay, "calls or depends on", "regular", 16.5, TEXT, anchor="lm")


def main() -> int:
    c = Canvas()
    left, x0, x1 = 30, 196, 1800  # bands start at `left`; cards fill x0..x1
    side = Box(1830, 0, W - 30 - 1830, 0)  # the sidebar's x and width

    # ---- title ----------------------------------------------------------------
    c.text(left, 16, "nnc — Neural Net Compiler", "bold", 54, INK)
    c.text(left + 2, 94, "Runs Gemma 3, Gemma 4, Llama, Qwen and EmbeddingGemma GGUF models on the CPU through a "
                         "hand-written x86-64 AVX2 + FMA JIT · Windows x64 and Linux x64", "regular", 23, MUTED)
    legend(c, Box(side.x, 20, side.w, 112))

    # ---- appnnc ---------------------------------------------------------------
    c.band(Box(left, 150, x1 - left, 196), APP, [("appnnc", "bold", 27, APP.strong),
                                                ("the nnc app", "semibold", 17, TEXT),
                                                ("src/appnnc", "mono", 15, MUTED)])
    app = [Box(x, 166, w, 164) for x, w in grid(x0, x1, 6, 16)]
    for b, (title, lines) in zip(app, [
        ("main.cpp · cli.cpp", ["checks the CPU first", "parses the options", "runs the chosen mode"]),
        ("chat.cpp · embed.cpp", ["the REPL, -p, --translate;", "embeddings: similarities,",
                                  "or a JSON vector a line"]),
        ("models.cpp", ["finds models and their MTP", "drafters; downloads them",
                        ("model_catalog.def", "mono")]),
        ("devtools.cpp", ["--gguf-info, --inspect-*,", "--gemma-*, --perplexity", "reads model internals"]),
        ("tests.cpp", ["nnc --test: 63 tests of", "the JIT, kernels, GGUF,", "tokenizer; no model files"]),
        ("ui.cpp · platform.cpp", ["progress bars, prompts", "Ctrl-C scopes", "native calls via sys.h"]),
    ]):
        c.card(b, title, lines, APP, avx2=False)

    # ---- nnc.h ----------------------------------------------------------------
    api = Box(left, 360, x1 - left, 74)
    c.shadow(api, 14)
    c.rounded(api, 14, fill=API.strong)
    c.text(api.x + 22, api.y + 11, "nnc.h", "monobold", 27, WHITE)
    c.text(api.x + 22 + c.width("nnc.h", "monobold", 27) + 16, api.y + 16,
           "the public, plain-C API: the only installed header", "semibold", 21, "#e0e7ff")
    c.text(api.x + 22, api.y + 50, "nnc_model_load · nnc_session_create · eval · sample · generate · chat · "
                                   "translate · cancel · nnc_embed · nnc_last_error", "mono", 16, "#c7d2fe")
    c.text(api.right - 22, api.y + 18, "also for your programs:", "regular", 17, "#c7d2fe", anchor="ra")
    c.text(api.right - 22, api.y + 50, "find_package(nnc) → nnc::nnc · examples/*.c", "mono", 16, WHITE,
           anchor="ra")
    for i in (0, 1, 3, 4):  # main, chat, devtools and tests use the API
        c.arrow([(app[i].cx, app[i].bottom + 1), (app[i].cx, api.y - 1)])

    # ---- libnnc: engine ---------------------------------------------------------
    c.band(Box(left, 449, x1 - left, 398), ENGINE, [("libnnc", "bold", 27, ENGINE.strong),
                                                   ("engine", "semibold", 17, TEXT),
                                                   ("src/libnnc", "mono", 15, MUTED)])
    cols = grid(x0, x1, 4, 40)
    top = [Box(x, 465, w, 168) for x, w in cols]
    low = [Box(x, 657, w, 174) for x, w in cols]
    for b, (title, lines, avx2) in zip(top, [
        ("baseline.cpp", ["built without AVX, so it runs", "on any x86-64 CPU: the CPU", "check before model loading;",
                          "errors, logging, defaults"], False),
        ("nnc.cpp", ["the C API for models and", "sessions: argument checks,", "handle ownership; no C++",
                     "exception crosses the API"], True),
        ("session.cpp", ["context, K/V and recurrent state,", "put back on cancel; the sampler;",
                         "chat templates; MTP draft and", "verify; whole-character UTF-8"], True),
        ("tokenizer.cpp", ["SentencePiece, byte-level BPE", "and Gemma 4's BPE; user-defined",
                           "tokens kept whole; the turn", "format from the vocabulary"], True),
    ]):
        c.card(b, title, lines, ENGINE, avx2)
    for b, (title, lines) in zip(low, [
        ("gguf.cpp", ["GGUF v2 / v3, parsed in place", "from the memory map; every", "length checked, metadata",
                      "memory budgeted"]),
        ("model.cpp · load", ["gemma3 · gemma4 · llama · qwen,", "EmbeddingGemma, MTP drafters;",
                              "decodes weights in parallel", "into Q8_0, Q4_S or BF16"]),
        ("model.cpp · forward", ["embed → layers → norm → lm_head,", "or pool (embeddings); one",
                                 "layer_forward for decode, prefill,", "MoE, Gated DeltaNet, drafters"]),
        ("runtime.cpp", ["dtype table, tensor sizes,", "timers and the --perf", "phase profiler"]),
    ]):
        c.card(b, title, lines, ENGINE, avx2=True)
    for col in (0, 1):  # nnc.h is implemented by baseline.cpp and nnc.cpp
        c.arrow([(top[col].cx, api.bottom + 1), (top[col].cx, top[col].y - 1)])
    for a, b in zip(top, top[1:]):
        c.arrow([(a.right + 5, a.cy), (b.x - 5, b.cy)])
    for col in (1, 2):  # nnc.cpp loads models; session.cpp runs them
        c.arrow([(top[col].cx, top[col].bottom + 1), (low[col].cx, low[col].y - 1)])
    c.arrow([(low[1].x - 5, low[1].cy), (low[0].right + 5, low[0].cy)])

    # ---- libnnc: kernels and the JIT ---------------------------------------------
    c.band(Box(left, 862, x1 - left, 200), KERNEL, [("libnnc", "bold", 27, KERNEL.strong),
                                                   ("kernels & JIT", "semibold", 17, TEXT),
                                                   ("src/libnnc", "mono", 15, MUTED)])
    ker = [Box(x, 878, w, 168) for x, w in grid(x0, x1, 5, 56)]
    for b, (title, lines) in zip(ker, [
        ("nn_ops.cpp", ["SIMD norms, softmax, RoPE,", "GELU, SiLU, softcap;", "gemv dispatch; a worker",
                        "pool that sleeps when idle"]),
        ("jit_kernel.cpp", ["kernel cache keyed by", "(op, dtype, shape): JIT", "once, then reuse the",
                            "function pointer"]),
        ("jit_ops.cpp", ["gemv builders for BF16,", "Q8_0 and Q4_S weights;", "the shape is baked in;",
                         "4 FMA accumulators"]),
        ("emitter_*.cpp", ["x64: REX, ModR/M, SIB", "AVX2 and FMA: VEX", "Win64 ABI + SysV shuffle",
                           "vzeroupper before ret"]),
        ("jit_buffer.cpp", ["W^X code pages: written", "read/write, then sealed", "read/execute for good"]),
    ]):
        c.card(b, title, lines, KERNEL, avx2=True)
    for a, b, label in zip(ker, ker[1:], ("gemv", "miss", "emit", "bytes")):
        c.arrow([(a.right + 5, a.cy + 10), (b.x - 5, b.cy + 10)])
        c.tag((a.right + b.x) / 2, a.cy - 14, label, KERNEL.tint)
    trunk = 854  # model.cpp (load and forward) -> nn_ops.cpp, in the gap between the bands
    c.line([(low[1].cx, low[1].bottom + 1), (low[1].cx, trunk)])
    c.arrow([(low[2].cx, low[2].bottom + 1), (low[2].cx, trunk), (ker[0].cx, trunk), (ker[0].cx, ker[0].y - 1)])

    # ---- the OS layer -------------------------------------------------------------
    c.band(Box(left, 1077, x1 - left, 96), OS, [("OS", "bold", 27, OS.strong), ("sys.h", "mono", 16, TEXT)])
    oss = [Box(x, 1091, w, 68) for x, w in grid(x0, x1, 2, 16)]
    c.tab_card(oss[0], "sys_win.cpp", ["VirtualAlloc / VirtualProtect · file mapping · CPUID",
                                       "console code pages · UTF-8 argv · WinHTTP · MoveFileEx"], OS)
    c.tab_card(oss[1], "sys_linux.cpp", ["mmap / mprotect · file mapping · CPUID · SIGINT",
                                         "terminal · curl or wget downloads · rename"], OS)
    c.arrow([(ker[4].cx, ker[4].bottom + 1), (ker[4].cx, oss[1].y - 1)])
    channel = (x1 + side.x) / 2  # the app's console and downloads go straight to sys.h
    c.arrow([(app[5].right + 1, app[5].cy), (channel, app[5].cy), (channel, oss[1].cy),
             (oss[1].right + 1, oss[1].cy)])

    # ---- sidebar ------------------------------------------------------------------
    weights = Box(side.x, 150, side.w, 268)
    y = c.panel(weights, "Weights in memory")
    for fmt, size, flag in (("Q8_0", "1.0625 bytes per weight", "default"),
                            ("Q4_S", "0.625 bytes per weight", "--q4"),
                            ("BF16", "2 bytes per weight", "--bf16")):
        c.text(weights.x + 22, y, fmt, "monobold", 19, KERNEL.strong)
        c.text(weights.x + 100, y, size, "regular", 18, TEXT)
        c.text(weights.right - 22, y + 2, flag, "mono", 16, MUTED, anchor="ra")
        y += 31
    c.line([(weights.x + 22, y + 4), (weights.right - 22, y + 4)], RULE, 1.2)
    y += 16
    for line in c.wrap("Decoded once, at load, into nnc's split layout: all the quants, then one BF16 scale "
                       "per 32 weights (Q4_S adds a BF16 bias). BF16, not FP16, so a tiny scale cannot "
                       "flush to zero.", "regular", 16.5, weights.w - 44, "weights"):
        c.text(weights.x + 22, y, line, "regular", 16.5, TEXT)
        y += 16.5 * 1.38
    if y > weights.bottom - 8:
        c.problems.append(f"weights panel: {y - weights.bottom + 8:.0f}px too much text")

    rules = Box(side.x, 434, side.w, 402)
    c.bullets(rules, c.panel(rules, "Rules that keep it fast"), [
        "Decode is bound by memory bandwidth: judge a change by the GB/s that --perf reports.",
        "Prefill passes 16 tokens over each weight, so it is compute-bound: the opposite regime.",
        "One layer_forward for decode, prefill, embeddings and drafters.",
        "Every gemv keeps 4 independent FMA accumulators.",
        "Emitted code runs vzeroupper before every ret.",
        "No computation graph: model.cpp calls the kernels directly.",
        "MTP drafting changes the speed, never the output.",
        "Weights are decoded once, at load, straight into their final layout.",
        "The per-token path allocates nothing.",
    ])

    platforms = Box(side.x, 852, side.w, 321)
    c.bullets(platforms, c.panel(platforms, "Platforms"), [
        "Windows x64 with MSVC v143+; Linux x64 with g++ 10+ or clang 12+, WSL2 included.",
        "C++20, CMake and Ninja; .\\dd.ps1 build and .\\dd.ps1 test on Windows.",
        "x86-64 with AVX2 and FMA; AVX-512 is never used.",
        "No third-party code: no JIT library, ML framework or BLAS.",
        "One app binary, nnc (nnc-d in Debug), built into exe/; libnnc is a static library.",
        "CI builds Debug and Release on both and runs nnc --test.",
    ])

    # ---- how a model runs ------------------------------------------------------------
    c.line([(left, 1188), (W - 30, 1188)], RULE, 1.5)
    c.text(left, 1198, "How a model runs", "bold", 30, INK)
    c.text(left + c.width("How a model runs", "bold", 30) + 18, 1210,
           "colours match the layers above", "regular", 19, MUTED)

    flow(c, 1252, "Load", "nnc_model_load()", "", [
        (OS, "map the file", ["sys.h: mmap or", "MapViewOfFile"]),
        (ENGINE, "parse in place", ["gguf.cpp: header, metadata", "and tensor table, no copies"]),
        (ENGINE, "check the model", ["model.cpp: architecture,", "hparams, every tensor shape"]),
        (KERNEL, "decode the weights", ["16384-element chunks on all", "threads → Q8_0, Q4_S or BF16"]),
        (ENGINE, "build the vocabulary", ["tokenizer, special tokens", "and the chat format"]),
        (API, "nnc_model", ["immutable; shared by any", "number of sessions"]),
    ])
    reply = flow(c, 1368, "Reply", "nnc_session_chat()", "", [
        (ENGINE, "chat template", ["Gemma, Gemma 4, Llama 3 or", "ChatML turns, system prompt"]),
        (ENGINE, "tokenize", ["user-defined tokens whole,", "then SentencePiece or BPE"]),
        (ENGINE, "prefill", ["16 prompt tokens per", "pass over the weights"]),
        (ENGINE, "decode", ["a token per pass, or MTP", "guesses checked in one"]),
        (ENGINE, "sample", ["top-k → top-p → min-p", "→ temperature, or argmax"]),
        (API, "stream", ["whole UTF-8 characters", "to the on_text callback"]),
    ])
    loop = reply[-1].bottom + 20
    c.arrow([(reply[5].cx, reply[5].bottom + 1), (reply[5].cx, loop), (reply[3].cx, loop),
             (reply[3].cx, reply[3].bottom + 1)])
    c.tag((reply[3].cx + reply[5].cx) / 2, loop,
          "next token, until the end of the turn, max_tokens, a cancel or a full context", PAPER, MUTED, 16)

    flow(c, 1514, "Layer", "layer_forward()", "n_tok: 1, 16 or a text", [
        (KERNEL, "norm → Q · K · V", ["JIT gemv; shared-KV", "layers reuse K and V"]),
        (KERNEL, "RoPE", ["after any q/k norms;", "NeoX or adjacent pairs"]),
        (KERNEL, "attention", ["GQA over a BF16 K/V or a", "sliding ring; or DeltaNet"]),
        (KERNEL, "out + residual", ["JIT gemv; post-norm", "(Gemma)"]),
        (KERNEL, "norm → gate · up", ["JIT gemv"]),
        (KERNEL, "GELU or SiLU", ["× up (Gemma; Llama, Qwen)"]),
        (KERNEL, "down + residual", ["JIT gemv; + 8 routed", "experts (26B A4B)"]),
        (KERNEL, "PLE", ["per-layer embeddings", "(Gemma 4)"]),
    ], gap=30)
    y = 1624
    for line in c.wrap("JIT gemv: machine code specialised to the matrix shape; large ones split their rows across "
                       "the worker pool. Decode streams every weight once per token, so memory bandwidth sets "
                       "its speed; prefill reuses each weight for 16 tokens and is compute-bound.",
                       "regular", 17, W - 60, "footnote"):
        c.text(left, y, line, "regular", 17, MUTED)
        y += 24

    c.img.convert("RGB").resize((W * OUTPUT_SCALE, H * OUTPUT_SCALE), Image.LANCZOS).save(OUT, optimize=True)
    print(f"nnc-diagram: wrote {OUT} ({W * OUTPUT_SCALE}x{H * OUTPUT_SCALE})")
    if c.problems:
        print("nnc-diagram: some text does not fit:", *c.problems, sep="\n  ")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
