#!/usr/bin/env python3
"""Make ncc-prompt.gif, the animation at the top of README.md, from a real
nnc chat.

    python3 docs/nnc-demo.py record     # on Linux or WSL: run nnc and record it
    python  docs/nnc-demo.py render     # anywhere with Pillow: write the GIF

`record` runs nnc on a pseudo-terminal, types the questions below the way a
person would, and saves everything nnc writes with the moment it arrived
(build/nnc-demo.json). `render` replays that through a small terminal
emulator and draws a frame at every change, so the GIF shows nnc's real
output at its real speed; only the shell prompt and the typing of the
command are drawn rather than recorded. Recording needs a pseudo-terminal,
because nnc streams token by token, draws progress bars and colours, and
prompts only when it talks to a terminal, and Python offers one only on Unix.

The load time in the recording is the disk's: a model read through WSL's
/mnt/c loads several times slower than on Windows itself. So record from a
native directory holding exe/nnc and models/ (the model and its mtp- drafter):

    python3 /mnt/c/path/to/nnc/docs/nnc-demo.py record --cwd ~/nnc-demo

Needs Python 3.8+; rendering also needs Pillow and the Cascadia Mono or
Consolas font (DejaVu Sans Mono elsewhere).
"""

from __future__ import annotations

import argparse
import codecs
import json
import os
import random
import re
import shlex
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RECORDING = REPO / "build" / "nnc-demo.json"
GIF = REPO / "ncc-prompt.gif"

COMMAND = "./exe/nnc -m gemma-4-e2b-q4km"
QUESTIONS = [
    "Why is the sky blue? Answer in two sentences.",
    "Then why are sunsets red?",
]
COLS, ROWS = 96, 22

# The escape sequences nnc writes: SGR (bold, faint, colour) and erase in line.
CSI = re.compile(r"\x1b\[([0-9;?]*)([@-~])")
CSI_START = re.compile(r"\x1b(\[[0-9;?]*)?$")  # one cut off by the end of a chunk


def keystroke_delay(rng: random.Random, ch: str) -> float:
    """Seconds before the next key: a quick, slightly uneven typist."""
    delay = rng.uniform(0.045, 0.11)
    if ch in " ,.?":
        delay += rng.uniform(0.04, 0.12)
    return delay


# ---- record ---------------------------------------------------------------------


def record(args: argparse.Namespace) -> int:
    import fcntl
    import pty
    import select
    import struct
    import termios

    argv = shlex.split(args.command)
    pid, fd = pty.fork()
    if pid == 0:  # the child: nnc, on the other end of the terminal
        try:
            os.chdir(os.path.expanduser(args.cwd))
            fcntl.ioctl(0, termios.TIOCSWINSZ, struct.pack("HHHH", args.rows, args.cols, 0, 0))
            env = dict(os.environ, TERM="xterm-256color")
            env.pop("NO_COLOR", None)
            os.execvpe(argv[0], argv, env)
        except OSError as e:
            print(f"nnc-demo: cannot run {argv[0]} in {args.cwd}: {e}", file=sys.stderr)
        os._exit(127)

    rng = random.Random(args.seed)
    start = time.monotonic()
    events: list[list] = []  # [seconds, text]
    seen = ""  # everything nnc wrote, without escape sequences or carriage returns
    decode = codecs.getincrementaldecoder("utf-8")("replace").decode

    def pump(seconds: float) -> bool:
        """Collect nnc's output for `seconds`. False once nnc has exited."""
        nonlocal seen
        until = time.monotonic() + seconds
        while True:
            left = until - time.monotonic()
            if left <= 0:
                return True
            ready, _, _ = select.select([fd], [], [], left)
            if not ready:
                return True
            try:
                data = os.read(fd, 65536)
            except OSError:  # EIO: nnc closed the terminal
                return False
            if not data:
                return False
            text = decode(data)
            events.append([round(time.monotonic() - start, 4), text])
            seen += CSI.sub("", text).replace("\r", "")

    def wait_for_prompt(count: int, limit: float) -> None:
        """Wait until nnc has shown its "> " prompt `count` times."""
        deadline = time.monotonic() + limit
        while seen.count("\n> ") < count:
            if time.monotonic() > deadline or not pump(0.05):
                sys.exit("nnc-demo: nnc did not get to its prompt. It wrote:\n" + seen[-3000:])

    def type_line(text: str) -> None:
        for ch in text:
            os.write(fd, ch.encode())
            pump(keystroke_delay(rng, ch))
        pump(0.35)
        os.write(fd, b"\r")

    wait_for_prompt(1, 300)
    pump(1.0)
    for n, question in enumerate(QUESTIONS, start=2):
        type_line(question)
        wait_for_prompt(n, 600)
        pump(1.2)
    end = round(time.monotonic() - start, 4)
    os.write(fd, b"\x04")  # Ctrl-D: the end of input, so nnc exits
    pump(5.0)
    os.waitpid(pid, 0)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"command": args.command, "cols": args.cols, "rows": args.rows, "end": end,
                               "events": events}, ensure_ascii=False), encoding="utf-8")
    print(f"nnc-demo: recorded {end:.1f} s of nnc in {out}")
    return 0


# ---- render ---------------------------------------------------------------------

BG = "#0c0c0c"  # Windows Terminal's Campbell scheme
STYLES = {  # SGR attribute -> (font, colour)
    0: ("regular", "#cccccc"),
    1: ("bold", "#f2f2f2"),
    2: ("regular", "#767676"),
    31: ("regular", "#e74856"),
    32: ("regular", "#16c60c"),
}
SGR = {0: 0, 1: 1, 2: 2, 22: 0, 31: 31, 32: 32, 39: 0}
SHELL_PROMPT = "\x1b[32m$\x1b[0m "

FONT_SIZE = 17  # pixels at scale 1
FONT_FILES = {
    "regular": ("CascadiaMono.ttf", "consola.ttf", "DejaVuSansMono.ttf"),
    "bold": ("CascadiaMono.ttf", "consolab.ttf", "DejaVuSansMono-Bold.ttf"),
}
FPS = 25
BLINK = 0.53  # cursor blink half-period, seconds
HOLD = 2.5  # seconds on the last screen before the GIF starts again


class Terminal:
    """Just enough of a VT100 for nnc: text, CR, LF, backspace, tab, line
    wrap and scrolling, SGR bold / faint / red / green, erase in line and
    display, cursor moves and cursor visibility."""

    def __init__(self, cols: int, rows: int) -> None:
        self.cols, self.rows = cols, rows
        self.grid = [self.blank() for _ in range(rows)]
        self.x = self.y = 0
        self.style = 0
        self.wrap_next = False  # the last column is written: the next character wraps
        self.cursor = True
        self.pending = ""

    def blank(self) -> list[tuple[str, int]]:
        return [(" ", 0)] * self.cols

    def snapshot(self) -> tuple:
        return tuple(map(tuple, self.grid)), self.x, self.y, self.cursor

    def feed(self, text: str) -> None:
        text, self.pending = self.pending + text, ""
        i = 0
        while i < len(text):
            ch = text[i]
            if ch == "\x1b":
                m = CSI.match(text, i)
                if m:
                    self.csi(m.group(1), m.group(2))
                    i = m.end()
                    continue
                if CSI_START.match(text, i):
                    self.pending = text[i:]
                    return
            elif ch == "\r":
                self.x, self.wrap_next = 0, False
            elif ch == "\n":
                self.line_feed()
            elif ch == "\b":
                self.x, self.wrap_next = max(0, self.x - 1), False
            elif ch == "\t":
                self.x = min(self.cols - 1, (self.x // 8 + 1) * 8)
            elif ch >= " " and ch != "\x7f":
                self.put(ch)
            i += 1

    def put(self, ch: str) -> None:
        if self.wrap_next:
            self.x, self.wrap_next = 0, False
            self.line_feed()
        row = self.grid[self.y] = list(self.grid[self.y])
        row[self.x] = (ch, self.style)
        if self.x == self.cols - 1:
            self.wrap_next = True
        else:
            self.x += 1

    def line_feed(self) -> None:
        if self.y == self.rows - 1:
            self.grid = self.grid[1:] + [self.blank()]
        else:
            self.y += 1

    def csi(self, params: str, final: str) -> None:
        private = params.startswith("?")
        nums = [int(p) if p.isdigit() else 0 for p in params.lstrip("?").split(";")]
        n = nums[0]
        if final == "m":
            for p in nums:
                self.style = SGR.get(p, self.style)
        elif final == "K":
            row = self.grid[self.y] = list(self.grid[self.y])
            lo, hi = {0: (self.x, self.cols), 1: (0, self.x + 1)}.get(n, (0, self.cols))
            row[lo:hi] = [(" ", 0)] * (hi - lo)
        elif final == "J" and n == 2:
            self.grid = [self.blank() for _ in range(self.rows)]
        elif final in "ABCD":
            dy, dx = {"A": (-1, 0), "B": (1, 0), "C": (0, 1), "D": (0, -1)}[final]
            step = max(n, 1)
            self.y = min(max(self.y + dy * step, 0), self.rows - 1)
            self.x = min(max(self.x + dx * step, 0), self.cols - 1)
            self.wrap_next = False
        elif final == "G":
            self.x, self.wrap_next = min(max(n, 1), self.cols) - 1, False
        elif final in "Hf":
            r, c = (nums + [1, 1])[:2]
            self.y, self.x, self.wrap_next = min(max(r, 1), self.rows) - 1, min(max(c, 1), self.cols) - 1, False
        elif final in "hl" and private and n == 25:
            self.cursor = final == "h"


def render(args: argparse.Namespace) -> int:
    from PIL import Image, ImageDraw, ImageFont

    def load_font(style: str, size: int) -> ImageFont.FreeTypeFont:
        for name in FONT_FILES[style]:
            try:
                font = ImageFont.truetype(name, size)
            except OSError:
                continue
            if name.startswith("Cascadia"):  # a variable font: pick the weight
                try:
                    font.set_variation_by_name("Bold" if style == "bold" else "Regular")
                except (OSError, ValueError):
                    if style == "bold":
                        continue
            return font
        sys.exit(f"nnc-demo: needs one of these fonts: {', '.join(FONT_FILES[style])}")

    rec = json.loads(Path(args.input).read_text(encoding="utf-8"))
    cols, rows = rec["cols"], rec["rows"]
    scale = args.scale
    fonts = {style: load_font(style, round(FONT_SIZE * scale)) for style in FONT_FILES}
    cell_w = fonts["regular"].getlength("M")
    ascent, descent = fonts["regular"].getmetrics()
    line_h = round((ascent + descent) * 1.08)
    text_dy = (line_h - ascent - descent) // 2
    pad_x, pad_y = round(18 * scale), round(14 * scale)
    width, height = round(2 * pad_x + cols * cell_w), 2 * pad_y + rows * line_h

    # What happens when, in seconds: the shell prompt, the command typed,
    # then everything nnc wrote, as recorded.
    rng = random.Random(args.seed)
    steps = [(0.0, SHELL_PROMPT)]
    t = 0.8
    for ch in rec["command"]:
        steps.append((t, ch))
        t += keystroke_delay(rng, ch)
    t += 0.3
    steps.append((t, "\r\n"))
    offset = t + 0.1
    steps += [(offset + at, text) for at, text in rec["events"] if at <= rec["end"]]
    total = offset + rec["end"] + HOLD

    # The screen at every tick; a run of identical ticks is one frame.
    term = Terminal(cols, rows)
    frames: list[list] = []  # [screen, seconds]
    i, tick = 0, 0
    while tick / FPS < total:
        while i < len(steps) and steps[i][0] <= tick / FPS:
            term.feed(steps[i][1])
            i += 1
        screen = term.snapshot()
        if frames and frames[-1][0] == screen:
            frames[-1][1] += 1 / FPS
        else:
            frames.append([screen, 1 / FPS])
        tick += 1

    # The cursor blinks while nothing happens, as a terminal's does.
    shown: list[tuple[tuple, bool, float]] = []
    for screen, seconds in frames:
        on = True
        while seconds > 1e-6:
            part = seconds if seconds <= BLINK * 1.2 else BLINK
            shown.append((screen, on, part))
            on, seconds = not on, seconds - part

    # One palette for every frame: the background, and 15 steps from it to
    # each text colour for the anti-aliased edges.
    def rgb(c: str) -> tuple[int, int, int]:
        return int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16)

    bg = rgb(BG)
    entries = [bg]
    for _, colour in STYLES.values():
        fg = rgb(colour)
        entries += [tuple(round(b + (f - b) * k / 15) for b, f in zip(bg, fg)) for k in range(1, 16)]
    palette = Image.new("P", (1, 1))
    palette.putpalette([v for e in entries for v in e] + [0] * (768 - 3 * len(entries)))

    cursor_w = max(2, round(2 * scale))
    images = []
    for (grid, cx, cy, visible), on, _ in shown:
        img = Image.new("RGB", (width, height), BG)
        draw = ImageDraw.Draw(img)
        for r, row in enumerate(grid):
            y = pad_y + r * line_h + text_dy
            c = 0
            while c < cols:
                style = row[c][1]
                end = c
                while end < cols and row[end][1] == style:
                    end += 1
                text = "".join(ch for ch, _ in row[c:end]).rstrip()
                if text.strip():
                    font, colour = STYLES[style]
                    draw.text((pad_x + c * cell_w, y), text, font=fonts[font], fill=colour)
                c = end
        if visible and on:
            x = round(pad_x + cx * cell_w)
            y = pad_y + cy * line_h + text_dy
            draw.rectangle([x, y, x + cursor_w - 1, y + ascent + descent - 1], fill=STYLES[0][1])
        images.append(img.quantize(palette=palette, dither=Image.Dither.NONE))

    # GIF delays are in hundredths of a second; round the running total so
    # the rounding does not add up.
    durations, clock, last = [], 0.0, 0
    for _, _, seconds in shown:
        clock += seconds
        cs = round(clock * 100)
        durations.append(10 * (cs - last))
        last = cs

    out = Path(args.out)
    images[0].save(out, save_all=True, append_images=images[1:], duration=durations, loop=0, optimize=False)
    print(f"nnc-demo: wrote {out} ({width}x{height}, {len(images)} frames, {total:.1f} s, "
          f"{out.stat().st_size / 1e6:.1f} MB)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="step", required=True)

    rec = sub.add_parser("record", help="run nnc on a pseudo-terminal and record it (Linux or WSL)")
    rec.add_argument("--cwd", default=str(REPO), help="where to run nnc (default: the repository)")
    rec.add_argument("--command", default=COMMAND, help=f"the command to run (default: {COMMAND})")
    rec.add_argument("--cols", type=int, default=COLS)
    rec.add_argument("--rows", type=int, default=ROWS)
    rec.add_argument("--seed", type=int, default=1, help="for the typing rhythm")
    rec.add_argument("--out", default=str(RECORDING))

    ren = sub.add_parser("render", help="draw the recording as an animated GIF (needs Pillow)")
    ren.add_argument("--in", dest="input", default=str(RECORDING))
    ren.add_argument("--out", default=str(GIF))
    ren.add_argument("--scale", type=float, default=2.0, help="pixels per unit; 2 keeps text sharp when zoomed")
    ren.add_argument("--seed", type=int, default=2, help="for the typing rhythm of the command")

    args = parser.parse_args()
    return record(args) if args.step == "record" else render(args)


if __name__ == "__main__":
    sys.exit(main())
