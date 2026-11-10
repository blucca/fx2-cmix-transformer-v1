#!/usr/bin/env python3
"""Draw the complete q7 online/native figure from the public JSON (stdlib only)."""
import argparse
import html
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / 'results/q7-teacher-only-terminal.json'
OUTPUT = HERE / 'q7-teacher-only-terminal.svg'
PAPER = '#ffffff'
INK = '#303e47'
GRID = '#dce5ed'
BLUE = '#245f90'
LONG = '#6192b4'
SHORT = '#8ca4b7'
RED = '#ad5143'


def draw(data):
    """Return a deterministic SVG; each plotted coordinate comes from data."""
    assert data['coverage']['complete32'] and len(data['online_epochs']) == 32
    assert [r['epoch'] for r in data['exports']] == list(range(4, 33, 4))
    assert data['selected_epoch'] == 28 and data['final_evaluated_epoch'] == 32
    out = ['<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="660" viewBox="0 0 1120 660" role="img" aria-labelledby="title desc">',
           '<title id="title">Complete full12 seven-symbol teacher-only trajectory</title>',
           '<desc id="desc">Left: all 32 target-weighted hard-online CE, teacher KL and objective means. Right: separate q7 e0 control and eight frozen native exports, expressed as raw NLL residuals to W0 for full held, contained long-three, remaining short and independent giant. Epoch 28 is selected; epoch 32 is final. The table gives each control and frozen export its own raw native NLL column.</desc>',
           '<style>text{font-family:"Source Sans 3","DejaVu Sans",sans-serif;fill:'+INK+';font-size:12px}.heading{font-family:"STIX Two Text","DejaVu Serif",serif}.num{font-variant-numeric:tabular-nums}.muted{fill:#566773}</style>',
           f'<rect width="1120" height="660" fill="{PAPER}"/>']

    def text(x, y, value, size=12, anchor='start', cls='', color=None, extra=''):
        style = f'font-size:{size}px' + (f';fill:{color}' if color else '')
        out.append(f'<text x="{x:.2f}" y="{y:.2f}" style="{style}" text-anchor="{anchor}" class="{cls}" {extra}>{html.escape(str(value))}</text>')

    def line(x1, y1, x2, y2, color=GRID, width=1, dash=''):
        dashed = f' stroke-dasharray="{dash}"' if dash else ''
        out.append(f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="{color}" stroke-width="{width}"{dashed}/>')

    def polyline(points, color, width=2, dash=''):
        dashed = f' stroke-dasharray="{dash}"' if dash else ''
        coords = ' '.join(f'{x:.3f},{y:.3f}' for x, y in points)
        out.append(f'<polyline fill="none" stroke="{color}" stroke-width="{width}" stroke-linejoin="round" points="{coords}"{dashed}/>')

    def marker(x, y, color, kind='circle', radius=3.4):
        if kind == 'selected':
            pts = [(x,y-5.2),(x+5.2,y),(x,y+5.2),(x-5.2,y)]
            out.append('<polygon points="'+' '.join(f'{a:.3f},{b:.3f}' for a,b in pts)+f'" fill="{color}" stroke="{PAPER}" stroke-width="1"/>')
        elif kind == 'final':
            out.append(f'<rect x="{x-4:.3f}" y="{y-4:.3f}" width="8" height="8" fill="{PAPER}" stroke="{color}" stroke-width="1.7"/>')
        else:
            fill = PAPER if kind == 'control' else color
            out.append(f'<circle cx="{x:.3f}" cy="{y:.3f}" r="{radius}" fill="{fill}" stroke="{color}" stroke-width="1.5"/>')

    def key(x, y, label, color, dash='', width=2):
        line(x, y-4, x+25, y-4, color, width, dash)
        text(x+33, y, label, 12)

    text(66, 37, 'Seven-symbol full12: complete learning and native error', 25, cls='heading')
    text(66, 62, 'Fixed-scale hard STE; 2KL + 2H/586459020. Native errors use the retained W0 reference.', 13, cls='muted')
    text(76, 101, '(a) Online training', 19, cls='heading')
    text(76, 124, 'Target-weighted means; weights change within each epoch.', 12, cls='muted')
    text(626, 101, '(b) Frozen native loss residual', 19, cls='heading')
    text(626, 124, 'Fixed held inputs; higher residual means larger prediction loss.', 12, cls='muted')
    key(76, 151, 'Objective', BLUE, width=2.6)
    key(213, 151, 'Teacher KL', LONG, '5 3')
    key(369, 151, 'Detached CE', INK, width=1.8)
    key(626, 149, 'Full held', BLUE, width=2.6)
    key(807, 149, 'Long3 (contained)', LONG, '5 3', 1.8)
    key(626, 169, 'Short', SHORT, '2 3', 1.7)
    key(807, 169, 'Giant (independent)', RED, width=2.3)

    top, bottom = 193, 424
    left, lright, right, rright = 76, 520, 626, 1063
    lx = lambda epoch: left + (epoch-1)/31*(lright-left)
    ly = lambda value: bottom - value/1.8*(bottom-top)
    rx = lambda epoch: right + epoch/32*(rright-right)
    ry = lambda value: bottom - value/1.35*(bottom-top)

    for value in (0, .4, .8, 1.2, 1.6):
        y = ly(value)
        line(left, y, lright, y)
        text(left-12, y+4, f'{value:.1f}', 11, 'end', 'num')
    for value in (0, .4, .8, 1.2):
        y = ry(value)
        line(right, y, rright, y, INK if value == 0 else GRID)
        text(right-12, y+4, f'{value:.1f}', 11, 'end', 'num')
    line(left, top, left, bottom, INK)
    line(left, bottom, lright, bottom, INK)
    line(right, top, right, bottom, INK)
    text(31, (top+bottom)/2, 'Online mean / objective', 12, 'middle',
         extra=f'transform="rotate(-90 31 {(top+bottom)/2})"')
    text(574, (top+bottom)/2, 'Δ raw NLL vs W0 (nat / target)', 12, 'middle',
         extra=f'transform="rotate(-90 574 {(top+bottom)/2})"')
    for epoch in (1, 8, 16, 24, 32):
        x = lx(epoch)
        line(x, bottom, x, bottom+5, INK)
        text(x, bottom+22, epoch, 11, 'middle', 'num')
    for epoch in (0, 4, 8, 12, 16, 20, 24, 28, 32):
        x = rx(epoch)
        line(x, bottom, x, bottom+5, INK)
        text(x, bottom+22, epoch, 11, 'middle', 'num')
    text((left+lright)/2, 469, 'Completed hard epoch', 12, 'middle')
    text((right+rright)/2, 469, 'Frozen hard export epoch (e0 is the separate control)', 12, 'middle')
    for epoch in (0, data['selected_epoch'], data['final_evaluated_epoch']):
        line(rx(epoch), top, rx(epoch), bottom, GRID, 1, '3 3')
    text(right+4, 186, 'e0 control', 10, cls='muted')
    text(rx(28)-4, 186, 'e28 selected', 10, 'end', 'muted')
    text(rright, 186, 'e32 final', 10, 'end', 'muted')
    text(right+9, bottom-9, 'W0 residual = 0', 11, cls='muted')

    for name, color, width, dash in (('objective', BLUE, 2.6, ''),
                                    ('teacher_kl_mean', LONG, 2, '5 3'),
                                    ('ce_mean', INK, 1.8, '')):
        points = [(lx(r['epoch']), ly(r[name])) for r in data['online_epochs']]
        polyline(points, color, width, dash)
        marker(*points[-1], color, 'final', 3)
    for scope, color, width, dash in (('held_short', SHORT, 1.7, '2 3'),
                                    ('held_long', LONG, 1.8, '5 3'),
                                    ('full_held', BLUE, 2.6, ''),
                                    ('giant', RED, 2.3, '')):
        initial = data['initial_student']['native'][scope]['delta_nll_w0']
        rows = [(0, initial)] + [(r['epoch'], r['native'][scope]['delta_nll_w0']) for r in data['exports']]
        points = [(rx(epoch), ry(value)) for epoch,value in rows]
        polyline(points, color, width, dash)
        for (epoch, _), (x, y) in zip(rows, points):
            kind = 'control' if epoch == 0 else 'selected' if epoch == data['selected_epoch'] else 'final' if epoch == data['final_evaluated_epoch'] else 'circle'
            marker(x, y, color, kind, 3.2 if scope in ('held_long','held_short') else 3.6)

    line(66, 490, 1063, 490, GRID)
    text(76, 516, 'Frozen native readout', 14, cls='heading')
    text(76, 536, 'Raw NLL / target', 11, cls='muted')
    selected = next(r for r in data['exports'] if r['selected'])
    final = data['exports'][-1]
    table = [(374, 'e0 control', data['initial_student']),
             (673, f"e{selected['epoch']} selected", selected),
             (972, f"e{final['epoch']} final", final)]
    for x, label, row in table:
        text(x, 516, label, 14, 'middle', 'heading')
        text(x, 536, f"W {row['weight_bytes']:,} B", 11, 'middle', 'num muted')
    for index, (scope, label, color) in enumerate((('full_held', 'Full held', BLUE),
                                                ('held_long', 'Long3 (contained)', LONG),
                                                ('held_short', 'Short', SHORT),
                                                ('giant', 'Giant (independent)', RED))):
        y = 560 + 21*index
        text(76, y, label, 12, color=color)
        for x, _, row in table:
            text(x, y, f"{row['native'][scope]['raw_nll']:.9f}", 13, 'middle', 'num', color=color)
    text(76, 647, 'All 32 hard-online epochs; 94,121,824 training targets; 2,527 updates. Lines join the measured native points.', 11, cls='muted')
    out.append('</svg>')
    return '\n'.join(out) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=DATA)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    data = json.loads(args.input.read_text())
    args.output.write_text(draw(data))
    print(f'Wrote {args.output.name}: 32 hard-online epochs, e0 control and 8 frozen native exports.')


if __name__ == '__main__':
    main()
