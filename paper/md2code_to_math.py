#!/usr/bin/env python3
"""Convert the DLA paper/docs code-styled formulas into real LaTeX math.

Both `paper/DLA_paper_draft.md` and `docs/mechanism_chain.md` historically wrote every
equation as a backticked code span or fenced code block, so pandoc/Chrome typeset them
as monospace text with literal `_` and `^{}` -- readable in the .md, but garbled in the
PDF and on GitHub. This script rewrites *formulas* as math while leaving genuine *code*
as code:

  converted     -> display math ($$..$$) for equation blocks, inline math ($..$) for
                   expressions and bare symbols (W_fast, Q, P, phi, gamma, ...)
  kept as code  -> experiment variant names (direct/shufwrite/nocons/nosleep/qonly/full),
                   code identifiers (success, load_body, tempos, store[key][...]),
                   file paths, and non-equation blocks (stats summaries, ASCII diagrams)

Idempotent-ish: run it on a fresh checkout of the pre-conversion file. `paper/md2pdf.py`
must be invoked with `--mathml` so Chrome renders the math natively (no LaTeX, no
network, no JS).

Usage:
    python3 paper/md2code_to_math.py --profile paper            # dry run
    python3 paper/md2code_to_math.py --profile paper --apply
    python3 paper/md2code_to_math.py --profile chain --apply
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FENCE = re.compile(r"```\n(.*?)```", re.S)
SPAN = re.compile(r"`([^`\n]+)`")


def aligned(*lines: str) -> str:
    """One display block whose rows are aligned on their first relation."""
    return "\\begin{aligned}\n" + "\\\\\n".join(lines) + "\n\\end{aligned}"


# =====================================================================================
# Profile: paper
# =====================================================================================
DISPLAY_PAPER: list[tuple[str, list[str]]] = [
    (
        "W_eff  = W_slow + softplus(P) * W_fast",
        [r"W_{\mathrm{eff}} = W_{\mathrm{slow}} + \mathrm{softplus}(P)\odot W_{\mathrm{fast}}"],
    ),
    (
        "dw     = -eta_fast * softplus(P) * adam - fast_decay * W_fast\nW_fast += dw",
        [r"\mathrm{d}w_t = -\eta_{\mathrm{fast}}\,\mathrm{softplus}(P_t)\odot \hat{a}_t"
         r" - \lambda_{\mathrm{fast}}\,W_{\mathrm{fast},t},"
         r"\qquad W_{\mathrm{fast},t+1} = W_{\mathrm{fast},t} + \mathrm{d}w_t"],
    ),
    (
        "W_slow += beta * Q + gamma * W_fast     # designed (Q) + unselective direct copy\n"
        "W_fast *= 0.5 ; Q *= 0.7 ; reset moments",
        [r"W_{\mathrm{slow}} \leftarrow W_{\mathrm{slow}} + \beta Q + \gamma W_{\mathrm{fast}}",
         r"W_{\mathrm{fast}} \leftarrow \delta\,W_{\mathrm{fast}},\qquad Q \leftarrow \rho\,Q,"
         r"\qquad m,v \leftarrow 0"],
    ),
    (
        "W_fast,t+1 = (1\u2212\u03bb) W_fast,t \u2212 \u03b7 \u03c6(P_t) \u2299 \u00e2_t\n"
        "\u21d2 W_fast,T = \u2212\u03b7 \u03a3_{k<T} (1\u2212\u03bb)^{T\u22121\u2212k} \u03c6(P_k) \u2299 \u00e2_k"
        " = \u2212\u03b7 \u03a3_k \u03ba(T,k) \u00e2_k^\u03c6 ,  \u03ba(T,k)=(1\u2212\u03bb)^{T\u22121\u2212k}",
        [r"W_{\mathrm{fast},t+1} = (1-\lambda)\,W_{\mathrm{fast},t}"
         r" - \eta\,\varphi(P_t)\odot \hat{a}_t",
         r"\Rightarrow\quad W_{\mathrm{fast},T} = -\eta \sum_{k=0}^{T-1}(1-\lambda)^{T-1-k}"
         r"\,\varphi(P_k)\odot \hat{a}_k = -\eta \sum_{k} \kappa(T,k)\,\hat{a}^{\varphi}_k,"
         r"\qquad \kappa(T,k)=(1-\lambda)^{T-1-k}"],
    ),
    (
        "\u0394_hist = W_fast^HE \u2212 W_fast^EH = \u2212\u03b7 \u03a3_k \u03ba(T,k)"
        "[\u00e2_k^\u03c6(HE) \u2212 \u00e2_k^\u03c6(EH)]",
        [r"\Delta_{\mathrm{hist}} = W_{\mathrm{fast}}^{\mathrm{HE}}"
         r" - W_{\mathrm{fast}}^{\mathrm{EH}} = -\eta \sum_{k}\kappa(T,k)"
         r"\left[\hat{a}^{\varphi}_k(\mathrm{HE}) - \hat{a}^{\varphi}_k(\mathrm{EH})\right]"],
    ),
    (
        "W_slow \u2190 W_slow + \u03b2 Q + \u03b3 W_fast ,   W_fast \u2190 \u03b4 W_fast ,"
        "  Q \u2190 \u03c1 Q ,  m,v \u2190 0",
        [r"W_{\mathrm{slow}} \leftarrow W_{\mathrm{slow}} + \beta Q + \gamma W_{\mathrm{fast}},"
         r"\qquad W_{\mathrm{fast}} \leftarrow \delta W_{\mathrm{fast}},\qquad "
         r"Q \leftarrow \rho Q,\qquad m,v \leftarrow 0"],
    ),
    (
        "selective  \u21d4  \u03c4_shuf := E[G(\u03a6_shuf)] \u2212 E[G(\u03a6_direct)] < 0"
        "   at fixed write energy",
        [r"\text{selective} \iff \tau_{\mathrm{shuf}} := "
         r"\mathbb{E}\!\left[G(\Phi_{\mathrm{shuf}})\right] - "
         r"\mathbb{E}\!\left[G(\Phi_{\mathrm{direct}})\right] < 0"
         r"\qquad\text{at fixed write energy}"],
    ),
    (
        "\u0394L_D \u2248 \u27e8 \u2207_{W_slow} L_D , A \u27e9 + O(\u2016A\u2016\u00b2)",
        [r"\Delta L_D \approx \langle \nabla_{W_{\mathrm{slow}}} L_D,\, A\rangle"
         r" + O(\|A\|^2)"],
    ),
]

INLINE_PAPER: dict[str, str] = {
    "W_eff = W_slow + softplus(P)\u00b7W_fast":
        r"$W_{\mathrm{eff}} = W_{\mathrm{slow}} + \mathrm{softplus}(P)\odot W_{\mathrm{fast}}$",
    "g = dL/dW_eff": r"$g = \mathrm{d}L/\mathrm{d}W_{\mathrm{eff}}$",
    "g_t = \u2207_{W_eff} L_t": r"$g_t = \nabla_{W_{\mathrm{eff}}} L_t$",
    "\u03c6(P)=softplus(P)": r"$\varphi(P)=\mathrm{softplus}(P)$",
    "\u00e2_t = m\u0302_t/(\u221av\u0302_t+\u03b5)":
        r"$\hat{a}_t = \hat{m}_t/(\sqrt{\hat{v}_t}+\epsilon)$",
    "1/\u03bb = 50": r"$1/\lambda = 50$",
    "\u0394L_D \u2248 \u27e8\u2207L_D, A\u27e9":
        r"$\Delta L_D \approx \langle \nabla L_D, A\rangle$",
    "A \u221d W_fast": r"$A \propto W_{\mathrm{fast}}$",
    "A = \u03b2Q + \u03b3W_fast": r"$A = \beta Q + \gamma W_{\mathrm{fast}}$",
    "A = \u03b3W_fast": r"$A = \gamma W_{\mathrm{fast}}$",
    "A_i = \u03b3 W_fast,i": r"$A_i = \gamma W_{\mathrm{fast},i}$",
    "A' = \u03a0_\u03c0 A": r"$A' = \Pi_{\pi} A$",
    "E\u27e8\u2207L_D, \u03a0A\u27e9 \u2248 0":
        r"$\mathbb{E}\langle \nabla L_D, \Pi A\rangle \approx 0$",
    "Q \u2190 (1\u2212\u03b1)Q + \u03b1\u00b7(dw\u00b7success)":
        r"$Q \leftarrow (1-\alpha)Q + \alpha\,(dw\cdot\mathrm{success})$",
    "cos(W_fast^EH, W_fast^HE)":
        r"$\cos(W_{\mathrm{fast}}^{\mathrm{EH}}, W_{\mathrm{fast}}^{\mathrm{HE}})$",
    "cos(g0, \u0394W_fast)": r"$\cos(g_0, \Delta W_{\mathrm{fast}})$",
    "\u03c1 = cos(g_0, \u0394_hist)": r"$\rho = \cos(g_0, \Delta_{\mathrm{hist}})$",
    "cos(g_0, \u0394_hist)": r"$\cos(g_0, \Delta_{\mathrm{hist}})$",
    "\u2016A'\u2016 = \u2016A\u2016": r"$\|A'\| = \|A\|$",
    "\u2016Q\u2016 \u2248 0.002": r"$\|Q\| \approx 0.002$",
    "\u2016W_fast\u2016 \u2248 3.6": r"$\|W_{\mathrm{fast}}\| \approx 3.6$",
    "\u2016\u0394_hist\u2016": r"$\|\Delta_{\mathrm{hist}}\|$",
    "|\u03c1| \u2248 0.02": r"$|\rho| \approx 0.02$",
    "\u03c4_shuf \u2248 \u03c4_unif \u2248 \u03c4_top \u2248 0":
        r"$\tau_{\mathrm{shuf}} \approx \tau_{\mathrm{unif}} \approx \tau_{\mathrm{top}} \approx 0$",
    "\u03c4_shuf": r"$\tau_{\mathrm{shuf}}$",
    "gamma\u00b7W_fast_i": r"$\gamma W_{\mathrm{fast},i}$",
    "gamma\u00b7W_fast": r"$\gamma W_{\mathrm{fast}}$",
    "\u03b3\u00b7W_fast": r"$\gamma W_{\mathrm{fast}}$",
    "\u03b3_scale = 0.05": r"$\gamma_{\mathrm{scale}} = 0.05$",
    "\u03b3_scale = 1": r"$\gamma_{\mathrm{scale}} = 1$",
    "gamma": r"$\gamma$",
    "W_fast^EH": r"$W_{\mathrm{fast}}^{\mathrm{EH}}$",
    "W_fast^HE": r"$W_{\mathrm{fast}}^{\mathrm{HE}}$",
    "\u0394W_fast": r"$\Delta W_{\mathrm{fast}}$",
    "\u0394_hist": r"$\Delta_{\mathrm{hist}}$",
    "W_fast": r"$W_{\mathrm{fast}}$",
    "W_slow": r"$W_{\mathrm{slow}}$",
    "W_eff": r"$W_{\mathrm{eff}}$",
    "\u03b3_scale": r"$\gamma_{\mathrm{scale}}$",
    "\u03a6": r"$\Phi$",
    "\u03c6": r"$\varphi$",
    "\u03b3": r"$\gamma$",
    "m,v": r"$m,v$",
    "Q": r"$Q$", "P": r"$P$", "A": r"$A$", "G": r"$G$", "L_D": r"$L_D$",
}

# =====================================================================================
# Profile: chain  (docs/mechanism_chain.md)
# =====================================================================================
DISPLAY_CHAIN: list[tuple[str, list[str]]] = [
    ("g_t        = \u2207_{W_eff} L_t",
     [aligned(r"g_t &= \nabla_{W_{\mathrm{eff}}} L_t",
              r"m_t &= \beta_1 m_{t-1} + (1-\beta_1) g_t",
              r"v_t &= \beta_2 v_{t-1} + (1-\beta_2) g_t \odot g_t",
              r"\hat{a}_t &= \hat{m}_t/(\sqrt{\hat{v}_t}+\epsilon)"
              r"\qquad(\hat{m},\hat{v}\ \text{bias-corrected})",
              r"W_{\mathrm{fast},t+1} &= (1-\lambda) W_{\mathrm{fast},t}"
              r" - \eta\,\varphi(P_t)\odot \hat{a}_t")]),
    ("W_fast,T = \u2212\u03b7 \u03a3_{k=0}^{T-1}",
     [aligned(r"W_{\mathrm{fast},T} &= -\eta \sum_{k=0}^{T-1}(1-\lambda)^{T-1-k}"
              r"\,\varphi(P_k)\odot \hat{a}_k",
              r"&= -\eta \sum_{k} \kappa(T,k)\,\hat{a}^{\varphi}_k,"
              r"\qquad \kappa(T,k) = (1-\lambda)^{T-1-k}")]),
    ("\u0394_hist = W_fast^HE \u2212 W_fast^EH",
     [r"\Delta_{\mathrm{hist}} = W_{\mathrm{fast}}^{\mathrm{HE}}"
      r" - W_{\mathrm{fast}}^{\mathrm{EH}} = -\eta \sum_k \kappa(T,k)"
      r"\left[\hat{a}^{\varphi}_k(\mathrm{HE}) - \hat{a}^{\varphi}_k(\mathrm{EH})\right]"]),
    ("W_slow \u2190 W_slow + \u03b2 Q + \u03b3 W_fast",
     [r"W_{\mathrm{slow}} \leftarrow W_{\mathrm{slow}} + \beta Q + \gamma W_{\mathrm{fast}}"
      r"\qquad\text{(Q-pathway} + \text{DIRECT pathway)}",
      r"W_{\mathrm{fast}} \leftarrow \delta\,W_{\mathrm{fast}};\qquad Q \leftarrow \rho\,Q;"
      r"\qquad m,v \leftarrow 0"]),
    ("\u03c4_shuf < 0  with",
     [r"\tau_{\mathrm{shuf}} < 0 \quad\text{with}\quad \|A'\| = \|A\|"
      r"\quad\Rightarrow\quad \text{the coordinate placement of the write is"
      r" functionally necessary}"]),
    ("\u0394L_D \u2248 \u27e8 \u2207_{W_slow} L_D",
     [r"\Delta L_D \approx \langle \nabla_{W_{\mathrm{slow}}} L_D,\, A\rangle"
      r" + O(\|A\|^2)"]),
]

INLINE_CHAIN: dict[str, str] = {
    "W_slow \u2208 R^d": r"$W_{\mathrm{slow}} \in \mathbb{R}^d$",
    "W_fast \u2208 R^d": r"$W_{\mathrm{fast}} \in \mathbb{R}^d$",
    "P \u2208 R^d": r"$P \in \mathbb{R}^d$",
    "Q \u2208 R^d": r"$Q \in \mathbb{R}^d$",
    "m, v \u2208 R^d": r"$m, v \in \mathbb{R}^d$",
    "\u03c6(P) = softplus(P) > 0": r"$\varphi(P) = \mathrm{softplus}(P) > 0$",
    "g_t = \u2207_{W_eff} L_t": r"$g_t = \nabla_{W_{\mathrm{eff}}} L_t$",
    "\u03bb, \u03b7, \u03b3, \u03b2, \u03b4, \u03c1":
        r"$\lambda, \eta, \gamma, \beta, \delta, \rho$",
    "W_eff = W_slow + \u03c6(P) \u2299 W_fast":
        r"$W_{\mathrm{eff}} = W_{\mathrm{slow}} + \varphi(P) \odot W_{\mathrm{fast}}$",
    "G(\u03a6)": r"$G(\Phi)$",
    "\u03a6 = (W_slow, W_fast, P, Q, m, v)":
        r"$\Phi = (W_{\mathrm{slow}}, W_{\mathrm{fast}}, P, Q, m, v)$",
    "G = gain@40 = (PPL_pre \u2212 PPL_40)/PPL_pre":
        r"$G = \text{gain@40} = (\mathrm{PPL}_{\mathrm{pre}}"
        r" - \mathrm{PPL}_{40})/\mathrm{PPL}_{\mathrm{pre}}$",
    "\u03bb = 0.02": r"$\lambda = 0.02$",
    "\u03b7 = 6e-4": r"$\eta = 6\times10^{-4}$",
    "\u03b2 \u2248 1.0": r"$\beta \approx 1.0$",
    "\u03b3 \u2248 0.15": r"$\gamma \approx 0.15$",
    "\u03b4 = 0.5": r"$\delta = 0.5$",
    "\u03c1 = 0.7": r"$\rho = 0.7$",
    "\u03b21=0.9, \u03b22=0.95": r"$\beta_1=0.9,\ \beta_2=0.95$",
    "\u03b5=1e-8": r"$\epsilon=10^{-8}$",
    "\u03b1_q = 0.3": r"$\alpha_q = 0.3$",
    "1/\u03bb = 50": r"$1/\lambda = 50$",
    "\u2016\u0394_hist\u2016": r"$\|\Delta_{\mathrm{hist}}\|$",
    "\u2016W_fast,i \u2212 W_fast,j\u2016":
        r"$\|W_{\mathrm{fast},i} - W_{\mathrm{fast},j}\|$",
    "cos(W_fast^EH, W_fast^HE)":
        r"$\cos(W_{\mathrm{fast}}^{\mathrm{EH}}, W_{\mathrm{fast}}^{\mathrm{HE}})$",
    "|\u03c1| \u2248 0.02": r"$|\rho| \approx 0.02$",
    "|\u03c1|": r"$|\rho|$",
    "A = \u03b2 Q + \u03b3 W_fast": r"$A = \beta Q + \gamma W_{\mathrm{fast}}$",
    "A_i = \u03b3 \u00b7 W_fast,i": r"$A_i = \gamma \cdot W_{\mathrm{fast},i}$",
    "A_i = \u03b3W_fast,i": r"$A_i = \gamma W_{\mathrm{fast},i}$",
    "A = \u03b2Q": r"$A = \beta Q$",
    "A = 0": r"$A = 0$",
    "A' = \u03a0_\u03c0 A": r"$A' = \Pi_{\pi} A$",
    "A'_i = \u03b3\u2016W_fast\u2016/\u221ad":
        r"$A'_i = \gamma\|W_{\mathrm{fast}}\|/\sqrt{d}$",
    "A \u221d W_fast": r"$A \propto W_{\mathrm{fast}}$",
    "\u2016A'\u2016 = \u2016A\u2016": r"$\|A'\| = \|A\|$",
    "q_inc = dw\u00b7success": r"$q_{\mathrm{inc}} = dw\cdot\mathrm{success}$",
    "\u2016Q\u2016 \u2248 0.002": r"$\|Q\| \approx 0.002$",
    "\u2016W_fast\u2016 \u2248 3.6": r"$\|W_{\mathrm{fast}}\| \approx 3.6$",
    "|W_fast,i|": r"$|W_{\mathrm{fast},i}|$",
    "\u03a6 \u21a6 I\u00b7\u03a6": r"$\Phi \mapsto I\cdot\Phi$",
    "W_fast \u2190 W_fast^{EH}":
        r"$W_{\mathrm{fast}} \leftarrow W_{\mathrm{fast}}^{\mathrm{EH}}$",
    "E\u27e8\u2207L_D, \u03a0A\u27e9 \u2248 0":
        r"$\mathbb{E}\langle \nabla L_D, \Pi A\rangle \approx 0$",
    "\u03c4_shuf \u2248 \u03c4_unif \u2248 \u03c4_top \u2248 0":
        r"$\tau_{\mathrm{shuf}} \approx \tau_{\mathrm{unif}} \approx \tau_{\mathrm{top}} \approx 0$",
    "\u03c6(P)": r"$\varphi(P)$",
    "\u0394_hist": r"$\Delta_{\mathrm{hist}}$",
    "W_fast": r"$W_{\mathrm{fast}}$",
    "W_slow": r"$W_{\mathrm{slow}}$",
    "W_eff": r"$W_{\mathrm{eff}}$",
    "\u03a0A": r"$\Pi A$",
    "\u00e2": r"$\hat{a}$",
    "g_0": r"$g_0$",
    "\u03c1": r"$\rho$", "\u03b3": r"$\gamma$",
    "A": r"$A$", "Q": r"$Q$", "G": r"$G$", "D": r"$D$", "T": r"$T$",
    "g": r"$g$", "d": r"$d$", "i": r"$i$",
}

PROFILES = {
    "paper": (ROOT / "paper" / "DLA_paper_draft.md", DISPLAY_PAPER, INLINE_PAPER),
    "chain": (ROOT / "docs" / "mechanism_chain.md", DISPLAY_CHAIN, INLINE_CHAIN),
}


def block_math(maths: list[str]) -> str:
    """Canonical display form: $$ on its own line, blank line after each block.

    The trailing blank line matters: two consecutive display blocks separated only by a
    newline land in the same paragraph, and their block-level <math> elements then
    render overlapping on top of each other.
    """
    return "".join("$$\n" + m + "\n$$\n\n" for m in maths)


def convert(text: str, display, inline):
    unmatched: list[str] = []
    used = set()

    def fence_rep(m: re.Match) -> str:
        inner = m.group(1).rstrip("\n")
        for key, maths in display:
            if inner == key or inner.split("\n")[0].startswith(key):
                used.add(key)
                return block_math(maths)
        unmatched.append(inner.split("\n")[0][:64])
        return m.group(0)

    text = FENCE.sub(fence_rep, text)
    parts = re.split(r"(\$\$.*?\$\$)", text, flags=re.S)
    n = 0
    for i, part in enumerate(parts):
        if part.startswith("$$"):
            continue

        def span_rep(m: re.Match) -> str:
            nonlocal n
            rep = inline.get(m.group(1))
            if rep is not None:
                n += 1
                return rep
            return m.group(0)

        parts[i] = SPAN.sub(span_rep, part)
    text = "".join(parts)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text, n, unmatched, [k for k, _ in display if k not in used]


def main() -> None:
    prof = "paper"
    if "--profile" in sys.argv:
        prof = sys.argv[sys.argv.index("--profile") + 1]
    path, display, inline = PROFILES[prof]
    src = path.read_text(encoding="utf-8")
    out, n_inline, unmatched, unused = convert(src, display, inline)
    n_disp = len(re.findall(r"^\$\$$", out, re.M))
    print(f"profile={prof}  file={path.relative_to(ROOT)}")
    print(f"  display math delimiters: {n_disp}"
          f" (expect {2 * sum(len(m) for _, m in display)})")
    print(f"  inline spans converted : {n_inline}")
    print(f"  fenced blocks kept as-is: {unmatched}")
    print(f"  display keys never used : {unused}")
    leftover = [s for s in SPAN.findall(re.sub(r"\$\$.*?\$\$", "", out, flags=re.S))
                if re.search(r"[=\u2248<>+\u00d7\u00b7]|_\{|\^|\u03a3|\u27e8|\u2016|\u2207", s)]
    print(f"  math-ish spans left as code: {leftover}")
    if "--apply" in sys.argv:
        path.write_text(out, encoding="utf-8")
        print("  APPLIED")


if __name__ == "__main__":
    main()
