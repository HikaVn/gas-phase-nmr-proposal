#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
気相NMR 精度設計ツール (GUI + バッチ)
=====================================

2つの解析を1つにまとめた設計支援ツール:
  (A) 分解度バジェット   R(N) = s0(2N+1)/sqrt(w1^2+(aN)^2+(bN)^2)
        ・ν0低減でRを稼ぐ、天井 R_max=2s0/sqrt(a^2+b^2) を確認
  (B) ベイズ判定        コム(分裂)を「ただの広がり」と区別できるか
        ・K_comb を達成するのに必要なイオン数(パワー解析)
  (A)で「ある装置・N往復だと分解度Rはいくつか」を出し、
  (B)で「そのRでコムを証明するのに何イオン要るか」を出す。両者をつなぐ。

使い方:
  GUI    : python3 gasnmr_tool.py
  バッチ : python3 gasnmr_tool.py --batch config.json
  雛形生成: python3 gasnmr_tool.py --make-config config.json

依存: numpy, scipy(必須), matplotlib(GUI/図), tkinter(GUI)
コア計算は bayes_comb_reanalysis.py を import して再利用。
"""

import argparse
import json
import csv
import os
import sys
import threading
import queue
from math import sqrt

import numpy as np

# --- ベイズのコアを再利用 ---
from bayes_comb_reanalysis import (
    Priors, sample_comb, log10_bayes_factors, power_analysis,
    broadening_factor, comb_density, pascal_weights, mtotal_offsets,
)
from scipy.stats import norm


# ===========================================================================
# (A) 分解度バジェットのコア
# ===========================================================================
S0_REF = 0.03      # ν0=100 m/s 基準の 1往復あたり間隔 [m/s]
NU0_REF = 100.0


def s0_of(nu0):
    """初速 ν0 での 1往復あたり間隔 s0 ∝ 1/ν0。"""
    return S0_REF * NU0_REF / nu0


def resolution_R(N, nu0, w1, a, b):
    """分解度 R(N)。a,b は ν0=100 基準値で 1/ν0 自動スケール。"""
    sc = NU0_REF / nu0
    s0, A, B = S0_REF * sc, a * sc, b * sc
    return s0 * (2 * N + 1) / sqrt(w1 * w1 + (A * N) ** 2 + (B * N) ** 2)


def ceiling_R(a, b):
    """到達天井 R_max = 2 s0 / sqrt(a^2+b^2)。ν0 に依らず一定。"""
    return 2 * S0_REF / sqrt(a * a + b * b + 1e-18)


def n_to_resolve(nu0, w1, a, b, Rtarget=2.0, Nmax=500):
    """R=Rtarget に達する最小往復回数 N。届かなければ None。"""
    for N in range(1, Nmax + 1):
        if resolution_R(N, nu0, w1, a, b) >= Rtarget:
            return N
    return None


# ===========================================================================
# バッチ用メトリクス
# ===========================================================================
def budget_metrics(run):
    """1つの装置設定の分解度バジェット指標。"""
    nu0, w1, a, b = run["nu0"], run["w1"], run["a"], run["b"]
    Nr = run.get("N_round", 10)
    R = resolution_R(Nr, nu0, w1, a, b)
    return {
        "s0[m/s]": round(s0_of(nu0), 4),
        "R@N": round(R, 3),
        "ceiling": round(ceiling_R(a, b), 2),
        "N_to_R2": n_to_resolve(nu0, w1, a, b),
        "broadening%@R": round((broadening_factor(run["n"], R) - 1) * 100, 1),
    }


def bayes_required_ions(rng, R, n, FWHM=0.4, sizes=(100, 300, 1000, 3000),
                        trials=25, p_target=0.5):
    """そのRでコム(K_comb>100)を p_target 以上の確率で証明できる最小イオン数。
       届かなければ '>max' を返す。median log10 K_comb も返す。"""
    sigma = FWHM / 2.3548
    delta = R * FWHM
    req = None
    med_at_max = None
    for N, med, p10, p100 in power_analysis(rng, list(sizes), delta, sigma, n,
                                            trials=trials):
        med_at_max = med
        if req is None and p100 >= p_target:
            req = N
    return (req if req is not None else f">{sizes[-1]}"), round(med_at_max, 2)


def run_batch(config_path):
    cfg = json.load(open(config_path, encoding="utf-8"))
    FWHM = cfg.get("FWHM", 0.4)
    trials = cfg.get("trials", 25)
    sizes = tuple(cfg.get("sizes", [100, 300, 1000, 3000]))
    out_csv = cfg.get("output_csv", "batch_results.csv")
    rng = np.random.default_rng(cfg.get("seed", 0))

    rows = []
    for i, run in enumerate(cfg["runs"], 1):
        label = run.get("label", f"run{i}")
        print(f"[{i}/{len(cfg['runs'])}] {label} ...", flush=True)
        bm = budget_metrics(run)
        req, med = bayes_required_ions(rng, bm["R@N"], run["n"], FWHM=FWHM,
                                       sizes=sizes, trials=trials)
        row = {
            "label": label, "nu0": run["nu0"], "w1": run["w1"],
            "a": run["a"], "b": run["b"], "n": run["n"],
            "N_round": run.get("N_round", 10),
            **bm,
            "req_ions(Kcomb>100)": req,
            "median_log10_Kcomb": med,
        }
        rows.append(row)
        print(f"      R@N={bm['R@N']}, 天井={bm['ceiling']}, "
              f"必要イオン={req}", flush=True)

    fields = list(rows[0].keys())
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\n[保存] {out_csv} ({len(rows)} 行)")
    return rows


def make_config(path):
    template = {
        "output_csv": "batch_results.csv",
        "FWHM": 0.4,
        "trials": 25,
        "sizes": [100, 300, 1000, 3000],
        "seed": 0,
        "runs": [
            {"label": "現状", "nu0": 100, "w1": 0.4, "a": 0.03, "b": 0.03,
             "n": 4, "N_round": 10},
            {"label": "②④解決", "nu0": 100, "w1": 0.4, "a": 0.005,
             "b": 0.0003, "n": 4, "N_round": 10},
            {"label": "②④+ν0低減(50)", "nu0": 50, "w1": 0.4, "a": 0.005,
             "b": 0.0003, "n": 4, "N_round": 10},
            {"label": "②④+ν0低減(25)", "nu0": 25, "w1": 0.4, "a": 0.005,
             "b": 0.0003, "n": 4, "N_round": 10},
        ],
    }
    json.dump(template, open(path, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"[雛形を生成] {path}\n  編集して  python3 gasnmr_tool.py --batch {path}")


# ===========================================================================
# (C) GUI
# ===========================================================================
def launch_gui():
    import tkinter as tk
    from tkinter import ttk, messagebox
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    import matplotlib
    matplotlib.rcParams["font.size"] = 9
    # 日本語フォント(あれば)
    for fam in ["Hiragino Sans", "Yu Gothic", "Noto Sans CJK JP", "AppleGothic"]:
        try:
            matplotlib.rcParams["font.family"] = fam
            break
        except Exception:
            pass

    TEAL, CORAL, GRAY, PURPLE = "#1d9e75", "#d85a30", "#888780", "#534ab7"

    root = tk.Tk()
    root.title("気相NMR 精度設計ツール")
    root.geometry("980x680")
    nb = ttk.Notebook(root)
    nb.pack(fill="both", expand=True)
    msg_queue = queue.Queue()

    # ---------- 共通ヘルパ ----------
    def labeled_entry(parent, label, default, width=8):
        fr = ttk.Frame(parent)
        ttk.Label(fr, text=label, width=14, anchor="e").pack(side="left")
        var = tk.StringVar(value=str(default))
        ttk.Entry(fr, textvariable=var, width=width).pack(side="left", padx=4)
        fr.pack(anchor="w", pady=2)
        return var

    # =====================================================================
    # タブ1: 分解度バジェット
    # =====================================================================
    t1 = ttk.Frame(nb)
    nb.add(t1, text="  (A) 分解度バジェット  ")
    left1 = ttk.Frame(t1)
    left1.pack(side="left", fill="y", padx=12, pady=12)
    ttk.Label(left1, text="装置パラメータ", font=("", 12, "bold")).pack(anchor="w")
    v_nu0 = labeled_entry(left1, "初速 ν0 [m/s]", 100)
    v_w1 = labeled_entry(left1, "① 冷却 w1 [m/s]", 0.4)
    v_a = labeled_entry(left1, "② 浮遊電場 a", 0.03)
    v_b = labeled_entry(left1, "④ RF誤差 b", 0.03)
    v_n1 = labeled_entry(left1, "等価プロトン n", 4)

    def set_preset(a, b):
        v_a.set(str(a)); v_b.set(str(b)); draw_budget()

    pf = ttk.Frame(left1); pf.pack(anchor="w", pady=6)
    ttk.Button(pf, text="現状", width=8,
               command=lambda: set_preset(0.03, 0.03)).pack(side="left", padx=2)
    ttk.Button(pf, text="②④解決後", width=10,
               command=lambda: set_preset(0.005, 0.0003)).pack(side="left", padx=2)
    ttk.Button(left1, text="更新", command=lambda: draw_budget()).pack(anchor="w", pady=4)
    budget_result = tk.StringVar(value="")
    ttk.Label(left1, textvariable=budget_result, foreground=TEAL,
              wraplength=210, justify="left").pack(anchor="w", pady=8)

    fig1 = Figure(figsize=(6.4, 5.2), dpi=100)
    ax1 = fig1.add_subplot(111)
    cv1 = FigureCanvasTkAgg(fig1, master=t1)
    cv1.get_tk_widget().pack(side="left", fill="both", expand=True, padx=8, pady=8)

    def draw_budget():
        try:
            nu0 = float(v_nu0.get()); w1 = float(v_w1.get())
            a = float(v_a.get()); b = float(v_b.get()); n = int(float(v_n1.get()))
        except ValueError:
            return
        Ns = np.arange(1, 31)
        Rcur = [resolution_R(N, nu0, w1, a, b) for N in Ns]
        Rbase = [resolution_R(N, 100, w1, a, b) for N in Ns]
        ax1.clear()
        ax1.plot(Ns, Rcur, color=TEAL, lw=2.4, label=f"ν0={nu0:.0f} m/s")
        ax1.plot(Ns, Rbase, color=GRAY, lw=1.6, ls="--", label="基準 ν0=100")
        ax1.axhline(2.0, color=CORAL, lw=1.4, ls="--", label="分解閾値 R=2")
        ax1.set_xlabel("往復回数 N"); ax1.set_ylabel("分解度 R = 間隔/線幅")
        ax1.set_ylim(0, 6); ax1.legend(fontsize=8); ax1.grid(alpha=0.25)
        cR = ceiling_R(a, b); nh = n_to_resolve(nu0, w1, a, b)
        nh100 = n_to_resolve(100, w1, a, b)
        if nh:
            budget_result.set(f"✓ R=2 到達: N={nh} 往復\n天井 R_max≈{cR:.1f}\n"
                              f"(ν0=100なら N={nh100})\ns0={s0_of(nu0):.3f} m/s")
            ax1.set_title(f"分解可能 (N={nh}, 天井 {cR:.1f})", color=TEAL)
        else:
            budget_result.set(f"✗ 分解不可\n天井 R_max≈{cR:.2f} < 2\n"
                              f"→ まず ②④(a,b) を下げる")
            ax1.set_title(f"分解不可 (天井 {cR:.2f}<2)", color=CORAL)
        fig1.tight_layout(); cv1.draw()

    # =====================================================================
    # タブ2: ベイズ判定
    # =====================================================================
    t2 = ttk.Frame(nb)
    nb.add(t2, text="  (B) ベイズ判定 (コム vs 広がり)  ")
    left2 = ttk.Frame(t2)
    left2.pack(side="left", fill="y", padx=12, pady=12)
    ttk.Label(left2, text="データ・モデル設定", font=("", 12, "bold")).pack(anchor="w")
    v_fwhm = labeled_entry(left2, "線幅 FWHM [m/s]", 0.4)
    v_R = labeled_entry(left2, "分解度 R", 0.5)
    v_n2 = labeled_entry(left2, "等価プロトン n", 4)
    v_nion = labeled_entry(left2, "イオン数 N_ions", 3000)
    v_trials = labeled_entry(left2, "試行回数 trials", 25)
    bayes_status = tk.StringVar(value="")
    ttk.Button(left2, text="単発 K 計算 + 図",
               command=lambda: single_k()).pack(anchor="w", pady=3)
    ttk.Button(left2, text="パワー解析 (必要イオン数)",
               command=lambda: start_power()).pack(anchor="w", pady=3)
    ttk.Label(left2, textvariable=bayes_status, foreground=PURPLE,
              wraplength=220, justify="left").pack(anchor="w", pady=8)

    right2 = ttk.Frame(t2)
    right2.pack(side="left", fill="both", expand=True, padx=8, pady=8)
    # 上: 単発K(分布) / 下: パワー解析(必要イオン数) を別グラフに分離
    fig2a = Figure(figsize=(6.4, 2.7), dpi=100)
    ax2a = fig2a.add_subplot(111)
    cv2a = FigureCanvasTkAgg(fig2a, master=right2)
    cv2a.get_tk_widget().pack(side="top", fill="both", expand=True, pady=(0, 4))
    fig2b = Figure(figsize=(6.4, 2.7), dpi=100)
    ax2b = fig2b.add_subplot(111)
    ax2b.set_title("パワー解析(ボタンを押すと下に表示)", color=GRAY, fontsize=9)
    cv2b = FigureCanvasTkAgg(fig2b, master=right2)
    cv2b.get_tk_widget().pack(side="top", fill="both", expand=True, pady=(4, 0))

    def read_bayes():
        return (float(v_fwhm.get()), float(v_R.get()),
                int(float(v_n2.get())), int(float(v_nion.get())),
                int(float(v_trials.get())))

    def single_k():
        try:
            FWHM, R, n, N_ions, _ = read_bayes()
        except ValueError:
            return
        sigma = FWHM / 2.3548; delta = R * FWHM
        rng = np.random.default_rng()
        data = sample_comb(rng, N_ions, delta, sigma, n)
        pr = Priors(sigma=sigma, delta_theory=delta, n_candidates=(n,))
        kc, kk = log10_bayes_factors(data, pr)
        B = broadening_factor(n, R)
        x = np.linspace(data.min() - 0.2, data.max() + 0.2, 600)
        w = pascal_weights(n); off = mtotal_offsets(n)
        ax2a.clear()
        ax2a.hist(data, bins=60, density=True, color=CORAL, alpha=0.45,
                  label="実データ")
        ax2a.plot(x, comb_density(x, 0, delta, sigma, w, off), color=TEAL,
                  lw=2, label="M1: コム")
        ax2a.plot(x, norm.pdf(x, 0, sigma * B), color=PURPLE, lw=1.6, ls="--",
                  label="M2: ただの広がり")
        ax2a.set_xlabel("速度シフト [m/s]"); ax2a.set_ylabel("密度")
        ax2a.legend(fontsize=8)
        verdict = ("決定的" if kk > 2 else "支持" if kk > 1 else "不十分")
        ax2a.set_title(f"単発K: R={R} 広がり{(B-1)*100:.0f}%  "
                       f"log10 K_comb={kk:.1f} ({verdict})")
        bayes_status.set(f"log10 K_change(広がり)={kc:.1f}\n"
                         f"log10 K_comb(コム)={kk:.1f}\n→ {verdict}")
        fig2a.tight_layout(); cv2a.draw()

    def start_power():
        try:
            params = read_bayes()
        except ValueError:
            return
        bayes_status.set("パワー解析 計算中…")
        threading.Thread(target=_power_worker, args=(params,),
                         daemon=True).start()

    def _power_worker(params):
        FWHM, R, n, N_ions, trials = params
        sigma = FWHM / 2.3548; delta = R * FWHM
        rng = np.random.default_rng()
        sizes = [100, 300, 1000, 3000]
        rows = power_analysis(rng, sizes, delta, sigma, n, trials=trials)
        msg_queue.put(("power", rows, params))

    def _draw_power(rows, params):
        FWHM, R, n, N_ions, trials = params
        Ns = [r[0] for r in rows]
        p100 = [r[3] * 100 for r in rows]
        p10 = [r[2] * 100 for r in rows]
        ax2b.clear()
        ax2b.plot(Ns, p100, "o-", color=TEAL, label="P(K_comb>100) 決定的")
        ax2b.plot(Ns, p10, "s--", color=GRAY, label="P(K_comb>10) 支持")
        ax2b.axhline(50, color=CORAL, lw=1, ls=":")
        ax2b.set_xscale("log"); ax2b.set_xlabel("イオン数 N_ions")
        ax2b.set_ylabel("達成確率 [%]"); ax2b.set_ylim(-3, 103)
        ax2b.legend(fontsize=8); ax2b.grid(alpha=0.25)
        B = broadening_factor(n, R)
        ax2b.set_title(f"パワー解析: 必要イオン数 (R={R}, 広がり{(B-1)*100:.0f}%)")
        req = next((N for N, _, _, q in rows if q >= 0.5), None)
        bayes_status.set(f"R={R} のとき\nK_comb>100 を50%以上で達成:\n"
                         + (f"≈ {req} イオン" if req else f">{Ns[-1]} イオン(困難)"))
        fig2b.tight_layout(); cv2b.draw()

    def poll():
        try:
            while True:
                kind, *payload = msg_queue.get_nowait()
                if kind == "power":
                    _draw_power(*payload)
        except queue.Empty:
            pass
        root.after(150, poll)

    draw_budget()
    single_k()
    poll()
    root.mainloop()


# ===========================================================================
def main():
    ap = argparse.ArgumentParser(description="気相NMR 精度設計ツール")
    ap.add_argument("--batch", metavar="config.json", help="バッチ実行")
    ap.add_argument("--make-config", metavar="config.json", help="設定の雛形を生成")
    args = ap.parse_args()
    if args.make_config:
        make_config(args.make_config)
    elif args.batch:
        run_batch(args.batch)
    else:
        launch_gui()


if __name__ == "__main__":
    main()
