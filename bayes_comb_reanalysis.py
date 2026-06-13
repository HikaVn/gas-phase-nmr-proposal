#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
気相NMR コム検出のベイズ再解析モデル
=====================================

冨宅式・気相NMR(磁気共鳴加速法)の原理検証データ(例: TEA+)に対して、
「核スピン分極による分裂(コム)が在るか」をベイズ因子で判定する。

★ 重要な設計(スレッドの核心):3つのモデルを比べる
  - M0 : 単一ピーク・線幅 σ 固定(=RF OFFと同じ) … 「何も起きていない」
  - M2 : 単一ピーク・線幅 σ 自由              … 「広がったが構造なし(ただの広がり)」
  - M1 : 二項コム(n+1本, パスカル比, 間隔δは理論値の狭い事前) … 「分裂あり(構造)」

  → K_change = M1/M0 :「そもそも広がったか」… 易しい(広がりがあれば巨大)
  → K_comb   = M1/M2 :「その広がりは"コム構造"か」… これが本当の難問。
                       未分解(R<1)では M2(ただの広がり)とほぼ区別できず ~1。
                       分解(R≳1.5)で初めて K_comb が大きくなる。
  これは「未分解だと"コム"と"ただの広がり"を区別できない」という
  分解能の話を、そのまま統計量で表したもの。

考え方(対話 C節):
  データ D = RF ON で測った個々のイオンの到達位置(速度シフト)の点列 {x_i}
  尤度 L = Π_i p(x_i|model)、事前 = 既知の物理(δ理論値・パスカル比・線幅σ・候補n)。
  周辺尤度はグリッド数値積分(log-sum-exp)で安定計算。

依存: numpy, scipy のみ。実行: python3 bayes_comb_reanalysis.py
"""

import numpy as np
from scipy.special import comb, logsumexp
from scipy.stats import norm

LOG10 = np.log(10.0)


# ---------------------------------------------------------------------------
# モデルの構成要素
# ---------------------------------------------------------------------------
def pascal_weights(n):
    """等価プロトン n 個の二項(パスカル)強度比 C(n,k)/2^n。"""
    k = np.arange(n + 1)
    return comb(n, k) / (2.0 ** n)


def mtotal_offsets(n):
    """各ピークの中心オフセット係数 M_total = k - n/2。"""
    return np.arange(n + 1) - n / 2.0


def comb_density(x, mu, delta, sigma, weights, offsets):
    """M1: コム(n+1本のガウス)の確率密度。"""
    centers = mu + offsets * delta
    pdf = norm.pdf(x[:, None], centers[None, :], sigma)
    return (pdf * weights[None, :]).sum(axis=1)


def broadening_factor(n, R):
    """分解度 R(=δ/FWHM)のコムが単峰に対し何倍に広がって見えるか。
       B = sqrt(1 + Var(M_total)*(δ/σ)^2), Var=n/4, δ/σ=2.3548*R。"""
    return np.sqrt(1.0 + (n / 4.0) * (2.3548 * R) ** 2)


# ---------------------------------------------------------------------------
# 事前分布
# ---------------------------------------------------------------------------
class Priors:
    def __init__(self, sigma, delta_theory, delta_frac=0.06,
                 mu_sd=None, n_candidates=(4,), f_bg=0.01,
                 sigma_max_factor=3.0):
        self.sigma = sigma                       # 線幅(RF OFFから既知, 固定)
        self.delta_theory = delta_theory         # 間隔の理論値 s0*(2N+1)
        self.delta_frac = delta_frac             # δ事前の相対幅(既知性 ~6%)
        self.mu_sd = mu_sd if mu_sd else 1.5 * sigma
        self.n_candidates = tuple(n_candidates)
        self.f_bg = f_bg
        self.sigma_max_factor = sigma_max_factor  # M2 の σ 上限(=広がり許容)
        nmax = max(self.n_candidates)
        self.window = (nmax / 2.0) * delta_theory + 6.0 * sigma

    def bg_density(self):
        return self.f_bg / (2.0 * self.window)


def _grid(center, half, num):
    g = np.linspace(center - half, center + half, num)
    return g, g[1] - g[0]


# ---------------------------------------------------------------------------
# 周辺尤度(エビデンス)
# ---------------------------------------------------------------------------
def log_evidence_M0(data, pr, n_mu=61):
    """M0: 単峰・σ固定・μ自由。"""
    mu_grid, dmu = _grid(0.0, 4.0 * pr.sigma, n_mu)
    bg = pr.bg_density()
    pdf = norm.pdf(data[:, None], mu_grid[None, :], pr.sigma)
    dens = (1 - pr.f_bg) * pdf + bg
    ll = np.log(dens).sum(axis=0)
    return logsumexp(ll + norm.logpdf(mu_grid, 0.0, pr.mu_sd) + np.log(dmu))


def log_evidence_M2(data, pr, n_mu=41, n_sig=41):
    """M2: 単峰・σ自由・μ自由(=ただの広がり)。"""
    mu_grid, dmu = _grid(0.0, 4.0 * pr.sigma, n_mu)
    sig_lo, sig_hi = 0.8 * pr.sigma, pr.sigma_max_factor * pr.sigma
    sig_grid = np.linspace(sig_lo, sig_hi, n_sig)
    dsig = sig_grid[1] - sig_grid[0]
    log_prior_mu = norm.logpdf(mu_grid, 0.0, pr.mu_sd)
    log_prior_sig = -np.log(sig_hi - sig_lo)      # 一様
    bg = pr.bg_density()
    terms = []
    for s in sig_grid:
        pdf = norm.pdf(data[:, None], mu_grid[None, :], s)
        dens = (1 - pr.f_bg) * pdf + bg
        ll = np.log(dens).sum(axis=0)
        log_int_mu = logsumexp(ll + log_prior_mu + np.log(dmu))
        terms.append(log_int_mu + log_prior_sig + np.log(dsig))
    return logsumexp(np.array(terms))


def log_evidence_M1(data, pr, n_mu=41, n_delta=41):
    """M1: コム・per-peak σ固定・μ自由・δは理論値の狭い事前・n候補で和。"""
    mu_grid, dmu = _grid(0.0, 4.0 * pr.sigma, n_mu)
    half_d = 3.0 * pr.delta_frac * pr.delta_theory
    delta_grid, ddelta = _grid(pr.delta_theory, half_d, n_delta)
    bg = pr.bg_density()
    log_prior_mu = norm.logpdf(mu_grid, 0.0, pr.mu_sd)
    log_prior_n = -np.log(len(pr.n_candidates))
    terms = []
    for n in pr.n_candidates:
        w = pascal_weights(n)
        off = mtotal_offsets(n)
        for d in delta_grid:
            centers = mu_grid[None, :, None] + off[None, None, :] * d
            pdf = norm.pdf(data[:, None, None], centers, pr.sigma)
            sig = (pdf * w[None, None, :]).sum(axis=2)
            dens = (1 - pr.f_bg) * sig + bg
            ll = np.log(dens).sum(axis=0)
            log_int_mu = logsumexp(ll + log_prior_mu + np.log(dmu))
            log_prior_d = norm.logpdf(d, pr.delta_theory,
                                      pr.delta_frac * pr.delta_theory)
            terms.append(log_int_mu + log_prior_d + np.log(ddelta) + log_prior_n)
    return logsumexp(np.array(terms))


def log10_bayes_factors(data, pr):
    """(log10 K_change=M1/M0, log10 K_comb=M1/M2) を返す。"""
    e0 = log_evidence_M0(data, pr)
    e2 = log_evidence_M2(data, pr)
    e1 = log_evidence_M1(data, pr)
    return (e1 - e0) / LOG10, (e1 - e2) / LOG10


# ---------------------------------------------------------------------------
# 疑似データ生成
# ---------------------------------------------------------------------------
def sample_comb(rng, n_ions, delta, sigma, n, mu=0.0):
    w = pascal_weights(n)
    off = mtotal_offsets(n)
    comp = rng.choice(n + 1, size=n_ions, p=w)
    return rng.normal(mu + off[comp] * delta, sigma)


def sample_single(rng, n_ions, sigma, mu=0.0):
    return rng.normal(mu, sigma, size=n_ions)


# ---------------------------------------------------------------------------
# パワー解析 ― K_comb(コム vs ただの広がり)で何イオン必要か
# ---------------------------------------------------------------------------
def power_analysis(rng, sizes, delta, sigma, n, trials=60, delta_frac=0.06):
    pr = Priors(sigma=sigma, delta_theory=delta, delta_frac=delta_frac,
                n_candidates=(n,))
    rows = []
    for N in sizes:
        l10 = []
        for _ in range(trials):
            data = sample_comb(rng, N, delta, sigma, n)
            _, lkc = log10_bayes_factors(data, pr)
            l10.append(lkc)
        l10 = np.array(l10)
        rows.append((N, np.median(l10),
                     np.mean(l10 > 1.0),     # K_comb > 10
                     np.mean(l10 > 2.0)))    # K_comb > 100
    return rows


# ---------------------------------------------------------------------------
# デモ
# ---------------------------------------------------------------------------
def demo():
    rng = np.random.default_rng(42)
    FWHM = 0.4
    sigma = FWHM / 2.3548
    n = 4                      # テスト用(TEA+ なら CH3:9, CH2:6)

    print("=" * 72)
    print(" 気相NMR コム検出 ベイズ再解析 ― デモ")
    print("=" * 72)
    print(f"  線幅 FWHM={FWHM} m/s (σ={sigma:.3f}),  等価プロトン n={n}")
    print(f"  K_change=M1/M0(広がったか) / K_comb=M1/M2(コム構造か=本命)")
    print(f"  値は log10。 log10 K=2 → K=100(決定的), =1 → K=10(支持)\n")

    # サニティ: 真に単峰なら両方とも負(M0/M2支持)
    delta = 0.2
    pr = Priors(sigma=sigma, delta_theory=delta, n_candidates=(n,))
    d_null = sample_single(rng, 2000, sigma)
    kc, kk = log10_bayes_factors(d_null, pr)
    print(f"[サニティ] 真に単峰(2000イオン): log10 K_change={kc:+.1f}, "
          f"log10 K_comb={kk:+.1f}  → 負ならコム否定で正常\n")

    # R を変えて K_change と K_comb を比較
    print(f"{'R':>5} {'広がり%':>8} {'N_ions':>7} {'log10 K_change':>15} {'log10 K_comb':>14}")
    print("-" * 56)
    for R in (0.2, 0.5, 1.5):
        delta = R * FWHM
        B = broadening_factor(n, R)
        pr = Priors(sigma=sigma, delta_theory=delta, n_candidates=(n,))
        for N in (1000, 3000):
            data = sample_comb(rng, N, delta, sigma, n)
            kc, kk = log10_bayes_factors(data, pr)
            print(f"{R:>5} {(B-1)*100:>7.0f}% {N:>7} {kc:>15.1f} {kk:>14.1f}")
        print()

    # パワー解析(K_comb で判定)
    print("=" * 72)
    print(" パワー解析: K_comb(コム vs ただの広がり)が閾値を超える割合")
    print("=" * 72)
    sizes = [300, 1000, 3000]
    for R in (0.2, 0.5, 1.5):
        delta = R * FWHM
        B = broadening_factor(n, R)
        print(f"\n  R={R} (広がり {(B-1)*100:.0f}%, δ={delta:.2f} m/s):")
        print(f"   {'N_ions':>7} | {'median log10 Kc':>15} | "
              f"{'P(Kc>10)':>9} | {'P(Kc>100)':>10}")
        print("   " + "-" * 52)
        for N, med, p10, p100 in power_analysis(rng, sizes, delta, sigma, n,
                                                 trials=40):
            print(f"   {N:>7} | {med:>15.1f} | {p10*100:>8.0f}% | "
                  f"{p100*100:>9.0f}%")

    print("\n" + "=" * 72)
    print(" 読み筋:")
    print("  ・K_change(広がったか)は易しい ― 少数でも巨大。")
    print("  ・K_comb(コム構造か)が本命 ― R=0.2(現状TEA+級)では")
    print("    数千イオンでも伸び悩む=未分解では構造を証明しにくい。")
    print("  ・R=1.5(分解)に上げると K_comb が一気に立つ=少数で決定的。")
    print("  → 結論: 統計だけでは限界。Rを上げる(ν0低減/②④)のが本質的。")
    print("=" * 72)

    # 図(matplotlib があれば)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
        for ax, R in zip(axes, (0.2, 0.5, 1.5)):
            delta = R * FWHM
            data = sample_comb(rng, 4000, delta, sigma, n)
            x = np.linspace(-1.6, 1.6, 700)
            w = pascal_weights(n); off = mtotal_offsets(n)
            ax.hist(data, bins=70, density=True, color="#d85a30", alpha=0.45,
                    label="実データ")
            ax.plot(x, comb_density(x, 0, delta, sigma, w, off),
                    color="#1d9e75", lw=2, label="M1: コム")
            B = broadening_factor(n, R)
            ax.plot(x, norm.pdf(x, 0, sigma * B), color="#534ab7", lw=1.6,
                    ls="--", label="M2: 広がり")
            ax.set_title(f"R={R}  (広がり{(B-1)*100:.0f}%)")
            ax.set_xlabel("速度シフト [m/s]"); ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig("bayes_comb_demo.png", dpi=130)
        print("\n[図を保存] bayes_comb_demo.png")
    except Exception as e:
        print(f"\n(図はスキップ: {e})")


if __name__ == "__main__":
    demo()
