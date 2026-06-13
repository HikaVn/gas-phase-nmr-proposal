# 気相NMR分光法 ― 研究提案資料

冨宅喜代一氏が開発した **気相NMR分光法(磁気共鳴加速法)** ― 質量選別した気相分子イオンを超伝導磁石内で往復させ、核スピン状態を飛行時間(TOF)の分裂として読み、質量分析(MS)に化学シフトという直接的な構造情報を付与する手法 ― についての、研究状況・課題・精度向上ロードマップ・競合分析の資料集。

## 公開サイト(GitHub Pages)

👉 **https://hikavn.github.io/gas-phase-nmr-proposal/**

## 収録物

| ファイル | 内容 |
|---|---|
| [`index.html`](index.html) | ランディングページ |
| [`気相NMR_状況とロードマップ.html`](気相NMR_状況とロードマップ.html) | **メイン提案書**(要旨・原理・現状・精度の不等式 R=δ/w&gt;2・10因子・ロードマップ・競合・PoC精度仮説。対話的シミュレーション付き) |
| [`気相NMR_理解のための解説.html`](気相NMR_理解のための解説.html) | 原理の解説(姉妹資料) |
| [`bayes_comb_reanalysis.py`](bayes_comb_reanalysis.py) | **ベイズ再解析シミュレーター**。TOF分裂の有意性(ベイズ因子)と必要ショット数を試算 |
| [`gasnmr_tool.py`](gasnmr_tool.py) | 精度設計ツール(GUI+バッチ)。分解度バジェット R(N) とベイズ判定を統合 |
| [`run_bayes.bat`](run_bayes.bat) | **Windows用**: ベイズシミュレーターをダブルクリックで実行 |
| [`gasnmr_tool.bat`](gasnmr_tool.bat) | **Windows用**: 設計ツール(GUI)をダブルクリックで起動 |
| [`分析まとめ_気相NMR.md`](分析まとめ_気相NMR.md) | 研究概要・ボトルネック10項目 |
| [`文献調査_最新ヒント.md`](文献調査_最新ヒント.md) | 第1〜3回の文献調査ログ(R期待値の根拠つき) |

## 実行方法

### Windows(かんたん)
`.bat` を**ダウンロードしてダブルクリック**するだけ。Python本体さえ入っていれば、必要なスクリプトの取得・ライブラリ(numpy / scipy / matplotlib)の導入・実行まで自動で行います。

- `run_bayes.bat` … ベイズ再解析シミュレーター
- `gasnmr_tool.bat` … 設計ツール(GUI)

> Python が未導入の場合は [python.org](https://www.python.org/downloads/) からインストールし、インストーラの **「Add Python to PATH」にチェック**してください。`.bat` 単体をダウンロードすれば、不足する `.py` は GitHub から自動取得します。

### macOS / Linux(手動)
```bash
pip install numpy scipy matplotlib
python3 bayes_comb_reanalysis.py        # ベイズ再解析
python3 gasnmr_tool.py                   # 設計ツール(GUI)
python3 gasnmr_tool.py --batch config.json   # バッチ → batch_results.csv
```

## 原典・ことわり

- 中核総説 Fuke, *J. Mass Spectrom. Soc. Jpn.* **70**, 4 (2022) は [J-STAGE で無料公開](https://doi.org/10.5702/massspec.S22-03)。**著作権の都合により本リポジトリにPDFは再配布していません。**
- 「PoCで精度が落ちた可能性のある箇所」等の分析は、公開文献の記述に基づく **仮説・第三者による考察** であり断定ではありません。定量値の一部は説明用の見積もりです。
</content>
