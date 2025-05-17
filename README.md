# andersoned

*Bethe 格子バスをもつ単一不純物アンダーソン模型（SIAM）の厳密対角化 (ED) 研究を手軽に行うための、小さく自己完結型の Python ツールボックスです。*

---

## ✨  概要

`andersoned` は 2 つの機能をまとめています。

| モジュール | 目的 | 主な公開シンボル |
| ---------- | ---- | ---------------- |
| `andersoned.fit_bethe` | **非相互作用系**: Bethe 格子のハイブリダイゼーション関数を高精度で再現するように有限個の浴 (bath) 軌道をあてはめます。 | `fit_bethe_bath`, `bethe_green` |
| `andersoned.ed` | **相互作用系**: 多体ハミルトニアンを構築し、密な ED を行い、Matsubara 領域および実周波数のグリーン関数を評価します。 | `build_hamiltonian`, `diagonalize`, `green_matsubara`, `green_realaxis`, `spectral_function` |

実装は純粋な NumPy / SciPy のみで書かれており、Fortran や C のバイナリは不要です。

---

## 📦  インストール

```bash
git clone https://github.com/your-org/andersoned.git
cd andersoned
python -m pip install -e .
```

### 依存パッケージ

* NumPy & SciPy  
* Matplotlib（例や可視化に使用）

---

## 🚀  使い方クイックガイド

### 1. Bethe 格子バスのフィッティング

```python
from andersoned import fit_bethe_bath, bethe_green
eps_p, V_p, chi2 = fit_bethe_bath(nbath=4, beta=50.0, nw_fit=200)
print(f"χ² = {chi2:.2e}")
```

フルデモ（プロット付き）を実行:

```bash
python examples/bethe_fit_demo.py
```

### 2. フル ED とスペクトル関数

```bash
python examples/ed_full_demo.py
```

---

## 🛠️  最小 API リファレンス

```python
from andersoned import (
    fit_bethe_bath,      # バスのフィット（非相互作用）
    bethe_green,         # Bethe 格子の解析的 G

    build_hamiltonian,   # 多体ハミルトニアンの構築
    diagonalize,         # ED (SciPy LAPACK ラッパー)
    green_matsubara,     # G(iω_n)
    green_realaxis,      # G(ω+iη)
    spectral_function,   # A(ω) = -Im G / π
)
```

---

## 🗂️  リポジトリ構成

```
andersoned-project/
│
├─ andersoned/            # import 可能なモジュール
│   ├─ __init__.py
│   ├─ fit_bethe.py
│   └─ ed.py
│
├─ examples/              # 実行可能なデモ
│   ├─ bethe_fit_demo.py
│   └─ ed_full_demo.py
│
├─ pyproject.toml         # ビルド情報
└─ README.md              # ← 本ファイル
```

---

## 🧪  開発のヒント

* **整形**: `black .`  
* **静的チェック**: `ruff .`  
* **テスト**: `pytest`（準備中）

Numba / Cython による高速化 PR も歓迎します！

---

## 📜  ライセンス

MIT License で配布しています。詳細は `LICENSE` を参照してください。

---

## 🙏  謝辞

この小さなツールボックスは ChatGPT とのブレーンストーミングの中で生まれ、以下にインスパイアされています。

* 不純物問題に対する数値的 RG  
* ハバード・アンダーソン模型の標準的な ED チュートリアル  
* **DMFT** おもちゃコード など

楽しい対角化ライフを！
