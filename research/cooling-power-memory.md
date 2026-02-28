# NVIDIA AI データセンター: 冷却・電力インフラとメモリ階層

調査日: 2026-02-28

---

## 第1部: 冷却・電力インフラ

### 1. AI データセンターの電力構造

#### 1.1 DGX SuperPOD の電力要件（GB200 NVL72 ベース）

NVIDIA DGX SuperPOD は、GB200 NVL72 をベースとしたスケーラブルユニット（SU）で構成される。

| 項目 | 仕様 |
|------|------|
| **ラック単体（DGX GB200 NVL72）** | 120 kW TDP |
| **スケーラブルユニット（SU）** | 1.2 MW TDP（8 ラック構成） |
| **重量（単体ラック）** | 1.36 メトリックトン（約 3,000 lbs） |
| **電源シェルフ** | 8 基/ラック、各シェルフ 6 x 5.5kW PSU（合計 33kW/シェルフ） |
| **電源方式** | AC → 50-51V DC 変換、バスバーで分配 |

**SuperPOD 構成の電力階層:**

```
SuperPOD（フルスケール）
├── Scalable Unit (SU) x N
│   ├── DGX GB200 NVL72 ラック x 8 = 1.2 MW
│   │   ├── 72 Blackwell GPU（B200）
│   │   ├── 36 Grace CPU
│   │   ├── 864 GB メモリ/Superchip（480GB LPDDR5x + 384GB HBM3e）
│   │   └── NVLink Switch System: 130 TB/s GPU 間通信
│   └── ネットワーク: Quantum-X800 InfiniBand / Spectrum-X800 Ethernet（800 Gb/s）
└── 性能: 1.44 exaFLOPS/ラック（FP4）
```

**出典:**
- https://docs.nvidia.com/dgx-superpod/reference-architecture-scalable-infrastructure-gb200/latest/dgx-superpod-components.html
- https://www.nvidia.com/en-us/data-center/gb200-nvl72/

#### 1.2 ラックあたり消費電力の変遷

| 世代 | ラック電力密度 | 備考 |
|------|---------------|------|
| **従来のデータセンター** | 10-20 kW/ラック | 汎用サーバー |
| **初期 AI DC（A100 時代）** | 30-40 kW/ラック | DGX A100 |
| **Hopper 世代（H100）** | 40-56 kW/ラック | DGX H100 SuperPOD |
| **Blackwell 世代（GB200 NVL72）** | 120-132 kW/ラック | 液冷必須 |
| **Blackwell Ultra / B300** | 250 kW/ラック | 2025-2026 |
| **Rubin（NVL144）** | 120-130 kW/ラック | 既存インフラ互換、2026 H2 |
| **Rubin Ultra（Kyber ラック）** | 600 kW/ラック | 2027、完全液浸冷却 |
| **将来予測（2029）** | ~1 MW/ラック | Vertiv 予測 |

**OCP（Open Compute Project）の動向:**
- Meta、Google、Microsoft が ORv3-HPR V3 ラックを採用（300 kW/キャビネット対応）
- NVIDIA は OCP 2025 で 1 MW ラックデザインを発表

**出典:**
- https://introl.com/blog/high-density-racks-100kw-ai-data-center-ocp-2025
- https://www.trendforce.com/insights/data-center-power
- https://www.kad8.com/server/data-center-rack-density-in-2025-how-high-can-it-scale/

#### 1.3 PUE（Power Usage Effectiveness）の実績値

| 冷却方式 | PUE 範囲 | 備考 |
|----------|----------|------|
| **従来の空冷 DC** | 1.4-1.8 | 業界平均 1.56 |
| **最適化された空冷** | 1.2-1.3 | 理想的条件下 |
| **液冷（Direct-to-Chip）** | 1.05-1.15 | 外気温に依存しない |
| **液冷（最適化）** | 1.02-1.08 | pPUE（冷却のみ） |
| **Google（2024 年間平均）** | 1.09 | フリート全体 |
| **Google（2025 Q1）** | 1.08 | 四半期値 |

PUE の計算式: `PUE = 総施設消費電力 / IT 機器消費電力`

Google の実績:
- 2024 年のフリート平均 PUE は 1.09（業界平均 1.56 と比較して冷却オーバーヘッド 84% 削減）
- 6 年ぶりに 1.10 を下回った
- 2024 年のデータセンター総電力消費: 30.8 百万 MWh（2020 年の 14.4 百万 MWh から倍増以上）

**出典:**
- https://datacenters.google/efficiency/
- https://introl.com/blog/liquid-vs-air-cooling-ai-data-centers

#### 1.4 電力供給がボトルネックになっている具体的事例

**事例 1: NVIDIA 本社所在地 Santa Clara の電力不足**
- Digital Realty Trust が 2019 年にデータセンター建設を申請
- 約 6 年経過した 2025 年時点で、ローカルユーティリティが電力を供給できず空のまま
- NVIDIA の本拠地でありながら、AI データセンターが電力待ちで稼働できない事態

**事例 2: グローバルな電力需要の急増**
- AI データセンターへの 2025 年の投資額: 約 5,800 億ドル
- グローバル DC 電力消費: 460 TWh（2024）→ 1,300 TWh（2035）予測
- 電力需給ギャップにより、多くのデータセンタープロジェクトが遅延

**NVIDIA の対策:**
- 800 VDC 電源アーキテクチャへの移行（高電圧 DC 配電で効率向上）
- PJM、National Grid 等のグリッドオペレーターとのパイロットプロジェクト
- Omniverse DSX Blueprint によるギガワット級 AI ファクトリーの標準設計

**出典:**
- https://fortune.com/2025/11/10/nvidia-hometown-santa-clara-california-data-centers-empty-power-grid/
- https://enkiai.com/nvidia/nvidia-power-strategy-2025-inside-the-ai-energy-pivot
- https://www.digitimes.com/news/a20251216PD205/nvidia-data-power-supply-growth-electricity.html

---

### 2. 液冷技術

#### 2.1 直接液冷（Direct-to-Chip / DLC）の仕組み

**基本アーキテクチャ:**

```
外部冷却水（施設側）
    │
    ▼
┌──────────────────┐
│  CDU（冷却分配装置）  │  ← 施設水系統とサーバー冷却系統の熱交換
│  Coolant Dist. Unit │
└──────┬───────────┘
       │ 内部冷却水（サーバー側）
       ▼
┌──────────────────┐
│  ラックマニフォールド  │  ← ラック全体に冷却水を分配
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│  コールドプレート     │  ← GPU/CPU に直接装着し、熱を吸収
│  （銅製微細流路）    │
└──────────────────┘
```

**DLC の主要コンポーネント:**

1. **コールドプレート（Cold Plate）**: GPU/CPU ダイに直接装着。ロウ付け銅製で、内部に微細流路（マイクロチャネル）を持つ。空気の 30 倍の熱伝導効率
2. **マニフォールド（Manifold）**: ラック内の冷却水分配。Blind mate QD（クイックディスコネクト）でリークフリー接続
3. **CDU（Coolant Distribution Unit）**: 施設冷却水系統とサーバー冷却系統間の熱交換器。冗長ポンプ搭載

**GB200 NVL72 の冷却仕様:**

| パラメータ | 値 |
|-----------|-----|
| 冷却方式 | Direct-to-Chip 液冷（必須） |
| 冷却容量 | 最大 250 kW/ラック |
| 冷却水入口温度 | 最大 45°C |
| 冷却水出口温度 | 最大 65°C |
| 流量 | 最大 130 LPM（リットル/分）/ラック |
| 熱吸収率 | 発熱の 98%（Supermicro DLC-2 実績） |

**マニフォールドアーキテクチャ（Boyd 社の実装例）:**
- **インナーマニフォールド**: サーバーシャーシ内でコールドプレートに冷却水を分配
- **ラックマニフォールド**: ラックレベルで集中型冷却水インターフェースを提供。銅製、垂直/水平配置可能

**出典:**
- https://docs.nvidia.com/dgx/dgxgb200-user-guide/hardware.html
- https://www.boydcorp.com/blog/cooling-nvidia-gb200-nvl72-artificial-intelligence.html
- https://www.lorithermal.com/lori-data-center-server-liquid-cooling-components-precise-and-high-quality-cooling-assurance-for-your-nvidia-gb200-ai-se

#### 2.2 NVIDIA の液冷リファレンスデザイン

NVIDIA は GB200 NVL72 向けに推奨ベンダーリスト（RVL）を整備し、液冷エコシステムを標準化している。

**認定ベンダー例:**
- **Boyd Corporation**: ロウ付け銅製コールドプレートアセンブリ（左右オリエンテーション対応）
- **CoolIT Systems**: 250 kW 対応ラック冷却ソリューション（入水温度 40°C）
- **Accelsius**: 高密度液冷ソリューション
- **JetCool**: SmartSense CDU
- **QCT**: QOOLRACK Stand-Alone Advanced Liquid Cooling

**使用冷媒:**
- **標準（DLC）**: PG-25（25% プロピレングリコール水溶液）。天然の殺菌作用あり。追加化学物質不要でそのまま使用
- **浸漬冷却用**: Fluorinert（3M 製の誘電性液体）。高い熱伝導性だが高コスト（100 ドル+/リットル）

**出典:**
- https://www.boydcorp.com/uncategorized/boyd-validated-for-nvidia-gb200-nvl72-recommended-vendor-list.html
- https://jetcool.com/coolant-distribution-units-smartsense-cdu/

#### 2.3 空冷 vs 液冷のコスト・効率比較

| 比較項目 | 空冷 | 液冷（DLC / 浸漬） |
|----------|------|---------------------|
| **PUE** | 1.2-1.8 | 1.02-1.15 |
| **冷却電力消費** | 0.5-1.2 kW/kW IT | 0.1-0.3 kW/kW IT |
| **最大ラック電力** | ~41.3 kW（物理限界） | 100-200+ kW |
| **冷却エネルギー削減** | ベースライン | 最大 90% 削減 |
| **初期コスト** | 低い | 高い（2-3 倍） |
| **運用コスト** | 高い | 低い（電力費 31-37% 削減） |
| **10 年 TCO（10MW 施設）** | 高い | ~$111M 削減（39% TCO 削減） |
| **年間電力コスト差（10MW）** | ベースライン | $3-7M/年の節約 |
| **サーバー寿命** | 3-4 年（ファン摩耗・粉塵） | 5-7 年（可動部品なし） |
| **メンテナンス頻度** | 高い（フィルター交換等） | 低い |
| **サーバー密度** | ベースライン | 58% 向上 |

**市場予測:**
- データセンター液冷市場: 49 億ドル（2024）→ 213 億ドル（2030）、CAGR 27.6%
- 2025 年以降の新規ハイパースケール DC では液冷が主流化
- 既存施設やスモール DC では空冷が継続

**出典:**
- https://introl.com/blog/liquid-vs-air-cooling-ai-data-centers
- https://www.profileits.com/understanding-the-total-cost-of-ownership-tco-for-a-10-mw-ai-data-center-air-cooling-vs-immersion-cooling/

---

### 3. Blackwell / Rubin の冷却要件

#### 3.1 Blackwell B200 の冷却要件

| パラメータ | 空冷版 | 液冷版 |
|-----------|--------|--------|
| **TDP** | 1,000 W | 1,200 W |
| **冷却方式** | 空冷（低電力モード） | DLC 必須 |
| **性能** | 制限あり | フル性能（20 PFLOPS FP4） |
| **ラック密度** | 低い | 高い（58% 向上） |

**GB200 Superchip（Grace CPU + 2x B200 GPU）:**
- TDP: 2,700 W
- 冷却方式: DLC 必須（空冷不可）

**DLC コールドプレート要件:**
- 最大 1,600 W/コンポーネントの熱処理能力
- 微細流路銅製コールドプレート（MCCP: Microchannel Cold Plate）
- 冷却水入口温度: 25°C（推奨）、最大 45°C
- 冷却水出口温度: 最大 65°C

#### 3.2 NVIDIA Rubin の電力・冷却要件

**Rubin（R200）- 2026 年後半出荷:**

| パラメータ | 仕様 |
|-----------|------|
| **GPU TDP** | 最大 2,300 W |
| **プロセス** | TSMC 3nm クラス |
| **メモリ** | 288 GB HBM4（8 スタック） |
| **帯域幅** | ~20.5 TB/s（HBM4 @ 10 Gbps） |
| **性能** | 50 PFLOPS FP4 / ~16 PFLOPS FP8 |
| **NVL144 ラック電力** | 120-130 kW（既存インフラ互換） |

**Rubin Ultra（VR200）- 2027 年:**

| パラメータ | 仕様 |
|-----------|------|
| **GPU TDP** | 最大 2,300 W（CPX で最大 3,700 W） |
| **ラック電力（Kyber）** | 600 kW |
| **冷却方式** | 完全液浸冷却（ファンゼロ） |
| **ラック構成** | コンピュートラック + サイドカー（電力・冷却用） |
| **電力配電** | 800 VDC |
| **GPU 数/ラック** | NVL576 構成で最大 576 GPU |

**冷却技術の進化:**
- Rubin Ultra 向けに Asia Vital Components に MCCP（マイクロチャネルコールドプレート）の設計を依頼
- CoolIT Systems / Accelsius が 250 kW ラック（入水温度 40°C）対応を実証済み
- 600 kW ラック対応冷却ソリューションの開発が進行中

**出典:**
- https://introl.com/blog/nvidia-vera-rubin-gpu-600kw-racks-2027
- https://www.tomshardware.com/pc-components/gpus/nvidia-shows-off-rubin-ultra-with-600-000-watt-kyber-racks-and-infrastructure-coming-in-2027
- https://www.tweaktown.com/news/108068/nvidia-could-change-cooling-solution-for-rubin-ultra-ai-gpus-for-huge-2300w-thermal-concerns/index.html

#### 3.3 GPU 世代別 TDP の推移

```
TDP (W)
3700 ┤                                           ● Rubin Ultra CPX
     │
2300 ┤                                    ● Rubin Ultra (VR200)
     │
1200 ┤                          ● B200 (液冷)
1000 ┤                     ● B200 (空冷)
 700 ┤               ● H100 SXM
 400 ┤         ● A100 SXM
 300 ┤    ● V100
     └──────────────────────────────────────────→ 世代
      2017   2020    2022    2024    2026   2027
```

---

### 4. サステナビリティ

#### 4.1 Google PUE 1.09 達成の手法

Google が業界平均 1.56 に対して PUE 1.09 を達成した主要手法:

1. **カスタム冷却システム**: 外気冷却（フリークーリング）の最大活用。寒冷地への戦略的立地
2. **AI による冷却最適化**: DeepMind の機械学習でデータセンター冷却を最適化（冷却エネルギー 40% 削減の事例）
3. **効率的な電力配電**: 高効率 UPS と電力変換、最小限のオーバーヘッド
4. **カスタムサーバー設計**: Google 自社設計のサーバー、不要コンポーネント排除
5. **再生可能エネルギー**: 2017 年以降 100% 再生可能エネルギーマッチを維持。2024 年に 60 件の新規クリーンエネルギー契約（2.5 GW）

#### 4.2 廃熱の再利用（ディストリクトヒーティング）

液冷データセンターの出水温度（45-65°C）は地域暖房に最適であり、北欧を中心に実用化が進んでいる。

**主要事例:**

| 事業者 | 場所 | 規模 | 状況 |
|--------|------|------|------|
| **Google** | Hamina, Finland | 地域暖房需要の 80% を供給 | 稼働中 |
| **Microsoft / VEKS** | Denmark | 住宅向け暖房 | 2025-2026 暖房シーズンから |
| **atNorth** | Copenhagen, Denmark | 8,000 世帯分の熱エネルギー | 2028 年から供給予定 |

**効果:**
- データセンターの実質消費電力を 10-30% 削減
- 地域住民の暖房コストと排出量の低減
- DLC（直接液冷）による高温排水が地域暖房ネットワークとの親和性が高い

**出典:**
- https://www.weforum.org/stories/2025/06/sustainable-data-centre-heating/
- https://www.ehn.org/nordic-homes-are-being-warmed-by-waste-heat-from-massive-data-centers
- https://www.bloomberg.com/news/features/2025-05-14/finland-s-data-centers-are-heating-cities-too

#### 4.3 再生可能エネルギーとの統合

**原子力（SMR: Small Modular Reactor）への投資:**

| 企業 | パートナー | 規模 | 時期 |
|------|-----------|------|------|
| **Google** | Kairos Power | 500 MW（米国初の企業向け SMR フリート契約） | 2030 年以降 |
| **Amazon** | Susquehanna 拠点転用 | 200 億ドル+ 投資 | 進行中 |
| **Meta** | RFP 発行 | 1-4 GW の新規原子力 | 計画中 |
| **Microsoft** | 各種原子力パートナーシップ | 複数プロジェクト | 進行中 |

**NuScale SMR:**
- 2025 年 5 月に Standard Design Approval を取得（462 MW の NuScale US 460）
- 2030 年以降、ベースロード低排出電力として DC 向け供給開始

**現在の米国 DC 電源構成:**
- 天然ガス: 40%+
- 再生可能エネルギー: ~24%
- 原子力: 15-20%
- 石炭: ~15%

**課題:**
- 間欠性のある再エネ（太陽光・風力）だけでは 24/7 のデータセンター電力需要を満たせない
- 原子力は 24/7 カーボンフリー電力を提供できるが、建設リードタイムが長い
- SMR は 2030 年以降の実用化を目指すが、それまでのギャップをどう埋めるかが課題

**出典:**
- https://introl.com/blog/nuclear-power-ai-data-centers-microsoft-google-amazon-2025
- https://introl.com/blog/smr-nuclear-power-ai-data-centers-2025
- https://spectrum.ieee.org/nuclear-powered-data-center

---

## 第2部: メモリ階層

### 1. GPU メモリ階層の全体像

#### 1.1 メモリ階層の構造

```
レイテンシ（低い）← ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ →（高い）
帯域幅（高い）← ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ →（低い）
容量（小さい）← ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ →（大きい）

┌────────────────┐
│   レジスタ        │  ~1 サイクル / ~0.3 ns / ~数十 TB/s
├────────────────┤
│   L1 / 共有メモリ  │  30-40 サイクル / ~10-20 ns / ~数十 TB/s（SM 内）
├────────────────┤
│   L2 キャッシュ    │  273-358 サイクル / ~100-200 ns / ~数 TB/s
├────────────────┤
│   HBM (VRAM)    │  420-1000 サイクル / ~300-500 ns / 1-8 TB/s
├────────────────┤
│   CPU メモリ      │  ~数千サイクル / ~100-200 ns (CPU側) / 100-500 GB/s
│   (DDR5/LPDDR5x) │  ※ PCIe/NVLink 経由で追加レイテンシ
├────────────────┤
│   NVMe SSD      │  ~数万サイクル / ~10-100 μs / 数-数十 GB/s
├────────────────┤
│ ネットワークストレージ│  ~100 μs-1 ms / 数-数百 GB/s（ファブリック依存）
└────────────────┘
```

#### 1.2 NVIDIA Blackwell（B200）のメモリ階層詳細

| メモリ階層 | 容量 | レイテンシ | 帯域幅 | 備考 |
|-----------|------|-----------|--------|------|
| **レジスタ** | 256 KB/SM | ~1 サイクル | - | 最速、SM ごとに専用 |
| **L1 / 共有メモリ** | 128 KB/SM（構成可能: 0-228 KB） | 30-40 サイクル | SM ローカル | Hopper の 256 KB/SM から削減 |
| **TMEM（Tensor Memory）** | Blackwell 新規 | 420 サイクル（キャッシュミス時） | - | Hopper のグローバルメモリ 1000 サイクルから 58% 削減 |
| **L2 キャッシュ** | 126 MB（モノリシック） | ~358 サイクル | - | Hopper の 50 MB（2 パーティション）から 2.5 倍 |
| **HBM3e (VRAM)** | 192 GB | - | 8 TB/s | Hopper HBM3 の 3.35 TB/s から 2.4 倍 |

**Hopper（H100）との比較:**

| メモリ階層 | H100（Hopper） | B200（Blackwell） | 倍率 |
|-----------|----------------|-------------------|------|
| **L1/SM** | 256 KB/SM | 128 KB/SM | 0.5x |
| **L2 キャッシュ** | 50 MB | 126 MB | 2.5x |
| **L2 レイテンシ** | ~273 サイクル | ~358 サイクル | 1.3x（増加） |
| **HBM 容量** | 80 GB（HBM3） | 192 GB（HBM3e） | 2.4x |
| **HBM 帯域幅** | 3.35 TB/s | 8 TB/s | 2.4x |
| **NVLink 帯域幅** | 900 GB/s | 1.8 TB/s | 2x |

Blackwell は L1 を縮小する代わりにモノリシック L2 を大幅に拡大し、HBM 帯域幅を 2.4 倍に強化する設計思想。TMEM による高速テンソルアクセスが新たに追加された。

**出典:**
- https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html
- https://arxiv.org/pdf/2507.10789 (Blackwell microbenchmarks)
- https://arxiv.org/html/2512.02189v1
- https://charlesgrassi.dev/blog/gpu-cache-hierarchy/

---

### 2. HBM の進化

#### 2.1 HBM 世代別スペック比較

| 仕様 | HBM2E | HBM3 | HBM3E | HBM4 |
|------|-------|-------|--------|-------|
| **JEDEC 規格** | - | JESD238A | - | JESD270-4 |
| **データレート/ピン** | 3.2 Gb/s | 6.4 Gb/s | 9.6 Gb/s | 8 Gb/s（規格値）〜11.7 Gb/s（実製品） |
| **バス幅** | 1024-bit | 1024-bit | 1024-bit | 2048-bit |
| **チャネル数** | 8 | 16 | 16 | 32（擬似チャネル: 2/ch） |
| **帯域幅/スタック** | ~410 GB/s | ~819 GB/s | ~1,229 GB/s | ~2 TB/s（規格）〜3.3 TB/s（Samsung 実装） |
| **最大スタック高** | 8-Hi | 12-Hi | 12-Hi | 4/8/12/16-Hi |
| **ダイ密度** | 16 Gb | 16 Gb | 24 Gb | 24/32 Gb |
| **最大容量/スタック** | 16 GB | 24 GB | 36 GB | 64 GB（32Gb 16-Hi） |
| **採用 GPU** | A100 | H100 | H200, B200 | Rubin (R200) |
| **量産時期** | 2020 | 2022 | 2023-2024 | 2026 H2 |

**世代間の帯域幅向上率:**
- HBM2E → HBM3: ~2.0x
- HBM3 → HBM3E: ~1.5x
- HBM3E → HBM4: ~1.6-2.7x（実装により異なる）

#### 2.2 HBM4 の詳細

**JEDEC HBM4 規格（JESD270-4、2025 年 4 月発表）:**
- インターフェース幅: 2048-bit（HBM3 の 2 倍）
- 独立チャネル: 32（HBM3 の 16 から倍増）
- 最大転送速度: 8 Gb/s/ピン
- 最大帯域幅: 2 TB/s/スタック
- スタック構成: 4/8/12/16-Hi
- 最大容量: 64 GB/スタック

**製造元の開発状況（2025-2026）:**

| 製造元 | 製品 | 仕様 | 状況 |
|--------|------|------|------|
| **SK Hynix** | 16-Hi HBM4 | 48 GB, 11.7 Gbps, 2 TB/s | 世界初の HBM4 開発完了。2026 Q3 量産開始 |
| **Samsung** | HBM4 | 36 GB, 3.3 TB/s/チャネル | ISSCC 2026 で発表 |
| **Micron** | HBM4 | 開発中 | 詳細未公表 |

**HBM4 の NVIDIA Rubin での利用:**
- R200 GPU: 288 GB HBM4（8 スタック x 36 GB）
- メモリ帯域幅: ~20.5 TB/s（10 Gbps で動作、規格の 8 Gbps を超過）
- 帯域幅は B200 の HBM3e（8 TB/s）から ~2.6 倍

**出典:**
- https://www.jedec.org/news/pressreleases/jedec%C2%AE-and-industry-leaders-collaborate-release-jesd270-4-hbm4-standard-advancing
- https://news.skhynix.com/sk-hynix-completes-worlds-first-hbm4-development-and-readies-mass-production/
- https://www.tomshardware.com/tech-industry/semiconductors/hbm-roadmaps-for-micron-samsung-and-sk-hynix-to-hbm4-and-beyond
- https://www.eetimes.com/the-state-of-hbm4-chronicled-at-ces-2026/

---

### 3. メモリ帯域幅がボトルネックになるケース

#### 3.1 Compute-bound vs Memory-bound の判定

**Roofline モデル:**

```
性能
(FLOPS)
  │
  │         メモリルーフライン        演算ルーフライン
  │              ╱                ─────────────
  │            ╱                │
  │          ╱                  │
  │        ╱                    │
  │      ╱                      │
  │    ╱   Memory-bound    Compute-bound
  │  ╱     （帯域幅制限）     （演算能力制限）
  │╱                            │
  └──────────────────────────────→ Arithmetic Intensity
                               (FLOP/Byte)
     転換点 = Peak FLOPS / Peak Bandwidth (FLOP/Byte)
```

**判定基準:**

```
Arithmetic Intensity (AI) = 実行される演算量 (FLOPs) / 転送されるデータ量 (Bytes)

if AI < Peak FLOPS / Peak Bandwidth:
    → Memory-bound（メモリ帯域幅がボトルネック）
else:
    → Compute-bound（演算能力がボトルネック）
```

**B200 の転換点:**
- Peak FP16 Tensor: ~4.5 PFLOPS = 4,500 TFLOPS
- Peak HBM Bandwidth: 8 TB/s = 8,000 GB/s
- 転換点: 4,500 / 8 = 562.5 FLOP/Byte

→ AI が 562.5 未満のカーネルは Memory-bound

#### 3.2 LLM 推論が Memory-bound である理由

**Prefill（プロンプト処理）フェーズ vs Decode（トークン生成）フェーズ:**

| フェーズ | 特性 | ボトルネック |
|----------|------|-------------|
| **Prefill** | 大きなテンソル、高い再利用性、バッチ処理可能 | Compute-bound |
| **Decode** | 1 トークンずつ生成、頻繁なメモリアクセス | Memory-bound |

**Decode フェーズが Memory-bound である理由:**

1. **低い Arithmetic Intensity**: Decode ステップでは各トークン生成に対してモデルの全パラメータ（数十〜数百 GB）を読み込む必要があるが、実行される演算量は少ない
2. **Attention カーネルの AI が定数的**: バッチサイズを変えても Attention カーネルの AI はほぼ一定で、Memory-bound 領域に留まる
3. **行列ベクトル積（GEMV）**: Decode 時の主要演算は GEMV（行列 x ベクトル）であり、行列全体をロードするのに対して演算が O(n) しかない
4. **KV キャッシュアクセス**: 生成が進むにつれ KV キャッシュが増大し、メモリアクセスが支配的になる

**具体例（70B パラメータモデル、FP16）:**
- モデルサイズ: ~140 GB
- 1 トークン生成に必要な読み込み: ~140 GB（全パラメータ）
- 1 トークン生成の演算量: ~140 GFLOP（2 x パラメータ数）
- AI = 140 GFLOP / 140 GB = 1 FLOP/Byte
- B200 の転換点 562.5 FLOP/Byte に対して圧倒的に低い → **完全に Memory-bound**

**最適化戦略:**

| 戦略 | 効果 | 説明 |
|------|------|------|
| **量子化（INT8/INT4/FP4）** | AI 向上 | パラメータサイズ削減で転送量を減らし、AI を向上 |
| **バッチ処理** | AI 向上 | 複数リクエストの同時処理で演算量/データ転送比を改善 |
| **カーネルフュージョン** | レイテンシ削減 | 複数操作を統合してメモリアクセス回数を削減 |
| **Speculative Decoding** | スループット向上 | 小モデルで候補生成、大モデルで検証 |
| **KV キャッシュ圧縮** | メモリ帯域幅節約 | PagedAttention 等でキャッシュ効率化 |

**出典:**
- https://arxiv.org/html/2402.16363v4 (LLM Inference Roofline Model)
- https://arxiv.org/html/2512.01644v1 (Systematic Characterization of LLM Inference)
- https://arxiv.org/html/2503.08311v2 (Mind the Memory Gap)
- https://pub.towardsai.net/the-engineering-guide-to-efficient-llm-inference-metrics-memory-and-mathematics-3aead91c99cc

---

### 4. GPUDirect テクノロジーのメモリ階層への影響

#### 4.1 GPUDirect ファミリーの概要

GPUDirect は NVIDIA Magnum IO の一部であり、GPU とのデータ移動を最適化する技術群。

```
従来のデータパス:
  NIC/SSD → CPU メモリ（バウンスバッファ） → GPU メモリ
  （CPU がボトルネック、追加のメモリコピー、高レイテンシ）

GPUDirect:
  NIC/SSD → GPU メモリ（ダイレクト DMA）
  （CPU バイパス、ゼロコピー、低レイテンシ）
```

#### 4.2 GPUDirect RDMA: NIC → GPU メモリ直接転送

**概要:**
- InfiniBand / Ethernet NIC から GPU メモリへの直接 DMA 転送
- CPU とシステムメモリのバウンスバッファを完全にバイパス

**性能:**
- 従来パス比で **10 倍**の性能向上
- CPU 負荷の大幅削減
- 通信レイテンシの低減、実効帯域幅の向上

**メモリ階層への影響:**

```
最適化前:                         最適化後:
ネットワーク                      ネットワーク
    ↓                                ↓
NIC                              NIC
    ↓                                ↓（GPUDirect RDMA）
CPU メモリ（バウンスバッファ）         GPU メモリ（HBM）← 直接転送
    ↓
GPU メモリ（HBM）
```

→ **ネットワークストレージ → CPU メモリ → GPU メモリ** の 2 段階を **ネットワーク → GPU メモリ** の 1 段階に短縮

#### 4.3 GPUDirect Storage: NVMe → GPU メモリ直接転送

**概要:**
- NVMe SSD から GPU メモリへの直接 DMA 転送
- CPU とシステムメモリのバウンスバッファを完全にバイパス

**性能ベンチマーク:**

| メトリック | 従来パス | GPUDirect Storage | 向上率 |
|-----------|---------|-------------------|--------|
| **スループット** | 最大 33 GB/s（CPU 飽和） | 84 GB/s | **460% 向上** |
| **CPU 使用率** | 高い（ボトルネック） | 15% | 大幅削減 |
| **GPU 単体** | - | 8+ GB/s over PCIe | - |
| **帯域幅** | ベースライン | 2-8 倍 | - |

**CUDA 12.8 以降の改善:**
- P2P モード対応により、nvidia-fs カーネルモジュールが不要に（特定構成）
- カスタムカーネルモジュールなしでデバイス間 P2P DMA が可能

**メモリ階層への影響:**

```
最適化前:                         最適化後:
NVMe SSD                        NVMe SSD
    ↓                                ↓（GPUDirect Storage）
CPU メモリ（バウンスバッファ）         GPU メモリ（HBM）← 直接転送
    ↓
GPU メモリ（HBM）
```

→ **NVMe → CPU メモリ → GPU メモリ** を **NVMe → GPU メモリ** に短縮

#### 4.4 GPUDirect P2P: GPU → GPU 直接転送

**概要:**
- 同一ノード内の複数 GPU 間でメモリを直接転送（PCIe / NVLink 経由）
- CPU やシステムメモリを経由しない

**NVLink 世代別帯域幅:**

| 世代 | GPU | 帯域幅/GPU | 備考 |
|------|-----|-----------|------|
| NVLink 3.0 | A100 | 600 GB/s | 12 リンク |
| NVLink 4.0 | H100 | 900 GB/s | 18 リンク |
| NVLink 5.0 | B200 | 1,800 GB/s | NVLink Switch System |

**メモリ階層への影響:**

```
最適化前:                         最適化後:
GPU 0 メモリ                     GPU 0 メモリ
    ↓                                ↓（GPUDirect P2P / NVLink）
CPU メモリ（中継）                   GPU 1 メモリ ← 直接転送
    ↓
GPU 1 メモリ
```

→ マルチ GPU トレーニング/推論で **GPU 間の HBM を仮想的に統合したメモリプール** として活用可能

#### 4.5 GPUDirect テクノロジーのメモリ階層最適化マップ

```
┌─────────────────────────────────────────────────────────┐
│                   メモリ階層                              │
├────────────┬────────────────────────────────────────────┤
│ GPU 内     │ レジスタ → L1/共有メモリ → L2 → HBM        │
│            │ （GPU 内部最適化は CUDA カーネルの責務）      │
├────────────┼────────────────────────────────────────────┤
│ GPU ↔ GPU  │ GPUDirect P2P / NVLink                     │
│            │ → GPU 間の HBM 直接アクセス                  │
│            │ → CPU メモリ中継を排除                       │
├────────────┼────────────────────────────────────────────┤
│ GPU ↔ NIC  │ GPUDirect RDMA                             │
│            │ → ネットワーク ↔ GPU メモリ直接転送          │
│            │ → CPU バウンスバッファを排除                  │
├────────────┼────────────────────────────────────────────┤
│ GPU ↔ SSD  │ GPUDirect Storage                          │
│            │ → NVMe ↔ GPU メモリ直接転送                 │
│            │ → CPU バウンスバッファを排除                  │
├────────────┼────────────────────────────────────────────┤
│ 統合効果    │ CPU メモリ（DDR/LPDDR）を迂回し、            │
│            │ GPU メモリ（HBM）を中心とした                 │
│            │ フラットなデータアクセスを実現                │
└────────────┴────────────────────────────────────────────┘
```

**出典:**
- https://developer.nvidia.com/gpudirect
- https://docs.nvidia.com/cuda/gpudirect-rdma/
- https://docs.nvidia.com/gpudirect-storage/configuration-guide/index.html
- https://docs.nvidia.com/gpudirect-storage/design-guide/index.html

---

## 付録: GPU 世代別総合スペック比較表

| 仕様 | A100 SXM | H100 SXM | B200 | R200（Rubin） |
|------|----------|----------|------|---------------|
| **アーキテクチャ** | Ampere | Hopper | Blackwell | Rubin |
| **プロセス** | 7nm | 4nm | 4nm (2 ダイ) | 3nm (2 ダイ) |
| **トランジスタ数** | 54.2B | 80B | 208B | - |
| **TDP** | 400W | 700W | 1,000-1,200W | 2,300W |
| **冷却** | 空冷可 | 空冷可 | 液冷推奨/必須 | 液冷必須 |
| **メモリ** | HBM2E 80GB | HBM3 80GB | HBM3e 192GB | HBM4 288GB |
| **メモリ帯域幅** | 2.0 TB/s | 3.35 TB/s | 8 TB/s | ~20.5 TB/s |
| **L2 キャッシュ** | 40 MB | 50 MB | 126 MB | - |
| **NVLink 帯域幅** | 600 GB/s | 900 GB/s | 1,800 GB/s | - |
| **FP16 Tensor** | 312 TFLOPS | 990 TFLOPS | - | - |
| **FP8 Tensor** | - | 1,979 TFLOPS | 4,500 TFLOPS | ~16 PFLOPS |
| **FP4 Tensor** | - | - | 9,000 TFLOPS | ~50 PFLOPS |
| **ラック構成** | DGX A100 | DGX H100 | GB200 NVL72 | NVL144 / NVL576 |
| **ラック電力** | ~6.5 kW | ~10.2 kW | 120 kW | 120-600 kW |
| **出荷時期** | 2020 | 2022 | 2024 | 2026 H2 |

※ ラック電力は DGX ノード単体ではなくフルラック構成時の値（世代により構成が異なる）

---

## 調査ソース一覧

### 冷却・電力インフラ
- NVIDIA DGX SuperPOD Reference Architecture: https://docs.nvidia.com/dgx-superpod/reference-architecture-scalable-infrastructure-gb200/latest/dgx-superpod-components.html
- NVIDIA GB200 NVL72: https://www.nvidia.com/en-us/data-center/gb200-nvl72/
- NVIDIA DGX GB200 User Guide: https://docs.nvidia.com/dgx/dgxgb200-user-guide/hardware.html
- Introl Blog - High-Density Racks: https://introl.com/blog/high-density-racks-100kw-ai-data-center-ocp-2025
- Introl Blog - Vera Rubin 600kW Racks: https://introl.com/blog/nvidia-vera-rubin-gpu-600kw-racks-2027
- Tom's Hardware - Rubin Ultra: https://www.tomshardware.com/pc-components/gpus/nvidia-shows-off-rubin-ultra-with-600-000-watt-kyber-racks-and-infrastructure-coming-in-2027
- Tweaktown - Rubin Ultra Cooling: https://www.tweaktown.com/news/108068/nvidia-could-change-cooling-solution-for-rubin-ultra-ai-gpus-for-huge-2300w-thermal-concerns/index.html
- Google Data Center Efficiency: https://datacenters.google/efficiency/
- Fortune - Santa Clara Power Crisis: https://fortune.com/2025/11/10/nvidia-hometown-santa-clara-california-data-centers-empty-power-grid/
- Boyd Corporation GB200 NVL72: https://www.boydcorp.com/blog/cooling-nvidia-gb200-nvl72-artificial-intelligence.html
- Introl Blog - Liquid vs Air Cooling: https://introl.com/blog/liquid-vs-air-cooling-ai-data-centers
- Profile IT Solutions - TCO Comparison: https://www.profileits.com/understanding-the-total-cost-of-ownership-tco-for-a-10-mw-ai-data-center-air-cooling-vs-immersion-cooling/
- WEF - Data Centre Heating: https://www.weforum.org/stories/2025/06/sustainable-data-centre-heating/
- Introl Blog - Nuclear Power AI DC: https://introl.com/blog/nuclear-power-ai-data-centers-microsoft-google-amazon-2025
- IEEE Spectrum - Nuclear Data Center: https://spectrum.ieee.org/nuclear-powered-data-center

### メモリ階層
- NVIDIA Blackwell Tuning Guide: https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html
- Blackwell Microbenchmarks (arXiv): https://arxiv.org/pdf/2507.10789
- Blackwell Architecture Microbenchmarking (arXiv): https://arxiv.org/html/2512.02189v1
- GPU Cache Hierarchy: https://charlesgrassi.dev/blog/gpu-cache-hierarchy/
- JEDEC HBM4 Standard: https://www.jedec.org/news/pressreleases/jedec%C2%AE-and-industry-leaders-collaborate-release-jesd270-4-hbm4-standard-advancing
- SK Hynix HBM4: https://news.skhynix.com/sk-hynix-completes-worlds-first-hbm4-development-and-readies-mass-production/
- Tom's Hardware - HBM Roadmaps: https://www.tomshardware.com/tech-industry/semiconductors/hbm-roadmaps-for-micron-samsung-and-sk-hynix-to-hbm4-and-beyond
- EE Times - HBM4 at CES 2026: https://www.eetimes.com/the-state-of-hbm4-chronicled-at-ces-2026/
- Tom's Hardware - Vera Rubin: https://www.tomshardware.com/tech-industry/artificial-intelligence/nvidia-delivers-first-vera-rubin-ai-gpu-samples-to-customers-88-core-vera-cpu-paired-with-rubin-gpus-with-288-gb-of-hbm4-memory-apiece
- LLM Inference Roofline (arXiv): https://arxiv.org/html/2402.16363v4
- LLM Inference Characterization (arXiv): https://arxiv.org/html/2512.01644v1
- Mind the Memory Gap (arXiv): https://arxiv.org/html/2503.08311v2
- NVIDIA GPUDirect: https://developer.nvidia.com/gpudirect
- NVIDIA GPUDirect RDMA: https://docs.nvidia.com/cuda/gpudirect-rdma/
- NVIDIA GPUDirect Storage: https://docs.nvidia.com/gpudirect-storage/design-guide/index.html
- Exxact Blog - GPU Comparison: https://www.exxactcorp.com/blog/hpc/comparing-nvidia-tensor-core-gpus
