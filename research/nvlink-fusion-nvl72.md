# NVLink Fusion と Vera Rubin NVL72 アーキテクチャ 徹底調査

調査日: 2026-02-28

---

## 目次

1. [NVLink Fusion](#1-nvlink-fusion)
   - [1.1 物理層アーキテクチャ](#11-物理層アーキテクチャ)
   - [1.2 パートナーエコシステム](#12-パートナーエコシステム)
   - [1.3 ビジネスモデル](#13-ビジネスモデル)
   - [1.4 技術的制約](#14-技術的制約)
2. [Vera Rubin NVL72](#2-vera-rubin-nvl72)
   - [2.1 物理アーキテクチャ](#21-物理アーキテクチャ)
   - [2.2 性能スペック](#22-性能スペック)
   - [2.3 Blackwell NVL72 との比較](#23-blackwell-nvl72-との比較)
   - [2.4 DGX SuperPOD との関係](#24-dgx-superpod-との関係)

---

## 1. NVLink Fusion

NVLink Fusion は、NVIDIA が Computex 2025（2025年5月）で発表した技術プログラムで、サードパーティのカスタム CPU / XPU（アクセラレータ）を NVIDIA の NVLink スケールアップインターコネクトおよび OCP MGX ラックスケールサーバーアーキテクチャに統合することを可能にする。

### 1.1 物理層アーキテクチャ

#### 1.1.1 NVLink-C2C IP の詳細

NVLink-C2C（Chip-to-Chip）は、NVLink 技術をチップ間インターコネクトに拡張したもので、元々は Grace Superchip や Grace Hopper Superchip 内のプロセッサダイ間接続に使用されていた技術である。

**主要スペック:**

| 項目 | 値 |
|------|-----|
| コヒーレント帯域幅 | 900 GB/s 以上 |
| PCIe Gen 5 比エネルギー効率 | 25倍 |
| PCIe Gen 5 比面積効率 | 90倍 |
| PCIe レーン比帯域幅 | 14倍（PCIe 標準レーンの14倍） |
| 信号方式 | PCIe シグナリング上に独自プロトコル |
| サポートプロトコル | Arm AMBA CHI / CXL |

**Chiplet 技術:**

NVLink Fusion の中核には **NVLink Fusion チップレット**がある。これは NVIDIA が設計した NVLink 5/6 対応のチップレットで、サードパーティのカスタムアクセラレータ設計に統合（ドロップイン）することで NVLink 機能を付与する。ハイパースケーラーは、このチップレットを自社カスタム ASIC 設計に組み込み、NVLink スケールアップインターコネクトおよび NVLink Switch に接続できる。

ただし、NVIDIA はチップレット自体のハードウェア仕様（サイズ、消費電力、NVLink レーン数など）を公開していない。

**UCIe との関係:**

NVLink Fusion では、サードパーティチップの接続方式として2つのパスが提供される:

1. **NVLink-C2C IP 統合（推奨）**: カスタム CPU の場合、NVIDIA NVLink-C2C IP を直接統合する方式。最適な低レイテンシ・高帯域幅・高電力効率を実現。
2. **UCIe ブリッジチップレット**: カスタム XPU（アクセラレータ）の場合、Universal Chiplet Interconnect Express (UCIe) IP を使用し、NVIDIA 提供の UCIe-to-NVLink ブリッジチップレットを介して接続する方式。UCIe はオープンスタンダードであり、現行・将来プラットフォームへの柔軟性を提供。

NVLink-C2C は UCIe と比較して、低レイテンシ・高帯域幅・高電力効率に最適化されている（NVIDIA 独自技術）。UCIe はオープンスタンダードとして相互運用性の柔軟性を提供する。

#### 1.1.2 NVLink Fusion Switch チップの仕様

NVLink Fusion では、Vera Rubin NVL72 用の第6世代 NVLink Switch トレイが提供される。

**NVLink 6 Switch チップ仕様:**

| 項目 | 値 |
|------|-----|
| Switch チップ帯域幅 | 28.8 TB/s（NVLink 5 Switch と同値） |
| ポート数 | NVLink 5 Switch の半数 |
| SerDes レート | 400G 双方向 SerDes（NVLink 5 の2倍速） |
| GPU あたり帯域幅 | 3.6 TB/s（全対全） |
| 最大 XPU 接続数 | 72 台（全対全） |
| 合計スケールアップ帯域幅 | 260 TB/s |
| インネットワーク演算 | SHARP FP8 対応（4倍帯域幅効率） |
| SHARP 演算性能 | 3.6 TFLOP |

NVL72 ラック内では、9 基の NVSwitch 6 ブレード（各ブレードに4つの Switch チップ = 合計36チップ）が全 GPU を全対全の非ブロッキングファブリックで接続する。

#### 1.1.3 サードパーティチップの接続要件

**接続方式（2パス）:**

- **カスタム CPU**: NVLink-C2C IP を直接統合 → NVIDIA GPU とのコヒーレント接続
- **カスタム XPU/アクセラレータ**: NVLink Fusion チップレットをパッケージ内に co-package → UCIe 経由で NVLink ファブリックに接続

**NVIDIA CPU/GPU とのペアリング条件:**

- NVLink Fusion は、カスタムシリコンを NVIDIA GPU と同一 NVLink ドメイン内でペアリング可能にする
- カスタム CPU は NVIDIA GPU の CPU パートナーとして機能（Grace CPU の代替）
- カスタム XPU は NVIDIA GPU と並列に NVLink Switch に接続
- 最大72 XPU の全対全接続（NVL72 ドメイン）
- Arm AMBA CHI C2C プロトコルによるコヒーレンシ確保が必須

**設計サービスパートナー:**

| パートナー | 役割 |
|-----------|------|
| MediaTek | カスタム ASIC 設計、高速インターコネクト IP |
| Marvell | カスタムクラウドプラットフォームシリコン設計 |
| Alchip | カスタム ASIC 設計サービス |
| Astera Labs | IP ブロック・接続ソリューション |
| Synopsys | EDA・IP・設計サービス |
| Cadence | EDA・IP・設計サービス |
| Samsung Foundry | 製造（NVLink Fusion プログラム参加） |
| GUC (Global Unichip) | ASIC 設計サービス |

### 1.2 パートナーエコシステム

#### 1.2.1 MediaTek

- **役割**: カスタム AI ASIC の設計・製造パートナー
- **提供内容**: ASIC 設計サービス + 高速インターコネクト IP ポートフォリオ（SerDes、光学、高速 I/O、die-to-die インターコネクト、メモリ技術）
- **NVLink Fusion での位置付け**: 最初期の採用パートナーの一つ。ハイパースケーラー向けにカスタム AI シリコンを NVLink エコシステム内で設計・提供
- **技術的詳細**: Computex 2025 で MediaTek 副会長兼CEO Rick Tsai がパートナーシップの詳細を発表
- **製造ノード**: 2nm チップのテープアウトを2025年9月に予定（スマートフォン / NVLink カスタム ASIC 向け）
- **ステータス**: 設計サービス・ソリューション提供開始済み

#### 1.2.2 Marvell

- **役割**: カスタムクラウドプラットフォームシリコン + NVLink Fusion 統合
- **提供内容**: ハイパースケーラー向けカスタムシリコンに NVLink Fusion を組み込んだソリューション
- **詳細**: Marvell のカスタムシリコンに NVLink Fusion ポートを統合し、モデルトレーニングおよびエージェント型 AI 推論の要求を満たすカスタムスケールアップソリューションを提供
- **対象**: 兆パラメータ AI モデル向けの帯域幅・信頼性・俊敏性を持つ次世代 AI インフラ基盤
- **ステータス**: 設計サービス・ソリューション提供開始済み

#### 1.2.3 AWS Trainium4

- **統合方式**: Trainium4 に NVLink 6 を統合し、NVIDIA MGX ラックアーキテクチャと組み合わせ
- **世代間連携**: NVIDIA と AWS の NVLink Fusion に関する**マルチジェネレーション・コラボレーション**の第一弾
- **具体的な統合内容**:
  - Trainium4、Graviton CPU、Elastic Fabric Adapter (EFA) が共通の MGX ベースラック内で相互運用
  - GPU サーバーと Trainium システムの両方をホストできる柔軟なラックスケール設計
  - NVLink Fusion により最大72カスタム ASIC を第6世代 NVLink Switch で全対全接続
- **Trainium4 性能（vs Trainium3）**:
  - FP4 スループット: 6倍
  - FP8 性能: 3倍
  - メモリ帯域幅: 4倍
  - HBM4 8スタック搭載（容量2倍、帯域幅4倍）
- **戦略的意義**: AWS は自社カスタムチップで NVIDIA GPU と相互運用可能にしつつ、低コストなサーバーラック技術を活用
- **発表時期**: AWS re:Invent 2025（2025年12月）
- **出荷時期**: 2026年後半（推定）

#### 1.2.4 Fujitsu: FUGAKU NEXT での採用計画

- **プロジェクト名**: FugakuNEXT（理化学研究所「富岳」後継機）
- **CPU**: FUJITSU-MONAKA-X（FUJITSU-MONAKA の後継。2nm、Arm ベース、極限的電力効率を目指す）
- **NVLink Fusion の利用方法**:
  - MONAKA-X CPU に NVLink Fusion チップレットを統合し、NVLink ポートを装備
  - NVSwitch 経由で NVIDIA GPU とメモリ共有（NUMA 的なアクセス）
  - Fujitsu CPU + NVIDIA GPU のヘテロジニアスノードを NVLink ファブリックで接続
- **スケジュール**:
  - 基本設計: 2025年度内完了（～2026年2月27日）
  - 詳細設計: 2026年度～
  - MONAKA チップ出荷: 2027年（予定）
  - MONAKA-X: 2027年以降（予定）
  - FugakuNEXT 運用開始: 2030年頃（理化学研究所神戸キャンパス）
- **予算**: 約7億4000万ドル（約1,100億円）
- **目標性能**: ゼタスケール

#### 1.2.5 Qualcomm

- **対象チップ**: Arm ベースのデータセンター向け CPU（Nuvia アーキテクチャベース）
- **統合方式**: NVLink Fusion ポートを Qualcomm サーバー CPU に統合し、NVIDIA GPU とラックスケールアーキテクチャで接続
- **背景**: Qualcomm は2021年に買収した Nuvia（元 Apple チップエンジニアが設立）の技術を基盤に、サーバー市場に再参入
- **パートナーシップ**: サウジアラビアの AI 企業 Humain とカスタムデータセンター CPU の共同開発で合意
- **発表時期**: Computex 2025
- **ステータス**: 具体的なチップ仕様は「今後数か月以内」に発表予定（2025年5月時点）

#### 1.2.6 その他のパートナー

- **Arm**: Neoverse プラットフォームに AMBA CHI C2C プロトコルの最新版を統合し、NVLink Fusion との C2C 互換性を確保
- **SiFive**: データセンター製品向けに NVLink Fusion をライセンス
- **Samsung Foundry**: NVLink Fusion プログラムに参加。カスタムプロセッサを NVIDIA の独自インターコネクトファブリックに直接接続する半導体製造を提供

### 1.3 ビジネスモデル

#### 1.3.1 ライセンスモデル

NVLink Fusion は**ライセンス方式**で提供される:

- カスタム CPU またはアクセラレータを製造するパートナーは、**NVLink ポート設計**と**メモリアトミックプロトコル**（NVLink 上で動作する NUMA メモリ共有プロトコル）をライセンスする
- NVLink-C2C IP および NVLink チップレットを外部ファウンドリにライセンス
- 具体的なライセンス料は非公開

#### 1.3.2 NVIDIA にとっての戦略的意義

**プラットフォーム化戦略:**

NVLink Fusion により、NVIDIA は「チップサプライヤー」から「プラットフォームプロバイダー」へ変革を図る。

1. **GPU を売らなくてもインフラで収益**: ハイパースケーラーが自社カスタム ASIC を使用する場合でも、NVLink Switch、MGX ラック、ソフトウェアスタックで NVIDIA が収益を得る
2. **ライセンス収益の経常化**: NVLink Fusion からのライセンス収益が NVIDIA の経常収入ストリームを大幅に強化する可能性
3. **エコシステムロックイン**: Google、Microsoft、Amazon 等のクラウド大手がカスタムシリコンを開発するトレンドに対抗するのではなく、そのトレンドに**乗る**ことで全ての「セミカスタム」システムを NVIDIA エコシステム経由にする
4. **UALink への対抗**: 競合技術である UALink（オープン仕様）は複数の有力企業間の利害対立（「共有地の悲劇」）により仕様策定が遅延する傾向がある。NVLink Fusion は既に動作するソリューションを提供し、UALink の市場参入を先制

**「Embrace, Extend, Extinguish」戦略（Fabricated Knowledge 分析）:**

- **Embrace**: NVLink を開放し、競合チップメーカーを NVLink エコシステムに取り込む
- **Extend**: NVIDIA のロードマップを UALink より遥かに高速に進化させる
- **Extinguish**: 競合がNVLink エコシステムに依存した段階で、徐々に代替技術を排除

Jensen Huang の哲学: 差別化の少ない領域ではオープンソース化・エコシステム採用を許容し、NVIDIA が唯一の動作製品を持つ状態で技術をライセンスする。顧客が検討する間に、NVIDIA の製品・計画が常に競合より優れていることを示す。

#### 1.3.3 参考: ネットワーキング収益

Wells Fargo のアナリスト推定では、Spectrum-X 単体の収益が年間100億ドル超のランレート（NVIDIA の GPU 事業より高い成長率）に達している。NVLink Fusion はこのネットワーキング収益をさらに拡大する手段となる。

### 1.4 技術的制約

#### 1.4.1 NVLink Fusion vs NVLink ネイティブの性能差

**帯域幅:**

| 接続方式 | GPU あたり帯域幅 | 備考 |
|---------|-----------------|------|
| NVLink 6 ネイティブ（Vera Rubin） | 3.6 TB/s | GPU-GPU 直接接続 |
| NVLink Fusion（NVLink-C2C） | 900 GB/s 以上 | チップ間コヒーレント接続 |
| NVLink Fusion（UCIe ブリッジ） | UCIe 帯域幅依存 | ブリッジチップレット経由 |
| PCIe Gen 5 | 約128 GB/s（x16） | 比較参考値 |

**レイテンシ:**

- NVLink: PCIe 比で最大約10倍低レイテンシ（約2 us vs 約20 us、P2P テスト）
- NVLink-C2C: ネイティブ NVLink と同等の低レイテンシを目指す設計
- NVLink Fusion（UCIe ブリッジ）: UCIe-to-NVLink 変換によるレイテンシ追加の可能性（具体的数値は非公開）

**性能差の概要:**

- NVLink Fusion は、Grace Hopper / Grace Blackwell のコデザインモデルをサードパーティに拡張したもの
- NVLink-C2C 経由の Fusion 接続は、ネイティブ NVLink と同等のコヒーレンシ・帯域幅特性を維持
- UCIe ブリッジ経由の場合、変換オーバーヘッドにより若干の性能低下が想定されるが、NVIDIA は「最高性能と統合容易性」を提供すると表明
- 具体的な性能差の数値は NVIDIA 未公開

#### 1.4.2 対応メモリモデル（コヒーレンシ）

- **フルキャッシュコヒーレンシ**: NVLink-C2C は GPU/CPU 間の完全なキャッシュコヒーレンシをサポート
- **NUMA メモリ共有**: CPU とアクセラレータ間で NUMA スタイルのメモリ共有（数十年の CPU 技術と同等）
- **アトミック操作**: 共有データへの高速同期・高頻度更新のためのアトミック操作をサポート
- **対応プロトコル**: Arm AMBA CHI、CXL（業界標準プロトコルとの相互運用）
- **メモリセマンティクアクセス**: デバイス間でメモリセマンティックアクセスを保持

#### 1.4.3 サポートされるトポロジ

| トポロジ | 詳細 |
|---------|------|
| NVL72 全対全 | 最大72 XPU を NVLink Switch で全対全接続（3.6 TB/s/XPU、260 TB/s 合計） |
| ヘテロジニアスラック | カスタム CPU + NVIDIA GPU、またはカスタム XPU + NVIDIA GPU の混在構成 |
| MGX ラックスケール | OCP MGX ラックアーキテクチャ準拠のラックスケール構成 |
| GPU + カスタム ASIC 混在 | GPU サーバーとカスタム ASIC（例: Trainium4）サーバーの同一ラック内共存 |

**制約:**

- NVLink Fusion は NVLink Switch ベースのラックスケール構成に限定（ポイントツーポイント接続のみでは NVLink Fusion の価値が限定的）
- カスタムシリコンの最大接続数は72 XPU（NVL72 ドメイン上限）
- NVIDIA NVLink Switch チップの使用が必須（サードパーティ Switch は不可）

---

## 2. Vera Rubin NVL72

NVIDIA が CES 2026（2026年1月）で正式発表した次世代 AI プラットフォーム。第3世代ラックスケールアーキテクチャで、6種のコデザインチップを単一統合システムに構成する。

### 2.1 物理アーキテクチャ

#### 2.1.1 72 Rubin GPU + 36 Vera CPU の物理配置

**ラック構成:**

| コンポーネント | 数量 | 詳細 |
|---------------|------|------|
| コンピュートブレード | 18基 | 各ブレードに GPU 4基 + CPU 2基 |
| NVSwitch 6 ブレード | 9基 | 各ブレードに NVLink 6 Switch チップ4基 |
| Rubin GPU (R200) | 72基 | 2ダイ構成、TSMC 3nm クラス |
| Vera CPU | 36基 | 88コア、Arm Olympus カスタムコア |
| BlueField-4 DPU | 18基 | データ処理ユニット |
| ConnectX-9 SuperNIC | 搭載 | ネットワーク接続 |

**6種のコデザインチップ:**

1. Vera CPU
2. Rubin GPU (R200)
3. NVLink 6 Switch
4. ConnectX-9 SuperNIC
5. BlueField-4 DPU
6. Spectrum-6 Ethernet Switch

**物理設計の特徴:**

- **ケーブルレス設計**: モジュラー式、ケーブル不要、ファン不要、ホース不要のトレイ設計
- **ブラインドメイトコネクタ**: コンピュートトレイはラックに挿入するだけで接続完了
- **ミッドプレーン導入**: VR200 NVL72 コンピュートトレイは初めてミッドプレーンを採用。仕様: 44層（22+22）、M9 CCL (EM896K3)、サイズ約 420 x 60 mm
- **組み立て時間**: 従来（GB300 NVL72）の約100分 → **18倍短縮**
- **ゼロダウンタイムメンテナンス**: NVLink Switch トレイは稼働中に取り外し・部分装着可能

#### 2.1.2 NVLink Spine の構造

**NVLink 6 Switch 構成:**

| 項目 | 値 |
|------|-----|
| Switch ブレード数 | 9基 |
| Switch チップ/ブレード | 4基 |
| Switch チップ総数 | 36基（Blackwell NVL72 の2倍） |
| チップあたり帯域幅 | 28.8 TB/s |
| GPU あたり帯域幅 | 3.6 TB/s（Blackwell の2倍） |
| ラック合計帯域幅 | 260 TB/s |
| SerDes | 400G 双方向カスタム SerDes |
| トポロジ | 全対全、非ブロッキング |
| インネットワーク演算 | SHARP FP8（3.6 TFLOP） |

Switch チップのレイアウトは NVIDIA 従来世代と同一構造（2面 IO + 中央ロジックセクションクロスバー）を維持。ポート数は NVLink 5 Switch の半分だが、SerDes レートが2倍（400G）のため帯域幅は同等。

#### 2.1.3 ラックサイズ、重量、電力要件

**電力要件:**

| 項目 | 値 |
|------|-----|
| Rubin GPU TDP（Max Q） | 約 1,800W/GPU |
| Rubin GPU TDP（Max P） | 約 2,300W/GPU |
| ラック電力（Max Q） | 約 190 kW |
| ラック電力（Max P / 2300W SKU） | 最大 220 kW |
| 電源構成 | 4 x 110 kW 電源シェルフ（N+1 冗長） |
| 電源シェルフ内 PSU | 6 x 18.3 kW PSU/シェルフ |

**冷却:**

| 項目 | 詳細 |
|------|------|
| 冷却方式 | 完全液冷（液冷必須） |
| 冷却液温度 | 45°C（エネルギー効率最適化） |
| 液冷バスバー | 新設計（より高性能） |
| CDU 圧力ヘッド | Blackwell と同等（増加なし） |
| 熱性能 | Blackwell 比で**約2倍**（追加冷却コスト・複雑性なし） |
| エネルギー貯蔵 | Blackwell 比 20倍（安定電力供給） |

**物理サイズ:**

- 第3世代 NVIDIA MGX NVL72 ラック設計
- 具体的なラック寸法・重量は NVIDIA 未公開（標準19インチラック準拠と推定）

### 2.2 性能スペック

#### 2.2.1 演算性能

**Rubin GPU (R200) 単体性能:**

| 精度 | 性能 | Blackwell (GB300) 比 |
|------|------|---------------------|
| NVFP4 推論 | 50 PFLOPS | 5倍 |
| NVFP4 トレーニング | 35 PFLOPS | 3.5倍 |
| FP8 | 約 16 PFLOPS | — |
| BF16 | Blackwell と同等水準 | 約 1.6倍 |
| FP64（倍精度） | 33 TFLOPS | 減少（FP64 ALU を FP4/FP6 ALU に置換） |

- 224 SM（Streaming Multiprocessor）搭載
- 第5世代 Tensor Core（低精度 NVFP4 / FP8 に最適化）
- 第3世代 Transformer Engine

**Vera Rubin NVL72 ラック性能:**

| 精度 | 性能 |
|------|------|
| NVFP4 推論 | 3.6 EFLOPS |
| NVFP4 トレーニング | 2.5 EFLOPS |

#### 2.2.2 メモリ

**Rubin GPU (R200) メモリ:**

| 項目 | 値 | Blackwell 比 |
|------|-----|-------------|
| メモリタイプ | HBM4 | HBM3e → HBM4 |
| 容量/GPU | 288 GB | 同等 |
| HBM スタック数 | 8スタック | — |
| HBM レート | 6.4 GT/s → 10.8 GT/s | — |
| 帯域幅/GPU | 約 22 TB/s | 2.8倍（8 TB/s → 22 TB/s） |
| バス幅 | 2倍（HBM4 によりスタックあたり2倍） | — |
| 製造プロセス | TSMC 3nm クラス（2ダイ + 2 I/O ダイ） | — |

**Vera CPU メモリ:**

| 項目 | 値 |
|------|-----|
| メモリタイプ | LPDDR5x (SOCAMM) |
| 容量/CPU | 最大 1.5 TB |
| 帯域幅/CPU | 最大 1.2 TB/s |

**Vera Rubin NVL72 ラック メモリ合計:**

| 項目 | 値 |
|------|-----|
| HBM4 総容量 | 20.7 TB |
| LPDDR5x 総容量 | 54 TB |
| HBM 総帯域幅 | 1.6 PB/s |

#### 2.2.3 Scale-up 帯域幅（260 TB/s の内訳）

NVLink 6 による 260 TB/s の内訳:

| 項目 | 値 | 計算根拠 |
|------|-----|---------|
| GPU あたり NVLink 帯域幅 | 3.6 TB/s | — |
| GPU 数 | 72基 | — |
| 合計（単純計算） | 259.2 TB/s ≈ 260 TB/s | 72 x 3.6 = 259.2 |
| Switch チップ帯域幅 | 28.8 TB/s x 9ブレード | — |
| Blackwell NVL72 比 | 2倍 | 130 TB/s → 260 TB/s |

**帯域幅の意義:**

- ラック内でのモデルパーティショニングが不要になるレベルの NVLink 帯域幅
- MoE（Mixture of Experts）モデルやロングコンテキストワークロードに最適

#### 2.2.4 Vera CPU 仕様

| 項目 | 値 |
|------|-----|
| トランジスタ数 | 2,270億個（227 billion） |
| コアアーキテクチャ | NVIDIA カスタム Arm "Olympus" コア |
| コア数 | 88コア |
| スレッド数 | 176スレッド（Spatial Multi-Threading） |
| ISA | Armv9.2 完全互換 |
| NVLink 接続 | NVLink-C2C による超高速接続 |
| メモリ | 最大 1.5 TB LPDDR5x (SOCAMM) |
| メモリ帯域幅 | 最大 1.2 TB/s |

### 2.3 Blackwell NVL72 との比較

#### 2.3.1 主要スペック比較表

| 項目 | GB300 NVL72 (Blackwell Ultra) | Vera Rubin NVL72 | 向上率 |
|------|-------------------------------|-------------------|--------|
| **GPU** | B300 (Blackwell Ultra) | R200 (Rubin) | — |
| **CPU** | Grace | Vera (Olympus) | — |
| **GPU TDP** | 約 1,400W | 約 1,800～2,300W | — |
| **ラック電力** | 約 120 kW | 約 190～220 kW | — |
| **GPU メモリ** | 288 GB HBM3e | 288 GB HBM4 | 同等容量 |
| **メモリ帯域幅/GPU** | 約 8 TB/s | 約 22 TB/s | **2.8倍** |
| **NVLink 帯域幅/GPU** | 1.8 TB/s (NVLink 5) | 3.6 TB/s (NVLink 6) | **2倍** |
| **NVLink 合計帯域幅** | 130 TB/s | 260 TB/s | **2倍** |
| **NVSwitch チップ数** | 18基 | 36基 | 2倍 |
| **FP4 推論/GPU** | 10 PFLOPS | 50 PFLOPS | **5倍** |
| **FP4 トレーニング/GPU** | 10 PFLOPS | 35 PFLOPS | **3.5倍** |
| **FP4 推論/ラック** | 約 0.72 EFLOPS (GB200) / 1.1 EFLOPS (GB300) | 3.6 EFLOPS | — |
| **HBM 総容量/ラック** | 約 37 TB | 20.7 TB | メモリタイプ差 |
| **HBM 総帯域幅/ラック** | — | 1.6 PB/s | — |

#### 2.3.2 性能向上率（NVIDIA 公称）

**トレーニング:**

- MoE モデルトレーニング: Blackwell の **1/4 の GPU 数**で同等性能
- FP4 トレーニング性能/GPU: **3.5倍**

**推論:**

- FP4 推論性能/GPU: **5倍**
- 推論コスト（100万トークンあたり）: **1/10**
- 高度にインタラクティブなディープリーズニング・エージェント型 AI に最適化

#### 2.3.3 電力効率の改善

- **性能/ワット**: Blackwell 比 **最大10倍**（NVIDIA 公称）
- 具体的な計算:
  - GB300: 1,400W で約 10-15 PFLOPS FP4 → 約 7-10 TFLOPS/W
  - R200 (Max Q): 1,800W で 50 PFLOPS FP4 → 約 28 TFLOPS/W
  - → FP4 推論の TFLOPS/W は約3-4倍改善
- **冷却効率**: 液冷流量増加で熱性能2倍（CDU 圧力増加なし・追加コストなし）
- **製造プロセス**: TSMC 3nm（Blackwell の4nm からシュリンク）

### 2.4 DGX SuperPOD との関係

#### 2.4.1 DGX SuperPOD with DGX Vera Rubin NVL72（8ラック構成）

一部のソースでは8ラック構成の SuperPOD が記載されている:

| 項目 | 値 |
|------|-----|
| NVL72 ラック数 | 8 |
| GPU 総数 | 576 |
| CPU 総数 | 288 |
| メモリ総容量 | 約 600 TB |
| NVFP4 推論性能 | 28.8 EFLOPS |

#### 2.4.2 DGX SuperPOD with DGX Vera Rubin NVL72（14ラック構成、フル構成）

NVIDIA 公式の DGX SuperPOD フル構成:

| 項目 | 値 |
|------|-----|
| DGX Vera Rubin NVL72 数 | 14システム |
| Rubin GPU 総数 | 1,008 |
| FP4 演算性能 | **50.4 EFLOPS** |
| 高速メモリ（HBM4）総容量 | **1,046 TB（約 1 PB）** |
| Scale-out ネットワーク | Spectrum-6 Ethernet / ConnectX-9 SuperNIC |

#### 2.4.3 576 GPU / 1 PB/s の詳細

「576 GPU / 1 PB/s」は以下の文脈で言及される:

- **576 GPU**: 8ラック構成（8 x 72 = 576 GPU）の SuperPOD バリアント。28.8 EFLOPS の NVFP4 性能
- **1 PB/s（HBM 帯域幅）**: 単一 Vera Rubin NVL72 ラック内の HBM4 総帯域幅が 1.6 PB/s。これはラック単位の HBM メモリ帯域幅を指す

**SuperPOD のスケールアウト:**

- 各 Vera Rubin NVL72 ラック内は NVLink 6 でスケールアップ（260 TB/s）
- ラック間は Spectrum-6 Ethernet Switch + ConnectX-9 SuperNIC + BlueField-4 DPU でスケールアウト
- ラック内でモデルパーティショニングが不要なため、スケールアウトの複雑性が大幅低減

#### 2.4.4 可用性

| モデル | 出荷時期 |
|-------|---------|
| DGX Vera Rubin NVL72 | 2026年下半期 |
| DGX Rubin NVL8 (HGX) | 2026年下半期 |
| DGX SuperPOD with DGX Vera Rubin NVL72 | 2026年下半期 |

---

## ソース一覧

### NVLink Fusion

- https://www.nvidia.com/en-us/data-center/nvlink-fusion/
- https://www.nvidia.com/en-us/data-center/nvlink-c2c/
- https://nvidianews.nvidia.com/news/nvidia-opens-nvlink-for-custom-silicon-integration
- https://nvidianews.nvidia.com/news/nvidia-nvlink-fusion-semi-custom-ai-infrastructure-partner-ecosystem
- https://developer.nvidia.com/blog/scaling-ai-inference-performance-and-flexibility-with-nvidia-nvlink-and-nvlink-fusion/
- https://developer.nvidia.com/blog/aws-integrates-ai-infrastructure-with-nvidia-nvlink-fusion-for-trainium4-deployment/
- https://developer.nvidia.com/blog/integrating-custom-compute-into-rack-scale-architecture-with-nvidia-nvlink-fusion/
- https://www.servethehome.com/nvidia-announces-nvlink-fusion-bringing-nvlink-to-third-party-cpus-and-accelerators/
- https://www.servethehome.com/nvidia-nvlink-fusion-tapped-for-future-aws-trainium4-deployments/
- https://www.nextplatform.com/2025/05/19/nvidia-licenses-nvlink-memory-ports-to-cpu-and-accelerator-makers/
- https://www.theregister.com/2025/05/19/nvidia_nvlink_fusion/
- https://www.theregister.com/2025/12/02/amazon_nvidia_trainium/
- https://www.fabricatedknowledge.com/p/nvlink-fusion-embrace-extend-extinguish
- https://www.tomshardware.com/pc-components/cpus/nvidia-announces-nvlink-fusion-to-allow-custom-cpus-and-ai-accelerators-to-work-with-its-products
- https://www.mediatek.com/tek-talk-blogs/mediateks-nvlink-fusion-partnership-pioneering-ai-innovation-with-custom-asic
- https://www.marvell.com/company/newsroom/marvell-nvidia-provide-to-custom-solutions-advanced-ai-infrastructure.html
- https://newsroom.arm.com/news/arm-neoverse-nvidia-nvlink
- https://www.guc-asic.com/en/news/PressRelease/PR_ENG_20250903
- https://hoti.org/assets/slides/2025_08_21_day2_Invited_talk_NVIDIA_NVLink_Fusion.pdf
- https://www.datacenterdynamics.com/en/news/qualcomm-announces-data-center-cpus-will-support-nvidias-nvlink-fusion/

### Fujitsu FugakuNEXT

- https://www.fujitsu.com/global/about/resources/news/press-releases/2025/0618-01.html
- https://blogs.nvidia.com/blog/fugakunext/
- https://www.riken.jp/en/news_pubs/news/2025/20250822_1/index.html
- https://www.nextplatform.com/2025/08/22/nvidia-tapped-to-accelerate-rikens-fugakunext-supercomputer/
- https://www.tomshardware.com/tech-industry/supercomputers/nvidia-gpus-and-fujitsu-arm-cpus-will-power-japans-next-usd750m-zetta-scale-supercomputer-fugakunext-aims-to-revolutionize-ai-driven-science-and-global-research

### Vera Rubin NVL72

- https://www.nvidia.com/en-us/data-center/vera-rubin-nvl72/
- https://www.nvidia.com/en-us/data-center/dgx-vera-rubin-nvl72/
- https://nvidianews.nvidia.com/news/rubin-platform-ai-supercomputer
- https://developer.nvidia.com/blog/inside-the-nvidia-rubin-platform-six-new-chips-one-ai-supercomputer
- https://blogs.nvidia.com/blog/dgx-superpod-rubin/
- https://videocardz.com/newz/nvidia-vera-rubin-nvl72-detailed-72-gpus-36-cpus-260-tb-s-scale-up-bandwidth
- https://www.storagereview.com/news/nvidia-launches-vera-rubin-architecture-at-ces-2026-the-vr-nvl72-rack
- https://www.theregister.com/2026/01/05/ces_rubin_nvidia/
- https://www.nextplatform.com/ai/2026/01/06/nvidias-vera-rubin-platform-obsoletes-current-ai-iron-six-months-ahead-of-launch/4092179
- https://www.tomshardware.com/pc-components/gpus/nvidia-launches-vera-rubin-nvl72-ai-supercomputer-at-ces-promises-up-to-5x-greater-inference-performance-and-10x-lower-cost-per-token-than-blackwell-coming-2h-2026
- https://www.tomshardware.com/pc-components/gpus/nvidias-vera-rubin-platform-in-depth-inside-nvidias-most-complex-ai-and-hpc-platform-to-date
- https://www.glennklockwood.com/garden/processors/R200
- https://newsletter.semianalysis.com/p/vera-rubin-extreme-co-design-an-evolution
- https://www.hpcwire.com/2026/01/05/nvidia-says-rubin-will-deliver-5x-ai-inference-boost-over-blackwell/
- https://www.techradar.com/pro/the-battle-of-the-superpods-nvidia-challenges-huawei-with-vera-rubin-powered-dgx-cluster-that-can-deliver-28-8-exaflops-with-only-576-gpus
- https://www.wheelersnetwork.com/2025/11/decoding-nvidias-rubin-networking-math.html
- https://en.wikipedia.org/wiki/NVLink
- https://en.wikipedia.org/wiki/Rubin_(microarchitecture)
