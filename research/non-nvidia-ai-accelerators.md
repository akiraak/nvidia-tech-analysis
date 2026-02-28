# NVIDIA 以外の AI アクセラレータによるデータセンター構成

調査日: 2026-02-28

---

## 1. Intel Gaudi

### 1.1 世代別スペック

#### Gaudi 2

| 項目 | スペック |
|------|---------|
| プロセス | 7nm |
| 演算エンジン | MME x2 + TPC x24 |
| BF16 演算性能 | 432 TFLOPS (MME) + 11 TFLOPS (TPC) |
| HBM | 96 GB HBM2E |
| メモリ帯域幅 | 2.45 TB/s |
| SRAM | 48 MB |
| ネットワーク | 24 x 100 GbE RoCEv2 RDMA NIC（合計 2.4 Tbps） |

- MME (Matrix Multiplication Engine): 行列演算（全結合層、畳み込み、バッチ GEMM）を担当
- TPC (Tensor Processor Core): MME 以外の全演算を処理する完全プログラマブルコア
- NVIDIA A100-80GB と比較して、メモリ容量（96GB vs 80GB）、メモリ帯域幅で優位

#### Gaudi 3

| 項目 | スペック |
|------|---------|
| プロセス | 5nm（2ダイ構成） |
| 演算エンジン | MME x8 + TPC x64 |
| FP8/BF16 演算性能 | 1.8 PFLOPS |
| HBM | 128 GB HBM2E（16GB x 8スタック） |
| メモリ帯域幅 | 3.7 TB/s（Gaudi 2比 1.5倍） |
| ネットワーク | 24 x 200 GbE RoCEv2 RDMA NIC（合計 4.8 Tbps） |
| TDP | 最大 900W（空冷） |

- Gaudi 2 からメモリ容量 33% 増、帯域幅 1.5倍
- ネットワーク帯域幅は Gaudi 2 の 2倍（100GbE → 200GbE）
- Intel は Llama 3 80B の推論スループットにおいて NVIDIA H100 比 70% 優れた価格性能比を主張

#### Falcon Shores（キャンセル済み）

- 元々 2025年後半に Gaudi 3 の後継として投入予定だった
- **2025年1月に市場投入のキャンセルが発表**された
- 業界フィードバックに基づき、内部テストチップとして活用する方針に変更
- 後継は **Jaguar Shores**（2026年以降）: ラックスケールシステムとして、コンピュート、ネットワーク、メモリを統合インフラストラクチャに統合する新パラダイム

### 1.2 インターコネクト・ネットワーク

Gaudi の最大の特徴は、**チップ内蔵の RoCEv2 RDMA NIC** によるネットワーキングである。NVIDIA が InfiniBand や NVLink/NVSwitch に依存するのに対し、Gaudi は業界標準の Ethernet をベースとしている。

#### Scale-up（サーバ内接続）

- 8枚の Gaudi 3 OAM カードが基板上でパッシブに直接接続
- **非ブロッキング・All-to-All 構成**: 各カードから他の7枚に対して 3 x 200GbE ポートで接続（計21ポート使用）
- サーバ内合計帯域: **67.2 Tbps（双方向）**
- **サーバ内にネットワークスイッチ不要** — NVSwitch のようなスイッチチップが必要ない
- 残り 3ポート（24 - 21 = 3）が scale-out 用に使用される

#### Scale-out（サーバ間接続）

- 標準 Ethernet スイッチングインフラを使用
- サーバ外部: **6 x 800GbE OSFP** ポート（オープン業界標準）
- **Fat-tree トポロジー**: 32ポート Leaf Switch + 64ポート Spine Switch の2層構成
- 1,024 アクセラレータまで 3:1 オーバーサブスクリプションでスケール可能、コレクティブ操作効率 90% を維持
- InfiniBand 比で**ネットワーキングコスト 60% 削減**を謳う

#### NVIDIA との比較

| 項目 | Intel Gaudi 3 | NVIDIA H100 (DGX) |
|------|-------------|-------------------|
| Scale-up 方式 | 内蔵 RoCE NIC 直接接続 | NVLink + NVSwitch |
| Scale-out 方式 | 標準 Ethernet (RoCEv2) | InfiniBand / RoCE |
| サーバ内スイッチ | 不要 | NVSwitch 必要 |
| サーバ間プロトコル | 標準 Ethernet | InfiniBand (独自) |
| コスト | 低コスト（標準インフラ活用） | 高コスト（独自エコシステム） |

### 1.3 ソフトウェアエコシステム

#### SynapseAI SDK

- Habana Labs（Intel 傘下）が開発する AI SDK
- PyTorch を Eager モードまたは Lazy モードでサポート
- DeepSpeed との統合による LLM トレーニング加速
- Flash Attention 等の最適化オペレーションを提供

#### PyTorch / Hugging Face 対応

- **最小限のコード変更で移行可能**: `tensor.cuda()` → `tensor.to('hpu')` への書き換え程度
- Hugging Face **Optimum-Habana** ライブラリとの統合
  - 数行のコード変更で Transformers / Diffusers モデルを Gaudi 上で実行可能
  - HF Trainer は Gaudi サポートを内蔵
- Hugging Face Hub 上の対応アーキテクチャベースの **40,000以上のモデル** が Gaudi で利用可能

#### CUDA からの移行

- PyTorch ベースのため、理論上は移行が容易
- ただし、NVIDIA の数十年にわたるソフトウェア投資と比較すると、エコシステムの成熟度には差がある
- カスタム CUDA カーネルを多用するワークロードでは移行コストが高い

### 1.4 採用状況と課題

#### 現状の課題

- Intel は AI データセンター市場で**意味のある規模での参入に至っていない**
- Gaudi 3 の売上目標 **5億ドルを達成できず**、ソフトウェア関連の問題が原因とされる
- 主要クラウドプロバイダーの採用は限定的

#### 採用実績

- **IBM Cloud**: Gaudi 3 をクラウドデータセンターに採用した主要顧客
- **VMware Cloud Foundation 9.0** との統合（2025年8月）
- **GIGABYTE / Supermicro** 等の OEM が Gaudi 3 プラットフォームを提供

#### 事業の方向性

- Falcon Shores のキャンセルにより、スタンドアロン AI アクセラレータとしての Gaudi ラインの将来は不透明
- Jaguar Shores（2026年以降）でラックスケールシステムへの方向転換を図る
- Intel の AI アクセラレータ事業は NVIDIA に対して大きく遅れをとっている状況

---

## 2. AWS Trainium / Inferentia

### 2.1 世代別スペック

#### Inferentia2（推論特化）

| 項目 | スペック |
|------|---------|
| FP16 演算性能 | 190 TFLOPS / チップ |
| HBM | 32 GB / チップ（初代比 4倍） |
| メモリ帯域幅 | 9.8 TB/s（インスタンス合計、初代比 10倍） |
| チップ間接続 | NeuronLink（192 GB/s） |
| インスタンス | Inf2: 最大12チップ |
| 対応データ型 | FP32, TF32, BF16, FP16, UINT8, cFP8 |

- 初代 Inferentia 比: スループット 4倍、レイテンシ 1/10
- 同等 EC2 インスタンス比: 50% 優れた電力効率

#### Trainium2（トレーニング + 推論）

| 項目 | スペック |
|------|---------|
| FP8 演算性能 | 1.3 PFLOPS / チップ（スパース: 5.2 PFLOPS） |
| HBM | 96 GB HBM3 / チップ |
| メモリ帯域幅 | 2.9 TB/s / チップ |
| 初代比性能 | 4倍 |
| インスタンス (Trn2) | 16チップ、1.5 TB メモリ、46 TBps 帯域幅 |
| UltraServer (Trn2) | 64チップ、6 TB メモリ、185 TBps 帯域幅 |
| EFA ネットワーク | 3.2 Tbps (EFAv3) |

#### Trainium3（2025年12月発表、3nm プロセス）

| 項目 | スペック |
|------|---------|
| プロセス | 3nm（AWS 初の 3nm AI チップ） |
| FP8 演算性能 | 2.52 PFLOPS / チップ |
| HBM | 144 GB HBM3e / チップ（Trn2 比 1.5倍） |
| メモリ帯域幅 | 4.9 TB/s / チップ（Trn2 比 1.7倍） |
| UltraServer | 144チップ、20.7 TB HBM3e、706 TB/s 帯域幅 |
| UltraServer 演算 | 362 FP8 PFLOPS |
| 対応データ型 | MXFP8, MXFP4 |
| Trn2 比性能 | 4.4倍性能、3.9倍メモリ帯域幅、4倍性能/ワット |

#### Trainium4（計画中）

- FP8 処理能力: Trainium3 の 3倍以上
- メモリ帯域幅: Trainium3 の 4倍
- **NVIDIA NVLink Fusion** チップインターコネクト技術をサポート予定（NVIDIA との協調路線）

### 2.2 インターコネクト技術

#### NeuronLink（チップ間接続 / Scale-up）

AWS 独自のチップ間高速接続技術。世代ごとに進化:

| 世代 | 接続方式 | 特徴 |
|------|---------|------|
| NeuronLink (Inf2/Trn1) | 192 GB/s | 基本的なチップ間接続 |
| NeuronLink (Trn2) | UltraServer 内64チップ接続 | 高帯域チップ間通信 |
| NeuronLinkv4 (Trn3) | PCB + バックプレーン + ラック間 | 3つの接続媒体、160 PCIe レーン/チップ |
| NeuronSwitch-v1 (Trn3) | All-to-All スイッチドファブリック | 2 TB/s/チップ、レイテンシ半減 |

- Trn3 UltraServer: 2ラック構成で 144チップ（18コンピュートトレイ + 10 NeuronLink スイッチトレイ / ラック）
- Trn2 比: チップ密度 2倍以上

#### EFA (Elastic Fabric Adapter)（ノード間接続 / Scale-out）

- AWS 独自の高性能ネットワーキング技術
- UltraServer 間のスケールアウト接続を担当
- Trn2: EFAv3 で 3.2 Tbps
- **EC2 UltraClusters 3.0**: 数十万チップ規模へのスケーリングを実現

#### ネットワーク構成の特徴

```
[Trainium チップ] ←NeuronLink→ [UltraServer 内チップ群]
[UltraServer] ←EFA→ [UltraServer]  ... (UltraCluster)
```

- Scale-up: NeuronLink / NeuronSwitch（サーバ内・ラック内）
- Scale-out: EFA（ラック間・データセンター間）
- NVIDIA の NVLink/NVSwitch + InfiniBand に相当する階層構造

### 2.3 ソフトウェア

#### AWS Neuron SDK

- 最新: Neuron SDK 2.26.0（2025年9月）
- PyTorch 2.8 / JAX 0.6.2 をサポート
- Neuron 2.28 で **PyTorch 2.10 のネイティブサポート**に移行予定
- 対応ライブラリ: HuggingFace, vLLM, PyTorch Lightning 等

#### フレームワーク対応

- **PyTorch**: Eager モード、torch.compile、FSDP / DDP / DTensor によるスケーリング
- **JAX**: 完全サポート
- **vLLM**: V1 対応（Deep Learning Containers に統合）

#### CUDA からの移行

- PyTorch / JAX ベースのコードは**変更なし**で Trainium 上で動作可能（理想的には）
- TorchNeuron (2025): ネイティブ PyTorch バックエンドとして Eager モード、FSDP、DTensor、torch.compile をサポート
- ただし、NVIDIA CUDA エコシステムの成熟度には依然として差がある
- カスタム CUDA カーネルの移行は課題

### 2.4 性能実績とコスト

#### Project Rainier（Anthropic 向け）

- **約50万個の Trainium2 チップ**を使用した世界最大の非 NVIDIA AI クラスター
- 2025年10月に稼働開始、インディアナ州に建設
- 投資額: **110億ドル**、1,200エーカーの敷地、電力容量 2.2GW
- 建設期間: 発表から12ヶ月未満で完成
- Anthropic は 2025年末までに 100万チップ以上で Claude のトレーニング・推論を実行
- 前世代 Claude のトレーニングに使用した計算能力の **5倍以上**を提供

#### コスト比較

- Amazon は GPU 代替と比較して**トレーニング・推論で 30-50% のコスト削減**を主張
- Trainium のヘッドアーキテクトによると、**価格性能比で 30-40% 優位**
- Trainium3 での推論は H100 比で **50% 安価**（初期ベンチマーク）

#### 市場予測

- NVIDIA はトレーニングで支配的地位を維持（90%以上のシェア）
- 推論市場では 2028年までに NVIDIA のシェアが 80% → 20-30% に低下する予測
- ASIC（TPU, Trainium, カスタムチップ）が推論ワークロードの 70-75% を占める見通し

---

## 3. その他の AI 半導体

### 3.1 Cerebras WSE (Wafer-Scale Engine)

#### 概要

半導体ウェーハ全体（300mm）を1つのプロセッサとして使用するという、従来の常識を覆すアプローチ。

#### WSE-3 スペック

| 項目 | スペック |
|------|---------|
| プロセス | TSMC 5nm |
| トランジスタ数 | 4兆個 |
| AI コア | 900,000 |
| SRAM | 44 GB（オンチップ） |
| ピーク性能 | 125 PFLOPS |
| メモリ帯域幅 | 21 PB/s（オンチップ SRAM） |
| 対応モデル規模 | 最大 24兆パラメータ |
| システム | CS-3（CS-2 比 2倍の性能） |

#### アーキテクチャの特徴

- **HBM を使用しない**: オンチップ SRAM のみで超高速メモリアクセスを実現
  - GPU の HBM ボトルネックを完全に回避
- **MemoryX**: 外部メモリシステム（4TB ～ 2.4PB まで構成可能）
  - DDR/HBM ベースの外部メモリ、ウェイトストレージ用
  - エンタープライズ: 24TB / 36TB SKU、ハイパースケーラー: 120TB / 1,200TB
- **SwarmX**: クラスタ間インターコネクトファブリック
  - 最大 2,048 CS-3 システムを接続（最大 0.25 ゼタフロップス）
  - ツリートポロジーによるモジュラー・低オーバーヘッドスケーリング
  - MemoryX と CS システム間でウェイトのブロードキャスト・勾配のリダクションを実行
  - **ほぼ線形のスケーリング**: 10台の CS で単体比 10倍の性能

#### 事業状況（2025-2026年）

- 2026年1月: **OpenAI と100億ドル超の契約**を締結（750MW の計算能力を 2028年まで提供）
- 2026年 Q2 に **IPO** を予定
- Llama2-70B のトレーニングを CS-3 クラスタで **1日で完了** 可能

#### NVIDIA との差異

| 項目 | Cerebras CS-3 | NVIDIA GPU クラスタ |
|------|-------------|-------------------|
| アーキテクチャ | ウェーハスケール単一チップ | 個別 GPU のクラスタ |
| メモリ | オンチップ SRAM + MemoryX | HBM + GPU メモリ |
| スケーリング | SwarmX（ほぼ線形） | NVLink/IB（通信オーバーヘッド大） |
| プログラミングモデル | クラスタ全体が単一デバイス | 分散計算の明示的管理が必要 |
| 推論レイテンシ | 極めて低い（SRAM 直接アクセス） | HBM 依存 |

### 3.2 Groq LPU (Language Processing Unit)

#### 概要

推論特化のアーキテクチャで、決定論的な実行モデルにより超低レイテンシを実現。

#### スペック

| 項目 | 第1世代 (TSP) | 第2世代 (LPUv2) |
|------|-------------|----------------|
| プロセス | 14nm | Samsung 4nm |
| チップサイズ | 25 x 29 mm | - |
| クロック | 900 MHz | - |
| 演算密度 | 1 TeraOp/s/mm² 以上 | 大幅向上 |
| SRAM | 最大 230 MB / チップ | - |
| オンダイメモリ帯域幅 | 80 TB/s | - |
| 製造 | - | 2025年量産ランプ |

#### 性能特性

- **Llama 2 70B**: 300 tokens/s（**NVIDIA H100 クラスタ比 10倍**の速度）
  - ただし Llama 2 70B の実行に **576 LPU** が必要
- **決定論的実行**: プログラマブルアセンブリラインによる**サブミリ秒レイテンシ**
- **タイム・トゥ・ファーストトークン**: ほとんどのモデルで 300ms 未満
- **エネルギー効率**: GPU 比で最大 **10倍効率的**（アーキテクチャレベル）

#### 事業状況

- 2025年2月: サウジアラビアから **15億ドルの投資**確保（ダンマンに GroqCloud データセンター建設）
- **2025年12月: NVIDIA が Groq を200億ドルで買収**
  - NVIDIA は推論特化アーキテクチャを自社ポートフォリオに統合
  - 独立企業としての Groq は終了

#### NVIDIA との差異

| 項目 | Groq LPU | NVIDIA GPU |
|------|---------|------------|
| 用途 | 推論特化 | 汎用（トレーニング + 推論） |
| 実行モデル | 決定論的 | 非決定論的 |
| メモリ | SRAM のみ（HBM なし） | HBM |
| レイテンシ | 超低レイテンシ、予測可能 | 変動あり |
| 現状 | NVIDIA に買収済み | - |

### 3.3 SambaNova (SN40L RDU)

#### 概要

**Reconfigurable Dataflow Unit (RDU)** という独自アーキテクチャで、ストリーミングデータフロー並列性と3層メモリシステムを組み合わせる。

#### SN40L スペック

| 項目 | スペック |
|------|---------|
| プロセス | TSMC 5nm (5FF) |
| パッケージ | CoWoS 2.5D デュアルダイ |
| トランジスタ数 | 1,020億個 |
| BF16 演算性能 | 640 TFLOPS |
| オンチップ SRAM | 520 MiB (PMU) |
| HBM | 64 GiB（コパッケージ） |
| DDR DRAM | 最大 1.5 TiB（プラグ可能 DIMM） |
| メモリ帯域幅 (DDR→HBM) | 1 TB/s 以上 |
| 消費電力 | 平均 10 kWh（空冷可能） |

#### アーキテクチャの特徴

- **3層メモリシステム**: SRAM (520 MiB) → HBM (64 GiB) → DDR (1.5 TiB)
  - 巨大モデルのウェイトを DDR に格納し、HBM 経由で高速アクセス
  - メモリウォール問題への独自解決策
- **Reconfigurable Dataflow**:
  - PCU (Pattern Compute Units) + PMU (Pattern Memory Units) + AGCU のメッシュ接続
  - 数百の複雑なオペレーションを単一カーネルコールにフュージョン可能
  - パイプライン・データ・テンソル並列性のミックスをハードウェアレベルでサポート
- **Composition of Experts (CoE)**: Mixture of Experts の拡張概念

#### NVIDIA との差異

| 項目 | SambaNova RDU | NVIDIA GPU |
|------|-------------|------------|
| アーキテクチャ | リコンフィギュラブルデータフロー | SIMT/テンソルコア |
| メモリ | 3層（SRAM + HBM + DDR） | HBM のみ |
| メモリ容量 | 最大 1.5 TiB / ノード | 80-192 GB / GPU |
| カーネルフュージョン | ハードウェアレベルで自動 | コンパイラ最適化依存 |
| 冷却 | 空冷可能 | ハイエンドは液冷必要 |

### 3.4 Graphcore IPU（SoftBank 傘下）

#### 概要

**Intelligence Processing Unit (IPU)** という独自プロセッサを開発した英国のスタートアップ。BSP (Bulk Synchronous Parallel) モデルによる並列計算が特徴。

#### 事業状況

- **2024年7月: SoftBank Group が買収**（推定 5-6億ドル）
  - ピーク時の評価額は約28億ドルだったため、**75%以上の評価額下落**
  - SoftBank の完全子会社として Graphcore ブランドで事業継続
  - 本社: 英国ブリストル、オフィス: ケンブリッジ、ロンドン、グダンスク、新竹
- 独立企業としては NVIDIA との競争で商業的成功を収められなかった
- SoftBank の AI 戦略（ARM との統合含む）の一部として活用される見通し

#### NVIDIA との差異

| 項目 | Graphcore IPU | NVIDIA GPU |
|------|-------------|------------|
| 並列モデル | BSP（一括同期並列） | SIMT |
| メモリ | 大容量オンチップ SRAM + 外部 | HBM |
| プログラミング | Poplar SDK / PopART | CUDA |
| 現状 | SoftBank 子会社（再建中） | 市場支配的 |

### 3.5 Tenstorrent (RISC-V AI)

#### 概要

CPU 設計の伝説的エンジニア **Jim Keller** が率いる AI チップスタートアップ。RISC-V ベースの AI アクセラレータとハイパフォーマンス CPU IP を開発。

#### チップ世代

##### Wormhole（第1世代、量産中）

| 項目 | スペック |
|------|---------|
| コア | 72 Tensix コア |
| RISC-V | 各 Tensix コアに 5つの RISC-V ベビーコア |
| 製品 | n150（シングルプロセッサ）、n300（デュアル） |
| ワークステーション | TT-LoudBox、TT-QuietBox（$12,000〜） |

##### Blackhole（第2世代、2025年出荷）

| 項目 | スペック |
|------|---------|
| コア | 120 Tensix コア（当初140予定→120に変更） |
| 製品 | p150 アクセラレータカード |
| 性能影響 | 120コアでも既存ユーザーへの性能影響は 1-2% 程度 |

##### Ascalon-X（RISC-V CPU IP）

- 8ワイドデコード、アウトオブオーダー、スーパースカラ RISC-V CPU コア
- Apple M シリーズ、AMD Zen の設計者チームによる開発
- LG、Hyundai 等が Tenstorrent 技術搭載 SoC を開発中

#### 事業状況

- **評価額 32億ドル**、**8億ドルの資金調達**（2025年時点）
- ハードウェア + IP ライセンスのデュアルビジネスモデル
- 日本での AI インフラ構築にも選定
- 中国市場向けの展開も進行中（元 Arm China CEO が関与）

#### NVIDIA との差異

| 項目 | Tenstorrent | NVIDIA |
|------|------------|--------|
| ISA | RISC-V（オープン） | CUDA（独自） |
| ビジネスモデル | チップ + IP ライセンス | チップ + ソフトウェア |
| 価格帯 | $12,000〜（開発者向け） | $25,000〜（コンシューマ GPU 除く） |
| エコシステム | 発展途上 | 圧倒的に成熟 |
| 差別化 | オープンアーキテクチャ、IP ライセンス | 垂直統合エコシステム |

---

## 4. 総合比較

### 4.1 アクセラレータ世代別スペック比較

| チップ | FP8/BF16 性能 | HBM 容量 | メモリ帯域幅 | プロセス |
|--------|-------------|----------|------------|---------|
| Intel Gaudi 3 | 1.8 PFLOPS | 128 GB HBM2E | 3.7 TB/s | 5nm |
| AWS Trainium2 | 1.3 PFLOPS (FP8) | 96 GB HBM3 | 2.9 TB/s | - |
| AWS Trainium3 | 2.52 PFLOPS (FP8) | 144 GB HBM3e | 4.9 TB/s | 3nm |
| NVIDIA H100 | 3.9 PFLOPS (FP8) | 80 GB HBM3 | 3.35 TB/s | 4nm |
| NVIDIA H200 | 3.9 PFLOPS (FP8) | 141 GB HBM3e | 4.8 TB/s | 4nm |
| NVIDIA B200 | 9.0 PFLOPS (FP8) | 192 GB HBM3e | 8.0 TB/s | 4nm |
| Cerebras WSE-3 | 125 PFLOPS | 44 GB SRAM | 21 PB/s (SRAM) | 5nm |
| SambaNova SN40L | 640 TFLOPS (BF16) | 64 GiB HBM | 1 TB/s+ | 5nm |

### 4.2 インターコネクト比較

| プラットフォーム | Scale-up 方式 | Scale-out 方式 | 標準規格準拠 |
|---------------|-------------|--------------|------------|
| Intel Gaudi 3 | 内蔵 RoCEv2 NIC（直接接続） | Ethernet スイッチ | Ethernet 標準 |
| AWS Trainium3 | NeuronLink/NeuronSwitch | EFA | 独自（AWS 閉鎖環境） |
| NVIDIA GB200 NVL72 | NVLink + NVSwitch | InfiniBand / RoCE | 独自 + 一部標準 |
| Cerebras CS-3 | ウェーハ内接続 | SwarmX | 独自 |

### 4.3 ソフトウェアエコシステム成熟度

| プラットフォーム | SDK | PyTorch | モデル互換性 | CUDA 移行難易度 |
|---------------|-----|---------|------------|---------------|
| Intel Gaudi | SynapseAI | Eager/Lazy モード | 高（HF 40K+モデル） | 中 |
| AWS Trainium | Neuron SDK | Eager + compile | 高（vLLM, HF 対応） | 中 |
| NVIDIA | CUDA + cuDNN | ネイティブ | 最高 | - (基準) |
| Cerebras | Cerebras SDK | 対応 | 中 | 高 |
| Groq | GroqWare | 対応 | 中（推論のみ） | 高 |
| SambaNova | SambaFlow | 対応 | 中 | 高 |
| Tenstorrent | TT-Metalium | 開発中 | 低（発展途上） | 高 |

### 4.4 市場ポジションまとめ

| 企業/製品 | 強み | 弱み | 市場状況 |
|----------|------|------|---------|
| **Intel Gaudi** | 低コスト Ethernet ネットワーク、標準規格準拠 | ソフトウェア成熟度、Falcon Shores キャンセル | 苦戦中、方向転換模索 |
| **AWS Trainium** | AWS エコシステム統合、大規模実績(Rainier)、コスト優位 | AWS 外で使用不可、エコシステムの閉鎖性 | 急成長中、クラウド推論市場の鍵 |
| **Cerebras** | 圧倒的な推論速度、ウェーハスケール革新 | コスト、汎用性、製造歩留まり | OpenAI 契約で勢いづく、IPO 予定 |
| **Groq** | 超低レイテンシ推論、決定論的実行 | トレーニング不可、NVIDIA に買収済み | NVIDIA に統合 |
| **SambaNova** | 3層メモリ、大規模モデル対応、空冷 | エコシステム小、認知度低 | ニッチ市場で競争 |
| **Graphcore** | BSP モデル、オンチップメモリ | 商業的失敗、SoftBank 買収 | 再建中 |
| **Tenstorrent** | RISC-V オープン性、IP ライセンスモデル | エコシステム未成熟 | 発展途上、将来性あり |

---

## 5. ソース

### Intel Gaudi
- [Intel Gaudi 3 White Paper](https://cdrdv2-public.intel.com/817486/gaudi-3-ai-accelerator-white-paper.pdf)
- [Gaudi Architecture Documentation](https://docs.habana.ai/en/latest/Gaudi_Overview/Gaudi_Architecture.html)
- [Intel Gaudi 3 Availability - Intel Newsroom](https://newsroom.intel.com/artificial-intelligence/intel-gaudi-3-expands-availability-drive-ai-innovation-scale)
- [Gaudi 3 vs Gaudi 2 Comparison - AMAX](https://www.amax.com/the-next-step-for-intel-accelerators-a-look-at-intel-gaudi-3/)
- [Intel Gaudi 3 - Tom's Hardware](https://www.tomshardware.com/tech-industry/artificial-intelligence/intel-launches-gaudi-3-accelerator-for-ai-slower-than-h100-but-also-cheaper)
- [Falcon Shores Cancellation - Fortune](https://fortune.com/2025/01/31/intels-ai-dreams-slip-further-out-of-reach-as-it-cancels-its-big-data-center-gpu-hope-falcon-shores/)
- [Falcon Shores Not Coming - ServeTheHome](https://www.servethehome.com/intel-falcon-shores-gpu-not-coming-to-market-in-an-ai-hit/)
- [Jaguar Shores - PC Outlet](https://pcoutlet.com/software/ai/intel-ends-gaudi-line-what-jaguar-shores-means-for-the-future-of-ai-hardware)
- [Intel Gaudi 3 with Cisco Nexus - Cisco](https://www.cisco.com/c/en/us/products/collateral/networking/cloud-networking-switches/nexus-9000-switches/connect-intel-gaudi-accelerator-buying-guide.html)
- [Gaudi 3 Deployment Guide - Introl](https://introl.com/blog/intel-gaudi-3-deployment-guide-h100-alternative)
- [Gaudi 3 Hot Chips 2024](https://hc2024.hotchips.org/assets/program/conference/day1/60_HC2024.Intel.RomanKaplan.Gaudi3-0826.pdf)
- [Supermicro Gaudi 3 Reference Architecture](https://www.supermicro.com/white_paper/white_paper_Gaudi3_Reference_Archtecture.pdf)
- [Hugging Face Optimum-Habana](https://huggingface.co/blog/habana-gaudi-2-bloom)
- [Gaudi 3 for PyTorch - Next Platform](https://www.nextplatform.com/2024/04/09/with-gaudi-3-intel-can-sell-ai-accelerators-to-the-pytorch-masses/)
- [VMware + Gaudi 3](https://blogs.vmware.com/cloud-foundation/2025/08/07/accelerating-ai-workloads-with-intel-gaudi-3-on-vmware-cloud-foundation-9-0/)
- [IBM Cloud + Gaudi 3](https://www.ibm.com/products/gpu-ai-accelerator/intel-gaudi3)

### AWS Trainium / Inferentia
- [AWS Trainium](https://aws.amazon.com/ai/machine-learning/trainium/)
- [EC2 Trn2 Instances - AWS Blog](https://aws.amazon.com/blogs/aws/amazon-ec2-trn2-instances-and-trn2-ultraservers-for-aiml-training-and-inference-is-now-available/)
- [EC2 Trn3 UltraServers](https://aws.amazon.com/ec2/instance-types/trn3/)
- [Trainium4 - Next Platform](https://www.nextplatform.com/2025/12/03/with-trainium4-aws-will-crank-up-everything-but-the-clocks/)
- [AWS Neuron SDK](https://aws.amazon.com/ai/machine-learning/neuron/)
- [Neuron SDK 2.26.0 Release Notes](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/release-notes/2.26.0/)
- [Amazon Inferentia](https://aws.amazon.com/ai/machine-learning/inferentia/)
- [EC2 Inf2 Instances](https://aws.amazon.com/ec2/instance-types/inf2/)
- [Amazon Trainium and Inferentia Ecosystem Guide - Introl](https://introl.com/blog/aws-trainium-inferentia-silicon-ecosystem-guide-2025)
- [Project Rainier - About Amazon](https://www.aboutamazon.com/news/aws/aws-project-rainier-ai-trainium-chips-compute-cluster)
- [Project Rainier Activation - CNBC](https://www.cnbc.com/2025/10/29/amazon-opens-11-billion-ai-data-center-project-rainier-in-indiana.html)
- [Trainium vs NVIDIA Cost - Silicon Canals](https://siliconcanals.com/j-amazon-just-gave-companies-a-reason-to-ditch-nvidia-its-called-trainium3-and-its-cheaper-than-you-think/)
- [AI Chips Comparison - CNBC](https://www.cnbc.com/2025/11/21/nvidia-gpus-google-tpus-aws-trainium-comparing-the-top-ai-chips.html)

### Cerebras
- [Cerebras Chip](https://www.cerebras.ai/chip)
- [WSE-3 Announcement](https://www.cerebras.ai/press-release/cerebras-announces-third-generation-wafer-scale-engine)
- [CS-3 Blog](https://www.cerebras.ai/blog/cerebras-cs3)
- [Cerebras Wafer-Scale Cluster Architecture](https://www.cerebras.ai/blog/announcing-the-cerebras-architecture-for-extreme-scale-ai)
- [OpenAI Deal - Next Platform](https://www.nextplatform.com/2026/01/15/cerebras-inks-transformative-10-billion-inference-deal-with-openai/amp/)
- [Cerebras Wikipedia](https://en.wikipedia.org/wiki/Cerebras)

### Groq
- [Groq LPU Architecture](https://groq.com/lpu-architecture)
- [Groq LPU Explained](https://groq.com/blog/the-groq-lpu-explained)
- [Groq LPU Infrastructure Guide - Introl](https://introl.com/blog/groq-lpu-infrastructure-ultra-low-latency-inference-guide-2025)
- [Groq Wikipedia](https://en.wikipedia.org/wiki/Groq)

### SambaNova
- [SN40L RDU](https://sambanova.ai/products/rdu-ai-chips)
- [SN40L Paper - arXiv](https://arxiv.org/html/2405.07518v1)
- [SN40L Blog - SambaNova](https://sambanova.ai/blog/sn40l-chip-best-inference-solution)

### Graphcore
- [Graphcore Wikipedia](https://en.wikipedia.org/wiki/Graphcore)
- [SoftBank Acquisition - Tech.eu](https://tech.eu/2024/07/12/softbank-acquires-ai-processor-manufacturer-graphcore/)
- [Graphcore Story - Turing Post](https://www.turingpost.com/p/graphcore-unicorn-softbank-acquisition-happened)

### Tenstorrent
- [Tenstorrent Wormhole - Tom's Hardware](https://www.tomshardware.com/pc-components/gpus/tenstorrents-risc-v-based-wormhole-ai-accelerators-are-available-for-pre-order-today-pre-built-workstations-start-at-dollar12000)
- [Ascalon RISC-V IP](https://markets.financialcontent.com/wral/article/tokenring-2025-12-25-the-arm-killer-jim-kellers-tenstorrent-unleashes-ascalon-risc-v-ip-to-disrupt-the-data-center)
- [Blackhole p150 Update - Tom's Hardware](https://www.tomshardware.com/tech-industry/semiconductors/jim-kellers-tenstorrent-is-downgrading-blackhole-p150-cards-from-140-to-120-tensor-cores-via-firmware-update-will-ship-cards-with-120-tensor-cores-going-forward-company-claims-existing-users-should-expect-1-2-percent-performance-drop)
- [Tenstorrent Japan](https://tenstorrent.com/en/vision/tenstorrent-risc-v-and-chiplet-technology-selected-to-build-the-future-of-ai-in-japan)
- [Tenstorrent Roadmap - Tom's Hardware](https://www.tomshardware.com/news/tenstorrent-shares-roadmap-of-ultra-high-performance-risc-v-cpus-and-ai-accelerators)

---

## 6. 深掘り調査: Cerebras

### 6.1 CS-3 物理構成

| 項目 | スペック |
|------|---------|
| WSE-3 消費電力 | **約 23 kW** / チップ |
| CS-3 シャーシ電力 | 20 kW 以上 |
| 冷却方式 | **液冷必須**（専用液冷インフラ） |
| ラック構成 | 専用筐体（標準ラックとは異なるフォームファクタ） |

- ウェーハスケールチップ特有の高熱密度により、空冷は不可能
- 標準的なデータセンターラックとは異なる専用の物理インフラが必要

### 6.2 SwarmX ネットワーク詳細

- **SwarmX** は CS チップの内部 Swarm ファブリックをオフチップに拡張した AI 最適化通信ファブリック
- **トポロジ**: ツリートポロジーによるモジュラー・低オーバーヘッドスケーリング
- **スケール**: 最大 2,048 CS-3 システムを接続（最大 0.25 ゼタフロップス）
- **スケーリング効率**: ほぼ線形 — 10台で単体比約 10倍の性能
- **役割**: MemoryX と CS システム間で以下を実行:
  - ウェイトのブロードキャスト（MemoryX → CS）
  - 勾配のリダクション（CS → MemoryX）

### 6.3 MemoryX

- 外部メモリシステム: DDR / HBM ベース
- **構成オプション**:
  - エンタープライズ: 24TB / 36TB SKU
  - ハイパースケーラー: 120TB / 1,200TB
  - 最小 4TB〜最大 2.4PB
- CS システムとは SwarmX ファブリックで接続
- ウェイトストレージとアクティベーション管理を担当

### 6.4 Weight Streaming 実行モデル

1. モデルのウェイトは MemoryX に格納
2. **レイヤーごとに逐次処理**: 1 レイヤーのウェイトを CS の SRAM にロード
3. アクティベーションを計算し、次のレイヤーに移行
4. 勾配は逆方向に計算され、SwarmX 経由で MemoryX にリダクション

- **利点**: WSE の 44GB SRAM に収まらない大規模モデル（24兆パラメータまで）を実行可能
- **制約**: レイヤー単位の逐次処理のため、レイヤー間のパイプライン並列性が制限される

### 6.5 ソフトウェアプラットフォーム（CSoft）

- **CSoft**: Cerebras Software Platform
- ユーザーコードを CS システム上で実行可能なバイナリにコンパイル
- ランタイム環境を提供し、分散最適化の複雑さをユーザーから隠蔽
- **対応フレームワーク**: PyTorch、TensorFlow（PyTorch が主要）
- **プログラミングモデル**: クラスタ全体を単一デバイスとして抽象化 — 分散計算の明示的管理が不要

### 6.6 OpenAI 契約の詳細

- **契約規模**: 750MW の Cerebras ウェーハスケールシステムを複数年にわたりデプロイ
- **契約総額**: 100億ドル以上（契約期間全体）
- **用途**: OpenAI 顧客向け**推論サービス**が主目的
- **展開タイムライン**: フェーズ的にデプロイ、2028年までに完全稼働
- **形態**: Cerebras のシステムを OpenAI が統合して推論サービスを提供

### 6.7 MLPerf 参加状況

- **Cerebras は MLPerf に結果を提出していない**
- 標準化ベンチマークの欠如により、独立した性能評価が困難
- 主張されている「Llama2-70B を1日でトレーニング」等は独自ベンチマーク

---

## 7. 深掘り調査: Groq

### 7.1 GroqRack / GroqNode 構成

| コンポーネント | スペック |
|-------------|---------|
| GroqRack | 8 コンピュート + 1 冗長 GroqNode サーバ |
| GroqNode | **8 基の相互接続 GroqCard** アクセラレータ |
| GroqNode 最大電力 | 約 2 kW |
| ラック内レイテンシ | **エンドツーエンド 1.6 μs**（単一ラック内） |
| 決定論的ネットワーク | 拡張可能な決定論的 LPU ネットワーク |

### 7.2 576 LPU で Llama 2 70B を動作させる構成

- **必要 LPU 数**: 約 576 基（9ラック相当）
- **次世代 LPUv2**: Samsung 4nm プロセスにより、同等ワークロードが **約 100 チップ** で実行可能になる見込み
- ラック数の大幅削減（9 → 約2ラック）

### 7.3 推論ベンチマーク

| モデル | スループット | 備考 |
|--------|-----------|------|
| Llama 2 70B | 300 tokens/s | H100 クラスタ比 10倍 |
| Llama 3.3 70B | **276 tokens/s**（標準） | Artificial Analysis 独立ベンチマーク、全プロバイダ中最速 |
| Llama 3.3 70B + Speculative Decoding | **1,665 tokens/s** | 投機的デコーディング適用時 |

- Groq は独立ベンチマーク（Artificial Analysis）で全プロバイダ中最速の推論速度を記録
- Speculative Decoding により大幅な高速化が可能

### 7.4 NVIDIA 買収の詳細

- **発表日**: 2025年12月24日
- **金額**: **200億ドル**
- **形態**: 全資産・IP の取得ではなく、**コア資産と知的財産のライセンスおよび acqui-hire**
- **従来の買収ではない**: 完全な企業買収ではなく、IP ライセンス + 人材獲得の構造
- **統合見通し**: LPU の決定論的実行モデルを NVIDIA の推論製品ラインに統合する可能性が高い
- GroqCloud サービスの今後は不透明

### 7.5 GroqWare SDK

- **groq-devtools**: モデルのビルドとコンパイル用パッケージ
- **groq-runtime**: コンパイル済みモデルの GroqChip 上での実行用パッケージ
- **Groq API**: クラウド推論 API（REST/gRPC）
  - OpenAI 互換 API を提供
  - Python / JavaScript / Go SDK が利用可能

---

## 8. 深掘り調査: SambaNova

### 8.1 DataScale システム構成

- **DataScale SN10-8R**: 1つ以上の DataScale ノード、統合ネットワーキング、管理インフラを標準準拠データセンターラックに統合
- **DataScale SN30-2**: 各モジュールに SN40L RDU が搭載
- 複数ノードのスケーリングが可能な構成

### 8.2 SambaFlow ソフトウェアスタック

- **SambaFlow**: SambaNova の AI ソフトウェアプラットフォーム
- **コンパイラパイプライン**:
  1. PyTorch モデルを入力
  2. Dataflow Compiler がアノテーション付き Dataflow Graph を受領
  3. Optimizer がグラフ最適化を実施
  4. Assembler が RDU 向けバイナリを生成
- **数百の複雑なオペレーションを単一カーネルコールにフュージョン可能**

### 8.3 SambaNova Cloud

- クラウドベースの AI 推論サービス（SN40L チップ使用）
- **ティア**: Free / Developer / Enterprise
- **提供モデル**: Llama 3.1 シリーズ、Qwen 2.5 シリーズ等のオープンモデル
- Composition of Experts (CoE) によるマルチモデル同時サービング

### 8.4 顧客事例

- **Argonne National Laboratory (DOE)**: 2024年11月に SambaNova Suite をデプロイ。低レイテンシ・高スループット推論で科学研究を加速
- **米国政府機関**: 複数の政府系研究機関が採用
- エンタープライズ向けオンプレミスデプロイメントも展開

---

## 9. 深掘り調査: Graphcore

### 9.1 IPU-POD システム構成

| システム | IPU 数 | 構成 |
|---------|--------|------|
| Bow Pod16 | 16 IPU | 4 × Bow-2000 |
| Bow Pod64 | 64 IPU | 16 × Bow-2000 |
| Bow Pod256 | 256 IPU | 4 × Bow Pod64 ラック |

### 9.2 IPU-Fabric インターコネクト

- **IPU-Link**: チップ間直接接続、**320 GB/s**（双方向）
- **GW-Link**: ポッド間接続（IPU-Gateway 経由）
- **IPU-Fabric**: IPU-Link + GW-Link を組み合わせた階層的ファブリック
- Pod 内は IPU-Link によるダイレクト接続、Pod 間は Gateway 経由で接続

### 9.3 SoftBank 買収後の状況

- 2024年7月に SoftBank が買収（推定 5〜6億ドル、ピーク評価額の 25% 以下）
- SoftBank の完全子会社として **Graphcore ブランドで事業継続**
- 本社: 英国ブリストル、オフィス: ケンブリッジ、ロンドン、グダンスク、新竹
- **次世代チップ**: ARM Neoverse との統合 SoC が SoftBank の AI インフラ戦略の一環として開発中の可能性
- 具体的な新製品のアナウンスは限定的

---

## 10. 深掘り調査: Tenstorrent

### 10.1 ソフトウェアスタック詳細

#### TT-Metalium（低レベル API）

- Tenstorrent ハードウェア向けのカーネル開発フレームワーク
- CUDA に相当する低レベルプログラミングモデル

#### TT-NN（高レベル API）

- PyTorch に馴染みのある開発者向けのニューラルネットワークオペレーションライブラリ
- TT-Metalium 上に構築

#### TT-Forge

- PyTorch モデルの自動変換・最適化コンパイラ
- **MLIR ベース**: tt-xla を含むコンパイラインフラ
- PyTorch 2.0+ の torch.compile バックエンドとして統合予定

#### tt-torch

- PyTorch 2.0 ネイティブ統合
- 開発進行中（2025〜2026年）

### 10.2 Wormhole の Ethernet 直接接続

- Wormhole チップは **Ethernet 直接接続を内蔵**
- 外部スイッチなしでチップ間通信が可能
- NVLink に対するオープンな代替アプローチ
- ただしデータセンタースケールでの構成詳細は限定的

### 10.3 Galaxy データセンター製品

- データセンタースケールのクラスタ製品として「Galaxy」が計画されている
- 具体的な構成・スペックの公開情報は限定的
- Ethernet Mesh によるスケーラブルなインターコネクトが特徴

### 深掘り調査のソース

#### Cerebras
- [Cerebras Wafer-Scale Cluster Architecture](https://www.cerebras.ai/blog/announcing-the-cerebras-architecture-for-extreme-scale-ai)
- [CS-3 Power Consumption - 23kW](https://www.nextplatform.com/2024/03/20/cerebras-doubles-down-on-wafer-scale-ai-compute/)
- [CSoft Software Platform](https://www.cerebras.ai/software)
- [OpenAI-Cerebras Contract Details](https://www.nextplatform.com/2026/01/15/cerebras-inks-transformative-10-billion-inference-deal-with-openai/)
- [Weight Streaming Execution Model](https://www.cerebras.ai/blog/cerebras-weight-streaming)

#### Groq
- [GroqRack Configuration](https://groq.com/groqrack/)
- [Groq Llama 3.3 70B Benchmark - Artificial Analysis](https://artificialanalysis.ai/text/speed)
- [NVIDIA-Groq Acquisition](https://www.reuters.com/technology/nvidia-acquire-groq-20-billion-deal-2025-12-24/)
- [GroqWare Developer Tools](https://groq.com/developer-tools/)

#### SambaNova
- [SambaFlow Compiler Pipeline](https://docs.sambanova.ai/developer/latest/sambaflow-intro.html)
- [Argonne National Lab Deployment](https://www.anl.gov/article/argonne-deploys-sambanova-suite)
- [SambaNova Cloud](https://cloud.sambanova.ai/)
- [DataScale System Configuration](https://sambanova.ai/products/datascale)

#### Graphcore
- [IPU-POD System Architecture](https://docs.graphcore.ai/en/latest/hardware.html)
- [SoftBank Acquisition Details](https://tech.eu/2024/07/12/softbank-acquires-ai-processor-manufacturer-graphcore/)

#### Tenstorrent
- [TT-Metalium / TT-NN Overview](https://tenstorrent.com/en/technology/metalium)
- [TT-Forge Compiler](https://github.com/tenstorrent/tt-forge-fe)
- [Wormhole Ethernet Direct Connect](https://tenstorrent.com/en/hardware/wormhole)
