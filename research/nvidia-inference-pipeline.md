# NVIDIA AI データセンターにおける推論パイプライン

調査日: 2026-02-28

---

## 目次

1. [推論 vs トレーニングの技術的な違い](#1-推論-vs-トレーニングの技術的な違い)
2. [KV キャッシュ管理](#2-kv-キャッシュ管理)
3. [Speculative Decoding](#3-speculative-decoding)
4. [推論サービングの最適化技術](#4-推論サービングの最適化技術)
5. [推論時のネットワーク要件](#5-推論時のネットワーク要件)
6. [NVIDIA の推論製品スタック](#6-nvidia-の推論製品スタック)

---

## 1. 推論 vs トレーニングの技術的な違い

### 1.1 メモリ使用パターン

#### トレーニング時のメモリ構成

トレーニング時の GPU メモリは以下の要素で構成される:

| メモリ要素 | 説明 | 概算サイズ (FP16 学習) |
|---|---|---|
| モデルパラメータ | 重み行列 | 2 bytes/param |
| 勾配 (Gradients) | 各パラメータの勾配 | 2 bytes/param |
| オプティマイザ状態 | Adam の場合 m, v | 8 bytes/param (FP32 m + FP32 v) |
| アクティベーション | 中間結果（逆伝播用） | バッチサイズ・シーケンス長に依存 |

例: 70B パラメータのモデルでは、パラメータだけで 140GB、オプティマイザ状態を含めると **約 840GB** のメモリが必要になる。

#### 推論時のメモリ構成

推論ではオプティマイザ状態・勾配・アクティベーションの保持が不要であり、メモリ構成が大幅に異なる:

| メモリ要素 | 説明 | 概算サイズ |
|---|---|---|
| モデルパラメータ | 重み行列（量子化可能） | FP16: 2 bytes/param, INT4: 0.5 bytes/param |
| KV キャッシュ | 各リクエストの Key/Value テンソル | シーケンス長に比例して増大 |
| 一時バッファ | 推論計算用 | 比較的小さい |

**重要な違い**: トレーニングではオプティマイザ状態がメモリの支配的要素であるのに対し、推論では **KV キャッシュ** がメモリのボトルネックとなる。推論は量子化（INT8, INT4, FP8, FP4）を積極的に適用でき、勾配の精度を保つ必要がないため、メモリ使用量を大幅に削減できる。

### 1.2 計算パターン

#### Prefill フェーズ（計算バウンド）

推論の最初のフェーズである Prefill（プロンプト処理）は、入力トークン全体に対して並列に行列乗算を実行するため、**計算バウンド (Compute-Bound)** となる。

- 高い Arithmetic Intensity（演算密度）を持つ
- Tensor Core の利用率が高い
- QKV-Proj, O-Proj, FlashAttention カーネルが計算バウンド領域で動作
- FFN カーネルもデータセンター GPU では計算バウンド

#### Decode フェーズ（メモリバウンド）

トークン生成を1つずつ行う Decode フェーズは、**メモリバウンド (Memory-Bound)** となる。

- KV キャッシュへの頻繁なアクセスが発生
- Arithmetic Intensity が低く、メモリレイテンシを隠蔽できない
- Attention カーネルの L2 キャッシュヒット率が 73-82% 低下（Chat タスクで 81.9% 低下、Summary タスクで 73.3% 低下）
- Prefill の集約アクセスがデータ再利用を最大化するのに対し、Decode の散発的な KV キャッシュ読み取りはローカリティを破壊する

#### Roofline モデルによる分析

Roofline モデルは Arithmetic Intensity (AI) と達成可能な性能の関係を表す:

```
性能 (FLOPS) = min(ピーク計算性能, メモリ帯域幅 × Arithmetic Intensity)
```

- **Prefill**: 高い AI を持ち、Roofline の計算バウンド領域（右側）に位置
- **Decode**: 低い AI を持ち、Roofline のメモリバウンド領域（左側）に位置

この Phase 間のボトルネックの差異が、Disaggregated Serving（Prefill と Decode の分離実行）の理論的根拠となっている。

### 1.3 レイテンシ vs スループットのトレードオフ

| 指標 | トレーニング | 推論 |
|---|---|---|
| 主な最適化目標 | スループット（全体処理量の最大化） | レイテンシ（応答速度）+ スループット |
| バッチサイズ | 大きい（数千～数万トークン） | 動的に変動 |
| 典型的な制約 | GPU 利用率、通信オーバーヘッド | TTFT, ITL, TPS/user |

推論では以下の2つのレイテンシ指標が重要:

- **TTFT (Time To First Token)**: 最初のトークンが出力されるまでの時間。Prefill フェーズの性能に依存
- **ITL (Inter-Token Latency)**: トークン間の生成時間。Decode フェーズの性能に依存

バッチサイズを1から64に増やすと（A100 の場合）スループットは大幅に向上するが、TTFT と ITL も同時に増加するというトレードオフが存在する。

---

## 2. KV キャッシュ管理

### 2.1 KV キャッシュとは何か

Transformer の Self-Attention 機構では、各トークンの出力を計算するために、過去の全トークンの Key (K) と Value (V) テンソルを参照する必要がある。Autoregressive な生成（1トークンずつ生成）では、過去のトークンの K, V を毎回再計算するのは非効率であるため、一度計算した K, V をキャッシュして再利用する。これが **KV キャッシュ** である。

```
Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V
```

KV キャッシュがない場合、N トークンの生成に O(N^2) の計算が必要になるが、キャッシュすることで O(N) に削減できる。

### 2.2 なぜ推論のボトルネックになるか

KV キャッシュが推論のボトルネックとなる理由は以下の通り:

1. **メモリ消費量**: コンテキスト長とバッチサイズに比例して線形に増大
2. **メモリ帯域幅**: Decode 時に毎ステップで全キャッシュを読み出す必要がある
3. **断片化**: 従来のシステムでは KV キャッシュメモリの 60-80% が断片化により無駄になっていた
4. **動的サイズ**: 各リクエストの出力長が事前に不明なため、事前割り当てが困難

### 2.3 KV キャッシュサイズの計算式

KV キャッシュのサイズは以下の式で計算される:

```
KV Cache Size = 2 × n_layers × n_heads × d_head × seq_len × batch_size × bytes_per_element
```

各変数の意味:

| 変数 | 説明 |
|---|---|
| 2 | Key と Value の2つの行列 |
| n_layers | Transformer レイヤー数 |
| n_heads | Attention ヘッド数 |
| d_head | 各ヘッドの次元数 |
| seq_len | シーケンス長（コンテキスト長） |
| batch_size | 同時リクエスト数 |
| bytes_per_element | 精度ごとのバイト数（FP16=2, FP8=1, INT8=1） |

#### 具体的な計算例

**Llama 2 70B (FP16)** の場合:
- n_layers = 80, n_heads = 64 (GQA: 8 KV heads), d_head = 128
- GQA を考慮した KV キャッシュ（1トークンあたり）:

```
1トークンあたり KV = 2 × 80 × 8 × 128 × 2 bytes = 327,680 bytes ≈ 320 KB
```

| コンテキスト長 | バッチサイズ 1 | バッチサイズ 8 | バッチサイズ 64 |
|---|---|---|---|
| 4,096 | 1.25 GB | 10 GB | 80 GB |
| 32,768 | 10 GB | 80 GB | 640 GB |
| 128,000 | 39 GB | 312 GB | 2,500 GB |
| 1,000,000 | 305 GB | 2,440 GB | 19,520 GB |

**コンテキスト長の爆発的影響**: コンテキスト長が 4K から 1M に伸びると KV キャッシュは **約 244 倍** に増大する。これが 100 万トークン超のコンテキストを扱う際の最大の技術的課題である。

### 2.4 PagedAttention (vLLM) の仕組みと効果

#### 従来のアプローチの問題

従来の KV キャッシュ管理では:
- リクエストごとに最大シーケンス長分の連続メモリを事前確保
- 実際に使用されるのは割り当て済みメモリの 20-38% のみ
- 内部断片化と外部断片化で 60-80% のメモリが浪費

#### PagedAttention の仕組み

PagedAttention は OS の仮想メモリとページング技術に着想を得た Attention アルゴリズムであり、vLLM で実装されている:

1. **ブロック分割**: KV キャッシュを固定サイズのブロック（例: 16 トークン分）に分割
2. **非連続メモリ配置**: 各ブロックは GPU メモリの任意の場所に配置可能（連続メモリ不要）
3. **論理→物理マッピング**: 論理ブロックから物理ブロックへのマッピングテーブルを管理
4. **動的割り当て**: トークン生成に応じてブロックを動的に割り当て・解放
5. **Copy-on-Write**: Beam Search などでの KV キャッシュ共有を効率化

#### PagedAttention の効果

| 指標 | 従来手法 | PagedAttention |
|---|---|---|
| KV キャッシュ無駄率 | 60-80% | 4% 未満 |
| スループット向上 | ベースライン | 2-4x 向上 |
| 同時リクエスト数 | 制限的 | 大幅に増加 |

vLLM は FasterTransformer や Orca と同等のレイテンシで、スループットを **2-4 倍** 向上させることが実証されている。

#### 2025年の進展: PagedEviction

2025年には PagedEviction と呼ばれるブロック単位の退避アルゴリズムが導入され、CUDA カーネルを変更することなく、重要度の低いブロックを特定・除去することで、さらなるメモリ効率化を実現している。

---

## 3. Speculative Decoding

### 3.1 仕組み（ドラフトモデル + 検証モデル）

Speculative Decoding は、推論時のレイテンシを削減する手法であり、出力品質を一切低下させない点が特徴的である。

#### 基本アルゴリズム

```
ループ:
  1. ドラフトモデル（小型・高速）が K 個のトークンを投機的に生成
  2. ターゲットモデル（大型・高精度）が K 個のトークンを並列に検証
  3. ターゲットモデルが同意する最長の接頭辞（h 個）を受理
  4. ターゲットモデルが (h+1) 番目のトークンを自ら生成
  5. 受理されたトークンと新しいトークンを出力に追加
```

#### 数学的保証

Speculative Decoding の重要な性質は、**ターゲットモデル単体で生成した場合と完全に同一の出力分布を保証する** ことである。検証ステップで修正された拒絶サンプリングを使用することで、この性質が成り立つ。

#### 受理率 (Acceptance Rate, α)

- **受理率 α**: ドラフトトークンがターゲットモデルに受理される確率
- α が高いほど、1ラウンドあたりの有効トークン数が増加し、ターゲットモデルの Forward Pass 回数が減少
- α はドラフトモデルとターゲットモデルの分布の類似度に依存
- ドラフトモデルの予測エントロピーが高い場合、α が低下する傾向がある

#### 2025年の進展: 動的ドラフト長

2025年の研究では、ドラフトモデルのエントロピーを参照してドラフトシーケンスの長さを適応的に決定する **SVIP** 手法が提案されている。エントロピーが高い（モデルが不確実な）区間ではドラフト長を短くし、低い区間では長くすることで効率を最適化する。

### 3.2 NVIDIA TensorRT-LLM での実装

TensorRT-LLM は複数の Speculative Decoding 手法をサポートしている:

#### サポートされる手法

| 手法 | 説明 | 特徴 |
|---|---|---|
| Draft Model Speculative Decoding | 別の小型モデルをドラフトに使用 | 最も一般的な手法 |
| Medusa | 追加の Attention Head でドラフト生成 | 単一モデルで完結 |
| ReDrafter | RNN ドラフトモデル + Beam Search + 動的 Tree Attention | Medusa の進化版 |
| EAGLE | 追加レイヤーでドラフト生成 | 高い受理率 |
| Lookahead Decoding | Jacobi 反復を利用 | ドラフトモデル不要 |

#### ReDrafter の詳細

Apple と NVIDIA が共同開発した ReDrafter は、以下の革新を導入:

- **RNN ドラフトモデル**: 各ドラフトトークンが前のトークンに依存するリカレント予測器を使用
- **Beam Search**: より有望なドラフトトークンの候補を探索
- **動的 Tree Attention**: 複数のドラフト候補を木構造で同時検証

### 3.3 性能向上の実測データ

| ベンチマーク | モデル | GPU | 手法 | スループット向上 |
|---|---|---|---|---|
| TensorRT-LLM 公式 | Llama 3.1 405B | 4x H200 | Speculative Decoding | 最大 3.6x |
| TensorRT-LLM 公式 | Llama 3.3 70B | - | Speculative Decoding | 3x |
| ReDrafter (Apple) | オープンソース LLM | H100 (TP8) | ReDrafter | 最大 2.7x スループット向上 |
| ReDrafter | オープンソース LLM | - | ReDrafter | 最大 3.5 トークン/生成ステップ |

#### 測定条件の注意

- Llama 3.1 405B の 3.6x 向上は 2024年11月18日に 4x NVIDIA H200 で測定
- Llama 3.3 70B の 3x 向上は 2025年1月に報告
- 性能向上はタスクの性質（受理率に影響）と設定に大きく依存

---

## 4. 推論サービングの最適化技術

### 4.1 Continuous Batching (In-Flight Batching)

#### 従来の Static Batching の問題

Static Batching では:
- バッチ内の全リクエストが完了するまで次のバッチを開始できない
- 短い応答を返すリクエストが終了しても GPU は待機状態
- GPU 利用率が著しく低下

#### Continuous Batching の仕組み

Continuous Batching（TensorRT-LLM では In-Flight Batching と呼称）は:

1. **イテレーションレベルのスケジューリング**: 各 Decode ステップの後にスケジューリング判断を行う
2. **動的なリクエスト追加・除去**: 完了したリクエストは即座にバッチから除去され、新しいリクエストが即座に追加される
3. **Prefill と Decode の混在実行**: コンテキスト処理と生成を同一バッチ内で同時実行

#### 効果

- GPU の無稼働時間を最小化
- スループットを大幅に向上（Static Batching 比で数倍）
- 現代の推論エンジン（vLLM, TensorRT-LLM, SGLang）の標準機能

### 4.2 Tensor Parallelism vs Pipeline Parallelism の推論での使い分け

#### Tensor Parallelism (TP)

```
1つのレイヤーの重み行列を複数 GPU に分割
→ 各 GPU が部分的な出力を計算
→ AllReduce で結果を統合
```

- **利点**: 単一リクエストのレイテンシを削減（モデルが GPU メモリに収まらない場合に必須）
- **欠点**: AllReduce 通信がレイヤーごとに必要（高帯域 NVLink 必須）
- **最適な用途**: Decode フェーズ（レイテンシが重要）
- **通信量**: 固定（Transformer レイヤー数と Decode ステップ数に依存）

#### Pipeline Parallelism (PP)

```
モデルをレイヤーグループに分割し、各 GPU に割り当て
→ マイクロバッチが各ステージを順次通過
→ Point-to-Point 通信のみ
```

- **利点**: 通信オーバーヘッドが最小（AllReduce 不要）
- **欠点**: パイプラインバブル（各ステージの待機時間）でレイテンシが増加
- **最適な用途**: Prefill フェーズ（スループットが重要）

#### 推論での使い分け指針

| 観点 | Tensor Parallelism | Pipeline Parallelism |
|---|---|---|
| 主な通信パターン | AllReduce / AllGather | Point-to-Point |
| レイテンシ | 低い（Decode に最適） | 高い（バブル発生） |
| スループット | 通信オーバーヘッドあり | 高い（Prefill に最適） |
| 必要な帯域幅 | 非常に高い (NVLink) | 比較的低い |
| 推奨 GPU 配置 | ノード内 (NVLink 接続) | ノード間 (InfiniBand 接続) |

#### 2025年の進展: Seesaw と N-D Parallelism

- **Seesaw**: 動的なモデル再シャーディングにより、Prefill と Decode のフェーズ間で並列化戦略を動的に再構成
- **Meta の N-D Parallelism**: Context Parallelism (CP), Pipeline Parallelism (PP), Expert Parallelism (EP), Tensor Parallelism (TP) を組み合わせたハイブリッドアプローチ

### 4.3 Quantization の推論時の効果

#### 精度フォーマットの比較

| フォーマット | ビット幅 | メモリ削減率 (vs FP16) | スループット向上 | 品質維持率 |
|---|---|---|---|---|
| FP16 (ベースライン) | 16-bit | 1x | 1x | 100% |
| FP8 | 8-bit | 2x | 約 2.3x (H100) | 99.5%+ |
| INT8 (SmoothQuant) | 8-bit | 2x | 約 2x | 99%+ |
| INT4 (GPTQ) | 4-bit | 4x | 約 2.69x | 90% |
| INT4 (AWQ) | 4-bit | 4x | 約 2.5x | 95% |
| FP4 (NVFP4) | 4-bit | 4x | FP8 の 2x (Blackwell) | 98%+ |

#### 主要な量子化手法

**Weight-Only Quantization（重みのみ量子化）**:

- **AWQ (Activation-Aware Weight Quantization)**: アクティベーション分布に基づく重要な重みチャネルを保護。GPTQ より高品質
- **GPTQ**: レイヤーごとの最適化で重みを量子化。高速な量子化処理

**Weight + Activation Quantization（重み・アクティベーション両方を量子化）**:

- **SmoothQuant**: アクティベーションの外れ値をスムージングしてから INT8 量子化
- **FP8**: Hopper 以降でネイティブサポート。最も安定した品質・性能バランス

**NVIDIA 固有フォーマット**:

- **NVFP4**: Blackwell アーキテクチャで導入。FP4 精度で高い圧縮率と精度維持を両立

#### 実測データ

- FP8 + バッチサイズ16（H100）: FP16 比で **2.3x 推論高速化**（TTFT 500ms 以下の制約下）
- FP8 Mixtral 70B: FP16 比で **3.5x 高速化**、精度低下 0.5% 未満
- 4-bit 量子化: MMLU-Pro でベースライン推論能力の **98.1%** を維持

### 4.4 TensorRT-LLM の主要最適化

TensorRT-LLM は NVIDIA が提供する LLM 推論最適化フレームワークであり、以下の最適化を統合的に提供する:

#### コア最適化機能

| 機能 | 説明 |
|---|---|
| カスタム Attention カーネル | FlashAttention ベースの最適化された Attention 実装 |
| In-Flight Batching | Continuous Batching の TensorRT-LLM 実装 |
| Paged KV Cache | PagedAttention によるメモリ効率化 |
| KV Cache 量子化 | FP8/INT8 での KV キャッシュ圧縮 |
| Circular Buffer KV Cache | 循環バッファによる効率的なキャッシュ管理 |
| KV Cache Reuse | プレフィックス共有によるキャッシュ再利用 |
| 優先度ベース KV Cache 退避 | トークン範囲に優先度・期間を指定した退避制御 |
| Speculative Decoding | 複数手法（Draft Model, Medusa, ReDrafter 等）のサポート |
| 量子化 | FP8, FP4, INT4 AWQ, INT8 SmoothQuant 等 |
| Chunked Prefill | 長いプロンプトを分割して処理 |
| Tensor Parallelism | マルチ GPU 推論 |

#### 2025年の新機能

- **優先度ベース KV Cache 退避** (2025年4月): ユーザーがトークン範囲ごとに優先度と期間を指定可能。キャッシュヒット率が約 **20% 向上**
- **MultiShot AllReduce**: NVSwitch を活用した AllReduce の **3 倍高速化**

---

## 5. 推論時のネットワーク要件

### 5.1 トレーニングとの違い

#### トレーニング時の通信パターン

| 並列化手法 | 主な通信オペレーション | 通信頻度 | データ量 |
|---|---|---|---|
| Data Parallelism | AllReduce（勾配同期） | 各ステップ（逆伝播後） | 全パラメータの勾配 |
| Tensor Parallelism | AllReduce（Forward/Backward） | 各レイヤー | アクティベーション |
| Pipeline Parallelism | Point-to-Point | 各マイクロバッチ | アクティベーション |
| ZeRO-3 | AllGather + ReduceScatter | Forward/Backward 全体 | パラメータシャード |

トレーニングでは **AllReduce が支配的** であり、全 GPU 間で勾配を集約・配布する必要がある。これは全対全通信であり、GPU 数に応じて通信コストが増大する。

#### 推論時の通信パターン

| 並列化手法 | 主な通信オペレーション | 通信頻度 | データ量 |
|---|---|---|---|
| Tensor Parallelism | AllReduce / AllGather | 各レイヤー（Decode ごと） | 部分アクティベーション |
| Pipeline Parallelism | Point-to-Point | 各ステージ間 | アクティベーション |
| Disaggregated Serving | KV Cache 転送 | Prefill 完了時 | KV Cache 全体 |
| Expert Parallelism (MoE) | All-to-All | 各 MoE レイヤー | トークンルーティング |

**推論とトレーニングの通信の違い**:

1. **勾配同期が不要**: Data Parallelism の AllReduce（最大の通信コスト）が完全に不要
2. **AllGather 中心**: Tensor Parallelism で分割されたアクティベーションの再構築が主
3. **通信量が小さい**: バッチサイズが小さく、逆伝播がないため
4. **レイテンシ感度が高い**: 各 Decode ステップで通信が発生し、レイテンシに直結
5. **AllReduce オーバーヘッド**: Tensor Parallelism の AllReduce がエンドツーエンドレイテンシの **最大 30%** を占める（Meta の測定）

#### 推論における帯域幅要件

推論では通信量自体はトレーニングより小さいが、**レイテンシ要件** が厳しい。Decode ステップは各レイヤーで AllReduce/AllGather を待つ必要があり、100ms 以下の ITL を実現するには:

- ノード内: **NVLink** (900 GB/s, Blackwell) が事実上必須
- ノード間: **InfiniBand NDR/XDR** (400-800 Gbps) が推奨
- 高速ネットワークがないと、ワークロード時間の **最大 50%** が通信待ちに費やされる

### 5.2 Prefill vs Decode の分離（Disaggregated Serving）

#### アーキテクチャ概要

Disaggregated Serving は、Prefill と Decode を別々の GPU プールで実行するアーキテクチャである。2025年には事実上の標準的な推論アーキテクチャとなり、ほぼすべての主要な推論フレームワーク（NVIDIA Dynamo, llm-d, Ray Serve LLM, SGLang, vLLM, LMCache, MoonCake）が対応している。

```
クライアント → [ルーター] → [Prefill GPU プール] → KV Cache 転送 → [Decode GPU プール] → 出力
```

#### 分離の理由とメリット

| 観点 | Prefill | Decode |
|---|---|---|
| 計算特性 | 計算バウンド | メモリバウンド |
| バッチ戦略 | 大バッチで高スループット | 小バッチで低レイテンシ |
| GPU 最適化 | 高 FLOPS が重要 | 高メモリ帯域が重要 |
| レイテンシ指標 | TTFT に影響 | ITL に影響 |

**主なメリット**:
1. **独立スケーリング**: Prefill と Decode の GPU 数を独立に調整可能
2. **Decode レイテンシの安定化**: Decode GPU が Prefill に中断されないため、ITL が予測可能に
3. **性能向上**: 従来の統合アーキテクチャ比で **2-7x の性能向上**
4. **ヘテロジニアスハードウェア対応**: Prefill に計算特化 GPU、Decode にメモリ帯域特化 GPU を使用可能

#### KV Cache 転送の課題

Disaggregated Serving の最大の課題は Prefill GPU から Decode GPU への KV Cache 転送:

- **TTFT への影響**: KV Cache 転送に数十～数百ミリ秒を要し、TTFT が増加
- **必要帯域幅**: 長コンテキストでは GB 単位の KV Cache 転送が必要
- **RDMA サポート**: EFA や ConnectX NIC による RDMA を使用して低レイテンシ転送を実現

#### 2025年の先進研究: TraCT

TraCT は CXL 共有メモリを KV Cache 転送基盤として使用するラックスケール推論システム:
- GPU が CXL load/store および DMA で KV ブロックを直接読み書き
- NIC を経由する従来のパイプラインを排除
- プレフィックス認識の KV Cache をラック全体で共有

### 5.3 NVIDIA Dynamo（推論オーケストレーション）

NVIDIA Dynamo は GTC 2025 で発表された、データセンタースケールの分散推論サービングフレームワークである。

#### 主要コンポーネント

```
┌─────────────────────────────────────────────────┐
│                NVIDIA Dynamo                     │
│                                                  │
│  ┌──────────┐  ┌──────────────┐  ┌───────────┐ │
│  │  Smart   │  │    GPU       │  │ Low-Latency│ │
│  │  Router  │  │  Resource    │  │   Comms    │ │
│  │          │  │  Planner     │  │   Library  │ │
│  └────┬─────┘  └──────┬───────┘  └─────┬─────┘ │
│       │               │                │        │
│  ┌────▼─────┐  ┌──────▼───────┐  ┌─────▼─────┐ │
│  │ KV Cache │  │ Prefill/     │  │ KV Cache  │ │
│  │ Aware    │  │ Decode       │  │ Manager   │ │
│  │ Routing  │  │ Scaling      │  │           │ │
│  └──────────┘  └──────────────┘  └───────────┘ │
└─────────────────────────────────────────────────┘
```

#### 1. Smart Router（KV Cache 認識ルーティング）

- リクエストをハッシュ化し、GPU 全体のキャッシュ位置を追跡
- 受信リクエストとキャッシュ済み KV ブロックのオーバーラップスコアを計算
- キャッシュ再利用を最大化する GPU にルーティング
- ワークロードバランシングも同時に考慮
- **効果**: 不要な KV Cache 再計算を回避し、コストと遅延を削減

#### 2. GPU Resource Planner

- GPU 容量メトリクスを継続的にモニタリング
- TTFT, ITL などの SLO と組み合わせて判断
- Prefill と Decode のワーカー数を動的にスケーリング
- 例: 長コンテキストリクエストの急増を検知すると、Decode GPU を Prefill に動的に再割り当て

#### 3. Low-Latency Communication Library

- GPU 間の KV Cache 転送を高速化
- ヘテロジニアスメモリ・ストレージタイプ間でのデータ転送を加速

#### 4. KV Cache Manager

- 各種メモリ階層間での KV Cache 転送を管理
- 貴重な GPU メモリを解放しつつユーザー体験を維持

#### 性能実績

- DeepSeek-R1（NVIDIA Blackwell）: リクエスト処理数が最大 **30x 向上**
- GB200 NVL72 での Disaggregated Serving: DGX B200 の統合サービング比で GPU あたりスループットが **約 1.5x** 向上（Llama 3.1 405B, MLPerf 提出結果）

#### 2026年1月の拡張（Microsoft Azure との連携）

- Dynamo Planner がランタイムオーケストレーションエンジンとして機能
- LLM 認識の負荷分散（従来のロードバランサーとは異なる）
- SLO 違反前にプロアクティブにリソースを調整

---

## 6. NVIDIA の推論製品スタック

### 6.1 Blackwell の推論特化機能

#### アーキテクチャ概要 (B200 / GB200)

| 仕様 | B200 | H100 (比較) |
|---|---|---|
| FP8 性能 | 20 PFLOPS | 4 PFLOPS |
| FP6 性能 | 20 PFLOPS | N/A |
| FP4 性能 | 40 PFLOPS | N/A |
| HBM3e メモリ | 192 GB | 80 GB |
| メモリ帯域幅 | 8 TB/s | 3.35 TB/s |
| NVLink 帯域幅 | 1.8 TB/s | 900 GB/s |

#### 推論性能ベンチマーク

| ベンチマーク | モデル | 構成 | 結果 |
|---|---|---|---|
| DGX B200 vs DGX H100 | 汎用 | 推論 | **15x 推論性能向上** |
| GB200 NVL72 vs H100 同数 | LLM | 推論 | **30x 性能向上** |
| Llama 4 Maverick | 400B | DGX B200 (8x GPU) | **1,000+ TPS/user**, **72,000 TPS/server** |
| DeepSeek-R1 | 671B | DGX B200 (8x GPU, FP4) | **250+ TPS/user**, **30,000+ TPS** |
| InferenceMAX | - | B200 | **60,000 TPS/GPU** |

#### 推論特化機能

1. **第2世代 Transformer Engine**: TensorRT-LLM と NeMo Framework と連携し、LLM と MoE モデルの推論を加速
2. **FP4 ネイティブサポート**: Hopper の FP8 に加え、FP4 をハードウェアネイティブでサポート。FP8 比で **2 倍** のピークスループット
3. **FP6 サポート**: FP8 と FP4 の中間精度。精度と性能のバランスを提供
4. **MoE 最適化**: MoE アーキテクチャで **10x** の性能向上
5. **GB200 NVL72 ラックスケール設計**: 36 Grace CPU + 72 Blackwell GPU を液冷ラックに統合。コストとエネルギー消費を **25x 削減**

#### Blackwell Ultra (GB300)

MLPerf Inference v5.1 で Blackwell Ultra が初登場:
- GB200 NVL72 比で GPU あたり **最大 1.4x 性能向上**
- Hopper 比で GPU あたり **約 5x スループット向上**（DeepSeek-R1）
- DeepSeek-R1, Llama 3.1 405B で MLPerf 推論記録を更新

### 6.2 Rubin CPX（推論特化 GPU）

#### 設計思想

Rubin CPX は **推論ワークロードに特化** した全く新しいクラスの GPU であり、特に 100 万トークン超のロングコンテキスト処理に最適化されている。

#### 主要仕様

| 仕様 | Rubin CPX | GB300 (比較) |
|---|---|---|
| NVFP4 計算性能 | 30 PFLOPS | - |
| メモリ | 128 GB GDDR7 | HBM3e |
| ダイ設計 | モノリシック（コスト効率重視） | マルチチップレット |
| Attention 高速化 | GB300 NVL72 比 **3x** | ベースライン |
| ビデオ処理 | 4x NVENC + 4x NVDEC | - |
| 想定出荷時期 | 2026年末 | 2025年 |

#### Vera Rubin NVL144 CPX プラットフォーム

| 仕様 | Vera Rubin NVL144 CPX | GB300 NVL72 (比較) |
|---|---|---|
| GPU 構成 | 144 Rubin CPX + 144 Rubin GPU | 72 GB300 |
| CPU | 36 Vera CPU | 36 Grace CPU |
| AI 性能 | **8 EFLOPS** (NVFP4) | 約 1 EFLOPS |
| メモリ容量 | **100 TB** | - |
| メモリ帯域幅 | **1.7 PB/s** | - |
| 性能倍率 | GB300 NVL72 比 **7.5x** | ベースライン |

#### 推論特化の根拠

1. **GDDR7 メモリ採用**: HBM より低コストで大容量。推論ではメモリ容量（KV キャッシュ用）がより重要
2. **Attention 高速化 3x**: ロングコンテキストの Attention 計算がボトルネックとなるため、専用ハードウェアで加速
3. **モノリシックダイ**: 製造コストを抑え、推論の TCO（総所有コスト）を最小化
4. **Disaggregated Serving に最適**: Prefill は Rubin GPU、Decode は Rubin CPX という役割分担が可能

### 6.3 NIM (NVIDIA Inference Microservices)

#### 概要

NIM は AI モデルの推論をマイクロサービスとして提供する NVIDIA のソフトウェアプラットフォームであり、デプロイメント時間を数週間から数分に短縮する。

#### アーキテクチャ

```
┌──────────────────────────────────────┐
│           NIM コンテナ               │
│  ┌──────────────┐  ┌──────────────┐ │
│  │ AI モデル    │  │ 最適化       │ │
│  │ (事前最適化) │  │ 推論エンジン │ │
│  │              │  │ (TensorRT-LLM│ │
│  │              │  │  / Triton)   │ │
│  └──────────────┘  └──────────────┘ │
│  ┌──────────────────────────────────┐│
│  │ 業界標準 API (OpenAI 互換)      ││
│  └──────────────────────────────────┘│
│  ┌──────────────────────────────────┐│
│  │ ランタイム依存関係               ││
│  └──────────────────────────────────┘│
└──────────────────────────────────────┘
```

#### 主な特徴

- **事前最適化コンテナ**: TensorRT-LLM, Triton Inference Server を含む最適化済みコンテナイメージ
- **業界標準 API**: OpenAI 互換 API を提供
- **マルチクラウド対応**: AWS, Google Cloud, Microsoft Azure, Oracle Cloud でネイティブサポート
- **Red Hat, Nutanix, VMware, Canonical** でも展開可能

#### 性能実績

- **Llama 3.1 8B (H100)**: 既成のデプロイ比で **2.6x スループット向上**（1,201 vs 613 tokens/sec）
- **DeepSeek-R1**: 2025年1月にプレビューマイクロサービスとして追加

#### サポートモデル（2025-2026年時点）

- Llama 3.x ファミリー
- DeepSeek-R1
- Mixtral / Mistral
- その他主要オープンソース LLM
- ドメイン特化モデル（医療、法律、コーディング等）

### 6.4 Triton Inference Server

#### 概要

Triton Inference Server は、NVIDIA が提供するオープンソースの推論サービングソフトウェアであり、あらゆるフレームワークのモデルを統一的にデプロイ可能にする。

#### 主な機能

| 機能 | 説明 |
|---|---|
| マルチフレームワーク | TensorRT, PyTorch, ONNX, OpenVINO, Python, RAPIDS FIL 等をサポート |
| Dynamic Batching | 推論リクエストを動的にバッチ化してスループットを最大化 |
| Model Ensemble | 複数モデルをパイプラインとして連結 |
| Concurrent Model Execution | 複数モデルを同一 GPU 上で同時実行 |
| Model Analyzer | モデルの最適な設定を自動探索 |
| Audio Streaming | リアルタイム音声ストリーミング推論 |
| HTTP/REST + gRPC + C API | 複数のクライアントインターフェース |

#### Dynamic Batching の詳細

- 個別の推論リクエストをサーバー側で動的にバッチ化
- 遅延時間を設定可能（より多くのリクエストを収集してバッチ化）
- Ensemble 内の各コンポーネントモデルでも個別に Dynamic Batching を適用可能

#### Model Ensemble

- 複数のモデルを入出力テンソルで連結し、パイプラインとして定義
- 前処理 → 推論 → 後処理の一連の流れを単一のリクエストで実行
- 各モデルに対して独立した最適化（バッチング、並列化）が可能

#### NIM との関係

Triton Inference Server は NIM の内部で推論エンジンとして使用されている。NIM は Triton + TensorRT-LLM + 最適化済みモデルをパッケージ化し、エンドユーザーが推論インフラの複雑さを意識せずにデプロイできるようにする抽象化レイヤーである。

---

## まとめ: NVIDIA 推論パイプラインの全体像

```
┌─────────────────────────────────────────────────────────────────────┐
│                    NVIDIA 推論パイプライン全体像                      │
│                                                                     │
│  [クライアントリクエスト]                                             │
│         │                                                           │
│         ▼                                                           │
│  ┌─────────────┐                                                    │
│  │ NVIDIA Dynamo│ ← Smart Router (KV Cache 認識ルーティング)        │
│  │ (オーケスト  │ ← GPU Resource Planner (SLO 駆動スケーリング)     │
│  │  レーション) │                                                    │
│  └──────┬──────┘                                                    │
│         │                                                           │
│    ┌────▼────┐         ┌──────────┐                                 │
│    │ Prefill │─KV転送─→│ Decode   │  ← Disaggregated Serving       │
│    │ GPU Pool│         │ GPU Pool │                                 │
│    └─────────┘         └──────────┘                                 │
│         │                    │                                      │
│         ▼                    ▼                                      │
│  ┌──────────────────────────────────┐                               │
│  │      TensorRT-LLM                │                               │
│  │  ・In-Flight Batching            │                               │
│  │  ・Paged KV Cache                │                               │
│  │  ・Speculative Decoding          │                               │
│  │  ・量子化 (FP8/FP4/INT4)         │                               │
│  │  ・Tensor/Pipeline Parallelism   │                               │
│  └──────────────────────────────────┘                               │
│                    │                                                │
│         ┌──────────┼──────────┐                                     │
│         ▼          ▼          ▼                                     │
│   ┌──────────┐ ┌────────┐ ┌──────────────┐                         │
│   │ Blackwell│ │ Rubin  │ │ GB200 NVL72  │  ← ハードウェア          │
│   │ B200/B300│ │ CPX    │ │ / NVL144 CPX │                         │
│   └──────────┘ └────────┘ └──────────────┘                         │
│                                                                     │
│  ┌──────────────────────────────────────┐                           │
│  │ NIM (Inference Microservices)        │ ← デプロイメント抽象化     │
│  │  ・事前最適化コンテナ                 │                           │
│  │  ・OpenAI 互換 API                   │                           │
│  │  ・Triton Inference Server 内蔵      │                           │
│  └──────────────────────────────────────┘                           │
└─────────────────────────────────────────────────────────────────────┘
```

### 推論パイプラインの進化のポイント (2025-2026)

1. **Disaggregated Serving が標準に**: Prefill/Decode の分離が事実上の業界標準
2. **推論特化ハードウェアの登場**: Rubin CPX はトレーニングとは明確に異なる推論専用 GPU
3. **FP4 量子化の普及**: Blackwell の NVFP4 により、品質を維持した 4-bit 推論が実用化
4. **インテリジェントなオーケストレーション**: NVIDIA Dynamo による SLO 駆動の動的リソース管理
5. **1,000+ TPS/user の実現**: Blackwell + TensorRT-LLM の組み合わせで人間の読書速度を大幅に超える生成速度

---

## 参考資料

### 推論 vs トレーニング
- [Training vs. inference: The two worlds of AI compute](https://rcrtech.com/semiconductor-news/training-vs-inference-compute/)
- [AI Computing Tutorial: Training vs. Inference Compute Needs](https://economistwritingeveryday.com/2025/12/02/ai-computing-tutorial-training-vs-inference-compute-needs-and-gpu-vs-tpu-processors/)
- [A Systematic Characterization of LLM Inference on GPUs (arXiv)](https://arxiv.org/html/2512.01644v1)
- [Prefill vs. Decode Bottlenecks: SRAM-Frequency Tradeoffs (arXiv)](https://arxiv.org/html/2512.22066v1)

### KV キャッシュ
- [KV Cache Optimization: Memory Efficiency for Production LLMs](https://introl.com/blog/kv-cache-optimization-memory-efficiency-production-llms-guide)
- [Efficient Memory Management for Large Language Model Serving with PagedAttention (arXiv)](https://arxiv.org/abs/2309.06180)
- [vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention](https://blog.vllm.ai/2023/06/20/vllm.html)
- [How to calculate size of KV cache](https://www.rohan-paul.com/p/how-to-calculate-size-of-kv-cache)
- [Techniques for KV Cache Optimization in Large Language Models](https://www.omrimallis.com/posts/techniques-for-kv-cache-optimization/)

### Speculative Decoding
- [TensorRT-LLM Speculative Decoding Boosts Inference Throughput by up to 3.6x](https://developer.nvidia.com/blog/tensorrt-llm-speculative-decoding-boosts-inference-throughput-by-up-to-3-6x/)
- [Boost Llama 3.3 70B Inference Throughput 3x with TensorRT-LLM](https://developer.nvidia.com/blog/boost-llama-3-3-70b-inference-throughput-3x-with-nvidia-tensorrt-llm-speculative-decoding/)
- [Accelerating LLM Inference on NVIDIA GPUs with ReDrafter](https://machinelearning.apple.com/research/redrafter-nvidia-tensorrt-llm)
- [An Introduction to Speculative Decoding (NVIDIA)](https://developer.nvidia.com/blog/an-introduction-to-speculative-decoding-for-reducing-latency-in-ai-inference/)
- [Speculative Decoding - LLM Inference Handbook](https://bentoml.com/llm/inference-optimization/speculative-decoding)

### 推論最適化
- [Mastering LLM Techniques: Inference Optimization (NVIDIA)](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/)
- [Speed up inference with SOTA quantization techniques in TRT-LLM](https://nvidia.github.io/TensorRT-LLM/blogs/quantization-in-TRT-LLM.html)
- [Quantization Tradeoffs: 4-bit vs 8-bit vs FP16 in Production](https://dasroot.net/posts/2026/02/quantization-tradeoffs-4-bit-8-bit-fp16-production/)
- [Scaling LLM Inference: Innovations in Tensor Parallelism (Meta)](https://engineering.fb.com/2025/10/17/ai-research/scaling-llm-inference-innovations-tensor-parallelism-context-parallelism-expert-parallelism/)
- [Introducing New KV Cache Reuse Optimizations in TensorRT-LLM](https://developer.nvidia.com/blog/introducing-new-kv-cache-reuse-optimizations-in-nvidia-tensorrt-llm/)

### ネットワーク・Disaggregated Serving
- [Characterizing Communication Patterns in Distributed LLM Inference (arXiv)](https://arxiv.org/html/2507.14392v1)
- [Disaggregated Inference: 18 Months Later](https://haoailab.com/blogs/distserve-retro/)
- [Disaggregated Prefill and Decode (Perplexity)](https://www.perplexity.ai/hub/blog/disaggregated-prefill-and-decode)
- [TraCT: Disaggregated LLM Serving with CXL Shared Memory (arXiv)](https://arxiv.org/abs/2512.18194)

### NVIDIA Dynamo
- [NVIDIA Dynamo: Low-Latency Distributed Inference Framework](https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models/)
- [KV Cache Aware Routing (NVIDIA Dynamo Documentation)](https://docs.nvidia.com/dynamo/latest/user-guides/kv-cache-aware-routing)
- [Scaling multi-node LLM inference with NVIDIA Dynamo on AKS](https://blog.aks.azure.com/2025/10/24/dynamo-on-aks)
- [NVIDIA Dynamo Planner: SLO-Driven Automation (InfoQ)](https://www.infoq.com/news/2026/01/nvidia-dynamo-ai-kubernetes/)

### Blackwell
- [NVIDIA Blackwell Architecture](https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/)
- [Blackwell Breaks the 1,000 TPS/User Barrier with Llama 4 Maverick](https://developer.nvidia.com/blog/blackwell-breaks-the-1000-tps-user-barrier-with-metas-llama-4-maverick/)
- [NVIDIA Blackwell Delivers World-Record DeepSeek-R1 Inference Performance](https://developer.nvidia.com/blog/nvidia-blackwell-delivers-world-record-deepseek-r1-inference-performance/)
- [SemiAnalysis InferenceMAX: vLLM and NVIDIA Accelerate Blackwell](https://blog.vllm.ai/2025/10/09/blackwell-inferencemax.html)
- [NVIDIA Blackwell Delivers Massive Performance Leaps in MLPerf Inference v5.0](https://developer.nvidia.com/blog/nvidia-blackwell-delivers-massive-performance-leaps-in-mlperf-inference-v5-0/)
- [NVIDIA Blackwell Ultra Sets New Inference Records in MLPerf](https://developer.nvidia.com/blog/nvidia-blackwell-ultra-sets-new-inference-records-in-mlperf-debut/)

### Rubin CPX
- [NVIDIA Unveils Rubin CPX](https://nvidianews.nvidia.com/news/nvidia-unveils-rubin-cpx-a-new-class-of-gpu-designed-for-massive-context-inference)
- [NVIDIA Rubin CPX Accelerates Inference Performance for 1M+ Token Context](https://developer.nvidia.com/blog/nvidia-rubin-cpx-accelerates-inference-performance-and-efficiency-for-1m-token-context-workloads/)
- [Inside the NVIDIA Rubin Platform: Six New Chips](https://developer.nvidia.com/blog/inside-the-nvidia-rubin-platform-six-new-chips-one-ai-supercomputer)
- [A Deep Dive into NVIDIA Rubin CPX](https://www.chiplog.io/p/a-deep-dive-into-nvidia-rubin-cpx)

### NIM / Triton
- [NVIDIA NIM Microservices](https://www.nvidia.com/en-us/ai-data-science/products/nim-microservices/)
- [NVIDIA NIM Offers Optimized Inference Microservices for Deploying AI Models at Scale](https://developer.nvidia.com/blog/nvidia-nim-offers-optimized-inference-microservices-for-deploying-ai-models-at-scale/)
- [NVIDIA Triton Inference Server Documentation](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/index.html)
- [Triton Architecture](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/architecture.html)

### GB200 NVL72 ベンチマーク
- [Deploying DeepSeek on GB200 NVL72: 3.8x Prefill, 4.8x Decode (LMSYS)](https://lmsys.org/blog/2025-09-25-gb200-part-2/)
- [How NVIDIA GB200 NVL72 and Dynamo Boost MoE Inference](https://developer.nvidia.com/blog/how-nvidia-gb200-nvl72-and-nvidia-dynamo-boost-inference-performance-for-moe-models/)
- [3x Faster AllReduce with NVSwitch and TensorRT-LLM MultiShot](https://developer.nvidia.com/blog/3x-faster-allreduce-with-nvswitch-and-tensorrt-llm-multishot/)
