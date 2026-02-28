# NVIDIA AI データセンター技術 詳細調査

調査日: 2026-02-27

---

## 1. RDMA (Remote Direct Memory Access) とは何か

### 基本概念

RDMA (Remote Direct Memory Access) とは、あるコンピュータのメモリから別のコンピュータのメモリへ、CPU やオペレーティングシステムを介さずに直接データを転送する技術である。

#### 例え話: 荷物の配送

従来の TCP/IP 通信を「宅配便」に例えると:
1. 送り主（アプリケーション）が荷物を玄関に出す（ユーザー空間 → カーネル空間へコピー）
2. 宅配業者（CPU）が荷物を受け取り、伝票を貼る（TCP/IP ヘッダの付与）
3. 配送センター（ネットワークスタック）で仕分けする
4. 届け先で逆の手順を踏む

RDMA は「専用のベルトコンベア」のようなもの:
- 送り主の倉庫から届け先の倉庫まで直通のベルトコンベアが設置されている
- 荷物を載せれば、誰の手も借りずに自動的に届く
- 宅配業者（CPU）は他の仕事に集中できる

### 従来の TCP/IP 通信との違い

| 項目 | TCP/IP | RDMA |
|------|--------|------|
| データ経路 | アプリ → カーネル → NIC → ネットワーク → NIC → カーネル → アプリ | アプリのメモリ → NIC → ネットワーク → NIC → アプリのメモリ |
| CPU 関与 | 全パケットの処理に CPU が必要 | 初期設定後は CPU 不要 |
| メモリコピー | 複数回のコピーが発生 | ゼロコピー（コピーなし） |
| レイテンシ | 数十〜数百マイクロ秒 | 1〜5 マイクロ秒（InfiniBand で 1μs 以下） |
| CPU 使用率 | 高い（パケット処理に消費） | 極めて低い |
| カーネルバイパス | なし | あり（OS カーネルを迂回） |

### なぜ AI トレーニングで重要なのか

AI トレーニングでは、数千の GPU が頻繁にデータ（勾配情報）を交換する必要がある。従来の TCP/IP では:
- CPU がネットワーク処理に時間を取られる
- データコピーの回数が多く、レイテンシが増大する
- GPU が通信完了を待つ「アイドル時間」が発生する

RDMA を使うことで、これらのオーバーヘッドが大幅に削減され、GPU の計算能力を最大限に活用できる。

### GPUDirect RDMA がどう改善するか

通常の RDMA でも、GPU メモリのデータを送信するには以下の手順が必要だった:
1. GPU メモリ → CPU のシステムメモリにコピー
2. システムメモリ → NIC 経由で RDMA 送信

**GPUDirect RDMA** はこのボトルネックを解消する:
- NIC が GPU メモリ（VRAM）に**直接アクセス**できる
- GPU メモリ → NIC → ネットワーク → NIC → リモート GPU メモリという**直通経路**を実現
- CPU のシステムメモリを一切経由しない「ゼロコピー転送」
- PCIe バス上の memory-mapped I/O (MMIO) を使って NIC が GPU の BAR1 メモリ領域に直接読み書きする

#### 実績例
- Cloudian が PyTorch に GPUDirect 対応を追加し、データ処理性能が **74% 向上**、CPU 使用率が **43% 削減**
- Meta は 24,000 GPU クラスタで GPUDirect RDMA over RoCE を使用して Llama モデルをトレーニング
- IBM も Granite モデルのトレーニングに GPUDirect RDMA を採用

### 情報ソース
- https://developer.nvidia.com/gpudirect
- https://docs.nvidia.com/cuda/gpudirect-rdma/
- https://www.weka.io/learn/glossary/gpu/what-is-gpudirect-rdma/
- https://www.fs.com/blog/a-quick-look-at-the-differences-rdma-vs-tcpip-2945.html
- https://www.digitalocean.com/community/conceptual-articles/rdma-high-performance-networking
- https://cloudswit.ch/blogs/rdma-and-tcpip-the-ultimate-guide-for-aidc/
- https://developers.redhat.com/articles/2025/04/29/accelerate-model-training-openshift-ai-nvidia-gpudirect-rdma

---

## 2. Collective Communication（集団通信）とは何か

### 基本概念

Collective Communication（集団通信）とは、分散コンピューティングにおいて複数のプロセス（GPU）が協調してデータを交換・集約する通信パターンのことである。NVIDIA の **NCCL (NVIDIA Collective Communications Library)** がこれらの操作を最適化して提供している。

#### 例え話: クラスの平均点計算

30人のクラスで全員のテストの平均点を計算する場面を想像してほしい:

- **AllReduce（全体集約）**: 全員が自分の点数を出し合い、合計して平均を計算し、**全員が最終結果を受け取る**。「みんなの点数を集めて平均を出して、その結果を全員に配る」操作。
- **AllGather（全体収集）**: 全員が自分の点数を出し合い、**全員が全員分の点数一覧を受け取る**。「全員の答案用紙のコピーを全員に配る」操作。
- **ReduceScatter（集約分散）**: 全員の点数を集めて合計を計算するが、結果の**一部分だけを各人が受け取る**。「合計を分担して、1番の人は国語の合計、2番の人は数学の合計を担当」のような操作。

### 各操作の詳細

#### AllReduce

- **処理内容**: k 個の GPU がそれぞれ N 個の値を持つ配列を提供し、要素ごとの演算（通常は合計）を行い、**全 GPU が同一の結果を受け取る**
- **AI での用途**: データ並列トレーニングで、各 GPU が計算した勾配（gradient）を全 GPU で合計して平均化する
- **アルゴリズム**: Ring AllReduce が代表的。各 GPU がリング状に接続され、ReduceScatter フェーズで部分的に集約し、AllGather フェーズで結果を全体に配布する

#### AllGather

- **処理内容**: k 個の GPU がそれぞれ N 個の値を持ち、すべてを集めて k×N のバッファを**全 GPU に配布**する
- **AI での用途**: Tensor Parallelism で分割されたモデルの重みを、必要に応じて全 GPU に集約する

#### ReduceScatter

- **処理内容**: 全 GPU のデータを演算で集約するが、結果を均等に分割して**各 GPU に 1 ブロックずつ配布**する
- **AI での用途**: ZeRO（DeepSpeed）での optimizer state の分散管理に使用される

### なぜ AI トレーニングのボトルネックになるか

1. **同期的な性質**: AllReduce は全 GPU の完了を待つ必要があり、最も遅い GPU がボトルネックになる
2. **線形スケーリング**: AllReduce の完了時間は GPU 数に対して線形に増加する。少数 GPU では問題ないが、数千 GPU では深刻な遅延になる
3. **バースト的なトラフィック**: AI トレーニングの通信は「計算 → 一斉に通信 → 計算」というパターンで、通信フェーズに大量のデータが集中する
4. **Incast 問題**: 多数の GPU が同時にスイッチにデータを送り、輻輳が発生する

### 数千 GPU での通信量

具体的な数値例:

| モデル | パラメータ数 | 1 イテレーションあたりの勾配データ量（FP16） |
|--------|------------|-------------------------------------------|
| ResNet-50 | 2,500万 | 約 100 MB |
| 2.2 億パラメータモデル | 2.2 億 | 約 880 MB |
| GPT-3 175B | 1,750 億 | 約 350 GB（FP16） |
| Llama 3.1 405B | 4,050 億 | 約 810 GB（FP16） |

- データ並列で 1,000 GPU を使用する場合、各イテレーションで**全 GPU が上記のデータ量を交換**する必要がある
- Ring AllReduce では、各 GPU が送受信するデータ量は `2 × (N-1)/N × データサイズ` で、N が大きいとほぼ `2 × データサイズ` に近づく
- 1,000 GPU × 880 MB の場合、ネットワーク全体で数百 GB〜数 TB のトラフィックが毎イテレーション発生する

### NCCL の最適化

NCCL 2.27 以降では:
- SHARP による AllGather と ReduceScatter のサポートが追加
- LLM トレーニングでは AllReduce よりも AllGather + ReduceScatter の組み合わせが好まれるようになった（計算と通信のオーバーラップが容易なため）

### 情報ソース
- https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html
- https://developer.nvidia.com/blog/enabling-fast-inference-and-resilient-training-with-nccl-2-27/
- https://arxiv.org/html/2510.20171v1
- https://hedgehog.cloud/blog/collective-communications-explained-the-hidden-coordination-behind-distributed-ai-training
- https://medium.com/@pranay.janupalli/introduction-to-nccl-communication-operators-the-backbone-of-efficient-distributed-training-d8b4b2f990a6
- https://www.keysight.com/blogs/en/inds/ai/benchmarking-collective-operations

---

## 3. AI ファクトリとは何か

### NVIDIA の定義

**AI ファクトリ**とは、NVIDIA CEO Jensen Huang が提唱した概念で、「インテリジェンス（知性）を大規模に製造する工場」としてのデータセンターを指す。従来のデータセンターがデータの保存・処理を行うのに対し、AI ファクトリは**AI トークン（推論結果）を生産する施設**である。

#### 例え話: 自動車工場との類比

| 項目 | 自動車工場 | AI ファクトリ |
|------|-----------|-------------|
| 原材料 | 鉄鋼・部品 | データ（テキスト、画像、センサーデータ） |
| 製造装置 | プレス機・ロボット | GPU クラスタ・高速ネットワーク |
| 製品 | 自動車 | AI トークン（予測・推論結果・意思決定） |
| エネルギー | 電力 → 動力 | 電力 → 計算能力 → インテリジェンス |
| 生産性指標 | 台数/時間 | トークンスループット |

自動車工場が鉄を車に変えるように、AI ファクトリは**電力をインテリジェンスに変える**施設である。

### 従来のデータセンターとの違い

| 項目 | 従来のデータセンター | AI ファクトリ |
|------|-------------------|-------------|
| 主な目的 | 多様なワークロードの処理・データ保存 | AI モデルのトレーニング・推論に特化 |
| 計算資源 | 汎用 CPU 中心 | GPU 中心（10 万 GPU 以上） |
| ネットワーク | 標準的な Ethernet | 高帯域・低遅延ネットワーク（InfiniBand / Spectrum-X） |
| 規模 | 数 MW〜数十 MW | ギガワット級（1 GW 以上） |
| 電力効率 | PUE 重視 | GPU あたりのトークンスループット重視 |
| 冷却 | 空冷中心 | 液冷が主流（高密度 GPU の発熱対策） |
| 成果物 | Web サービス・データ処理結果 | AI モデル・リアルタイム推論結果 |

### 2025 年の動向

- NVIDIA は Siemens, Schneider Electric, Trane などと連携し、**AI ファクトリのリファレンスデザイン**を策定
- デジタルツインを使った設計、電力・冷却・制御アーキテクチャの標準化
- **800V DC 電力アーキテクチャ**の推進（効率性・スケーラビリティの向上）
- マルチギガワット規模の AI ファクトリを、より迅速かつ効率的に展開可能にすることが目標

### 情報ソース
- https://blogs.nvidia.com/blog/ai-factory/
- https://www.nvidia.com/en-us/solutions/ai-factories/
- https://www.datacenterfrontier.com/machine-learning/article/55286658/inside-nvidias-vision-for-ai-factories-wade-vinsons-data-center-world-2025-keynote
- https://www.datacenterfrontier.com/design/article/55338673/nvidia-and-partners-define-a-repeatable-blueprint-for-ai-factory-data-centers
- https://developer.nvidia.com/blog/ai-factories-physical-ai-and-advances-in-models-agents-and-infrastructure-that-shaped-2025

---

## 4. なぜ AI トレーニングに大量の GPU が必要なのか

### LLM のパラメータ数とメモリ要件

#### パラメータとは

ニューラルネットワークの「パラメータ」とは、モデルが学習する重み（weight）とバイアス（bias）の数値のこと。パラメータ数が多いほどモデルが複雑なパターンを学習できるが、それだけメモリと計算資源が必要になる。

#### 例え話: 百科事典の暗記

- 7B パラメータのモデル = 70 億個の数値を記憶する脳
- 70B パラメータのモデル = 700 億個 → 10 倍の記憶容量が必要
- 405B パラメータのモデル = 4,050 億個 → さらに巨大

これは「百科事典 1 冊を暗記するか、図書館全体を暗記するか」の違いに近い。

### 1 GPU では足りない理由: 具体的な数値

#### 推論時のメモリ要件（モデルの重みだけ）

| モデル | パラメータ数 | FP16 メモリ | FP32 メモリ |
|--------|------------|------------|------------|
| Llama 3 8B | 80 億 | 16 GB | 32 GB |
| Llama 3 70B | 700 億 | 140 GB | 280 GB |
| Llama 3.1 405B | 4,050 億 | 810 GB | 1,620 GB |
| GPT-3 | 1,750 億 | 350 GB | 700 GB |

※ FP16: 1 パラメータ = 2 バイト、FP32: 1 パラメータ = 4 バイト

#### トレーニング時のメモリ要件（推論の約 4〜6 倍）

トレーニングでは、モデルの重みに加えて以下のデータがメモリに必要:

| 項目 | FP16/FP32 混合精度での容量（パラメータ数 Φ あたり） |
|------|--------------------------------------------------|
| モデルパラメータ（FP16） | 2Φ バイト |
| 勾配（FP16） | 2Φ バイト |
| Optimizer の FP32 パラメータコピー | 4Φ バイト |
| Optimizer のモーメンタム（Adam, FP32） | 4Φ バイト |
| Optimizer の分散（Adam, FP32） | 4Φ バイト |
| **合計** | **16Φ バイト** |

つまり、**トレーニング時は推論時の約 8 倍のメモリ**が必要（FP16 推論の 2Φ に対して 16Φ）。

#### 具体的な計算例

**Llama 3.1 405B のトレーニング:**
- パラメータ数: 4,050 億 = 4.05 × 10^11
- 必要メモリ: 16 × 4.05 × 10^11 バイト = **約 6.48 TB**
- さらに Activation メモリが数 TB 追加で必要（バッチサイズ・シーケンス長に依存）

**NVIDIA H100 80GB の場合:**
- 1 GPU のメモリ: 80 GB
- 必要な GPU 数（メモリだけで）: 6,480 GB ÷ 80 GB = **最低 81 GPU**
- 実際には Activation メモリ、通信バッファ等を含めると **数百〜数千 GPU** が必要

**実際の事例:**
- Meta は Llama 3.1 405B を **16,384 台の H100 GPU** で約 54 日間トレーニング
- GPT-4 は推定 **約 25,000 台の A100 GPU** で 90〜100 日間トレーニング
- トレーニング中、平均 3 時間に 1 回のハードウェア障害が発生（Meta の報告、419 件の障害）

### なぜ「速度」のためにも多くの GPU が必要か

メモリだけでなく、計算速度の面でも大量の GPU が不可欠:

- Llama 3.1 405B のトレーニングは 16,384 GPU で 54 日かかった
- 仮に 1,000 GPU で行うと（線形スケーリングと仮定）、約 **885 日（2.4 年）** かかる計算
- 1 GPU では物理的に数百年〜数千年かかり、現実的に不可能

### 情報ソース
- https://www.propelrc.com/llm-gpu-vram-requirements-explained/
- https://www.hyperstack.cloud/blog/case-study/how-much-vram-do-you-need-for-llms
- https://medium.com/@xiaxiami/calculating-memory-footprint-for-large-language-models-llms-a-complete-guide-98ac3fdfdbf6
- https://shreyans92.github.io/2025-05-23-LLMMemory/
- https://ai.meta.com/blog/meta-llama-3-1/
- https://www.tomshardware.com/tech-industry/artificial-intelligence/faulty-nvidia-h100-gpus-and-hbm3-memory-caused-half-of-the-failures-during-llama-3-training-one-failure-every-three-hours-for-metas-16384-gpu-training-cluster

---

## 5. RoCE (RDMA over Converged Ethernet) とは

### 基本概念

**RoCE (RDMA over Converged Ethernet)** は、標準的な Ethernet ネットワーク上で RDMA を実現するプロトコルである。InfiniBand が専用のネットワークインフラを必要とするのに対し、RoCE は既存の Ethernet インフラを活用できる。

#### 例え話: 高速道路 vs 一般道の改良

- **InfiniBand** = 専用の高速道路。最初から高速走行のために設計されており、信号も渋滞もない。ただし建設コストが高く、既存の道路とは接続しにくい。
- **RoCE** = 一般道を高速道路並みに改良したもの。既存の道路インフラを活用しつつ、信号の最適化や車線拡張で高速走行を可能にする。コスト効率は良いが、完全な専用高速道路にはやや劣る。

### InfiniBand の RDMA と RoCE の違い

| 項目 | InfiniBand | RoCE v2 |
|------|-----------|---------|
| ネットワーク基盤 | 専用のInfiniBand ファブリック | 標準 Ethernet |
| レイテンシ | 1μs 以下（サブマイクロ秒） | 約 5μs |
| 設計思想 | 最初から RDMA 用に設計 | Ethernet に RDMA を適応 |
| コスト | 高い（専用ハードウェア） | 比較的低い（既存 Ethernet 活用） |
| TCO（3年間） | 高い | InfiniBand 比で **55% 削減**（Juniper の報告） |
| 輻輳制御 | ハードウェアレベルで組込み済み | ソフトウェア・追加設定が必要 |
| 適応ルーティング | ネイティブサポート | 従来は限定的（Spectrum-X で改善） |
| パケットロス | ロスレス設計 | PFC（Priority Flow Control）で対応 |
| ベンダーロックイン | NVIDIA（旧 Mellanox）がほぼ独占 | 複数ベンダーから選択可能 |
| 市場シェア（2023年） | AI トレーニングの約 80% | 約 20% |
| 市場シェア（2025年中頃） | 減少傾向 | **Ethernet がリード**（Meta 等の大規模採用） |

### Spectrum-X がどのように RoCE を改善しているか

NVIDIA **Spectrum-X** は、Ethernet ベースの AI ネットワーキングを InfiniBand に近い性能に引き上げるプラットフォームである。主な改善点:

#### 1. RoCE Adaptive Routing（適応型ルーティング）
- 従来の Ethernet は ECMP（Equal-Cost Multi-Path）で固定的に経路を振り分けていた
- Spectrum-X は **フロー単位で動的に経路を変更**し、輻輳ポイントを回避
- ネットワーク資源の利用効率を最大化

#### 2. RoCE Congestion Control（輻輳制御）
- **In-band Network Telemetry** でスイッチからリアルタイムのネットワーク状態データを収集
- BlueField-3 SuperNIC が収集したテレメトリデータを**深層学習モデルで分析**し、最適な送信レートを決定
- 従来の ECN/PFC ベースの輻輳制御より高精度

#### 3. RoCE Performance Isolation（性能分離）
- マルチテナント環境で、テナント間の干渉を防止
- 各 AI ワークロードに対して予測可能な性能を保証

#### 4. 結果
- 従来の Ethernet 比で **2 倍の実効ネットワーク性能**
- ハイパースケール環境で **95% の実効帯域幅**を達成
- Meta は「RoCE と InfiniBand をチューニングして同等の性能を達成した」と報告

### 情報ソース
- https://developer.nvidia.com/blog/turbocharging-ai-workloads-with-nvidia-spectrum-x-networking-platform/
- https://www.naddod.com/blog/spectrum-x-nvidia-s-answer-to-gen-ai-ethernet-challenges
- https://cloudswit.ch/blogs/roce-or-infiniband-technical-comparison/
- https://www.vitextech.com/blogs/blog/infiniband-vs-ethernet-for-ai-clusters-effective-gpu-networks-in-2025
- https://www.juniper.net/content/dam/www/assets/white-papers/us/en/2024/juniper-artificial-intelligence-data-center-comparison-of-infiniband-and-rdma-over-converged-ethernet.pdf
- https://www.trendforce.com/insights/infiniband-vs-ethernet

---

## 6. In-Network Computing / SHARP の仕組み

### 基本概念

**SHARP (Scalable Hierarchical Aggregation and Reduction Protocol)** は、NVIDIA のスイッチ上で集団通信の計算処理を直接実行する In-Network Computing 技術である。

#### 例え話: 投票の集計

1,000 人が全国投票の結果を知りたい場合を考える:

**従来の方法（ホストベースの AllReduce）:**
1. 全国の各投票所が全票を中央選管に送る（大量のデータが中央に集中）
2. 中央選管が集計する
3. 結果を全投票所に送り返す
4. → ネットワーク上を大量のデータが行き来する

**SHARP（In-Network Computing）:**
1. 各地域のスイッチ（= 地方選管）が、その地域の票をその場で集計する
2. 集計結果（票数だけ）を上位のスイッチに送る
3. 最上位で最終集計し、結果を下流に配布する
4. → ネットワーク上を流れるデータ量が大幅に削減される

### 具体的な仕組み

#### Aggregation Tree（集約ツリー）

SHARP は物理的なネットワークトポロジーの上に**論理的な集約ツリー**を構築する:

```
        [Root スイッチ]          ← 最終集約
       /              \
  [Spine スイッチ]  [Spine スイッチ]  ← 中間集約
   /      \          /      \
[Leaf]  [Leaf]   [Leaf]  [Leaf]    ← 初段集約
 |  |    |  |     |  |    |  |
GPU GPU  GPU GPU  GPU GPU  GPU GPU  ← データ源
```

#### 2 フェーズの処理

1. **Reduce フェーズ（上り）**: 各 GPU からデータがツリーの葉から根に向かって流れる。各スイッチノードで**受信したデータをその場で集約**（合計、最大値など）して、集約結果だけを上位に転送する。
2. **Broadcast フェーズ（下り）**: 根ノードの最終結果をツリーに沿って全 GPU に配布する。

### データ量が半減する理由

従来の Ring AllReduce と SHARP の比較:

#### 従来の Ring AllReduce（N 台の GPU の場合）

- 各 GPU がデータを隣の GPU に送信し、リングを一周する
- ReduceScatter フェーズ: 各 GPU が N-1 回の送受信を行う
- AllGather フェーズ: 各 GPU がさらに N-1 回の送受信を行う
- **ネットワーク上の総データ転送量**: 約 `2 × (N-1) × データサイズ`
- データが何度もネットワークリンクを通過する

#### SHARP の In-Network Aggregation

- 各 GPU はデータを**1 回だけ**ネットワークに送信する
- スイッチが途中でデータを集約するため、**上位に行くほどデータ量が減る**
- 例: 8 GPU が各 100 MB を送信する場合
  - Leaf スイッチ: 4 GPU × 100 MB を受信 → 集約して 100 MB だけ上位へ送信
  - Spine スイッチ: 2 × 100 MB を受信 → 集約して 100 MB だけ Root へ送信
  - Root: 最終結果 100 MB を下流に配布
- **結果としてネットワーク全体のトラフィックが約半分に削減**

#### 性能数値

- 8 バイトの MPI_Allreduce() で 128 ホスト: 6.01μs → **2.83μs**（2.1 倍の改善）
- 中〜大サイズのデータでは、リダクション帯域幅が **2〜5 倍** 向上
- 実際の AI ワークロードで **10〜20% の性能改善** が報告されている

### NCCL との連携

NCCL 2.27 以降、SHARP は AllGather と ReduceScatter もサポートするようになり、LLM トレーニングでの活用範囲が拡大した。

### 情報ソース
- https://developer.nvidia.com/blog/advancing-performance-with-nvidia-sharp-in-network-computing/
- https://training.continuumlabs.ai/infrastructure/networking-and-connectivity/scalable-hierarchical-aggregation-and-reduction-protocol-sharp
- https://lambda.ai/blog/nvidia-sharp-on-lambda-1cc
- https://docs.nvidia.com/networking/display/sharpv300
- https://network.nvidia.com/pdf/solutions/hpc/paperieee_copyright.pdf
- https://link.springer.com/chapter/10.1007/978-3-030-50743-5_3

---

## 7. NVLink と PCIe の違い

### 基本概念

**NVLink** は NVIDIA が開発した GPU 間の高速インターコネクト技術で、**PCIe (Peripheral Component Interconnect Express)** は汎用的なコンポーネント間接続規格である。

#### 例え話: 水道管の太さ

- **PCIe Gen5** = 一般家庭の水道管。日常的な使用には十分だが、プールを満たすには時間がかかる。
- **NVLink 5** = 工業用の大口径パイプ。PCIe の 14 倍の太さがあり、大量の水を高速に送れる。

GPU 間のデータ転送を「水の流れ」に例えると、AI トレーニングでは「プール何杯分もの水を秒単位で移動させる」必要があり、家庭用水道管（PCIe）では到底追いつかない。

### 具体的な帯域幅の比較

| 世代 | 技術 | 双方向帯域幅（1 リンク / 1 GPU） | 対 PCIe 比 |
|------|------|-------------------------------|-----------|
| — | PCIe Gen3 x16 | 約 32 GB/s | 1x (基準) |
| — | PCIe Gen4 x16 | 約 64 GB/s | 2x |
| — | PCIe Gen5 x16 | 約 128 GB/s | 4x |
| — | PCIe Gen6 x16 | 約 256 GB/s | 8x |
| 2016 | NVLink 1.0 (Pascal P100, 4 リンク) | 160 GB/s（総帯域） | 5x vs Gen3 |
| 2017 | NVLink 2.0 (Volta V100, 6 リンク) | 300 GB/s（総帯域） | 9x vs Gen3 |
| 2020 | NVLink 3.0 (Ampere A100, 12 リンク) | 600 GB/s（総帯域） | 9x vs Gen4 |
| 2022 | NVLink 4.0 (Hopper H100, 18 リンク) | 900 GB/s（総帯域） | 7x vs Gen5 |
| 2024 | **NVLink 5.0 (Blackwell, 18 リンク)** | **1,800 GB/s（1.8 TB/s, 総帯域）** | **14x vs Gen5** |
| 2026 | NVLink 6.0 (Rubin) | 3,600 GB/s（3.6 TB/s） | 14x vs Gen6 |

### なぜ PCIe では不十分なのか

#### 1. 帯域幅の絶対的な差

PCIe Gen5 の 128 GB/s に対し、NVLink 5.0 は **1,800 GB/s（14 倍）**。AI トレーニングで頻繁に交換される数百 GB のデータを考えると、この差は計算効率に直結する。

#### 2. GPU 間のダイレクト接続

- PCIe: GPU → PCIe バス → CPU（Root Complex）→ PCIe バス → GPU という経路で、CPU がボトルネックになりうる
- NVLink: GPU 間を**直接接続**し、CPU を介さない。NVSwitch を使えば全 GPU 間で均等な帯域幅を実現

#### 3. スケーラビリティ

- NVLink 5.0 + NVSwitch: GB200 NVL72 では **72 GPU を 1 つの NVLink ドメイン**として接続し、合計 **1 PB/s 以上**の総帯域幅を実現
- PCIe ではこのようなスケールの GPU 間接続は不可能

#### 4. 実際の影響

例えば、100 GB のモデル重みを GPU 間で転送する場合:
- PCIe Gen5: 100 GB ÷ 128 GB/s = 約 **0.78 秒**
- NVLink 5.0: 100 GB ÷ 1,800 GB/s = 約 **0.056 秒**

AI トレーニングでは毎イテレーション数百〜数千回のデータ交換が発生するため、この差の蓄積は膨大になる。

### NVLink の世代別進化まとめ

| 世代 | GPU アーキテクチャ | リンク数 | 1 リンク帯域幅 | 総帯域幅 |
|------|-------------------|---------|-------------|---------|
| NVLink 1.0 | Pascal (P100) | 4 | 40 GB/s | 160 GB/s |
| NVLink 2.0 | Volta (V100) | 6 | 50 GB/s | 300 GB/s |
| NVLink 3.0 | Ampere (A100) | 12 | 50 GB/s | 600 GB/s |
| NVLink 4.0 | Hopper (H100) | 18 | 50 GB/s | 900 GB/s |
| NVLink 5.0 | Blackwell (B200) | 18 | 100 GB/s | 1,800 GB/s |
| NVLink 6.0 | Rubin | — | — | 3,600 GB/s |

8 年間で帯域幅は **160 GB/s → 3,600 GB/s** と **22.5 倍** に成長。同期間の PCIe の成長は Gen3→Gen6 で **8 倍** にとどまり、GPU の帯域幅需要の増加に追いつけていない。

### 情報ソース
- https://en.wikipedia.org/wiki/NVLink
- https://intuitionlabs.ai/articles/nvidia-nvlink-gpu-interconnect
- https://hardwarenation.com/resources/blog/nvidia-nvlink-5-0-accelerating-multi-gpu-communication/
- https://network-switch.com/blogs/networking/the-evolution-of-nvidia-nvlink-technology
- https://www.nvidia.com/en-us/data-center/nvlink/
- https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/
- https://www.hpcwire.com/2024/03/25/nvlink-faster-interconnects-and-switches-to-help-relieve-data-bottlenecks/
