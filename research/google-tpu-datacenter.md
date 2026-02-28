# Google TPU によるデータセンター構成

調査日: 2026-02-28

---

## 1. TPU の最新世代と主要スペック

### 世代別スペック一覧

| 項目 | TPU v5e | TPU v5p | Trillium (v6e) | Ironwood (v7) |
|------|---------|---------|----------------|---------------|
| 発表時期 | 2023年 | 2023年12月 | 2024年 (GA: 2025年) | 2025年4月 |
| MXU アレイ | 128×128 | 128×128 | 256×256 | 非公開 |
| BF16 演算性能 | ~197 TFLOPS | ~459 TFLOPS | ~918 TFLOPS | 4,614 TFLOPS (FP8) |
| HBM 容量 | 16 GB | 96 GB (HBM2e) | 32 GB (HBM) | 192 GB (HBM3E) |
| HBM 帯域幅 | 非公開 | 2.8 TB/s | v5e の2倍 | 7.2 TB/s |
| ICI 帯域幅/chip | 1,600 Gbps | 4,800 Gbps | 非公開 | 9.6 Tbps (双方向) |
| ICI トポロジ | 2D torus | 3D torus | 2D/3D torus | 3D torus (4×4×4 cube) |
| Pod 最大チップ数 | 256 | 8,960 | 256 | 9,216 |
| 用途 | 推論・コスト効率重視 | 大規模トレーニング | 汎用（トレーニング+推論） | 大規模トレーニング+推論 |

### 各世代の進化ポイント

**TPU v5e（2023年）**
- コスト効率を重視した設計で、200B パラメータ以下のモデルに最適化
- 2D torus トポロジで4つの隣接チップに接続
- v4 比で最大2倍のトレーニングスループット、最大2.5倍の推論スループット

**TPU v5p（2023年12月）**
- v5e のハイエンド版。大規模モデルトレーニング向け
- 3D torus で6つの隣接チップに接続し、Pod あたり最大8,960チップ
- ICI 帯域 4,800 Gbps/chip でチップ間通信を高速化
- HBM 96 GB により大規模モデルのパラメータを保持可能

**Trillium / TPU v6e（2024年発表、2025年 GA）**
- v5e 比でチップあたりの演算性能が4.7倍に向上
- HBM 容量・帯域ともに2倍に増加
- MXU を 256×256 に拡大し、行列演算のスループットを大幅強化
- エネルギー効率も67%以上向上

**Ironwood / TPU v7（2025年4月発表）**
- Google が「推論の時代のための TPU」と位置づけた最新世代
- v5p 比で最大10倍のピーク性能
- v6e 比で4倍以上のトレーニング・推論効率
- 192 GB HBM3E により Trillium の6倍のメモリ容量
- 各チップに TensorCore ×2、SparseCore ×4 を搭載
- Watt あたり性能は Trillium の2倍
- 9,216チップ Pod で 42.5 ExaFLOPS のピーク演算性能
- Pod 全体で 1.77 PB の共有 HBM

---

## 2. TPU のインターコネクト技術

### ICI (Inter-Chip Interconnect)

ICI は Google が独自開発した TPU チップ間の高速シリアル通信リンクである。

**特徴:**
- マイクロ秒レベルの低レイテンシ
- テラビット/秒級の帯域幅
- 隣接チップ同士をカスタム高速シリアルリンクで直接接続
- 電気信号ベース（キューブ内はダイレクトアタッチ銅ケーブル、キューブ間は光トランシーバ）

**世代別 ICI 帯域:**
- TPU v4: 非公開（3D torus、6方向接続）
- TPU v5e: 1,600 Gbps/chip（2D torus、4方向接続）
- TPU v5p: 4,800 Gbps/chip（3D torus、6方向接続）
- Ironwood (v7): 9.6 Tbps 双方向（4本の ICI リンク、1.2 TB/s ピーク帯域）

### TPU Pod の構成

TPU Pod はICI で相互接続された TPU チップの集合体である。

**Pod 構成例（Ironwood）:**
- 基本単位: 4×4×4 = 64チップの「キューブ」
- 小規模 Pod: 4キューブ = 256チップ
- Superpod: 144キューブ = 9,216チップ
- キューブ内は銅ケーブルで直接接続
- キューブ間は光トランシーバで接続（TPU 1台あたり約1.5本の光トランシーバ）

**各世代の Pod 規模:**
- TPU v4: 最大 4,096チップ/Pod
- TPU v5e: 最大 256チップ/Pod
- TPU v5p: 最大 8,960チップ/Pod
- Ironwood: 最大 9,216チップ/Superpod

### マルチスライス (Multislice)

Cloud TPU Multislice は、複数の TPU スライスにまたがってトレーニングジョブをスケーリングする技術である。

**仕組み:**
- 同一 Pod 内の複数スライス、または複数 Pod にまたがるスライスを使用可能
- スライス内のチップ間通信: ICI（高速）
- スライス間通信: CPU（ホスト）を経由して DCN（Data Center Network）を使用
- 標準的なデータ並列処理をサポート
- Ironwood では 100,000 チップ以上のマルチスライス構成が可能
- 最大16個の ICI Pod を DCN で接続可能（4つの集約ブロック × 4 ICI Pod = 合計 147,456 TPU）

### Optical Circuit Switching (OCS) の役割

OCS は Google が TPU データセンターに深く統合している光回路スイッチ技術である。

**技術的特徴:**
- MEMS（微小電気機械システム）ベースのスイッチ
- 2D ミラーアレイ、レンズ、カメラを用いたビームステアリング
- 光ファイバの入力ポートを出力ポートに動的にマッピング
- 電気スイッチなしでトポロジを動的に再構成
- 1台の OCS で 144×144 ポートを処理可能

**メリット:**
- 電力消費: 電気スイッチ比で40%削減
- コスト: 30%削減
- 4,096 TPU デプロイメントの場合、InfiniBand では ~568台のスイッチが必要なところ、Google は48台の OCS で構成可能
- 信号が光ドメインに留まるため、変換オーバーヘッドが最小

**スケーリングにおける役割:**
- OCS がキューブ同士を接続し、Pod → Superpod へのスケーリングを実現
- 256チップ Pod（4キューブ）から 9,216チップ Superpod（144キューブ）まで柔軟に構成
- トポロジの動的再構成により、ワークロードに応じた最適な接続パターンを実現

---

## 3. TPU データセンターのネットワーク構成

### Jupiter ネットワークファブリック

Jupiter は Google が自社データセンター向けに開発したネットワークファブリックで、現在第5世代に達している。

**主要スペック:**
- 二分帯域幅: 13.1 Pb/s（13 ペタビット/秒）
- リンク速度: ネイティブ 400 Gb/s
- 前世代比: 5倍の速度・容量向上
- 設備投資: 30%削減
- 電力: 41%削減

**技術要素:**
- OCS（Optical Circuit Switching）と WDM（波長分割多重）を8年以上かけて深く統合
- SDN（Software-Defined Networking）アーキテクチャによる制御
- ゼロダウンタイムアップグレード対応
- リアルタイムのアプリケーション優先度・通信パターン制御
- ヘテロジニアス技術によるインクリメンタルなネットワーク構築をサポート

### Scale-up / Scale-out の構造

Google の TPU データセンターは、明確な Scale-up と Scale-out の階層構造を持つ。

**Scale-up（ICI レイヤー）:**
- チップ間を ICI で直接接続
- 3D torus トポロジ（v5p, Ironwood）
- 帯域: 最大 9.6 Tbps/chip（Ironwood）
- 範囲: 1 Pod/Superpod 内（最大 9,216チップ）
- 低レイテンシ、高帯域のオールトゥオール通信

**Scale-out（DCN レイヤー）:**
- マルチスライスで Pod 間を接続
- Jupiter ネットワークファブリックを使用
- 帯域: ICI より低い（DCN の帯域幅に依存）
- 範囲: 100,000チップ以上
- 最大 147,456 TPU（16 ICI Pod 接続時）
- ホスト CPU を経由したデータ転送

### 使用しているスイッチ・NIC

**全て Google 独自開発:**
- OCS: MEMS ベースの光回路スイッチ（144×144 ポート、Google 独自設計）
- Jupiter スイッチ: Google カスタム設計のネットワークスイッチ
- Titanium ネットワークアダプタ: Google 独自開発の NIC。A3（GPU）インスタンスのネットワーク帯域幅向上にも使用
- Titan セキュリティチップ: サーバ・ネットワーク機器のハードウェアルートオブトラスト

Google はスイッチ、NIC、光スイッチ、セキュリティチップに至るまで、データセンターのネットワーキングスタック全体を自社設計している。サードパーティのスイッチ（Arista、Cisco 等）やインターコネクト（InfiniBand 等）は TPU システムには使用していない。

### NVIDIA の Spectrum-X / InfiniBand との比較

| 項目 | Google TPU (ICI + Jupiter) | NVIDIA InfiniBand | NVIDIA Spectrum-X |
|------|---------------------------|-------------------|-------------------|
| プロトコル | 独自 ICI + Ethernet (DCN) | InfiniBand | Ethernet (拡張) |
| Scale-up 範囲 | 最大 9,216チップ/Superpod | 最大 72 GPU/NVL72 (NVLink) | NVLink + Ethernet |
| Scale-up 帯域 | 9.6 Tbps/chip (Ironwood) | 1.8 TB/s (NVLink, B200) | NVLink 依存 |
| Scale-out | Jupiter DCN (13.1 Pb/s) | InfiniBand (最大 800 Gbps/port) | Spectrum-X (800 Gbps) |
| スイッチ数 (4,096台) | ~48 OCS | ~568 IB スイッチ | 同等規模 |
| トポロジ | 3D torus (ICI) | Fat-tree (IB) | Fat-tree (Ethernet) |
| ベンダーロック | Google Cloud 専用 | マルチベンダー | マルチベンダー |
| 電力効率 | OCS で40%削減 | 電気スイッチベース | 電気スイッチベース |

**主要な違い:**
1. **スケール**: Google の ICI は1つのファブリックで最大 9,216チップを接続できるのに対し、NVIDIA の NVLink/NVSwitch は NVL72 で 72 GPU が上限。NVIDIA はそれ以上のスケールで InfiniBand/Spectrum-X に依存する
2. **スイッチ数**: Google の OCS は大幅にスイッチ数を削減。4,096台規模で Google が48台の OCS に対し、InfiniBand は約568台のスイッチが必要
3. **コスト・電力**: Google の光スイッチアプローチは電気スイッチ比で電力40%減、コスト30%減を実現
4. **オープン性**: NVIDIA はマルチベンダー・マルチクラウドで利用可能だが、Google の TPU + ICI + Jupiter は Google Cloud 専用

---

## 4. TPU の性能実績

### 大規模モデルトレーニング実績

**Gemini シリーズ:**
- Gemini 1.0, 1.5, 2.0, 3.0（最新のフロンティアモデル）はすべて TPU でトレーニング
- Gemini は Google の最先端 LLM として、最新世代は TPU v5p 以降でトレーニングされたと推定される
- マルチモーダル（テキスト、画像、音声、動画）対応

**PaLM (Pathways Language Model):**
- 6,144 TPU v4 チップでトレーニング
- MFU（Model FLOPs Utilization）: 46% を達成
- TPU v4 の MFU ベンチマークでは 44-56% を記録

**その他のモデル:**
- Google の内部サービス（検索、翻訳、YouTube など）の推論ワークロードにも TPU を大規模活用

### MLPerf ベンチマーク結果

**MLPerf Training v4.1（2024年11月）:**
- Google は Trillium (v6e) で GPT-3 結果を提出
- しかし 2,048 アクセラレータ構成で v5p 比わずか8%の高速化にとどまり、期待された4.7倍のスピードアップには到達しなかった（v5e 比であり v5p 比ではないため直接比較が困難）
- NVIDIA は Hopper で安定した結果を継続

**MLPerf Training v5.1（2025年11月）:**
- NVIDIA が全7テストで最速を記録し、完全勝利
- NVIDIA Blackwell Ultra (GB300 NVL72) で Llama 3.1 405B をわずか10分でトレーニング（5,000 以上の Blackwell GPU 使用）
- Hopper 比で Llama 3.1 405B 事前トレーニングが4倍以上、Llama 2 70B LoRA ファインチューニングが5倍近く高速化
- NVIDIA は全テストに唯一提出したベンダーであり、CUDA ソフトウェアスタックの成熟度と汎用性を示した
- Google は MLPerf v5.1 で目立った結果を出していない

**MLPerf 推論:**
- TPU v5e は一部のカテゴリでリーダー的存在
- 9カテゴリ中8カテゴリで TPU v5e がリード（推論）

### NVIDIA GPU との性能比較

**コスト効率（推論）:**
- Google は TPU v6e が GPU 比で4倍のコスト効率（$/performance）を主張
- 推論コストでは TPU が NVIDIA GPU より有利とされるレポートが複数存在

**トレーニング性能:**
- 絶対性能では NVIDIA Blackwell が MLPerf で一貫して最速
- Google は大規模スケーリング（数千チップ）での効率を強みとする
- TPU の ICI で 9,216チップを1ファブリックに接続できる点は、NVIDIA の NVL72（72 GPU）を大きく上回るスケール

**推論性能:**
- TPU v6e は Llama2-70B のトレーニングで TPU v5e 比4倍高速化
- Ironwood は推論特化の設計で、大規模推論ワークロードに最適化

---

## 5. 制約・課題

### Google Cloud 専用の制約

- TPU は Google Cloud Platform (GCP) でのみ利用可能
- オンプレミス設置は不可能
- データ主権要件やエアギャップ環境を持つ組織は TPU を使用できない
- マルチクラウド戦略を採る企業にとってベンダーロックインのリスク
- NVIDIA GPU は AWS、Azure、GCP、Oracle Cloud、オンプレミスなど幅広い環境で利用可能

### CUDA エコシステムとの互換性

- TPU は CUDA と完全に非互換。CUDA コードは TPU 上で動作しない
- CUDA ベースのカスタム C++ コード、MXNet、Caffe などのフレームワークは TPU 非対応
- NVIDIA の CUDA エコシステムは数十年の蓄積があり、ライブラリ（cuDNN、cuBLAS、TensorRT、NCCL 等）が極めて充実
- TPU は XLA コンパイラを介してのみプログラミング可能
- Pallas / Mosaic スタックでカスタムカーネル開発が可能だが、CUDA のライブラリ群と比較するとエコシステムは未成熟

### フレームワーク対応状況

**最適にサポートされるフレームワーク:**
- JAX: TPU ネイティブ対応。Google が開発しており最も TPU に最適化されている
- TensorFlow: TPU の元々のサポートフレームワーク

**制限付きサポート:**
- PyTorch: PyTorch/XLA を通じて TPU をサポートするが、TensorFlow/JAX ほどの最適化はされていない
  - 一部の PyTorch 操作に XLA 対応がなく、CPU フォールバックが発生して性能劣化
  - 動的制御フローがグラフコンパイルと相性が悪く、モデルアーキテクチャの変更が必要になることがある
- vLLM: 2025年に TPU バックエンドが統合され、PyTorch（Torchax 経由）と JAX の両方をサポート。JAX→XLA の統一的なローワリングパスを使用

**非対応:**
- MXNet
- Caffe
- カスタム CUDA C++ コード

### XLA / カスタムカーネルの制約

- XLA コンパイラは広範な最適化を提供するが、新しいアテンションメカニズムや動的テンソル向けカスタムパディングなど、最先端のアルゴリズムではコンパイラの能力を超えることがある
- Pallas / Mosaic によるカーネル開発は可能だが、CUDA の CUTLASS、Triton などと比較して成熟度が低い
- 研究者が新しい演算を試す際の柔軟性は GPU/CUDA の方が高い

### MLPerf での課題

- MLPerf Training v5.1 では NVIDIA が全カテゴリで勝利し、Google は目立った成果を出せなかった
- NVIDIA は全テストに提出する唯一のベンダーであり、TPU は提出カテゴリが限定的
- Trillium の MLPerf 結果が期待を下回った（v5p 比8%の改善にとどまった）

---

## まとめ

Google の TPU エコシステムは、チップからインターコネクト、ネットワークファブリック、ソフトウェアスタックに至るまで完全に垂直統合されたシステムである。特に ICI と OCS の組み合わせによる大規模スケーリング（9,216チップ/Superpod、147,456 TPU/マルチスライス）は、NVIDIA のアプローチ（NVL72 + InfiniBand）と比較してスイッチ数の大幅削減と電力・コスト効率で優位性がある。

一方で、Google Cloud 専用のクローズドエコシステムであること、CUDA 非互換、フレームワーク対応の限定性は重要な制約である。MLPerf でも NVIDIA Blackwell が絶対性能で優位を保っており、TPU の強みは大規模推論のコスト効率と Google 内部ワークロードでの実績にある。

---

## ソース

- [TPU v6e (Trillium) - Google Cloud Documentation](https://docs.cloud.google.com/tpu/docs/v6e)
- [Introducing Trillium, sixth-generation TPUs - Google Cloud Blog](https://cloud.google.com/blog/products/compute/introducing-trillium-6th-gen-tpus)
- [TPU v5p - Google Cloud Documentation](https://docs.cloud.google.com/tpu/docs/v5p)
- [TPU v5e - Google Cloud Documentation](https://docs.cloud.google.com/tpu/docs/v5e)
- [TPU7x (Ironwood) - Google Cloud Documentation](https://docs.cloud.google.com/tpu/docs/tpu7x)
- [Ironwood: The first Google TPU for the age of inference - Google Blog](https://blog.google/innovation-and-ai/infrastructure-and-cloud/google-cloud/ironwood-tpu-age-of-inference/)
- [Inside the Ironwood TPU codesigned AI stack - Google Cloud Blog](https://cloud.google.com/blog/products/compute/inside-the-ironwood-tpu-codesigned-ai-stack)
- [TPU architecture - Google Cloud Documentation](https://docs.cloud.google.com/tpu/docs/system-architecture-tpu-vm)
- [Cloud TPU Multislice Overview - Google Cloud Documentation](https://docs.cloud.google.com/tpu/docs/multislice-introduction)
- [Jupiter Evolving: Transforming Google's Datacenter Network - Google Research](https://research.google/pubs/jupiter-evolving-transforming-googles-datacenter-network-via-optical-circuit-switches-and-software-defined-networking/)
- [Jupiter now scales to 13 Petabits per second - Google Cloud Blog](https://cloud.google.com/blog/products/networking/speed-scale-reliability-25-years-of-data-center-networking)
- [The evolution of Google's Jupiter data center network - Google Cloud Blog](https://cloud.google.com/blog/topics/systems/the-evolution-of-googles-jupiter-data-center-network)
- [NVIDIA Wins Every MLPerf Training v5.1 Benchmark - NVIDIA Blog](https://blogs.nvidia.com/blog/mlperf-training-benchmark-blackwell-ultra/)
- [Nvidia and Google Train to Win - XPU.pub](https://xpu.pub/2024/11/14/mlperf-training-4-1/)
- [Google TPU vs NVIDIA GPU - Introl Blog](https://introl.com/blog/google-tpu-vs-nvidia-gpu-infrastructure-decision-framework-2025)
- [Google TPU Architecture: 7 Generations Explained - Introl Blog](https://introl.com/blog/google-tpu-architecture-complete-guide-7-generations)
- [Google TPUs Explained: Architecture & Performance for Gemini 3 - IntuitionLabs](https://intuitionlabs.ai/articles/google-tpu-architecture-gemini-3)
- [Introducing Cloud TPU v5p and AI Hypercomputer - Google Cloud Blog](https://cloud.google.com/blog/products/ai-machine-learning/introducing-cloud-tpu-v5p-and-ai-hypercomputer)
- [Google AI Infrastructure Supremacy - SemiAnalysis](https://semianalysis.com/2023/04/12/google-ai-infrastructure-supremacy/)
- [Highly Customized Optical Networking Critical for Google's TPUs - NextBigFuture](https://www.nextbigfuture.com/2025/11/highly-customized-optical-networking-critical-for-googles-tensor-processing-units-tpus.html)
- [The Ironwood: Google TPU Rack & Optical Circuit Switch System - Global Tech Research](https://globaltechresearch.substack.com/p/the-ironwood-an-introduction-to-google)
- [vLLM TPU: A New Unified Backend - vLLM Blog](https://blog.vllm.ai/2025/10/16/vllm-tpu.html)
- [Building production AI on Cloud TPUs with JAX - Google Cloud Documentation](https://docs.cloud.google.com/tpu/docs/jax-ai-stack)
- [Tensor Processing Unit - Wikipedia](https://en.wikipedia.org/wiki/Tensor_Processing_Unit)

---

## 7. 電力・冷却インフラ（深掘り調査）

### 7.1 世代別消費電力

| 世代 | TDP / チップ | 備考 |
|------|------------|------|
| TPU v4 | 175〜250W（実運用平均 200W） | 液冷必須 |
| TPU v5p | 250〜300W | 液冷 |
| TPU v6e (Trillium) | 120〜300W（構成による） | |
| Ironwood (v7) | **約 157W** | 9,216チップ Superpod で約 10MW |
| NVIDIA B200 | **1,000〜1,200W** | フルスペック 1,200W、DGX 最適化版 1,000W |

- Ironwood は 157W/チップと極めて低消費電力。9,216チップの Superpod 全体で約 10MW
- NVIDIA B200 は 1,000〜1,200W と Ironwood の約 6〜8 倍の消費電力
- Ironwood の perf/watt は Trillium 比 2倍、FP8 性能（4,614 TFLOPS）は B200（4,500 TFLOPS）とほぼ同等ながら消費電力は大幅に低い

### 7.2 液冷システム

- TPU v4 以降、Google は**全世代で液冷（Direct-to-Chip Cold Plate）**を採用
- マニフォールド、フレキシブルホース、コールドプレートを高発熱チップに直接装着
- **冷却水温度**: Google 独自の温水冷却対応（標準的な施設水で運用可能）
- Hot Chips 2025 で Google が「データセンター規模の TPU 液冷」に関する発表を実施（Project Deschutes）
- Ironwood Superpod（9,216チップ、144ラック）は liquid-cooled 構成

### 7.3 PUE（Power Usage Effectiveness）

- Google データセンター全体の PUE: **1.09〜1.11**（2025年 TTM ベース）
- 四半期 PUE は 1.08〜1.11 の範囲で推移
- 業界平均の PUE 1.5〜1.6 と比較して極めて効率的

---

## 8. ソフトウェアスタック（深掘り調査）

### 8.1 JAX / Flax / Pax のスタック階層

```
┌─────────────────────────────────────────────┐
│  Application Layer                           │
│  (Gemini, PaLM, Gemma 等のモデル)             │
├─────────────────────────────────────────────┤
│  Framework Layer                             │
│  ├─ Pax: 大規模モデルトレーニング              │
│  ├─ Flax (NNX): モデルオーサリング             │
│  ├─ Optax: 最適化戦略                         │
│  └─ Grain: 決定論的データパイプライン           │
├─────────────────────────────────────────────┤
│  JAX Core                                    │
│  (配列計算、自動微分、JIT コンパイル、vmap)      │
├─────────────────────────────────────────────┤
│  XLA Compiler                                │
│  (HLO → 最適化 → ターゲットコード生成)          │
├─────────────────────────────────────────────┤
│  Pathways Runtime                            │
│  (分散ジョブスケジューリング、リソース管理)       │
├─────────────────────────────────────────────┤
│  TPU Hardware (ICI / OCS / Jupiter)          │
└─────────────────────────────────────────────┘
```

### 8.2 Pathways 分散ランタイム

- 2022年に Google Research が論文発表した大規模アクセラレータ向けオーケストレーション層
- **シャード化データフローグラフ**: 非同期オペレータの有向グラフで、future を生成・消費
- **Gang スケジューリング**: 異種計算リソースの効率的な一括割り当て
- **目的**: 新しいシステム研究・ML 研究の探索を可能にしつつ、現行モデルで最先端の性能を維持
- NVIDIA の Megatron-LM が手動で並列化戦略を指定するのに対し、Pathways は自動的なリソース管理を志向

### 8.3 XLA コンパイラの最適化パイプライン

1. **HLO (High-Level Optimizer)**: JAX/TensorFlow からの入力を HLO IR に変換
2. **最適化パス**: フュージョン、レイアウト割り当て、定数畳み込み、デッドコード削除等
3. **GSPMD による自動並列化**: シングルデバイスプログラムをパーティション化し、適切な Collective 通信を挿入
4. **ターゲットコード生成**: TPU 固有の命令セットへの変換
5. **オートチューニング**: 実行時パラメータの自動最適化

### 8.4 GSPMD（General and Scalable Parallelization for ML Computation Graphs）

- ML ワークロードの**自動並列化システム**
- プログラミング（モデル定義）と並列化の関心を分離
- 開発者はシングルデバイス向けにコードを書き、GSPMD が自動的にマルチデバイス向けに変換
- データ並列、モデル並列、テンソル並列、パイプライン並列を自動的に組み合わせ

### 8.5 NVIDIA Megatron-LM / NeMo との対比

| 項目 | Google (JAX + Pathways) | NVIDIA (Megatron-LM + NeMo) |
|------|------------------------|---------------------------|
| 並列化 | GSPMD で自動（アノテーションベース） | 手動で TP/PP/DP を指定 |
| フレームワーク | JAX（関数型） | PyTorch（命令型） |
| ランタイム | Pathways（自動リソース管理） | Slurm + torchrun（手動構成） |
| コンパイラ | XLA（HLO → TPU コード） | torch.compile / TensorRT |
| エコシステム | Google Cloud 専用 | マルチベンダ対応 |
| スケーリング | Pod/Superpod（数千〜万チップ） | DGX SuperPOD（数百〜千 GPU） |

---

## 9. MLPerf 結果の具体的数値（深掘り調査）

### 9.1 MLPerf Training v5.1（2025年）

| ベンチマーク | NVIDIA Blackwell | Google TPU | 備考 |
|------------|-----------------|-----------|------|
| Llama 3.1 405B | **10分**（5,120 GPU） | 提出なし | NVIDIA が全カテゴリ最速 |
| Llama 3.1 405B | 18.79分（2,560 GPU） | — | Hopper 比 45% 高速 |
| GPT-3 | — | 2分短縮（2,048 TPU、Trillium） | v4.1 比で改善 |

- NVIDIA Blackwell が MLPerf Training v5.1 の全ベンチマークで最速を記録
- Google は Trillium で一部カテゴリに提出しているが、Ironwood の結果はまだ

### 9.2 MLPerf Inference v5.1（2025年9月）

- 新規ベンチマーク: DeepSeek-R1、Llama 3.1 8B、Whisper Large V3
- LLM Q&A (Llama-2-70B): Conversational 2000ms/200ms、Interactive 450ms/40ms ターゲット
- Google は TPU での提出を行っているが、NVIDIA が多数のカテゴリでリード

### 9.3 Google TPU の MLPerf 参加状況

- Google は MLPerf への提出を一部カテゴリに限定する傾向
- Trillium (v6e) は Training v5.0 で「期待値に対して 8% にとどまった」との報告
- Ironwood (v7) の MLPerf 結果は今後の提出に期待

---

## 10. 料金・TCO 分析（深掘り調査）

### 10.1 GCP TPU 料金体系

| モデル | オンデマンド ($/chip/h) | 1年コミット | 3年コミット |
|--------|---------------------|-----------|-----------|
| TPU v5e | $1.20 | $0.84 | $0.54 |
| TPU v5p | $4.20 | $2.94 | $1.89 |
| TPU v6e (Trillium) | $1.375 | — | $0.39〜 |

### 10.2 NVIDIA GPU インスタンスとの比較

- GCP A3 (H100 80GB): 約 $3.50〜/GPU/h（オンデマンド）
- GCP A3 Ultra (H200 141GB): さらに高額
- **TPU v5e は H100 比で大幅に安価**（$1.20 vs $3.50+）

### 10.3 「GPU 比 4倍のコスト効率」の根拠

- **TPU v5e は推論で H100 比 4倍の performance-per-dollar** を達成（Google 主張）
- AssemblyAI: Cloud TPU v5e が「一貫して 4倍の performance per dollar」を報告
- Trillium: TPU v5e 比 2.1倍、TPU v5p 比 2.5倍の performance-per-dollar（LLM トレーニング）
- **Midjourney の事例**: 月間推論コストが $210万 → $70万未満に削減（**65% 削減、年間 $1,680万のコスト削減**）

---

## 11. 外部ユーザーの実デプロイメント事例（深掘り調査）

### 11.1 Anthropic

- 2025年10月: Google Cloud TPU の利用を大幅拡大し、**最大100万チップへのアクセス**を発表
- Claude モデルの次世代トレーニング・サービングに TPU を使用
- Google Cloud を選択した理由: スケーラビリティ、長期的パートナーシップ

### 11.2 Apple

- Apple Intelligence のモデル（Apple Foundation Model / AFM）は **Google Cloud TPU クラスタでトレーニング**
- NVIDIA GPU ではなく TPU を選択
- **規模**: 2,048 TPU v5p チップを使用した事例が確認されている

### 11.3 Midjourney

- 2025年 Q2: 画像生成の推論ワークロードの大部分を GPU → TPU に移行
- Stable Diffusion XL と Flux の推論を TPU で実行
- **コスト削減**: 月間 $210万 → $70万未満（65% 削減）
- 同等の出力量を維持しながらのコスト削減

### 11.4 リージョン展開

- **北米**: US Central / East / South / West（全 TPU 世代）
- **ヨーロッパ**: West リージョン（TPU v5e、v6e）
- **アジア太平洋**: Southeast リージョン（TPU v5e、v6e）

---

## 12. 3D Torus vs Fat-tree 通信性能分析（深掘り調査）

### 12.1 バイセクション帯域幅

| トポロジ | バイセクション帯域幅 | 特徴 |
|---------|-------------------|------|
| Fat-tree | **フルバイセクション帯域幅** | 任意の2分割間で等量帯域 |
| 3D Torus | **低バイセクション帯域幅** | 次元あたりの帯域に制約 |

- Fat-tree はクラスタの任意の半分同士の間で、総帯域の半分を保証
- Torus は次元ごとのリンク帯域に制約されるため、バイセクション帯域が低い

### 12.2 Collective 通信の特性

- **AllReduce**: Fat-tree は Ring AllReduce / Recursive Halving-Doubling で高い帯域利用率。Torus は Dimension-Order Routing により各次元で順次実行
- **AllGather**: Fat-tree が帯域利用率で有利
- **ただし**: Google は ICI の圧倒的な帯域幅（9.6 Tbps/chip）とカスタム通信パターンにより、Torus の理論的不利を補償

### 12.3 耐障害性

| 項目 | 3D Torus | Fat-tree |
|------|---------|---------|
| 冗長パス | 複数の異なるパスが存在（8×8×8 で 6ホップパスが90通り） | スイッチレベルの冗長性 |
| リンク障害の影響 | 局所的（隣接ノード間のみ） | スイッチ障害で広範囲に影響 |
| ルーティング | 適応型ルーティングで回避可能 | スイッチ冗長化で対応 |
| 利点 | **分散型で障害の局所化** | **帯域幅の均一性** |

- 3D Torus は障害が局所化される利点があり、負荷分散と障害回避に複数の代替パスを活用できる
- Google は OCS（光回路スイッチ）により動的なトポロジ再構成でさらに耐障害性を強化

### 深掘り調査のソース

- [Google TPU v4 Liquid Cooling - ISCA 2023](https://dl.acm.org/doi/10.1145/3579371.3589350)
- [Google Hot Chips 2025 - Datacenter Cooling for TPUs](https://www.hc2025.hotchips.org/)
- [MLPerf Training v5.1 Results - NVIDIA](https://nvidianews.nvidia.com/news/nvidia-blackwell-sets-new-ai-training-records)
- [Google Cloud TPU Pricing](https://cloud.google.com/tpu/pricing)
- [Anthropic Expands Use of Google Cloud TPUs](https://cloud.google.com/blog/products/ai-machine-learning/anthropic-expands-use-of-google-cloud-tpus)
- [Apple Foundation Model Training on TPUs](https://machinelearning.apple.com/research/apple-foundation-models)
- [Midjourney TPU Migration - Cost Savings](https://cloud.google.com/blog/products/ai-machine-learning/how-midjourney-reduced-costs-with-tpus)
- [Ironwood Power Consumption - 157W per chip](https://www.nextplatform.com/2025/04/10/google-ironwood-tpu-challenges-nvidias-ai-training-empire/)
- [NVIDIA B200 TDP Specifications](https://www.nvidia.com/en-us/data-center/b200/)
- [Google Data Center PUE - 2025](https://www.google.com/about/datacenters/efficiency/)
