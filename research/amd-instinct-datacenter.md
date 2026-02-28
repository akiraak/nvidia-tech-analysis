# AMD Instinct によるデータセンター構成

調査日: 2026-02-28

---

## 1. AMD Instinct の最新世代と主要スペック

### 製品ラインナップ概要

AMD Instinct は AMD のデータセンター向け GPU アクセラレータブランドであり、CDNA アーキテクチャを基盤としている。2024〜2025年にかけて MI300 シリーズから MI350 シリーズへと世代交代が進んでいる。

### 各モデルの主要スペック

#### MI300X（CDNA 3、2023年12月出荷開始）

- **アーキテクチャ**: CDNA 3、TSMC 5nm/6nm
- **HBM メモリ**: 192 GB HBM3（8スタック）
- **メモリ帯域幅**: 5.325 TB/s
- **演算性能**:
  - FP32: 163.4 TFLOPS
  - FP16 / BF16: 1,307.4 TFLOPS
  - FP8: 2,614.9 TFLOPS
  - Sparsity 有効時 FP8: 5,229.8 TFLOPS
- **TDP**: 750W
- **GPU コンピュートユニット (CU)**: 304 CU（8 XCD、各38 CU）

#### MI300A（CDNA 3 APU、2023年）

- **特徴**: CPU（Zen 4）と GPU（CDNA 3）を統合した APU
- **CPU**: 24コア Zen 4（3 チップレット x 8コア）
- **GPU**: 228 CU（6 XCD、各38 CU）
- **HBM メモリ**: 128 GB HBM3（統合メモリ、CPU/GPU 共有）
- **メモリ帯域幅**: 5.325 TB/s
- **共有 LLC**: 256 MB
- **TDP**: 550W（空冷/液冷）/ 760W（液冷）
- **特徴**: CPU と GPU が完全にコヒーレントなメモリ空間を共有。明示的なメモリコピーが不要

#### MI325X（CDNA 3 改良版、2024年10月出荷開始）

- **アーキテクチャ**: CDNA 3（MI300X の HBM 強化版）
- **HBM メモリ**: 256 GB HBM3E
- **メモリ帯域幅**: 6.0 TB/s
- **演算性能**: MI300X と同等のコンピュート性能
- **TDP**: 750W
- **位置づけ**: NVIDIA H200 の競合製品

#### MI350X（CDNA 4、2025年6月発表）

- **アーキテクチャ**: CDNA 4、TSMC 3nm
- **HBM メモリ**: 288 GB HBM3E
- **メモリ帯域幅**: 8.0 TB/s
- **演算性能**:
  - FP64: 78.6 TFLOPS
  - FP16: 約 5.0 PFLOPS
  - FP8: 約 10.1 PFLOPS（Sparsity 含む推定）
- **TDP**: 1,000W
- **GPU 構成**: 8 XCD、各32 CU
- **新機能**: MXFP6 / MXFP4 データタイプのネイティブサポート、ハードウェアスパーシティ対応
- **世代間性能向上**: AI 演算で前世代比最大 4倍、推論で最大 35倍の向上

#### MI355X（CDNA 4、2025年）

- **MI350X の上位モデル**
- **FP64**: 78.6 TFLOPS
- **FP16**: 約 5.0 PFLOPS
- **FP8**: 約 10.1 PFLOPS
- **位置づけ**: NVIDIA B200 の競合製品

### NVIDIA Blackwell / Hopper との比較

| 項目 | AMD MI300X | AMD MI325X | AMD MI350X | NVIDIA H100 SXM | NVIDIA H200 SXM | NVIDIA B200 |
|------|-----------|-----------|-----------|-----------------|-----------------|-------------|
| HBM 容量 | 192 GB (HBM3) | 256 GB (HBM3E) | 288 GB (HBM3E) | 80 GB (HBM3) | 141 GB (HBM3E) | 192 GB (HBM3E) |
| メモリ帯域幅 | 5.3 TB/s | 6.0 TB/s | 8.0 TB/s | 3.35 TB/s | 4.8 TB/s | 8.0 TB/s |
| FP16 Tensor | 1,307 TFLOPS | 1,307 TFLOPS | ~5,000 TFLOPS | 989 TFLOPS | 989 TFLOPS | 2,250 TFLOPS |
| FP8 Tensor | 2,615 TFLOPS | 2,615 TFLOPS | ~10,100 TFLOPS | 1,979 TFLOPS | 1,979 TFLOPS | 4,500 TFLOPS |
| TDP | 750W | 750W | 1,000W | 700W | 700W | 1,000W |

**注意点**: AMD MI300X はカタログスペック（理論性能）では NVIDIA H100/H200 を上回るが、実際の LLM 推論ベンチマークでは H100/H200 の 37〜66% の実効性能にとどまるケースが報告されている。これはソフトウェアスタック（ROCm vs CUDA）の最適化の差や、ライブラリ（cuDNN 等）の成熟度の違いに起因する。

ソース:
- https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html
- https://www.amd.com/en/products/accelerators/instinct/mi300/mi325x.html
- https://www.amd.com/en/products/accelerators/instinct/mi350.html
- https://www.clarifai.com/blog/mi300x-vs-b200
- https://arxiv.org/pdf/2510.27583

---

## 2. AMD のインターコネクト技術

### Infinity Fabric の概要

AMD Infinity Fabric は、AMD のモジュラー型パケットベースのインターコネクト技術であり、CPU、GPU、メモリコントローラ間を接続する。元々は AMD の CPU チップレットアーキテクチャ（EPYC）のダイ間接続として開発され、GPU 間接続にも拡張された。

### xGMI（External Global Memory Interconnect）

xGMI は Infinity Fabric 上に構築された GPU 間高速インターコネクトプロトコルである。

#### xGMI の世代と帯域幅

| 世代 | 搭載製品 | ビット幅 | 転送レート | 片方向帯域幅（リンクあたり） |
|------|---------|---------|-----------|--------------------------|
| xGMI 2 | MI250X | 16 bit | 25 GT/s | 50 GB/s |
| xGMI 3 | MI300A/MI300X | 16 bit | 32 GT/s | 64 GB/s |

#### MI300X の GPU 間接続トポロジ

- 8 GPU が **フルメッシュ（全結合）トポロジ** で接続
- 各 GPU は他の 7 つの GPU それぞれに対して専用リンクを保持
- **リンクあたりの帯域幅**: 128 GB/s（双方向）/ 約 50 GB/s（片方向、実効値）
- **システム全体の集約帯域幅**: 896 GB/s
- フルメッシュにより、マルチホップ通信が不要で、任意の GPU ペア間で直接通信が可能
- 理論帯域幅の約 81% の実効帯域幅を達成

#### Infinity Fabric のリンクタイプ

| リンクタイプ | 用途 | 双方向帯域幅 |
|------------|------|------------|
| Quad リンク | GPU 内 GCD 間（イントラ GPU） | 200+200 GB/s |
| Dual リンク | GPU 間（インター GPU） | 100+100 GB/s |
| Single リンク | GPU 間（インター GPU） | 50+50 GB/s |

### NVLink との比較

| 項目 | AMD Infinity Fabric (MI300X) | NVIDIA NVLink (H100 SXM) | NVIDIA NVLink (B200) |
|------|---------------------------|-------------------------|---------------------|
| GPU 間帯域幅（双方向） | 128 GB/s/リンク | 900 GB/s（全18リンク合計） | 1,800 GB/s |
| トポロジ | フルメッシュ（8 GPU） | NVSwitch 経由全結合（8 GPU） | NVSwitch 経由（72 GPU） |
| 接続可能 GPU 数 | 8 GPU（ノード内） | 8 GPU（ノード内） | 72 GPU（NVLink Switch） |
| プロトコル | xGMI / Infinity Fabric | NVLink | NVLink 5th Gen |
| In-Network Reduction | 非対応 | NVLS（NVLink SHARP）対応 | NVLS 対応 |

**重要な差異**:
- NVIDIA NVLink は NVSwitch を使用した **スイッチ型トポロジ** で、H100 では 8 GPU 全体で 900 GB/s の全二重帯域幅を提供。AMD のフルメッシュは点対点接続で、各リンクは 128 GB/s
- NVLink は In-Network Reduction（NVLS）をサポートし、Collective 通信で大きなアドバンテージを持つ
- NVLink 5th Gen（Blackwell）は最大 72 GPU をスイッチ接続でき、スケーラビリティで大きく優位
- AMD Infinity Fabric は CPU-GPU 間のコヒーレント接続が可能（MI300A の統合メモリ）

ソース:
- https://rocm.blogs.amd.com/software-tools-optimization/mi300x-rccl-xgmi/README.html
- https://arxiv.org/html/2410.00801v1
- https://en.wikichip.org/wiki/amd/infinity_fabric
- https://massedcompute.com/faq-answers/?question=How+does+NVLink+compare+to+AMD's+Infinity+Fabric+in+terms+of+performance?

---

## 3. AMD データセンターのネットワーク構成

### ノード内通信

- **GPU 間**: Infinity Fabric（xGMI）によるフルメッシュ接続（8 GPU）
- **CPU-GPU 間**: PCIe Gen 5 / Infinity Fabric
- 上記のノード内通信はネットワーク機器を必要としない

### ノード間通信（クラスタネットワーク）

AMD Instinct クラスタのノード間通信では、以下のプロトコルが利用可能:

1. **RoCE v2（RDMA over Converged Ethernet）**: Ethernet ベースの RDMA。InfiniBand に匹敵する速度を低コストで実現
2. **InfiniBand**: 低遅延・高帯域の伝統的な HPC ネットワーク
3. **TCP/IP**: 汎用 Ethernet（RDMA 非対応、低性能）

### 検証済み NIC（ネットワークインターフェースカード）

AMD の公式ドキュメントで検証済みとされている NIC:

| ベンダー | 製品 | 速度 | プロトコル |
|---------|------|------|----------|
| **Broadcom** | BCM957608（Thor2） | 400 Gbps | RoCE v2、Peer Memory Direct |
| **NVIDIA (Mellanox)** | ConnectX-7 | 400 Gbps | RoCE v2、InfiniBand NDR |
| **AMD** | Pensando Pollara 400 | 400 Gbps | RoCE v2（UEC 対応） |

#### AMD Pensando Pollara 400 AI NIC

AMD は自社の NIC ソリューションとして **Pensando Pollara 400** を提供:

- **速度**: 400 Gbps Ethernet
- **特徴**: 業界初の **Ultra Ethernet Consortium（UEC）対応** AI NIC
- **プログラマビリティ**: 第3世代 Pensando P4 エンジンによる完全ハードウェアプログラマブル
- **フォームファクタ**: OCP 3.0 準拠
- **用途**: GPU-GPU 間の高速通信、ネットワーク監視、ロードバランシング

#### AMD Pensando Salina 400 DPU

- **速度**: デュアル 400GE、PCIe Gen 5
- **プロセッサ**: 232 P4 MPU エンジン + 16x Arm Neoverse-N1 コア
- **メモリ**: 最大 128 GB DDR5
- **機能**: SDN、ファイアウォール、暗号化、ロードバランシング、ストレージオフロード

### NVIDIA ネットワーク機器との互換性

#### NVIDIA ConnectX-7 との組み合わせ

- AMD Instinct クラスタで **NVIDIA ConnectX-7 が検証済み NIC として公式にリストされている**
- ConnectX-7 は RoCE v2 および InfiniBand NDR（400 Gbps）の両方をサポート
- AMD GPU クラスタで InfiniBand ネットワークを構築する場合、ConnectX-7 が主要な選択肢の一つ

#### NVIDIA Spectrum-X / InfiniBand スイッチとの組み合わせ

- **InfiniBand**: NVIDIA の InfiniBand スイッチ（Quantum-2 NDR 等）と ConnectX-7 の組み合わせは、AMD GPU クラスタでも利用可能。InfiniBand プロトコル自体はベンダー非依存
- **Spectrum-X**: NVIDIA Spectrum-X は Ethernet ベースの AI ネットワークソリューションだが、NVIDIA GPU（特に Spectrum-4 スイッチ + ConnectX-7 + NVIDIA AIR の組み合わせ）に最適化されている。AMD GPU との組み合わせでは、Spectrum-X の特定の最適化機能（NVIDIA AIR によるパケットスプレイ等）が利用できない可能性がある
- **実態**: 多くの AMD GPU クラスタでは、Broadcom や AMD Pensando の Ethernet スイッチ + NIC の組み合わせ、または NVIDIA ConnectX-7 + InfiniBand の構成が採用されている

### RoCE / InfiniBand の設定要件

RoCE クラスタの構築には以下の設定が重要:

1. **NIC 側の設定**:
   - PCIe Relaxed Ordering の有効化
   - RDMA サポートの有効化
   - RoCE パフォーマンスプロファイルの設定

2. **スイッチ側の設定**:
   - RoCE/CNP の DSCP 値を NIC と一致させる
   - ECN（Explicit Congestion Notification）の有効化と閾値設定
   - PFC（Priority Flow Control）の設定

3. **NIC-GPU の NUMA マッピング**:
   - NIC と GPU の NUMA ノードを適切にマッピングすることで最適なパフォーマンスを実現

ソース:
- https://instinct.docs.amd.com/projects/gpu-cluster-networking/en/latest/reference/hardware-support.html
- https://instinct.docs.amd.com/projects/gpu-cluster-networking/en/latest/how-to/roce-network-config.html
- https://docs.broadcom.com/doc/957608-AN2XX
- https://www.amd.com/en/products/network-interface-cards/pensando.html
- https://www.storagereview.com/news/amds-pensando-pollara-400-nic-brings-programmability-and-performance-to-ai-networking

---

## 4. ソフトウェアエコシステム

### ROCm プラットフォームの現状

ROCm（Radeon Open Compute）は AMD の GPU コンピューティングプラットフォームであり、NVIDIA CUDA に対応する位置づけにある。

#### ROCm バージョン推移

| バージョン | リリース時期 | 主な特徴 |
|-----------|------------|---------|
| ROCm 6.x | 2024年 | MI300X 対応の安定版 |
| ROCm 7.0 | 2025年初頭 | AI/HPC インフラの大幅強化、vLLM/SGLang ネイティブ統合 |
| ROCm 7.1.1 | 2025年中頃 | PyTorch 2.9 対応 |
| ROCm 7.2.0 | 2025年後半 | JAX 0.8.0 対応 |

#### ROCm 7.0 の主な特徴

- vLLM および SGLang のネイティブ統合
- RCCL によるマルチ GPU スケーリングの強化
- MIOpen ライブラリの最適化
- オープンソースを基本方針とし、ベンダーロックインを回避

### CUDA との互換性（HIP による移植）

**HIP（Heterogeneous-compute Interface for Portability）** は AMD が提供する CUDA 互換レイヤーであり、CUDA コードを AMD GPU 上で動作させるための移植ツール。

- **hipify**: CUDA コードを HIP コードに自動変換するツール
- **互換性**: 大部分の CUDA API を HIP API にマッピング可能
- **制約**: 一部の CUDA 固有機能（Tensor Core 専用命令等）は直接変換できない場合がある
- **ランタイム**: HIP コードは NVIDIA GPU 上でも AMD GPU 上でも実行可能（ポータブル）

### フレームワーク対応状況

| フレームワーク | ROCm 対応状況 | 対応バージョン（ROCm 7.x） |
|--------------|-------------|------------------------|
| **PyTorch** | 公式対応 | PyTorch 2.7, 2.8, 2.9 |
| **TensorFlow** | 公式対応 | TensorFlow 2.19.1 |
| **JAX** | 公式対応 | JAX 0.6.x (ROCm 7.0), JAX 0.8.0 (ROCm 7.2) |
| **vLLM** | ネイティブ統合 | ROCm 7.0 以降 |
| **SGLang** | ネイティブ統合 | ROCm 7.0 以降 |

#### 性能差について

- PyTorch on ROCm は CUDA 版とほぼ同等の性能を達成しつつあるが、Attention メカニズムなど特定の演算では CUDA の cuDNN 統合が有利
- ML フレームワーク全般で、NVIDIA ハードウェアの方が **10〜20% 高速** な傾向（2025〜2026年時点）。ただしこのギャップは急速に縮小中

### RCCL（ROCm Communication Collectives Library）

RCCL は NCCL（NVIDIA Collective Communications Library）の AMD 版であり、マルチ GPU の集団通信プリミティブを提供する。

#### 対応する集団通信操作

- AllReduce
- AllGather
- ReduceScatter
- Broadcast
- Reduce

#### RCCL の性能と課題

- **API 互換性**: NCCL と概ね互換性のある API を提供
- **性能課題**: NVIDIA の NCCL と比較して、特にスケールアウト（マルチノード）性能で差がある
  - NVIDIA は NCCL + InfiniBand/Spectrum-X + NVSwitch の垂直統合で最適化
  - AMD はネットワーク垂直統合の度合いが低い
- **新興ライブラリとの比較**:
  - MSCCL++: RCCL に対して小メッセージで最大 3.8 倍、大メッセージで最大 2.2 倍の高速化
  - UCCL: NCCL/RCCL のドロップイン置き換えとして、レイテンシ・スループットの両面で優位
  - HiCCL: AMD、NVIDIA、Intel GPU で RCCL/NCCL と同等以上の性能

ソース:
- https://rocm.docs.amd.com/en/latest/about/release-notes.html
- https://research.aimultiple.com/cuda-vs-rocm/
- https://rocm.blogs.amd.com/ecosystems-and-partners/rocm-7.0-blog/README.html
- https://www.amd.com/content/dam/amd/en/documents/partner-hub/instinct/why-choose-rocm-platform.pdf
- https://rocm.docs.amd.com/en/latest/compatibility/ml-compatibility/pytorch-compatibility.html

---

## 5. 性能実績と採用事例

### MLPerf ベンチマーク結果

#### MLPerf Training v5.0（2025年）

- AMD Instinct MI300X による初の MLPerf Training 公式提出
- **Llama 2-70B LoRA (FP8) ベンチマーク**:
  - MI300X 8-GPU プラットフォーム（Dell）: 28.99 分で学習完了
  - MI300X 8-GPU プラットフォーム（Oracle）: 30.42 分で学習完了
- MI325X プラットフォームは Llama 2-70B-LoRA ファインチューニングで NVIDIA H200 を最大 **8% 上回る** 性能

#### MLPerf Training v5.1（2025年11月）

- **MI355X / MI350X** による初の Training 提出
- MI355X は MI300X と比較して学習時間を約 28 分から約 10 分に短縮（Llama 2-70B LoRA）

#### MLPerf Inference v5.1（2025年9月）

- MI300X の推論性能を ROCm ソフトウェアスタックの能力とともに実証
- MangoBoost が MI300X 4 ノード構成で Llama 2 70B ベンチマークのオフライン性能最高記録を達成
- パートナー提出（Dell、MangoBoost）が AMD リファレンス性能の 1% 以内を達成し、デプロイメント品質の一貫性を実証

### 主要採用事例

#### Microsoft Azure

- **Azure ND MI300X v5 VM** として MI300X を大規模デプロイ
- **Azure OpenAI Service** のワークロードに MI300X を活用
- Microsoft EVP Scott Guthrie が MI300X を「最もコスト効率の良い GPU」と評価
- Canada Central リージョンなどで利用可能

#### Meta

- **Meta のすべてのライブトラフィック** に MI300X を独占的にデプロイ（特定のワークロード）
- 大容量メモリと TCO（Total Cost of Ownership）の優位性が採用理由

#### Oracle Cloud Infrastructure (OCI)

- **OCI Compute Supercluster** に AMD Instinct アクセラレータを採用
- 生成 AI、コンピュータビジョン、予測分析などの高負荷 AI ワークロードを処理
- MI355X on OCI のパフォーマンス・技術詳細も公開済み

### El Capitan スーパーコンピュータ

- **場所**: Lawrence Livermore National Laboratory（LLNL）、カリフォルニア州
- **正式稼働**: 2025年2月
- **用途**: 国家安全保障向けの世界初のエクサスケールシステム
- **TOP500 ランキング**: **第1位**（2025年11月時点も維持）
- **HPL ベンチマーク**: **1.809 Exaflop/s**
- **ピーク性能**: 2.79 Exaflop/s
- **HPCG ベンチマーク**: 17.4 PFLOPS（デビュー時最高スコア）
- **ハードウェア構成**:
  - 43,808 x AMD EPYC "Genoa" 24コア CPU（1,051,392 CPU コア）
  - 43,808 x AMD Instinct MI300A APU（9,988,224 GPU コンピュートユニット）
  - 合計 11,039,616 CPU+GPU コア
- **システム**: HPE Cray EX255a

### その他の HPC 実績

- AMD は TOP500 リストにおいてスーパーコンピュータリーダーシップを維持
- El Capitan が MI300A（APU）の統合メモリアーキテクチャの優位性を実証

### 市場シェアと将来展望

- AMD はデータセンター AI 収益で今後 3〜5 年間で **年平均成長率 80% 以上** を見込む
- 2026年末までに AI GPU 市場で **15〜20% のシェア** を獲得する見通し（アナリスト予測）
- 将来のロードマップ: MI450 シリーズ、Helios ラックスケールソリューション

ソース:
- https://www.amd.com/en/blogs/2025/amd-drives-ai-gains-with-mlperf-training-results.html
- https://www.amd.com/en/blogs/2025/amd-instinct-gpus-continue-ai-momentum-across-indu.html
- https://www.amd.com/en/newsroom/press-releases/2024-5-21-amd-instinct-mi300x-accelerators-power-microsoft-a.html
- https://blogs.oracle.com/cloud-infrastructure/post/llm-performance-results-amd-instinct-mi300x-gpus
- https://www.amd.com/en/blogs/2025/el-capitan-takes-exascale-computing-to-new-heights.html
- https://en.wikipedia.org/wiki/El_Capitan_(supercomputer)
- https://top500.org/news/el-capitan-retains-top-spot-65th-top500-list-exascale-era-expands/

---

## 6. 総合評価: AMD Instinct vs NVIDIA プラットフォーム

### AMD の強み

1. **HBM メモリ容量**: MI300X（192 GB）、MI325X（256 GB）は同世代の NVIDIA 製品を大幅に上回るメモリ容量を持ち、大規模モデルの推論で有利
2. **コストパフォーマンス**: TCO ベースで NVIDIA より優れるケースがある（Meta、Microsoft の評価）
3. **APU アーキテクチャ**: MI300A の統合メモリは HPC ワークロードで独自の優位性（El Capitan で実証）
4. **オープンソース志向**: ROCm のオープンソースアプローチはベンダーロックインを回避
5. **Ultra Ethernet**: Pensando Pollara 400 による UEC 対応は次世代 Ethernet 標準への先行投資

### AMD の課題

1. **ソフトウェアエコシステムの成熟度**: ROCm は CUDA に対して依然として最適化の深さで劣る（実効性能で 10〜20% の差）
2. **ネットワーク垂直統合**: NVIDIA の NVLink + NVSwitch + InfiniBand/Spectrum-X の垂直統合に対して、AMD はネットワーキングの統合度が低い
3. **スケールアウト性能**: マルチノードのスケールアウトでは NCCL + NVLink の最適化が RCCL + xGMI を上回る
4. **GPU 間帯域幅**: NVLink（900 GB/s @H100）対 Infinity Fabric（896 GB/s @MI300X 集約値）は近似だが、NVSwitch による効率的なルーティングで NVIDIA が有利
5. **開発者エコシステム**: CUDA の膨大な開発者コミュニティ・ライブラリに対して ROCm はまだ追いつく途上

### 結論

AMD Instinct は、特に **メモリ容量が重要な推論ワークロード** や **コスト効率を重視するデプロイメント** において競争力のある選択肢となっている。El Capitan の成功は HPC 領域での実力を示し、Microsoft・Meta・Oracle の採用はエンタープライズ市場での認知を証明している。しかし、大規模学習クラスタにおけるスケールアウト性能や、ソフトウェアスタックの成熟度では NVIDIA が依然としてリードしており、2026年末までの ROCm エコシステムの成熟とネットワーク統合の強化が AMD の市場シェア拡大の鍵となる。

---

## 7. 物理ラック構成・リファレンスプラットフォーム（深掘り調査）

### 7.1 OAM / UBB プラットフォーム

- **OAM (OCP Accelerator Module)**: OCP 標準のアクセラレータモジュール規格
- **UBB 2.0 (Universal Baseboard)**: 8 基の OAM アクセラレータを搭載するベースボード
  - MI300X: 8 OAM × 192GB HBM3 = 合計 1.5TB HBM
  - MI325X: 8 OAM × 256GB HBM3E = 合計 2TB HBM
- xGMI による GPU 間フルメッシュ接続はベースボード上のパッシブ配線で実現

### 7.2 主要サーバプラットフォーム

#### Supermicro 8U GPU System

| 項目 | スペック |
|------|---------|
| フォームファクタ | 8U |
| GPU | 8 × MI300X OAM |
| CPU | AMD EPYC（ホスト CPU） |
| NIC | **8 × 400GbE NIC**（GPU と 1:1 マッピング） |
| NIC-GPU 接続 | GPU と NIC が同一 PCIe root complex に配置（NUMA アライメント） |
| ネットワーク構成 | GPU ごとに専用 NIC による Rail-optimized トポロジ対応 |

#### Dell PowerEdge XE9680

| 項目 | スペック |
|------|---------|
| GPU | 8 × MI300X 192GB 750W OAM |
| CPU | デュアルソケット AMD EPYC |
| 用途 | ML/DL トレーニング・推論 |

#### HPE 系

- **注意**: HPE ProLiant DL384 Gen12 は NVIDIA GH200 向けであり、MI300X には対応していない
- AMD Instinct 向けは HPE Cray EX 等の HPC ラインが中心（El Capitan 等）

### 7.3 GPU-NIC-CPU 物理トポロジ

```
[MI300X GPU 0] ←PCIe→ [ConnectX-7 NIC 0] → Leaf Switch Rail 0
[MI300X GPU 1] ←PCIe→ [ConnectX-7 NIC 1] → Leaf Switch Rail 1
  ...                    ...
[MI300X GPU 7] ←PCIe→ [ConnectX-7 NIC 7] → Leaf Switch Rail 7
      ↕ xGMI (Full Mesh)          ↕ Ethernet/IB
```

- 各 GPU に対して 1:1 で NIC を割り当て
- GPU と NIC を同一 NUMA ノード / PCIe root complex に配置することで、GPU→NIC 間のレイテンシを最小化
- AMD 公式ドキュメントで NUMA トポロジマッピングの最適化手順が公開されている

---

## 8. 大規模クラスタのネットワークトポロジ設計（深掘り調査）

### 8.1 AMD Instinct Reference Cluster Design

AMD は公式リファレンスクラスタ設計（Version 4.0、2025年8月）を公開しており、以下の規模をカバー:

| クラスタ規模 | ノード数 | ネットワーク階層 | トポロジ |
|------------|---------|---------------|---------|
| 128〜1,024 GPU | 16〜128 | 2 Tier | Rail-optimized / Fat-tree |
| 1,024〜8,192 GPU | 128〜1,024 | 2 Tier | Rail-optimized / Fat-tree |

### 8.2 Rail-Optimized トポロジ

- GPU の NIC ランク（Rail）ごとに専用の Leaf スイッチに集約
- 例: 全ノードの GPU 0 の NIC → Rail 0 Leaf Switch、GPU 1 → Rail 1 Leaf Switch、...
- **利点**: Collective 通信（AllReduce 等）のトラフィックパターンに最適化され、ネットワーク輻輳を低減
- Fat-tree と比較して、Rail 内の通信でスイッチホップ数を削減

### 8.3 推奨 NIC / スイッチ構成

| 規模 | NIC | スイッチ |
|------|-----|---------|
| 小〜中規模 | Broadcom Thor-2 400G / ConnectX-7 400G | Broadcom / Cisco 等 |
| 大規模（1,024+ GPU） | **AMD Pollara 400G** | 各社 400G スイッチ |
| 次世代 | AMD Vulcano 800GbE（2026年下期予定） | 800G 対応スイッチ |

- AMD Pollara 400G: 業界初の UEC（Ultra Ethernet Consortium）対応 AI NIC
- 次世代 Vulcano 800GbE は NVIDIA ConnectX-8 より約1年遅れの見込み

### 8.4 オーバーサブスクリプション比

- AI/ML クラスタでは **1:1（ノンブロッキング）** が推奨
- コスト制約がある場合、**3:1** まで許容（コレクティブ効率 90% 維持の目安）
- Leaf-Spine 構成: 32ポート Leaf + 64ポート Spine の2層設計が一般的

---

## 9. RCCL vs NCCL の定量的性能比較（深掘り調査）

### 9.1 AllReduce 帯域幅比較

Semi Analysis のベンチマークレポートによる実測データ:

| 構成 | メッセージサイズ (16MiB〜256MiB) | 相対性能 |
|------|-------------------------------|---------|
| NVIDIA H100 + InfiniBand (Non-SHARP) | 基準 (100%) | — |
| NVIDIA H100 + InfiniBand + **SHARP** | **約 130〜150%** | SHARP による In-Network Reduction |
| NVIDIA H100 + Spectrum-X (Ethernet) | **約 90〜95%** | IB Non-SHARP に近い性能 |
| AMD MI300X + ConnectX-7 + **RoCEv2** | **約 50%** | IB Non-SHARP の約半分 |

- MI300X + RoCEv2 は実用的なメッセージサイズ（16MiB〜256MiB）で InfiniBand Non-SHARP 比約半分の速度
- NVIDIA Spectrum-X が IB Non-SHARP に近い性能を達成できるのは NCCL との垂直統合が要因
- RCCL の NVIDIA ネットワーク技術との統合度の差が性能ギャップの主因

### 9.2 MSCCL++ / UCCL の性能

| ライブラリ | 小メッセージ（NCCL 比） | 大メッセージ（NCCL 比） |
|-----------|---------------------|---------------------|
| MSCCL++ vs NCCL | **最大 2.8倍** | **最大 2.4倍** |
| MSCCL++ vs MSCCL | 最大 1.6倍 | 最大 2.0倍 |
| MSCCL++ vs RCCL | **最大 3.8倍** | — |

- MSCCL++（Microsoft）は NCCL/RCCL を大幅に上回る性能を示す
- 特に RCCL に対しては小メッセージで最大 3.8倍の高速化
- UCCL は NCCL/RCCL のドロップイン代替として開発中（UC Berkeley）

---

## 10. 冷却・電力インフラ（深掘り調査）

### 10.1 世代別冷却・電力仕様

| モデル | TDP | 冷却方式 | ラック電力密度 |
|--------|-----|---------|-------------|
| MI300X | 750W | 空冷 / 液冷 | — |
| MI325X | 750W | 空冷 / 液冷 | — |
| MI350X | **1,000W** | 空冷 / 液冷 | — |
| MI355X | **最大 1,400W** | **液冷（DLC）必須** | 高密度 |

- MI355X は Direct Liquid Cooling（DLC）環境でのみ 1,400W 動作が可能、空冷版の MI350X は 1,000W
- **冷却水温度**: ASHRAE W3/W4 温水冷却に対応（DCX HYDRO コールドプレートで 330W 以上の熱を移送）

### 10.2 電力効率比較

| チップ | TDP | FP8 性能 | GFLOPS/W (FP8) |
|--------|-----|---------|---------------|
| MI300X | 750W | 2,615 TFLOPS | **3.49** |
| NVIDIA H100 | 700W | 3,958 TFLOPS | **5.65** |
| NVIDIA H200 | 700W | 3,958 TFLOPS | **5.65** |
| NVIDIA B200 | 1,000W | 9,000 TFLOPS | **9.00** |
| MI350X | 1,000W | ~10,100 TFLOPS | **~10.10** |

- MI300X のカタログスペック上の GFLOPS/W は H100 の約 62%
- MI350X は B200 と同等以上の電力効率を達成する見込み
- **実効性能ベースでは差がさらに大きい**: MI300X は実ワークロードで H100 の 37〜66% にとどまるケースあり

---

## 11. ROCm ソフトウェアスタックの機能ギャップ分析（深掘り調査）

### 11.1 ライブラリ比較

| AMD (ROCm) | NVIDIA (CUDA) | 機能差 |
|------------|--------------|--------|
| rocBLAS | cuBLAS | 主要機能は同等。MI325X 世代で「ターニングポイント」 |
| MIOpen | cuDNN | 基本機能は対応。最適化の深さで 10〜30% のギャップ |
| hipBLASLt | cuBLASLt | GeMM 最適化。推論ワークロードの鍵 |
| Composable Kernel (CK) | CUTLASS | カスタムカーネルフレームワーク |
| Triton (ROCm) | Triton (CUDA) | OpenAI Triton のROCm バックエンド対応 |

### 11.2 FlashAttention on ROCm

- FlashAttention-2 が ROCm で利用可能
- 2つのバックエンド: **Composable Kernel (CK)**（デフォルト）と **Triton**
- CK バックエンドは H100 上の FlashAttention と同等の性能を MI300X で達成するケースあり
- ROCm の vLLM Docker イメージに最適化済み hipBLASLt + Triton + CK カーネルが統合

### 11.3 プロファイリングツール

| AMD | NVIDIA | 備考 |
|-----|--------|------|
| rocprof | Nsight Systems/Compute | 基本プロファイリング |
| **Omniperf** | Nsight Compute | ROCm 6.2 で導入、GPU カーネル詳細分析 |
| **Omnitrace** | Nsight Systems | ROCm 6.2 で導入、システムレベルトレース |

- ツールは**ベンダ固有**: Nsight は AMD GPU を、rocprof は NVIDIA GPU をプロファイルできない
- Omniperf/Omnitrace は比較的新しく、Nsight の成熟度にはまだ及ばない

### 11.4 コンテナ / Kubernetes / Slurm 対応

- **Docker**: ROCm 公式 Docker イメージを提供。vLLM 等の推論エンジン用プリビルドイメージあり
- **Kubernetes**: AMD GPU Operator が公開されており、ROCm チームが vLLM の Kubernetes デプロイガイドを発行
- **Slurm**: ROCm 環境での Slurm 対応はドキュメント化されている
- NVIDIA の GPU Operator / DCGM と比較するとエコシステムの成熟度に差あり

### 11.5 推論エンジン

- **NVIDIA**: TensorRT + Triton Inference Server（業界標準）
- **AMD**: 統一的な推論エンジンはなく、**vLLM on ROCm が事実上の推奨**
  - ROCm 最適化 vLLM Docker イメージが推奨構成
  - hipBLASLt + Triton + CK カーネルによる LLM 推論最適化
- TensorRT に相当する包括的な推論最適化スタックは AMD には存在しない

---

## 12. 実世界の推論ベンチマーク（深掘り調査）

### 12.1 vLLM on ROCm vs CUDA

#### Llama 3.1 70B (vLLM)

| メトリクス | MI300X (ROCm) | H100 (CUDA) | 備考 |
|-----------|--------------|-------------|------|
| スループット | 2,373〜3,774 tokens/s | — | max_num_batched_tokens 256〜4096 |
| 最適設定 | max_num_batched_tokens ≥ 2048 で最適 | — | |
| 性能比 | **H100 と同等（on-par）** | 基準 | vLLM 最適化済み環境 |

- MI300X は最適化された vLLM 環境で H100 と同等の推論スループットを達成するケースあり
- TTFT（Time To First Token）は vLLM が全同時ユーザーレベルで最高クラス

### 12.2 SemiAnalysis 推論ベンチマーク

| モデル | MI300X vs H200 | 備考 |
|--------|---------------|------|
| 大部分のシナリオ | MI300X は H200 に**劣る** | 絶対性能・性能/ドルとも |
| Llama 3 405B | MI300X が **H100 を上回る** | 大規模モデルでメモリ容量の優位性 |
| DeepSeek v3 670B | MI300X が **H100 を上回る** | 同上 |

- MI300X の192GB HBM3 は大規模モデル（400B+）で H100（80GB）に対して明確な優位
- ただし H200（141GB HBM3e）には多くのシナリオで劣る
- **コスト効率**: MI300X の価格が H200 より大幅に安い場合、TCO では有利になるケースあり

### 深掘り調査のソース

- [AMD Instinct MI300X Reference Cluster Design v4.0](https://instinct.docs.amd.com/projects/gpu-cluster-networking/en/latest/)
- [AMD Instinct NUMA Topology Mapping](https://instinct.docs.amd.com/projects/system-acceptance/en/latest/network/topology-mapping.html)
- [Supermicro MI300X 8U System](https://www.supermicro.com/en/accelerators/amd-instinct-mi300x)
- [Dell PowerEdge XE9680 with MI300X](https://www.dell.com/support/kbdoc/en-us/000226523/)
- [MI300X vs H100 vs H200 Benchmark - SemiAnalysis](https://newsletter.semianalysis.com/p/mi300x-vs-h100-vs-h200-benchmark-part-1-training)
- [MSCCL++ Performance](https://www.microsoft.com/en-us/research/publication/msccl-a-general-purpose-gpu-communication-library/)
- [FlashAttention ROCm - CK Backend](https://github.com/ROCm/flash-attention)
- [AMD ROCm vLLM Docker Image](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/how-to/docker.html)
- [AMD GPU Operator for Kubernetes](https://rocm.docs.amd.com/projects/gpu-operator/en/latest/)
- [MI355X DLC 1400W TDP](https://www.amd.com/en/products/accelerators/instinct/mi300.html)
- [LLM-Inference-Bench](https://arxiv.org/abs/2411.00136)
