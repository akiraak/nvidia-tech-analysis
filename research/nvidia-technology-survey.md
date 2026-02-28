# NVIDIA 主要技術・製品・プラットフォーム 網羅的調査メモ

調査日: 2026-02-27

---

## 目次

1. [ネットワーキング（NIC, スイッチ, DPU など）](#1-ネットワーキングnic-スイッチ-dpu-など)
2. [GPU / アクセラレータ](#2-gpu--アクセラレータ)
3. [インターコネクト技術（NVLink, NVSwitch など）](#3-インターコネクト技術nvlink-nvswitch-など)
4. [ソフトウェアプラットフォーム / SDK](#4-ソフトウェアプラットフォーム--sdk)
5. [AI / データセンター基盤](#5-ai--データセンター基盤)
6. [その他の主要技術](#6-その他の主要技術)

---

## 1. ネットワーキング（NIC, スイッチ, DPU など）

### 1.1 NIC / SuperNIC 製品ライン

#### ConnectX-9 SuperNIC（次世代・Rubin 世代）

- **概要**: NVIDIA Rubin プラットフォーム向けの次世代ネットワークアクセラレータカード。GPU あたり 1.6 Tb/s のネットワーク帯域幅を提供し、RDMA（Remote Direct Memory Access）機能を内蔵する。
- **用途**: Rubin 世代の AI ファクトリにおける scale-out ネットワーキング。NVIDIA Spectrum-X Ethernet と組み合わせて、RoCE（RDMA over Converged Ethernet）性能を最適化し、大規模 AI ワークロードに対して一貫した予測可能なネットワーク性能を実現する。
- **主な仕様**:
  - 1.6 Tb/s の帯域幅（GPU あたり）
  - InfiniBand / Ethernet 両対応
  - Vera Rubin NVL72 プラットフォームの標準構成要素
- **出荷時期**: 2026 年後半（Rubin プラットフォームと同時期）
- **参考URL**: [NVIDIA Rubin Platform](https://nvidianews.nvidia.com/news/rubin-platform-ai-supercomputer), [Inside the NVIDIA Rubin Platform](https://developer.nvidia.com/blog/inside-the-nvidia-rubin-platform-six-new-chips-one-ai-supercomputer)

#### ConnectX-8 SuperNIC（現行最新世代・Blackwell 世代）

- **概要**: 最高性能の 800G ネットワーキングを実現する SuperNIC。PCIe Gen6 スイッチを内蔵した統合設計により、ディスクリートな PCIe スイッチが不要になり、システム設計の簡素化と電力・コスト効率の向上を実現する。
- **用途**: Blackwell 世代の AI プラットフォーム（DGX B200/B300 など）における高速ネットワーキング。AI ワークロードの collective communication を加速し、AI ファクトリ全体のスケーラビリティを向上させる。
- **主な仕様**:
  - 最大 800 Gb/s（InfiniBand / Ethernet 両対応）
  - PCIe Gen6 x48 レーン対応、内蔵 PCIe Gen6 スイッチ
  - NVIDIA MGX ConnectX-8 SuperNIC Switch 構成では 4 枚の ConnectX-8 で合計 3,200 Gb/s の帯域幅
  - GPU あたりのネットワーク帯域幅を 2 倍に拡大
  - NCCL all-to-all 性能が最大 2 倍向上
- **参考URL**: [ConnectX-8 SuperNIC Technical Blog](https://developer.nvidia.com/blog/nvidia-connectx-8-supernics-advance-ai-platform-architecture-with-pcie-gen6-connectivity/), [ConnectX-8 SuperNIC at ServeTheHome](https://www.servethehome.com/nvidia-connectx-8-supernic-pcie-gen6-800g-nic-detailed/)

#### ConnectX-7（前世代）

- **概要**: 前世代の SmartNIC / NIC。400 Gb/s 対応の InfiniBand NDR / Ethernet ネットワークアダプタ。Hopper 世代のシステムで広く使用される。
- **用途**: H100/H200 ベースのデータセンターにおけるネットワーキング基盤。
- **主な仕様**:
  - 最大 400 Gb/s（InfiniBand NDR / Ethernet）
  - PCIe Gen5 x16 対応

---

### 1.2 DPU（Data Processing Unit）

#### BlueField-4 DPU（次世代・Rubin 世代）

- **概要**: NVIDIA Grace CPU と ConnectX-9 ネットワーキングを統合した次世代 DPU。BlueField-3 と比較して 6 倍のコンピュート性能を備え、最大 4 倍大規模な AI ファクトリをサポートする。800 Gb/s のスループットに対応。
- **用途**: AI ファクトリにおけるネットワーキング、ストレージ、セキュリティ機能のオフロード。CPU を解放してアプリケーションとワークロードの実行に専念させる。
- **主な仕様**:
  - NVIDIA Grace CPU コア統合
  - ConnectX-9 ネットワーキング統合
  - 800 Gb/s スループット
  - BlueField-3 比 6 倍のコンピュート性能
- **新機能**: BlueField Astra（Advanced Secure Trusted Resource Architecture）を搭載。BlueField-4 DPU と ConnectX-9 SuperNIC の間の専用接続を通じて、East-West ファブリックへの管理、プロビジョニング、ポリシー適用を拡張する。
- **出荷時期**: 2026 年後半
- **参考URL**: [BlueField-4 AI Factory Blog](https://blogs.nvidia.com/blog/bluefield-4-ai-factory/), [BlueField Astra Technical Blog](https://developer.nvidia.com/blog/redefining-secure-ai-infrastructure-with-nvidia-bluefield-astra-for-nvidia-vera-rubin-nvl72)

#### BlueField-3 DPU / SuperNIC（現行世代）

- **概要**: 現行世代の DPU。Arm コア、400 Gb/s ネットワーキング、DPU 機能を統合。SuperNIC モード（B3140H）ではネットワークアクセラレータとして機能し、DPU モードではフルスタックのインフラストラクチャ処理が可能。
- **用途**: Blackwell 世代までのデータセンターにおけるネットワーキング、ストレージ、セキュリティのオフロード。Spectrum-X プラットフォームの構成要素としてRoCE 最適化にも使用。
- **主な仕様**:
  - 8 Arm コア、16GB DDR5 メモリ
  - 最大 400 Gb/s（Ethernet / InfiniBand NDR）
  - QSFP112 ポート x1
  - PCIe 5.0 x16
  - Half-height half-length（HHHL）フォームファクタ
  - 統合 BMC
- **参考URL**: [BlueField Networking Platform](https://www.nvidia.com/en-us/networking/products/data-processing-unit/), [BlueField-3 Datasheet](https://resources.nvidia.com/en-us-accelerated-networking-resource-library/datasheet-nvidia-bluefield)

#### BlueField-2 DPU（前世代）

- **概要**: BlueField シリーズの初期世代 DPU。8 Arm コアと 200 Gb/s ネットワーキングを搭載。
- **用途**: レガシーデータセンターにおけるインフラストラクチャのオフロード。
- **参考URL**: [BlueField-2 Datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/documents/datasheet-nvidia-bluefield-2-dpu.pdf)

---

### 1.3 Ethernet スイッチ

#### Spectrum-X Photonics Ethernet スイッチ（次世代・2026 年）

- **概要**: Co-Packaged Optics（CPO）技術を採用した次世代 Ethernet スイッチ。Silicon Photonics により、AI ファクトリを数百万 GPU 規模にスケールさせることが可能。200G SerDes 通信回路と AI 最適化ファブリックを備える。
- **用途**: 超大規模 AI ファクトリの scale-out ネットワーキング。数百万 GPU を接続する大規模クラスタの構築。
- **主な仕様**:
  - 128 ポート x 800 Gb/s（合計 100 Tb/s）構成
  - 512 ポート x 800 Gb/s（合計 400 Tb/s）構成
  - 3D スタック Silicon Photonics エンジン（世界初）
  - 従来比 3.5 倍の電力効率、63 倍の信号完全性、10 倍のネットワーク回復力
- **出荷時期**: 2026 年後半
- **参考URL**: [Spectrum-X Photonics Announcement](https://nvidianews.nvidia.com/news/nvidia-spectrum-x-co-packaged-optics-networking-switches-ai-factories), [Silicon Photonics Networking](https://www.nvidia.com/en-us/networking/products/silicon-photonics/)

#### Spectrum-X Ethernet ネットワーキングプラットフォーム（現行世代）

- **概要**: AI ワークロードに最適化されたエンドツーエンドの Ethernet ネットワーキングプラットフォーム。Spectrum-4 スイッチ ASIC と BlueField-3 SuperNIC を組み合わせ、RoCE adaptive routing、RoCE congestion control、RoCE performance isolation などの AI 最適化機能を提供する。
- **用途**: 生成 AI（LLM トレーニング・推論）、分散学習、NLP ワークロード向けの Ethernet ベースの AI ネットワーキング。InfiniBand の代替として Ethernet で高性能 AI ネットワーキングを実現。
- **主な特徴**:
  - RoCE adaptive routing: ECMP リンク上の細粒度ロードバランシング
  - RoCE congestion control: テレメトリベースの輻輳制御
  - RoCE performance isolation: マルチテナント環境での性能分離
  - 最大 95% の有効帯域幅を実現
  - ストレージ読み取り帯域幅を最大 48%、書き込み帯域幅を最大 41% 向上
- **採用事例**: Meta、Oracle が導入
- **参考URL**: [Spectrum-X Platform](https://www.nvidia.com/en-us/networking/spectrumx/), [Spectrum-X Technical Blog](https://developer.nvidia.com/blog/turbocharging-ai-workloads-with-nvidia-spectrum-x-networking-platform/)

#### Spectrum-4 スイッチ ASIC（現行世代）

- **概要**: 世界初の 400 Gb/s エンドツーエンド Ethernet プラットフォーム向けスイッチ ASIC。前世代比 4 倍の switching throughput を実現する 51.2 Tb/s のスイッチングプラットフォーム。
- **用途**: AI データセンター、HPC、クラウド向けの高性能 Ethernet スイッチング。
- **主な仕様**:
  - 51.2 Tb/s 双方向帯域幅
  - 最大 128 ポート x 400 Gb/s
  - 66.5 billion packets per second
  - ナノ秒精度のタイミング（ミリ秒ベースと比較して 5〜6 桁の改善）
  - MACsec / VXLANsec セキュリティ機能
  - Hardware Root of Trust によるセキュアブート
  - 前世代比 40% 低消費電力
- **対応 NOS**: NVIDIA Cumulus Linux, SONiC, Linux Switch driver
- **スイッチ製品**: SN5600 シリーズ（2U フォームファクタ）
- **参考URL**: [Spectrum-4 ASIC Datasheet](https://resources.nvidia.com/en-us-accelerated-networking-resource-library/ethernet-switches), [NVIDIA Ethernet Switching](https://www.nvidia.com/en-us/networking/ethernet-switching/)

---

### 1.4 InfiniBand スイッチ

#### Quantum-X Photonics InfiniBand スイッチ（次世代・2026 年）

- **概要**: Co-Packaged Optics（CPO）技術を採用した次世代 InfiniBand スイッチ。Silicon Photonics による超高速・低消費電力の光通信を実現。
- **用途**: 次世代 AI ファクトリの scale-out InfiniBand ネットワーキング。
- **主な仕様**:
  - 144 ポート x 800 Gb/s InfiniBand
  - 200 Gb/s SerDes ベース
  - 115 Tb/s スループット
  - 液冷設計
- **出荷時期**: 2026 年前半
- **参考URL**: [Quantum-X Photonics Announcement](https://nvidianews.nvidia.com/news/nvidia-spectrum-x-co-packaged-optics-networking-switches-ai-factories), [Tom's Hardware Coverage](https://www.tomshardware.com/networking/nvidias-silicon-photonics-based-1-6-tb-s-switch-platforms-enable-clusters-with-millions-of-gpus)

#### Quantum-2 InfiniBand スイッチ（QM9700 シリーズ・現行世代）

- **概要**: NDR（Next Data Rate）400 Gb/s InfiniBand に対応した現行世代のスイッチファミリ。ソフトウェア定義ネットワーキング、In-Network Computing、SHARP、adaptive routing などの先進機能を搭載。
- **用途**: HPC スーパーコンピュータ、AI トレーニングクラスタの InfiniBand ファブリック構築。
- **主な仕様**:
  - 64 ポート x 400 Gb/s（または 128 ポート x 200 Gb/s）
  - 51.2 Tb/s 双方向スループット
  - 66.5 billion packets per second
  - 1U コンパクト設計（空冷・液冷バージョン）
  - RDMA, adaptive routing, SHARP 対応
  - Fat Tree, SlimFly, DragonFly+, multi-dimensional Torus など多様なトポロジをサポート
- **参考URL**: [Quantum-2 Platform](https://www.nvidia.com/en-us/networking/quantum2/), [QM9700 Datasheet](https://solutions.asbis.com/api/uploads/files/40/infiniband-quantum-2-qm9700-series-datasheet-us-nvidia-1751454-r8-web-1.pdf)

---

### 1.5 X1600 スイッチ（Rubin 世代）

- **概要**: Rubin プラットフォームの一部として 2026 年に登場予定の InfiniBand / Ethernet スイッチ。Spectrum-6 / Quantum-X 世代の技術を統合すると推測される。
- **参考URL**: [Rubin Platform Announcement](https://nvidianews.nvidia.com/news/rubin-platform-ai-supercomputer)

---

## 2. GPU / アクセラレータ

### 2.1 Rubin アーキテクチャ（次世代・2026 年〜）

#### Rubin GPU

- **概要**: Blackwell の次世代となる GPU アーキテクチャ。HBM4 メモリと新しい NVIDIA Transformer Engine を搭載。Blackwell 比でトレーニング 3.5 倍、推論 5 倍の性能向上を実現する。
- **用途**: 次世代 AI ファクトリにおけるトレーニングと推論。大規模 MoE モデルのトレーニングに必要な GPU 数を 4 分の 1 に削減、推論トークンコストを 10 分の 1 に削減。
- **主な仕様**:
  - 最大 50 PFLOPS
  - HBM4 メモリ（288 GB）
  - 新世代 Transformer Engine
  - NVLink 6 対応（3.6 TB/s GPU-to-GPU 帯域幅）
- **出荷時期**: 2026 年後半
- **参考URL**: [NVIDIA Rubin Platform](https://nvidianews.nvidia.com/news/rubin-platform-ai-supercomputer), [Inside the Rubin Platform](https://developer.nvidia.com/blog/inside-the-nvidia-rubin-platform-six-new-chips-one-ai-supercomputer)

#### Rubin Ultra GPU（2027 年予定）

- **概要**: Rubin GPU の強化版。さらなる性能向上を予定。
- **出荷時期**: 2027 年
- **参考URL**: [Tom's Hardware Rubin Roadmap](https://www.tomshardware.com/pc-components/gpus/nvidia-announces-rubin-gpus-in-2026-rubin-ultra-in-2027-feynam-after)

#### Rubin CPX

- **概要**: 大規模コンテキスト推論に特化した新しいクラスの GPU。100 万トークン級のコーディングや生成動画アプリケーション向け。
- **出荷時期**: 2026 年末
- **参考URL**: [NVIDIA Rubin CPX](https://nvidianews.nvidia.com/news/nvidia-unveils-rubin-cpx-a-new-class-of-gpu-designed-for-massive-context-inference)

---

### 2.2 Blackwell アーキテクチャ（現行世代・2024〜2025 年）

#### Blackwell Ultra B300 GPU

- **概要**: Blackwell アーキテクチャの強化版。AI 推論のリーズニングとテスト時計算に最適化。デュアルレチクル設計で 208B トランジスタ、160 Streaming Multiprocessors（SM）を 2 ダイに搭載し、NV-HBI（High-Bandwidth Interface）で接続。
- **用途**: AI ファクトリにおけるトレーニングと推論。特にリーズニング・エージェント型 AI ワークロードに最適化。
- **主な仕様**:
  - 15 PFLOPS dense NVFP4
  - 288 GB HBM3e（12-high スタック）
  - 8 TB/s メモリ帯域幅
  - 1,400 W TDP
  - NVFP4 精度フォーマット（FP8 比 1.8 倍のメモリフットプリント削減）
  - NVLink 5 対応
- **出荷時期**: 2025 年後半
- **参考URL**: [Blackwell Ultra Technical Blog](https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/), [Tom's Hardware B300](https://www.tomshardware.com/pc-components/gpus/nvidia-announces-blackwell-ultra-b300-1-5x-faster-than-b200-with-288gb-hbm3e-and-15-pflops-dense-fp4)

#### Blackwell B200 GPU

- **概要**: Blackwell アーキテクチャの初期バージョン。前世代 Hopper 比で最大 25 倍のコスト・エネルギー効率改善を実現。
- **用途**: 兆パラメータ規模の大規模言語モデル（LLM）のリアルタイム生成 AI。
- **主な仕様**:
  - 208B トランジスタ
  - TSMC 4NP プロセス
  - HBM3e メモリ
  - NVLink 5 対応
- **参考URL**: [Blackwell Architecture](https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/), [Blackwell Platform Announcement](https://nvidianews.nvidia.com/news/nvidia-blackwell-platform-arrives-to-power-a-new-era-of-computing)

---

### 2.3 Hopper アーキテクチャ（前世代・2022〜2024 年）

#### H100 GPU

- **概要**: Hopper アーキテクチャの主力データセンター GPU。第 4 世代 Tensor Core と FP8 精度対応の Transformer Engine を搭載。現在も広く運用されている世代。
- **用途**: AI トレーニング・推論、HPC、科学計算。GPT-3（175B）モデルのトレーニングで前世代比最大 4 倍高速化。
- **主な仕様**:
  - 80 GB HBM3e メモリ
  - 3.35 TB/s メモリ帯域幅
  - 67 TFLOPS FP32（SXM5）
  - 1,979 TFLOPS FP16
  - 700 W TDP（SXM5）
  - NVLink 4.0（900 GB/s per GPU）
  - Grace CPU と NVLink-C2C で接続可能（900 GB/s）
- **参考URL**: [H100 GPU](https://www.nvidia.com/en-us/data-center/h100/), [Hopper Architecture In-Depth](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)

#### H200 GPU

- **概要**: H100 のメモリ強化版。コンピュートは H100 と同一だが、HBM3e メモリ容量と帯域幅を大幅に拡張。
- **用途**: LLM 推論の高速化（H100 比最大 1.7 倍）、HPC（最大 1.3 倍性能向上）。
- **主な仕様**:
  - 141 GB HBM3e メモリ（GPU 初の HBM3e 搭載）
  - 4.8 TB/s メモリ帯域幅
  - 241.3 TFLOPS FP16
  - NVLink 4.0 対応（最大 4 GPU 接続）
- **参考URL**: [H200 GPU](https://www.nvidia.com/en-us/data-center/h200/)

---

### 2.4 CPU

#### Vera CPU（次世代・Rubin 世代・2026 年）

- **概要**: NVIDIA が独自設計した Arm ベースのデータセンター CPU。88 個のカスタム設計 Olympus コアを搭載し、AI ファクトリに最適化。CPU として初の FP8 精度サポート。
- **用途**: Rubin GPU と組み合わせた Vera Rubin Superchip として AI ファクトリの CPU コンポーネントを担う。
- **主な仕様**:
  - 88 Olympus コア（Armv9.2 互換）
  - NVIDIA Spatial Multithreading による 176 スレッド
  - 最大 1.5 TB SOCAMM LPDDR5X メモリ
  - 最大 1.2 TB/s メモリ帯域幅
  - 3.4 TB/s bisection bandwidth の SCF（第 2 世代）
  - NVLink-C2C: 1.8 TB/s coherent bandwidth
  - TDP: 50 W
  - Grace CPU 比 2 倍の性能
- **出荷時期**: 2026 年後半
- **参考URL**: [NVIDIA Vera CPU](https://www.nvidia.com/en-us/data-center/vera-cpu/), [Vera Rubin NVL72 Details](https://videocardz.com/newz/nvidia-vera-rubin-nvl72-detailed-72-gpus-36-cpus-260-tb-s-scale-up-bandwidth)

#### Grace CPU（現行世代）

- **概要**: NVIDIA 初のデータセンター向け Arm ベース CPU。72 個の Arm Neoverse V2 コアを搭載し、高性能と優れたエネルギー効率を両立。サーバクラスの LPDDR5X メモリを初めて採用した CPU。
- **用途**: AI、HPC、クラウドゲーミングワークロード。Grace Blackwell / Grace Hopper Superchip として GPU と組み合わせて使用。
- **主な仕様**:
  - 72 Arm Neoverse V2 コア
  - 最大 500 GB/s LPDDR5X メモリ帯域幅（DDR 比 1/5 の消費電力）
  - 3.2 TB/s bisection bandwidth の SCF（Scalable Coherency Fabric）
  - Grace CPU Superchip: 2 CPU を NVLink-C2C（900 GB/s 双方向）で接続
    - 144 コア、最大 1 TB/s メモリ帯域幅
  - 従来サーバ比 2 倍の性能/ワット、2 倍の実装密度
  - AArch64 バイナリ完全互換
- **参考URL**: [Grace CPU](https://www.nvidia.com/en-us/data-center/grace-cpu/), [Grace CPU Superchip Architecture](https://developer.nvidia.com/blog/nvidia-grace-cpu-superchip-architecture-in-depth/)

---

## 3. インターコネクト技術（NVLink, NVSwitch など）

### 3.1 NVLink

NVLink は NVIDIA GPU 間の高速インターコネクト技術。世代を重ねるごとに帯域幅が大幅に向上している。

| 世代 | 帯域幅（per GPU） | リンク数 | 対応アーキテクチャ | 登場年 |
|------|-------------------|----------|-------------------|--------|
| NVLink 4.0 | 900 GB/s | 18 links x 50 GB/s | Hopper (H100/H200) | 2022 |
| NVLink 5.0 | 1.8 TB/s | 18 links x 100 GB/s | Blackwell (B200/B300) | 2024 |
| NVLink 6.0 | 3.6 TB/s | - | Rubin | 2026 |

- **概要**: GPU 間の直接接続を提供する高帯域幅・低レイテンシのインターコネクト。PCIe の帯域幅制限を大幅に超える性能を実現する。NVLink 5 は PCIe Gen5 の 14 倍の帯域幅を提供。
- **用途**: マルチ GPU システム内の GPU 間通信。AI トレーニングにおける大規模モデルの分散処理に必須。
- **参考URL**: [NVLink & NVSwitch](https://www.nvidia.com/en-us/data-center/nvlink/), [NVLink Explained](https://intuitionlabs.ai/articles/nvidia-nvlink-gpu-interconnect)

### 3.2 NVSwitch

- **概要**: NVLink ファブリックをノード規模に拡張するスイッチチップ。複数の GPU を non-blocking で接続し、全対全通信を可能にする。
- **用途**: DGX / NVL72 システム内の GPU 間通信のバックボーン。
- **世代**:
  - **NVSwitch（Blackwell 世代）**: GB200 NVL72 で 72 GPU を 130 TB/s の aggregate bandwidth で接続。576 GPU を non-blocking ファブリックで 1 PB/s の合計帯域幅で接続可能。
  - **NVLink 6 Switch（Rubin 世代）**: Rubin プラットフォーム向け。3.6 TB/s GPU-to-GPU 帯域幅を提供。
- **参考URL**: [NVLink & NVSwitch](https://www.nvidia.com/en-us/data-center/nvlink/), [NVLink Wikipedia](https://en.wikipedia.org/wiki/NVLink)

### 3.3 NVLink Spine

- **概要**: ラック内で最大 72 GPU を接続する物理的なインターコネクト構造体。数千本の超高密度同軸リンクと複数の NVLink スイッチチップで構成される。
- **用途**: NVL72 ラックスケールシステムの内部 GPU 間接続。
- **主な仕様**:
  - 130 TB/s の持続スループット（単一 Spine で全インターネットのピーク帯域幅を上回る）
  - NVLink Fusion スイッチチップ: 144 NVLink ポート、14.4 TB/s non-blocking 帯域幅
- **参考URL**: [NVLink Spine Article](https://medium.com/@fahey_james/nvidias-nvlink-spine-reinventing-the-data-center-for-the-ai-age-0d298bd29d17), [Glitchwire NVLink Spine](https://www.glitchwire.com/news/nvidia-s-nv-link-spine-the-quiet-backbone-of-ai-s-next-leap)

### 3.4 NVLink Fusion

- **概要**: ハイパースケーラーやカスタム ASIC 設計者が、カスタム CPU や XPU を NVLink scale-up インターコネクトおよび OCP MGX ラックスケールサーバアーキテクチャと統合できるようにするプラットフォーム。NVLink のエコシステムをサードパーティに開放する戦略的な取り組み。
- **用途**: NVIDIA GPU 以外のプロセッサ（カスタム CPU、XPU）を NVLink ファブリックに統合。セミカスタム AI インフラストラクチャの構築。
- **パートナー**: MediaTek, Marvell, Alchip Technologies, Astera Labs, Synopsys, Cadence, Fujitsu, Qualcomm Technologies
- **発表**: 2025 年 5 月 COMPUTEX
- **参考URL**: [NVLink Fusion](https://www.nvidia.com/en-us/data-center/nvlink-fusion/), [NVLink Fusion Announcement](https://nvidianews.nvidia.com/news/nvidia-nvlink-fusion-semi-custom-ai-infrastructure-partner-ecosystem)

### 3.5 NVLink-C2C（Chip-to-Chip）

- **概要**: チップ間を直接接続する超高速・低レイテンシのインターコネクト。Grace CPU と GPU の間、または 2 つの Grace CPU の間のコヒーレント接続に使用。
- **用途**: Grace Hopper / Grace Blackwell / Vera Rubin Superchip 内の CPU-GPU 接続。
- **主な仕様**:
  - Grace-Hopper: 900 GB/s 双方向
  - Vera CPU: 1.8 TB/s coherent bandwidth

---

## 4. ソフトウェアプラットフォーム / SDK

### 4.1 CUDA

- **概要**: NVIDIA GPU でアクセラレーテッドコンピューティングを実現するための並列計算プラットフォームおよびプログラミングモデル。C++、Python、Fortran などの言語をサポートし、PyTorch などのフレームワークの基盤となる。
- **用途**: GPU プログラミングの基盤。全ての NVIDIA GPU アクセラレーション技術の根幹をなすソフトウェアプラットフォーム。
- **参考URL**: [CUDA Platform](https://developer.nvidia.com/cuda), [CUDA Toolkit](https://developer.nvidia.com/cuda/toolkit)

### 4.2 DOCA（Data Center Infrastructure on a Chip Architecture）

- **概要**: BlueField DPU / SuperNIC 上でアプリケーションやサービスを迅速に作成・管理するためのソフトウェアフレームワーク。GPU における CUDA に相当するもの。業界標準 API を活用し、ネットワーキング、セキュリティ、ストレージの性能を向上させる。
- **用途**: BlueField DPU / SuperNIC 上でのアプリケーション開発。ネットワーキング、セキュリティ、ストレージ機能のプログラミング。
- **含まれる SDK**:
  - RDMA acceleration SDK
  - Network acceleration SDK
  - Security acceleration SDK
  - Storage acceleration SDK
  - Data path acceleration SDK
  - Management SDK
- **対応 API**: DPDK, SPDK, P4, Linux Netlink
- **最新バージョン**: DOCA 3.2.1 LTS（2025 年 11 月）、DOCA 2.9.4 LTS / 2.5.5 LTS（2026 年 1 月）
- **参考URL**: [DOCA Framework](https://developer.nvidia.com/networking/doca), [DOCA Overview](https://docs.nvidia.com/doca/sdk/doca-overview/index.html)

### 4.3 NCCL（NVIDIA Collective Communications Library）

- **概要**: GPU 間およびマルチノード間の通信を最適化するライブラリ。AI / HPC における効率的な並列計算に不可欠。Magnum IO スタックの一部。
- **用途**: 分散 AI トレーニングにおける collective communication（AllReduce, AllGather, ReduceScatter など）の高速化。
- **主な特徴**:
  - インターコネクト帯域幅の 100% 近くを達成
  - GPU カーネルおよびネットワークプロファイラサポート
  - QoS サポート（重要なネットワーク通信の優先化）
  - RAS 改善（診断出力と安定性の向上）
- **最新バージョン**: NCCL 2.26
- **参考URL**: [NCCL Developer Page](https://developer.nvidia.com/nccl), [NCCL 2.26 Blog](https://developer.nvidia.com/blog/improved-performance-and-monitoring-capabilities-with-nvidia-collective-communications-library-2-26/)

### 4.4 NVSHMEM

- **概要**: NVIDIA GPU 向けの OpenSHMEM 実装。複数 GPU のメモリを PGAS（Partitioned Global Address Space）として統合し、CUDA カーネル内からの細粒度な GPU-to-GPU データ移動と同期を実現。
- **用途**: 分散 GPU コンピューティングにおける低レイテンシ通信。HPC アプリケーションの GPU 間通信最適化。
- **主な特徴**:
  - CUDA カーネル内からの one-sided read（get）/ write（put）/ atomic update
  - NVLink, PCIe, InfiniBand 上で動作
  - UCX リモートトランスポートに対応
- **最新バージョン**: NVSHMEM 3.0.6
- **参考URL**: [NVSHMEM GitHub](https://github.com/NVIDIA/NVSHMEM/releases), [NVSHMEM Documentation](https://docs.nvidia.com/nvshmem/release-notes-install-guide/prior-releases/release-3006.html)

### 4.5 Magnum IO

- **概要**: アクセラレーテッドデータセンター向けの IO ソフトウェアスタック。ハードウェアレベルのアクセラレーションエンジン（RDMA, GPUDirect, SHARP）を活用し、NVIDIA InfiniBand / NVLink ネットワークの高帯域幅・超低レイテンシを最大限に引き出す。
- **用途**: データセンターにおけるデータ移動の最適化。AI / HPC ワークロードのデータボトルネック解消。
- **含まれるコンポーネント**: NCCL, NVSHMEM, GPUDirect, SHARP, UCX など
- **参考URL**: [Magnum IO](https://www.nvidia.com/en-us/data-center/magnum-io/), [Magnum IO SDK](https://developer.nvidia.com/magnum-io)

### 4.6 Cumulus Linux

- **概要**: NVIDIA Spectrum スイッチ向けの Linux ベースのネットワークオペレーティングシステム。オープンなクラウドネイティブアーキテクチャにより、既存の Linux ツール・自動化フレームワークとの統合が容易。
- **用途**: NVIDIA Spectrum Ethernet スイッチの管理・運用。AI ファブリック、データセンターネットワークの構築。
- **最新バージョン**: Cumulus Linux 5.15
- **参考URL**: [Cumulus Linux](https://www.nvidia.com/en-us/networking/ethernet-switching/cumulus-linux/), [Cumulus Linux User Guide](https://docs.nvidia.com/networking-ethernet-software/cumulus-linux/)

### 4.7 NetQ

- **概要**: 高度にスケーラブルなネットワーク運用ツールセット。NVIDIA NVLink スイッチおよび Cumulus ファブリックの可視化、トラブルシューティング、相関分析、検証をリアルタイムで提供。
- **用途**: AI ネットワークファブリックのモニタリングと運用管理。CI/CD ワークフローによるネットワーク要素の管理・プロビジョニング。
- **主な特徴**:
  - OpenTelemetry と時系列データベースによるテレメトリデータの収集・相関
  - コンピュート、ファブリック、ワークロード層にまたがるデータ分析
  - Kubernetes 上でデプロイ
  - NVIDIA Mission Control との統合
- **最新バージョン**: NetQ 5.0（オンプレミス専用）
- **参考URL**: [NetQ](https://www.nvidia.com/en-us/networking/ethernet-switching/netq/), [NetQ 5.0 User Guide](https://docs.nvidia.com/networking-ethernet-software/cumulus-netq/)

### 4.8 UCX（Unified Communication X）

- **概要**: 複数のネットワーク API、プログラミングモデル、プロトコル、実装を抽象化するオープンソース通信フレームワーク。
- **用途**: GPU-centric 通信レイヤの実装。分散 GPU コンピューティングにおけるプロセス間通信の基盤。
- **主な特徴**:
  - get/put, send/receive, active messages (RPC) プリミティブ
  - CPU, GPU 間のエンドポイント間通信
  - CUDA / ROCm 対応
- **参考URL**: [UCX in Multi-Node Guide](https://docs.nvidia.com/multi-node-nvlink-systems/multi-node-tuning-guide/ucx.html)

---

## 5. AI / データセンター基盤

### 5.1 DGX システム

#### DGX B300（最新世代）

- **概要**: NVIDIA Blackwell Ultra GPU を搭載した AI ファクトリ向けサーバ。推論 192 PFLOPS、トレーニング 70 PFLOPS を実現する。
- **用途**: エンタープライズ AI ファクトリにおけるトレーニングと推論。リアルタイムエージェント応答。
- **主な仕様**:
  - 36 NVIDIA Blackwell Ultra GPU（GB300 NVL72 構成時）
  - 液冷ラックスケールアーキテクチャ
  - 推論 192 PFLOPS / トレーニング 70 PFLOPS
- **参考URL**: [DGX B300](https://www.nvidia.com/en-us/data-center/dgx-b300/)

#### DGX B200

- **概要**: 8 基の NVIDIA Blackwell GPU を搭載した AI プラットフォーム。第 5 世代 NVLink で GPU 間を接続し、前世代比トレーニング 3 倍、推論 15 倍の性能を実現。
- **用途**: エンタープライズ AI のトレーニング・推論パイプライン全体をカバーする統合プラットフォーム。
- **主な仕様**:
  - 8x NVIDIA Blackwell GPU
  - 1.4 TB GPU メモリ、64 TB/s メモリ帯域幅
  - 72 PFLOPS AI トレーニング性能
  - 10U フォームファクタ
  - Dual Intel Xeon Platinum 8570、4 TB DDR5
  - 8x 3.84 TB NVMe SSD（50 GB/s ピーク帯域幅）
  - NVIDIA Mission Control / AI Enterprise ソフトウェア含む
- **参考URL**: [DGX B200](https://www.nvidia.com/en-us/data-center/dgx-b200/), [DGX B200 Datasheet](https://resources.nvidia.com/en-us-dgx-systems/dgx-b200-datasheet)

### 5.2 DGX SuperPOD

- **概要**: 複数の DGX システムを統合したターンキー AI データセンターソリューション。世界クラスのコンピューティング、ソフトウェアツール、専門知識、継続的イノベーションをシームレスに提供。
- **用途**: エンタープライズ、高等教育、研究機関、公共セクターにおける最も困難な AI トレーニング・推論ワークロード。
- **構成オプション**:
  - DGX SuperPOD with DGX GB300: 36 Grace Blackwell Ultra Superchip（72 B300 GPU）、液冷、1.1 exaFLOPS dense FP4
  - DGX SuperPOD with DGX B300: Blackwell Ultra GPU ベース
  - DGX SuperPOD with DGX B200: Blackwell GPU ベース
  - DGX SuperPOD with DGX H100/H200: Hopper GPU ベース（レガシー）
- **導入事例**:
  - Lilly: 1,016 Blackwell Ultra GPU を搭載した製薬企業最大の AI ファクトリ
  - Mayo Clinic: DGX SuperPOD with DGX B200 による医療 AI ファクトリ
- **参考URL**: [DGX SuperPOD](https://www.nvidia.com/en-us/data-center/dgx-superpod/), [Blackwell Ultra DGX SuperPOD](https://nvidianews.nvidia.com/news/blackwell-ultra-dgx-superpod-supercomputer-ai-factories)

### 5.3 Vera Rubin NVL72

- **概要**: Rubin プラットフォームの旗艦システム。72 Rubin GPU と 36 Vera CPU を 1 つのシステムに統合した AI スーパーコンピュータ。
- **用途**: 次世代 AI ファクトリの基本単位。複数の NVL72 を組み合わせて DGX SuperPOD を構成。
- **主な仕様**:
  - 72 Rubin GPU（HBM4、各 288 GB）
  - 36 Vera CPU（各 88 Olympus コア）
  - 260 TB/s scale-up 帯域幅
  - NVLink 6 接続
  - Blackwell 比 5 倍の推論性能、10 倍の推論トークンコスト削減
- **出荷時期**: 2026 年後半
- **参考URL**: [Vera Rubin NVL72](https://videocardz.com/newz/nvidia-vera-rubin-nvl72-detailed-72-gpus-36-cpus-260-tb-s-scale-up-bandwidth), [CES 2026 Announcement](https://www.tomshardware.com/pc-components/gpus/nvidia-launches-vera-rubin-nvl72-ai-supercomputer-at-ces-promises-up-to-5x-greater-inference-performance-and-10x-lower-cost-per-token-than-blackwell-coming-2h-2026)

### 5.4 NVIDIA Instant AI Factory

- **概要**: Equinix マネージドサービスとして提供される Blackwell Ultra DGX SuperPOD。NVIDIA Mission Control ソフトウェアを搭載し、完全にプロビジョニングされた AI ファクトリを企業に提供。
- **用途**: 自社でインフラを構築せずに AI ファクトリを利用したいエンタープライズ向けのマネージドサービス。
- **参考URL**: [DGX SuperPOD Blog](https://blogs.nvidia.com/blog/dgx-superpod-rubin/)

### 5.5 NVIDIA Mission Control

- **概要**: AI ファクトリの運用管理ソフトウェア。NetQ を含む各種管理ツールを統合し、コンピュート・ファブリック・ワークロード全体の統合管理を実現。
- **用途**: DGX SuperPOD / AI ファクトリの管理・監視・プロビジョニング。
- **参考URL**: [NetQ Mission Control Integration](https://docs.nvidia.com/mission-control/docs/nmc-software-installation-guide/2.1.0/netq-installation.html)

---

## 6. その他の主要技術

### 6.1 GPUDirect ファミリ

#### GPUDirect RDMA

- **概要**: NVIDIA GPU とサードパーティデバイス（NIC、DPU、ビデオキャプチャアダプタなど）の間で PCI Express を介した直接データ交換を可能にする技術。GPU メモリへの直接読み書きにより、不要なメモリコピーを排除し、CPU オーバーヘッドとレイテンシを削減。
- **用途**: 分散 AI トレーニングにおけるマルチノード GPU 間通信。HPC の科学シミュレーション。
- **対応インターコネクト**: InfiniBand、RoCE（RDMA over Converged Ethernet）
- **参考URL**: [GPUDirect](https://developer.nvidia.com/gpudirect), [GPUDirect RDMA Documentation](https://docs.nvidia.com/cuda/gpudirect-rdma/index.html)

#### GPUDirect Storage

- **概要**: ストレージデバイスと GPU メモリ間のデータ移動を加速する技術。ストレージからの直接データロードにより、CPU メモリを経由するオーバーヘッドを排除。
- **用途**: AI トレーニングデータのロード高速化、HPC データ処理パイプラインの最適化。
- **参考URL**: [GPUDirect](https://developer.nvidia.com/gpudirect)

### 6.2 SHARP（Scalable Hierarchical Aggregation and Reduction Protocol）

- **概要**: InfiniBand スイッチ ASIC に組み込まれた In-Network Computing 技術。collective communication 操作（all-reduce, reduce, broadcast）をサーバのコンピュートエンジンからネットワークスイッチにオフロードする。
- **用途**: 分散 AI トレーニングおよび HPC における collective communication の高速化。MPI 操作の時間短縮。
- **性能**:
  - MPI AllReduce で 5 倍の性能向上
  - MPI Barrier で最大 9 倍の性能向上
  - AllReduce 帯域幅（GB/s）を 2 倍に
  - ネットワーク上の転送データ量を半減
  - SHARPv4: 前世代比 2 倍の帯域幅、9 倍の in-network compute
- **参考URL**: [SHARP Blog](https://developer.nvidia.com/blog/advancing-performance-with-nvidia-sharp-in-network-computing/), [SHARP Resource](https://resources.nvidia.com/en-us-accelerated-networking-resource-library/network-computing-nvidia-sharp)

### 6.3 Silicon Photonics / Co-Packaged Optics (CPO)

- **概要**: 従来の pluggable 光トランシーバに代わり、シリコンフォトニクスをスイッチ ASIC にダイレクトに統合する技術。TSMC の 3D Hybrid Bonding 技術を採用した世界初の 3D スタック Silicon Photonics エンジン。
- **用途**: 次世代 AI ファクトリにおける超大規模ネットワーク接続。数百万 GPU の接続を可能にする省電力・高性能光通信。
- **主な利点**:
  - 従来比 3.5 倍の電力効率
  - 63 倍の信号完全性
  - 10 倍のネットワーク回復力
  - 4 分の 1 のレーザー数
  - 1.3 倍の高速デプロイ
- **パートナー**: Lumentum（レーザー供給）、Coherent（Silicon Photonics 協業）
- **出荷時期**: Quantum-X: 2026 年前半、Spectrum-X Photonics: 2026 年後半
- **参考URL**: [Silicon Photonics Networking](https://www.nvidia.com/en-us/networking/products/silicon-photonics/), [CPO Technical Blog](https://developer.nvidia.com/blog/scaling-ai-factories-with-co-packaged-optics-for-better-power-efficiency/), [CPO Era Blog](https://developer.nvidia.com/blog/a-new-era-in-data-center-networking-with-nvidia-silicon-photonics-based-network-switching/)

### 6.4 BlueField Astra（Advanced Secure Trusted Resource Architecture）

- **概要**: BlueField-4 DPU 上で動作するセキュリティアーキテクチャ。BlueField-4 DPU と ConnectX-9 SuperNIC の間の専用接続を通じて、East-West ファブリックへの管理性、プロビジョニング、ポリシー適用を拡張。
- **用途**: AI ファクトリのセキュアインフラストラクチャ。Vera Rubin NVL72 におけるセキュリティ・管理基盤。
- **発表**: CES 2026
- **参考URL**: [BlueField Astra Technical Blog](https://developer.nvidia.com/blog/redefining-secure-ai-infrastructure-with-nvidia-bluefield-astra-for-nvidia-vera-rubin-nvl72)

### 6.5 NVIDIA Air

- **概要**: クラウドホスト型のネットワークシミュレーションプラットフォーム。実際の本番環境と同じ動作をするデジタルツインを作成し、フルスケールのネットワークアーキテクチャを検証できる。
- **用途**: データセンターネットワークのデジタルツイン。ネットワークプロビジョニング、自動化、セキュリティポリシーの事前テスト・検証。Spectrum-X AI ファブリックのデプロイ加速。
- **対応機器**: Spectrum Ethernet スイッチ（Cumulus Linux / SONiC）、BlueField DPU / SuperNIC、NetQ
- **特徴**: ベアメタルシミュレーション、x86 サーバエミュレーション、ビルド済みネットワークテンプレート、エミュレーション ASIC
- **参考URL**: [NVIDIA Air](https://www.nvidia.com/en-us/networking/ethernet-switching/air/), [Air Introduction Blog](https://developer.nvidia.com/blog/an-introduction-to-nvidia-air/)

### 6.6 NVIDIA Omniverse

- **概要**: 物理的 AI アプリケーション（産業デジタルツイン、ロボティクスシミュレーションなど）を開発するためのライブラリおよびマイクロサービスのコレクション。
- **用途**: 産業デジタル化、物理 AI シミュレーション。DGX Cloud 上の Omniverse によるアプリケーションストリーミング。
- **参考URL**: [Omniverse](https://www.nvidia.com/en-us/omniverse/), [Omniverse on DGX Cloud](https://www.nvidia.com/en-us/data-center/omniverse-dgx-cloud/)

---

## 技術ロードマップ（まとめ）

| 時期 | GPU | CPU | NIC | DPU | スイッチ | NVLink |
|------|-----|-----|-----|-----|---------|--------|
| 2022-2024 | Hopper (H100/H200) | Grace (72 Neoverse V2) | ConnectX-7 (400G) | BlueField-3 | Quantum-2 / Spectrum-4 | NVLink 4.0 (900 GB/s) |
| 2024-2025 | Blackwell (B200/B300) | Grace | ConnectX-8 (800G) | BlueField-3 | Quantum-2 / Spectrum-4 | NVLink 5.0 (1.8 TB/s) |
| 2026- | Rubin / Rubin CPX | Vera (88 Olympus) | ConnectX-9 (1.6T) | BlueField-4 | Quantum-X / Spectrum-X Photonics / X1600 | NVLink 6.0 (3.6 TB/s) |
| 2027- | Rubin Ultra | - | - | - | - | - |
| 2028- | Feynman（発表済み） | - | - | - | - | - |

---

## 参考 URL 一覧

### 公式ページ
- [NVIDIA NVLink & NVSwitch](https://www.nvidia.com/en-us/data-center/nvlink/)
- [NVIDIA Blackwell Architecture](https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/)
- [NVIDIA Hopper Architecture](https://www.nvidia.com/en-us/data-center/technologies/hopper-architecture/)
- [NVIDIA Grace CPU](https://www.nvidia.com/en-us/data-center/grace-cpu/)
- [NVIDIA Vera CPU](https://www.nvidia.com/en-us/data-center/vera-cpu/)
- [NVIDIA BlueField Networking Platform](https://www.nvidia.com/en-us/networking/products/data-processing-unit/)
- [NVIDIA Spectrum-X](https://www.nvidia.com/en-us/networking/spectrumx/)
- [NVIDIA Ethernet SuperNIC](https://www.nvidia.com/en-us/networking/products/ethernet/supernic/)
- [NVIDIA InfiniBand](https://www.nvidia.com/en-us/networking/products/infiniband/)
- [NVIDIA Quantum-2](https://www.nvidia.com/en-us/networking/quantum2/)
- [NVIDIA Ethernet Switching](https://www.nvidia.com/en-us/networking/ethernet-switching/)
- [NVIDIA Cumulus Linux](https://www.nvidia.com/en-us/networking/ethernet-switching/cumulus-linux/)
- [NVIDIA NetQ](https://www.nvidia.com/en-us/networking/ethernet-switching/netq/)
- [NVIDIA DGX SuperPOD](https://www.nvidia.com/en-us/data-center/dgx-superpod/)
- [NVIDIA DGX B200](https://www.nvidia.com/en-us/data-center/dgx-b200/)
- [NVIDIA DGX B300](https://www.nvidia.com/en-us/data-center/dgx-b300/)
- [NVIDIA NVLink Fusion](https://www.nvidia.com/en-us/data-center/nvlink-fusion/)
- [NVIDIA Silicon Photonics](https://www.nvidia.com/en-us/networking/products/silicon-photonics/)
- [NVIDIA Magnum IO](https://www.nvidia.com/en-us/data-center/magnum-io/)
- [NVIDIA CUDA](https://developer.nvidia.com/cuda)
- [NVIDIA DOCA](https://developer.nvidia.com/networking/doca)
- [NVIDIA NCCL](https://developer.nvidia.com/nccl)
- [NVIDIA GPUDirect](https://developer.nvidia.com/gpudirect)
- [NVIDIA Omniverse](https://www.nvidia.com/en-us/omniverse/)
- [NVIDIA Air](https://www.nvidia.com/en-us/networking/ethernet-switching/air/)

### ニュース・発表
- [Rubin Platform Announcement (CES 2026)](https://nvidianews.nvidia.com/news/rubin-platform-ai-supercomputer)
- [BlueField-4 AI Factory Blog](https://blogs.nvidia.com/blog/bluefield-4-ai-factory/)
- [Blackwell Ultra Platform Announcement (GTC 2025)](https://investor.nvidia.com/news/press-release-details/2025/NVIDIA-Blackwell-Ultra-AI-Factory-Platform-Paves-Way-for-Age-of-AI-Reasoning/default.aspx)
- [Spectrum-X Photonics Announcement](https://nvidianews.nvidia.com/news/nvidia-spectrum-x-co-packaged-optics-networking-switches-ai-factories)
- [NVLink Fusion Announcement (COMPUTEX 2025)](https://nvidianews.nvidia.com/news/nvidia-nvlink-fusion-semi-custom-ai-infrastructure-partner-ecosystem)
- [Blackwell Ultra DGX SuperPOD](https://nvidianews.nvidia.com/news/blackwell-ultra-dgx-superpod-supercomputer-ai-factories)
- [Rubin CPX Announcement](https://nvidianews.nvidia.com/news/nvidia-unveils-rubin-cpx-a-new-class-of-gpu-designed-for-massive-context-inference)

### 技術ブログ
- [Inside the Rubin Platform](https://developer.nvidia.com/blog/inside-the-nvidia-rubin-platform-six-new-chips-one-ai-supercomputer)
- [Inside Blackwell Ultra](https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/)
- [ConnectX-8 SuperNIC Architecture](https://developer.nvidia.com/blog/nvidia-connectx-8-supernics-advance-ai-platform-architecture-with-pcie-gen6-connectivity/)
- [BlueField Astra Security](https://developer.nvidia.com/blog/redefining-secure-ai-infrastructure-with-nvidia-bluefield-astra-for-nvidia-vera-rubin-nvl72)
- [SHARP In-Network Computing](https://developer.nvidia.com/blog/advancing-performance-with-nvidia-sharp-in-network-computing/)
- [Silicon Photonics Network Switching](https://developer.nvidia.com/blog/a-new-era-in-data-center-networking-with-nvidia-silicon-photonics-based-network-switching/)
- [Co-Packaged Optics Efficiency](https://developer.nvidia.com/blog/scaling-ai-factories-with-co-packaged-optics-for-better-power-efficiency/)
- [Spectrum-X AI Workloads](https://developer.nvidia.com/blog/turbocharging-ai-workloads-with-nvidia-spectrum-x-networking-platform/)
- [NCCL 2.26 Release](https://developer.nvidia.com/blog/improved-performance-and-monitoring-capabilities-with-nvidia-collective-communications-library-2-26/)
- [NVLink Fusion Inference Scaling](https://developer.nvidia.com/blog/scaling-ai-inference-performance-and-flexibility-with-nvidia-nvlink-and-nvlink-fusion/)
- [Grace CPU Architecture](https://developer.nvidia.com/blog/nvidia-grace-cpu-superchip-architecture-in-depth/)
- [Hopper Architecture In-Depth](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)

### 外部記事・分析
- [ServeTheHome: Rubin Platform at CES 2026](https://www.servethehome.com/nvidia-launches-next-generation-rubin-ai-compute-platform-at-ces-2026/)
- [Tom's Hardware: Blackwell Ultra B300](https://www.tomshardware.com/pc-components/gpus/nvidia-announces-blackwell-ultra-b300-1-5x-faster-than-b200-with-288gb-hbm3e-and-15-pflops-dense-fp4)
- [Tom's Hardware: Silicon Photonics Switches](https://www.tomshardware.com/networking/nvidias-silicon-photonics-based-1-6-tb-s-switch-platforms-enable-clusters-with-millions-of-gpus)
- [Network World: InfiniBand Hardware](https://www.networkworld.com/article/970450/nvidia-announces-new-infiniband-networking-hardware.html)
- [Network World: Networking Roadmap](https://www.networkworld.com/article/4050881/nvidia-networking-roadmap-ethernet-infiniband-co-packaged-optics-will-shape-data-center-of-the-future.html)
- [SDxCentral: Inside Spectrum-X](https://www.sdxcentral.com/analysis/inside-spectrum-x-nvidias-ethernet-networking-platform/)
- [SDxCentral: BlueField-4 DPU](https://www.sdxcentral.com/news/nvidia-reveals-next-gen-dpu-to-help-offload-gigascale-ai-infrastructure/)
- [NextPlatform: GPU System Roadmap to 2028](https://www.nextplatform.com/2025/03/19/nvidia-draws-gpu-system-roadmap-out-to-2028/)
- [IntuitionLabs: NVLink Guide](https://intuitionlabs.ai/articles/nvidia-nvlink-gpu-interconnect)
- [IntuitionLabs: Data Center GPU Specs](https://intuitionlabs.ai/articles/nvidia-data-center-gpu-specs)
- [HPCwire: UALink Competition](https://www.hpcwire.com/2025/12/02/upscale-ai-eyes-late-2026-for-scale-up-ualink-switch/)
