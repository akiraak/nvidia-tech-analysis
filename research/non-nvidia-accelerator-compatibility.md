# 他社 AI 半導体と NVIDIA GPU 以外の技術の組み合わせ可能性

調査日: 2026-02-28

## 概要

NVIDIA は GPU 以外にも NIC/DPU、Ethernet/InfiniBand スイッチ、高速インターコネクト、通信ライブラリ、SDK など多数のデータセンター技術を保有している。本調査では、これらの技術が他社の AI 半導体（Google TPU、AMD Instinct、Intel Gaudi、AWS Trainium 等）と組み合わせて使用可能かどうかを技術的・ビジネス的観点から整理する。

---

## 1. ConnectX SuperNIC / BlueField DPU

### 1.1 AMD GPU + ConnectX NIC の組み合わせ

**結論: 技術的に可能であり、実際に広く使われている。**

ConnectX NIC は標準的な PCIe デバイスであり、NVIDIA GPU に限定されない。AMD Instinct GPU と ConnectX NIC の組み合わせは実運用で確認されている。

- **Oracle Cloud Infrastructure (OCI)**: AMD Instinct MI355X インスタンスで ConnectX-7 (8 x 400 Gb/s Ethernet) を採用している
  - 出典: https://blogs.oracle.com/cloud-infrastructure/amd-instinct-mi355x-on-oci-performance-technical-details
- **Meta**: MI355X クラスタで ConnectX-7 NIC を採用する計画が報道されている
  - 出典: https://newsletter.semianalysis.com/p/mi300x-vs-h100-vs-h200-benchmark-part-1-training
- **AMD Instinct システムガイド**: ConnectX-7 NIC との NUMA ノードアライメントに関する公式ドキュメントが存在し、GPU と NIC を同一 PCIe root complex に配置する最適化手順が記載されている
  - 出典: https://instinct.docs.amd.com/projects/system-acceptance/en/latest/network/topology-mapping.html

### 1.2 GPUDirect RDMA の他社 GPU 対応

**結論: NVIDIA 独自の GPUDirect RDMA は NVIDIA GPU 専用だが、AMD は同等機能を別実装で提供している。**

- **GPUDirect RDMA** は NVIDIA の CUDA と密に結合した技術であり、Kepler 世代以降の NVIDIA GPU と ConnectX-3 以降の NIC で動作する
  - 出典: https://docs.nvidia.com/cuda/gpudirect-rdma/index.html
- **AMD の対応**: AMD は ROCnRDMA（旧名）および PeerDirect インターフェースを通じて、RDMA 対応 NIC が GPU メモリに直接アクセスする同等機能を ROCm プラットフォームで提供している。AMD カーネルドライバは PeerDirect インターフェースを通じて RDMA を公開し、NIC が GPU デバイスメモリへ直接読み書きできる
  - 出典: https://instinct.docs.amd.com/projects/gpu-cluster-networking/en/latest/how-to/gpu-enabled-mpi.html
  - 出典: https://github.com/rocmarchive/ROCnRDMA
- **重要**: GPUDirect RDMA という名前の技術自体は NVIDIA GPU 専用だが、「NIC から GPU メモリへの直接アクセス」という機能は AMD も ConnectX NIC 経由で実現できる。これは PeerDirect API が NIC ベンダ非依存に設計されているためである

### 1.3 BlueField DPU のオフロード機能の GPU 非依存性

**結論: BlueField DPU の基本的なインフラオフロード機能は GPU 非依存で動作する。**

BlueField DPU は以下のコンポーネントで構成される:
- Arm CPU コア（プログラマブル）
- ConnectX NIC（ネットワーク機能）
- セキュリティ、ストレージ、ネットワークアクセラレータ

BlueField DPU は「ホスト CPU からソフトウェア定義ネットワーク/ストレージをオフロード・分離する」ことを主目的としており、以下の機能は GPU の種類に依存しない:
- ネットワーク仮想化（OVS オフロード等）
- ストレージオフロード（NVMe-oF 等）
- セキュリティ機能（暗号化、ファイアウォール等）
- テレメトリと監視

ただし、BlueField-4 では GPU-to-GPU 通信の最適化（east-west トラフィック）が強化されており、この部分は NVIDIA GPU との組み合わせで最大の効果を発揮する。
- 出典: https://www.chiplog.io/p/analysis-of-nvidias-bluefield-4-dpu
- 出典: https://developer.nvidia.com/blog/offloading-and-isolating-data-center-workloads-with-bluefield-dpu/

### 1.4 Intel Gaudi との組み合わせ

**結論: Intel Gaudi は独自の RoCE v2 NIC をオンチップ搭載しており、ConnectX NIC の必要性が低い。**

- Gaudi は業界初の DL トレーニングプロセッサとして RoCE v2 エンジンをオンチップ統合している
- Gaudi 3 は 24 x 200 Gb/s の RoCE v2 インターフェースを内蔵し、スケールアップもスケールアウトも標準 Ethernet で対応
- Intel は「プロプライエタリなネットワーキングファブリックにロックインされない柔軟性」を Gaudi の利点として訴求している
  - 出典: https://docs.habana.ai/en/latest/Gaudi_Overview/Gaudi_Architecture.html

---

## 2. Spectrum-X (Ethernet スイッチ)

### 2.1 非 NVIDIA アクセラレータとの組み合わせ

**結論: 技術的には標準 Ethernet ベースだが、NVIDIA エコシステムとの密結合で最大効果を発揮する設計。**

Spectrum-X は以下の特徴を持つ:
- **標準 Ethernet ベース**: InfiniBand ではなく Ethernet プロトコルを使用しており、標準的な Ethernet 機器との互換性がある
- **SONiC 対応**: オープンな Ethernet スタックである SONiC をサポートしている
  - 出典: https://www.dell.com/en-us/blog/open-ethernet-for-ai-nvidia-spectrum-x-with-dell-sonic/
- **ただし密結合設計**: 「Spectrum-X の真の力は、ハードウェアとソフトウェアの密結合にあり、単一の部品を単独で使用しても最大効率を発揮しない」とされている
  - 出典: https://www.fibermall.com/blog/understanding-nvidias-spectrum-x-solution.htm

### 2.2 Adaptive Routing / Congestion Control の GPU 非依存性

**結論: ネットワーク機能自体は GPU 非依存だが、NCCL との連携で最適化されている。**

Spectrum-X の主要機能:
- **パケット単位の Adaptive Routing**: トラフィックを動的に分散し、輻輳を防止しつつ帯域幅を最大化する。これはスイッチ側の機能であり、原理的にはエンドポイントの GPU 種類に依存しない
- **テレメトリベースの Congestion Control**: ローカルおよびグローバルのネットワーク状況に基づくトラフィック分散
- **ロスレスネットワーキング**: 再送遅延を排除

ただし、性能最適化の観点では NVIDIA NCCL との垂直統合が重要な役割を果たしている。Spectrum-X Ethernet の性能が InfiniBand（Non-SHARP）に近いのは、「NCCL コレクティブライブラリとの垂直統合、および優れた congestion control と adaptive routing の活用」による。

- 出典: https://www.sdxcentral.com/analysis/inside-spectrum-x-nvidias-ethernet-networking-platform/
- 出典: https://newsletter.semianalysis.com/p/mi300x-vs-h100-vs-h200-benchmark-part-1-training

### 2.3 他社 GPU + Spectrum-X の事例

**結論: 公開された具体的な事例は限定的。**

- Spectrum-X は主に NVIDIA GPU（H100/H200/B200）クラスタ向けに販売・最適化されている
- Meta や Oracle が Spectrum-X を採用しているが、これは NVIDIA GPU クラスタ向けである
  - 出典: https://nvidianews.nvidia.com/news/nvidia-spectrum-x-ethernet-switches-speed-up-networks-for-meta-and-oracle
- AMD MI300X のリファレンスデザインでは、Spectrum-X ではなく Broadcom Ethernet を推奨している
- 技術的には標準 Ethernet であるため接続自体は可能だが、Spectrum-X + ConnectX SuperNIC の密結合最適化は NVIDIA NCCL に依存するため、AMD GPU 環境では最適化効果が限定される

### 2.4 Spectrum-XGS（2025年発表）

Spectrum-XGS は分散データセンターを接続して「ギガスケール AI スーパーファクトリー」を構築する技術として発表された。距離に応じた動的 congestion control、精密レイテンシ管理、エンドツーエンドテレメトリを特徴とする。
- 出典: https://nvidianews.nvidia.com/news/nvidia-introduces-spectrum-xgs-ethernet-to-connect-distributed-data-centers-into-giga-scale-ai-super-factories

---

## 3. Quantum (InfiniBand スイッチ)

### 3.1 InfiniBand の非 NVIDIA GPU 対応

**結論: InfiniBand は元来 GPU 非依存のオープン標準であり、歴史的に AMD/Intel CPU の HPC システムで広く使われてきた。**

- InfiniBand はもともと HPC（高性能計算）用のオープンなインターコネクト標準として開発された
- NVIDIA が Mellanox を買収（2020年）する以前から、InfiniBand は Intel/AMD CPU ベースの HPC クラスタで広く使われていた
- InfiniBand Host Channel Adapter (HCA) は標準 PCIe デバイスであり、任意の PCIe 対応プロセッサと接続可能
  - 出典: https://en.wikipedia.org/wiki/InfiniBand

### 3.2 SHARP (In-Network Computing) の GPU 依存性

**結論: SHARP は NVIDIA Quantum InfiniBand スイッチの機能であり、スイッチ側で動作する。ただし、NCCL との統合が前提。**

- SHARP は集団通信操作（allReduce、allGather 等）を CPU/GPU からネットワークスイッチにオフロードする技術
- SHARP 自体はスイッチのハードウェアで勾配集約を実行するため、理論的にはエンドポイントの GPU 種類に依存しない
- ただし、SHARP の利用は NCCL（NVIDIA Collective Communications Library）を通じて行われる。NCCL は NVIDIA GPU 専用であるため、**実質的に SHARP は NVIDIA GPU 環境でのみフル活用できる**
- AMD の RCCL は NCCL の fork だが、SHARP プラグインのサポートは確認されていない
  - 出典: https://developer.nvidia.com/blog/advancing-performance-with-nvidia-sharp-in-network-computing/
  - 出典: https://docs.nvidia.com/networking/display/sharpv300

### 3.3 AMD GPU + InfiniBand の構成事例

**結論: 歴史的には多数の事例が存在するが、最新の AMD AI GPU では Ethernet 志向が強い。**

- **歴史的事例**: AMD CPU（Opteron 等）+ InfiniBand の HPC クラスタは多数存在した
- **現在の AMD Instinct**: AMD は RoCEv2（RDMA over Converged Ethernet）を推奨する方向に移行しており、MI300X のリファレンスデザインでは Broadcom Thor-2 NIC + RoCE を採用
- **将来の AMD 戦略**: AMD は Ultra Ethernet Consortium (UEC) に注力しており、Broadcom Tomahawk 6 Ethernet スイッチ上で Ultra Accelerator Link (UALink) をトンネリングする方向に進んでいる
  - 出典: https://newsletter.semianalysis.com/p/amd-advancing-ai-mi350x-and-mi400-ualoe72-mi500-ual256

---

## 4. NVLink / NVSwitch

### 4.1 NVLink の NVIDIA GPU 専用性

**結論: NVLink は従来 NVIDIA GPU 専用のプロプライエタリインターコネクトだったが、2025年に NVLink Fusion で方針転換した。**

- NVLink は 2014年に発表され、2016年の Pascal 世代（Tesla P100）で初実装された
- NVHS（NVIDIA High Speed）というプロプライエタリな高速シグナリングを使用
- NVLink 1.0 から NVLink 6 まで、世代ごとに帯域幅がほぼ倍増してきた
- **従来、NVLink は完全に NVIDIA GPU 専用のクローズドな技術だった**
  - 出典: https://en.wikipedia.org/wiki/NVLink

### 4.2 NVLink Fusion（2025年5月 Computex で発表）

**結論: NVLink Fusion は他社チップとの接続を明確に想定した新戦略であり、NVIDIA のエコシステム戦略の大きな転換点。**

NVLink Fusion は、サードパーティの CPU やカスタム ASIC が NVLink インターコネクトを利用できるようにする技術である。

**主な特徴**:
- NVLink IP をカスタムチップ設計に統合するか、インターコネクトチプレットを介して接続する2つの方式を提供
- NVIDIA の Grace CPU や将来の Vera CPU と、非 NVIDIA 製アクセラレータを接続可能
- ただし、**完全なオープン化ではない**: 「NVLink を自社 ASIC で使いたければ、NVIDIA の CPU を使うか、逆に NVIDIA GPU を使う必要がある」という制約がある。つまり Intel CPU + AMD GPU のような組み合わせに NVLink は使えない
  - 出典: https://www.theregister.com/2025/05/19/nvidia_nvlink_fusion/

**パートナーエコシステム**:
- **MediaTek**: ASIC 設計サービスと高速インターコネクトの専門知識を活用して NVLink Fusion 対応チップを開発
- **Marvell**: ハイパースケーラ向けカスタムシリコンに NVLink Fusion を統合し、モデルトレーニングとエージェンティック AI 推論のスケールアップソリューションを提供
- **Alchip, Astera Labs, Synopsys, Cadence**: シリコン設計サービスとソリューションを提供
- **Fujitsu, Qualcomm**: NVLink Fusion を CPU に統合する最初のパートナー
  - 出典: https://nvidianews.nvidia.com/news/nvidia-nvlink-fusion-semi-custom-ai-infrastructure-partner-ecosystem
  - 出典: https://embeddedcomputing.com/technology/ai-machine-learning/ai-logic-devices-worload-acceleration/synopsys-mediatek-marvell-and-others-embrace-nvidias-nvlink-fusion-for-ai-workloads

**AWS Trainium4 との統合（2025年12月発表）**:
- AWS は Trainium4 を NVLink 6 および NVIDIA MGX ラックアーキテクチャと統合する設計を発表
- Trainium4 アクセラレータ、Graviton CPU、EFA ネットワーキング技術が NVIDIA の MGX ラック上でシームレスに通信可能
- これは NVIDIA と AWS の「NVLink Fusion に関する複数世代にわたるコラボレーション」の最初の成果とされている
  - 出典: https://www.theregister.com/2025/12/02/amazon_nvidia_trainium/
  - 出典: https://developer.nvidia.com/blog/aws-integrates-ai-infrastructure-with-nvidia-nvlink-fusion-for-trainium4-deployment/

### 4.3 UALink との競合

- 2025年4月、UALink Consortium が UALink 200G 1.0 Specification をリリース
- UALink は NVLink に対抗するオープン標準インターコネクトとして設計されている
- AMD、Intel、Google、Meta、Microsoft 等が参加
- Arm も UALink と NVLink Fusion の両方をサポートする姿勢を表明
  - 出典: https://www.networkworld.com/article/4091468/arm-jumps-on-the-nvidia-nvlink-fusion-bandwagon-at-sc25.html

---

## 5. NCCL / Magnum IO

### 5.1 NCCL の GPU 依存性

**結論: NCCL は NVIDIA GPU 専用であり、CUDA ランタイムに依存する。**

- NCCL（NVIDIA Collective Communications Library）は NVIDIA GPU 向けに性能最適化されたマルチ GPU / マルチノード通信プリミティブを実装している
- CUDA ランタイムに直接依存しており、AMD GPU や他社アクセラレータでは動作しない
  - 出典: https://developer.nvidia.com/nccl

### 5.2 RCCL と NVIDIA ネットワーク技術の連携

**結論: RCCL は NCCL の fork であり API 互換だが、NVIDIA 固有の最適化（SHARP 等）は利用できない。一方、ConnectX NIC 経由の RDMA は利用可能。**

- **RCCL**（ROCm Communication Collectives Library）は NCCL を fork した AMD GPU 向けの集団通信ライブラリ
- NCCL Net Plugin API と互換性のあるネットワークプラグインアーキテクチャを持つ
- ConnectX NIC の RDMA 機能は RCCL からも利用可能（PeerDirect 経由）
- ただし、NCCL 固有の最適化（NVLink SHARP、GPU-Initiated Networking 等）は RCCL では利用できない
  - 出典: https://rocm.docs.amd.com/projects/rccl/en/develop/how-to/using-nccl.html

### 5.3 UCCL（新しい統合アプローチ）

- UCCL は NCCL/RCCL のネットワークプラグインとして実装された新しい通信レイヤ
- NVIDIA RDMA NIC（RC/UC）、Broadcom RDMA NIC（RC）、AWS EFA NIC（UD/SRD）、AWS ENA 非 RDMA NIC（AF_XDP）をサポート
- ベンダを跨いだ NIC サポートを提供する統合的なアプローチ
  - 出典: https://arxiv.org/html/2504.17307v2

### 5.4 異種 GPU 混在トレーニングの最新研究（2025-2026年）

異種 GPU クラスタ（AMD + NVIDIA 混在）でのトレーニングに関する研究が進展している:

- **Joint Training on AMD and NVIDIA GPUs** (2026年2月): Device-Direct Communication アプローチにより、GPUDirect RDMA と CPU オフロード P2P 通信スキームを統合し、異なるベンダの GPU 間で直接データ転送を実現。NVIDIA 同種システムのスループットの最大 98% を達成
  - 出典: https://arxiv.org/abs/2602.18007
- **HetCCL** (2025年1月): ベンダ固有のバックエンド（NCCL、RCCL）を統合し、ドライバ変更なしで異種 GPU 間の RDMA ベース通信を可能にする集団通信ライブラリ。同種環境では NCCL/RCCL と同等性能を発揮しつつ、異種環境でもスケーリング可能
  - 出典: https://arxiv.org/abs/2601.22585

---

## 6. DOCA SDK

### 6.1 DOCA の GPU 非依存性

**結論: DOCA の基本機能は GPU 非依存だが、NVIDIA GPU との統合機能も多数含まれる。**

- DOCA（Data Center Infrastructure On A Chip Architecture）は BlueField DPU/SuperNIC 向けのプログラミング SDK
- 業界標準のオープン API とソフトウェアフレームワークを提供
- 複数の OS やディストリビューションをサポートし、ドライバ、ライブラリ、ツール、ドキュメント、サンプルアプリケーションを含む
  - 出典: https://developer.nvidia.com/networking/doca

**GPU 非依存の機能**:
- ネットワークオフロード（OVS、IPsec、正規表現マッチング等）
- ストレージオフロード（NVMe-oF、virtio-blk 等）
- セキュリティ機能
- テレメトリと監視
- これらの機能は BlueField DPU の Arm CPU とアクセラレータで処理され、ホスト側の GPU 種類に依存しない

**NVIDIA GPU 依存の機能**:
- DOCA GPUNetIO: GPU-to-NIC の直接通信（CUDA 依存）
- GPU + DPU コンバージドアクセラレータ向けの機能
- CUDA との連携機能
  - 出典: https://docs.nvidia.com/doca/sdk/index.html
  - 出典: https://developer.nvidia.com/blog/developing-applications-with-bluefield-dpu-and-doca-libraries/

### 6.2 他社 GPU 環境での BlueField + DOCA

**結論: インフラオフロード機能は他社 GPU 環境でも使用可能だが、GPU-NIC 連携の高度な機能は NVIDIA GPU 専用。**

- BlueField DPU をサーバに搭載した場合、ネットワーク仮想化やストレージオフロードは GPU の種類に関係なく動作する
- ただし、DOCA のマーケティングと文書は NVIDIA エコシステム（CUDA + BlueField）を前提としており、他社 GPU 環境での検証情報は限定的

---

## 7. 実現可能性の総合評価

### 7.1 組み合わせ別の評価マトリクス

| NVIDIA 技術 | 他社アクセラレータ | 技術的可否 | 性能制約 | 実事例 | ビジネス制約 |
|---|---|---|---|---|---|
| ConnectX NIC | AMD Instinct | **可能** | GPUDirect RDMA 相当は PeerDirect で対応 | OCI MI355X、Meta MI355X | なし（標準 PCIe デバイス） |
| ConnectX NIC | Intel Gaudi | **不要**（オンチップ NIC） | Gaudi は独自 NIC を搭載 | 該当なし | N/A |
| ConnectX NIC | AWS Trainium | **限定的** | EFA/Nitro が標準 | AWS は独自 NIC を使用 | AWS エコシステム制約 |
| ConnectX NIC | Google TPU | **不可** | 完全独自インフラ | なし | Google の閉鎖的アーキテクチャ |
| BlueField DPU（インフラ） | 任意 | **可能** | なし | 限定的 | なし |
| BlueField DPU（GPU連携） | 非 NVIDIA | **不可** | CUDA 依存 | なし | NVIDIA エコシステムに依存 |
| Spectrum-X スイッチ | AMD Instinct | **接続可能** | NCCL 最適化が欠如し性能低下 | 公開事例なし | NVIDIA は NVIDIA GPU 向けに販売 |
| Spectrum-X スイッチ | Intel Gaudi | **接続可能** | Gaudi は独自 NIC でありSuperNIC 非使用 | 公開事例なし | アーキテクチャ不一致 |
| Quantum InfiniBand | AMD Instinct | **可能** | SHARP 利用不可で性能低下 | 歴史的 HPC 事例あり | AMD は Ethernet 志向に移行 |
| Quantum InfiniBand | Intel Gaudi | **限定的** | Gaudi は RoCE 専用設計 | なし | アーキテクチャ不一致 |
| NVLink（従来） | 非 NVIDIA | **不可** | NVIDIA GPU 専用 | なし | プロプライエタリ |
| NVLink Fusion | カスタム ASIC | **可能**（条件付き） | NVIDIA CPU/GPU との組み合わせ必須 | AWS Trainium4（計画中） | NVIDIA エコシステム内に限定 |
| NCCL | 非 NVIDIA GPU | **不可** | CUDA 専用 | なし | NVIDIA エコシステム制約 |
| RCCL + ConnectX | AMD GPU | **可能** | SHARP 等の高度な最適化は不可 | あり | AMD が RCCL を独自開発 |
| DOCA（インフラ） | 任意 | **可能** | なし | 限定的 | なし |
| DOCA（GPUNetIO） | 非 NVIDIA | **不可** | CUDA 依存 | なし | NVIDIA エコシステム制約 |

### 7.2 性能影響の分析

ConnectX NIC を AMD GPU で使用した場合の性能について、Semi Analysis のベンチマークレポートが重要な知見を提供している:

- **MI300X + RoCEv2**: 実用的なメッセージサイズ（16MiB - 256MiB）で、InfiniBand Non-SHARP と比較して約半分の速度
- **NVIDIA Spectrum-X Ethernet**: InfiniBand Non-SHARP に近い性能を達成（NCCL との垂直統合が要因）
- **NVIDIA InfiniBand + SHARP**: 最高性能だが、NCCL（= NVIDIA GPU）が必要
  - 出典: https://newsletter.semianalysis.com/p/mi300x-vs-h100-vs-h200-benchmark-part-1-training

つまり、NVIDIA のネットワーク技術を他社 GPU と組み合わせた場合、ハードウェアレベルでは動作するが、ソフトウェアスタック（NCCL）との統合がないため性能上のギャップが生じる。

### 7.3 NVIDIA のビジネス戦略分析

NVIDIA のエコシステム戦略は以下の層で構成されている:

1. **CUDA ソフトウェアエコシステム**: 400万人以上の開発者、20年近い蓄積。最も強力なロックイン要因
2. **ネットワーキング事業**: Q4 で 110億ドルに達し、「世界最大のネットワーキング事業」。GPU 性能リーダーシップよりも耐久性のあるロックイン機構
3. **垂直統合**: GPU + NIC + スイッチ + 通信ライブラリ + SDK を一貫して提供することで、個々のコンポーネントの価値以上のシステム価値を創出

**NVLink Fusion の戦略的意義**:
- 表面的にはオープン化だが、実質的には「NVIDIA インフラを使いたければ NVIDIA CPU/GPU が必要」という条件付き
- AWS Trainium4 の事例は、競合の AI チップですら NVIDIA のインフラエコシステム（NVLink、MGX）に取り込む戦略を示している
- カスタム ASIC 市場を NVIDIA エコシステムに取り込むことで、GPU 単体の競争力低下リスクをヘッジする狙い

**競合の対抗戦略**:
- **AMD**: Ultra Ethernet Consortium (UEC) と UALink で NVIDIA 非依存のオープンなインターコネクトを推進。自社 NIC（Pollara、次世代 Vulcano 800GbE）の開発を進めているが、800GbE 対応は NVIDIA ConnectX-8 より約1年遅れ（2026年下期）
- **Google/Meta**: PyTorch の TPU サポート強化など、CUDA モートへの直接的な対抗
- **AWS**: EFA + Nitro で独自ネットワークスタックを構築しつつ、Trainium4 では NVLink Fusion を採用するハイブリッド戦略

  - 出典: https://www.nasdaq.com/articles/nvidias-broadening-moat-securing-ai-ecosystem
  - 出典: https://fourweekmba.com/nvidia-chip-wars-networking-moat-model-makers/
  - 出典: https://techcrunch.com/2025/12/02/amazon-releases-an-impressive-new-ai-chip-and-teases-a-nvidia-friendly-roadmap/

---

## 8. 各社 AI 半導体のネットワーク戦略まとめ

### Google TPU
- **独自閉鎖型**: ICI（Inter-Chip Interface）による独自スケールアップ、OCS（Optical Circuit Switch）と 3D トーラスネットワークによるスケールアウト
- **NVIDIA 技術との互換性**: 基本的になし。完全に独自のインフラスタック
- **Google Cloud 内では NVIDIA GPU も提供**: H100/Blackwell インスタンスも利用可能だが、TPU とは別インフラ
  - 出典: https://introl.com/blog/google-tpu-vs-nvidia-gpu-infrastructure-decision-framework-2025

### AMD Instinct
- **オープンエコシステム志向**: 標準 Ethernet（RoCEv2）を採用し、ベンダロックインを回避
- **NVIDIA NIC との互換性**: ConnectX NIC は AMD GPU と組み合わせて使用可能（OCI、Meta 等で採用）
- **独自 NIC 開発**: Pollara 400G NIC（Ultra Ethernet 対応）、次世代 Vulcano 800GbE NIC を開発中
- **UALink**: NVLink に対抗するオープン標準インターコネクトを推進
  - 出典: https://www.amd.com/en/products/accelerators/instinct/mi300.html

### Intel Gaudi
- **標準 Ethernet 統合型**: RoCE v2 NIC をオンチップ統合し、外部 NIC への依存を排除
- **NVIDIA 技術との互換性**: 低い。独自の統合型アーキテクチャにより、NVIDIA NIC/スイッチの必要性が限定的
- **注**: Intel は Gaudi 事業の将来について不透明な状況が続いている
  - 出典: https://docs.habana.ai/en/latest/Gaudi_Overview/Gaudi_Architecture.html

### AWS Trainium
- **独自 + NVIDIA ハイブリッド**: Trainium2 は EFA + Nitro の独自ネットワークスタックを使用
- **Trainium4 で NVLink Fusion 採用**: NVIDIA の MGX ラックアーキテクチャと統合する方針を発表
- **EFA の評価**: NVIDIA GPU 上では Spectrum-X/InfiniBand/ConnectX に比べて性能面で劣るが、Trainium 上ではスタック全体を制御できるため良好
  - 出典: https://developer.nvidia.com/blog/aws-integrates-ai-infrastructure-with-nvidia-nvlink-fusion-for-trainium4-deployment/

---

## 9. 結論

### NVIDIA GPU 非依存で使える技術
1. **ConnectX NIC**（標準 NIC 機能、RDMA）: AMD GPU と組み合わせて実運用されている
2. **BlueField DPU**（インフラオフロード機能）: ネットワーク仮想化、ストレージ、セキュリティは GPU 非依存
3. **Quantum InfiniBand スイッチ**（基本接続）: 標準 InfiniBand として他社 GPU と接続可能
4. **Spectrum-X スイッチ**（基本 Ethernet 接続）: 標準 Ethernet として接続可能
5. **DOCA SDK**（インフラ機能）: BlueField のインフラオフロード開発は GPU 非依存

### NVIDIA GPU が必要な技術（ソフトウェア依存）
1. **GPUDirect RDMA**: CUDA 依存（AMD は PeerDirect で同等機能を独自実装）
2. **NCCL + Spectrum-X/InfiniBand の最適化連携**: NCCL は CUDA 専用
3. **SHARP In-Network Computing**: NCCL 経由でのみ利用可能
4. **DOCA GPUNetIO**: CUDA 依存
5. **NVLink（従来）**: NVIDIA GPU 専用ハードウェア

### 新たな展開（NVLink Fusion）
- NVLink Fusion により、サードパーティの ASIC/CPU が NVLink エコシステムに参加可能になった
- ただし、NVIDIA の CPU または GPU との組み合わせが必須条件であり、完全なオープン化ではない
- AWS Trainium4 が最初の大規模な非 NVIDIA アクセラレータ + NVLink の事例となる見込み

### 全体的な傾向
NVIDIA のデータセンター技術は、**ハードウェアレベルでは多くが標準規格（PCIe、Ethernet、InfiniBand）に準拠しており他社製品と接続可能だが、ソフトウェアレイヤ（NCCL、CUDA、GPUDirect）での最適化が NVIDIA GPU に限定されている**ことで、実質的なエコシステムロックインを形成している。物理的な接続は可能でも、性能の最大化には NVIDIA GPU が必要という構造が、NVIDIA の競争優位の核心である。

---

## 10. AllReduce ベンチマークデータ（深掘り調査）

### 10.1 構成別 AllReduce 性能（Semi Analysis 実測データ）

マルチノード AllReduce、実用的なメッセージサイズ（16MiB〜256MiB）での比較:

| 構成 | 相対性能 | 備考 |
|------|---------|------|
| H100 + ConnectX-7 + InfiniBand (Non-SHARP) | **100%**（基準） | |
| H100 + ConnectX-7 + InfiniBand + **SHARP** | **約 130〜150%** | In-Network Reduction で基準を上回る |
| H100 + ConnectX-7 + **Spectrum-X** (Ethernet) | **約 90〜95%** | NCCL 垂直統合により IB Non-SHARP に近い |
| MI300X + ConnectX-7 + **RoCEv2** | **約 50%** | IB Non-SHARP の約半分 |
| MI300X + Broadcom Thor-2 + RoCEv2 | **約 50%** | ConnectX-7 とほぼ同等 |

**重要な知見**:
- MI300X の AllReduce 性能が IB Non-SHARP の約半分にとどまる主因は **RCCL のネットワーク最適化不足**
- Spectrum-X が IB Non-SHARP に近い性能を達成できるのは **NCCL との垂直統合**（Adaptive Routing + Congestion Control の協調）が要因
- ハードウェア（NIC）が同じ ConnectX-7 でも、ソフトウェアスタック（NCCL vs RCCL）の差が性能を決定

### 10.2 異種 GPU 混在トレーニングの実測データ

#### Joint Training on AMD and NVIDIA GPUs（arXiv:2602.18007、2026年2月）

| アプローチ | スループット（NVIDIA 同種比） | 方式 |
|-----------|--------------------------|------|
| CPU-Forwarding Communication | **約 90〜95%** | CPU 経由の P2P 通信 |
| Device-Direct Communication | **最大 98%** | GPUDirect RDMA + CPU オフロード統合 |

- Device-Direct アプローチにより、**NVIDIA 同種システムのスループットの最大 98% を達成**
- GPUDirect RDMA と CPU オフロード P2P 通信スキームを統合
- 異なるベンダの GPU 間で RDMA を介した直接データ転送を実現

#### HetCCL（arXiv:2601.22585、2025年1月）

- ベンダ固有のバックエンド（NCCL + RCCL）を統合する集団通信ライブラリ
- **ドライバ変更なし**で異種 GPU 間の RDMA ベース通信を実現
- 同種環境では NCCL/RCCL と**同等性能**を維持しつつ、異種環境でもスケーリング可能
- AMD GPU と NVIDIA GPU が混在するクラスタでの LLM トレーニングを実証

---

## 11. UALink 1.0 仕様の技術分析（深掘り調査）

### 11.1 UALink 200G 1.0 の技術仕様

| 項目 | スペック |
|------|---------|
| データレート | **200 GT/s / レーン** |
| シグナリングレート | 212.5 GT/s（FEC エンコーディング込み） |
| レーン構成 | x1, x2, x4 |
| 4レーンリンク帯域幅 | **200 GB/s**（双方向） |
| プロトコル基盤 | **Ethernet Layer 1** を基盤として採用 |
| FEC | Ethernet Layer 1 の Forward Error Correction を使用 |
| 策定日 | 2025年4月リリース |

### 11.2 UALink vs NVLink 6 の定量比較

| 項目 | UALink 1.0 (x4) | NVLink 6 (Rubin) |
|------|-----------------|------------------|
| リンク帯域幅 | 200 GB/s | **3,600 GB/s** |
| 帯域幅比 | 1x | **18x** |
| 標準規格 | オープンコンソーシアム | プロプライエタリ |
| エコシステム | AMD, Intel, Google, Meta 等 | NVIDIA + NVLink Fusion パートナー |

- **NVLink 6 は UALink 1.0 の約 18 倍の帯域幅**
- UALink 1.0 は NVLink に対して帯域幅で大幅に劣るが、オープン標準であることが最大の差別化要因
- UALink 2.0 以降で帯域幅の向上が予定されている

### 11.3 UALink over Ethernet (UALoE)

- AMD が「Infinity Fabric Over Ethernet」を「**UALink Protocol over Ethernet**」にリブランド
- **MI400 シリーズ**の Scale-up ネットワークに UALoE を採用予定
- Broadcom Tomahawk 6 Ethernet スイッチ上で UALink をトンネリング
- Scale-up と Scale-out を統一的な Ethernet インフラで実現する戦略

### 11.4 実装タイムライン

| 製品 | 帯域幅 | 出荷時期 |
|------|--------|---------|
| AMD Pollara 400G NIC | 400 Gbps | 出荷中（UEC 対応） |
| AMD Vulcano 800GbE NIC | 800 Gbps | **2026年下期**（NVIDIA ConnectX-8 より約1年遅れ） |
| AMD MI400 + UALoE | — | 2026年以降 |
| UALink 2.0 仕様 | TBD | 策定中 |

---

## 12. NCCL 代替ライブラリ群の体系的比較（深掘り調査）

### 12.1 ライブラリ一覧と特徴

| ライブラリ | 開発元 | 対応 GPU | 特徴 |
|-----------|--------|---------|------|
| **NCCL** | NVIDIA | NVIDIA GPU 専用 | 業界標準、CUDA 依存 |
| **RCCL** | AMD | AMD GPU 専用 | NCCL fork、API 互換 |
| **MSCCL** | Microsoft | NVIDIA (Azure) | NCCL 上にカスタムアルゴリズム実行 |
| **MSCCL++** | Microsoft | NVIDIA / AMD | NCCL/RCCL 比 最大 3.8倍高速 |
| **UCCL** | UC Berkeley / UC Davis | マルチベンダ | NCCL/RCCL ドロップイン代替 |
| **HetCCL** | 研究 | **AMD + NVIDIA 混在** | ドライバ変更なしで異種 GPU 間通信 |
| **OneCCL** | Intel | Intel GPU / CPU | Intel oneAPI エコシステム |

### 12.2 MSCCL / MSCCL++

- **MSCCL**: NCCL 上に構築された Microsoft Azure 向けカスタム集団通信プラットフォーム
- **MSCCL++**: 低レベル通信プリミティブを提供する C++ ライブラリ
  - 小メッセージ: NCCL 比 **最大 2.8倍**、RCCL 比 **最大 3.8倍**
  - 大メッセージ: NCCL 比 **最大 2.4倍**、RCCL 比 **最大 2.0倍**
  - Azure 環境に最適化

### 12.3 UCCL

- **UCCL-collective**: NCCL/RCCL の**ドロップイン代替**
- 対応 NIC:
  - NVIDIA RDMA NIC (RC/UC)
  - Broadcom RDMA NIC (RC)
  - AWS EFA NIC (UD/SRD)
  - AWS ENA 非 RDMA NIC (AF_XDP)
- **ベンダ非依存のマルチ NIC サポート**が最大の特徴
- UC Berkeley Sky Computing Lab と UC Davis ArtSy lab で開発中

### 12.4 HetCCL

- **異種 GPU クラスタ向け**集団通信ライブラリ
- NCCL（NVIDIA GPU 用）と RCCL（AMD GPU 用）のバックエンドを統合
- **ドライバ変更なし**で RDMA ベースの異種 GPU 間通信を実現
- 同種環境では NCCL/RCCL と同等性能を維持

---

## 13. Spectrum-X の非 NVIDIA GPU 環境での動作（深掘り調査）

### 13.1 技術的検証

- Spectrum-X の Adaptive Routing は **NCCL の集団通信パターンと連携して動作**する設計
- NCCL が「最大性能を達成する」ための前提条件として位置付けられている
- **非 NVIDIA GPU 環境での Adaptive Routing の動作は技術的に検証されていない**（公開情報なし）

### 13.2 SONiC 対応

- Spectrum-X スイッチは **SONiC**（Software for Open Networking in the Cloud）をサポート
- SONiC 自体はオープンな NOS であり、理論的には非 NVIDIA NIC とも連携可能
- ただし、Spectrum-X の AI 最適化機能（Adaptive Routing、RoCE 拡張等）は **ConnectX SuperNIC との協調動作**を前提としている

### 13.3 UFM（Unified Fabric Manager）

- InfiniBand / Ethernet ファブリック管理ソフトウェア
- ファブリックの監視、設定、最適化を提供
- 非 NVIDIA GPU 環境での UFM の対応状況は限定的な公開情報

---

## 14. NVLink Fusion の技術的詳細（深掘り調査）

### 14.1 物理層とチップレット接続

- **NVLink-C2C (Chip-to-Chip)**: 2チップ間のメモリコヒーレンシ接続を提供
- NVLink は **物理層 (L1)** と **制御層 (L2)** で構成
- **2つの統合方式**:
  1. **NVLink-C2C IP ブロック統合**: ライセンシーのチップ設計に NVLink-C2C IP を直接統合
  2. **NVLink チップレット**: 外付けチップレットとしてインターフェースを追加

### 14.2 メモリコヒーレンシ

- NVLink-C2C は**メモリコヒーレンシ**をサポートし、接続された2チップ間で統一されたメモリ空間を提供
- 他社チップとの **coherency domain** の管理方式の詳細は公開されていない
- AWS Trainium4 + NVLink Fusion でのメモリモデル（統一アドレス空間、キャッシュ一貫性）の詳細は今後の発表待ち

### 14.3 NVLink C2C vs NVLink Fusion

- **NVLink C2C**: NVIDIA チップ間（Grace CPU ↔ Hopper GPU 等）の直接接続技術
- **NVLink Fusion**: NVLink C2C 技術を**サードパーティに開放**するブランド名
- 技術的には同一の NVLink-C2C プロトコルを使用
- Fusion は「非 NVIDIA チップが NVLink エコシステムに参加するためのプログラム」

---

## 15. SHARP 世代別機能（深掘り調査）

### 15.1 SHARP バージョンと対応機能

| 機能 | SHARP v2 | SHARP v3 |
|------|---------|---------|
| AllReduce | 対応 | 対応 |
| AllGather | — | **対応** |
| ReduceScatter | — | **対応** |
| NVLink SHARP | — | **対応**（NVLink + IB SHARP 統合） |
| NCCL バージョン | NCCL 2.x | **NCCL 2.27+** |

- NCCL 2.27 で NVLink SHARP と InfiniBand SHARP の両方のサポートが追加
- AllGather と ReduceScatter への SHARP サポート拡張により、LLM トレーニングの主要な通信パターンをカバー

### 15.2 SHARP の NCCL 依存

- SHARP は **NCCL プラグインとしてのみ**公式サポート
- **UCC (Unified Collective Communication)** フレームワーク経由での利用可能性は限定的
- RCCL（AMD）からの SHARP 利用は**未サポート** — API 差異ではなく、SHARP プラグインが NCCL 向けにのみ提供されていることが技術的理由

### 深掘り調査のソース

- [MI300X vs H100 vs H200 Benchmark - SemiAnalysis](https://newsletter.semianalysis.com/p/mi300x-vs-h100-vs-h200-benchmark-part-1-training)
- [Joint Training on AMD and NVIDIA GPUs - arXiv:2602.18007](https://arxiv.org/abs/2602.18007)
- [HetCCL - arXiv:2601.22585](https://arxiv.org/abs/2601.22585)
- [UALink 200G 1.0 Specification](https://www.ualink.org/specifications)
- [AMD Advancing AI: MI350X, MI400, UALoE - SemiAnalysis](https://newsletter.semianalysis.com/p/amd-advancing-ai-mi350x-and-mi400-ualoe72-mi500-ual256)
- [MSCCL++ Performance](https://www.microsoft.com/en-us/research/publication/msccl-a-general-purpose-gpu-communication-library/)
- [UCCL - arXiv:2504.17307](https://arxiv.org/html/2504.17307v2)
- [NCCL 2.27 SHARP Support](https://docs.nvidia.com/networking/display/sharpv300)
- [NVLink-C2C Architecture](https://developer.nvidia.com/blog/nvidia-nvlink-c2c/)
- [Spectrum-X Adaptive Routing + NCCL](https://www.sdxcentral.com/analysis/inside-spectrum-x-nvidias-ethernet-networking-platform/)
- [RCCL Net Plugin Architecture](https://rocm.docs.amd.com/projects/rccl/en/develop/how-to/using-nccl.html)
