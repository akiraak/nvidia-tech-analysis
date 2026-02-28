# MoE アーキテクチャ・耐障害性・実デプロイメント事例

調査日: 2026-02-28

---

## 1. MoE（Mixture of Experts）アーキテクチャとネットワーク要件

### 1.1 MoE の基本概念

#### Dense モデル vs MoE モデルの違い

| 特性 | Dense モデル | MoE（Sparse）モデル |
|------|------------|-------------------|
| パラメータ活性化 | 全パラメータを全入力に使用 | 入力ごとに一部の Expert のみ活性化 |
| 計算コスト | パラメータ数に比例して増大 | 総パラメータ数より大幅に少ない |
| メモリ要件 | 全パラメータをロード | 全パラメータのロードは必要だが計算は一部 |
| スケーリング | 計算量とパラメータ数が線形関係 | パラメータ数を増やしても計算量を一定に保てる |
| 通信パターン | AllReduce が主体 | All-to-All 通信が必要 |

Dense モデルでは全パラメータが全入力に対して使用されるのに対し、MoE モデルでは条件付き計算（Conditional Computation）の概念を導入し、入力に応じてモデルの一部のみを実行する。これにより、モデル容量（知識量）を増やしながら、推論・学習時の計算コストを実質的に一定に保つことが可能になる。

#### Expert の選択メカニズム（Router / Gating Network）

MoE レイヤーの構造:
1. **Expert 群**: 各レイヤーに複数の Feed-Forward Network（FFN）サブレイヤーが並列に配置される
2. **Router（Gating Network）**: 入力トークンに対して確率分布を生成し、最も関連性の高い Expert のサブセットを選択する
3. **Top-K 選択**: 一般的に K=2（Mixtral）または K=8（DeepSeek-V3）の Expert が選択される
4. **出力の統合**: 選択された Expert の出力を加重和で統合する

Router の動作原理:
- 入力トークン x に対して、Gating Network G(x) が各 Expert の重みを計算
- Softmax を適用して確率分布を生成
- Top-K の Expert を選択し、残りをゼロにする（Sparse Gating）
- 選択された Expert の出力を重み付きで加算

#### なぜ MoE が注目されているか

1. **計算効率**: 同じパラメータ数の Dense モデルと比較して、はるかに少ない計算量で同等以上の性能を実現
   - Mixtral 8x7B: 総パラメータ 46.7B、活性化パラメータ 13B → LLaMA 2 70B と同等性能
   - DeepSeek-V3: 総パラメータ 671B、活性化パラメータ 37B → GPT-4 クラスの性能
2. **推論コスト**: トークンあたりのコストが大幅に削減される
3. **スケーラビリティ**: Expert 数を増やすことでモデル容量を拡張可能
4. **フロンティアモデルの標準**: GPT-4、Gemini、DeepSeek-V3/R1、Mixtral など主要モデルが MoE を採用

### 1.2 MoE のネットワーク要件

#### All-to-All 通信パターン

**なぜ AllReduce ではなく All-to-All が必要か:**

- **Data Parallelism（Dense モデル）**: 各 GPU が同じモデルの複製を持ち、勾配を AllReduce で集約する。全 GPU が同じ計算を行うため、通信パターンは予測可能
- **Expert Parallelism（MoE モデル）**: Expert が異なる GPU に分散配置される。各トークンは Router によって動的に異なる Expert に送られるため、**All-to-All 通信**が必要になる
  - 各 GPU は入力トークンを適切な Expert が存在する GPU に送信する必要がある
  - どの GPU にどのトークンを送るかは入力依存で動的に変化する
  - これにより通信量が不均一かつ予測困難になる

All-to-All 通信の課題:
- **動的な通信量**: Expert の選択は入力依存であり、イテレーションごとに通信パターンが変化
- **ネットワーク輻輳リスク**: 特定の Expert に偏った場合、特定のリンクに通信が集中
- **レイテンシ感度**: Expert 間の通信がクリティカルパスに存在

#### Expert Parallelism の通信量

Expert Parallelism（EP）における通信コスト:
- **Dispatch フェーズ**: 各 GPU からトークンを対応する Expert の GPU に送信
- **Combine フェーズ**: Expert の計算結果を元の GPU に返送
- 通信量は `tokens × hidden_dim × top_k / EP_degree` に比例
- EP 度数を上げると Expert あたりの計算量は減るが、通信量は増加

Capacity Factor:
- Expert の処理可能なトークン数の上限を決める係数
- 高い Capacity Factor → 品質向上、通信コスト増大
- 低い Capacity Factor → 通信削減、一部トークンのドロップリスク

#### ConnectX SuperNIC の All-to-All 最適化

**ConnectX-8 SuperNIC（2025年量産開始）:**
- PCIe Gen6 スイッチングと 800 Gb/s ネットワーキングを単一デバイスに統合
- GPU あたり最大 400 Gb/s のネットワーク帯域幅（前世代比2倍）
- NCCL All-to-All 性能で最大 2倍の高速化を実現
- HGX B300 および GB300 NVL72 システムに統合

**Blackwell での具体的な最適化:**
- NVIDIA RTX PRO Server + ConnectX-8 SuperNIC: NCCL All-to-All 性能 2倍向上
- 専用 PCIe スイッチを ConnectX-8 SuperNIC に置き換え、IO ボトルネックを排除
- GPU-NIC-ストレージ間のデータ移動を高速化

**ConnectX-9 SuperNIC（次世代）:**
- Vera Rubin プラットフォーム向け
- さらなる帯域幅向上が計画されている

### 1.3 主要な MoE モデルの事例

#### Mixtral 8x7B / 8x22B（Mistral AI）

| 仕様 | Mixtral 8x7B | Mixtral 8x22B |
|------|-------------|--------------|
| 総パラメータ数 | 46.7B | 141B |
| 活性化パラメータ数 | 13B | 39B |
| Expert 数 | 8 | 8 |
| Top-K | 2 | 2 |
| コンテキスト長 | 32K | 64K |
| 比較対象 Dense モデル | LLaMA 2 70B と同等 | GPT-3.5 175B を上回る |

- 各レイヤーで Router が 8 つの Expert から 2 つを選択
- 選択された Expert の出力を加算的に統合
- 発表: 2024年1月（論文: arXiv:2401.04088）

#### GPT-4（OpenAI、推定 MoE 構造）

- **公式未確認**: OpenAI は技術仕様を公開していない
- **推定アーキテクチャ（SemiAnalysis 等の分析）**:
  - 総パラメータ数: 約 1.8 兆（1.8T）
  - レイヤー数: 約 120
  - Expert 数: 16（推定）、各 Expert 約 111B パラメータ
  - 別の推定: 8 Expert、各 220B パラメータ
- **業界コンセンサス**: MoE アーキテクチャを採用していることは広く信じられている
- スパースモデルにより、全パラメータを活性化せず推論コストを抑制

#### DeepSeek V3 / R1

**DeepSeek-V3:**

| 仕様 | 値 |
|------|-----|
| 総パラメータ数 | 671B |
| 活性化パラメータ数 | 37B |
| Routed Expert 数 | 256 |
| Shared Expert 数 | 1 |
| Top-K | 8 |
| コンテキスト長 | 128K |
| 学習データ | 14.8 兆トークン |
| 学習コスト | 2.788M H800 GPU 時間（約 $5.576M） |

技術的革新:
- **Auxiliary-Loss-Free Load Balancing**: 従来の補助損失関数を排除し、ゲーティング値にバイアス項を導入。Expert の過負荷/低負荷時にのみ手動調整
- **Multi-head Latent Attention（MLA）**: DeepSeek-V2 で検証された効率的な Attention 機構
- **Multi-Token Prediction（MTP）**: 学習目標として複数トークン予測を設定
- V2 比で Routed Expert 数を 160 → 256 に 60% 増加
- 全 Expert 活性化レイヤーを 1 → 3 に増加

**DeepSeek-R1:**
- V3 と同一のベースアーキテクチャ（671B / 37B 活性化）を使用
- 強化学習による推論能力の強化が主な差分
- MoE アーキテクチャとしての構造は V3 と共通

#### Gemini（Google DeepMind、MoE ベース）

- **Gemini 1.5（2024年）**: MoE アーキテクチャを初採用
- **Gemini 2.5（2025年）**: Sparse MoE Transformer、マルチモーダル対応（テキスト・画像・音声）
- **Gemini 3 Pro（2025年末）**: Sparse MoE Transformer
  - 推定: 総パラメータ 1兆以上、活性化パラメータ 5B〜30B
  - 入力に応じて関連する Expert パスウェイのみを選択的に活性化
  - Dense モデルの推論深度を、はるかに小さいモデルの推論効率で実現
- Google は MoE の先駆的研究者: Sparsely-Gated MoE、GShard、Switch-Transformer、M4 等

### 1.4 NVIDIA の MoE 最適化

#### NVL72 での MoE 配置戦略

**GB200 NVL72 のアーキテクチャ:**
- 72 Blackwell GPU を NVLink Switch で接続
- 単一 NVLink ドメインとして 130 TB/s（双方向）の帯域幅
- 全 GPU 間でコヒーレントなメモリアクセス

**Wide Expert Parallelism（Wide-EP）:**
- TensorRT-LLM の Wide-EP は NVL72 のラックスケール NVLink ドメインを最大限に活用
- **EP=32 で EP=8 比 最大 1.8 倍の per-GPU スループット向上**（100 tokens/sec/user 条件）
- 重みロード圧力の削減、GroupGEMM 効率の改善
- 130 TB/s NVLink 帯域で通信オーバーヘッドを相殺

**MoE 推論性能（Blackwell NVL72 vs Hopper H200）:**
- MoE モデル（DeepSeek-R1、Kimi K2 Thinking、Mistral Large 3）で **10 倍の性能向上**
- トークンあたりのコストを **1/10 に削減**
- DeepSeek-R1: 単一 DGX B200（8 GPU）で 250 tokens/sec/user、最大スループット 30,000 tokens/sec
- GB300 NVL72: Hopper 比 **5 倍の per-GPU スループット**（MLPerf Inference v5.1）

**DeepSeek-R1 on GB200 NVL72（LMSYS ベンチマーク）:**
- Prefill スループット: 3.8 倍向上
- Decode スループット: 4.8 倍向上

#### NCCL の All-to-All 最適化

**Hybrid-EP ライブラリ:**
- NVIDIA が開発した Expert Parallel 通信最適化ライブラリ
- NVLink、Quantum InfiniBand、Spectrum-X Ethernet、RDMA、TMA コマンド、IBGDA を統合活用
- **わずか 4 SM で NVLink 帯域幅をほぼ最大化**
- GB200 NVL36 構成では 16 SM で NVLink 帯域を飽和

性能改善実績:
- DeepSeek-V3（256 Expert）: **約 14% の性能改善**
- Qwen 3 235B: BF16 で **5.5%**、MXFP8 で **約 9.9%** の改善
- CUDA ブロックレベルの並列性で Dispatch/Combine オペレータを最適化
- FP8/BF16 精度のネイティブサポート
- 通信と計算の完全なオーバーラップ

**DeepEP（DeepSeek 発）:**
- Expert Parallel 専用の効率的通信ライブラリ
- 64 以上の EP 度数にスケール
- Permute/Unpermute 融合オプション
- NVIDIA Hybrid-EP ブランチとして統合

**LatentMoE（Nemotron 3 向け）:**
- トークンをモデル隠れ次元からより小さいラテント次元に射影してから Expert ルーティング
- All-to-All トラフィックを d/l 倍（通常約 **4 倍**）削減

#### TensorRT-LLM の MoE サポート

- Wide-EP のネイティブサポート
- NVFP4（4ビット浮動小数点）精度での MoE 推論
- Multi-Token Prediction（MTP）最適化
- Disaggregated Serving（Prefill/Decode 分離）
- **3ヶ月間で per-GPU スループット最大 2.8 倍向上**（継続的最適化）
- SGLang、vLLM との連携によるオープンソースエコシステム

---

## 2. 耐障害性・高可用性

### 2.1 大規模 GPU クラスタの障害率

#### 数万 GPU クラスタでの MTBF（平均故障間隔）

GPU クラスタサイズと MTBF の関係（研究データ: arXiv:2410.21680）:

| クラスタサイズ（GPU 数） | MTBF / MTTF |
|----------------------|-------------|
| 8 GPU | 約 47.7 日 |
| 1,024 GPU | 約 7.9 時間 |
| 3,000 GPU（375ノード） | ピーク 56.6 時間、平均 33.0 時間 |
| 16,384 GPU | 約 1.8 時間（推定） |
| 131,072 GPU | 約 0.23 時間（約 14 分、推定） |

重要な知見:
- GPU 数の増加に対して MTBF は指数的に減少する
- 障害は全ジョブの 0.2% にしか影響しないが、**総ランタイムの 18.7% を消費**
- 大規模・長時間ジョブほど障害に遭遇する確率が高い
- 150M A100 GPU 時間、400 万ジョブの分析に基づくデータ

#### Meta Llama 3.1 405B トレーニングの障害レポート

**トレーニング構成:**
- GPU: 16,384 台の NVIDIA H100 80GB
- モデル: Llama 3 405B パラメータ
- 期間: 54 日間

**障害統計:**
- **総障害数: 419 回**（54 日間）
- **障害頻度: 約 3 時間に 1 回**（7.76 回/日）
- **有効トレーニング時間: 90% 以上を維持**
- チェックポイント・障害回復に **総トレーニング時間の約 2.1%** を消費
- 最適チェックポイント頻度: 4 分ごと、所要時間 2.5 秒

**障害の種類別内訳:**

| 障害種別 | 件数 | 割合 |
|---------|------|------|
| GPU 関連障害（NVLink 含む） | 148 | 30.1% |
| HBM3 メモリ障害 | 72 | 17.2% |
| GPU + HBM3 合計 | 220 | 約 58.7% |
| SRAM 障害 | 19 | 4.5% |
| GPU プロセッサ障害 | 17 | 4.1% |
| サイレントデータ破損 | 6 | 1.4% |
| 温度インターフェースセンサー | 6 | 1.4% |
| その他（ソフトウェアバグ、ネットワークケーブル、ネットワークアダプタ等） | 173 | 41.3% |

**追加の課題:**
- 数万 GPU の同時電力消費変動が数十 MW に達し、データセンター電力グリッドに負荷
- 電力変動が電力網の限界に接近

#### 障害の種類の一般分類

1. **ハードウェア障害**
   - GPU コア障害、HBM3 メモリ障害、SRAM 障害
   - NIC 障害、ネットワークケーブル障害
   - 電源障害、冷却系障害
   - NVLink リンク障害

2. **ソフトウェア障害**
   - CUDA ドライバ障害
   - NCCL 通信タイムアウト
   - アプリケーションバグ（OOM、数値異常）
   - OS / カーネル障害

3. **インフラ障害**
   - ネットワークスイッチ障害
   - ストレージ障害
   - 電力供給の不安定性

### 2.2 チェックポイント技術

#### 分散チェックポイントの仕組み

**基本アーキテクチャ:**
- 各 GPU のモデルパラメータ、オプティマイザ状態、学習状態を定期的にストレージに保存
- 障害発生時に最新のチェックポイントから学習を再開
- PyTorch Distributed Checkpoint（DCP）が標準的なフレームワーク

**チェックポイントの課題:**
- 大規模モデルのチェックポイントサイズは数百 GB〜数 TB に達する
- 同期チェックポイント中は全 GPU がアイドル状態になる
- ストレージ帯域がボトルネックになりやすい

#### チェックポイント頻度とオーバーヘッド

**Meta Llama 3 の実績:**
- チェックポイント頻度: **4 分ごと**
- チェックポイント所要時間: **2.5 秒**
- チェックポイント + 障害回復の総オーバーヘッド: **トレーニング時間の約 2.1%**

**一般的なトレードオフ:**
- 高頻度チェックポイント → 障害時のロールバック損失が小さい、オーバーヘッドが増加
- 低頻度チェックポイント → オーバーヘッドが小さい、障害時の損失が増大
- 最適頻度は MTBF とチェックポイントコストのバランスで決定

#### 非同期チェックポイント

**動作原理:**
1. GPU メモリからチェックポイントデータを CPU メモリにコピー（短時間の GPU ブロッキング）
2. CPU スレッドがバックグラウンドでストレージにデータを書き出し
3. GPU はステップ1完了後すぐにトレーニングを再開

**主要技術:**
- **PyTorch async_save() API**: DCP の非同期保存機能、トレーニングのクリティカルパスからチェックポイント生成を除外
- **Asynchronous Redundant Copying（ARC）**: 冗長スナップショットをホスト-デバイス協調で非同期実行
- **Asynchronous Erasure Coding（AEC）**: イレージャーコーディングによるデータ保護を非同期で実行
- **Asynchronous Optimizer Recomputing（AOR）**: オプティマイザ状態の再計算による容量削減

**先進的システム:**
- **Gemini（ByteDance）**: CPU メモリにチェックポイントを保存し、マシン間バックアップで高頻度チェックポイントを実現
- **ByteCheckpoint**: 異なるフレームワークのチェックポイントを並列性非依存の表現に統一
- **TorchFT**: PyTorch の per-step 耐障害性ライブラリ。個々のノード/GPU が障害を起こしてもジョブ全体の再起動なしに継続可能

**効果:**
- 同期チェックポイント比で GPU ブロッキング時間を **10 倍以上削減**
- トレーニングの goodput（有効処理率）を大幅に改善

### 2.3 NVIDIA の高可用性技術

#### NVIDIA Mission Control

**概要:**
- 大規模 AI クラスタの運用管理プラットフォーム
- BCM（Base Command Manager）統合インフラサービスを含む
- 最新版: Mission Control 2.2（2025年）

**主要コンポーネント:**

1. **Autonomous Hardware Recovery（AHR）**
   - ハードウェア障害の自動検出と修復
   - NVSwitch コンポーネントの RMA 後の自動復帰ワークフロー
   - NVLink リンクの状態確認: リンク活性、速度、ファブリック登録、帯域幅、NVLink ドメイン/パーティション
   - 非同期操作でラック間の並列修復を実行
   - Mission Control 2.2: AHR 28.4

2. **Autonomous Job Recovery（AJR）**
   - AI トレーニングジョブの自動障害検出・隔離・回復
   - 数千 GPU 規模のトレーニングジョブの手動介入を大幅に削減
   - **Anomaly Detection**: テレメトリデータを使用したリアルタイム異常検出（性能低下、クラッシュ、ハング）
   - **Failure Attribution and Characterization（FACT）エンジン**: システムログ、アプリケーションログ、ネットワークメトリクスを分析して障害原因を特定
   - コンピュート/インフラ障害（ネットワーク、ストレージ、ノード/GPU）とソフトウェア障害の両方に対応
   - Mission Control 2.2: AJR 1.5

3. **Autonomous Recovery Engine（ARE）**
   - AHR と AJR を統合した統一レジリエンシフレームワーク
   - Mission Control 2.2 で導入

#### NetQ テレメトリ

- NVLink と Ethernet の統合可視化
- リアルタイムのネットワークファブリック監視・トラブルシューティング
- NVLink Switch テレメトリの収集・分析
- Spectrum-X Ethernet からのテレメトリデータ統合（スイッチ、SuperNIC、GPU）
- パケットロス、ハードウェア障害、設定エラーの即時検出
- NVIDIA Cumulus Linux ファブリックの検証

#### GPU Direct RDMA のフォールバック

- GPU-NIC 間の直接データパスを提供（CPU バイパス）
- PCIe 経由の P2P 通信が障害時にはシステムメモリ経由のフォールバックパスを使用
- Grace Blackwell プラットフォームでは ConnectX-8 と Data Direct テクノロジーで最適化
- ドライバレベルでの障害検出と代替パスへの切り替え

#### NVLink の障害検出と切り離し

- Mission Control AHR が NVLink リンク状態を継続的に監視
- 障害リンクの検出: 帯域幅低下、CRC エラー、リンクダウンの検知
- 障害 NVLink の切り離し（Isolation）: 障害リンクを NVLink ドメインから除外
- 縮退動作: 残存リンクでの通信を継続（帯域幅は低下）
- RMA 後の自動復帰: AHR が新コンポーネントを検証し、NVLink ドメインに再統合

### 2.4 ジョブスケジューリング

#### Slurm / Kubernetes の GPU スケジューリング

**Slurm:**
- NVIDIA DGX SuperPOD / BasePOD の標準スケジューラ
- GPU を消費可能リソースとして認識（ジョブアカウンティング連携）
- ジョブごとに任意数の GPU を要求可能
- Pyxis / Enroot をコンテナプラグイン・ランタイムとして使用
- Base Command Manager により自動デプロイ・設定

**Kubernetes:**
- NVIDIA GPU Operator によるクラウドネイティブ GPU 管理
- MIG（Multi-Instance GPU）対応
- GPU リソースの動的割り当て

**ハイブリッドデプロイメント:**
- NVIDIA DeepOps: Ansible ベースの自動デプロイメントツール
- Slurm + Kubernetes のハイブリッド構成をサポート
- 監視サービス等の付随機能も自動展開

#### NVIDIA Base Command / DGX Cloud

**Base Command Manager（BCM）:**
- DGX SuperPOD / BasePOD のクラスタ管理プラットフォーム
- Slurm ワークロードマネージャの統合管理
- Mission Control との連携（AHR、AJR、NetQ、Observability Stack）
- ハードウェアインベントリ管理、ファームウェア更新

**DGX Cloud:**
- NVIDIA が管理するクラウド GPU サービス
- Slurm ベースのジョブスケジューリングを提供
- マルチテナント環境での GPU リソース分離
- Azure、GCP、Oracle Cloud 上で提供

---

## 3. 実デプロイメント事例

### 3.1 DGX SuperPOD 導入事例

#### Eli Lilly（1,016 GPU）: 創薬 AI

| 項目 | 詳細 |
|------|------|
| 発表時期 | 2025年（GTC Washington, D.C.） |
| システム | 世界初の DGX SuperPOD with DGX B300 |
| GPU | 1,016 NVIDIA Blackwell Ultra GPU |
| ネットワーク | 統一ネットワーキングファブリック（GPU・ストレージ・関連システム間） |
| 用途 | 創薬 AI |
| 投資規模 | Lilly と NVIDIA の共同イノベーションラボ（サンフランシスコ）: $1B |

活用分野:
- 数百万の実験データで AI モデルをトレーニング
- ゲノム全配列の解析
- 患者アウトカムの予測
- 生化学的可能性の網羅的探索
- 創薬プロセスの範囲と精度を飛躍的に拡大

#### Mayo Clinic: 医療 AI

| 項目 | 詳細 |
|------|------|
| システム | DGX SuperPOD with DGX B200 |
| 管理基盤 | NVIDIA Mission Control |
| データ資産 | 2,000 万枚のデジタル化病理スライド、世界最大級の患者データベース |
| 主要用途 | Pathomics（病理画像解析の計算科学） |

活用分野:
- パソミクス（Pathomics）: 病理画像からの疾患研究
- 創薬
- 精密医療（Precision Medicine）

#### xAI Colossus（100,000+ GPU）: 世界最大の GPU クラスタ

**Colossus 1.0（2024年）:**

| 項目 | 詳細 |
|------|------|
| 所在地 | Memphis, Tennessee |
| 構築期間 | **122 日間**で 100,000 GPU クラスタを構築 |
| GPU 構成 | 100,000 NVIDIA H100 GPU |
| ネットワーク | NVIDIA Spectrum-X Ethernet（SN5600 スイッチ + BlueField-3 SuperNIC） |
| 電力 | 250 MW（Memphis 電力網から供給） |
| 用途 | Grok ファミリーの大規模言語モデルのトレーニング |

**ネットワーク性能:**
- Spectrum-X によるフロー衝突ゼロ、パケットロスゼロ
- **データスループット 95%**（標準 Ethernet では 60%）
- アダプティブルーティング、輻輳制御、Direct Data Placement

**拡張フェーズ（2025年）:**
- 200,000 GPU に倍増（+92 日間で完了）
- 構成: H100 150,000 + H200 50,000 + GB200 30,000 GPU

**Colossus 2.0（2026年1月発表）:**

| 項目 | 詳細 |
|------|------|
| 総電力容量 | **2 ギガワット** |
| GPU 数 | **555,000 NVIDIA GPU** |
| 投資額 | **約 $18B**（GPU 購入費） |
| 施設 | Memphis に 3 棟目の建物を取得 |

**将来計画:**
- 100 万 GPU 以上への拡張を計画
- 2026年 Q1: 3 棟目のデータセンター変換開始

**冷却:**
- 約半分: xAI の中水（graywater）施設による水冷
- 約半分: 空冷

### 3.2 クラウドプロバイダ

#### Microsoft Azure: GB200/GB300 NVL72 ベースの AI インフラ

**世界初の GB300 NVL72 大規模クラスタ（2025年10月）:**

| 項目 | 詳細 |
|------|------|
| VM シリーズ | NDv6 GB300 |
| システム数 | **4,600 以上の GB300 NVL72 システム**（64 ラック） |
| GPU 総数 | **4,608 GB300 GPU** |
| 構成 | 各ラック 72 Blackwell Ultra GPU + 36 Grace CPU |
| ネットワーク | 次世代 NVIDIA InfiniBand |
| 性能 | 各システム 1.44 ExaFLOPS（FP4）、全体 92.1 ExaFLOPS |
| メモリ | 各システム 37 TB の統合高速メモリ |
| 主要顧客 | OpenAI |

**MLPerf Inference v5.1 成績:**
- DeepSeek-R1（671B）で Hopper 比 **最大 5 倍の per-GPU スループット**

**将来計画:**
- 数十万台規模の Blackwell Ultra GPU をグローバルデータセンターに展開
- Vera Rubin（2026年後半）対応の戦略的データセンター計画

#### Oracle Cloud: Blackwell クラスタ

**OCI Supercluster（2024年9月発表、2025年提供開始）:**

| 項目 | 詳細 |
|------|------|
| GPU 規模 | 最大 **131,072 NVIDIA Blackwell GPU** |
| 性能 | **2.4 ゼタFLOPS**（FP4、1 ゼタFLOPS = 1,000 エクサFLOPS） |
| 位置づけ | 世界初のゼタスケール AI スーパーコンピュータ |
| 比較 | Frontier スパコンの 3 倍以上、他ハイパースケーラの 6 倍以上 |

GB200 NVL72 on OCI:
- 各 GB200 NVL72 で 1 ExaFLOPS 以上のトレーニング性能
- NVLink + NVLink Switch で 72 GPU 間 129.6 TB/s
- 2025年に一般提供開始（GA）

#### CoreWeave: GPU-as-a-Service

**概要:**

| 項目 | 詳細 |
|------|------|
| 設立 | 元暗号通貨マイニング企業から AI クラウドに転身 |
| データセンター数 | 32（2025年時点） |
| 総 GPU 数 | **250,000 GPU** |
| 2024年売上 | $1.92B（前年比 **737%** 増） |
| 2025年売上 | **$5.1B**（前年比 **168%** 増） |
| 2026年売上見通し | **$12B〜$13B** |
| 受注残高 | **$66.8B**（年初の $15B から 4 倍以上） |

**主要マイルストーン:**
- 2025年2月: 世界初の商用 GB200 NVL72 提供開始
- 2025年7月: 世界初の商用 GB300 NVL72（Blackwell Ultra）提供開始

**主要顧客契約:**
- OpenAI: 5年間 **$12B** のクラウドコンピューティング契約（2025年3月）
- Meta: **$14.2B** の契約

### 3.3 国家プロジェクト

#### Stargate（$500B 投資計画）

| 項目 | 詳細 |
|------|------|
| 発表日 | 2025年1月21日（Donald Trump 大統領が発表） |
| 法人 | Stargate LLC（デラウェア州設立） |
| 総投資額 | **$500B**（4年間） |
| 即時展開 | **$100B** |

**参加企業と出資:**

| 企業 | 役割 | 出資額 | 持分 |
|------|------|--------|------|
| OpenAI | 共同設立者 | $19B | 40% |
| SoftBank | 共同設立者 | $19B | 40% |
| Oracle | パートナー | $7B | - |
| MGX | パートナー | $7B | - |
| Microsoft | テクノロジーパートナー | - | - |
| NVIDIA | テクノロジーパートナー | - | - |
| Arm | テクノロジーパートナー | - | - |

**インフラ計画:**
- 旗艦サイト: テキサス州アビリーン（2025年9月オープン）
- 追加 5 サイト: ウィスコンシン、ニューメキシコ、オハイオ等
- **総計画容量: 約 7 ギガワット**
- **投資総額: $400B 以上**（3年間）
- CoreWeave との継続プロジェクトも含む

**期待効果:**
- 米国内で **100,000 人以上の雇用創出**（Trump 大統領発言）
- 米国の AI インフラにおけるリーダーシップ確保

**課題:**
- OpenAI、Oracle、SoftBank 間の未解決の紛争により一部停滞の報道あり

#### 各国のソブリン AI

**日本: RIKEN + NVIDIA**

| プロジェクト | 詳細 |
|------------|------|
| AI for Science スパコン | 1,600 Blackwell GPU（GB200 NVL4）、Quantum-X800 InfiniBand |
| 量子コンピューティングシステム | 540 Blackwell GPU |
| 合計 | **2,140 NVIDIA Blackwell GPU** |
| 稼働開始 | **2026年春** |
| 目的 | 生命科学、材料科学、気候予測、製造、ラボオートメーション |

**FugakuNEXT（2030年目標）:**
- Fujitsu と NVIDIA の共同設計
- FUJITSU-MONAKA-X CPU + NVIDIA NVLink Fusion
- 既存スパコン比 **100 倍のアプリケーション性能**
- 2030年稼働予定

**Fujitsu ソブリン AI サーバ製造（2026年3月開始）:**
- 笠島工場（Fugaku 製造実績あり）で "Made in Japan" AI サーバを製造
- NVIDIA HGX B300 および RTX PRO 6000 Blackwell Server Edition GPU 搭載

**日本の AI インフラ投資:**
- 総投資計画: **$135B**（AI + 量子インフラ）

**その他の国家プロジェクト:**
- 各国が NVIDIA ベースのソブリン AI インフラを構築
- デジタル主権確保と国内 AI 能力強化が目的
- NVIDIA の Sovereign AI パートナーシッププログラムを通じて展開

---

## まとめ: 数値データ一覧

### MoE モデル比較

| モデル | 総パラメータ | 活性化パラメータ | Expert 数 | Top-K |
|--------|------------|----------------|----------|-------|
| Mixtral 8x7B | 46.7B | 13B | 8 | 2 |
| Mixtral 8x22B | 141B | 39B | 8 | 2 |
| GPT-4（推定） | ~1.8T | 不明 | 16（推定） | 不明 |
| DeepSeek-V3/R1 | 671B | 37B | 256+1 | 8 |
| Gemini 3 Pro（推定） | >1T | 5B〜30B | 不明 | 不明 |

### 障害統計

| メトリクス | 値 |
|-----------|-----|
| 16,384 GPU クラスタの MTBF | 約 1.8〜3 時間 |
| Llama 3 トレーニング障害数 | 419 回 / 54 日 |
| GPU/HBM3 起因の障害割合 | 58.7% |
| チェックポイント + 回復のオーバーヘッド | 約 2.1% |
| 有効トレーニング時間 | 90% 以上 |

### デプロイメント規模

| デプロイメント | GPU 数 | 投資額 |
|--------------|--------|--------|
| xAI Colossus（現在） | 230,000+ | - |
| xAI Colossus 2.0（計画） | 555,000 | $18B |
| Microsoft Azure GB300 | 4,608（初期） | - |
| Oracle OCI Supercluster | 最大 131,072 | - |
| CoreWeave | 250,000（総数） | - |
| Stargate | - | $500B（4年） |
| Eli Lilly | 1,016 | $1B（共同ラボ） |
| RIKEN（日本） | 2,140 | - |

---

## ソース

### MoE アーキテクチャ
- https://blogs.nvidia.com/blog/mixture-of-experts-frontier-models/
- https://developer.nvidia.com/blog/optimizing-communication-for-mixture-of-experts-training-with-hybrid-expert-parallel/
- https://developer.nvidia.com/blog/applying-mixture-of-experts-in-llm-architectures/
- https://developer.nvidia.com/blog/scaling-large-moe-models-with-wide-expert-parallelism-on-nvl72-rack-scale-systems/
- https://developer.nvidia.com/blog/delivering-massive-performance-leaps-for-mixture-of-experts-inference-on-nvidia-blackwell
- https://developer.nvidia.com/blog/how-nvidia-gb200-nvl72-and-nvidia-dynamo-boost-inference-performance-for-moe-models/
- https://huggingface.co/blog/moe
- https://arxiv.org/abs/2401.04088
- https://arxiv.org/abs/2412.19437
- https://semianalysis.com/2023/07/10/gpt-4-architecture-infrastructure/
- https://fireworks.ai/blog/deepseek-model-architecture
- https://developer.nvidia.com/blog/nvidia-blackwell-delivers-world-record-deepseek-r1-inference-performance/
- https://lmsys.org/blog/2025-09-25-gb200-part-2/
- https://github.com/deepseek-ai/DeepEP
- https://developer.nvidia.com/blog/nvidia-connectx-8-supernics-advance-ai-platform-architecture-with-pcie-gen6-connectivity/

### 耐障害性・高可用性
- https://www.tomshardware.com/tech-industry/artificial-intelligence/faulty-nvidia-h100-gpus-and-hbm3-memory-caused-half-of-the-failures-during-llama-3-training-one-failure-every-three-hours-for-metas-16384-gpu-training-cluster
- https://arxiv.org/html/2410.21680v1
- https://nebius.com/blog/posts/how-we-build-reliable-clusters
- https://pytorch.org/blog/distributed-checkpoint-efficient-checkpointing-in-large-scale-jobs/
- https://docs.nvidia.com/mission-control/docs/systems-administration-guide/2.2.0/autonomous-hardware-recovery.html
- https://docs.nvidia.com/mission-control/docs/systems-quick-start-guide/2.0.0/ajr/ajr-overview.html
- https://developer.nvidia.com/blog/next-generation-ai-factory-telemetry-with-nvidia-spectrum-x-ethernet/
- https://docs.nvidia.com/mission-control/index.html
- https://www.crusoe.ai/resources/blog/autoclusters-minimizing-hardware-failures-in-large-gpu-clusters

### 実デプロイメント事例
- https://blogs.nvidia.com/blog/lilly-ai-factory-nvidia-blackwell-dgx-superpod/
- https://hlth.com/insights/news/mayo-clinic-deploys-nvidia-infrastructure-to-drive-genai-solutions-in-medicine-2025-07-29
- https://nvidianews.nvidia.com/news/spectrum-x-ethernet-networking-xai-colossus
- https://introl.com/blog/xai-colossus-2-gigawatt-expansion-555k-gpus-january-2026
- https://azure.microsoft.com/en-us/blog/microsoft-azure-delivers-the-first-large-scale-cluster-with-nvidia-gb300-nvl72-for-openai-workloads/
- https://blogs.oracle.com/cloud-infrastructure/worlds-largest-ai-supercomputer-in-the-cloud
- https://www.coreweave.com/blog/coreweave-pushes-boundaries-with-gb200-and-more
- https://openai.com/index/announcing-the-stargate-project/
- https://nvidianews.nvidia.com/news/nvidia-and-riken-advance-japans-scientific-frontiers-with-new-supercomputers-for-ai-and-quantum-computing
- https://global.fujitsu/en-global/pr/news/2026/02/12-01
- https://introl.com/blog/japan-ai-infrastructure-135-billion-investment-2025
