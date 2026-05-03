# soarm101-isaac-lab-sagemaker-rl

SO-ARM101 の Reach タスクを Isaac Lab で強化学習し、Amazon SageMaker Training Job + Managed Spot Training で実行するサンプルコードです。

関連ブログ記事：[SO-ARM101 を Isaac Lab × Training Job (Managed Spot) で強化学習してみました](https://dev.classmethod.jp/articles/)（公開後にリンク差し替え）

> English version: [README.md](README.md)

## 概要

- ベースイメージ: `nvcr.io/nvidia/isaac-lab:2.3.2`（Isaac Lab 2.3.2 + Isaac Sim 5.1.x 同梱）
- タスク: `Isaac-SO-ARM101-Reach-v0`（[MuammerBay/isaac_so_arm101](https://github.com/MuammerBay/isaac_so_arm101) v1.2.0）
- 学習基盤: SageMaker Training Job、ml.g6.2xlarge（NVIDIA L4 24 GB）、Managed Spot
- リージョン: `ap-northeast-1`

## リポジトリ構成

```
.
├── cdk/                    # AWS CDK（TypeScript）: S3 / ECR / IAM Role / Budget
├── scripts/
│   └── push_to_ecr.sh      # 学習用 image を build して ECR に push
├── src/
│   ├── train.py            # SageMaker entrypoint（SIGTERM 対応、ckpt 自動再開）
│   └── entrypoint.sh
├── Dockerfile              # NGC isaac-lab:2.3.2 を継承
├── submit.py               # SageMaker Estimator 起動スクリプト
└── README.md / README.ja.md
```

## 前提条件

- SageMaker / S3 / ECR / Budgets 権限のある AWS アカウント
- `ap-northeast-1` 用に設定済みの AWS CLI v2
- Docker（`linux/amd64` ビルド対応）
- Node.js 20.x と AWS CDK v2（`pnpm add -g aws-cdk`）
- NVIDIA NGC アカウントと API key（`docker login nvcr.io`）
- Python 3.11 と SageMaker Python SDK（`pip install sagemaker`）

## セットアップ

### 1. リポジトリをクローン

```bash
git clone https://github.com/furuya02/soarm101-isaac-lab-sagemaker-rl.git
cd soarm101-isaac-lab-sagemaker-rl
```

### 2. CDK で AWS リソースを構築

```bash
cd cdk
pnpm install

export AWS_REGION=ap-northeast-1
export ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

cdk bootstrap aws://${ACCOUNT_ID}/${AWS_REGION}
cdk deploy \
  -c account_id=${ACCOUNT_ID} \
  -c region=${AWS_REGION}
```

このスタックで作成されるリソース：

- S3 バケット: `soarm101-isaac-lab-sagemaker-rl-<ACCOUNT_ID>`
- ECR リポジトリ: `soarm101-isaac-lab-sagemaker-rl`
- IAM ロール: `soarm101-isaac-lab-sagemaker-rl-sagemaker-execution-role`
- 月次 Budget アラート（USD 100、10/50/90 % しきい値、`hirauchi.shinichi@classmethod.jp` へ通知）

bucket suffix や Budget 通知先を上書きする場合：

```bash
cdk deploy \
  -c account_id=${ACCOUNT_ID} \
  -c bucket_suffix=20260503 \
  -c budget_email=you@example.com
```

### 3. 学習用 image を build して ECR に push

```bash
cd ..

# NGC にログイン（NGC API key が必要）
docker login nvcr.io

./scripts/push_to_ecr.sh
```

初回 push は約 15 GB の転送が発生し、回線速度に応じて 30〜60 分程度かかります。同一リージョンの ECR から SageMaker への pull はリージョン間データ転送料金が発生しません。

### 4. SageMaker Training Job を投入

```bash
export SAGEMAKER_ROLE_ARN=$(aws iam get-role \
  --role-name soarm101-isaac-lab-sagemaker-rl-sagemaker-execution-role \
  --query 'Role.Arn' --output text)
export ECR_IMAGE_URI=${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/soarm101-isaac-lab-sagemaker-rl:latest
export S3_BUCKET=soarm101-isaac-lab-sagemaker-rl-${ACCOUNT_ID}

# On-demand で動作確認
USE_SPOT=false MAX_RUN_HOURS=2 python submit.py

# Managed Spot 実行
USE_SPOT=true MAX_RUN_HOURS=2 MAX_WAIT_HOURS=6 python submit.py
```

### 5. 学習済みモデルの取得

```bash
JOB_NAME=<submit.py の出力に表示される job name>
aws s3 cp s3://${S3_BUCKET}/output/${JOB_NAME}/output/model.tar.gz .
tar xzf model.tar.gz
# rsl_rl/<task>/<run>/model_<iter>.pt が取り出せる
```

## コスト目安（ap-northeast-1、2026 年 5 月時点）

| リソース | 想定コスト |
|---|---|
| ml.g6.2xlarge（オンデマンド） | 約 $1.81 / 時 |
| ml.g6.2xlarge（Managed Spot、70 % off） | 約 $0.54 / 時 |
| ECR ストレージ（15 GB image） | 約 $1.50 / 月 |
| S3（成果物 + チェックポイント、1 GB 未満） | 月額 $0.10 未満 |

Reach タスクの典型的な学習（`--num_envs 64`、`--max_iterations 1000`）は、Managed Spot の場合 1 試行あたり USD 5 を十分下回る見込みです。

## クリーンアップ

```bash
cd cdk
cdk destroy
# S3 バケット内のオブジェクトと ECR の image は、必要に応じて手動で削除してください。
```

## 注意事項

- **Managed Spot にはチェックポイント実装が必須**。チェックポイントがないと `max_wait` の上限が 1 時間に制限されます。`src/train.py` は `/opt/ml/checkpoints/model_*.pt` から自動的に再開します。
- **`max_run` は必ず指定**。学習暴走による課金事故を防ぎます。
- **リージョン固定**。ECR / S3 / SageMaker をすべて `ap-northeast-1` に揃え、リージョン間データ転送料金を回避します。
- **Image サイズ**。ベース image は約 15 GB。SageMaker はジョブ起動時に pull するため、起動オーバーヘッドが 5〜10 分程度発生します。

## ライセンス

本サンプルコードは MIT License で公開しています。

`isaac_so_arm101` は BSD-3-Clause です。NVIDIA Isaac Sim および Isaac Lab は [NVIDIA Omniverse License Agreement](https://docs.omniverse.nvidia.com/install-guide/latest/common/NVIDIA_Omniverse_License_Agreement.html) に従います。
