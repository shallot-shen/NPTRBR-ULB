# NP-TRBR 在 ULB 公共数据集上的复现

[English](README.md)

本仓库复现论文中 **TabM + NP-TRBR** 在 ULB/Worldline 信用卡欺诈数据集（[OpenML 数据集 1597](https://www.openml.org/d/1597)）上的匹配实验。包含官方 TabM 骨干上的门控与损失四格消融，以及固定正类权重的 BCE 对照。这是论文 ULB 匹配结果所用版本，不是早期本地 STaR／TabM 风格实验。

压缩包只包含源码、配置、测试和公共数据实验的汇总参考结果，不包含交易明细、训练权重、逐行预测或非公开数据集。

## 1. 安装与检查

在仓库根目录运行以下单行命令，PowerShell、Bash 均可使用。建议使用 Python 3.11 或 3.12；CPU 检查使用 Python 3.12。先按 [PyTorch 安装页面](https://pytorch.org/get-started/locally/) 安装适合本机的版本。运行 `--device cuda:0` 需要支持 CUDA 的 PyTorch；已有可用 CUDA 环境时可继续使用。

```text
python -m pip install -r requirements.txt
python reproduce_ulb.py smoke
python -m unittest discover -s tests -v
```

冒烟测试使用少量合成数据在 CPU 离线运行，不下载或训练真实 ULB。

## 2. 准备公共数据划分

```text
python reproduce_ulb.py prepare
python reproduce_ulb.py audit
```

`prepare` 无需凭据，从 OpenML 下载数据集 1597。若已有原始顺序一致的公共 CSV，可运行 `python reproduce_ulb.py prepare --input creditcard.csv`。已有此前实验准备好的 CSV 时，放到 `data/ulb_prepared.csv`，运行 `python reproduce_ulb.py split`。命令不会覆盖已准备的 CSV。使用其他路径时，在各步骤统一指定 `--data` 和 `--split-dir`。

| 分区 | 记录数 | 正例数 |
|---|---:|---:|
| 训练 | 182,277 | 365 |
| 早停验证 | 30,839 | 32 |
| 校准 | 14,730 | 20 |
| 测试 | 56,961 | 75 |

有效变量只有 `V1`–`V28` 和 `Amount`，标签为 `Class`。实验所用导出没有 `Time`。划分保留原始行顺序，不将其解释为经核实的交易时序；行号和时间均不作为模型特征。预处理保留一个常量日期作为元数据，匹配的官方模型会去掉由它派生的常量列。划分元数据记录准备后 CSV 和划分清单的 SHA-256；分区不符时直接报错，不会偷偷重排。

## 3. 训练五组匹配实验

```text
python -u reproduce_ulb.py train --device cuda:0 --seeds 42 7 2027
python reproduce_ulb.py development --device cuda:0 --seeds 42 7 2027
```

CPU 运行可将 `cuda:0` 换成 `cpu`；同一次实验的训练与评估需用同一设备。默认共训练 **15 个模型**。建议先用一条命令跑一组：

```text
python -u reproduce_ulb.py train --device cuda:0 --arms nptrbr --seeds 42
```

| 实验选项 | MECG 门控 | NP-TRBR | BCE 正类权重 |
|---|---|---|---:|
| `bce` | 关闭 | 关闭 | 1 |
| `mecg_bce` | 开启 | 关闭 | 1 |
| `nptrbr` | 关闭 | 开启 | 1 |
| `mecg_nptrbr` | 开启 | 开启 | 1 |
| `wbce_w2` | 关闭 | 关闭 | 2 |

冻结配置使用 `tabm==0.0.3`、32 个成员、3 个宽度 128 的块、16 个分段线性分箱、8 维数值嵌入、FP32，成员正负采样目标比为 1:10，最多训练 100 轮。每个种子根据早停验证 AP 保留至多 5 个检查点。加权 BCE 固定正类权重为 2，复现论文已选定的对照，不重跑跨数据集权重筛选。

训练与开发评估不对测试集打分。原命令重跑时，会跳过已校验的完成任务；中断任务从最近一个已保存的 epoch 边界恢复，要求代码、数据、划分、配置、依赖版本及设备一致。更改设置请用新的 `--output`。只加载自己生成且信任的检查点及预处理文件。

## 4. 测试评估与生成汇总表

训练完成后，显式解锁回顾性测试评估：

```text
python reproduce_ulb.py evaluate --device cuda:0 --seeds 42 7 2027 --unlock-test
python summarize_ulb.py
```

评估先在每个种子内平均各检查点的预测概率，再对多个种子的概率取平均。Platt 校准和阈值**只用校准集**拟合。报告四档规则：校准 FPR 上限 0.1%、0.5%、校准后概率 0.5、校准集最大 F1。实际测试 FPR 可与校准上限不同。AP 和 AUROC 使用未校准的分数，不依赖阈值。

- `runs/ulb_paper/development_metrics.csv`：交叉拟合的校准诊断结果。
- `runs/ulb_paper/evaluation/test_metrics.csv`：各种子与集成的 AP、AUROC、TP/FP/TN/FN、精确率、召回率、F1 和 FPR。
- `runs/ulb_paper/evaluation/test_ensemble_metrics.csv`：仅集成的结果。
- `runs/ulb_paper/evaluation/paired_intervals.csv`：指定损失对照的 2,000 次配对、按类别分层的行重抽样区间，条件是已有拟合模型及冻结阈值。区间不能证明方法等效，也不代表未来上线表现。
- `tables/ulb/`：`summarize_ulb.py` 生成的排序、判定和匹配析因汇总表。

只测试一组时，在评估命令使用对应的 `--arms` 和 `--seeds`，并加 `--bootstrap-repeats 0 --report-dir runs/ulb_one_arm`。如更改评估组别、种子或重抽样设置，请指定不同的 `--report-dir`；已有模型仍从 `--output` 读取。

[归档 ULB 汇总结果](reference_results/) 是论文参考数值，不是随包提供的逐行预测。论文中 NP-TRBR 集成的 AP 为 **0.792128**、AUROC 为 **0.987409**；匹配 BCE 分别为 **0.798344** 和 **0.988100**。不同 GPU 和依赖版本不保证逐位相同。可不训练，直接从归档汇总生成对照表：

```text
python summarize_ulb.py --metrics reference_results/test_ensemble_metrics.csv --output tables/ulb_reference
```

## 5. 可选的独立训练基线

这些运行器使用同一公共划分，但模型、加权及检查点策略各自独立，不应将其结果用于隔离单个损失或门控的效应。结果写入不同目录。

```text
python -m pip install -r requirements-baselines.txt
python run_extra_baselines.py --models random_forest xgboost lightgbm ft_transformer official_tabm --device cuda:0 --tree-device cuda --unlock-test
python -m mecg_star_tabm suite --models logistic,catboost,vanilla_tabm --seeds 42,7,2027 --run-mode final --device cuda:0 --output-root runs/ulb_legacy_baselines --resume
```

额外基线的 CPU 运行需同时指定 `--device cpu --tree-device cpu`。省略其 `--unlock-test` 则只训练和评估开发集；对旧版 CLI 可用 `--run-mode tune` 并另设输出目录。旧版 CLI 还支持 `--models full --config mecg_star_tabm/configs/legacy_k32.json` 运行早期本地 32 成员 STaR 配置；该命令复跑冻结配置，不复跑此前的调参历史。独立官方 TabM 基线和本地 `vanilla_tabm` 均不是匹配 `bce` 对照。

## 引用与发布

数据和方法引用见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。第三方软件包与公共数据需单独获取。公开仓库前请与权利人确认版权署名与发布授权。
