# 校园论坛帖子推荐系统

本项目是《大数据原理与技术》课程大作业的推荐系统算法层原型。系统面向校园论坛帖子推荐，完整覆盖数据预处理、用户画像、多兴趣表示、多路召回、候选融合、DeepFM 排序、重排、离线评估和画像增量更新。

## 项目结构

```text
campus_recommender/
├── data/                    # CSV 数据与生成结果
├── models/                  # 训练后的 DeepFM 和特征编码器
├── scripts/                 # 测试脚本和模拟数据脚本
├── src/
│   ├── data_loader.py
│   ├── zanao_jsonl_parser.py
│   ├── seed_data_expander.py
│   ├── large_dataset_builder.py
│   ├── data_validator.py
│   ├── preprocess.py
│   ├── feature_engineering.py
│   ├── user_profile.py
│   ├── multi_interest.py
│   ├── recall/
│   ├── merge_candidates.py
│   ├── models/deepfm.py
│   ├── ranker.py
│   ├── rerank.py
│   ├── train.py
│   ├── evaluate.py
│   ├── update_profile.py
│   └── recommend.py
└── main.py
```

## 数据表

- `users.csv`：用户基础信息。
- `posts.csv`：帖子基础信息。
- `post_stats.csv`：浏览、点赞、评论、收藏、热度和质量分。
- `user_behaviors.csv`：用户行为日志。
- `post_tags.csv`：帖子标签关系。
- `questionnaire.csv`：新用户问卷。
- `user_profiles.csv`：用户画像输出。
- `training_samples.csv`：DeepFM 训练样本。

## 安装依赖

建议使用你的 `bigdata` 环境：

```bash
conda activate bigdata
pip install pandas numpy scikit-learn jieba scipy torch
```

如果只安装 CPU 版 PyTorch：

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## 如何运行

生成模拟数据：

```bash
python scripts/mock_data.py
```

使用约 100 条真实校园集市帖子作为种子样本，扩展构造课程实验数据集：

```bash
python scripts/build_dataset_from_zanao_jsonl.py ^
  --input data/raw/zanao_100_posts.jsonl ^
  --output data/ ^
  --num_posts 2000 ^
  --num_users 500 ^
  --num_behaviors 30000 ^
  --num_comments 6000
```

如果 JSONL 文件在其他目录，把 `--input` 改成实际路径即可。构建脚本会输出标准 CSV，并生成：

```text
outputs/dataset_build_report.md
outputs/dataset_validation_report.md
```

说明：该数据集以真实校园集市帖子为种子样本，结合校园论坛场景规则扩展构造，用于推荐系统算法流程验证。扩展数据不应被视作真实线上用户行为；真实联系方式不会保存，只保留 `has_contact_info` 布尔字段。

完整跑通主流程：

```bash
python main.py
```

主流程会完成：

1. 读取或生成模拟数据。
2. 数据预处理和正负样本构造。
3. 用户画像构建。
4. 多兴趣时间感知向量构建。
5. 训练样本生成。
6. DeepFM 训练。
7. 对指定用户生成 Top10 推荐。
8. 输出 Precision@K、Recall@K、HR@K、NDCG@K。

## 训练模型

```bash
python -m src.train
```

模型保存到：

```text
models/deepfm.pt
models/feature_encoder.json
```

## 生成推荐

```bash
python -m src.recommend
```

推荐结果包含：

```text
post_id, title, board, tags, rank_score, rerank_score, recall_sources, recommend_reason
```

## 离线评估

```bash
python -m src.evaluate
```

指标包括：

```text
Precision@5/10/20
Recall@5/10/20
HR@5/10/20
NDCG@5/10/20
```

## 用户画像增量更新

`src/update_profile.py` 支持新增行为后更新：

- `user_behaviors.csv`
- `user_profiles.csv`
- 用户多兴趣向量

不要求实时重训 DeepFM。

## 推荐系统流程

```text
数据读取与预处理
→ 用户画像构建
→ 帖子特征构建
→ 多兴趣时间感知用户表示
→ 多路召回
→ 候选集融合
→ DeepFM 排序
→ 多样性/新颖性/时效性重排
→ 输出 TopN 推荐
→ 新行为触发用户画像更新
```

## 创新点

- 多路召回：热门、最新、内容、ItemCF、画像、时间场景、位置场景、冷启动。
- 多兴趣时间感知用户表示：学习、生活、社交、发展四类兴趣向量。
- DeepFM 排序：结合稀疏类别特征、连续数值特征和交叉统计特征。
- 重排策略：控制多样性、新颖性和时效性。
- 冷启动处理：问卷优先，无问卷则热门、最新和场景召回兜底。
- 用户画像增量更新：新行为更新标签、板块、时间、位置和兴趣权重。

如果 Windows PowerShell 中看到中文乱码，通常只是终端代码页显示问题；CSV 使用 `utf-8-sig` 编码，Python、Excel 和 WPS 均可正常读取。
