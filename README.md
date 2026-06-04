# 校园论坛帖子推荐系统

本项目是《大数据原理与技术》课程大作业的推荐系统算法层原型。系统面向校园论坛帖子推荐，完整覆盖数据预处理、用户画像、多兴趣表示、多路召回、候选融合、DeepFM 排序、重排、离线评估和画像增量更新。

## 项目结构

```text
campus_recommender/
├── data/                    # 不再存放业务 CSV 数据
├── models/                  # 训练后的 DeepFM 和特征编码器
├── scripts/                 # 测试脚本
├── src/
│   ├── data_loader.py
│   ├── zanao_jsonl_parser.py
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

## 数据源

系统读取真实 SQLite 数据库：

```text
/Users/leslie/Documents/sysu_study/grade3/spring/Big Data Science and Technology/zanao.sqlite
```

业务数据表：

- `POSTS`：帖子基础信息与统计字段。
- `USERS`：用户基础信息。
- `COMMENTS`：评论数据。
- `USER_EVENTS`：用户行为事件。

有效板块只保留：

```text
打听求助、恋爱交友、校园趣事、兼职招聘、校园招聘、二手闲置
```

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

完整跑通主流程：

```bash
python main.py
```

主流程会完成：

1. 从 SQLite 读取真实数据。
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

- `USER_EVENTS`
- 内存用户画像
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

如果 Windows PowerShell 中看到中文乱码，通常只是终端代码页显示问题。
