# Dataset Build Report

Seed JSONL: `data\raw\zanao_100_posts.jsonl`

本数据集以约 100 条真实校园集市帖子为种子样本，结合校园论坛场景规则扩展构造，用于课程项目中的推荐系统算法流程验证。
数据构造过程参考了真实种子样本的板块、文本风格、帖子热度和评论结构，但扩展数据不应被视作真实线上用户行为。
真实联系方式未保存，模型字段仅保留 `has_contact_info`。

## Output Scale
- users: 500 rows
- posts: 2000 rows
- post_stats: 2000 rows
- comments: 6000 rows
- tags: 58 rows
- post_tags: 6913 rows
- user_behaviors: 30000 rows
- questionnaire: 79 rows

## Validation

- [OK] users.csv columns complete
- [OK] posts.csv columns complete
- [OK] post_stats.csv columns complete
- [OK] user_behaviors.csv columns complete
- [OK] behavior positive_rate=0.2385