"""Expand parsed Zanao seed data into a larger experimental dataset."""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    ACTION_WEIGHTS,
    ALL_BOARDS,
    BOARD_DEFAULT_TAGS,
    BOARD_TOPIC_MAP,
    DEFAULT_NUM_BEHAVIORS,
    DEFAULT_NUM_COMMENTS,
    DEFAULT_NUM_POSTS,
    DEFAULT_NUM_QUESTIONNAIRES,
    DEFAULT_NUM_USERS,
)


COLLEGE_MAJOR_MAP = {
    "计算机学院": ["软件工程", "人工智能", "数据科学", "网络工程", "信息安全"],
    "管理学院": ["会计学", "工商管理", "市场营销", "信息管理"],
    "外国语学院": ["英语", "日语", "翻译", "商务英语"],
    "公共卫生学院": ["预防医学", "公共卫生", "护理学"],
    "数学学院": ["数学与应用数学", "统计学", "信息与计算科学"],
    "电子信息学院": ["通信工程", "电子信息工程", "自动化", "物联网工程"],
    "中文系": ["汉语言文学", "新闻学", "传播学"],
    "法学院": ["法学", "知识产权"],
    "医学院": ["临床医学", "基础医学", "口腔医学"],
}
GRADES = ["大一", "大二", "大三", "大四", "研一", "研二", "研三"]
CAMPUSES = ["南校园", "东校园", "北校园", "珠海校区", "深圳校区"]
LOCATIONS = ["教学区", "宿舍区", "食堂区", "图书馆", "校外"]
DEVICES = ["mobile", "pc", "tablet"]
SCENES = ["普通浏览", "考试周", "开学季", "毕业季", "午餐时间", "晚间学习"]

BOARD_TEMPLATES = {
    "二手交易": [
        ("出二手自行车", "车况正常，适合校内通勤，{campus}附近交易，可自提。"),
        ("收二手电动车", "预算有限，想收一辆能正常骑的电动车，最好有校区牌。"),
        ("出考研资料", "资料保存完整，适合复习使用，价格可小刀。"),
        ("毕业季出闲置", "整理宿舍出一些闲置物品，优先校内自提。"),
    ],
    "问答求助": [
        ("求问南校哪里可以自习", "最近想找安静地方复习，想问问大家有什么推荐。"),
        ("求问校园卡补办流程", "校园卡丢了，有没有同学知道补办需要带什么材料。"),
        ("宿舍维修一般多久来", "已经报修了，想问问大家一般多久能处理。"),
        ("求推荐附近打印店", "需要打印课程资料，想找价格合适的位置。"),
    ],
    "校园生活": [
        ("今天食堂这个菜怎么样", "中午路过看到很多人排队，想问问味道如何。"),
        ("有人一起去操场跑步吗", "晚上想找搭子慢跑，节奏轻松就行。"),
        ("吐槽一下今天的校车", "等车时间有点久，大家最近有没有类似情况。"),
        ("宿舍区附近有什么好吃的", "想换换口味，求推荐附近小店。"),
    ],
    "恋爱交友": [
        ("找学习搭子", "希望互相监督学习，地点可以在图书馆。"),
        ("找健身搭子", "每周两三次，强度不用太高。"),
        ("周末有人一起打羽毛球吗", "新手友好，主要想活动一下。"),
        ("毕业照拍照搭子", "想找同学一起拍照，时间可以商量。"),
    ],
    "学习交流": [
        ("数据结构期末复习资料整理", "整理了一些重点和往年题型，欢迎补充。"),
        ("求操作系统实验报告参考", "实验有几个地方卡住了，想交流一下思路。"),
        ("这门课给分怎么样", "准备选课，想听听上过的同学评价。"),
        ("有没有学习经验分享", "想调整一下复习节奏，欢迎大家聊聊方法。"),
    ],
    "考研保研": [
        ("408 考研经验分享", "按阶段整理了复习计划和资料选择。"),
        ("有没有保研经验分享", "想了解材料准备和面试注意事项。"),
        ("复试经验整理", "把最近问到的问题和准备过程记录一下。"),
        ("考研学习搭子", "希望每天互相打卡，保持节奏。"),
    ],
    "课程评价": [
        ("这门课给分怎么样", "想问问作业量和期末难度。"),
        ("选课避坑指南", "整理一些个人体验，欢迎补充不同看法。"),
        ("期末重点有吗", "复习范围有点大，想交流一下重点。"),
        ("老师上课风格讨论", "准备下学期选课，想了解课堂节奏。"),
    ],
    "食堂生活": [
        ("食堂新窗口测评", "今天试了一下，味道和分量都还可以。"),
        ("午餐拼饭帖", "中午想找人一起吃饭，地点在食堂区。"),
        ("隐藏菜单分享", "最近发现一个不错的搭配，价格也还行。"),
        ("这个窗口确实不错", "排队人不少，不过出餐挺快。"),
    ],
    "社团活动": [
        ("社团活动招人", "活动时间比较灵活，欢迎感兴趣的同学加入。"),
        ("周末活动报名", "本周末有线下活动，人数有限。"),
        ("志愿服务招募", "需要几位同学协助现场秩序。"),
        ("社团招新公告", "面向所有年级，零基础也可以参加。"),
    ],
    "竞赛科研": [
        ("竞赛组队招募", "方向初步确定，缺会写代码和做展示的同学。"),
        ("科研项目招人", "每周有固定讨论，适合想了解科研的同学。"),
        ("论文阅读小组", "一起读论文，互相讲解重点。"),
        ("创新项目经验", "分享一下申报和结题过程。"),
    ],
    "实习就业": [
        ("暑期实习内推", "岗位适合大三大四，有相关项目经验更好。"),
        ("宣讲会信息分享", "时间地点已经整理，感兴趣可以看看。"),
        ("求简历修改建议", "准备投递实习，希望大家给点建议。"),
        ("有没有同学了解这个岗位", "想问问工作内容和面试形式。"),
    ],
    "兼职实习": [
        ("家教兼职招人", "时间可协调，地点离学校不远。"),
        ("短期兼职信息", "适合课余时间，有意可以留言。"),
        ("兼职避坑提醒", "整理一些常见注意事项。"),
        ("实习信息交流", "看到一些岗位，想和大家讨论一下。"),
    ],
    "失物招领": [
        ("捡到校园卡", "在教学楼附近捡到，失主可以联系认领。"),
        ("寻找钥匙", "可能丢在宿舍区或食堂附近。"),
        ("证件失物招领", "在图书馆座位旁看到。"),
        ("请帮忙找背包", "今天下午可能落在自习室。"),
    ],
}


@dataclass
class ExpansionConfig:
    num_posts: int = DEFAULT_NUM_POSTS
    num_users: int = DEFAULT_NUM_USERS
    num_behaviors: int = DEFAULT_NUM_BEHAVIORS
    num_comments: int = DEFAULT_NUM_COMMENTS
    num_questionnaires: int = DEFAULT_NUM_QUESTIONNAIRES
    seed: int = 2026


def join_tags(tags: list[str]) -> str:
    return "|".join(dict.fromkeys([tag for tag in tags if tag]))


def topic_for_board(board: str) -> str:
    return BOARD_TOPIC_MAP.get(board, (board, "生活"))[1]


def pick_user_tags(grade: str, college: str) -> list[str]:
    tags: list[str] = []
    if grade == "大一":
        tags += ["社团", "选课", "新生攻略", "校园生活"]
    elif grade == "大二":
        tags += ["课程资料", "竞赛", "社团", "二手交易"]
    elif grade == "大三":
        tags += ["考研", "保研", "实习", "竞赛"]
    elif grade == "大四":
        tags += ["就业", "租房", "二手交易", "毕业"]
    else:
        tags += ["科研", "实习", "招聘", "租房"]
    if college == "计算机学院":
        tags += ["编程", "竞赛", "课程资料", "实习", "考研"]
    elif college == "管理学院":
        tags += ["就业", "简历", "兼职", "社团"]
    elif college == "医学院":
        tags += ["考研", "科研", "课程资料", "实习"]
    return random.sample(list(dict.fromkeys(tags)), k=min(6, len(set(tags))))


def complete_seed_users(seed_users: pd.DataFrame, target_count: int, now: pd.Timestamp) -> pd.DataFrame:
    rows = seed_users.to_dict("records") if not seed_users.empty else []
    colleges = list(COLLEGE_MAJOR_MAP.keys())
    for row in rows:
        college = random.choice(colleges)
        grade = random.choice(GRADES)
        row["grade"] = row.get("grade") or grade
        row["college"] = row.get("college") or college
        row["major"] = row.get("major") or random.choice(COLLEGE_MAJOR_MAP[college])
        row["campus"] = row.get("campus") or random.choice(CAMPUSES)
        row["default_location"] = row.get("default_location") or random.choice(LOCATIONS)
        row["interest_tags"] = row.get("interest_tags") or join_tags(pick_user_tags(row["grade"], row["college"]))
        row["status"] = "normal"
    existing_ids = {str(row["user_id"]) for row in rows}
    while len(rows) < target_count:
        idx = len(rows) + 1
        user_id = f"u_ext_{idx:05d}"
        if user_id in existing_ids:
            continue
        grade = random.choice(GRADES)
        college = random.choice(colleges)
        register_days = random.randint(1, 1200)
        is_new = int(random.random() < random.uniform(0.15, 0.25))
        rows.append(
            {
                "user_id": user_id,
                "nickname": f"校园用户{idx:05d}",
                "user_level": random.randint(1, 8),
                "user_level_title": random.choice(["普通用户", "活跃同学", "资深坛友", "校园达人"]),
                "grade": grade,
                "college": college,
                "major": random.choice(COLLEGE_MAJOR_MAP[college]),
                "campus": random.choice(CAMPUSES),
                "interest_tags": join_tags(pick_user_tags(grade, college)),
                "register_time": (now - pd.Timedelta(days=register_days)).strftime("%Y-%m-%d %H:%M:%S"),
                "is_new_user": is_new,
                "questionnaire_filled": int(is_new and random.random() < 0.65),
                "default_location": random.choice(LOCATIONS),
                "status": "normal",
            }
        )
    return pd.DataFrame(rows).drop_duplicates("user_id", keep="last").head(target_count)


def generate_post_from_board(post_id: str, author_id: str, board: str, now: pd.Timestamp) -> dict[str, object]:
    title, content_template = random.choice(BOARD_TEMPLATES.get(board, BOARD_TEMPLATES["校园生活"]))
    campus = random.choice(CAMPUSES)
    tags = random.sample(BOARD_DEFAULT_TAGS.get(board, ["校园生活"]), k=min(random.randint(2, 5), len(BOARD_DEFAULT_TAGS.get(board, ["校园生活"]))))
    publish_time = now - pd.Timedelta(days=random.randint(0, 180), hours=random.randint(0, 23), minutes=random.randint(0, 59))
    post_age_hours = max((now - publish_time).total_seconds() / 3600, 0)
    content = content_template.format(campus=campus)
    img_count = random.choice([0, 0, 0, 1, 1, 2, 3, 5]) if board in {"二手交易", "食堂生活", "社团活动"} else random.choice([0, 0, 1])
    topic_type = topic_for_board(board)
    return {
        "post_id": post_id,
        "author_id": author_id,
        "cate_id": f"ext_{ALL_BOARDS.index(board) + 1:03d}" if board in ALL_BOARDS else "ext_000",
        "board": board,
        "title": title,
        "content": content,
        "img_count": img_count,
        "has_image": int(img_count > 0),
        "publish_time": publish_time.strftime("%Y-%m-%d %H:%M:%S"),
        "post_age_hours": round(post_age_hours, 4),
        "tags": join_tags(tags),
        "topic_type": topic_type,
        "location_scope": random.choice(["全校", "教学区", "宿舍区", "食堂区", "图书馆", "校外"]),
        "price": round(random.choice([0, 0, 0, 10, 20, 50, 90, 150]), 2) if board == "二手交易" else 0,
        "need_pay": int(board == "二手交易" and random.random() < 0.45),
        "has_contact_info": int(board in {"二手交易", "兼职实习", "实习就业"} and random.random() < 0.35),
        "report_status": "0",
        "finish_status": "10",
        "status": "normal",
        "content_length": len(content),
        "keyword_list": join_tags(tags[:5]),
    }


def expand_posts(seed_posts: pd.DataFrame, users: pd.DataFrame, target_count: int, now: pd.Timestamp) -> pd.DataFrame:
    rows = seed_posts.to_dict("records") if not seed_posts.empty else []
    board_values = seed_posts["board"].tolist() if not seed_posts.empty and "board" in seed_posts.columns else []
    board_pool = board_values + ALL_BOARDS * 3
    existing_ids = {str(row["post_id"]) for row in rows}
    while len(rows) < target_count:
        idx = len(rows) + 1
        post_id = f"p_ext_{idx:06d}"
        if post_id in existing_ids:
            continue
        board = random.choice(board_pool)
        author_id = str(users.sample(1).iloc[0]["user_id"])
        rows.append(generate_post_from_board(post_id, author_id, board, now))
    return pd.DataFrame(rows).drop_duplicates("post_id", keep="last").head(target_count)


def generate_post_stats(posts: pd.DataFrame, seed_stats: pd.DataFrame, now: pd.Timestamp) -> pd.DataFrame:
    seed_views = seed_stats["view_count"].tolist() if not seed_stats.empty and "view_count" in seed_stats.columns else [20, 50, 100]
    rows = []
    for post in posts.itertuples(index=False):
        if not str(post.post_id).startswith("p_ext_"):
            matched = seed_stats[seed_stats["post_id"].astype(str).eq(str(post.post_id))] if not seed_stats.empty else pd.DataFrame()
            if not matched.empty:
                rows.append(matched.iloc[0].to_dict())
                continue
        base_view = max(1, int(random.choice(seed_views) * random.uniform(0.7, 3.0) + random.randint(0, 200)))
        like_count = int(base_view * random.uniform(0.01, 0.18))
        comment_count = int(base_view * random.uniform(0.005, 0.12))
        collect_count = int(base_view * random.uniform(0.002, 0.08))
        dislike_count = int(base_view * random.uniform(0, 0.02))
        hot_score = base_view * 0.2 + like_count * 0.3 + comment_count * 0.3 + collect_count * 0.2
        final_hot_score = hot_score / (1 + float(post.post_age_hours) / 24)
        quality_score = (like_count * 0.35 + collect_count * 0.35 + comment_count * 0.30 - dislike_count * 0.20) / max(base_view, 1)
        rows.append(
            {
                "post_id": post.post_id,
                "view_count": base_view,
                "like_count": like_count,
                "comment_count": comment_count,
                "collect_count": collect_count,
                "dislike_count": dislike_count,
                "hot_val": round(hot_score, 4),
                "hot_rank": 0,
                "hot_score": round(hot_score, 4),
                "final_hot_score": round(final_hot_score, 4),
                "quality_score": round(max(quality_score, 0), 4),
                "update_time": now.strftime("%Y-%m-%d %H:%M:%S"),
            }
        )
    stats = pd.DataFrame(rows).drop_duplicates("post_id", keep="last")
    stats = stats.sort_values("final_hot_score", ascending=False).reset_index(drop=True)
    stats["hot_rank"] = range(1, len(stats) + 1)
    return stats


def match_score(user: pd.Series, post: pd.Series, timestamp: pd.Timestamp) -> float:
    user_tags = set(str(user["interest_tags"]).split("|"))
    post_tags = set(str(post["tags"]).split("|"))
    score = 0.12 + len(user_tags & post_tags) / max(len(user_tags | post_tags), 1) * 0.65
    if user["default_location"] == "食堂区" and post["board"] == "食堂生活":
        score += 0.2
    if user["default_location"] == "教学区" and post["board"] in {"学习交流", "课程评价"}:
        score += 0.18
    if user["default_location"] == "宿舍区" and post["board"] in {"二手交易", "校园生活"}:
        score += 0.18
    if user["default_location"] == "图书馆" and post["board"] in {"学习交流", "考研保研"}:
        score += 0.18
    if 11 <= timestamp.hour < 14 and post["board"] in {"食堂生活", "校园生活", "二手交易"}:
        score += 0.15
    if 18 <= timestamp.hour < 23 and post["board"] in {"学习交流", "考研保研", "课程评价"}:
        score += 0.15
    if timestamp.month in {1, 6, 12} and post["board"] in {"学习交流", "课程评价", "考研保研"}:
        score += 0.16
    if timestamp.month in {8, 9} and post["board"] in {"社团活动", "课程评价"}:
        score += 0.14
    if timestamp.month in {5, 6} and post["board"] in {"实习就业", "兼职实习", "二手交易"}:
        score += 0.14
    return min(score, 0.95)


def get_time_period(ts: pd.Timestamp) -> str:
    if 5 <= ts.hour < 11:
        return "早上"
    if 11 <= ts.hour < 14:
        return "中午"
    if 14 <= ts.hour < 18:
        return "下午"
    if 18 <= ts.hour < 23:
        return "晚上"
    return "深夜"


def generate_behaviors(users: pd.DataFrame, posts: pd.DataFrame, stats: pd.DataFrame, target_count: int, now: pd.Timestamp) -> pd.DataFrame:
    post_stats = posts.merge(stats[["post_id", "final_hot_score"]], on="post_id", how="left")
    hot_weights = (post_stats["final_hot_score"].fillna(0) + 1).to_numpy()
    rows = []
    for idx in range(1, target_count + 1):
        user = users.sample(1).iloc[0]
        if random.random() < 0.65:
            user_tags = set(str(user["interest_tags"]).split("|"))
            candidates = post_stats[post_stats["tags"].apply(lambda v: bool(user_tags & set(str(v).split("|"))))]
            post = candidates.sample(1).iloc[0] if not candidates.empty else post_stats.sample(1, weights=hot_weights).iloc[0]
        else:
            post = post_stats.sample(1, weights=hot_weights).iloc[0]
        timestamp = now - pd.Timedelta(days=random.randint(0, 180), hours=random.randint(0, 23), minutes=random.randint(0, 59))
        score = match_score(user, post, timestamp)
        roll = random.random()
        if roll < score * 0.08:
            action = "collect"
            dwell = random.randint(50, 260)
        elif roll < score * 0.18:
            action = "comment"
            dwell = random.randint(40, 220)
        elif roll < score * 0.38:
            action = "like"
            dwell = random.randint(20, 160)
        elif roll < score:
            action = "view"
            dwell = random.randint(10, 140)
        elif random.random() < 0.04:
            action = "dislike"
            dwell = random.randint(2, 15)
        else:
            action = "skip"
            dwell = random.randint(1, 8)
        if random.random() < 0.08:
            action = "expose"
            dwell = 0
        is_positive = int(action in {"like", "comment", "collect"} or (action == "view" and dwell >= 10))
        if action == "view" and dwell < 5:
            is_positive = 0
        rows.append(
            {
                "behavior_id": f"b{idx:08d}",
                "user_id": user["user_id"],
                "post_id": post["post_id"],
                "action_type": action,
                "action_weight": ACTION_WEIGHTS[action],
                "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "dwell_time": dwell,
                "time_period": get_time_period(timestamp),
                "location": user["default_location"],
                "device_type": random.choice(["mobile", "pc", "tablet"]),
                "scene": random.choice(SCENES),
                "is_positive": is_positive,
                "source": "expanded",
            }
        )
    return pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)


def generate_comments(posts: pd.DataFrame, users: pd.DataFrame, seed_comments: pd.DataFrame, target_count: int, now: pd.Timestamp) -> pd.DataFrame:
    rows = seed_comments.to_dict("records") if not seed_comments.empty else []
    comment_templates = {
        "学习": ["感谢分享", "求资料", "期末重点有吗", "蹲一个复习整理"],
        "生活": ["还在吗", "多少钱", "可以自提吗", "在哪个校区交易", "这个窗口确实不错"],
        "社交": ["我也想参加", "时间合适的话可以", "蹲一个搭子", "活动还有名额吗"],
        "发展": ["需要什么技术栈", "可以内推吗", "简历要怎么投", "这个岗位适合大三吗"],
    }
    while len(rows) < target_count:
        idx = len(rows) + 1
        post = posts.sample(1).iloc[0]
        user = users.sample(1).iloc[0]
        topic = post["topic_type"]
        rows.append(
            {
                "comment_id": f"c_ext_{idx:08d}",
                "post_id": post["post_id"],
                "user_id": user["user_id"],
                "root_comment_id": "",
                "reply_comment_id": "",
                "content": random.choice(comment_templates.get(topic, comment_templates["生活"])),
                "like_count": random.randint(0, 30),
                "dislike_count": random.randint(0, 3),
                "publish_time": (now - pd.Timedelta(days=random.randint(0, 120), minutes=random.randint(0, 1440))).strftime("%Y-%m-%d %H:%M:%S"),
                "user_level_title": user["user_level_title"],
                "is_author": int(user["user_id"] == post["author_id"]),
                "is_hide": 0,
                "status": "normal",
            }
        )
    return pd.DataFrame(rows).drop_duplicates("comment_id", keep="last").head(target_count)


def generate_tags_and_post_tags(posts: pd.DataFrame, seed_tags: pd.DataFrame, seed_post_tags: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    tag_map = {}
    if not seed_tags.empty:
        for row in seed_tags.itertuples(index=False):
            tag_map[str(row.tag_name)] = {"tag_id": str(row.tag_id), "tag_name": str(row.tag_name), "tag_type": str(row.tag_type), "view_count": int(getattr(row, "view_count", 0)), "source": str(getattr(row, "source", "content_extract"))}
    post_tag_rows = seed_post_tags.to_dict("records") if not seed_post_tags.empty else []
    for post in posts.itertuples(index=False):
        for tag in str(post.tags).split("|")[:5]:
            if not tag:
                continue
            if tag not in tag_map:
                tag_map[tag] = {
                    "tag_id": f"tag_{len(tag_map) + 1:04d}",
                    "tag_name": tag,
                    "tag_type": post.topic_type,
                    "view_count": random.randint(10, 5000),
                    "source": "content_extract",
                }
            post_tag_rows.append(
                {
                    "post_id": post.post_id,
                    "tag_id": tag_map[tag]["tag_id"],
                    "tag_name": tag,
                    "tag_weight": round(random.uniform(0.5, 1.0), 4),
                    "source": "extract",
                }
            )
    return pd.DataFrame(tag_map.values()), pd.DataFrame(post_tag_rows).drop_duplicates(["post_id", "tag_name"], keep="last")


def generate_questionnaire(users: pd.DataFrame, target_count: int, now: pd.Timestamp) -> pd.DataFrame:
    new_users = users[users["is_new_user"].astype(int).eq(1)]
    selected = new_users.head(target_count) if not new_users.empty else users.head(target_count)
    rows = []
    for user in selected.itertuples(index=False):
        tags = str(user.interest_tags).split("|")
        selected_tags = random.sample(tags, k=min(3, len(tags))) if tags else ["校园生活"]
        boards = [board for board, board_tags in BOARD_DEFAULT_TAGS.items() if set(selected_tags) & set(board_tags)]
        rows.append(
            {
                "user_id": user.user_id,
                "selected_tags": join_tags(selected_tags),
                "selected_boards": join_tags(boards[:3] or ["校园生活"]),
                "selected_scene": random.choice(SCENES),
                "filled_time": (now - pd.Timedelta(days=random.randint(0, 30))).strftime("%Y-%m-%d %H:%M:%S"),
            }
        )
    return pd.DataFrame(rows)


def expand_seed_data(parsed: dict[str, pd.DataFrame], config: ExpansionConfig) -> dict[str, pd.DataFrame]:
    random.seed(config.seed)
    np.random.seed(config.seed)
    now = pd.Timestamp.now()
    users = complete_seed_users(parsed["raw_users_df"], config.num_users, now)
    posts = expand_posts(parsed["raw_posts_df"], users, config.num_posts, now)
    post_stats = generate_post_stats(posts, parsed["raw_post_stats_df"], now)
    tags, post_tags = generate_tags_and_post_tags(posts, parsed["raw_tags_df"], parsed["raw_post_tags_df"])
    behaviors = generate_behaviors(users, posts, post_stats, config.num_behaviors, now)
    comments = generate_comments(posts, users, parsed["raw_comments_df"], config.num_comments, now)
    questionnaire = generate_questionnaire(users, config.num_questionnaires, now)
    return {
        "users": users,
        "posts": posts,
        "post_stats": post_stats,
        "comments": comments,
        "tags": tags,
        "post_tags": post_tags,
        "user_behaviors": behaviors,
        "questionnaire": questionnaire,
    }
