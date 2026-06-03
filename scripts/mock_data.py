"""Generate normalized mock data for the campus forum recommender."""

from __future__ import annotations

import argparse
import csv
import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

ACTION_WEIGHTS = {
    "expose": 0,
    "skip": -1,
    "view": 1,
    "like": 2,
    "comment": 3,
    "collect": 4,
    "dislike": -2,
    "report": -4,
}

COLLEGE_MAJOR_MAP = {
    "计算机学院": ["软件工程", "人工智能", "数据科学", "网络工程"],
    "管理学院": ["会计学", "工商管理", "市场营销", "信息管理"],
    "外国语学院": ["英语", "日语", "翻译", "商务英语"],
    "公共卫生学院": ["临床医学", "预防医学", "护理学", "公共卫生"],
}

CAMPUS = ["南校园", "东校园", "北校园", "珠海校区", "深圳校区"]
GRADES = ["大一", "大二", "大三", "大四", "研一", "研二"]
LOCATIONS = ["教学区", "宿舍区", "食堂区", "图书馆", "校外"]
DEVICES = ["mobile", "pc", "tablet"]
SCENES = ["普通浏览", "考试周", "开学季", "毕业季", "午餐时间", "晚间学习"]

BOARD_CONFIG = {
    "二手交易": {
        "cate_id": "c001",
        "topic_type": "生活",
        "tags": ["二手交易", "教材", "自行车", "电子产品", "闲置", "租房"],
        "titles": ["出二手教材", "求购自行车", "闲置电子产品转让", "毕业季物品处理"],
        "locations": ["全校", "宿舍区", "校外"],
    },
    "问答求助": {
        "cate_id": "c002",
        "topic_type": "生活",
        "tags": ["打听求助", "校园卡", "选课", "宿舍", "课程资料", "校内通知"],
        "titles": ["求助选课问题", "打听校园卡补办流程", "宿舍问题咨询", "课程资料求分享"],
        "locations": ["全校", "教学区", "宿舍区"],
    },
    "恋爱交友": {
        "cate_id": "c003",
        "topic_type": "社交",
        "tags": ["恋爱交友", "交友", "校园墙", "活动"],
        "titles": ["周末交友活动", "校园墙投稿", "恋爱经验交流", "交友局报名"],
        "locations": ["全校", "校外"],
    },
    "校园生活": {
        "cate_id": "c004",
        "topic_type": "社交",
        "tags": ["校园生活", "宿舍", "校内通知", "校园趣事", "新生攻略"],
        "titles": ["校园生活小技巧", "宿舍收纳经验", "新生入校攻略", "本周校内通知"],
        "locations": ["全校", "宿舍区", "图书馆"],
    },
    "兼职实习": {
        "cate_id": "c005",
        "topic_type": "发展",
        "tags": ["兼职", "实习", "简历", "面试", "就业"],
        "titles": ["兼职招聘信息", "实习经验分享", "简历修改建议", "面试避坑指南"],
        "locations": ["全校", "校外", "教学区"],
    },
    "实习就业": {
        "cate_id": "c006",
        "topic_type": "发展",
        "tags": ["实习", "就业", "校招", "简历", "面试"],
        "titles": ["暑期实习内推", "校园招聘汇总", "就业经验分享", "面试题整理"],
        "locations": ["全校", "教学区", "校外"],
    },
    "学习交流": {
        "cate_id": "c007",
        "topic_type": "学习",
        "tags": ["课程资料", "学习经验", "考试", "复习资料", "选课"],
        "titles": ["期末复习资料整理", "学习小组招募", "课程重点总结", "选课经验分享"],
        "locations": ["全校", "教学区", "图书馆"],
    },
    "考研保研": {
        "cate_id": "c008",
        "topic_type": "学习",
        "tags": ["考研", "保研", "408", "复习资料", "学习经验"],
        "titles": ["考研经验分享", "保研流程答疑", "408 复习路线", "复试经验整理"],
        "locations": ["全校", "教学区", "图书馆"],
    },
    "课程评价": {
        "cate_id": "c009",
        "topic_type": "学习",
        "tags": ["课程评价", "选课", "考试", "课程资料"],
        "titles": ["课程评价汇总", "选课避坑指南", "考试经验分享", "老师上课风格讨论"],
        "locations": ["全校", "教学区"],
    },
    "食堂生活": {
        "cate_id": "c010",
        "topic_type": "生活",
        "tags": ["食堂", "拼饭", "校园生活", "优惠", "窗口测评"],
        "titles": ["食堂新窗口测评", "午餐拼饭帖", "本周食堂推荐", "隐藏菜单分享"],
        "locations": ["全校", "食堂区"],
    },
    "社团活动": {
        "cate_id": "c011",
        "topic_type": "社交",
        "tags": ["社团", "活动", "招新", "志愿服务", "交友"],
        "titles": ["社团招新公告", "周末活动报名", "志愿服务招募", "社团经验分享"],
        "locations": ["全校", "教学区"],
    },
    "竞赛科研": {
        "cate_id": "c012",
        "topic_type": "发展",
        "tags": ["竞赛", "科研项目", "组队", "论文", "实习"],
        "titles": ["竞赛组队招募", "科研项目招人", "论文阅读小组", "创新项目经验"],
        "locations": ["全校", "教学区", "图书馆"],
    },
}


@dataclass(frozen=True)
class MockConfig:
    user_count: int = 120
    post_count: int = 520
    behavior_count: int = 3600
    seed: int = 42


def join_tags(tags: list[str]) -> str:
    return "|".join(dict.fromkeys(tags))


def get_time_period(dt: datetime) -> str:
    if 5 <= dt.hour < 11:
        return "早上"
    if 11 <= dt.hour < 14:
        return "中午"
    if 14 <= dt.hour < 18:
        return "下午"
    if 18 <= dt.hour < 23:
        return "晚上"
    return "深夜"


def pick_user_tags(grade: str, college: str, location: str) -> list[str]:
    tags = []
    if grade in {"大三", "大四", "研一", "研二"}:
        tags.extend(random.sample(["考研", "保研", "实习", "就业", "竞赛"], 2))
    else:
        tags.extend(random.sample(["课程资料", "社团", "校园生活", "新生攻略", "食堂"], 2))
    if college == "计算机学院":
        tags.extend(random.sample(["408", "竞赛", "科研项目", "课程资料", "实习"], 3))
    elif college == "管理学院":
        tags.extend(random.sample(["就业", "简历", "二手交易", "社团", "兼职"], 3))
    elif college == "公共卫生学院":
        tags.extend(random.sample(["课程资料", "考研", "科研项目", "实习", "考试"], 3))
    else:
        tags.extend(random.sample(["活动", "交友", "学习经验", "实习", "社团"], 3))
    if location == "宿舍区":
        tags.append(random.choice(["二手交易", "宿舍", "校园生活"]))
    if location == "教学区":
        tags.append(random.choice(["学习交流", "课程评价", "课程资料"]))
    return random.sample(list(dict.fromkeys(tags)), k=min(6, len(set(tags))))


def generate_users(config: MockConfig, now: datetime) -> list[dict[str, object]]:
    rows = []
    colleges = list(COLLEGE_MAJOR_MAP.keys())
    for idx in range(1, config.user_count + 1):
        user_id = f"u{idx:04d}"
        grade = random.choice(GRADES)
        college = random.choice(colleges)
        major = random.choice(COLLEGE_MAJOR_MAP[college])
        default_location = random.choice(LOCATIONS)
        register_days = random.randint(1, 1000)
        is_new = int(register_days <= 30 or random.random() < 0.08)
        questionnaire_filled = int(is_new and random.random() < 0.65)
        rows.append(
            {
                "user_id": user_id,
                "nickname": f"同学{idx:04d}",
                "user_level": random.randint(1, 8),
                "user_level_title": random.choice(["萌新", "活跃同学", "资深坛友", "校园达人"]),
                "grade": grade,
                "college": college,
                "major": major,
                "campus": random.choice(CAMPUS),
                "interest_tags": join_tags(pick_user_tags(grade, college, default_location)),
                "register_time": (now - timedelta(days=register_days)).strftime("%Y-%m-%d %H:%M:%S"),
                "is_new_user": is_new,
                "questionnaire_filled": questionnaire_filled,
                "default_location": default_location,
                "status": random.choices(["normal", "blocked"], weights=[96, 4], k=1)[0],
            }
        )
    return rows


def generate_posts(config: MockConfig, users: list[dict[str, object]], now: datetime) -> list[dict[str, object]]:
    rows = []
    boards = list(BOARD_CONFIG.keys())
    for idx in range(1, config.post_count + 1):
        board = random.choice(boards)
        cfg = BOARD_CONFIG[board]
        tags = random.sample(cfg["tags"], k=random.randint(2, min(4, len(cfg["tags"]))))
        title_base = random.choice(cfg["titles"])
        title = f"{title_base}：{'/'.join(tags[:2])}"
        content = f"{title_base}，这里整理了{join_tags(tags).replace('|', '、')}相关信息，欢迎同学补充经验。"
        publish_time = now - timedelta(hours=random.randint(0, 24 * 150), minutes=random.randint(0, 59))
        post_age_hours = max((now - publish_time).total_seconds() / 3600, 0)
        has_contact = int(board in {"二手交易", "兼职实习", "实习就业"} and random.random() < 0.45)
        img_count = random.randint(0, 6) if board in {"二手交易", "食堂生活", "社团活动"} else random.randint(0, 2)
        rows.append(
            {
                "post_id": f"p{idx:05d}",
                "author_id": random.choice(users)["user_id"],
                "cate_id": cfg["cate_id"],
                "board": board,
                "title": title,
                "content": content,
                "img_count": img_count,
                "has_image": int(img_count > 0),
                "publish_time": publish_time.strftime("%Y-%m-%d %H:%M:%S"),
                "post_age_hours": round(post_age_hours, 2),
                "tags": join_tags(tags),
                "topic_type": cfg["topic_type"],
                "location_scope": random.choice(cfg["locations"]),
                "price": random.choice([0, 0, 0, 9.9, 19.9, 49.0, 99.0]) if board == "二手交易" else 0,
                "need_pay": int(board == "二手交易" and random.random() < 0.55),
                "has_contact_info": has_contact,
                "report_status": random.choices(["normal", "normal", "normal", "suspect"], weights=[95, 95, 95, 5], k=1)[0],
                "finish_status": random.choice(["open", "open", "open", "finished"]),
                "status": random.choices(["normal", "deleted", "blocked"], weights=[92, 5, 3], k=1)[0],
                "content_length": len(content),
                "keyword_list": join_tags(tags[:3]),
            }
        )
    return rows


def generate_post_stats(posts: list[dict[str, object]], now: datetime) -> list[dict[str, object]]:
    rows = []
    for idx, post in enumerate(posts, start=1):
        age = float(post["post_age_hours"])
        base = max(5, int(random.gauss(120, 60)))
        recency_bonus = int(120 / (1 + age / 24))
        view_count = max(1, base + recency_bonus + random.randint(0, 420))
        like_count = int(view_count * random.uniform(0.02, 0.18))
        comment_count = int(view_count * random.uniform(0.01, 0.12))
        collect_count = int(view_count * random.uniform(0.005, 0.08))
        dislike_count = int(view_count * random.uniform(0, 0.025))
        hot_score = view_count * 0.2 + like_count * 0.3 + comment_count * 0.3 + collect_count * 0.2
        final_hot_score = hot_score / (1 + age / 24)
        quality_score = (like_count * 0.35 + comment_count * 0.3 + collect_count * 0.35) / view_count
        rows.append(
            {
                "post_id": post["post_id"],
                "view_count": view_count,
                "like_count": like_count,
                "comment_count": comment_count,
                "collect_count": collect_count,
                "dislike_count": dislike_count,
                "hot_val": round(hot_score * random.uniform(0.9, 1.15), 4),
                "hot_rank": idx,
                "hot_score": round(hot_score, 4),
                "final_hot_score": round(final_hot_score, 4),
                "quality_score": round(min(quality_score, 1), 4),
                "update_time": now.strftime("%Y-%m-%d %H:%M:%S"),
            }
        )
    rows.sort(key=lambda row: row["final_hot_score"], reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["hot_rank"] = rank
    return rows


def generate_tags_and_post_tags(posts: list[dict[str, object]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    tag_type_lookup: dict[str, str] = {}
    for cfg in BOARD_CONFIG.values():
        for tag in cfg["tags"]:
            tag_type_lookup.setdefault(tag, cfg["topic_type"])
    tag_names = sorted(tag_type_lookup)
    tags = [
        {
            "tag_id": f"t{idx:04d}",
            "tag_name": tag,
            "tag_type": tag_type_lookup[tag],
            "view_count": random.randint(100, 8000),
            "source": random.choice(["hot_tag", "content_extract", "manual"]),
        }
        for idx, tag in enumerate(tag_names, start=1)
    ]
    tag_id_lookup = {row["tag_name"]: row["tag_id"] for row in tags}
    post_tags = []
    for post in posts:
        for tag in str(post["tags"]).split("|"):
            post_tags.append(
                {
                    "post_id": post["post_id"],
                    "tag_id": tag_id_lookup[tag],
                    "tag_name": tag,
                    "tag_weight": round(random.uniform(0.5, 1.0), 4),
                    "source": random.choice(["manual", "extract", "category"]),
                }
            )
    return tags, post_tags


def behavior_score(user: dict[str, object], post: dict[str, object], timestamp: datetime) -> float:
    user_tags = set(str(user["interest_tags"]).split("|"))
    post_tags = set(str(post["tags"]).split("|"))
    overlap = len(user_tags & post_tags) / max(len(user_tags | post_tags), 1)
    score = 0.12 + overlap * 0.65
    if user["college"] == "计算机学院" and post["board"] in {"考研保研", "竞赛科研", "学习交流", "实习就业"}:
        score += 0.18
    if get_time_period(timestamp) == "中午" and post["board"] == "食堂生活":
        score += 0.22
    if timestamp.month in {1, 6, 12} and post["board"] in {"学习交流", "课程评价", "考研保研"}:
        score += 0.18
    if user["default_location"] == "宿舍区" and post["board"] in {"二手交易", "校园生活"}:
        score += 0.16
    if user["default_location"] == "教学区" and post["board"] in {"学习交流", "课程评价"}:
        score += 0.16
    return min(score, 0.94)


def sample_action(score: float) -> tuple[str, int]:
    roll = random.random()
    if roll < 0.04:
        return "expose", 0
    if roll < score * 0.10:
        return "collect", random.randint(80, 320)
    if roll < score * 0.23:
        return "comment", random.randint(60, 260)
    if roll < score * 0.48:
        return "like", random.randint(35, 180)
    if roll < score:
        return "view", random.randint(15, 140)
    if random.random() < 0.12:
        return "dislike", random.randint(2, 18)
    return "skip", random.randint(1, 10)


def generate_behaviors(
    config: MockConfig,
    users: list[dict[str, object]],
    posts: list[dict[str, object]],
    now: datetime,
) -> list[dict[str, object]]:
    normal_posts = [post for post in posts if post["status"] == "normal"]
    rows = []
    for idx in range(1, config.behavior_count + 1):
        user = random.choice(users)
        if random.random() < 0.75:
            user_tags = set(str(user["interest_tags"]).split("|"))
            related = [post for post in normal_posts if user_tags & set(str(post["tags"]).split("|"))]
            post = random.choice(related or normal_posts)
        else:
            post = random.choice(normal_posts)

        timestamp = now - timedelta(days=random.randint(0, 120), hours=random.randint(0, 23), minutes=random.randint(0, 59))
        score = behavior_score(user, post, timestamp)
        action, dwell_time = sample_action(score)
        is_positive = int(action in {"like", "comment", "collect"} or (action == "view" and dwell_time >= 30))
        rows.append(
            {
                "behavior_id": f"b{idx:06d}",
                "user_id": user["user_id"],
                "post_id": post["post_id"],
                "action_type": action,
                "action_weight": ACTION_WEIGHTS[action],
                "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "dwell_time": dwell_time,
                "time_period": get_time_period(timestamp),
                "location": user["default_location"],
                "device_type": random.choice(DEVICES),
                "scene": random.choice(SCENES),
                "is_positive": is_positive,
                "source": "mock",
            }
        )
    rows.sort(key=lambda row: row["timestamp"])
    return rows


def generate_comments(posts: list[dict[str, object]], behaviors: list[dict[str, object]], now: datetime) -> list[dict[str, object]]:
    comment_behaviors = [row for row in behaviors if row["action_type"] == "comment"]
    rows = []
    for idx, behavior in enumerate(comment_behaviors[:900], start=1):
        post = next(post for post in posts if post["post_id"] == behavior["post_id"])
        rows.append(
            {
                "comment_id": f"cm{idx:06d}",
                "post_id": behavior["post_id"],
                "user_id": behavior["user_id"],
                "root_comment_id": "",
                "reply_comment_id": "",
                "content": "这个信息很有用，感谢分享。",
                "like_count": random.randint(0, 30),
                "dislike_count": random.randint(0, 3),
                "publish_time": behavior["timestamp"],
                "user_level_title": random.choice(["萌新", "活跃同学", "资深坛友", "校园达人"]),
                "is_author": int(behavior["user_id"] == post["author_id"]),
                "is_hide": 0,
                "status": "normal",
            }
        )
    if not rows:
        rows.append(
            {
                "comment_id": "cm000001",
                "post_id": posts[0]["post_id"],
                "user_id": posts[0]["author_id"],
                "root_comment_id": "",
                "reply_comment_id": "",
                "content": "欢迎补充。",
                "like_count": 0,
                "dislike_count": 0,
                "publish_time": now.strftime("%Y-%m-%d %H:%M:%S"),
                "user_level_title": "活跃同学",
                "is_author": 1,
                "is_hide": 0,
                "status": "normal",
            }
        )
    return rows


def generate_questionnaire(users: list[dict[str, object]], now: datetime) -> list[dict[str, object]]:
    rows = []
    for user in users:
        if int(user["questionnaire_filled"]) != 1:
            continue
        tags = str(user["interest_tags"]).split("|")
        selected_tags = random.sample(tags, k=min(3, len(tags)))
        selected_boards = []
        for board, cfg in BOARD_CONFIG.items():
            if set(selected_tags) & set(cfg["tags"]):
                selected_boards.append(board)
        rows.append(
            {
                "user_id": user["user_id"],
                "selected_tags": join_tags(selected_tags),
                "selected_boards": join_tags(selected_boards[:3] or ["校园生活"]),
                "selected_scene": random.choice(SCENES),
                "filled_time": (now - timedelta(days=random.randint(0, 30))).strftime("%Y-%m-%d %H:%M:%S"),
            }
        )
    return rows or [
        {
            "user_id": users[0]["user_id"],
            "selected_tags": users[0]["interest_tags"],
            "selected_boards": "校园生活",
            "selected_scene": "普通浏览",
            "filled_time": now.strftime("%Y-%m-%d %H:%M:%S"),
        }
    ]


def generate_user_profiles(users: list[dict[str, object]], behaviors: list[dict[str, object]], posts: list[dict[str, object]], now: datetime) -> list[dict[str, object]]:
    post_lookup = {post["post_id"]: post for post in posts}
    rows = []
    for user in users:
        user_behaviors = [row for row in behaviors if row["user_id"] == user["user_id"]]
        positives = [row for row in user_behaviors if int(row["is_positive"]) == 1]
        boards = [post_lookup[row["post_id"]]["board"] for row in positives if row["post_id"] in post_lookup]
        tags = []
        for row in positives:
            if row["post_id"] in post_lookup:
                tags.extend(str(post_lookup[row["post_id"]]["tags"]).split("|"))
        tag_counts = sorted(set(tags), key=tags.count, reverse=True)
        board_counts = sorted(set(boards), key=boards.count, reverse=True)
        total = max(len(user_behaviors), 1)
        rows.append(
            {
                "user_id": user["user_id"],
                "long_term_tags": join_tags(tag_counts[:8] or str(user["interest_tags"]).split("|")),
                "short_term_tags": join_tags(tag_counts[:5] or str(user["interest_tags"]).split("|")[:5]),
                "preferred_boards": join_tags(board_counts[:4] or ["校园生活"]),
                "active_time_period": max([row["time_period"] for row in user_behaviors] or ["晚上"], key=([row["time_period"] for row in user_behaviors] or ["晚上"]).count),
                "preferred_location": user["default_location"],
                "learning_interest_weight": round(sum(1 for tag in tags if tag in {"课程资料", "考研", "保研", "考试", "408"}) / max(len(tags), 1), 4),
                "life_interest_weight": round(sum(1 for tag in tags if tag in {"食堂", "宿舍", "二手交易", "租房", "校园生活"}) / max(len(tags), 1), 4),
                "social_interest_weight": round(sum(1 for tag in tags if tag in {"社团", "活动", "交友", "校园墙"}) / max(len(tags), 1), 4),
                "career_interest_weight": round(sum(1 for tag in tags if tag in {"实习", "就业", "竞赛", "科研项目"}) / max(len(tags), 1), 4),
                "avg_dwell_time": round(sum(int(row["dwell_time"]) for row in user_behaviors) / total, 4),
                "click_rate": round(len([row for row in user_behaviors if row["action_type"] == "view"]) / total, 4),
                "like_rate": round(len([row for row in user_behaviors if row["action_type"] == "like"]) / total, 4),
                "collect_rate": round(len([row for row in user_behaviors if row["action_type"] == "collect"]) / total, 4),
                "comment_rate": round(len([row for row in user_behaviors if row["action_type"] == "comment"]) / total, 4),
                "last_update_time": now.strftime("%Y-%m-%d %H:%M:%S"),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write for {path.name}")
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def generate_all(config: MockConfig) -> None:
    random.seed(config.seed)
    now = datetime.now().replace(microsecond=0)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    users = generate_users(config, now)
    posts = generate_posts(config, users, now)
    post_stats = generate_post_stats(posts, now)
    tags, post_tags = generate_tags_and_post_tags(posts)
    behaviors = generate_behaviors(config, users, posts, now)
    comments = generate_comments(posts, behaviors, now)
    questionnaire = generate_questionnaire(users, now)
    user_profiles = generate_user_profiles(users, behaviors, posts, now)

    write_csv(DATA_DIR / "users.csv", users)
    write_csv(DATA_DIR / "posts.csv", posts)
    write_csv(DATA_DIR / "post_stats.csv", post_stats)
    write_csv(DATA_DIR / "comments.csv", comments)
    write_csv(DATA_DIR / "tags.csv", tags)
    write_csv(DATA_DIR / "post_tags.csv", post_tags)
    write_csv(DATA_DIR / "user_behaviors.csv", behaviors)
    write_csv(DATA_DIR / "questionnaire.csv", questionnaire)
    write_csv(DATA_DIR / "user_profiles.csv", user_profiles)

    print("Mock data generated successfully.")
    print(f"users.csv: {len(users)} rows")
    print(f"posts.csv: {len(posts)} rows")
    print(f"post_stats.csv: {len(post_stats)} rows")
    print(f"comments.csv: {len(comments)} rows")
    print(f"tags.csv: {len(tags)} rows")
    print(f"post_tags.csv: {len(post_tags)} rows")
    print(f"user_behaviors.csv: {len(behaviors)} rows")
    print(f"questionnaire.csv: {len(questionnaire)} rows")
    print(f"user_profiles.csv: {len(user_profiles)} rows")


def parse_args() -> MockConfig:
    parser = argparse.ArgumentParser(description="Generate normalized mock campus forum data.")
    parser.add_argument("--users", type=int, default=120)
    parser.add_argument("--posts", type=int, default=520)
    parser.add_argument("--behaviors", type=int, default=3600)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    return MockConfig(args.users, args.posts, args.behaviors, args.seed)


if __name__ == "__main__":
    generate_all(parse_args())
