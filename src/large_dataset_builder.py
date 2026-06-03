"""Build a standard experimental dataset from Zanao JSONL seed posts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import OUTPUTS_DIR
from src.data_loader import save_dataframe
from src.data_validator import write_validation_report
from src.feature_engineering import build_training_features, fit_post_text_vectors, save_training_samples
from src.preprocess import preprocess_all
from src.seed_data_expander import ExpansionConfig, expand_seed_data
from src.user_profile import build_user_profile_table, build_user_profiles, save_user_profiles
from src.zanao_jsonl_parser import parse_zanao_jsonl


def save_standard_tables(tables: dict[str, pd.DataFrame], output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    filename_map = {
        "users": "users.csv",
        "posts": "posts.csv",
        "post_stats": "post_stats.csv",
        "comments": "comments.csv",
        "tags": "tags.csv",
        "post_tags": "post_tags.csv",
        "user_behaviors": "user_behaviors.csv",
        "questionnaire": "questionnaire.csv",
    }
    for key, filename in filename_map.items():
        save_dataframe(tables[key], output_dir / filename)


def build_training_samples_with_existing_pipeline(output_dir: str | Path) -> None:
    """Reuse existing preprocessing and feature engineering modules."""

    from src.data_loader import load_all_data

    data = load_all_data(output_dir)
    result = preprocess_all(data)
    profile_table = build_user_profile_table(result.users, result.posts, result.behaviors)
    save_user_profiles(profile_table, Path(output_dir) / "user_profiles.csv")
    text_bundle = fit_post_text_vectors(result.posts)
    profiles = build_user_profiles(
        users=result.users,
        posts=result.posts,
        behaviors=result.behaviors,
        user_profiles_df=profile_table,
        post_id_to_index=text_bundle.post_id_to_index,
        text_matrix=text_bundle.post_text_matrix,
    )
    training_features = build_training_features(
        samples=result.samples,
        users=result.users,
        posts=result.posts,
        behaviors=result.behaviors,
        user_profiles_df=profile_table,
        profiles=profiles,
        text_bundle=text_bundle,
    )
    save_training_samples(training_features, str(Path(output_dir) / "training_samples.csv"))


def write_dataset_build_report(
    output_path: str | Path,
    input_path: str | Path,
    tables: dict[str, pd.DataFrame],
    validation_messages: list[str],
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Dataset Build Report",
        "",
        f"Seed JSONL: `{input_path}`",
        "",
        "本数据集以约 100 条真实校园集市帖子为种子样本，结合校园论坛场景规则扩展构造，用于课程项目中的推荐系统算法流程验证。",
        "数据构造过程参考了真实种子样本的板块、文本风格、帖子热度和评论结构，但扩展数据不应被视作真实线上用户行为。",
        "真实联系方式未保存，模型字段仅保留 `has_contact_info`。",
        "",
        "## Output Scale",
    ]
    for key, df in tables.items():
        lines.append(f"- {key}: {len(df)} rows")
    lines.extend(["", "## Validation", ""])
    lines.extend(f"- {message}" for message in validation_messages)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def build_large_dataset(
    input_path: str | Path,
    output_dir: str | Path,
    config: ExpansionConfig,
    build_training_samples: bool = True,
) -> dict[str, pd.DataFrame]:
    parsed = parse_zanao_jsonl(input_path)
    tables = expand_seed_data(parsed, config)
    save_standard_tables(tables, output_dir)
    if build_training_samples:
        build_training_samples_with_existing_pipeline(output_dir)
    validation_path = OUTPUTS_DIR / "dataset_validation_report.md"
    ok, messages = write_validation_report(output_dir, validation_path)
    report_path = OUTPUTS_DIR / "dataset_build_report.md"
    write_dataset_build_report(report_path, input_path, tables, messages)
    print(f"dataset saved to: {output_dir}")
    print(f"validation report: {validation_path}")
    print(f"build report: {report_path}")
    print(f"validation status: {'PASS' if ok else 'CHECK REQUIRED'}")
    return tables
