import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

N = 900
CATEGORY_MIX = {
    "Web Application": 0.35,
    "Mobile Application": 0.30,
    "AI/ML & Data": 0.15,
    "Cloud/DevOps & Infrastructure": 0.20,
}

# category -> (prob of binary flag = 1)
BINARY_PRIORS = {
    "Web Application":                 dict(web=0.95, mobile=0.15, ai_ml=0.10, cloud=0.20, data_pipe=0.20, deadline=0.55),
    "Mobile Application":              dict(web=0.25, mobile=0.95, ai_ml=0.10, cloud=0.25, data_pipe=0.15, deadline=0.55),
    "AI/ML & Data":                    dict(web=0.40, mobile=0.05, ai_ml=0.90, cloud=0.45, data_pipe=0.75, deadline=0.45),
    "Cloud/DevOps & Infrastructure":   dict(web=0.30, mobile=0.05, ai_ml=0.15, cloud=0.90, data_pipe=0.30, deadline=0.60),
}

# category -> (mean, std) for numeric features
NUMERIC_PRIORS = {
    "Web Application":                 dict(n_req=(8, 3), n_int=(2.0, 1.3), n_tech=(5, 2.0), complexity=(2.0, 1.5), text_len=(500, 150)),
    "Mobile Application":              dict(n_req=(9, 3), n_int=(3.0, 1.5), n_tech=(5, 2.0), complexity=(2.5, 1.5), text_len=(550, 150)),
    "AI/ML & Data":                    dict(n_req=(7, 3), n_int=(2.5, 1.5), n_tech=(7, 2.5), complexity=(4.5, 2.0), text_len=(650, 200)),
    "Cloud/DevOps & Infrastructure":   dict(n_req=(5, 2.5), n_int=(3.0, 1.8), n_tech=(6, 2.5), complexity=(3.5, 1.8), text_len=(450, 150)),
}

BASE_EFFORT = {
    "Web Application": 25,
    "Mobile Application": 30,
    "AI/ML & Data": 45,
    "Cloud/DevOps & Infrastructure": 35,
}

def clip_pos_int(x):
    return int(max(0, round(x)))

rows = []
categories = list(CATEGORY_MIX.keys())
probs = list(CATEGORY_MIX.values())
true_cats = rng.choice(categories, size=N, p=probs)

for true_cat in true_cats:
    bp = BINARY_PRIORS[true_cat]
    npz = NUMERIC_PRIORS[true_cat]

    requires_web = int(rng.random() < bp["web"])
    requires_mobile = int(rng.random() < bp["mobile"])
    requires_ai_ml = int(rng.random() < bp["ai_ml"])
    requires_cloud = int(rng.random() < bp["cloud"])
    requires_data_pipe = int(rng.random() < bp["data_pipe"])
    has_deadline = int(rng.random() < bp["deadline"])

    num_functional_requirements = clip_pos_int(rng.normal(*npz["n_req"]))
    num_integrations = clip_pos_int(rng.normal(*npz["n_int"]))
    num_technologies_mentioned = clip_pos_int(rng.normal(*npz["n_tech"]))
    complexity_keyword_count = clip_pos_int(rng.normal(*npz["complexity"]))
    requirement_text_length = clip_pos_int(rng.normal(*npz["text_len"]))
    deadline_days = clip_pos_int(rng.normal(60, 30)) if has_deadline else np.nan

    effort = (
        BASE_EFFORT[true_cat]
        + 1.8 * num_functional_requirements
        + 2.5 * num_integrations
        + 8.0 * requires_ai_ml
        + 6.0 * requires_cloud
        + 4.0 * requires_data_pipe
        + 1.5 * complexity_keyword_count
        + 0.02 * requirement_text_length
        + rng.normal(0, 8)
    )
    effort = max(5.0, round(effort, 1))

    rows.append(dict(
        num_functional_requirements=num_functional_requirements,
        num_integrations=num_integrations,
        requires_web_component=requires_web,
        requires_mobile_component=requires_mobile,
        requires_ai_ml_component=requires_ai_ml,
        requires_cloud_infra=requires_cloud,
        requires_data_pipeline=requires_data_pipe,
        num_technologies_mentioned=num_technologies_mentioned,
        has_deadline=has_deadline,
        deadline_days=deadline_days,
        complexity_keyword_count=complexity_keyword_count,
        requirement_text_length=requirement_text_length,
        project_category=true_cat,
        estimated_effort_person_days=effort,
    ))

df = pd.DataFrame(rows)

# --- inject label noise: flip ~7% of category labels to simulate ambiguous requests ---
noise_mask = rng.random(len(df)) < 0.07
other_cats = df["project_category"].copy()
for idx in df.index[noise_mask]:
    current = df.loc[idx, "project_category"]
    choices = [c for c in categories if c != current]
    other_cats.loc[idx] = rng.choice(choices)
df["project_category"] = other_cats

# impute missing deadline_days with median (has_deadline=0 rows)
median_deadline = df.loc[df["has_deadline"] == 1, "deadline_days"].median()
df["deadline_days"] = df["deadline_days"].fillna(median_deadline)

df.to_csv("synthetic_project_requests.csv", index=False)
print(f"Generated {len(df)} rows -> synthetic_project_requests.csv")
print("\nClass distribution:")
print(df["project_category"].value_counts())
print("\nEffort stats by category:")
print(df.groupby("project_category")["estimated_effort_person_days"].describe()[["mean", "std", "min", "max"]])
