"""
بیمارستان موتور جت — داشبورد نگهداری پیش‌بینانه (نسخه‌ی Streamlit)
اپلیکیشنی که آرتیفکت‌های تولیدشده در main.ipynb را بارگذاری می‌کند
و برای هر موتور و سیکل انتخاب‌شده، RUL، ریسک خرابی، ناهنجاری، و توصیه‌ی
نگهداری را نمایش می‌دهد.

نحوه‌ی اجرا (لوکال):
    cd app
    pip install -r requirements.txt
    streamlit run streamlit_app.py

ساختار موردانتظار پوشه‌ها (نسبت به این فایل):
    ../data/raw/test_FD001.txt
    ../data/raw/RUL_FD001.txt
    ../artifacts/rul_model_FD001.joblib
    ../artifacts/classification_models_FD001.joblib
    ../artifacts/anomaly_models_FD001.joblib
    ../artifacts/metadata_FD001.json

روی Streamlit Community Cloud (share.streamlit.io)، این فایل باید در ریشه‌ی
مخزن GitHub باشد و پوشه‌های data/ و artifacts/ به‌صورت مسطح کنارش قرار بگیرند؛
کد به‌طور خودکار بین دو چیدمان تشخیص می‌دهد.
"""

import json
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy.stats import rankdata

# ============================================================
# تنظیمات مسیر — به‌صورت خودکار بین چیدمان لوکال و چیدمان مسطح تشخیص می‌دهد
# ============================================================
FD = "FD001"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def resolve_dir(candidates):
    """اولین مسیر موجود از میان چند حالت ممکن را برمی‌گرداند (مطلق، بر اساس محل اسکریپت)."""
    for rel in candidates:
        full = os.path.normpath(os.path.join(SCRIPT_DIR, rel))
        if os.path.isdir(full):
            return full
    # اگر هیچ‌کدام پیدا نشد، اولین گزینه را برمی‌گرداند تا خطا پیام روشنی بدهد
    return os.path.normpath(os.path.join(SCRIPT_DIR, candidates[0]))


DATA_DIR = resolve_dir([
    "data/raw",                      # چیدمان مسطح کنار اسکریپت
    "../data/raw",                   # چیدمان لوکال قبلی (اسکریپت داخل app/)
    "../data/raw/CMAPSSData",        # چیدمان کامل پروژه با پوشه‌ی دانلودی اصلی
    "data/raw/CMAPSSData",
])
ARTIFACTS_DIR = resolve_dir([
    "artifacts",
    "../artifacts",
])

INDEX_COLS = ["engine_id", "cycle"]
SETTING_COLS = [f"op_setting_{i}" for i in range(1, 4)]
SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]
ALL_COLS = INDEX_COLS + SETTING_COLS + SENSOR_COLS


# ============================================================
# توابع بارگذاری و فیچرسازی (عیناً همان نوت‌بوک، برای تطابق استنتاج)
# ============================================================
def load_cmapss_file(path, columns=ALL_COLS):
    return pd.read_csv(path, sep=r"\s+", header=None, names=columns)


def load_rul_file(path):
    rul = pd.read_csv(path, sep=r"\s+", header=None, names=["RUL"])
    rul["engine_id"] = rul.index + 1
    return rul[["engine_id", "RUL"]]


def build_test_rul(test_df, rul_df):
    test_df = test_df.copy()
    last_cycle = test_df.groupby("engine_id")["cycle"].transform("max")
    rul_map = rul_df.set_index("engine_id")["RUL"]
    final_rul = test_df["engine_id"].map(rul_map)
    total_life_estimate = last_cycle + final_rul
    test_df["RUL"] = total_life_estimate - test_df["cycle"]
    return test_df


def add_rolling_features(df, sensor_cols, window=5):
    df = df.sort_values(["engine_id", "cycle"]).copy()
    grouped = df.groupby("engine_id")[sensor_cols]

    roll_mean = grouped.transform(lambda s: s.rolling(window, min_periods=1).mean())
    roll_std = grouped.transform(lambda s: s.rolling(window, min_periods=1).std().fillna(0))
    roll_min = grouped.transform(lambda s: s.rolling(window, min_periods=1).min())
    roll_max = grouped.transform(lambda s: s.rolling(window, min_periods=1).max())

    roll_mean.columns = [f"{c}_roll_mean_w{window}" for c in sensor_cols]
    roll_std.columns = [f"{c}_roll_std_w{window}" for c in sensor_cols]
    roll_min.columns = [f"{c}_roll_min_w{window}" for c in sensor_cols]
    roll_max.columns = [f"{c}_roll_max_w{window}" for c in sensor_cols]

    diff1 = grouped.transform(lambda s: s.diff().fillna(0))
    diff1.columns = [f"{c}_diff1" for c in sensor_cols]

    def rolling_slope(s):
        def slope(x):
            if len(x) < 2:
                return 0.0
            idx = np.arange(len(x))
            return np.polyfit(idx, x, 1)[0]
        return s.rolling(window, min_periods=2).apply(slope, raw=True).fillna(0)

    roll_slope = grouped.transform(rolling_slope)
    roll_slope.columns = [f"{c}_roll_slope_w{window}" for c in sensor_cols]

    result = pd.concat([df, roll_mean, roll_std, roll_min, roll_max, diff1, roll_slope], axis=1)
    return result


# ============================================================
# بارگذاری آرتیفکت‌ها و داده (کش می‌شود تا فقط یک‌بار اجرا شود)
# ============================================================
@st.cache_resource
def load_everything():
    rul_model = joblib.load(f"{ARTIFACTS_DIR}/rul_model_{FD}.joblib")
    clf_models = joblib.load(f"{ARTIFACTS_DIR}/classification_models_{FD}.joblib")
    anomaly_models = joblib.load(f"{ARTIFACTS_DIR}/anomaly_models_{FD}.joblib")
    with open(f"{ARTIFACTS_DIR}/metadata_{FD}.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)

    test_raw = load_cmapss_file(f"{DATA_DIR}/test_{FD}.txt")
    rul_raw = load_rul_file(f"{DATA_DIR}/RUL_{FD}.txt")
    test_df = build_test_rul(test_raw, rul_raw)
    test_feat = add_rolling_features(test_df, SENSOR_COLS, window=metadata["window"])

    feature_cols = metadata["feature_columns"]
    X_test_all = test_feat[feature_cols]
    X_test_scaled_all = anomaly_models["scaler"].transform(X_test_all)

    iso_scores = -anomaly_models["iso_forest"].score_samples(X_test_scaled_all)
    lof_scores = -anomaly_models["lof"].score_samples(X_test_scaled_all)
    svm_scores = -anomaly_models["oc_svm"].decision_function(X_test_scaled_all)

    iso_pct = rankdata(iso_scores) / len(iso_scores) * 100
    lof_pct = rankdata(lof_scores) / len(lof_scores) * 100
    svm_pct = rankdata(svm_scores) / len(svm_scores) * 100

    test_feat = test_feat.reset_index(drop=True)
    test_feat["امتیاز_ناهنجاری"] = (iso_pct + lof_pct + svm_pct) / 3
    test_feat["هشدار_لحظه‌ای"] = (
        test_feat["امتیاز_ناهنجاری"] >= np.percentile(test_feat["امتیاز_ناهنجاری"], metadata["anomaly_alert_percentile"])
    ).astype(int)

    return rul_model, clf_models, anomaly_models, metadata, test_feat


rul_model, clf_models, anomaly_models, metadata, test_feat = load_everything()

FEATURE_COLS = metadata["feature_columns"]
HORIZONS = metadata["horizons"]
DECISION_THRESHOLDS = metadata["decision_thresholds"]
CONFORMAL_MARGIN = metadata["conformal_margin"]
PERSISTENCE_RULE = metadata["anomaly_persistence_rule"]
ENGINE_IDS = sorted(test_feat["engine_id"].unique().tolist())


# ============================================================
# منطق تصمیم نگهداری
# ============================================================
def is_anomaly_persistent(engine_df, current_cycle, m=None, n=None):
    m = m or PERSISTENCE_RULE["m_alerts"]
    n = n or PERSISTENCE_RULE["n_window"]
    window_df = engine_df[engine_df["cycle"] <= current_cycle].sort_values("cycle").tail(n)
    return window_df["هشدار_لحظه‌ای"].sum() >= m


def maintenance_recommendation(rul_lower_bound, fail_prob_10, fail_prob_20, fail_prob_30, anomaly_persistent):
    if rul_lower_bound <= DECISION_THRESHOLDS["rul_critical"]:
        return "STOP", f"کران پایین بازه‌ی RUL ({rul_lower_bound:.0f} سیکل) از آستانه‌ی بحرانی کمتر است."
    if fail_prob_10 >= DECISION_THRESHOLDS["prob_10_critical"]:
        return "STOP", f"احتمال خرابی ظرف ده سیکل ({fail_prob_10:.2f}) از آستانه‌ی اعتبارسنجی‌شده بیشتر است."

    elevated_risk = (
        fail_prob_20 >= DECISION_THRESHOLDS["prob_20_elevated"]
        or fail_prob_30 >= DECISION_THRESHOLDS["prob_30_elevated"]
    )
    if elevated_risk:
        return "INSPECT", "احتمال خرابی در افق بیست یا سی سیکل از آستانه‌ی اعتبارسنجی‌شده بیشتر است."
    if anomaly_persistent:
        return "INSPECT", "ناهنجاری پایدار در سیکل‌های اخیر شناسایی شده است."

    return "CONTINUE", "کران پایین RUL مطمئن، احتمالات خرابی پایین، و امتیاز ناهنجاری پایدار است."


STATUS_COLOR = {"CONTINUE": "#2e7d32", "INSPECT": "#f9a825", "STOP": "#c62828"}
STATUS_LABEL_FA = {"CONTINUE": "ادامه‌ی کار", "INSPECT": "نیازمند بازرسی", "STOP": "توقف فوری"}


# ============================================================
# رابط کاربری Streamlit
# ============================================================
st.set_page_config(page_title="بیمارستان موتور جت", layout="wide")

st.title("بیمارستان موتور جت — داشبورد نگهداری پیش‌بینانه")
st.markdown(f"دیتاست فعال: **{FD}** — یک موتور و یک سیکل را برای مشاهده‌ی تحلیل کامل انتخاب کنید.")

col1, col2 = st.columns(2)
with col1:
    engine_id = st.selectbox("شناسه‌ی موتور", ENGINE_IDS, index=0)
with col2:
    engine_df_preview = test_feat[test_feat["engine_id"] == engine_id]
    max_cycle = int(engine_df_preview["cycle"].max())
    cycle = st.slider("سیکل", min_value=1, max_value=max_cycle, value=min(50, max_cycle))

# --- استنتاج ---
engine_df = test_feat[test_feat["engine_id"] == engine_id].sort_values("cycle").reset_index(drop=True)
available_cycles = engine_df["cycle"].tolist()
cycle = min(available_cycles, key=lambda c: abs(c - cycle))
row = engine_df[engine_df["cycle"] == cycle].iloc[0]
x_row = row[FEATURE_COLS].to_frame().T

rul_point = float(np.clip(rul_model.predict(x_row)[0], 0, None))
rul_lower = max(0.0, rul_point - CONFORMAL_MARGIN)
rul_upper = rul_point + CONFORMAL_MARGIN

fail_probs = {h: float(clf_models[h].predict_proba(x_row)[0, 1]) for h in HORIZONS}
anomaly_score = float(row["امتیاز_ناهنجاری"])
anomaly_persistent = bool(is_anomaly_persistent(engine_df, cycle))

action, reason = maintenance_recommendation(
    rul_lower, fail_probs[10], fail_probs[20], fail_probs[30], anomaly_persistent
)

# --- نوار وضعیت ---
st.markdown(
    f"<div style='padding:14px;border-radius:8px;background-color:{STATUS_COLOR[action]};"
    f"color:white;text-align:center;font-size:22px;font-weight:bold;'>"
    f"{STATUS_LABEL_FA[action]} ({action})</div>",
    unsafe_allow_html=True,
)
st.markdown(f"**دلیل محرک:** {reason}")
st.markdown("---")

# --- کارت‌ها ---
c1, c2 = st.columns(2)
with c1:
    st.subheader("کارت عمر مفید باقی‌مانده")
    st.write(f"**نقطه‌ی پیش‌بینی:** {rul_point:.0f} سیکل")
    st.write(f"**بازه‌ی اطمینان نود درصدی:** [{rul_lower:.0f}, {rul_upper:.0f}] سیکل")

with c2:
    st.subheader("کارت ریسک خرابی")
    st.write(f"**ظرف ده سیکل:** {fail_probs[10]:.1%} (آستانه: {DECISION_THRESHOLDS['prob_10_critical']:.2f})")
    st.write(f"**ظرف بیست سیکل:** {fail_probs[20]:.1%} (آستانه: {DECISION_THRESHOLDS['prob_20_elevated']:.2f})")
    st.write(f"**ظرف سی سیکل:** {fail_probs[30]:.1%} (آستانه: {DECISION_THRESHOLDS['prob_30_elevated']:.2f})")

c3, c4 = st.columns(2)
with c3:
    st.subheader("کارت ناهنجاری")
    st.write(f"**درصدک امتیاز نرمال‌شده:** {anomaly_score:.1f} از صد")
    st.write(f"**پایداری اخیر:** {'هشدار پایدار فعال است' if anomaly_persistent else 'هشدار پایدار فعال نیست'}")

with c4:
    st.subheader("متادیتای مدل")
    st.write(f"دیتاست: {FD}")
    st.write(f"نسخه‌ی مدل: {metadata['model_version']}")
    st.write(f"پنجره‌ی فیچر: {metadata['window']} سیکل")
    st.write(f"آخرین اجرای آموزش: {metadata['last_training_run'][:10]}")

st.markdown("---")

# --- نمودار تایم‌لاین موتور ---
fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
axes[0].plot(engine_df["cycle"], engine_df["sensor_4"], color="steelblue", label="سنسور ۴")
axes[0].plot(engine_df["cycle"], engine_df["sensor_11"], color="darkorange", label="سنسور ۱۱")
axes[0].axvline(cycle, color="red", linestyle="--", label="سیکل فعلی")
axes[0].set_ylabel("مقدار سنسور")
axes[0].legend(fontsize=8)
axes[0].set_title(f"تایم‌لاین موتور {engine_id}")

axes[1].plot(engine_df["cycle"], engine_df["امتیاز_ناهنجاری"], color="purple", label="امتیاز ناهنجاری")
axes[1].axhline(metadata["anomaly_alert_percentile"], color="gray", linestyle=":", label="آستانه‌ی هشدار (درصدک)")
axes[1].axvline(cycle, color="red", linestyle="--")
axes[1].set_xlabel("سیکل")
axes[1].set_ylabel("درصدک ناهنجاری")
axes[1].legend(fontsize=8)

plt.tight_layout()
st.pyplot(fig)
