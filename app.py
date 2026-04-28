import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import psycopg2

# ==============================================================================
# PAGE CONFIG
# ==============================================================================
st.set_page_config(
    page_title="Import Risk Classifier",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==============================================================================
# CSS
# ==============================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

* { font-family: 'Inter', sans-serif; }

/* ── Hero Banner ── */
.hero {
    background: linear-gradient(135deg, #0f3460, #16213e, #1a1a2e);
    border: 1px solid #e94560;
    border-radius: 14px;
    padding: 1.8rem 2.2rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 0 35px rgba(233,69,96,0.25);
}
.hero h1 { color: #fff; font-size: 2rem; margin: 0 0 0.3rem 0; }
.hero p  { color: #94a3b8; font-size: 0.95rem; margin: 0; }

/* ── Section Card ── */
.card {
    background: #1e1e2e;
    border: 1px solid #2d3748;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}
.card-title {
    font-size: 1rem;
    font-weight: 600;
    color: #e2e8f0;
    margin-bottom: 1rem;
    border-bottom: 1px solid #2d3748;
    padding-bottom: 0.6rem;
}

/* ── Result Boxes ── */
.result-high {
    background: linear-gradient(135deg, #7f1d1d, #991b1b);
    border: 2px solid #ef4444;
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
    box-shadow: 0 0 25px rgba(239,68,68,0.4);
}
.result-low {
    background: linear-gradient(135deg, #0c4a6e, #075985);
    border: 2px solid #0ea5e9;
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
    box-shadow: 0 0 25px rgba(14,165,233,0.4);
}
.result-label { font-size: 1.6rem; font-weight: 700; color: #fff; margin-bottom: 0.3rem; }
.result-sub   { font-size: 0.9rem; color: rgba(255,255,255,0.75); }

/* ── Stat Pill ── */
.stat-pill {
    background: #1e1e2e;
    border: 1px solid #374151;
    border-radius: 10px;
    padding: 0.9rem 0.5rem;
    text-align: center;
}
.stat-val { font-size: 1.4rem; font-weight: 700; color: #e94560; }
.stat-lbl { font-size: 0.72rem; color: #64748b; margin-top: 2px; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] > div {
    background: #0f172a;
    border-right: 1px solid #1e293b;
}
section[data-testid="stSidebar"] label { color: #94a3b8 !important; }

/* ── Primary Button ── */
div.stButton > button[kind="primary"],
div.stButton > button {
    background: linear-gradient(135deg, #e94560, #b91c1c) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    width: 100% !important;
    padding: 0.65rem !important;
    box-shadow: 0 3px 12px rgba(233,69,96,0.45) !important;
}

/* ── DB status dot ── */
.dot-green { color: #22c55e; font-size: 0.85rem; }
.dot-red   { color: #ef4444; font-size: 0.85rem; }

/* ── Progress bar color ── */
div[data-testid="stProgressBar"] > div { background: #e94560 !important; }

/* ── Tab styling ── */
button[data-baseweb="tab"] { font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# CONSTANTS
# ==============================================================================
FEATURES = [
    'Year', 'Log_Imports', 'Decade',
    'Continent_Encoded', 'Subregion_Encoded', 'Income_Group_Encoded'
]
IMPORTANCES = [0.0053, 0.0511, 0.0005, 0.1326, 0.2690, 0.5414]

CONTINENT_MAP = {
    "Africa": 0, "Asia": 1, "Europe": 2,
    "North America": 3, "Oceania": 4, "South America": 5,
}
SUBREGION_MAP = {
    "Eastern Africa": 0,   "Eastern Asia": 1,     "Eastern Europe": 2,
    "Middle Africa": 3,    "Northern Africa": 4,   "Northern America": 5,
    "Northern Europe": 6,  "South-Eastern Asia": 7, "Southern Africa": 8,
    "Southern Asia": 9,    "Southern Europe": 10,  "Western Africa": 11,
    "Western Asia": 12,    "Western Europe": 13,   "Latin America": 14,
    "Oceania": 15,         "Other": 16,
}
INCOME_MAP = {
    "Low income": 0, "Lower middle income": 1,
    "Upper middle income": 2, "High income": 3,
}

# ==============================================================================
# LOAD MODEL
# ==============================================================================


@st.cache_resource
def load_model():
    return joblib.load("best_classification_model.pkl")


model = load_model()

# ==============================================================================
# DATABASE HELPERS
# ==============================================================================


DATABASE_URL = "postgresql://myapp_db_vm7v_user:q3Gg9HX4bGwj94B5n8RzHgvs6R9Avpjk@dpg-d7o9j0nlk1mc73ck3r1g-a.ohio-postgres.render.com/myapp_db_vm7v"


def get_conn(cfg=None):
    return psycopg2.connect(
        DATABASE_URL,
        sslmode="require",
        connect_timeout=5,
    )


def test_connection(cfg):
    conn = get_conn(cfg)
    conn.close()
    return True


def init_table(cfg):
    conn = get_conn(cfg)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS import_predictions (
            id             SERIAL PRIMARY KEY,
            predicted_at   TIMESTAMP DEFAULT NOW(),
            year           INT,
            log_imports    FLOAT,
            decade         INT,
            continent      TEXT,
            subregion      TEXT,
            income_group   TEXT,
            prediction     INT,
            risk_label     TEXT,
            prob_low_risk  FLOAT,
            prob_high_risk FLOAT
        );
    """)
    conn.commit()
    cur.close()
    conn.close()


def save_to_db(cfg, data: dict):
    conn = get_conn(cfg)
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO import_predictions
          (year, log_imports, decade, continent, subregion, income_group,
           prediction, risk_label, prob_low_risk, prob_high_risk)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
    """, (
        data["year"], data["log_imports"], data["decade"],
        data["continent"], data["subregion"], data["income_group"],
        data["prediction"], data["risk_label"],
        data["prob_low_risk"], data["prob_high_risk"],
    ))
    conn.commit()
    cur.close()
    conn.close()


def save_batch_to_db(cfg, df: pd.DataFrame):
    conn = get_conn(cfg)
    cur = conn.cursor()
    rows = [(
        int(r.Year), float(r.Log_Imports), int(r.Decade),
        str(getattr(r, "Continent", "N/A")),
        str(getattr(r, "Subregion",  "N/A")),
        str(getattr(r, "Income_Group", "N/A")),
        int(r.Prediction), str(r.Risk_Label),
        float(r.Prob_Low_Risk), float(r.Prob_High_Risk),
    ) for r in df.itertuples()]
    cur.executemany("""
        INSERT INTO import_predictions
          (year, log_imports, decade, continent, subregion, income_group,
           prediction, risk_label, prob_low_risk, prob_high_risk)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
    """, rows)
    conn.commit()
    cur.close()
    conn.close()


def get_history(cfg, limit=100):
    conn = get_conn(cfg)
    df = pd.read_sql(f"""
        SELECT id, predicted_at, year, continent, subregion,
               income_group, risk_label,
               ROUND(prob_low_risk::numeric,  4) AS prob_low_risk,
               ROUND(prob_high_risk::numeric, 4) AS prob_high_risk
        FROM import_predictions
        ORDER BY predicted_at DESC
        LIMIT {limit}
    """, conn)
    conn.close()
    return df


def get_stats(cfg):
    conn = get_conn(cfg)
    cur = conn.cursor()
    cur.execute("""
        SELECT COUNT(*),
               COALESCE(SUM(prediction),0),
               COALESCE(AVG(prob_high_risk),0)
        FROM import_predictions;
    """)
    total, high, avg_p = cur.fetchone()
    cur.close()
    conn.close()
    return int(total), int(high), float(avg_p)


# ==============================================================================
# SESSION STATE
# ==============================================================================
if "db_connected" not in st.session_state:
    st.session_state.db_connected = False
if "db_cfg" not in st.session_state:
    st.session_state.db_cfg = {}
if "last_result" not in st.session_state:
    st.session_state.last_result = None

# ==============================================================================
# SIDEBAR — DB CONFIG
# ==============================================================================
with st.sidebar:
    st.markdown("## 🗄️ PostgreSQL Setup")
    st.markdown("---")

    db_host = st.text_input(
        "🌐 Host",     placeholder="db.render.com",    key="db_host")
    db_port = st.text_input("🔌 Port",     value="5432",
                            key="db_port")
    db_name = st.text_input(
        "📁 Database", placeholder="mydb",             key="db_name")
    db_user = st.text_input(
        "👤 Username", placeholder="postgres",         key="db_user")
    db_pass = st.text_input(
        "🔑 Password", type="password",                key="db_pass")

    cfg = dict(
        host=db_host, port=db_port,
        dbname=db_name, user=db_user, password=db_pass,
    )

    if st.button("🔗 Connect to Database"):
        if all([db_host, db_port, db_name, db_user, db_pass]):
            try:
                test_connection(cfg)
                init_table(cfg)
                st.session_state.db_connected = True
                st.session_state.db_cfg = cfg
                st.success("✅ Connected!")
            except Exception as e:
                st.session_state.db_connected = False
                st.error(f"❌ {e}")
        else:
            st.warning("⚠️ Sab fields fill karein.")

    st.markdown("---")
    if st.session_state.db_connected:
        st.markdown(
            '<span class="dot-green">🟢 Database Connected</span>', unsafe_allow_html=True)
    else:
        st.markdown(
            '<span class="dot-red">🔴 Database Not Connected</span>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**ℹ️ Model Info**")
    st.markdown("""
    - 🤖 GradientBoostingClassifier
    - 🌳 100 Estimators
    - 📐 Max Depth: 5
    - 🎯 Binary: Low / High Risk
    """)

# ==============================================================================
# HERO
# ==============================================================================
st.markdown("""
<div class="hero">
  <h1>🌍 Import Risk Classifier</h1>
  <p>Gradient Boosting Model &nbsp;·&nbsp; Binary Classification &nbsp;·&nbsp; PostgreSQL Integrated</p>
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# TABS
# ==============================================================================
tab_predict, tab_batch, tab_history = st.tabs([
    "🔮 Prediction",
    "📂 Batch Upload",
    "📊 DB History",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
with tab_predict:

    st.markdown("### Apni values enter karein aur Predict karein 👇")
    st.markdown("")

    # ── INPUT FORM ──────────────────────────────────────────────────────────
    with st.form(key="prediction_form"):

        col_a, col_b, col_c = st.columns(3)

        with col_a:
            st.markdown("**📅 Year**")
            year = st.number_input(
                "Year", min_value=1990, max_value=2030,
                value=2015, step=1, label_visibility="collapsed"
            )
            decade = (year // 10) * 10

            st.markdown("**📦 Log Imports**")
            log_imports = st.number_input(
                "Log Imports", min_value=0.0, max_value=25.0,
                value=12.0, step=0.1, format="%.2f",
                label_visibility="collapsed",
                help="Import value ka natural log (e.g. ln(market_value))"
            )

        with col_b:
            st.markdown("**🌐 Continent**")
            continent = st.selectbox(
                "Continent", list(CONTINENT_MAP.keys()),
                label_visibility="collapsed"
            )

            st.markdown("**📍 Subregion**")
            subregion = st.selectbox(
                "Subregion", list(SUBREGION_MAP.keys()),
                label_visibility="collapsed"
            )

        with col_c:
            st.markdown("**💰 Income Group**")
            income = st.selectbox(
                "Income Group", list(INCOME_MAP.keys()),
                label_visibility="collapsed"
            )

            st.markdown("**📆 Decade** *(auto)*")
            st.info(f"Decade = **{decade}** (Year se auto-calculate)")

        st.markdown("")
        submitted = st.form_submit_button(
            "🔮 Predict Now", use_container_width=True)

    # ── PREDICTION LOGIC ────────────────────────────────────────────────────
    if submitted:
        X = np.array([[
            year, log_imports, decade,
            CONTINENT_MAP[continent],
            SUBREGION_MAP[subregion],
            INCOME_MAP[income],
        ]])

        prediction = int(model.predict(X)[0])
        proba = model.predict_proba(X)[0]
        confidence = float(proba[prediction]) * 100
        risk_label = "⚠️ High Risk" if prediction == 1 else "✅ Low Risk"

        # Store in session so result stays visible
        st.session_state.last_result = {
            "year": year, "log_imports": log_imports, "decade": decade,
            "continent": continent, "subregion": subregion,
            "income_group": income, "prediction": prediction,
            "risk_label": risk_label,
            "prob_low_risk":  round(float(proba[0]), 4),
            "prob_high_risk": round(float(proba[1]), 4),
            "confidence": confidence,
            "proba": proba,
        }

        # Auto-save to DB if connected
        if st.session_state.db_connected:
            try:
                save_to_db(st.session_state.db_cfg,
                           st.session_state.last_result)
                st.toast("💾 Prediction PostgreSQL mein save ho gayi!", icon="✅")
            except Exception as e:
                st.toast(f"DB save failed: {e}", icon="⚠️")

    # ── SHOW RESULT ─────────────────────────────────────────────────────────
    if st.session_state.last_result:
        r = st.session_state.last_result
        st.markdown("---")
        st.markdown("### 🎯 Prediction Result")

        res_col1, res_col2 = st.columns([1, 1.4], gap="large")

        with res_col1:
            # Result Badge
            if r["prediction"] == 1:
                st.markdown(f"""
                <div class="result-high">
                  <div class="result-label">⚠️ HIGH RISK</div>
                  <div class="result-sub">Model ne Class 1 predict kiya</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-low">
                  <div class="result-label">✅ LOW RISK</div>
                  <div class="result-sub">Model ne Class 0 predict kiya</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("")

            # Confidence
            st.markdown(f"**Model Confidence: `{r['confidence']:.1f}%`**")
            st.progress(int(r["confidence"]))

            st.markdown("")

            # Stats row
            s1, s2, s3 = st.columns(3)
            s1.markdown(f"""<div class="stat-pill">
                <div class="stat-val">{r['year']}</div>
                <div class="stat-lbl">Year</div></div>""", unsafe_allow_html=True)
            s2.markdown(f"""<div class="stat-pill">
                <div class="stat-val">{r['log_imports']:.1f}</div>
                <div class="stat-lbl">Log Imports</div></div>""", unsafe_allow_html=True)
            s3.markdown(f"""<div class="stat-pill">
                <div class="stat-val">{r['decade']}</div>
                <div class="stat-lbl">Decade</div></div>""", unsafe_allow_html=True)

            st.markdown("")
            st.markdown(f"🌐 **Continent:** {r['continent']}")
            st.markdown(f"📍 **Subregion:** {r['subregion']}")
            st.markdown(f"💰 **Income Group:** {r['income_group']}")

            if st.session_state.db_connected:
                st.success("💾 DB mein save ho gayi")
            else:
                st.info("ℹ️ DB connect karein auto-save ke liye")

        with res_col2:
            # Probability Bar Chart
            st.markdown("**Class Probability Distribution**")
            fig1, ax1 = plt.subplots(figsize=(5, 3.2))
            fig1.patch.set_facecolor("#1e1e2e")
            ax1.set_facecolor("#1e1e2e")
            bars = ax1.bar(
                ["Class 0\n✅ Low Risk", "Class 1\n⚠️ High Risk"],
                r["proba"],
                color=["#0ea5e9", "#e94560"],
                width=0.45, edgecolor="none",
            )
            for bar, val in zip(bars, r["proba"]):
                ax1.text(
                    bar.get_x() + bar.get_width()/2,
                    val + 0.015,
                    f"{val:.4f}",
                    ha="center", va="bottom",
                    color="white", fontsize=12, fontweight="bold"
                )
            ax1.set_ylim(0, 1.15)
            ax1.tick_params(colors="#94a3b8", labelsize=9)
            ax1.spines[:].set_visible(False)
            ax1.yaxis.set_visible(False)
            ax1.set_title("Predicted Probabilities", color="#e2e8f0",
                          fontsize=11, pad=10)
            plt.tight_layout()
            st.pyplot(fig1, use_container_width=True)
            plt.close()

            # Feature Importance Chart
            st.markdown("**Feature Importance (Model)**")
            feat_df = pd.DataFrame({
                "Feature": FEATURES,
                "Importance": IMPORTANCES,
            }).sort_values("Importance", ascending=True)

            fig2, ax2 = plt.subplots(figsize=(5, 3.2))
            fig2.patch.set_facecolor("#1e1e2e")
            ax2.set_facecolor("#1e1e2e")
            colors = ["#334155", "#475569", "#0077b6",
                      "#00b4d8", "#e94560", "#ef4444"]
            ax2.barh(feat_df["Feature"], feat_df["Importance"],
                     color=colors, edgecolor="none", height=0.55)
            for i, (val, feat) in enumerate(zip(feat_df["Importance"], feat_df["Feature"])):
                ax2.text(val + 0.004, i, f"{val*100:.1f}%",
                         va="center", color="white", fontsize=8)
            ax2.tick_params(colors="#94a3b8", labelsize=8)
            ax2.spines[:].set_visible(False)
            ax2.xaxis.set_visible(False)
            ax2.set_title("Feature Importance",
                          color="#e2e8f0", fontsize=11, pad=10)
            plt.tight_layout()
            st.pyplot(fig2, use_container_width=True)
            plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — BATCH UPLOAD
# ══════════════════════════════════════════════════════════════════════════════
with tab_batch:
    st.markdown("### 📂 Batch CSV Prediction")
    st.info(f"**Required Columns:** `{', '.join(FEATURES)}`")

    uploaded = st.file_uploader("CSV file upload karein", type=["csv"])

    if uploaded:
        try:
            df_raw = pd.read_csv(uploaded)
            missing = [c for c in FEATURES if c not in df_raw.columns]

            if missing:
                st.error(f"❌ Ye columns nahi hain: `{missing}`")
            else:
                preds = model.predict(df_raw[FEATURES])
                probas = model.predict_proba(df_raw[FEATURES])

                df_out = df_raw.copy()
                df_out["Prediction"] = preds
                df_out["Prob_Low_Risk"] = probas[:, 0].round(4)
                df_out["Prob_High_Risk"] = probas[:, 1].round(4)
                df_out["Risk_Label"] = pd.Series(preds).map(
                    {0: "✅ Low Risk", 1: "⚠️ High Risk"}
                ).values

                # Summary metrics
                total = len(df_out)
                high = int((preds == 1).sum())
                low = total - high

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("📋 Total Records", total)
                m2.metric("⚠️ High Risk",     high)
                m3.metric("✅ Low Risk",       low)
                m4.metric("📊 High Risk %",    f"{high/total*100:.1f}%")

                st.markdown("---")
                st.markdown("**Prediction Results:**")
                st.dataframe(
                    df_out.style.apply(
                        lambda col: [
                            "background-color:#3b0f0f; color:#fca5a5"
                            if v == "⚠️ High Risk"
                            else "background-color:#0c2340; color:#7dd3fc"
                            if v == "✅ Low Risk"
                            else ""
                            for v in col
                        ] if col.name == "Risk_Label" else [""] * len(col),
                        axis=0,
                    ),
                    use_container_width=True,
                    height=350,
                )

                btn1, btn2 = st.columns(2)
                with btn1:
                    st.download_button(
                        "⬇️ Results Download (CSV)",
                        df_out.to_csv(index=False).encode(),
                        file_name="batch_predictions.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
                with btn2:
                    if st.button("💾 Sab Rows PostgreSQL mein Save Karein",
                                 use_container_width=True):
                        if st.session_state.db_connected:
                            try:
                                save_batch_to_db(
                                    st.session_state.db_cfg, df_out)
                                st.success(
                                    f"✅ {total} records DB mein save ho gaye!")
                            except Exception as e:
                                st.error(f"❌ Save failed: {e}")
                        else:
                            st.warning(
                                "⚠️ Pehle sidebar se DB connect karein.")

        except Exception as e:
            st.error(f"❌ File padhne mein error: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — DB HISTORY
# ══════════════════════════════════════════════════════════════════════════════
with tab_history:
    st.markdown("### 📊 PostgreSQL — Saved Predictions")

    if not st.session_state.db_connected:
        st.warning("⚠️ Pehle sidebar se PostgreSQL se connect karein.")
    else:
        limit = st.slider("Kitni records dikhani hain?", 10, 500, 50, 10)

        if st.button("🔄 Database se Load Karein", use_container_width=True):
            try:
                total, high, avg_p = get_stats(st.session_state.db_cfg)
                low = total - high

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Total Saved",    total)
                m2.metric("⚠️ High Risk",  high)
                m3.metric("✅ Low Risk",    low)
                m4.metric("Avg High Prob",  f"{avg_p:.4f}")

                st.markdown("---")

                hist_df = get_history(st.session_state.db_cfg, limit=limit)

                if hist_df.empty:
                    st.info("Abhi koi predictions save nahi hain.")
                else:
                    st.dataframe(
                        hist_df.style.apply(
                            lambda col: [
                                "background-color:#3b0f0f; color:#fca5a5"
                                if "High" in str(v)
                                else "background-color:#0c2340; color:#7dd3fc"
                                if "Low" in str(v)
                                else ""
                                for v in col
                            ] if col.name == "risk_label" else [""] * len(col),
                            axis=0,
                        ),
                        use_container_width=True,
                        height=380,
                    )

                    # Trend chart (agar 3+ records hain)
                    if len(hist_df) >= 3:
                        st.markdown("---")
                        st.markdown("**📈 High Risk Probability Trend**")
                        hist_plot = hist_df.sort_values("predicted_at")
                        fig3, ax3 = plt.subplots(figsize=(9, 3))
                        fig3.patch.set_facecolor("#1e1e2e")
                        ax3.set_facecolor("#1e1e2e")
                        ax3.plot(
                            range(len(hist_plot)),
                            hist_plot["prob_high_risk"].astype(float),
                            color="#e94560", linewidth=2,
                            marker="o", markersize=4,
                        )
                        ax3.fill_between(
                            range(len(hist_plot)),
                            hist_plot["prob_high_risk"].astype(float),
                            alpha=0.15, color="#e94560",
                        )
                        ax3.axhline(0.5, color="#475569", linestyle="--",
                                    linewidth=1, label="Threshold 0.5")
                        ax3.set_xlabel("Prediction Index",
                                       color="#94a3b8", fontsize=9)
                        ax3.set_ylabel("Prob High Risk",
                                       color="#94a3b8", fontsize=9)
                        ax3.tick_params(colors="#94a3b8", labelsize=8)
                        ax3.spines[:].set_visible(False)
                        ax3.legend(facecolor="#1e1e2e",
                                   labelcolor="white", fontsize=8)
                        plt.tight_layout()
                        st.pyplot(fig3, use_container_width=True)
                        plt.close()

                    st.markdown("")
                    st.download_button(
                        "⬇️ History Export (CSV)",
                        hist_df.to_csv(index=False).encode(),
                        file_name="db_history.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )

            except Exception as e:
                st.error(f"❌ DB Error: {e}")

# ==============================================================================
# FOOTER
# ==============================================================================
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#334155; font-size:0.8rem; padding:0.8rem;">
  🤖 GradientBoostingClassifier &nbsp;·&nbsp; scikit-learn &nbsp;·&nbsp;
  Streamlit &nbsp;·&nbsp; ☁️ PostgreSQL
</div>
""", unsafe_allow_html=True)
