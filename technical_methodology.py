import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ─────────────────────────────────────────────
#  HELPER: Styled Section Header
# ─────────────────────────────────────────────
def section_header(icon: str, title: str, subtitle: str = ""):
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            border-left: 5px solid #e94560;
            border-radius: 8px;
            padding: 1.2rem 1.5rem;
            margin: 1.5rem 0 1rem 0;
        ">
            <h2 style="color:#ffffff; margin:0; font-size:1.4rem;">
                {icon} &nbsp; {title}
            </h2>
            {"<p style='color:#a0aec0; margin:0.3rem 0 0 0; font-size:0.9rem;'>" + subtitle + "</p>" if subtitle else ""}
        </div>
        """,
        unsafe_allow_html=True,
    )


def pillar_card(number: str, title: str, description: str, color: str):
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, {color}22 0%, {color}08 100%);
            border: 1px solid {color}55;
            border-radius: 12px;
            padding: 1.2rem 1.4rem;
            text-align: center;
            height: 100%;
        ">
            <div style="font-size:2rem; font-weight:900; color:{color}; margin-bottom:0.3rem;">{number}</div>
            <div style="font-weight:700; font-size:1rem; color:#ffffff; margin-bottom:0.5rem;">{title}</div>
            <div style="color:#a0aec0; font-size:0.82rem; line-height:1.5;">{description}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def info_box(text: str, color: str = "#3182ce"):
    st.markdown(
        f"""<div style="
            background:{color}18;
            border-left:4px solid {color};
            border-radius:6px;
            padding:0.8rem 1rem;
            margin:0.6rem 0;
            color:#e2e8f0;
            font-size:0.88rem;
            line-height:1.6;
        ">{text}</div>""",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────
#  PILLAR 1 DEMO: Live Disparate Impact Calculator
# ─────────────────────────────────────────────
def _demo_di_calculator():
    st.markdown("#### 🧮 Live Disparate Impact Calculator")
    info_box(
        "Interact with the sliders below to see how the Four-Fifths Rule and "
        "Statistical Parity Difference respond to real numbers — exactly as FairFrame computes them.",
        "#805ad5",
    )

    col1, col2 = st.columns(2)
    with col1:
        rate_priv = st.slider(
            "Selection Rate — Privileged Group (%)",
            min_value=10, max_value=100, value=75, step=1,
            key="meth_rate_priv",
        )
    with col2:
        rate_unpriv = st.slider(
            "Selection Rate — Unprivileged Group (%)",
            min_value=1, max_value=100, value=45, step=1,
            key="meth_rate_unpriv",
        )

    p_priv   = rate_priv   / 100.0
    p_unpriv = rate_unpriv / 100.0

    di  = p_unpriv / p_priv if p_priv > 0 else 0.0
    spd = p_unpriv - p_priv
    gap = abs(spd) * 100

    # Risk classification (mirrors bias_detector.py logic)
    if gap > 15:
        risk_label, risk_color, risk_icon = "HIGH RISK",     "#e53e3e", "🔴"
    elif gap > 5:
        risk_label, risk_color, risk_icon = "MODERATE RISK", "#d69e2e", "🟡"
    else:
        risk_label, risk_color, risk_icon = "LOW RISK",      "#38a169", "🟢"

    eeoc_pass   = di >= 0.80
    eeoc_label  = "✅ PASSES (≥ 0.80)" if eeoc_pass else "❌ FAILS (< 0.80)"
    eeoc_color  = "#38a169" if eeoc_pass else "#e53e3e"

    # Metric cards
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Disparate Impact (DI)", f"{di:.3f}", help="Ratio of unprivileged to privileged selection rate")
    m2.metric("Stat. Parity Diff (SPD)", f"{spd:+.3f}", help="Difference in positive outcome probabilities")
    m3.metric("Fairness Gap", f"{gap:.1f}%", help="Absolute percentage difference between groups")
    m4.metric("Risk Level", f"{risk_icon} {risk_label}", help="Mirrors FairFrame's risk classification thresholds")

    st.markdown(
        f"<div style='background:{eeoc_color}22; border:1px solid {eeoc_color}88; border-radius:8px;"
        f"padding:0.7rem 1.2rem; margin-top:0.5rem; color:{eeoc_color}; font-weight:600;'>"
        f"EEOC Four-Fifths Rule: &nbsp; {eeoc_label} &nbsp;|&nbsp; DI = {di:.3f}"
        f"</div>",
        unsafe_allow_html=True,
    )

    # Visual bar chart
    fig, ax = plt.subplots(figsize=(7, 2.5))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")
    bars = ax.barh(
        ["Privileged Group", "Unprivileged Group"],
        [p_priv, p_unpriv],
        color=["#63b3ed", "#fc8181"],
        height=0.45,
    )
    ax.axvline(x=0.80 * p_priv, color="#faf089", linewidth=1.5,
               linestyle="--", label="80% threshold (EEOC)")
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Selection Rate", color="#a0aec0", fontsize=9)
    ax.tick_params(colors="#a0aec0", labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor("#2d3748")
    ax.legend(fontsize=8, facecolor="#1a202c", edgecolor="#4a5568",
              labelcolor="#a0aec0")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# ─────────────────────────────────────────────
#  PILLAR 2 DEMO: Proxy Correlation Heatmap
# ─────────────────────────────────────────────
def _demo_proxy_heatmap():
    st.markdown("#### 🕵️ Proxy Correlation Explorer")
    info_box(
        "This mirrors the exact logic in <code>bias_detector.py → show_proxy_warning()</code>. "
        "The system encodes categorical protected attributes to numeric codes, then computes "
        "Pearson absolute correlation against all other numeric features.",
        "#319795",
    )

    # Synthetic dataset that mimics COMPAS / credit data structure
    np.random.seed(42)
    n = 400
    race_code      = np.random.randint(0, 3, n)
    zip_code       = race_code * 10000 + np.random.randint(0, 3000, n)   # high proxy
    income         = 60000 - race_code * 8000 + np.random.normal(0, 5000, n)
    education      = 16 - race_code * 1.5 + np.random.normal(0, 1.2, n)
    age            = np.random.randint(22, 65, n)
    credit_score   = 700 - race_code * 30 + np.random.normal(0, 40, n)
    loan_approved  = (credit_score > 660).astype(int)

    demo_df = pd.DataFrame({
        "race_code":     race_code,
        "zip_code":      zip_code,
        "income":        income,
        "education_yrs": education,
        "age":           age,
        "credit_score":  credit_score,
        "loan_approved": loan_approved,
    })

    threshold = st.slider(
        "Proxy flag threshold (r >)",
        min_value=0.30, max_value=0.95, value=0.50, step=0.05,
        key="meth_proxy_thresh",
        help="Features whose |r| with race_code exceeds this are flagged as proxies",
    )

    corr = demo_df.corr(numeric_only=True)["race_code"].drop("race_code").abs().sort_values(ascending=False)

    flagged = corr[corr > threshold]
    safe    = corr[corr <= threshold]

    col_a, col_b = st.columns([2, 1])
    with col_a:
        fig2, ax2 = plt.subplots(figsize=(7, 3))
        fig2.patch.set_facecolor("#0d1117")
        ax2.set_facecolor("#0d1117")
        colors = ["#fc8181" if v > threshold else "#68d391" for v in corr.values]
        ax2.barh(corr.index[::-1], corr.values[::-1], color=colors[::-1], height=0.55)
        ax2.axvline(x=threshold, color="#faf089", linewidth=1.5, linestyle="--",
                    label=f"Proxy threshold (r = {threshold:.2f})")
        ax2.set_xlabel("|Pearson r| with race_code", color="#a0aec0", fontsize=9)
        ax2.tick_params(colors="#a0aec0", labelsize=9)
        ax2.set_xlim(0, 1.0)
        for spine in ax2.spines.values():
            spine.set_edgecolor("#2d3748")
        red_p  = mpatches.Patch(color="#fc8181", label="⚠ Proxy Detected")
        green_p = mpatches.Patch(color="#68d391", label="✅ Safe Feature")
        ax2.legend(handles=[red_p, green_p], fontsize=8, facecolor="#1a202c",
                   edgecolor="#4a5568", labelcolor="#a0aec0")
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)

    with col_b:
        st.markdown("**🚨 Flagged Proxies**")
        if not flagged.empty:
            for feat, val in flagged.items():
                st.markdown(
                    f"<div style='background:#e53e3e22;border:1px solid #e53e3e66;"
                    f"border-radius:6px;padding:0.4rem 0.7rem;margin:0.3rem 0;"
                    f"color:#fc8181;font-size:0.83rem;'>"
                    f"⚠ <b>{feat}</b><br/><span style='color:#a0aec0'>r = {val:.3f}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.success("No proxies flagged at this threshold.")

        st.markdown("**✅ Safe Features**")
        for feat, val in safe.items():
            st.markdown(
                f"<div style='background:#38a16922;border:1px solid #38a16966;"
                f"border-radius:6px;padding:0.4rem 0.7rem;margin:0.3rem 0;"
                f"color:#68d391;font-size:0.83rem;'>"
                f"✔ <b>{feat}</b> — r = {val:.3f}"
                f"</div>",
                unsafe_allow_html=True,
            )


# ─────────────────────────────────────────────
#  PILLAR 3 DEMO: Gemini Prompt Anatomy
# ─────────────────────────────────────────────
def _demo_gemini_prompt():
    st.markdown("#### 🤖 Gemini Prompt Anatomy — What We Actually Send")
    info_box(
        "This is a live reconstruction of the exact prompt template used in "
        "<code>ai_auditor.py → generate_ai_report()</code>. "
        "Change the parameters below to see how Gemini's context adapts.",
        "#d69e2e",
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        domain = st.selectbox(
            "Use-Case Domain",
            ["Loan Approval", "Recidivism Prediction", "Hiring / Recruitment",
             "University Admission", "Healthcare Triage"],
            key="meth_domain",
        )
    with col2:
        attribute = st.selectbox(
            "Protected Attribute",
            ["race", "gender", "age", "religion", "disability_status"],
            key="meth_attr",
        )
    with col3:
        gap_val = st.slider(
            "Bias Score (Fairness Gap %)",
            min_value=0.0, max_value=50.0, value=22.4, step=0.5,
            key="meth_gap",
        )

    # Risk tier — same logic as bias_detector.py
    if gap_val > 15:
        risk_tier, risk_color = "🔴 HIGH RISK",     "#e53e3e"
    elif gap_val > 5:
        risk_tier, risk_color = "🟡 MODERATE",      "#d69e2e"
    else:
        risk_tier, risk_color = "🟢 LOW RISK",      "#38a169"

    is_mitigated = st.checkbox("Mark as Post-Mitigation Report", key="meth_is_mitigated")
    status = "Post-Mitigation" if is_mitigated else "Initial Audit"

    prompt_str = f"""You are an expert AI Ethics Consultant.

Status    : {status}
Domain    : {domain}
Attribute : {attribute}
Metric    : Fairness Gap (%)
Value     : {gap_val:.1f}%

Write a professional report:
1. Summarize ethical risk
2. Explain real-world impact on {attribute} group in the context of {domain}
3. Give one expert recommendation

Keep it concise and clear."""

    st.markdown(
        f"""<div style="
            background:#1a202c;
            border:1px solid #4a5568;
            border-radius:10px;
            padding:1rem 1.2rem;
            font-family:'Courier New', monospace;
            font-size:0.82rem;
            color:#a0aec0;
            white-space:pre-wrap;
            line-height:1.7;
        ">{prompt_str}</div>""",
        unsafe_allow_html=True,
    )

    st.markdown(
        f"<div style='margin-top:0.8rem; background:{risk_color}18; border:1px solid {risk_color}66;"
        f"border-radius:8px; padding:0.7rem 1.2rem; color:{risk_color}; font-weight:600; font-size:0.9rem;'>"
        f"Risk Classification Engine → &nbsp; <b>{risk_tier}</b> &nbsp;"
        f"(Gap = {gap_val:.1f}% vs. threshold: &gt;15% = High, 5–15% = Moderate, &lt;5% = Low)"
        f"</div>",
        unsafe_allow_html=True,
    )

    st.caption(
        "💡 Gemini 2.0 Flash uses this structured prompt to adapt the ethical urgency of its "
        "narrative. A Recidivism domain triggers stronger language than Loan Approval because "
        "the model understands societal stakes — this is context-awareness, not hard-coding."
    )


# ─────────────────────────────────────────────
#  PILLAR 4 DEMO: Reweighing Math Walkthrough
# ─────────────────────────────────────────────
def _demo_reweighing():
    st.markdown("#### ⚖️ Kamiran-Calders Reweighing — Step-by-Step")
    info_box(
        "FairFrame's mitigation engine (<code>bias_fixer.py → apply_mitigation()</code>) uses "
        "the Kamiran &amp; Calders (2012) algorithm. It assigns each sample a <i>fairness weight</i> "
        "so that every demographic group carries equal statistical influence during model training.",
        "#e53e3e",
    )

    # Small worked example with fixed numbers for clarity
    st.markdown(
        """
| Symbol | Meaning | Example |
|--------|---------|---------|
| `P(G)` | Probability of belonging to group G | P(Black) = 0.35 |
| `P(Y)` | Probability of positive outcome | P(Approved) = 0.60 |
| `P(G, Y)` | Joint probability of group G *and* outcome Y | P(Black, Approved) = 0.18 |
| **`W`** | **Sample Weight assigned by FairFrame** | **(0.35 × 0.60) / 0.18 = 1.167** |
        """
    )

    st.markdown(
        r"""
$$
W_{G,Y} = \frac{P(G) \;\times\; P(Y)}{P(G,\; Y)}
$$

> **Intuition**: If a group is *underrepresented* among positive outcomes relative to its population share, 
> its weight `W > 1` — giving those samples more influence. Overrepresented combinations get `W < 1`.
        """
    )

    np.random.seed(7)
    n = 200
    group = np.random.choice(["Privileged", "Unprivileged"], n, p=[0.55, 0.45])
    outcome = np.where(
        group == "Privileged",
        np.random.choice([0, 1], n, p=[0.25, 0.75]),
        np.random.choice([0, 1], n, p=[0.60, 0.40]),
    )
    demo_df2 = pd.DataFrame({"group": group, "outcome": outcome})

    p_g  = demo_df2["group"].value_counts() / n
    p_y  = demo_df2["outcome"].value_counts() / n

    weights = []
    for _, row in demo_df2.iterrows():
        g, y = row["group"], row["outcome"]
        p_gy = len(demo_df2[(demo_df2["group"] == g) & (demo_df2["outcome"] == y)]) / n
        w    = (p_g[g] * p_y[y]) / (p_gy + 1e-9)
        weights.append(round(w, 4))
    demo_df2["weight"] = weights

    before_gap = (
        demo_df2.groupby("group")["outcome"].mean().max()
        - demo_df2.groupby("group")["outcome"].mean().min()
    ) * 100

    weighted_means = demo_df2.groupby("group").apply(
        lambda g: np.average(g["outcome"], weights=g["weight"])
    )
    after_gap = (weighted_means.max() - weighted_means.min()) * 100

    c1, c2, c3 = st.columns(3)
    c1.metric("Bias Before Reweighing", f"{before_gap:.1f}%")
    c2.metric("Bias After Reweighing",  f"{after_gap:.1f}%")
    c3.metric("Reduction",              f"-{before_gap - after_gap:.1f}%",
              delta=f"-{before_gap - after_gap:.1f}%", delta_color="normal")

    with st.expander("🔎 Inspect computed sample weights (first 10 rows)"):
        st.dataframe(demo_df2.head(10).style.format({"weight": "{:.4f}"}),
                     use_container_width=True)


# ─────────────────────────────────────────────
#  MAIN PAGE ENTRY POINT
# ─────────────────────────────────────────────
def show_technical_methodology():

    # ── Hero Banner ──────────────────────────────────────────────────────────
    st.markdown(
        """
        <div style="
            background: linear-gradient(135deg, #0d0d1a 0%, #1a1a3e 40%, #0f3460 100%);
            border-radius: 16px;
            padding: 2.5rem 2rem;
            text-align: center;
            margin-bottom: 2rem;
            border: 1px solid #e9456022;
        ">
            <h1 style="color:#ffffff; font-size:2rem; margin:0 0 0.5rem 0; letter-spacing:1px;">
                ⚙️ Technical Methodology
            </h1>
            <p style="color:#a0aec0; font-size:1rem; margin:0; max-width:700px; margin:auto; line-height:1.7;">
                A complete, transparent breakdown of the mathematical foundations, 
                algorithmic pipelines, and AI reasoning layers that power 
                <strong style="color:#e94560;">FairFrame</strong>. 
                This section answers: <em>how does the system actually detect and correct bias?</em>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── System Overview ───────────────────────────────────────────────────────
    st.markdown("### 🗺️ System Architecture at a Glance")
    st.markdown(
        """
        FairFrame is a **four-module pipeline**. Each module has a specific, 
        mathematically defined responsibility:
        """
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        pillar_card("M1", "Data Handler",
                    "Ingests CSV / built-in datasets, detects column types, validates schema",
                    "#63b3ed")
    with c2:
        pillar_card("M2a", "Bias Detector",
                    "Computes group disparity using Disparate Impact & Statistical Parity Difference",
                    "#e94560")
    with c3:
        pillar_card("M2b", "Bias Fixer",
                    "Applies Kamiran-Calders Reweighing — a pre-processing fairness algorithm",
                    "#f6ad55")
    with c4:
        pillar_card("M3", "AI Auditor",
                    "Gemini 2.0 Flash converts raw scores into a contextual ethical narrative",
                    "#68d391")

    st.markdown("")

    # Pipeline flow diagram (text-based, styled)
    st.markdown(
        """
        <div style="
            background:#1a202c;
            border-radius:10px;
            padding:1rem 1.5rem;
            font-family:'Courier New', monospace;
            font-size:0.85rem;
            color:#a0aec0;
            text-align:center;
            margin:1rem 0;
        ">
            📂 Raw Dataset &nbsp;→&nbsp; 
            <span style="color:#63b3ed;">[ M1: Ingest & Validate ]</span> &nbsp;→&nbsp; 
            <span style="color:#e94560;">[ M2a: Detect Bias ]</span> &nbsp;→&nbsp; 
            <span style="color:#f6ad55;">[ M2b: Mitigate Bias ]</span> &nbsp;→&nbsp; 
            <span style="color:#68d391;">[ M3: AI Narrative Report ]</span> &nbsp;→&nbsp; 
            📄 PDF Export
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════════
    #  PILLAR 1 — THE FAIRNESS MATH
    # ══════════════════════════════════════════════════════════════════════════
    section_header(
        "📐", "Pillar 1: The Fairness Mathematics",
        "The specific statistical standards and formulas FairFrame uses — not general heuristics."
    )

    tab1a, tab1b, tab1c = st.tabs([
        "  Four-Fifths Rule (EEOC)  ",
        "  Statistical Parity Difference  ",
        "  Live Calculator  ",
    ])

    with tab1a:
        st.markdown("#### The Four-Fifths Rule — U.S. EEOC Standard (29 CFR § 1607)")
        info_box(
            "The Equal Employment Opportunity Commission (EEOC) provides a bright-line legal test: "
            "if the selection rate for any group is <b>less than 80%</b> of the rate for the highest-selected group, "
            "it constitutes evidence of <b>Adverse Impact</b> — a form of systemic, structural discrimination.",
            "#e94560",
        )

        st.markdown(
            r"""
$$
\text{Disparate Impact (DI)} = \frac{P(\hat{Y} = 1 \;\mid\; \text{unprivileged})}{P(\hat{Y} = 1 \;\mid\; \text{privileged})}
$$

**Decision rule implemented in FairFrame:**

| DI Value | Interpretation | FairFrame Action |
|----------|----------------|-----------------|
| `DI ≥ 0.80` | Passes EEOC test — acceptable fairness | 🟢 LOW RISK |
| `0.65 ≤ DI < 0.80` | Borderline — warrants investigation | 🟡 MODERATE |
| `DI < 0.65` | Clear adverse impact detected | 🔴 HIGH RISK |

> **Why this formula?** DI is a *ratio* — it captures relative inequality. A 50% approval rate for Group A 
> vs. 70% for Group B looks small in absolute terms, but the ratio (0.714) fails the EEOC test.
            """
        )

        st.markdown("**Real-world application in FairFrame (from `bias_detector.py`)**")
        st.code(
            """
# bias_detector.py — run_audit() function
group_stats = df.groupby('Demographic Groups')[target].mean()

# Disparate Impact equivalent (max vs min group):
gap = (group_stats.max() - group_stats.min()) * 100  # classification
# → Risk thresholds: >15% = HIGH, >5% = MODERATE, ≤5% = LOW
""",
            language="python",
        )

    with tab1b:
        st.markdown("#### Statistical Parity Difference (SPD)")
        info_box(
            "SPD is the <b>signed difference</b> in positive outcome rates between groups. "
            "Unlike DI (which is a ratio), SPD captures the absolute gap — "
            "useful when both groups have very high or very low base rates.",
            "#805ad5",
        )

        st.markdown(
            r"""
$$
\text{SPD} = P(\hat{Y} = 1 \;\mid\; \text{unprivileged}) \;-\; P(\hat{Y} = 1 \;\mid\; \text{privileged})
$$

**Interpretation of SPD:**

- `SPD = 0.00` → **Perfect parity** — both groups are equally likely to receive a positive outcome.  
- `SPD = −0.20` → The unprivileged group has a **20 percentage-point lower** chance of a positive outcome.  
- `SPD = +0.10` → The unprivileged group is **favored** by 10 percentage points.

**Fairness Condition:** A system is considered fair under Statistical Parity when `|SPD| ≤ 0.05`

> **Together, DI and SPD form a complementary pair**: DI catches proportional gaps 
> (e.g., "60% as likely"), while SPD catches absolute gaps (e.g., "20 points lower").
            """
        )

        st.markdown("**Regression variant — Disparity Ratio:**")
        st.markdown(
            r"""
For continuous outcome targets (e.g., predicted salary), FairFrame switches from a gap-based 
measure to a **ratio**:

$$
\text{Disparity Ratio} = \frac{\mu_{\text{max\_group}}}{\mu_{\text{min\_group}}}
$$

> Thresholds: `> 1.5x` = HIGH RISK, `> 1.2x` = MODERATE, `≤ 1.2x` = LOW RISK
            """
        )

    with tab1c:
        _demo_di_calculator()

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════════
    #  PILLAR 2 — PROXY DETECTION
    # ══════════════════════════════════════════════════════════════════════════
    section_header(
        "🕵️", "Pillar 2: The Proxy Detection System",
        "Finding hidden bias encoded in seemingly neutral features."
    )

    tab2a, tab2b = st.tabs([
        "  How Proxies Work  ",
        "  Live Correlation Explorer  ",
    ])

    with tab2a:
        st.markdown("#### Why Removing the Sensitive Column Is Not Enough")
        info_box(
            "A landmark finding in algorithmic fairness research is <b>Proxy Discrimination</b>: "
            "a model can re-learn race, gender, or income from <i>other</i> columns — "
            "even after you delete the protected attribute entirely.",
            "#319795",
        )

        col_ex1, col_ex2 = st.columns(2)
        with col_ex1:
            st.markdown(
                """
**Classic proxy chains:**

| Protected Attribute | Proxy Variable | Mechanism |
|--------------------|----------------|-----------|
| **Race** | Zip Code | Residential segregation |
| **Race** | Credit Score | Systemic wealth inequality |
| **Gender** | Part-time Employment | Caregiving burden |
| **Income** | Education Level | Socioeconomic access |
| **Age** | Years of Experience | Direct linear mapping |
                """
            )
        with col_ex2:
            info_box(
                "📖 <b>Example:</b> The COMPAS recidivism tool (Northpointe) "
                "excluded explicit race, but its predictions were strongly correlated "
                "with neighbourhood data, which itself is racially stratified "
                "(ProPublica, 2016). FairFrame's proxy detector catches exactly this pattern.",
                "#744210",
            )

        st.markdown("#### The Algorithm — From `bias_detector.py: show_proxy_warning()`")
        st.code(
            """
def show_proxy_warning(df, protected_col):
    temp_df = df.copy()
    
    # Step 1: Encode the categorical protected attribute numerically
    temp_df[protected_col] = temp_df[protected_col].astype('category').cat.codes

    # Step 2: Compute absolute Pearson correlation with every other numeric column
    corr = temp_df.corr(numeric_only=True)[protected_col].abs().sort_values(ascending=False)
    
    # Step 3: Inspect the top 2 correlated non-target features
    proxies = corr[1:3]   # index 0 is self-correlation (= 1.0)

    # Step 4: Flag if top proxy correlation exceeds threshold (r > 0.50)
    if not proxies.empty and proxies.iloc[0] > 0.50:
        st.warning(f"⚠️ {proxies.index[0]} is a proxy for {protected_col} (r = {proxies.iloc[0]:.2f})")
""",
            language="python",
        )

        st.markdown(
            r"""
**The Pearson Correlation Coefficient:**

$$
r = \frac{\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})}
         {\sqrt{\sum_{i=1}^{n}(x_i-\bar{x})^2 \;\cdot\; \sum_{i=1}^{n}(y_i-\bar{y})^2}}
$$

Where `x` is the encoded protected attribute and `y` is the candidate proxy feature.

| `|r|` Range | Interpretation | FairFrame Flag |
|-------------|----------------|----------------|
| 0.00 – 0.30 | Weak / No correlation | ✅ Safe |
| 0.30 – 0.50 | Moderate correlation | 📋 Monitor |
| 0.50 – 0.70 | Strong correlation | ⚠ Likely Proxy |
| 0.70 – 1.00 | Very strong — near-deterministic proxy | 🚨 Critical |
            """
        )

    with tab2b:
        _demo_proxy_heatmap()

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════════
    #  PILLAR 3 — GEMINI AI REASONING LAYER
    # ══════════════════════════════════════════════════════════════════════════
    section_header(
        "🤖", "Pillar 3: Gemini 2.0 Flash — The Ethical Reasoning Layer",
        "Bridging the gap between raw numbers and human-understandable, context-aware ethical judgement."
    )

    tab3a, tab3b, tab3c = st.tabs([
        "  Why Gemini?  ",
        "  What We Send / Receive  ",
        "  Risk Categorisation  ",
    ])

    with tab3a:
        st.markdown("#### The Limitation of Pure Statistics")
        info_box(
            "A fairness score of <b>22.4%</b> is alarming in a hiring tool — but routine in sports performance analytics. "
            "No static formula can decide <i>how urgently</i> to frame a disparity. "
            "Gemini 2.0 Flash adds <b>domain awareness</b> to the pipeline.",
            "#d69e2e",
        )

        st.markdown(
            """
| Task Gemini Performs | Input | Output |
|---------------------|-------|--------|
| **Data-to-Narrative Conversion** | Raw disparity scores (gap %, DI ratio) | Human-readable audit report |
| **Risk Categorisation** | Gap value + domain context | Low / Medium / High / Critical level |
| **Contextual Bias Insights** | Domain label (e.g. "Loan Approval") | Domain-escalated ethical urgency |
| **Expert Recommendations** | Protected attribute + risk tier | Specific, actionable mitigation advice |
| **Post-Mitigation Comparison** | Before/after bias scores | Narrative of improvement + caveats |
            """
        )

        st.markdown("#### How Gemini Is Invoked — From `ai_auditor.py`")
        st.code(
            """
# ai_auditor.py — generate_ai_report()

# 1. Extract structured metrics from the bias engine
attr     = ", ".join(audit_results.get('protected_cols', []))
gap      = audit_results.get('gap', 0)              # e.g., 22.4
is_class = audit_results.get('is_classification', True)

# 2. Construct a structured natural-language prompt
prompt = f\"\"\"
You are an expert AI Ethics Consultant.
Status    : {status}          # "Initial Audit" OR "Post-Mitigation"
Attribute : {attr}            # e.g., "race, gender"
Metric    : Fairness Gap (%)
Value     : {gap}

Write a professional report:
1. Summarize ethical risk
2. Explain impact
3. Give one expert recommendation
\"\"\"

# 3. Send to Gemini 2.0 Flash via Google Gen AI SDK
response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents=prompt
)
report_text = response.text   # Natural-language ethical audit narrative
""",
            language="python",
        )

    with tab3b:
        _demo_gemini_prompt()

    with tab3c:
        st.markdown("#### Risk Tier Assignment — Bridging Statistics and Policy")
        info_box(
            "FairFrame assigns risk tiers <i>before</i> sending to Gemini, so the AI can anchor "
            "its narrative to a concrete severity label. The thresholds are aligned with "
            "EEOC practice and IBM AI Fairness 360 literature.",
            "#e53e3e",
        )

        st.markdown(
            """
| Risk Tier | Classification Gap | Regression Ratio | Legal / Ethical Framing |
|-----------|-------------------|------------------|------------------------|
| 🟢 **LOW RISK** | ≤ 5% | ≤ 1.2× | Within acceptable statistical noise |
| 🟡 **MODERATE** | 5% – 15% | 1.2× – 1.5× | Warrants monitoring; possible systemic issue |
| 🔴 **HIGH RISK** | > 15% | > 1.5× | Clear EEOC adverse impact; immediate review required |
            """
        )

        # Visual representation
        fig3, ax3 = plt.subplots(figsize=(8, 1.5))
        fig3.patch.set_facecolor("#0d1117")
        ax3.set_facecolor("#0d1117")
        ax3.barh(["Risk Scale"], [5],  color="#38a169", height=0.5, left=0)
        ax3.barh(["Risk Scale"], [10], color="#d69e2e", height=0.5, left=5)
        ax3.barh(["Risk Scale"], [30], color="#e53e3e", height=0.5, left=15)
        ax3.set_xlim(0, 45)
        ax3.set_xlabel("Fairness Gap (%)", color="#a0aec0", fontsize=9)
        ax3.tick_params(colors="#a0aec0", labelsize=9)
        for spine in ax3.spines.values():
            spine.set_edgecolor("#2d3748")
        g  = mpatches.Patch(color="#38a169", label="🟢 LOW (0–5%)")
        y  = mpatches.Patch(color="#d69e2e", label="🟡 MODERATE (5–15%)")
        r  = mpatches.Patch(color="#e53e3e", label="🔴 HIGH (>15%)")
        ax3.legend(handles=[g, y, r], loc="upper right", fontsize=8,
                   facecolor="#1a202c", edgecolor="#4a5568", labelcolor="#a0aec0")
        st.pyplot(fig3, use_container_width=True)
        plt.close(fig3)

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════════
    #  PILLAR 4 — BIAS MITIGATION ENGINE
    # ══════════════════════════════════════════════════════════════════════════
    section_header(
        "⚖️", "Pillar 4: Bias Mitigation — Kamiran-Calders Reweighing",
        "A principled pre-processing algorithm to correct structural imbalance in training data."
    )
    _demo_reweighing()

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════════
    #  GLOBAL SCAN MODULE
    # ══════════════════════════════════════════════════════════════════════════
    section_header(
        "🔭", "Bonus: Global Attribute Risk Scanner",
        "FairFrame doesn't only audit the column you specify — it scans every column automatically."
    )

    st.markdown(
        """
The `run_audit_all()` function in `bias_detector.py` performs an exhaustive sweep across 
**every column** in the uploaded dataset:

1. **Skip columns** with > 20 unique values (continuous) or < 2 unique values (constants)  
2. **Group** each remaining column and compute the mean target outcome per group  
3. **Score** using the same classification/regression risk logic  
4. **Rank** all columns by Fairness Score (descending) and present as a sortable table  

This allows auditors to **discover unexpected bias dimensions** — e.g., finding that 
`Employment Status` is more discriminatory than the explicitly specified `Gender` column.
        """
    )

    st.code(
        """
# bias_detector.py — run_audit_all()
for col in all_cols:
    if temp_df[col].nunique() > 20 or temp_df[col].nunique() < 2:
        continue                          # Skip unsuitable columns

    group_stats = temp_df.groupby(col)[target].mean().dropna()

    if is_classification:
        score = (group_stats.max() - group_stats.min()) * 100    # Fairness Gap %
        risk  = "HIGH" if score > 15 else "MODERATE" if score > 5 else "LOW"
    else:
        score = group_stats.max() / (group_stats.min() + 1e-6)   # Ratio
        risk  = "HIGH" if score > 1.5 else "MODERATE" if score > 1.2 else "LOW"
""",
        language="python",
    )

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════════
    #  STANDARDS & REFERENCES
    # ══════════════════════════════════════════════════════════════════════════
    section_header(
        "📚", "Standards, Frameworks & Research References",
        "FairFrame is grounded in established legal, statistical, and academic sources."
    )

    col_r1, col_r2 = st.columns(2)
    with col_r1:
        st.markdown(
            """
**Legal & Regulatory Standards**
- 📜 **U.S. EEOC Uniform Guidelines** (29 CFR § 1607) — Four-Fifths / 80% Rule
- 📜 **EU AI Act (2024)** — High-risk AI systems require fairness documentation
- 📜 **NYC Local Law 144** — Automated employment decision tools must be bias-audited

**Algorithmic Frameworks**
- 🔬 **IBM AI Fairness 360 (AIF360)** — Source for SPD, DI metric definitions
- 🔬 **Fairlearn (Microsoft)** — Constrained optimization approaches
- 🔬 **Google PAIR Guidebook** — Human-centered AI evaluation
            """
        )
    with col_r2:
        st.markdown(
            """
**Academic Research**
- 📖 Kamiran & Calders (2012) — *"Data preprocessing techniques for classification without discrimination"*
- 📖 Feldman et al. (2015) — *"Certifying and removing disparate impact"*
- 📖 Chouldechova (2017) — *"Fair prediction with disparate impact"*
- 📖 ProPublica (2016) — *"Machine Bias"* (COMPAS recidivism study)
- 📖 Barocas, Hardt & Narayanan — *"Fairness and Machine Learning"* (fairmlbook.org)

**AI Layer**
- 🤖 Google Gemini 2.0 Flash — via `google-generativeai` SDK
            """
        )

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown(
        """
        <div style="
            background: linear-gradient(135deg, #0d0d1a, #1a1a3e);
            border-radius: 12px;
            padding: 1.5rem 2rem;
            text-align: center;
            margin-top: 2rem;
            border: 1px solid #e9456022;
        ">
            <p style="color:#a0aec0; margin:0; font-size:0.85rem; line-height:1.8;">
                🏆 <strong style="color:#e94560;">FairFrame</strong> — 
                Google Solution Challenge 2025 &nbsp;|&nbsp;
                Built with Streamlit + Gemini 2.0 Flash &nbsp;|&nbsp;
                <em>Making AI Accountable, One Audit at a Time.</em>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
