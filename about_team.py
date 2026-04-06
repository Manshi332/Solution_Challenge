import streamlit as st
import streamlit.components.v1 as components
import os
import base64


def get_avatar_src(image_path):
    """Return a base64 data URI for local files, or the URL for remote images."""
    if os.path.exists(image_path):
        ext = image_path.rsplit(".", 1)[-1].lower()
        mime = "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}"
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        return f"data:{mime};base64,{b64}"
    return image_path


def team_card_html(name, role, image_path, linkedin, email, phone, tagline):
    avatar_src = get_avatar_src(image_path)
    return f"""
    <div class="card">
      <div class="avatar-ring">
        <img src="{avatar_src}" alt="{name}" />
      </div>
      <div class="name">{name}</div>
      <div class="role">{role}</div>
      <div class="tagline">{tagline}</div>
      <hr class="divider" />
      <div class="chips">
        <a class="chip chip-email" href="mailto:{email}" title="{email}">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
            <rect x="2" y="4" width="20" height="16" rx="2"/>
            <polyline points="22,6 12,13 2,6"/>
          </svg>
          Email
        </a>
        <a class="chip chip-phone" href="tel:{phone}" title="{phone}">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M22 16.92v3a2 2 0 01-2.18 2 19.79 19.79 0 01-8.63-3.07A19.5 19.5 0 013.07 10.8a19.79 19.79 0 01-3.07-8.7A2 2 0 012.18 0h3a2 2 0 012 1.72c.127.96.361 1.903.7 2.81a2 2 0 01-.45 2.11L6.91 7.09a16 16 0 006 6l.56-.56a2 2 0 012.11-.45c.907.339 1.85.573 2.81.7A2 2 0 0122 14.92z"/>
          </svg>
          Contact
        </a>
        <a class="chip chip-linkedin" href="{linkedin}" target="_blank" rel="noopener">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor">
            <path d="M16 8a6 6 0 016 6v7h-4v-7a2 2 0 00-4 0v7h-4V9h4v2a6 6 0 014-3zM2 9h4v12H2z"/>
            <circle cx="4" cy="4" r="2"/>
          </svg>
          LinkedIn
        </a>
      </div>
    </div>
    """

CARD_CSS = """
  @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@500;600;700&family=Nunito:wght@300;400;500;600&display=swap');

  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  html, body {
    background: transparent;
    font-family: 'Nunito', sans-serif;
    /* let content dictate height — no overflow clipping */
    overflow: visible;
  }

  .grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 18px;
    padding: 6px 4px 12px;   /* extra bottom pad so hover shadow isn't clipped */
  }

  .card {
    background: linear-gradient(160deg, rgba(255,255,255,0.048) 0%, rgba(255,255,255,0.016) 100%);
    border: 1px solid rgba(200,185,255,0.22);
    border-radius: 22px;
    padding: 26px 20px 20px;
    text-align: center;
    position: relative;
    /* NO overflow:hidden — was clipping hover shadow */
    transition: transform 0.28s ease, box-shadow 0.28s ease;
  }

  .card::before {
    content: '';
    position: absolute;
    top: 0; left: 12%; right: 12%;
    height: 1px;
    border-radius: 1px;
    background: linear-gradient(90deg, transparent, #d4c8ff, transparent);
  }

  .card:hover {
    transform: translateY(-5px);
    box-shadow: 0 22px 52px rgba(0,0,0,0.45), 0 0 0 1px rgba(184,169,232,0.2);
  }

  /* Avatar */
  .avatar-ring {
    display: inline-flex;
    padding: 3px;
    border-radius: 50%;
    background: linear-gradient(135deg, #c4b0f5 0%, #8ab4e8 100%);
    margin-bottom: 14px;
  }
  .avatar-ring img {
    width: 86px;
    height: 86px;
    border-radius: 50%;
    object-fit: cover;
    border: 3px solid #100e1e;
    display: block;
  }

  /* Text */
  .name {
    font-family: 'Cormorant Garamond', serif;
    font-size: 19px;
    font-weight: 600;
    color: #f0eafa;
    letter-spacing: 0.3px;
    margin-bottom: 5px;
  }
  .role {
    font-size: 9.5px;
    font-weight: 600;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: #d4c8ff;
    margin-bottom: 10px;
  }
  .tagline {
    font-size: 12.5px;
    font-weight: 400;
    color: #a89fc4;
    line-height: 1.7;
    min-height: 42px;
    margin-bottom: 15px;
    font-style: italic;
  }

  /* Divider */
  .divider {
    border: none;
    border-top: 1px solid rgba(200,185,255,0.08);
    margin: 0 0 15px;
  }

  /* Chips */
  .chips {
    display: flex;
    justify-content: center;
    gap: 7px;
    flex-wrap: wrap;
  }
  .chip {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 5px 11px;
    border-radius: 50px;
    font-size: 11px;
    font-weight: 600;
    font-family: 'Nunito', sans-serif;
    text-decoration: none;
    transition: filter 0.18s, transform 0.15s;
    white-space: nowrap;
    letter-spacing: 0.2px;
  }
  .chip:hover { transform: scale(1.07); filter: brightness(1.3); }

  .chip-email {
    background: rgba(255, 168, 176, 0.12);
    color: #ffa8b0;
    border: 1px solid rgba(255, 168, 176, 0.2);
  }
  .chip-phone {
    background: rgba(160, 210, 190, 0.11);
    color: #90d4b8;
    border: 1px solid rgba(160, 210, 190, 0.2);
  }
  .chip-linkedin {
    background: rgba(138, 180, 232, 0.12);
    color: #8ab4e8;
    border: 1px solid rgba(138, 180, 232, 0.2);
  }
"""


def render_team_grid(members):
    cards_html = "".join([
        team_card_html(
            m["name"], m["role"], m["image_path"],
            m["linkedin"], m["email"], m["phone"], m["tagline"]
        )
        for m in members
    ])

    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <style>{CARD_CSS}</style>
</head>
<body>
  <div class="grid">{cards_html}</div>
</body>
</html>"""
    components.html(html, height=820, scrolling=False)


PAGE_HEADER_CSS = """
<style>
  @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,400;0,500;0,600;1,400;1,500&family=Nunito:wght@300;400;500;600&display=swap');

  .stApp {
    background: radial-gradient(ellipse at 25% 0%, #1a1625 0%, #111018 55%, #0f0e16 100%)
  }
  /* Apply Nunito site-wide for body text */
  .stApp, .stApp * {
    font-family: 'Nunito', sans-serif;
  }

  .section-label {
    font-family: 'Nunito', sans-serif;
    font-size: 10.5px;
    font-weight: 600;
    letter-spacing: 4px;
    text-transform: uppercase;
    color: #c8bbf5;
    margin-bottom: 6px;
    opacity: 0.8;
  }
  .page-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 46px;
    font-weight: 600;
    color: #f0eafa;
    line-height: 1.1;
    margin: 0 0 14px 0;
    letter-spacing: -0.5px;
  }
  .page-subtitle {
    font-family: 'Nunito', sans-serif;
    font-size: 15px;
    font-weight: 400;
    color: #9590b0;
    max-width: 500px;
    line-height: 1.8;
    margin-bottom: 38px;
  }
  .team-section-title {
    font-family: 'Nunito', sans-serif;
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 3.5px;
    text-transform: uppercase;
    color: #5a527a;
    margin-bottom: 16px;
  }
  .accent-line {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(184,169,232,0.25), transparent);
    border: none;
    margin: 38px 0 28px;
  }
  .vision-box {
    background: linear-gradient(135deg, rgba(184,169,232,0.06) 0%, rgba(138,180,232,0.04) 100%);
    border: 1px solid rgba(184,169,232,0.14);
    border-radius: 18px;
    padding: 28px 32px;
  }
  .vision-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 22px;
    font-weight: 600;
    color: #f0eafa;
    margin-bottom: 10px;
    letter-spacing: 0.2px;
  }
  .vision-text {
    font-family: 'Nunito', sans-serif;
    font-size: 14.5px;
    font-weight: 400;
    color: #9e99b8;
    line-height: 1.9;
  }
</style>
"""


def show_about_team():
    st.markdown(PAGE_HEADER_CSS, unsafe_allow_html=True)

    st.markdown('<p class="section-label">The People Behind It</p>', unsafe_allow_html=True)
    st.markdown('<h1 class="page-title">Meet the Team</h1>', unsafe_allow_html=True)
    st.markdown("""
        <p class="page-subtitle">
            We&#39;re building <strong style="color:#eaf2ff;">FairFrame Pro</strong> &mdash;
            an AI system designed to detect, analyze, and mitigate bias in algorithmic decision-making.
        </p>
        <p class="team-section-title">Our Members</p>
    """, unsafe_allow_html=True)

    members = [
        dict(
            name="Manshi Prajapati",
            role="Model Bias & UI Engineer",
            image_path="https://i.pravatar.cc/150?img=5",
            linkedin="https://www.linkedin.com/in/manshi-prajapati-a407192aa/",
            email="manshiprajapati332@gmail.com",
            phone="+91-7489378054",
            tagline="Focused on building intuitive and fair AI systems."
        ),
        dict(
            name="Ishika Gupta",
            role="Bias Detection Engineer",
            image_path="assets/ishika.jpeg",
            linkedin="https://www.linkedin.com/in/ishika-gupta-5302452bb/",
            email="ishikagupta2202@gmail.com",
            phone="+91-9301689801",
            tagline="Specializes in fairness metrics and intersectional analysis."
        ),
        dict(
            name="Sakshi Saxena",
            role="AI Ethical Auditor",
            image_path="https://i.pravatar.cc/150?img=32",
            linkedin="https://www.linkedin.com/in/sakshi-saxena-a01392287/",
            email="sakshisaxena7121@gmail.com",
            phone="+91-7428955499",
            tagline="Transforms technical outputs into human ethical insights."
        ),
        dict(
            name="Mihika Saxena",
            role="Data & Integration Engineer",
            image_path="assets/mihika.jpeg",
            linkedin="https://www.linkedin.com/in/mihika-saxena-b5bb8a28b/",
            email="mihika11saxena@gmail.com",
            phone="+91-7740910965",
            tagline="Ensures clean data pipelines and seamless integration."
        ),
    ]

    render_team_grid(members)

    st.markdown('<hr class="accent-line">', unsafe_allow_html=True)
    st.markdown("""
    <div class="vision-box">
      <p class="vision-title">&#127757; Our Vision</p>
      <p class="vision-text">
        To make AI systems <strong style="color:#eaf2ff;">transparent, fair, and accountable</strong> &mdash;
        ensuring that technology benefits everyone equally, regardless of background, identity, or circumstance.
      </p>
    </div>
    """, unsafe_allow_html=True)