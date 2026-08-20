# Implementation Plan: DROPOUT RISK Retro OS / Neo-Brutalist Website

Build a complete, standalone, production-ready single-page website (`index.html`) for **DROPOUT RISK** — an ML-powered early warning system predicting student dropout risk in Indian higher education institutions (built for Avishkar 2026 at Pillai HOC College).

---

## User Review Required

> [!IMPORTANT]
> - **Visual Architecture**: The site adheres strictly to the **Retro OS meets Neo-Brutalism** aesthetic (`#B8DDE1` wallpaper, `#1A1A1A` titlebars with `#FF5F57`, `#FFBD2E`, `#28CA41` dots, thick 2px black borders, hard `4px 4px 0px #1A1A1A` drop shadows, `Space Mono` font, zero rounded corners on boxes).
> - **External Links**: Direct links to the live Streamlit app (`https://dropout-risk-system.streamlit.app`) and GitHub repo (`https://github.com/anewsevile/dropout_risk_prediction`).
> - **Interactive Features**: In addition to the 6 core windows, an interactive **Live Risk Sandbox (`demo_simulator.exe`)** allows competition judges to test the risk scoring logic dynamically with real-time explainability output.

---

## Proposed Architecture & Structure

The single-page web app will be crafted in `index.html` (comprising embedded clean CSS and vanilla JavaScript with zero external runtime dependencies except Google Fonts for `Space Mono`).

```
c:\antigravity xoxo\
└── index.html  # Complete single-page application with HTML, CSS & Vanilla JS
```

### Key Components

1. **Retro OS Top Menu Bar / Taskbar**:
   - Left: `DROPOUT RISK OS [v2.6.4]` with pulsating status dot (`SYS_ONLINE`)
   - Center Navigation: Smooth jump links to `#problem`, `#architecture`, `#features`, `#metrics`, `#simulator`
   - Right: Quick CTA `VIEW DASHBOARD` (yellow `#FFBD2E`) + Live system clock (`HH:MM:SS IST`)

2. **Desktop Canvas & Desktop Shortcuts (`#B8DDE1`)**:
   - Authentic retro desktop background with interactive desktop icons on the side (quick links to sections and live apps)
   - Layout grid with staggered, slightly offset windows on desktop screens that stack smoothly into a responsive single column on mobile.

3. **Window 1: Hero / Desktop Header (`intro.exe`)**:
   - Headline: *"IDENTIFY AT-RISK STUDENTS BEFORE THEY SLIP AWAY."*
   - Subtext: *"ML-powered early warning system for Indian higher education institutions."*
   - Competition Badge: `AVISHKAR 2026 — PILLAI HOC COLLEGE`
   - Action CTAs: `OPEN DASHBOARD →` and `SEE HOW IT WORKS`
   - Retro terminal preview chip showing real-time inference throughput & tier-2/3 college adaptation matrix.

4. **Window 2: The Problem (`problem.exe`)**:
   - Headline: *"THE CRISIS IN INDIAN HIGHER ED."*
   - Barrier 01: Socio-economic pressures pushing students out silently (fee cycles, family distress, first-generation learner hurdles)
   - Barrier 02: Absence of proactive early warning systems (late semester discoveries after irreversible backlogs)
   - Highlight Stat: `10%+ dropout rates in certain regions (NIRF 2023)`

5. **Window 3: How It Works (`system_architecture.exe`)**:
   - 3-Step Neo-Brutalist Pipeline with step connectors:
     - `01 / DATA INPUT`: Attendance, CGPA, assignment submission velocity, backlog history, LMS engagement.
     - `02 / RISK ANALYSIS`: Gradient Boosting classifier computing calibrated risk scores (0–100).
     - `03 / INTERVENTION`: Plain-English LLM explanation + faculty guidance action plan.

6. **Window 4: Key Features (`features.exe`)**:
   - 2x2 Neo-Brutalist Grid:
     - `F-01: RISK SCORING` (0-100 Probabilistic Calibration)
     - `F-02: EXPLAINABILITY` (Feature Importance & Plain-English Rationale)
     - `F-03: INTERVENTION SUGGESTIONS` (Role-tailored Faculty Playbooks)
     - `F-04: REAL-TIME DASHBOARD` (Live Streamlit Integration)

7. **Window 5: Interactive Simulator Easter Egg (`demo_simulator.exe`)**:
   - Interactive sliders for Attendance %, Cumulative Backlogs, Current CGPA, and Assignment Submissions.
   - Calculates dynamic risk level (Low / Medium / High / Critical) and updates the gauge and Gemini-powered plain-English summary card in real-time.

8. **Window 6: Proven Metrics (`results.log`)**:
   - 3 Neo-brutalist highlight stat blocks:
     - `92.5%` — ACCURACY
     - `17.7%` — STUDENTS FLAGGED
     - `90.9%` — HIGH-RISK RECALL
   - Primary External CTAs:
     - `VIEW LIVE DASHBOARD →` (`https://dropout-risk-system.streamlit.app`)
     - `VIEW GITHUB →` (`https://github.com/anewsevile/dropout_risk_prediction`)

9. **Window 7: Footer / System Info (`credits.txt`)**:
   - `Built solo by Nia · Avishkar 2026 · Pillai HOC College`
   - `Powered by: Python · scikit-learn · Streamlit · Gemini`
   - Retro system checksum & license banner.

---

## Verification Plan

### Automated / Syntax & Functional Checks:
- Validate `index.html` structure, semantic HTML tags, link targets (`_blank`, `rel="noopener noreferrer"`).
- Test JavaScript event handlers: smooth scrolling, interactive risk score simulator calculation, live clock updates, window active focus states.
- Responsive design check across mobile (<600px), tablet (600px-1024px), and desktop (>1024px) viewport widths.

### Visual Quality Review:
- Verify strict adherence to color palette:
  - Background: `#B8DDE1`
  - Title bars: `#1A1A1A` with `#FF5F57`, `#FFBD2E`, `#28CA41` dots
  - Shadows: `4px 4px 0px #1A1A1A`
  - Font: Space Mono loaded from Google Fonts
- No prohibited clichés (no purple on dark, no blurry gradients, no default rounded cards).
