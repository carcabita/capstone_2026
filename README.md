# ⚽ Football Performance Analysis & Scouting Platform

> **Capstone Project — Bachelor in Data & Business Analytics**  
> IEU School of Science and Technology · 2026  
> Author: Pablo Cárcaba García · Supervisor: Alberto Martín Izquierdo

---

## Overview

A data-driven desktop scouting application that helps football clubs identify squad weaknesses from match loss data and recruit the players best suited to address them — within budget, age, and tactical constraints.

The platform integrates four analytical modules into a single end-to-end pipeline:

| Module | Description |
|--------|-------------|
| 🔍 **Pattern Recognition** | Detects tactical weaknesses from a team's losing matches across 8 analytical dimensions |
| 🏆 **Archetype-Based Scoring (ABS)** | Evaluates players across 21 position-specific archetypes using percentile normalisation over 100+ FBRef statistics |
| 🤝 **k-NN Similarity Matrix** | Retrieves statistically similar players via ball-tree indexed Euclidean distance in z-score standardised space |
| 📈 **Semi-Markov Monte Carlo Simulation** | Projects a player's seasonal performance through 6 discrete form states including injury and suspension mechanics |

---

## Screenshots

<img width="1919" height="1125" alt="image" src="https://github.com/user-attachments/assets/bce70d56-57d8-40a0-a90f-dac0a6ce543a" />
<img width="1919" height="1123" alt="image" src="https://github.com/user-attachments/assets/97a9eb1e-82c6-4b39-8467-711e12865dbc" />
<img width="1919" height="1126" alt="image" src="https://github.com/user-attachments/assets/1281cbaf-0f24-4819-b332-7e38e8db332e" />
<img width="1051" height="1118" alt="image" src="https://github.com/user-attachments/assets/7ba1fe3d-7aed-4adf-89fe-b7a18cef5d43" />




---

## Tech Stack

- **Language:** Python 3
- **GUI:** PyQt6
- **Data sources:** FBRef (2024–25, top 5 leagues), Transfermarkt, Kaggle match dataset (FBRef scraped)
- **Key libraries:**

```
pandas · numpy · scikit-learn · rapidfuzz · matplotlib · reportlab · PyQt6
```

---

## Project Structure

```
capstone_2026/
│
├── main.py                      # Application entry point — launches PyQt6 GUI
├── analizar.py                  # Core engine: PatternAnalyzer, PlayerEvaluator,
│                                #   RecommendationEngine, k-NN module
├── markov_monte_carlo.py        # Semi-Markov Chain Monte Carlo simulator
│
├── data/
│   ├── player_profiles_final.csv    # Merged FBRef + Transfermarkt player dataset
│   └── matches.csv                  # Club match data (2000–2025)
│
├── preprocessing/
│   └── TFG_DATA_PREPPING.ipynb      # Jupyter notebook: UTF-8 encoding, schema
│                                    #   cleaning, RapidFuzz name matching
│
└── assets/                          # UI assets, icons
```

---

## Installation

**Requirements:** Python 3.10+, pip

```bash
# 1. Clone the repository
git clone https://github.com/carcabita/capstone_2026.git
cd capstone_2026

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the application
python main.py
```

> **Note:** PyQt6 requires Qt libraries. On most systems `pip install PyQt6` handles this automatically. On Linux you may need `sudo apt install libxcb-xinerama0`.

---

## How It Works

### 1 — Load your data
Use the left panel to load a **matches file** (CSV or XLSX) and a **players file**. The application auto-detects delimiters and formats.

### 2 — Select a team and season
The platform filters the match dataset to the selected team's losses and runs the `PatternAnalyzer`.

### 3 — Review detected patterns
The **Detected Issues** tab lists all tactical weaknesses found, categorised by dimension (defensive, offensive, tactical, physical, mental, set piece, game management, opponent-specific) and ranked by severity.

### 4 — Generate recommendations
Select the issues to address. The `RecommendationEngine` maps each issue to relevant archetypes and scores all eligible players using the **Archetype-Based Scoring** system. Recommendations appear as interactive player cards with Transfermarkt images.

### 5 — Explore similar players (k-NN)
From any player card, open the **similarity panel**, select the per-90 metrics to compare on, and retrieve the *k* nearest statistical neighbours — useful for finding budget-accessible alternatives.

### 6 — Simulate a signing
Click **Simulate Season** on any player card to run the **Semi-Markov Monte Carlo** engine (1,000 seasons by default). The output shows expected goals, assists, appearances, cards, and progressive actions with best/expected/worst-case scenarios.

### 7 — Export
Download the recommendations shortlist as **CSV** or export a full **PDF report** (visualisations + patterns + player table) via ReportLab, ready to share with scouts and sporting directors.

---

## Methodology Highlights

### Percentile-based scoring vs. min-max

The platform offers both normalisation methods but defaults to **percentile ranking**:

$$\hat{x}_i = \frac{\text{rank}(x_i)}{n}, \quad \hat{x}_i \in (0, 1)$$

Percentile ranking is robust to the outlier compression that makes min-max scores unreliable in football contexts (e.g. Mbappé's goal rate collapsing every other striker's score toward zero). It is the industry standard adopted by FBRef, StatsBomb, and SofaScore.

### Intra-positional normalisation

Players are ranked **against peers in their own positional group**, not the full population. A centre-back's tackle rate is percentiled among centre-backs only, preventing cross-positional contamination.

### Archetype-Based Scoring (ABS)

Each archetype score is a weighted linear combination of position-relevant normalised per-90 statistics:

$$S_a(p) = \sum_i w_i \cdot P(m_i(p)), \quad \sum_i w_i = 1$$

Weights were selected by expert assessment informed by Football Manager's documented key statistics per position. The open-source codebase allows users to adjust weights directly.

### k-NN with ball-tree indexing

Players are represented as vectors in ℝᵈ (up to 20 per-90 features across 4 domains). Features are z-score standardised before distance computation:

$$z_{ij} = \frac{x_{ij} - \mu_j}{\sigma_j}$$

$$d(p, q) = \sqrt{\sum_{j=1}^{d}(z_{pj} - z_{qj})^2}$$

A **ball-tree** spatial index reduces query complexity from O(dn) to O(d log n), enabling sub-second retrieval across the full player pool.

### Semi-Markov Chain Monte Carlo

The simulator models six performance states with position-specific transition matrices:

| State | Description |
|-------|-------------|
| `HOT` | Above-baseline form; streak dynamics increase persistence |
| `GOOD` | Season-average baseline |
| `COLD` | Below-average; reflects fatigue, loss of confidence |
| `INJURED` | Absent; duration sampled from age/position-calibrated distributions |
| `SUSPENDED` | Absent; yellow card accumulation triggers automatic bans |
| `RECOVERY` | Transitional; reduced minutes on return from injury |

---

## Data Sources

| Dataset | Source | Description |
|---------|--------|-------------|
| Football Players Stats 2024–25 | FBRef / Kaggle | 100+ statistical dimensions per player, top 5 European leagues |
| Player Profiles | Transfermarkt / Kaggle | Market value, contract expiry, preferred foot, player images |
| Club Match Data 2000–2025 | Kaggle | Results, shots, ELO ratings, half-time scores |

> Raw data files are not included in this repository due to licensing restrictions. See the **Data** section of the thesis for download links and preprocessing instructions.

---

## Limitations

- Static dataset (2024–25 season); no live API integration
- Pattern thresholds are hard-coded against league averages — ML-driven detection is a planned future development
- Simulation assumes 90 minutes played per active match (minutes model planned)
- Fixture difficulty not factored into simulation outputs

---

## Future Work

- **ML pattern detection** — XGBoost + SHAP to replace hard-coded thresholds
- **Cloud deployment** — REST API backend + web frontend; SQLite/PostgreSQL persistence
- **Interactive visualisations** — Plotly/Altair charts, radar comparison views, shot maps
- **NLP scouting reports** — LLM-generated player narrative from statistical + Transfermarkt profile data
- **Injury history integration** — Individual injury probability conditioning via Opta/StatsBomb API
- **Learned archetype weights** — PCA or supervised regression to replace expert-assigned weights
- **Soft multi-role archetypes** — Continuous archetype membership scores for multi-positional players

---

## Academic Context

This project was developed as a Bachelor's Capstone (TFG) at IE University, Madrid. The full thesis document covers the literature review, detailed methodology, results, conclusions, and references.

**Key citations:**
- Zimmerman & Zumbo (2005) — percentile transformation methodology
- Decroos et al. / VAEP (2019) — multi-dimensional player evaluation
- Lahvička (2015) & Stock et al. (2021) — Monte Carlo methods in sports
- Bergkamp et al. (2022) — human–data complementarity in scouting

---

## License

This project is released for academic and educational purposes.  
© 2026 Pablo Cárcaba García — IE University
