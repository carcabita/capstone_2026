"""
main.py
============
PyQt6 Desktop Application for Football Performance Analyzer & Recommender
"""

import sys
import pandas as pd
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QComboBox, QSpinBox, QTableWidget, QTableWidgetItem,
    QFileDialog, QMessageBox, QProgressDialog, QTabWidget, QGroupBox,
    QRadioButton, QButtonGroup, QListWidget, QLineEdit, QTextEdit,
    QSplitter, QHeaderView, QScrollArea, QGridLayout, QFrame, QCheckBox, QSlider, QDialog
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QUrl, QByteArray
from PyQt6.QtGui import QFont, QColor, QPixmap, QImage, QPainter, QBrush, QPen
from PyQt6.QtNetwork import QNetworkAccessManager, QNetworkRequest, QNetworkReply

# Matplotlib imports for charts
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

# PDF generation imports
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import io
from datetime import datetime

# Import analyzer functions
from analizar import (
    load_matches_data,
    load_players_data,
    clean_data,
    compute_team_table,
    PatternAnalyzer,
    PlayerEvaluator,
    RecommendationEngine,
    ARCHETYPE_DEFINITIONS,
    parse_age_range,
    get_team_statistics,
    get_all_issues,
    calculate_betting_odds_analysis
)

# Import Markov Monte Carlo
from markov_monte_carlo import MonteCarloSimulator, IMPORTANCE_PROFILES

# KNN similarity dependencies
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors


# ===================== Per-90 Column Definitions =====================
# These map exactly to the `_p90` columns computed by
# PlayerEvaluator._calculate_per90_metrics() in analizar.py.
# Euclidean distance is used (rather than cosine) because per-90 stats are on
# ratio scales whose magnitudes carry real meaning – a player producing
# 0.5 goals/90 is categorically different from one producing 0.0, not merely
# pointing in a different direction.
PER90_COLS = [
    'Gls_p90', 'Ast_p90', 'xG_p90', 'npxG_p90', 'xAG_p90',
    'Sh_p90', 'SoT_p90',
    'PrgC_p90', 'PrgP_p90', 'PrgR_p90',
    'Tkl_p90', 'TklW_p90', 'Int_p90', 'Blocks_p90', 'Clr_p90',
    'SCA_p90', 'GCA_p90',
    'Touches_p90', 'Carries_p90',
    'Recov_p90',
]

# Human-readable labels for every PER90_COLS entry (used in checkboxes + chips)
PER90_LABELS = {
    'Gls_p90':    'Goals/90',
    'Ast_p90':    'Assists/90',
    'xG_p90':     'xG/90',
    'npxG_p90':   'npxG/90',
    'xAG_p90':    'xAG/90',
    'Sh_p90':     'Shots/90',
    'SoT_p90':    'SoT/90',
    'PrgC_p90':   'Prog Carries/90',
    'PrgP_p90':   'Prog Passes/90',
    'PrgR_p90':   'Prog Runs/90',
    'Tkl_p90':    'Tackles/90',
    'TklW_p90':   'Tkl Won/90',
    'Int_p90':    'Interceptions/90',
    'Blocks_p90': 'Blocks/90',
    'Clr_p90':    'Clearances/90',
    'SCA_p90':    'SCA/90',
    'GCA_p90':    'GCA/90',
    'Touches_p90':'Touches/90',
    'Carries_p90':'Carries/90',
    'Recov_p90':  'Recoveries/90',
}

# Market-value column name candidates (evaluated_players may use any of these)
_VALUE_COLS = ['market_value', 'Market Value', 'Value', 'Transfer Value', 'market_val', 'value']

# Budget dropdown options: (display label, max value in euros | None = no limit)
_BUDGET_OPTIONS = [
    ('No limit',  None),
    ('€5M',        5_000_000),
    ('€10M',      10_000_000),
    ('€20M',      20_000_000),
    ('€30M',      30_000_000),
    ('€50M',      50_000_000),
    ('€75M',      75_000_000),
    ('€100M',    100_000_000),
    ('€150M',    150_000_000),
    ('€200M',    200_000_000),
]


# ===================== KNN Similarity Engine =====================

class _KNNCache:
    """
    Module-level cache for the fitted StandardScaler + NearestNeighbors index.
    The model is invalidated and rebuilt only when the candidate DataFrame
    changes (detected via a lightweight hash of its length and index).
    """

    def __init__(self):
        self._hash: int | None = None
        self._scaler: StandardScaler | None = None
        self._knn: NearestNeighbors | None = None
        self._valid_index = None   # pandas Index of rows that survived NaN-drop
        self._scaled_matrix = None # cached scaled feature matrix

    @staticmethod
    def _pool_hash(df: pd.DataFrame, cols: list) -> int:
        available = [c for c in cols if c in df.columns]
        return hash((len(df), tuple(df.index.tolist()), tuple(available)))

    def get(self, candidates_df: pd.DataFrame, per90_cols: list):
        """Return (knn, scaler, valid_index, scaled_matrix) if cache is fresh, else None."""
        h = self._pool_hash(candidates_df, per90_cols)
        if h == self._hash and self._knn is not None:
            return self._knn, self._scaler, self._valid_index, self._scaled_matrix
        return None

    def store(self, candidates_df, per90_cols, knn, scaler, valid_index, scaled_matrix):
        self._hash = self._pool_hash(candidates_df, per90_cols)
        self._knn = knn
        self._scaler = scaler
        self._valid_index = valid_index
        self._scaled_matrix = scaled_matrix


# Single shared cache instance for the process lifetime
_knn_cache = _KNNCache()


def build_per90_feature_matrix(players_df: pd.DataFrame, per90_cols: list):
    """
    Build a numeric feature matrix from whichever per90 columns exist in
    `players_df`.

    Strategy for missing values:
      • Rows where *all* features are NaN are dropped entirely (unusable players).
      • Remaining NaNs are imputed with the column median so the scaler
        receives no NaN inputs.

    Returns
    -------
    feature_matrix : np.ndarray  shape (n_valid, n_features), or None
    valid_index    : pd.Index     – positional labels of retained rows
    used_cols      : list[str]    – columns actually included
    """
    used_cols = [c for c in per90_cols if c in players_df.columns]
    if not used_cols:
        return None, None, []

    sub = players_df[used_cols].apply(pd.to_numeric, errors='coerce')
    sub = sub.dropna(how='all')           # remove fully-empty rows
    sub = sub.fillna(sub.median())        # impute remaining NaNs

    return sub.values, sub.index, used_cols


def fit_knn_model(feature_matrix: np.ndarray, k: int = 6):
    """
    Fit a StandardScaler then a ball-tree NearestNeighbors on the feature
    matrix.  k is set to k+1 internally to account for the player being their
    own nearest neighbour.

    Returns
    -------
    knn           : fitted NearestNeighbors
    scaler        : fitted StandardScaler
    scaled_matrix : np.ndarray – the scaled features (reused for queries)
    """
    scaler = StandardScaler()
    scaled = scaler.fit_transform(feature_matrix)
    n_neighbors = min(k + 1, len(feature_matrix))
    knn = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean',
                           algorithm='ball_tree')
    knn.fit(scaled)
    return knn, scaler, scaled


def get_similar_players(
    player_name: str,
    candidates_df: pd.DataFrame,
    per90_cols: list = None,
    k: int = 5,
) -> list:
    """
    Find the top-k players most similar to `player_name` within `candidates_df`
    using Euclidean distance in standardised per-90 feature space.

    The fitted scaler + NearestNeighbors index are cached in `_knn_cache` and
    only rebuilt when `candidates_df` changes (detected by hash).

    Parameters
    ----------
    player_name   : name to look up (matched case-insensitively after strip)
    candidates_df : filtered player pool (should exclude target team if desired)
    per90_cols    : columns to use; defaults to module-level PER90_COLS
    k             : number of similar players to return

    Returns
    -------
    list of {'row': pd.Series, 'distance': float} dicts, length ≤ k
    """
    if per90_cols is None:
        per90_cols = PER90_COLS

    cached = _knn_cache.get(candidates_df, per90_cols)
    if cached is not None:
        knn, scaler, valid_index, scaled_matrix = cached
        # Rebuild the raw matrix just to locate the query row (cheap)
        feat_matrix, _, used_cols = build_per90_feature_matrix(candidates_df, per90_cols)
    else:
        feat_matrix, valid_index, used_cols = build_per90_feature_matrix(
            candidates_df, per90_cols
        )
        if feat_matrix is None or len(feat_matrix) < 2:
            return []
        knn, scaler, scaled_matrix = fit_knn_model(feat_matrix, k=k)
        _knn_cache.store(candidates_df, per90_cols, knn, scaler, valid_index, scaled_matrix)

    if feat_matrix is None or len(feat_matrix) < 2:
        return []

    valid_df = candidates_df.loc[valid_index]
    name_clean = player_name.strip().lower()
    mask = valid_df['Player'].str.strip().str.lower() == name_clean
    if not mask.any():
        return []

    target_pos = int(np.where(mask.values)[0][0])
    query_vec = scaled_matrix[target_pos].reshape(1, -1)
    n_query = min(k + 1, len(feat_matrix))
    distances, indices = knn.kneighbors(query_vec, n_neighbors=n_query)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == target_pos:          # skip the player themselves
            continue
        results.append({
            'row': valid_df.iloc[idx],
            'distance': float(dist),
        })
        if len(results) >= k:
            break

    return results


# ===================== Chart Widgets =====================
class PieChartWidget(FigureCanvasQTAgg):
    """Widget for displaying win rate pie chart"""
    def __init__(self, wins, draws, losses, parent=None):
        fig = Figure(figsize=(5, 4), facecolor='white')
        super().__init__(fig)
        self.setParent(parent)
        
        ax = fig.add_subplot(111)
        
        # Data
        sizes = [wins, draws, losses]
        labels = [f'Wins\n{wins}', f'Draws\n{draws}', f'Losses\n{losses}']
        colors = ['#28a745', '#6c757d', '#dc3545']
        explode = (0.1, 0, 0)  # Explode wins slice
        
        # Create pie chart
        wedges, texts, autotexts = ax.pie(
            sizes, 
            explode=explode, 
            labels=labels, 
            colors=colors, 
            autopct='%1.1f%%',
            shadow=True, 
            startangle=90,
            textprops={'fontsize': 10, 'weight': 'bold'}
        )
        
        # Make percentage text white for better contrast
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(11)
        
        ax.axis('equal')
        ax.set_title('Win Rate Distribution', fontsize=12, weight='bold', pad=20)
        
        fig.tight_layout()


class HomeAwayChartWidget(FigureCanvasQTAgg):
    """Widget for displaying home vs away comparison bar chart"""
    def __init__(self, home_stats, away_stats, parent=None):
        fig = Figure(figsize=(6, 4), facecolor='white')
        super().__init__(fig)
        self.setParent(parent)
        
        ax = fig.add_subplot(111)
        
        # Data
        categories = ['Win Rate', 'Goals For', 'Goals Against']
        x = np.arange(len(categories))
        width = 0.35
        
        # Create bars
        bars1 = ax.bar(x - width/2, home_stats, width, label='Home', color='#0066cc', alpha=0.8)
        bars2 = ax.bar(x + width/2, away_stats, width, label='Away', color='#ff6600', alpha=0.8)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}',
                       ha='center', va='bottom', fontsize=9, weight='bold')
        
        # Styling
        ax.set_ylabel('Value', fontsize=10, weight='bold')
        ax.set_title('Home vs Away Performance', fontsize=12, weight='bold', pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(categories, fontsize=10)
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        fig.tight_layout()


class BettingOddsChartWidget(FigureCanvasQTAgg):
    """Widget for displaying expected vs actual results based on betting odds"""
    def __init__(self, expected_data, actual_data, parent=None):
        fig = Figure(figsize=(7, 5), facecolor='white')
        super().__init__(fig)
        self.setParent(parent)
        
        ax = fig.add_subplot(111)
        
        # Data
        categories = ['Wins', 'Draws', 'Losses']
        x = np.arange(len(categories))
        width = 0.35
        
        # Create bars
        bars1 = ax.bar(x - width/2, expected_data, width, label='Expected (Odds)', 
                       color='#6c757d', alpha=0.7)
        bars2 = ax.bar(x + width/2, actual_data, width, label='Actual', 
                       color='#0066cc', alpha=0.8)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}',  # Show expected as decimal
                   ha='center', va='bottom', fontsize=9, weight='bold')
        
        for bar in bars2:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',  # Show actual as integer
                   ha='center', va='bottom', fontsize=10, weight='bold')
        
        # Styling
        ax.set_ylabel('Number of Games', fontsize=11, weight='bold')
        ax.set_title('Expected vs Actual Results (Based on Betting Odds)', 
                    fontsize=12, weight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(categories, fontsize=11)
        ax.legend(fontsize=11, loc='upper right')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add zero line
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
        
        # Add more space at top for labels
        ax.set_ylim(bottom=0, top=max(max(expected_data), max(actual_data)) * 1.15)
        
        fig.tight_layout(pad=2.0)


class FormEloTimelineWidget(FigureCanvasQTAgg):
    """Widget for displaying form and Elo rating evolution over the season"""
    def __init__(self, df_team, parent=None):
        fig = Figure(figsize=(10, 5), facecolor='white')
        super().__init__(fig)
        self.setParent(parent)
        
        ax1 = fig.add_subplot(111)
        ax2 = ax1.twinx()  # Secondary y-axis
        
        # Get match numbers
        match_nums = list(range(1, len(df_team) + 1))
        
        # Elo ratings (left axis)
        elo_values = []
        for _, row in df_team.iterrows():
            if row.get('is_home', False):
                elo_values.append(row.get('HomeElo', 0))
            else:
                elo_values.append(row.get('AwayElo', 0))
        
        # Form values (right axis)
        form3_values = []
        form5_values = []
        for _, row in df_team.iterrows():
            if row.get('is_home', False):
                form3_values.append(row.get('Form3Home', 0))
                form5_values.append(row.get('Form5Home', 0))
            else:
                form3_values.append(row.get('Form3Away', 0))
                form5_values.append(row.get('Form5Away', 0))
        
        # Plot Elo on left axis
        line1 = ax1.plot(match_nums, elo_values, color='#0066cc', linewidth=2.5, 
                        label='Elo Rating', marker='o', markersize=4, alpha=0.8)
        ax1.set_xlabel('Match Number', fontsize=11, weight='bold')
        ax1.set_ylabel('Elo Rating', fontsize=11, weight='bold', color='#0066cc')
        ax1.tick_params(axis='y', labelcolor='#0066cc')
        ax1.grid(axis='y', alpha=0.3, linestyle='--', color='#0066cc')
        
        # Plot Form on right axis
        line2 = ax2.plot(match_nums, form3_values, color='#28a745', linewidth=2, 
                        linestyle='--', label='Form (Last 3)', marker='s', markersize=3, alpha=0.7)
        line3 = ax2.plot(match_nums, form5_values, color='#ffc107', linewidth=2, 
                        linestyle='--', label='Form (Last 5)', marker='^', markersize=3, alpha=0.7)
        ax2.set_ylabel('Form Points', fontsize=11, weight='bold', color='#28a745')
        ax2.tick_params(axis='y', labelcolor='#28a745')
        
        # Title
        ax1.set_title('Team Strength & Momentum Evolution', fontsize=13, weight='bold', pad=15)
        
        # Combined legend
        lines = line1 + line2 + line3
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper left', fontsize=10)
        
        # Highlight wins/losses with background color
        for i, (_, row) in enumerate(df_team.iterrows()):
            result = row.get('Result', '')
            if result == 'W':
                ax1.axvspan(i+0.5, i+1.5, alpha=0.1, color='green')
            elif result == 'L':
                ax1.axvspan(i+0.5, i+1.5, alpha=0.1, color='red')
        
        fig.tight_layout()


class HalfTimeFullTimeWidget(FigureCanvasQTAgg):
    """Widget showing half-time vs full-time performance analysis"""
    def __init__(self, df_team, parent=None):
        fig = Figure(figsize=(8, 6), facecolor='white')
        super().__init__(fig)
        self.setParent(parent)
        
        ax = fig.add_subplot(111)
        
        # Calculate HT and FT goal differences from team perspective
        ht_diffs = []
        ft_diffs = []
        colors = []
        sizes = []
        
        for _, row in df_team.iterrows():
            is_home = row.get('is_home', False)
            
            if is_home:
                ht_diff = row.get('HTHome', 0) - row.get('HTAway', 0)
                ft_diff = row.get('FTHome', 0) - row.get('FTAway', 0)
                total_goals = row.get('FTHome', 0) + row.get('FTAway', 0)
            else:
                ht_diff = row.get('HTAway', 0) - row.get('HTHome', 0)
                ft_diff = row.get('FTAway', 0) - row.get('FTHome', 0)
                total_goals = row.get('FTHome', 0) + row.get('FTAway', 0)
            
            ht_diffs.append(ht_diff)
            ft_diffs.append(ft_diff)
            sizes.append(total_goals * 50 + 50)  # Size based on total goals
            
            # Color by result
            result = row.get('Result', '')
            if result == 'W':
                colors.append('#28a745')
            elif result == 'D':
                colors.append('#ffc107')
            else:
                colors.append('#dc3545')
        
        # Scatter plot
        scatter = ax.scatter(ht_diffs, ft_diffs, c=colors, s=sizes, alpha=0.6, edgecolors='black', linewidth=1)
        
        # Diagonal line (no change HT to FT)
        max_val = max(max(abs(min(ht_diffs)), abs(max(ht_diffs))), 
                      max(abs(min(ft_diffs)), abs(max(ft_diffs)))) + 1
        ax.plot([-max_val, max_val], [-max_val, max_val], 'k--', alpha=0.3, linewidth=2, label='No Change')
        
        # Quadrant lines
        ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)
        ax.axvline(x=0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)
        
        # Labels
        ax.set_xlabel('Half-Time Goal Difference', fontsize=11, weight='bold')
        ax.set_ylabel('Full-Time Goal Difference', fontsize=11, weight='bold')
        ax.set_title('Half-Time to Full-Time Performance\n(Size = Total Goals)', 
                    fontsize=12, weight='bold', pad=15)
        
        # Add quadrant annotations
        ax.text(max_val*0.7, max_val*0.7, 'Leading &\nStayed Ahead', 
               ha='center', va='center', fontsize=9, alpha=0.5, style='italic')
        ax.text(-max_val*0.7, max_val*0.7, 'Comeback\nWins!', 
               ha='center', va='center', fontsize=9, alpha=0.5, style='italic', color='green')
        ax.text(max_val*0.7, -max_val*0.7, 'Collapsed\n2nd Half', 
               ha='center', va='center', fontsize=9, alpha=0.5, style='italic', color='red')
        ax.text(-max_val*0.7, -max_val*0.7, 'Behind &\nStayed Behind', 
               ha='center', va='center', fontsize=9, alpha=0.5, style='italic')
        
        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#28a745', label='Win'),
            Patch(facecolor='#ffc107', label='Draw'),
            Patch(facecolor='#dc3545', label='Loss')
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=10)
        
        ax.grid(alpha=0.3, linestyle='--')
        ax.set_aspect('equal')
        
        fig.tight_layout()


class HomeAwayRadarWidget(FigureCanvasQTAgg):
    """Widget for comprehensive home vs away radar comparison"""
    def __init__(self, home_stats, away_stats, parent=None):
        fig = Figure(figsize=(7, 7), facecolor='white')
        super().__init__(fig)
        self.setParent(parent)
        
        ax = fig.add_subplot(111, projection='polar')
        
        # 8 categories
        categories = ['Win Rate %', 'Goals\nScored', 'Goals\nConceded', 'Shots', 
                     'Shot\nAccuracy %', 'Corners', 'Fouls', 'Cards']
        N = len(categories)
        
        # Get values
        home_values = [
            home_stats.get('win_rate', 0) * 100,
            home_stats.get('goals_for', 0),
            home_stats.get('goals_against', 0),
            home_stats.get('shots', 0),
            home_stats.get('shot_accuracy', 0),
            home_stats.get('corners', 0),
            home_stats.get('fouls', 0),
            home_stats.get('cards', 0)
        ]
        
        away_values = [
            away_stats.get('win_rate', 0) * 100,
            away_stats.get('goals_for', 0),
            away_stats.get('goals_against', 0),
            away_stats.get('shots', 0),
            away_stats.get('shot_accuracy', 0),
            away_stats.get('corners', 0),
            away_stats.get('fouls', 0),
            away_stats.get('cards', 0)
        ]
        
        # Normalize to 0-10 scale for better visualization
        max_values = [100, 3, 3, 20, 100, 10, 20, 5]  # Max expected values
        home_normalized = [min(h / m * 10, 10) for h, m in zip(home_values, max_values)]
        away_normalized = [min(a / m * 10, 10) for a, m in zip(away_values, max_values)]
        
        # Angles for each axis
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        home_normalized += home_normalized[:1]  # Close the plot
        away_normalized += away_normalized[:1]
        angles += angles[:1]
        
        # Plot
        ax.plot(angles, home_normalized, 'o-', linewidth=2.5, color='#0066cc', label='Home', alpha=0.8)
        ax.fill(angles, home_normalized, alpha=0.15, color='#0066cc')
        
        ax.plot(angles, away_normalized, 'o-', linewidth=2.5, color='#ff6600', label='Away', alpha=0.8)
        ax.fill(angles, away_normalized, alpha=0.15, color='#ff6600')
        
        # Fix axis
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=9)
        ax.set_ylim(0, 10)
        ax.set_yticks([2, 4, 6, 8, 10])
        ax.set_yticklabels(['2', '4', '6', '8', '10'], fontsize=8, alpha=0.5)
        ax.grid(True, alpha=0.3)
        
        # Title and legend
        ax.set_title('Home vs Away Performance Radar\n(Normalized Scale)', 
                    fontsize=12, weight='bold', pad=20, y=1.08)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
        
        fig.tight_layout()


class ShotEfficiencyQuadrantWidget(FigureCanvasQTAgg):
    """Widget showing shot efficiency quadrant analysis"""
    def __init__(self, df_team, parent=None):
        fig = Figure(figsize=(8, 6), facecolor='white')
        super().__init__(fig)
        self.setParent(parent)
        
        ax = fig.add_subplot(111)
        
        # Calculate metrics for each match
        accuracies = []
        goals_per_shot = []
        colors = []
        
        for _, row in df_team.iterrows():
            is_home = row.get('is_home', False)
            
            if is_home:
                shots = row.get('HomeShots', 0)
                target = row.get('HomeTarget', 0)
                goals = row.get('FTHome', 0)
            else:
                shots = row.get('AwayShots', 0)
                target = row.get('AwayTarget', 0)
                goals = row.get('FTAway', 0)
            
            # Calculate metrics
            accuracy = (target / shots * 100) if shots > 0 else 0
            gps = (goals / shots) if shots > 0 else 0
            
            accuracies.append(accuracy)
            goals_per_shot.append(gps)
            
            # Color by result
            result = row.get('Result', '')
            if result == 'W':
                colors.append('#28a745')
            elif result == 'D':
                colors.append('#ffc107')
            else:
                colors.append('#dc3545')
        
        # Scatter plot
        ax.scatter(accuracies, goals_per_shot, c=colors, s=100, alpha=0.6, edgecolors='black', linewidth=1)
        
        # Add quadrant lines at median
        median_accuracy = np.median([a for a in accuracies if a > 0])
        median_gps = np.median([g for g in goals_per_shot if g > 0])
        
        ax.axhline(y=median_gps, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax.axvline(x=median_accuracy, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        
        # Labels
        ax.set_xlabel('Shot Accuracy (%)', fontsize=11, weight='bold')
        ax.set_ylabel('Goals per Shot', fontsize=11, weight='bold')
        ax.set_title('Shot Efficiency Analysis\n(4 Quadrants)', fontsize=12, weight='bold', pad=15)
        
        # Quadrant labels
        max_x = max(accuracies) if accuracies else 100
        max_y = max(goals_per_shot) if goals_per_shot else 0.5
        
        ax.text(median_accuracy + (max_x - median_accuracy)/2, median_gps + (max_y - median_gps)/2,
               'EFFICIENT\n& ACCURATE', ha='center', va='center', fontsize=9, 
               weight='bold', alpha=0.4, color='green')
        ax.text(median_accuracy/2, median_gps + (max_y - median_gps)/2,
               'Clinical\n(Lucky)', ha='center', va='center', fontsize=8, 
               style='italic', alpha=0.4)
        ax.text(median_accuracy + (max_x - median_accuracy)/2, median_gps/2,
               'Unlucky\n(Wasteful)', ha='center', va='center', fontsize=8, 
               style='italic', alpha=0.4, color='orange')
        ax.text(median_accuracy/2, median_gps/2,
               'POOR', ha='center', va='center', fontsize=9, 
               weight='bold', alpha=0.4, color='red')
        
        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#28a745', label='Win'),
            Patch(facecolor='#ffc107', label='Draw'),
            Patch(facecolor='#dc3545', label='Loss')
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=10)
        
        ax.grid(alpha=0.3, linestyle='--')
        ax.set_xlim(0, max_x * 1.1)
        ax.set_ylim(0, max_y * 1.1)
        
        fig.tight_layout()


class OddsVsRealityWidget(FigureCanvasQTAgg):
    """Widget showing betting odds vs actual results"""
    def __init__(self, df_team, parent=None):
        fig = Figure(figsize=(8, 6), facecolor='white')
        super().__init__(fig)
        self.setParent(parent)
        
        ax = fig.add_subplot(111)
        
        # Collect data
        team_odds = []  # Odds for team to win
        opponent_odds = []  # Odds for opponent to win
        colors = []
        sizes = []
        
        for _, row in df_team.iterrows():
            is_home = row.get('is_home', False)
            
            if is_home:
                team_odd = row.get('OddHome', 0)
                opp_odd = row.get('OddAway', 0)
            else:
                team_odd = row.get('OddAway', 0)
                opp_odd = row.get('OddHome', 0)
            
            if team_odd > 0 and opp_odd > 0:  # Valid odds
                team_odds.append(team_odd)
                opponent_odds.append(opp_odd)
                
                # Size based on odds discrepancy
                discrepancy = abs(team_odd - opp_odd)
                sizes.append(discrepancy * 10 + 50)
                
                # Color by result
                result = row.get('Result', '')
                if result == 'W':
                    colors.append('#28a745')
                elif result == 'D':
                    colors.append('#ffc107')
                else:
                    colors.append('#dc3545')
        
        if len(team_odds) == 0:
            # No odds data
            ax.text(0.5, 0.5, 'No betting odds data available', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            return
        
        # Scatter plot
        ax.scatter(team_odds, opponent_odds, c=colors, s=sizes, alpha=0.6, edgecolors='black', linewidth=1)
        
        # Diagonal line (even odds)
        max_odd = max(max(team_odds), max(opponent_odds))
        ax.plot([0, max_odd], [0, max_odd], 'k--', alpha=0.3, linewidth=2, label='Even Odds')
        
        # Labels
        ax.set_xlabel('Team Odds (to win)', fontsize=11, weight='bold')
        ax.set_ylabel('Opponent Odds (to win)', fontsize=11, weight='bold')
        ax.set_title('Betting Odds vs Reality\n(Size = Odds Discrepancy)', 
                    fontsize=12, weight='bold', pad=15)
        
        # Add regions
        ax.text(max_odd*0.3, max_odd*0.7, 'Team\nFavorite', 
               ha='center', va='center', fontsize=10, alpha=0.4, style='italic')
        ax.text(max_odd*0.7, max_odd*0.3, 'Team\nUnderdog', 
               ha='center', va='center', fontsize=10, alpha=0.4, style='italic')
        
        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#28a745', label='Win'),
            Patch(facecolor='#ffc107', label='Draw'),
            Patch(facecolor='#dc3545', label='Loss')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
        
        ax.grid(alpha=0.3, linestyle='--')
        ax.set_xlim(0, max_odd * 1.1)
        ax.set_ylim(0, max_odd * 1.1)
        
        fig.tight_layout()


# ===================== Helper Functions for Enhanced Stats =====================
def calculate_offensive_defensive_metrics(df_team):
    """Calculate offensive and defensive metrics from team data"""
    metrics = {}
    
    # Offensive metrics
    metrics['goals_per_game'] = df_team['GF'].mean() if 'GF' in df_team.columns else 0
    metrics['shots_per_game'] = df_team['ShotsFor'].mean() if 'ShotsFor' in df_team.columns else 0
    metrics['shots_on_target_per_game'] = df_team['ShotsOnTargetFor'].mean() if 'ShotsOnTargetFor' in df_team.columns else 0
    
    # Shot accuracy
    if 'ShotsFor' in df_team.columns and 'ShotsOnTargetFor' in df_team.columns:
        total_shots = df_team['ShotsFor'].sum()
        total_on_target = df_team['ShotsOnTargetFor'].sum()
        metrics['shot_accuracy'] = (total_on_target / total_shots * 100) if total_shots > 0 else 0
    else:
        metrics['shot_accuracy'] = 0
    
    # Conversion rate
    if 'GF' in df_team.columns and 'ShotsOnTargetFor' in df_team.columns:
        total_goals = df_team['GF'].sum()
        total_on_target = df_team['ShotsOnTargetFor'].sum()
        metrics['conversion_rate'] = (total_goals / total_on_target * 100) if total_on_target > 0 else 0
    else:
        metrics['conversion_rate'] = 0
    
    # Defensive metrics
    metrics['goals_conceded_per_game'] = df_team['GA'].mean() if 'GA' in df_team.columns else 0
    metrics['clean_sheets'] = len(df_team[df_team['GA'] == 0]) if 'GA' in df_team.columns else 0
    metrics['shots_allowed_per_game'] = df_team['ShotsAgainst'].mean() if 'ShotsAgainst' in df_team.columns else 0
    
    # Discipline
    metrics['yellow_cards_per_game'] = df_team['YellowCards'].mean() if 'YellowCards' in df_team.columns else 0
    metrics['red_cards_total'] = df_team['RedCards'].sum() if 'RedCards' in df_team.columns else 0
    metrics['fouls_per_game'] = df_team['FoulsFor'].mean() if 'FoulsFor' in df_team.columns else 0
    
    return metrics


def calculate_home_away_stats(df_team):
    """Calculate separate stats for home and away games (enhanced for radar chart)"""
    home_games = df_team[df_team['is_home'] == True]
    away_games = df_team[df_team['is_home'] == False]
    
    def calc_stats(games, is_home=True):
        if len(games) == 0:
            return {
                'games': 0, 'wins': 0, 'draws': 0, 'losses': 0, 'win_rate': 0,
                'goals_for': 0, 'goals_against': 0, 'shots': 0, 'shot_accuracy': 0,
                'corners': 0, 'fouls': 0, 'cards': 0
            }
        
        wins = len(games[games['Result'] == 'W'])
        
        stats = {
            'games': len(games),
            'wins': wins,
            'draws': len(games[games['Result'] == 'D']),
            'losses': len(games[games['Result'] == 'L']),
            'win_rate': (wins / len(games)) if len(games) > 0 else 0,
            'goals_for': games['GF'].mean() if 'GF' in games.columns else 0,
            'goals_against': games['GA'].mean() if 'GA' in games.columns else 0,
        }
        
        # Additional stats for radar
        if is_home:
            stats['shots'] = games['HomeShots'].mean() if 'HomeShots' in games.columns else 0
            stats['shot_accuracy'] = (
                (games['HomeTarget'].sum() / games['HomeShots'].sum() * 100)
                if 'HomeTarget' in games.columns and 'HomeShots' in games.columns 
                and games['HomeShots'].sum() > 0
                else 0
            )
            stats['corners'] = games['HomeCorners'].mean() if 'HomeCorners' in games.columns else 0
            stats['fouls'] = games['HomeFouls'].mean() if 'HomeFouls' in games.columns else 0
            yellow = games['HomeYellow'].mean() if 'HomeYellow' in games.columns else 0
            red = games['HomeRed'].mean() if 'HomeRed' in games.columns else 0
            stats['cards'] = yellow + (red * 3)  # Weight red cards more
        else:
            stats['shots'] = games['AwayShots'].mean() if 'AwayShots' in games.columns else 0
            stats['shot_accuracy'] = (
                (games['AwayTarget'].sum() / games['AwayShots'].sum() * 100)
                if 'AwayTarget' in games.columns and 'AwayShots' in games.columns
                and games['AwayShots'].sum() > 0
                else 0
            )
            stats['corners'] = games['AwayCorners'].mean() if 'AwayCorners' in games.columns else 0
            stats['fouls'] = games['AwayFouls'].mean() if 'AwayFouls' in games.columns else 0
            yellow = games['AwayYellow'].mean() if 'AwayYellow' in games.columns else 0
            red = games['AwayRed'].mean() if 'AwayRed' in games.columns else 0
            stats['cards'] = yellow + (red * 3)
        
        return stats
    
    home_stats = calc_stats(home_games, is_home=True)
    away_stats = calc_stats(away_games, is_home=False)
    
    return home_stats, away_stats


def calculate_predictive_metrics(df_team):
    """Calculate predictive metrics like xG difference"""
    metrics = {}
    
    # Basic form calculation
    if len(df_team) >= 5:
        recent_5 = df_team.tail(5)
        recent_wins = len(recent_5[recent_5['Result'] == 'W'])
        recent_form = recent_wins / 5 * 100
        
        if recent_form >= 60:
            metrics['form_trend'] = "↗️ Improving"
            metrics['momentum_score'] = 7.5 + (recent_form - 60) / 10
        elif recent_form >= 40:
            metrics['form_trend'] = "→ Stable"
            metrics['momentum_score'] = 5.0 + (recent_form - 40) / 10
        else:
            metrics['form_trend'] = "↘️ Declining"
            metrics['momentum_score'] = 2.5 + recent_form / 10
    else:
        metrics['form_trend'] = "→ Insufficient data"
        metrics['momentum_score'] = 5.0
    
    # Expected points (simple calculation based on goal difference)
    if 'GF' in df_team.columns and 'GA' in df_team.columns:
        total_gf = df_team['GF'].sum()
        total_ga = df_team['GA'].sum()
        goal_diff = total_gf - total_ga
        
        # Simple model: each goal difference ~ 0.3 points
        expected_points = len(df_team) * 1.5 + (goal_diff * 0.3)
        
        # Actual points
        wins = len(df_team[df_team['Result'] == 'W'])
        draws = len(df_team[df_team['Result'] == 'D'])
        actual_points = wins * 3 + draws
        
        metrics['expected_points'] = round(expected_points, 1)
        metrics['actual_points'] = actual_points
        metrics['points_difference'] = actual_points - expected_points
        
        if metrics['points_difference'] > 3:
            metrics['performance_status'] = "Overperforming! 🔥"
        elif metrics['points_difference'] < -3:
            metrics['performance_status'] = "Underperforming ⚠️"
        else:
            metrics['performance_status'] = "As expected âœ“"
    else:
        metrics['expected_points'] = 0
        metrics['actual_points'] = 0
        metrics['points_difference'] = 0
        metrics['performance_status'] = "N/A"
    
    return metrics


def get_best_worst_matches(df_team):
    """Get best victories and worst defeats"""
    best_matches = []
    worst_matches = []
    
    if 'GF' in df_team.columns and 'GA' in df_team.columns:
        # Best victories (highest goal difference in wins)
        wins = df_team[df_team['Result'] == 'W'].copy()
        if len(wins) > 0:
            wins['goal_diff'] = wins['GF'] - wins['GA']
            wins_sorted = wins.sort_values('goal_diff', ascending=False)
            
            for idx, row in wins_sorted.head(3).iterrows():
                location = "Home" if row.get('is_home', False) else "Away"
                match = {
                    'opponent': row.get('opponent', 'Unknown'),
                    'score': f"{int(row['GF'])}-{int(row['GA'])}",
                    'location': location,
                    'goal_diff': int(row['goal_diff'])
                }
                best_matches.append(match)
        
        # Worst defeats (highest negative goal difference in losses)
        losses = df_team[df_team['Result'] == 'L'].copy()
        if len(losses) > 0:
            losses['goal_diff'] = losses['GA'] - losses['GF']
            losses_sorted = losses.sort_values('goal_diff', ascending=False)
            
            for idx, row in losses_sorted.head(3).iterrows():
                location = "Home" if row.get('is_home', False) else "Away"
                match = {
                    'opponent': row.get('opponent', 'Unknown'),
                    'score': f"{int(row['GF'])}-{int(row['GA'])}",
                    'location': location,
                    'goal_diff': int(row['goal_diff'])
                }
                worst_matches.append(match)
    
    return best_matches, worst_matches


# ===================== Worker Thread for Long Operations =====================
class AnalysisWorker(QThread):
    """Worker thread for running analysis without blocking UI"""
    finished = pyqtSignal(object, object, object)  # df_team, patterns, evaluated_players
    error = pyqtSignal(str)
    
    def __init__(self, matches_df, players_df, team, use_percentiles=False):
        super().__init__()
        self.matches_df = matches_df
        self.players_df = players_df
        self.team = team
        self.use_percentiles = use_percentiles
    
    def run(self):
        try:
            # Compute team table
            df_team = compute_team_table(self.matches_df, self.team)
            if df_team.empty:
                self.error.emit(f"No match data found for team: {self.team}")
                return
            
            # Analyze patterns
            analyzer = PatternAnalyzer(df_team, self.matches_df)
            patterns = analyzer.analyze_all_patterns()
            
            # Evaluate players with chosen scoring method
            evaluator = PlayerEvaluator(self.players_df, use_percentiles=self.use_percentiles)
            evaluated_players = evaluator.evaluate_all_players()
            
            if evaluated_players.empty:
                self.error.emit("No players met minimum criteria for evaluation.")
                return
            
            self.finished.emit(df_team, patterns, evaluated_players)
        except Exception as e:
            self.error.emit(f"Analysis error: {str(e)}")


class RecommendationWorker(QThread):
    """Worker thread for generating recommendations"""
    finished = pyqtSignal(object, object)  # recommendations_df, archetype_needs
    error = pyqtSignal(str)
    
    def __init__(self, custom_patterns, evaluated_players, team, top_n, age_min, age_max, 
                 preferred_foot, free_agents_only, max_budget, scoring_method="min-max"):
        super().__init__()
        self.custom_patterns = custom_patterns
        self.evaluated_players = evaluated_players
        self.team = team
        self.top_n = top_n
        self.age_min = age_min
        self.age_max = age_max
        self.preferred_foot = preferred_foot
        self.free_agents_only = free_agents_only
        self.max_budget = max_budget
        self.scoring_method = scoring_method
    
    def run(self):
        try:
            engine = RecommendationEngine(
                self.custom_patterns, 
                self.evaluated_players, 
                self.team,
                scoring_method=self.scoring_method
            )
            recommendations_df = engine.generate_recommendations(
                self.top_n,
                age_min=self.age_min,
                age_max=self.age_max,
                preferred_foot=self.preferred_foot,
                free_agents_only=self.free_agents_only,
                max_budget=self.max_budget
            )
            archetype_needs = getattr(engine, 'archetype_needs', {})
            self.finished.emit(recommendations_df, archetype_needs)
        except Exception as e:
            self.error.emit(f"Recommendation error: {str(e)}")


# ===================== Image Loader for Player Photos =====================
class ImageLoader:
    """Singleton class to manage image loading from URLs with caching"""
    _instance = None
    _cache = {}
    _network_manager = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ImageLoader, cls).__new__(cls)
            cls._network_manager = QNetworkAccessManager()
        return cls._instance
    
    def load_image(self, url, callback, size=(80, 80)):
        """
        Load image from URL asynchronously
        Args:
            url: Image URL string
            callback: Function to call with QPixmap when loaded
            size: Tuple of (width, height) for scaling
        """
        if not url or pd.isna(url) or str(url).strip() == "":
            # Return placeholder if no URL
            callback(self._create_placeholder(size))
            return
        
        url_str = str(url).strip()
        
        # Check cache first
        cache_key = f"{url_str}_{size[0]}x{size[1]}"
        if cache_key in self._cache:
            callback(self._cache[cache_key])
            return
        
        # Download image
        request = QNetworkRequest(QUrl(url_str))
        request.setAttribute(QNetworkRequest.Attribute.RedirectPolicyAttribute, 
                           QNetworkRequest.RedirectPolicy.NoLessSafeRedirectPolicy)
        
        reply = self._network_manager.get(request)
        reply.finished.connect(lambda: self._on_image_loaded(reply, callback, size, cache_key))
    
    def _on_image_loaded(self, reply, callback, size, cache_key):
        """Handle image download completion"""
        if reply.error() == QNetworkReply.NetworkError.NoError:
            image_data = reply.readAll()
            pixmap = QPixmap()
            if pixmap.loadFromData(image_data):
                # Scale image to requested size while maintaining aspect ratio
                scaled_pixmap = pixmap.scaled(size[0], size[1], 
                                             Qt.AspectRatioMode.KeepAspectRatio,
                                             Qt.TransformationMode.SmoothTransformation)
                self._cache[cache_key] = scaled_pixmap
                callback(scaled_pixmap)
            else:
                callback(self._create_placeholder(size))
        else:
            # Error loading image, use placeholder
            callback(self._create_placeholder(size))
        
        reply.deleteLater()
    
    def _create_placeholder(self, size):
        """Create a placeholder image for missing photos"""
        pixmap = QPixmap(size[0], size[1])
        pixmap.fill(QColor("#555555"))
        return pixmap


# ===================== Player Profile Dialog =====================
class PlayerProfileDialog(QDialog):
    """Dialog showing detailed player profile"""
    def __init__(self, player_data, parent=None, candidates_df=None):
        super().__init__(parent)
        self.parent_window = parent
        self.recommendation_data = player_data  # Basic recommendation info

        # Pool of players used for KNN similarity (the currently filtered set).
        # Falls back to parent's evaluated_players when not provided explicitly.
        if candidates_df is not None:
            self.candidates_df = candidates_df
        elif hasattr(parent, 'evaluated_players') and parent.evaluated_players is not None:
            self.candidates_df = parent.evaluated_players
        else:
            self.candidates_df = None

        # CRITICAL: Get FULL player stats from evaluated_players dataset
        self.player_data = self._get_full_player_data(player_data)

        self.setWindowTitle(f"Player Profile: {self.player_data.get('Player', 'Unknown')}")
        self.setMinimumSize(700, 600)
        self.setup_ui()
    
    def _get_full_player_data(self, recommendation_row):
        """
        Get full player stats from evaluated_players by walking up the parent
        chain until FootballAnalyzerApp (or any window that holds
        evaluated_players) is reached.

        This handles the case where the dialog is opened from another
        PlayerProfileDialog (e.g., clicking a Similar Alternatives chip),
        because PlayerProfileDialog itself does not store evaluated_players.
        """
        player_name = recommendation_row.get('Player', '')
        player_name_clean = player_name.strip().lower()

        window = self.parent_window
        while window is not None:
            ep = getattr(window, 'evaluated_players', None)
            if ep is not None:
                if 'Player' in ep.columns:
                    player_match = ep[
                        ep['Player'].str.strip().str.lower() == player_name_clean
                    ]
                    if len(player_match) > 0:
                        return player_match.iloc[0]
                break  # Found the holder but player absent — stop searching
            window = getattr(window, 'parent_window', None)

        print(
            f"⚠️  WARNING: Could not find full stats for '{player_name}' "
            f"in evaluated_players (searched {type(self.parent_window).__name__} chain)"
        )
        return recommendation_row

    def setup_ui(self):
        # Outer layout: scroll area on top, pinned buttons at the bottom.
        outer_layout = QVBoxLayout(self)
        outer_layout.setSpacing(0)
        outer_layout.setContentsMargins(0, 0, 0, 10)

        # --- Scrollable content area ---
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        scroll_content = QWidget()
        layout = QVBoxLayout(scroll_content)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 10)

        scroll_area.setWidget(scroll_content)
        outer_layout.addWidget(scroll_area, stretch=1)

        # Header section with player image and name
        header_layout = QHBoxLayout()
        header_layout.setSpacing(20)
        
        # Player image
        self.image_label = QLabel()
        self.image_label.setFixedSize(120, 120)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("""
            QLabel {
                border: 3px solid #0066cc;
                border-radius: 8px;
                background-color: #555555;
            }
        """)
        
        # Load player image if available
        image_loader = ImageLoader()
        if 'player_image_url' in self.player_data and pd.notna(self.player_data['player_image_url']):
            image_loader.load_image(
                self.player_data['player_image_url'], 
                self._set_player_image,
                size=(120, 120)
            )
        else:
            # Set placeholder
            self._set_player_image(image_loader._create_placeholder((120, 120)))
        
        header_layout.addWidget(self.image_label)
        
        # Player name and title
        name_layout = QVBoxLayout()
        player_name = QLabel(self.player_data.get('Player', 'Unknown'))
        player_name.setFont(QFont("Arial", 22, QFont.Weight.Bold))
        player_name.setStyleSheet("color: #0066cc;")
        name_layout.addWidget(player_name)
        
        # Position and squad subtitle
        if 'Pos' in self.player_data and 'Squad' in self.player_data:
            subtitle = QLabel(f"{self.player_data.get('Pos', '')} • {self.player_data.get('Squad', '')}")
            subtitle.setFont(QFont("Arial", 12))
            subtitle.setStyleSheet("color: #999;")
            name_layout.addWidget(subtitle)
        
        name_layout.addStretch()
        header_layout.addLayout(name_layout, stretch=1)
        
        # Header container
        header_container = QWidget()
        header_container.setLayout(header_layout)
        header_container.setStyleSheet("padding: 15px; background-color: rgba(0, 102, 204, 0.1); border-radius: 5px;")
        layout.addWidget(header_container)
        
        # Basic info grid
        info_group = QGroupBox("📋 Basic Information")
        info_group.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        info_layout = QGridLayout()
        info_layout.setSpacing(15)
        
        row = 0
        for label_text, key, icon in [
            ("Position", "Pos", "⚽"),
            ("Squad", "Squad", "🏆"),
            ("Age", "Age", "📅"),
            ("Nationality", "Nation", "🌍"),
        ]:
            if key in self.player_data and pd.notna(self.player_data[key]):
                lbl = QLabel(f"{icon} {label_text}:")
                lbl.setFont(QFont("Arial", 11, QFont.Weight.Bold))
                val = QLabel(str(self.player_data[key]))
                val.setFont(QFont("Arial", 11))
                val.setStyleSheet("padding: 5px; background-color: #f0f0f0; color: #000000; border-radius: 3px;")
                info_layout.addWidget(lbl, row, 0)
                info_layout.addWidget(val, row, 1)
                row += 1
        
        # Additional info (foot, contract, value)
        for label_text, keys, icon in [
            ("Preferred Foot", ["foot", "Foot", "preferred_foot"], "👟"),
            ("Contract Expires", ["contract_expires", "Contract", "Contract Expires"], "📝"),
            ("Market Value", ["market_value", "Market Value", "Value", "value"], "💰"),
        ]:
            value = None
            for key in keys:
                if key in self.player_data and pd.notna(self.player_data[key]):
                    value = self.player_data[key]
                    break
            
            if value is not None:
                lbl = QLabel(f"{icon} {label_text}:")
                lbl.setFont(QFont("Arial", 11, QFont.Weight.Bold))
                
                # Format value
                if "Value" in label_text:
                    try:
                        val_num = float(value)
                        if val_num >= 1_000_000:
                            value = f"€{val_num/1_000_000:.1f}M"
                        elif val_num >= 1_000:
                            value = f"€{val_num/1_000:.0f}K"
                        else:
                            value = f"€{val_num:.0f}"
                    except:
                        value = str(value)
                
                val = QLabel(str(value))
                val.setFont(QFont("Arial", 11))
                val.setStyleSheet("padding: 5px; background-color: #f0f0f0; color: #000000; border-radius: 3px;")
                
                # Highlight free agents
                if "Contract" in label_text and "30/06/2026" in str(value):
                    val.setStyleSheet("padding: 5px; background-color: #add8e6; color: #00008b; border-radius: 3px; font-weight: bold;")
                
                info_layout.addWidget(lbl, row, 0)
                info_layout.addWidget(val, row, 1)
                row += 1
        
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)
        
        # Match Score
        score_group = QGroupBox("🎯 Recommendation Score")
        score_group.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        score_layout = QVBoxLayout()
        
        score_value = self.recommendation_data.get('Score', 0)  # Use recommendation_data which has the Score!
        try:
            score_num = float(score_value)
            score_pct = int(score_num * 100)
            
            # Score bar
            score_label = QLabel(f"{score_pct}%")
            score_label.setFont(QFont("Arial", 24, QFont.Weight.Bold))
            score_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            
            if score_num > 0.7:
                score_label.setStyleSheet("color: #28a745; padding: 10px;")
                rating = "⭐⭐⭐ Excellent Match!"
            elif score_num > 0.5:
                score_label.setStyleSheet("color: #ffc107; padding: 10px;")
                rating = "⭐⭐ Good Match"
            else:
                score_label.setStyleSheet("color: #6c757d; padding: 10px;")
                rating = "⭐ Potential Match"
            
            score_layout.addWidget(score_label)
            rating_label = QLabel(rating)
            rating_label.setFont(QFont("Arial", 12))
            rating_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            rating_label.setStyleSheet("color: #ffffff;")  # Ensure white text on dark background
            score_layout.addWidget(rating_label)
        except:
            score_layout.addWidget(QLabel("Score not available"))
        
        score_group.setLayout(score_layout)
        layout.addWidget(score_group)
        
        # Addresses (what issues this player solves)
        if 'Addresses' in self.player_data and pd.notna(self.player_data['Addresses']):
            addresses_group = QGroupBox("✅ Addresses Team Needs")
            addresses_group.setFont(QFont("Arial", 12, QFont.Weight.Bold))
            addresses_layout = QVBoxLayout()
            
            addresses_text = str(self.player_data['Addresses'])
            for need in addresses_text.split(','):
                need = need.strip()
                if need:
                    need_label = QLabel(f"• {need}")
                    need_label.setFont(QFont("Arial", 10))
                    need_label.setWordWrap(True)
                    need_label.setStyleSheet("padding: 5px; margin-left: 10px; color: #ffffff;")
                    addresses_layout.addWidget(need_label)
            
            addresses_group.setLayout(addresses_layout)
            layout.addWidget(addresses_group)

        # ── Similar Alternatives (KNN) ─────────────────────────────────────
        similar_section = self._build_similar_section()
        if similar_section is not None:
            layout.addWidget(similar_section)

        # Push content to the top inside the scroll area
        layout.addStretch()

        # ── Pinned button bar (lives OUTSIDE the scroll area) ───────────────
        buttons_layout = QHBoxLayout()
        buttons_layout.setContentsMargins(20, 6, 20, 6)

        # Simulate Season button
        simulate_btn = QPushButton("🎲 Simulate Season")
        simulate_btn.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        simulate_btn.clicked.connect(self.open_simulation)
        simulate_btn.setStyleSheet("""
            QPushButton {
                background-color: #28a745;
                color: white;
                padding: 10px 30px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #218838;
            }
        """)
        buttons_layout.addWidget(simulate_btn)

        # Close button
        close_btn = QPushButton("Close")
        close_btn.setFont(QFont("Arial", 11))
        close_btn.clicked.connect(self.accept)
        close_btn.setStyleSheet("""
            QPushButton {
                background-color: #0066cc;
                color: white;
                padding: 10px 30px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #0052a3;
            }
        """)
        buttons_layout.addWidget(close_btn)

        # Add button bar to the outer (non-scrolling) layout
        outer_layout.addLayout(buttons_layout)
    
    # ------------------------------------------------------------------ #
    #  KNN "Similar Alternatives" helpers                                #
    # ------------------------------------------------------------------ #

    def _build_similar_section(self):
        """
        Build the interactive "Similar Alternatives" panel.

        Layout inside the QGroupBox:
          1. Checkbox grid  — pick which per-90 stats drive the KNN
          2. Controls row   — max budget dropdown + K spinbox + Update button
          3. Results area   — lazily populated; auto-run on first open
        """
        group = QGroupBox("🔍 Similar Alternatives")
        group.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        outer = QVBoxLayout()
        outer.setSpacing(8)
        outer.setContentsMargins(8, 8, 8, 8)

        # ── 1. Metric checkboxes ────────────────────────────────────────────
        metrics_box = QGroupBox("Stats used for comparison")
        metrics_box.setFont(QFont("Arial", 9))
        metrics_box.setStyleSheet("""
            QGroupBox {
                color: #bbb;
                border: 1px solid #444;
                border-radius: 4px;
                margin-top: 6px;
                padding-top: 4px;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 8px; }
        """)
        grid = QGridLayout()
        grid.setSpacing(2)
        grid.setContentsMargins(6, 10, 6, 6)

        self._similar_checkboxes: dict[str, QCheckBox] = {}
        cols_present = (
            set(self.candidates_df.columns)
            if self.candidates_df is not None
            else set()
        )
        available_cols = [c for c in PER90_COLS if c in cols_present]

        for i, col in enumerate(available_cols):
            cb = QCheckBox(PER90_LABELS.get(col, col))
            cb.setChecked(True)
            cb.setFont(QFont("Arial", 9))
            cb.setStyleSheet("color: #ddd;")
            self._similar_checkboxes[col] = cb
            grid.addWidget(cb, i // 3, i % 3)

        metrics_box.setLayout(grid)
        outer.addWidget(metrics_box)

        # ── 2. Budget + K + Update button row ──────────────────────────────
        controls_row = QHBoxLayout()
        controls_row.setSpacing(10)

        budget_lbl = QLabel("Max budget:")
        budget_lbl.setFont(QFont("Arial", 10))
        budget_lbl.setStyleSheet("color: #ccc;")
        controls_row.addWidget(budget_lbl)

        self._similar_budget_combo = QComboBox()
        self._similar_budget_combo.setFont(QFont("Arial", 10))
        for label, _ in _BUDGET_OPTIONS:
            self._similar_budget_combo.addItem(label)
        self._similar_budget_combo.setFixedWidth(105)
        controls_row.addWidget(self._similar_budget_combo)

        controls_row.addSpacing(16)

        k_lbl = QLabel("Top:")
        k_lbl.setFont(QFont("Arial", 10))
        k_lbl.setStyleSheet("color: #ccc;")
        controls_row.addWidget(k_lbl)

        self._similar_k_spin = QSpinBox()
        self._similar_k_spin.setRange(1, 20)
        self._similar_k_spin.setValue(5)
        self._similar_k_spin.setSuffix(" players")
        self._similar_k_spin.setFont(QFont("Arial", 10))
        self._similar_k_spin.setFixedWidth(90)
        controls_row.addWidget(self._similar_k_spin)

        controls_row.addStretch()

        update_btn = QPushButton("🔄  Update")
        update_btn.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        update_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        update_btn.setStyleSheet("""
            QPushButton {
                background-color: #0066cc;
                color: white;
                padding: 5px 18px;
                border-radius: 4px;
            }
            QPushButton:hover { background-color: #0052a3; }
        """)
        update_btn.clicked.connect(self._refresh_similar)
        controls_row.addWidget(update_btn)

        outer.addLayout(controls_row)

        # ── 3. Results container ────────────────────────────────────────────
        self._similar_results_widget = QWidget()
        self._similar_results_layout = QVBoxLayout(self._similar_results_widget)
        self._similar_results_layout.setSpacing(4)
        self._similar_results_layout.setContentsMargins(0, 4, 0, 0)
        outer.addWidget(self._similar_results_widget)

        group.setLayout(outer)

        # Auto-run so results are visible immediately on open
        if self.candidates_df is not None and len(self.candidates_df) >= 2:
            self._refresh_similar()
        else:
            self._set_similar_status(
                "No candidate pool available — run analysis first."
            )

        return group

    def _refresh_similar(self):
        """
        Read the current control values, run the KNN query, and repopulate
        the results container.  Called automatically on first open and each
        time the user clicks "Update".
        """
        # ── selected per-90 columns ─────────────────────────────────────────
        selected_cols = [
            col for col, cb in self._similar_checkboxes.items()
            if cb.isChecked()
        ]
        if not selected_cols:
            self._set_similar_status("Select at least one stat to compare.")
            return

        k = self._similar_k_spin.value()

        # ── budget filter ───────────────────────────────────────────────────
        budget_idx = self._similar_budget_combo.currentIndex()
        max_budget = _BUDGET_OPTIONS[budget_idx][1]   # None = no limit

        pool = self.candidates_df
        if max_budget is not None and pool is not None:
            val_col = next((c for c in _VALUE_COLS if c in pool.columns), None)
            if val_col:
                mv = pd.to_numeric(pool[val_col], errors='coerce')
                # Keep players whose value is within budget OR unknown (NaN)
                pool = pool[mv.isna() | (mv <= max_budget)]

        if pool is None or len(pool) < 2:
            self._set_similar_status(
                "Too few players remain after applying the budget filter."
            )
            return

        # ── KNN query ───────────────────────────────────────────────────────
        player_name = self.player_data.get('Player', '')
        try:
            similar = get_similar_players(
                player_name=player_name,
                candidates_df=pool,
                per90_cols=selected_cols,
                k=k,
            )
        except Exception as exc:
            self._set_similar_status(f"KNN error: {exc}")
            return

        # ── repopulate results ──────────────────────────────────────────────
        while self._similar_results_layout.count():
            item = self._similar_results_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if not similar:
            self._set_similar_status(
                "No similar players found — try selecting more stats or raising the budget."
            )
            return

        for entry in similar:
            chip = self._make_similar_chip(
                entry['row'], entry['distance'], selected_cols
            )
            self._similar_results_layout.addWidget(chip)

    def _set_similar_status(self, message: str):
        """Clear the results area and show a single status/info label."""
        while self._similar_results_layout.count():
            item = self._similar_results_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        lbl = QLabel(message)
        lbl.setFont(QFont("Arial", 9))
        lbl.setStyleSheet("color: #888; font-style: italic; padding: 4px;")
        lbl.setWordWrap(True)
        self._similar_results_layout.addWidget(lbl)

    def _make_similar_chip(
        self, player_row, distance: float, selected_cols: list = None
    ) -> QFrame:
        """
        Build a compact, clickable row widget for one similar player.

        Shows (left → right):
          • Clickable player name
          • Squad
          • Up to 2 of the currently selected per-90 stats (from selected_cols)
          • Similarity distance badge
        """
        frame = QFrame()
        frame.setFrameShape(QFrame.Shape.StyledPanel)
        frame.setStyleSheet("""
            QFrame {
                background-color: #333;
                border: 1px solid #555;
                border-radius: 5px;
                padding: 4px 8px;
            }
            QFrame:hover {
                border-color: #4da6ff;
                background-color: #3a3a3a;
            }
        """)

        row_layout = QHBoxLayout(frame)
        row_layout.setContentsMargins(4, 4, 4, 4)
        row_layout.setSpacing(10)

        # Clickable name button
        name = str(player_row.get('Player', 'Unknown'))
        name_btn = QPushButton(name)
        name_btn.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        name_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        name_btn.setStyleSheet("""
            QPushButton {
                background: transparent;
                border: none;
                color: #4da6ff;
                text-align: left;
                text-decoration: underline;
            }
            QPushButton:hover { color: #80c4ff; }
        """)
        name_btn.clicked.connect(
            lambda _checked, r=player_row: self._open_similar_player(r)
        )
        row_layout.addWidget(name_btn, stretch=3)

        # Squad
        if 'Squad' in player_row and pd.notna(player_row.get('Squad')):
            team_lbl = QLabel(str(player_row['Squad']))
            team_lbl.setFont(QFont("Arial", 9))
            team_lbl.setStyleSheet("color: #bbb;")
            row_layout.addWidget(team_lbl, stretch=2)

        # Up to 2 of the currently selected per-90 stats
        display_cols = selected_cols if selected_cols else list(PER90_LABELS.keys())
        stat_parts = []
        for col in display_cols:
            if col not in player_row.index:
                continue
            try:
                val = float(player_row[col])
            except (TypeError, ValueError):
                continue
            stat_parts.append(f"{PER90_LABELS.get(col, col)}: {val:.2f}")
            if len(stat_parts) == 2:
                break

        if stat_parts:
            stats_lbl = QLabel("  |  ".join(stat_parts))
            stats_lbl.setFont(QFont("Arial", 9))
            stats_lbl.setStyleSheet("color: #999;")
            row_layout.addWidget(stats_lbl, stretch=3)

        # Similarity distance badge
        dist_lbl = QLabel(f"dist {distance:.2f}")
        dist_lbl.setFont(QFont("Arial", 8))
        dist_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        dist_lbl.setStyleSheet(
            "color: #28a745; background: #1e3a1e; border-radius: 3px; padding: 2px 5px;"
        )
        row_layout.addWidget(dist_lbl)

        return frame

    def _open_similar_player(self, player_row):
        """
        Open a new PlayerProfileDialog for the clicked similar player,
        passing the same candidate pool so the new card can also show
        its own similar alternatives.
        """
        dialog = PlayerProfileDialog(
            player_row,
            parent=self.parent_window,
            candidates_df=self.candidates_df,
        )
        dialog.exec()

    def _set_player_image(self, pixmap):
        """Set the player image when loaded"""
        if pixmap and not pixmap.isNull():
            # Simple rectangular scaling - no circular mask
            target_size = 120
            
            # Scale to fit within bounds, maintaining aspect ratio
            scaled = pixmap.scaled(target_size, target_size, 
                                  Qt.AspectRatioMode.KeepAspectRatio,
                                  Qt.TransformationMode.SmoothTransformation)
            
            self.image_label.setPixmap(scaled)
        else:
            self.image_label.setText("👤")
            self.image_label.setStyleSheet("""
                QLabel {
                    border: 3px solid #0066cc;
                    border-radius: 8px;
                    background-color: #555555;
                    font-size: 48px;
                }
            """)
    
    def open_simulation(self):
        """Open Monte Carlo simulation dialog"""
        sim_dialog = MonteCarloDialog(self.player_data, parent=self)
        sim_dialog.exec()


# ===================== Monte Carlo Simulation Components =====================

class SimulationWorker(QThread):
    """Worker thread for running simulations without freezing UI"""
    progress = pyqtSignal(int)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, player_data, n_simulations, importance="Starter — Full 90"):
        super().__init__()
        self.player_data = player_data
        self.n_simulations = n_simulations
        self.importance = importance

    def run(self):
        """Run simulation in background thread"""
        try:
            simulator = MonteCarloSimulator(
                self.player_data,
                n_simulations=self.n_simulations,
                importance=self.importance,
            )
            self.progress.emit(0)
            results = simulator.run_simulations()
            self.progress.emit(100)
            self.finished.emit(results)
        except Exception as e:
            self.error.emit(str(e))


class MonteCarloDialog(QDialog):
    """Dialog for Monte Carlo season simulation"""
    
    def __init__(self, player_data, parent=None):
        super().__init__(parent)
        self.parent_window = parent  # ADD THIS
        self.recommendation_data = player_data  # ADD THIS - stores Score
        
        # Get FULL player stats (needed for simulation)
        self.player_data = self._get_full_player_data(player_data)  # ADD THIS
        
        self.results = None
        
        self.setWindowTitle(f"🎲 Season Simulation: {self.player_data.get('Player', 'Unknown')}")
        self.setMinimumSize(1200, 800)
        
        # Main layout
        main_layout = QHBoxLayout(self)
        
        # Left panel (controls)
        left_panel = self.create_left_panel()
        main_layout.addWidget(left_panel, stretch=1)
        
        # Right panel (results)
        self.right_panel = self.create_right_panel()
        main_layout.addWidget(self.right_panel, stretch=3)
    
    def _get_full_player_data(self, recommendation_row):
        """
        Get full player stats from evaluated_players by walking up the parent
        chain.

        The dialog may be opened from PlayerProfileDialog (which has no
        evaluated_players), which itself was opened from FootballAnalyzerApp
        (which does).  A simple one-level parent check therefore misses the
        data; we traverse parent_window links until we find the DataFrame or
        run out of parents.
        """
        player_name = recommendation_row.get('Player', '')
        player_name_clean = player_name.strip().lower()

        # Walk up the window hierarchy to find evaluated_players.
        window = self.parent_window
        while window is not None:
            ep = getattr(window, 'evaluated_players', None)
            if ep is not None:
                if 'Player' in ep.columns:
                    player_match = ep[
                        ep['Player'].str.strip().str.lower() == player_name_clean
                    ]
                    if len(player_match) > 0:
                        return player_match.iloc[0]
                # Found the window with evaluated_players but player not in it —
                # no point searching further up.
                break
            window = getattr(window, 'parent_window', None)

        # Fallback: return original row (simulation will work with reduced stats)
        print(
            f"⚠️  WARNING [MonteCarlo]: Could not find full stats for "
            f"'{player_name}' in evaluated_players (searched "
            f"{type(self.parent_window).__name__} chain)"
        )
        return recommendation_row

    def _update_importance_desc(self, label: str):
        """Update the description label when the importance combo changes."""
        profile = IMPORTANCE_PROFILES.get(label, {})
        desc = profile.get("description", "")
        avg_min = profile.get("avg_minutes", 90)
        sel_pct = int(profile.get("selection_prob", 1.0) * 100)
        mult    = profile.get("perf_multiplier", 1.0)
        boost_note = f"  |  +{int((mult-1)*100)}% per-90 boost" if mult > 1.0 else (
                     f"  |  {int((mult-1)*100)}% per-90 penalty" if mult < 1.0 else ""
        )
        self.importance_desc.setText(
            f"{desc}\n~{avg_min} min/game  |  selected {sel_pct}% of matches{boost_note}"
        )

    def create_left_panel(self):
        """Create control panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(15)
        
        # Title
        title = QLabel("⚙️ Simulation Settings")
        title.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        title.setStyleSheet("color: #0066cc; padding: 10px;")
        layout.addWidget(title)
        
        # Player info
        info_group = QGroupBox("Player Information")
        info_layout = QVBoxLayout()
        
        player_name = QLabel(f"📊 {self.player_data.get('Player', 'Unknown')}")
        player_name.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        info_layout.addWidget(player_name)
        
        position = QLabel(f"Position: {self.player_data.get('Pos', 'N/A')}")
        info_layout.addWidget(position)
        
        squad = QLabel(f"Squad: {self.player_data.get('Squad', 'N/A')}")
        info_layout.addWidget(squad)
        
        age = QLabel(f"Age: {self.player_data.get('Age', 'N/A')}")
        info_layout.addWidget(age)
        
        rec_score = float(self.recommendation_data.get('Score', 0)) * 100  # Use recommendation_data!
        score_label = QLabel(f"Recommendation: {rec_score:.0f}%")
        score_label.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        
        # Color code by score
        if rec_score >= 70:
            score_label.setStyleSheet("color: #28a745; padding: 5px;")
            multiplier_text = "Team Fit: 1.50x (Excellent!)"
        elif rec_score >= 60:
            score_label.setStyleSheet("color: #0066cc; padding: 5px;")
            multiplier_text = "Team Fit: 1.30x (Good)"
        elif rec_score >= 50:
            score_label.setStyleSheet("color: #ffc107; padding: 5px;")
            multiplier_text = "Team Fit: 1.15x (Decent)"
        else:
            score_label.setStyleSheet("color: #6c757d; padding: 5px;")
            multiplier_text = "Team Fit: 1.00x (Neutral)"
        
        info_layout.addWidget(score_label)
        
        multiplier_label = QLabel(multiplier_text)
        multiplier_label.setStyleSheet("font-style: italic; color: #999;")
        info_layout.addWidget(multiplier_label)
        
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)
        
        # ── Importance / Role ──────────────────────────────────────────
        importance_group = QGroupBox("Player Role / Importance")
        importance_layout = QVBoxLayout()

        self.importance_combo = QComboBox()
        for label in IMPORTANCE_PROFILES:
            self.importance_combo.addItem(label)
        self.importance_combo.setStyleSheet("padding: 4px; font-size: 11px;")
        importance_layout.addWidget(self.importance_combo)

        self.importance_desc = QLabel()
        self.importance_desc.setWordWrap(True)
        self.importance_desc.setStyleSheet("color: #aaaaaa; font-size: 10px; font-style: italic; padding: 2px 4px;")
        importance_layout.addWidget(self.importance_desc)

        importance_group.setLayout(importance_layout)
        layout.addWidget(importance_group)

        # Populate description for default selection and wire the signal
        self._update_importance_desc(self.importance_combo.currentText())
        self.importance_combo.currentTextChanged.connect(self._update_importance_desc)

        # ── Simulation count ───────────────────────────────────────────
        settings_group = QGroupBox("Settings")
        settings_layout = QVBoxLayout()

        sims_label = QLabel("Number of Simulations:")
        settings_layout.addWidget(sims_label)

        self.sims_spin = QSpinBox()
        self.sims_spin.setRange(100, 10000)
        self.sims_spin.setValue(1000)
        self.sims_spin.setSingleStep(100)
        self.sims_spin.setSuffix(" simulations")
        self.sims_spin.setStyleSheet("padding: 5px; font-size: 11px;")
        settings_layout.addWidget(self.sims_spin)

        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)

        # Run button
        self.run_btn = QPushButton("🎲 Run Simulation")
        self.run_btn.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        self.run_btn.clicked.connect(self.run_simulation)
        self.run_btn.setStyleSheet("""
            QPushButton {
                background-color: #28a745;
                color: white;
                padding: 15px;
                border-radius: 8px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #218838;
            }
            QPushButton:disabled {
                background-color: #6c757d;
            }
        """)
        layout.addWidget(self.run_btn)
        
        # Progress bar
        self.progress_bar = QProgressDialog("Running simulations...", None, 0, 100, self)
        self.progress_bar.setWindowTitle("Simulating Season")
        self.progress_bar.setWindowModality(Qt.WindowModality.WindowModal)
        self.progress_bar.setAutoClose(True)
        self.progress_bar.setAutoReset(True)
        self.progress_bar.cancel()  # Hide initially
        
        layout.addStretch()
        
        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        close_btn.setStyleSheet("""
            QPushButton {
                background-color: #dc3545;
                color: white;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #c82333;
            }
        """)
        layout.addWidget(close_btn)
        
        return panel
    
    def create_right_panel(self):
        """Create results panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Title
        title = QLabel("📊 Simulation Results")
        title.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        title.setStyleSheet("color: #0066cc; padding: 10px;")
        layout.addWidget(title)
        
        # Info label (shown before simulation)
        self.info_label = QLabel("Click 'Run Simulation' to project the player's season performance...")
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.info_label.setStyleSheet("color: #6c757d; font-size: 12px; padding: 50px;")
        layout.addWidget(self.info_label)
        
        # Results container (hidden initially)
        self.results_container = QWidget()
        self.results_layout = QVBoxLayout(self.results_container)
        self.results_container.hide()
        
        layout.addWidget(self.results_container)
        
        return panel
    
    def run_simulation(self):
        """Run the Monte Carlo simulation"""
        n_sims = self.sims_spin.value()
        
        # Disable run button
        self.run_btn.setEnabled(False)
        self.run_btn.setText("Running...")
        
        # Show progress
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.show()
        
        importance = self.importance_combo.currentText()

        # Create worker thread
        self.worker = SimulationWorker(self.player_data, n_sims, importance=importance)
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.display_results)
        self.worker.error.connect(self.show_error)
        self.worker.start()
    
    def update_progress(self, value):
        """Update progress bar"""
        self.progress_bar.setValue(value)
    
    def show_error(self, error_msg):
        """Show error message"""
        self.progress_bar.cancel()
        self.run_btn.setEnabled(True)
        self.run_btn.setText("🎲 Run Simulation")
        
        QMessageBox.critical(self, "Simulation Error", f"Error running simulation:\n{error_msg}")
    
    def display_results(self, results):
        """Display simulation results"""
        self.progress_bar.cancel()
        self.run_btn.setEnabled(True)
        self.run_btn.setText("🎲 Run Simulation")
        
        self.results = results
        
        # Hide info label
        self.info_label.hide()
        
        # Clear previous results
        while self.results_layout.count():
            item = self.results_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        # Show results container
        self.results_container.show()
        
        # Add results widgets
        self.add_summary_stats()
        self.add_scenarios()
        
        # Add chart if goals data available
        if results.get('goals') and results['goals'].get('distribution'):
            self.add_goals_chart()
    
    def add_summary_stats(self):
        """Add summary statistics"""
        stats_group = QGroupBox("📈 Expected Season Performance")
        stats_group.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        stats_layout = QGridLayout()
        
        row = 0
        
        # Role / importance label
        importance_label = self.results.get('importance', 'Starter — Full 90')
        avg_min = int(self.results.get('avg_minutes', 90))
        self.add_stat_row(stats_layout, row, "Role",
                         f"{importance_label}  (~{avg_min} min/game)",
                         "🎭")
        row += 1

        # Availability breakdown
        games = self.results['games_played']
        self.add_stat_row(stats_layout, row, "Appearances",
                         f"{games['median']} ({games['p5']}-{games['p95']})",
                         "📅")
        row += 1

        not_sel = self.results.get('games_not_selected', {})
        inj     = self.results.get('games_injured',      {})
        sus     = self.results.get('games_suspended',     {})
        if not_sel and not_sel.get('median', 0) > 0:
            self.add_stat_row(stats_layout, row, "Not Selected",
                             f"{not_sel['median']} games (rotation/bench)",
                             "🔄")
            row += 1
        inj_med = inj.get('median', 0) if inj else 0
        sus_med = sus.get('median', 0) if sus else 0
        if inj_med > 0 or sus_med > 0:
            self.add_stat_row(stats_layout, row, "Unavailable",
                             f"{inj_med} inj. / {sus_med} susp.",
                             "🏥")
            row += 1

        # Position-specific stats
        if self.results.get('goals'):
            goals = self.results['goals']
            self.add_stat_row(stats_layout, row, "Goals", 
                             f"{goals['median']} ({goals['p5']}-{goals['p95']})", 
                             "⚽")
            row += 1
        
        if self.results.get('assists'):
            assists = self.results['assists']
            self.add_stat_row(stats_layout, row, "Assists", 
                             f"{assists['median']} ({assists['p5']}-{assists['p95']})", 
                             "🎯")
            row += 1
        
        if self.results.get('tackles'):
            tackles = self.results['tackles']
            self.add_stat_row(stats_layout, row, "Tackles", 
                             f"{tackles['median']}", 
                             "🛡️")
            row += 1
        
        if self.results.get('interceptions'):
            ints = self.results['interceptions']
            self.add_stat_row(stats_layout, row, "Interceptions", 
                             f"{ints['median']}", 
                             "✋")
            row += 1
        
        if self.results.get('saves'):
            saves = self.results['saves']
            self.add_stat_row(stats_layout, row, "Saves", 
                             f"{saves['median']}", 
                             "🧤")
            row += 1
            
            cs = self.results['clean_sheets']
            self.add_stat_row(stats_layout, row, "Clean Sheets", 
                             f"{cs['median']}", 
                             "🚫")
            row += 1
        
        # Progressive actions
        if self.results.get('prog_carries'):
            pc = self.results['prog_carries']
            self.add_stat_row(stats_layout, row, "Prog. Carries",
                             f"{pc['median']} ({pc['p5']}-{pc['p95']})",
                             "\U0001f3c3")
            row += 1

        if self.results.get('prog_passes'):
            pp = self.results['prog_passes']
            self.add_stat_row(stats_layout, row, "Prog. Passes",
                             f"{pp['median']} ({pp['p5']}-{pp['p95']})",
                             "\U0001f4e8")
            row += 1

        # Discipline
        yellows = self.results['discipline']['yellow_cards']['median']
        reds = self.results['discipline']['red_cards']['median']
        self.add_stat_row(stats_layout, row, "Cards", 
                         f"{yellows} yellow, {reds} red", 
                         "🟨🟥")
        
        stats_group.setLayout(stats_layout)
        self.results_layout.addWidget(stats_group)
    
    def add_stat_row(self, layout, row, label_text, value_text, icon):
        """Add a stat row to the grid"""
        icon_label = QLabel(icon)
        icon_label.setFont(QFont("Arial", 16))
        layout.addWidget(icon_label, row, 0)
        
        label = QLabel(label_text + ":")
        label.setFont(QFont("Arial", 11))
        layout.addWidget(label, row, 1)
        
        value = QLabel(value_text)
        value.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        value.setStyleSheet("color: #0066cc;")
        layout.addWidget(value, row, 2)
    
    def add_scenarios(self):
        """Add best/expected/worst scenarios (position-aware)"""
        scenarios_group = QGroupBox("🎯 Scenarios")
        scenarios_group.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        scenarios_layout = QVBoxLayout()
        
        # Determine PRIMARY and SECONDARY metrics based on position
        primary_metric = None
        primary_name = None
        secondary_metric = None
        secondary_name = None
        
        # Attackers: Goals primary, Assists secondary
        if self.results.get('goals'):
            primary_metric = self.results['goals']
            primary_name = "goals"
            if self.results.get('assists'):
                secondary_metric = self.results['assists']
                secondary_name = "assists"
        
        # Goalkeepers: Saves primary, Clean Sheets secondary
        elif self.results.get('saves'):
            primary_metric = self.results['saves']
            primary_name = "saves"
            if self.results.get('clean_sheets'):
                secondary_metric = self.results['clean_sheets']
                secondary_name = "clean sheets"
        
        # Defenders: Tackles primary, Interceptions secondary
        elif self.results.get('tackles'):
            primary_metric = self.results['tackles']
            primary_name = "tackles"
            if self.results.get('interceptions'):
                secondary_metric = self.results['interceptions']
                secondary_name = "interceptions"
        
        # Fallback: just show games and rating
        else:
            primary_metric = self.results['games_played']
            primary_name = "games"
            secondary_metric = self.results['rating']
            secondary_name = "avg rating"
        
        games = self.results['games_played']
        
        # Build scenario text
        def format_secondary(metric):
            """Format secondary metric value"""
            if secondary_name == "avg rating":
                return f"{metric:.2f}"
            else:
                return str(metric)
        
        # Best case (95th percentile)
        if secondary_metric and secondary_name != "avg rating":
            best_text = f"✨ Best Case (95th): {primary_metric['p95']} {primary_name}, {format_secondary(secondary_metric['p95'])} {secondary_name}"
        else:
            best_text = f"✨ Best Case (95th): {primary_metric['p95']} {primary_name} in {games['p95']} games"
        
        best = QLabel(best_text)
        best.setStyleSheet("color: #28a745; font-size: 11px; padding: 5px;")
        scenarios_layout.addWidget(best)
        
        # Expected (median)
        if secondary_metric and secondary_name != "avg rating":
            exp_text = f"📊 Expected (50th): {primary_metric['median']} {primary_name}, {format_secondary(secondary_metric['median'])} {secondary_name}"
        else:
            exp_text = f"📊 Expected (50th): {primary_metric['median']} {primary_name} in {games['median']} games"
        
        expected = QLabel(exp_text)
        expected.setStyleSheet("color: #0066cc; font-size: 11px; padding: 5px; font-weight: bold;")
        scenarios_layout.addWidget(expected)
        
        # Worst case (5th percentile)
        if secondary_metric and secondary_name != "avg rating":
            worst_text = f"⚠️ Worst Case (5th): {primary_metric['p5']} {primary_name}, {format_secondary(secondary_metric['p5'])} {secondary_name}"
        else:
            worst_text = f"⚠️ Worst Case (5th): {primary_metric['p5']} {primary_name} in {games['p5']} games"
        
        worst = QLabel(worst_text)
        worst.setStyleSheet("color: #dc3545; font-size: 11px; padding: 5px;")
        scenarios_layout.addWidget(worst)
        
        scenarios_group.setLayout(scenarios_layout)
        self.results_layout.addWidget(scenarios_group)
    
    def add_goals_chart(self):
        """Add goals distribution histogram"""
        chart_group = QGroupBox("📊 Goals Distribution")
        chart_group.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        chart_layout = QVBoxLayout()
        
        # Create matplotlib figure
        fig = Figure(figsize=(8, 4), facecolor='white')
        canvas = FigureCanvasQTAgg(fig)
        ax = fig.add_subplot(111)
        
        # Plot histogram
        goals_data = self.results['goals']['distribution']
        ax.hist(goals_data, bins=30, color='#0066cc', alpha=0.7, edgecolor='black')
        
        # Add median line
        median = self.results['goals']['median']
        ax.axvline(median, color='red', linestyle='--', linewidth=2, label=f'Median: {median}')
        
        # Styling
        ax.set_xlabel('Goals', fontsize=11, weight='bold')
        ax.set_ylabel('Frequency', fontsize=11, weight='bold')
        ax.set_title(f'Distribution of Goals Across {len(goals_data)} Simulations', 
                    fontsize=12, weight='bold', pad=10)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        fig.tight_layout()
        
        chart_layout.addWidget(canvas)
        chart_group.setLayout(chart_layout)
        self.results_layout.addWidget(chart_group)


# ===================== Main Application Window =====================
class FootballAnalyzerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("⚽ Football Performance Analyzer & Recommender")
        self.setGeometry(100, 100, 1400, 900)
        
        # Data storage
        self.matches_df = None
        self.matches_trimmed_df = None  # Dataset trimmed to last 4 seasons
        self.players_df = None
        self.df_team = None
        self.patterns = None
        self.evaluated_players = None
        self.current_team = None
        self.all_issues = []
        self.selected_season = "All seasons"
        
        # Setup UI
        self.init_ui()
        
    def init_ui(self):
        """Initialize the user interface"""
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)
        
        # Left panel (controls)
        left_panel = self.create_left_panel()
        left_panel.setMaximumWidth(450)  # Prevent left panel from being too wide
        splitter.addWidget(left_panel)
        
        # Right panel (results)
        right_panel = self.create_right_panel()
        splitter.addWidget(right_panel)
        
        # Set splitter sizes - left panel smaller for better space usage
        splitter.setSizes([350, 1050])
        splitter.setStretchFactor(0, 0)  # Left panel doesn't stretch
        splitter.setStretchFactor(1, 1)  # Right panel takes remaining space
        
        # Status bar
        self.statusBar().showMessage("Ready. Please load data files to begin.")
    
    def create_left_panel(self):
        """Create left control panel with scroll area for smaller screens"""
        # Create scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: transparent;
            }
            QScrollBar:vertical {
                border: none;
                background: #2b2b2b;
                width: 10px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background: #555555;
                border-radius: 5px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background: #666666;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)
        
        # Create the actual content panel
        panel = QWidget()
        panel.setMinimumWidth(320)  # Minimum usable width
        panel.setMaximumWidth(430)  # Maximum to prevent overflow
        layout = QVBoxLayout(panel)
        layout.setSpacing(12)  # Tighter spacing
        layout.setContentsMargins(8, 8, 8, 8)  # Tighter margins
        
        # === 1. DATA LOADING ===
        data_group = QGroupBox("1. Load Data")
        data_layout = QVBoxLayout()
        
        # Matches file
        self.matches_file_label = QLabel("No matches file loaded")
        self.matches_file_label.setWordWrap(True)
        btn_load_matches = QPushButton("📁 Load Matches File")
        btn_load_matches.clicked.connect(self.load_matches_file)
        
        # Players file
        self.players_file_label = QLabel("No players file loaded")
        self.players_file_label.setWordWrap(True)
        btn_load_players = QPushButton("📁 Load Players File")
        btn_load_players.clicked.connect(self.load_players_file)
        
        data_layout.addWidget(QLabel("Matches Data:"))
        data_layout.addWidget(self.matches_file_label)
        data_layout.addWidget(btn_load_matches)
        data_layout.addSpacing(10)
        data_layout.addWidget(QLabel("Players Data:"))
        data_layout.addWidget(self.players_file_label)
        data_layout.addWidget(btn_load_players)
        
        data_group.setLayout(data_layout)
        layout.addWidget(data_group)
        
        # === 2. TEAM SELECTION ===
        team_group = QGroupBox("2. Select Team")
        team_layout = QVBoxLayout()
        
        self.team_combo = QComboBox()
        self.team_combo.setEnabled(False)
        self.team_combo.setEditable(True)  # Allow typing to search
        self.team_combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)  # Don't add typed text as new item
        self.team_combo.completer().setCompletionMode(
            self.team_combo.completer().CompletionMode.PopupCompletion
        )  # Show popup with matches
        self.team_combo.completer().setFilterMode(Qt.MatchFlag.MatchContains)  # Match anywhere in string
        self.team_combo.setToolTip("Start typing to search for a team")
        self.team_combo.setPlaceholderText("Type to search...")
        
        self.top_n_spin = QSpinBox()
        self.top_n_spin.setRange(5, 50)
        self.top_n_spin.setValue(20)
        self.top_n_spin.setPrefix("Top ")
        self.top_n_spin.setSuffix(" players")
        
        self.btn_analyze = QPushButton("🔍 Run Full Analysis")
        self.btn_analyze.setEnabled(False)
        self.btn_analyze.clicked.connect(self.run_analysis)
        self.btn_analyze.setStyleSheet("QPushButton { background-color: #0066cc; color: white; font-weight: bold; padding: 8px; }")
        
        team_layout.addWidget(QLabel("Select Team:"))
        team_layout.addWidget(self.team_combo)
        team_layout.addSpacing(10)
        team_layout.addWidget(QLabel("Number of Recommendations:"))
        team_layout.addWidget(self.top_n_spin)
        team_layout.addSpacing(10)
        team_layout.addWidget(self.btn_analyze)
        
        team_group.setLayout(team_layout)
        layout.addWidget(team_group)
        
        # === 3. FILTERS ===
        filter_group = QGroupBox("3. Filters (Optional)")
        filter_layout = QVBoxLayout()
        filter_layout.setSpacing(10)
        
        # Age filter
        self.age_input = QLineEdit()
        self.age_input.setPlaceholderText("e.g., 18-25, u23, 30+")
        filter_layout.addWidget(QLabel("Age Filter:"))
        filter_layout.addWidget(self.age_input)
        
        # Foot preference
        self.foot_combo = QComboBox()
        self.foot_combo.addItems(["No preference", "Right-footed", "Left-footed"])
        filter_layout.addWidget(QLabel("Preferred Foot:"))
        filter_layout.addWidget(self.foot_combo)
        
        # Free agents filter
        self.free_agents_checkbox = QCheckBox("Future Free Agents Only (Contract expires 30/06/2026)")
        self.free_agents_checkbox.setFont(QFont("Arial", 9))
        filter_layout.addWidget(self.free_agents_checkbox)
        
        # Budget filter
        budget_label = QLabel("Max Budget (â‚¬):")
        filter_layout.addWidget(budget_label)
        
        budget_h_layout = QHBoxLayout()
        
        self.budget_slider = QSlider(Qt.Orientation.Horizontal)
        self.budget_slider.setMinimum(0)
        self.budget_slider.setMaximum(200)  # 200M euros
        self.budget_slider.setValue(200)  # Default: no limit
        self.budget_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.budget_slider.setTickInterval(20)
        self.budget_slider.valueChanged.connect(self.update_budget_label)
        
        self.budget_value_label = QLabel("No limit")
        self.budget_value_label.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        self.budget_value_label.setStyleSheet("color: #0066cc;")
        self.budget_value_label.setMinimumWidth(100)
        
        budget_h_layout.addWidget(self.budget_slider)
        budget_h_layout.addWidget(self.budget_value_label)
        
        filter_layout.addLayout(budget_h_layout)
        
        # Budget presets
        presets_layout = QHBoxLayout()
        presets_layout.addWidget(QLabel("Quick presets:"))
        
        btn_10m = QPushButton("â‚¬10M")
        btn_10m.clicked.connect(lambda: self.budget_slider.setValue(10))
        btn_10m.setMaximumWidth(60)
        presets_layout.addWidget(btn_10m)
        
        btn_25m = QPushButton("â‚¬25M")
        btn_25m.clicked.connect(lambda: self.budget_slider.setValue(25))
        btn_25m.setMaximumWidth(60)
        presets_layout.addWidget(btn_25m)
        
        btn_50m = QPushButton("â‚¬50M")
        btn_50m.clicked.connect(lambda: self.budget_slider.setValue(50))
        btn_50m.setMaximumWidth(60)
        presets_layout.addWidget(btn_50m)
        
        btn_100m = QPushButton("â‚¬100M")
        btn_100m.clicked.connect(lambda: self.budget_slider.setValue(100))
        btn_100m.setMaximumWidth(60)
        presets_layout.addWidget(btn_100m)
        
        btn_unlimited = QPushButton("Unlimited")
        btn_unlimited.clicked.connect(lambda: self.budget_slider.setValue(200))
        btn_unlimited.setMaximumWidth(80)
        presets_layout.addWidget(btn_unlimited)
        
        presets_layout.addStretch()
        filter_layout.addLayout(presets_layout)

        # Season filter
        filter_layout.addSpacing(10)
        filter_layout.addWidget(QLabel("Season:"))
        self.season_combo = QComboBox()
        self.season_combo.addItem("All seasons")
        self.season_combo.setEnabled(False)
        self.season_combo.setToolTip("Filter matches by season (last 4 seasons available)")
        self.season_combo.currentTextChanged.connect(self.on_season_changed)
        filter_layout.addWidget(self.season_combo)

        filter_group.setLayout(filter_layout)
        layout.addWidget(filter_group)
        
        
        # === 4. SCORING METHOD ===
        scoring_group = QGroupBox("4. Scoring Method")
        scoring_layout = QVBoxLayout()
        scoring_layout.setSpacing(10)
        
        # Info label
        scoring_info = QLabel("Choose how player scores are calculated:")
        scoring_info.setFont(QFont("Arial", 9))
        scoring_info.setWordWrap(True)
        scoring_layout.addWidget(scoring_info)
        
        # Radio buttons
        self.scoring_method_group = QButtonGroup()
        
        self.rb_minmax = QRadioButton("Min-Max Normalization (Absolute Performance)")
        self.rb_minmax.setFont(QFont("Arial", 9))
        self.rb_minmax.setToolTip(
            "Scores based on absolute stats normalized to dataset range.\n"
            "A 0.7 score means 70% between worst and best in dataset.\n"
            "Good for: Technical analysis, preserving performance gaps."
        )
        
        self.rb_percentile = QRadioButton("Percentile Ranking (Relative Standing) ⭐ RECOMMENDED")
        self.rb_percentile.setFont(QFont("Arial", 9, QFont.Weight.Bold))
        self.rb_percentile.setToolTip(
            "Scores based on percentile rank within position group.\n"
            "A 0.7 score means better than 70% of similar players.\n"
            "Good for: Scouting reports, fairer comparisons, outlier-robust."
        )
        
        self.scoring_method_group.addButton(self.rb_minmax, 0)
        self.scoring_method_group.addButton(self.rb_percentile, 1)
        
        self.rb_percentile.setChecked(True)  # Default to percentile (recommended)
        
        scoring_layout.addWidget(self.rb_percentile)
        scoring_layout.addWidget(self.rb_minmax)
        
        # Explanation text
        explanation = QLabel(
            "💡 Percentile ranking provides more intuitive scores and handles\n"
            "outliers better (e.g., elite players don't make everyone else look bad)."
        )
        explanation.setFont(QFont("Arial", 8))
        explanation.setStyleSheet("color: #666; padding: 5px; background-color: #f0f0f0; border-radius: 3px;")
        explanation.setWordWrap(True)
        scoring_layout.addWidget(explanation)
        
        scoring_group.setLayout(scoring_layout)
        layout.addWidget(scoring_group)
        
        # Reset button
        btn_reset = QPushButton("🔄 Reset All")
        btn_reset.clicked.connect(self.reset_app)
        layout.addWidget(btn_reset)
        
        layout.addStretch()
        
        # Set the panel as the scroll area's widget
        scroll.setWidget(panel)
        
        return scroll  # Return scroll area instead of panel
    
    def create_right_panel(self):
        """Create right results panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Title
        title = QLabel("Welcome to Football Analyzer")
        title.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        # Tab widget for different views
        self.tabs = QTabWidget()
        
        # Tab 1: Team Statistics
        self.stats_tab = self.create_stats_tab()
        self.tabs.addTab(self.stats_tab, "📊 Team Statistics")
        
        # Tab 2: Detected Issues
        self.issues_tab = self.create_issues_tab()
        self.tabs.addTab(self.issues_tab, "⚠️ Detected Issues")
        
        # Tab 3: Recommendations
        self.recs_tab = self.create_recommendations_tab()
        self.tabs.addTab(self.recs_tab, "âœ… Recommendations")
        
        layout.addWidget(self.tabs)
        
        return panel
    
    def create_stats_tab(self):
        """Create enhanced team statistics tab with charts and detailed metrics"""
        # Main widget with scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        
        # Content widget
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setSpacing(25)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Title
        self.stats_title = QLabel("Team Statistics")
        self.stats_title.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        self.stats_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.stats_title.setStyleSheet("color: #0066cc; padding: 10px;")
        layout.addWidget(self.stats_title)
        
        # === SECTION 1: OVERVIEW METRICS ===
        overview_group = QGroupBox("📊 Overview")
        overview_group.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        overview_group.setMinimumHeight(120)
        overview_layout = QGridLayout()
        overview_layout.setSpacing(15)
        
        self.overview_labels = {}
        metrics = ['Games', 'Wins', 'Draws', 'Losses', 'Win Rate']
        for i, metric in enumerate(metrics):
            label = QLabel(metric)
            label.setFont(QFont("Arial", 10, QFont.Weight.Bold))
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            
            value = QLabel("--")
            value.setFont(QFont("Arial", 20, QFont.Weight.Bold))
            value.setAlignment(Qt.AlignmentFlag.AlignCenter)
            value.setMinimumHeight(50)
            
            # Color coding
            if metric == 'Wins':
                value.setStyleSheet("color: #28a745; background-color: #d4edda; border-radius: 5px; padding: 10px;")
            elif metric == 'Losses':
                value.setStyleSheet("color: #dc3545; background-color: #f8d7da; border-radius: 5px; padding: 10px;")
            elif metric == 'Win Rate':
                value.setStyleSheet("color: #0066cc; background-color: #d1ecf1; border-radius: 5px; padding: 10px;")
            elif metric == 'Draws':
                value.setStyleSheet("color: #6c757d; background-color: #e2e3e5; border-radius: 5px; padding: 10px;")
            else:
                value.setStyleSheet("background-color: #d1ecf1; border-radius: 5px; padding: 10px;")
            
            overview_layout.addWidget(label, 0, i)
            overview_layout.addWidget(value, 1, i)
            self.overview_labels[metric] = value
        
        overview_group.setLayout(overview_layout)
        layout.addWidget(overview_group)
        
        # === SECTION 2: ADVANCED CHARTS (NEW!) ===
        charts_scroll = QScrollArea()
        charts_scroll.setWidgetResizable(True)
        charts_scroll.setMinimumHeight(600)
        
        charts_widget = QWidget()
        charts_main_layout = QVBoxLayout(charts_widget)
        
        # Row 1: Form & Elo Timeline (Full Width)
        self.form_elo_container = QWidget()
        self.form_elo_container.setMinimumSize(900, 400)
        form_elo_layout = QVBoxLayout(self.form_elo_container)
        self.form_elo_label = QLabel("📈 Run analysis to see Form & Elo Evolution")
        self.form_elo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        form_elo_layout.addWidget(self.form_elo_label)
        charts_main_layout.addWidget(self.form_elo_container)
        
        # Row 2: HT to FT Analysis (Left) + Home/Away Radar (Right)
        row2_layout = QHBoxLayout()
        row2_layout.setSpacing(15)
        
        self.ht_ft_container = QWidget()
        self.ht_ft_container.setMinimumSize(450, 450)
        ht_ft_layout = QVBoxLayout(self.ht_ft_container)
        self.ht_ft_label = QLabel("⚽ Run analysis to see HT-FT Performance")
        self.ht_ft_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ht_ft_layout.addWidget(self.ht_ft_label)
        row2_layout.addWidget(self.ht_ft_container)
        
        self.radar_container = QWidget()
        self.radar_container.setMinimumSize(450, 450)
        radar_layout = QVBoxLayout(self.radar_container)
        self.radar_label = QLabel("🏠 Run analysis to see Home vs Away Radar")
        self.radar_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        radar_layout.addWidget(self.radar_label)
        row2_layout.addWidget(self.radar_container)
        
        charts_main_layout.addLayout(row2_layout)
        
        # Row 3: Shot Efficiency Quadrant (Left) + Odds vs Reality (Right)
        row3_layout = QHBoxLayout()
        row3_layout.setSpacing(15)
        
        self.shot_efficiency_container = QWidget()
        self.shot_efficiency_container.setMinimumSize(450, 450)
        shot_layout = QVBoxLayout(self.shot_efficiency_container)
        self.shot_label = QLabel("🎯 Run analysis to see Shot Efficiency")
        self.shot_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        shot_layout.addWidget(self.shot_label)
        row3_layout.addWidget(self.shot_efficiency_container)
        
        self.odds_reality_container = QWidget()
        self.odds_reality_container.setMinimumSize(450, 450)
        odds_layout = QVBoxLayout(self.odds_reality_container)
        self.odds_reality_label = QLabel("🎲 Run analysis to see Odds vs Reality")
        self.odds_reality_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        odds_layout.addWidget(self.odds_reality_label)
        row3_layout.addWidget(self.odds_reality_container)
        
        charts_main_layout.addLayout(row3_layout)
        
        charts_scroll.setWidget(charts_widget)
        layout.addWidget(charts_scroll)
        
        # === SECTION 3: HOME VS AWAY TABLE ===
        homeaway_group = QGroupBox("🏠 Home vs Away Performance")
        homeaway_group.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        homeaway_group.setMinimumHeight(300)
        homeaway_layout = QVBoxLayout()
        
        self.homeaway_table = QTableWidget(7, 3)
        self.homeaway_table.setHorizontalHeaderLabels(['Metric', 'Home', 'Away'])
        self.homeaway_table.verticalHeader().setVisible(False)
        self.homeaway_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.homeaway_table.setMinimumHeight(280)
        self.homeaway_table.setFont(QFont("Arial", 10))
        
        # Style the table
        self.homeaway_table.setStyleSheet("""
            QTableWidget {
                gridline-color: #ddd;
                font-size: 11px;
            }
            QTableWidget::item {
                padding: 8px;
            }
            QHeaderView::section {
                background-color: #0066cc;
                color: white;
                font-weight: bold;
                padding: 10px;
                font-size: 11px;
            }
        """)
        
        homeaway_layout.addWidget(self.homeaway_table)
        homeaway_group.setLayout(homeaway_layout)
        layout.addWidget(homeaway_group)
        
        # === SECTION 4: OFFENSIVE & DEFENSIVE METRICS ===
        metrics_group = QGroupBox("⚔️ Offensive & Defensive Metrics")
        metrics_group.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        metrics_group.setMinimumHeight(300)
        metrics_layout = QHBoxLayout()
        metrics_layout.setSpacing(20)
        
        # Offensive metrics
        offensive_widget = QWidget()
        offensive_layout = QVBoxLayout(offensive_widget)
        offensive_title = QLabel("⚽ Offensive")
        offensive_title.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        offensive_title.setStyleSheet("color: #28a745; padding: 5px;")
        offensive_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        offensive_layout.addWidget(offensive_title)
        
        self.offensive_table = QTableWidget(5, 2)
        self.offensive_table.setHorizontalHeaderLabels(['Metric', 'Value'])
        self.offensive_table.verticalHeader().setVisible(False)
        self.offensive_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.offensive_table.setMinimumHeight(250)
        self.offensive_table.setFont(QFont("Arial", 10))
        self.offensive_table.setStyleSheet("""
            QTableWidget {
                gridline-color: #ddd;
            }
            QTableWidget::item {
                padding: 8px;
            }
            QHeaderView::section {
                background-color: #28a745;
                color: white;
                font-weight: bold;
                padding: 8px;
            }
        """)
        offensive_layout.addWidget(self.offensive_table)
        
        metrics_layout.addWidget(offensive_widget)
        
        # Defensive metrics
        defensive_widget = QWidget()
        defensive_layout = QVBoxLayout(defensive_widget)
        defensive_title = QLabel("🛡️ Defensive")
        defensive_title.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        defensive_title.setStyleSheet("color: #dc3545; padding: 5px;")
        defensive_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        defensive_layout.addWidget(defensive_title)
        
        self.defensive_table = QTableWidget(5, 2)
        self.defensive_table.setHorizontalHeaderLabels(['Metric', 'Value'])
        self.defensive_table.verticalHeader().setVisible(False)
        self.defensive_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.defensive_table.setMinimumHeight(250)
        self.defensive_table.setFont(QFont("Arial", 10))
        self.defensive_table.setStyleSheet("""
            QTableWidget {
                gridline-color: #ddd;
            }
            QTableWidget::item {
                padding: 8px;
            }
            QHeaderView::section {
                background-color: #dc3545;
                color: white;
                font-weight: bold;
                padding: 8px;
            }
        """)
        defensive_layout.addWidget(self.defensive_table)
        
        metrics_layout.addWidget(defensive_widget)
        
        metrics_group.setLayout(metrics_layout)
        layout.addWidget(metrics_group)
        
        # === SECTION 5: PREDICTIVE METRICS ===
        predictive_group = QGroupBox("🔮 Predictive Metrics")
        predictive_group.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        predictive_group.setMinimumHeight(150)
        predictive_layout = QGridLayout()
        predictive_layout.setSpacing(15)
        predictive_layout.setContentsMargins(15, 15, 15, 15)
        
        self.predictive_labels = {}
        pred_metrics = ['Form Trend', 'Momentum Score', 'Expected Points', 'Actual Points', 'Points Diff', 'Status']
        for i, metric in enumerate(pred_metrics):
            label = QLabel(metric + ":")
            label.setFont(QFont("Arial", 11))
            value = QLabel("--")
            value.setFont(QFont("Arial", 12, QFont.Weight.Bold))
            value.setMinimumHeight(35)
            value.setStyleSheet("background-color: transparent; border: 1px solid #444; border-radius: 3px; padding: 8px;")
            
            row = i // 2
            col = (i % 2) * 2
            predictive_layout.addWidget(label, row, col)
            predictive_layout.addWidget(value, row, col + 1)
            self.predictive_labels[metric] = value
        
        predictive_group.setLayout(predictive_layout)
        layout.addWidget(predictive_group)
        
        # === SECTION 5.5: BETTING ODDS ANALYSIS ===
        odds_group = QGroupBox("🎲 Betting Odds Analysis (Expected vs Actual)")
        odds_group.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        odds_group.setMinimumHeight(550)
        odds_layout = QVBoxLayout()
        odds_layout.setSpacing(15)
        
        # Performance label at top
        self.odds_performance_label = QLabel("")
        self.odds_performance_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.odds_performance_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        self.odds_performance_label.setStyleSheet("padding: 10px; border-radius: 5px;")
        self.odds_performance_label.setMinimumHeight(60)
        self.odds_performance_label.hide()
        odds_layout.addWidget(self.odds_performance_label)
        
        # Info label
        self.odds_info_label = QLabel("Betting odds data not available")
        self.odds_info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.odds_info_label.setFont(QFont("Arial", 10))
        self.odds_info_label.setStyleSheet("color: #6c757d; padding: 10px;")
        odds_layout.addWidget(self.odds_info_label)
        
        # Chart container
        self.odds_chart_container = QWidget()
        self.odds_chart_container.setMinimumSize(600, 450)
        self.odds_chart_container.hide()  # Hidden until data available
        odds_chart_layout = QVBoxLayout(self.odds_chart_container)
        odds_chart_layout.setContentsMargins(0, 0, 0, 0)
        odds_layout.addWidget(self.odds_chart_container)
        
        odds_group.setLayout(odds_layout)
        layout.addWidget(odds_group)
        
        # === SECTION 6: BEST & WORST MATCHES ===
        matches_group = QGroupBox("🏆 Best Victories & Worst Defeats")
        matches_group.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        matches_group.setMinimumHeight(250)
        matches_layout = QHBoxLayout()
        matches_layout.setSpacing(20)
        
        # Best victories
        best_widget = QWidget()
        best_layout = QVBoxLayout(best_widget)
        best_title = QLabel("🏆 Best Victories")
        best_title.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        best_title.setStyleSheet("color: #28a745; padding: 5px; background-color: #d4edda; border-radius: 3px;")
        best_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        best_layout.addWidget(best_title)
        
        self.best_matches_list = QListWidget()
        self.best_matches_list.setMinimumHeight(180)
        self.best_matches_list.setFont(QFont("Arial", 10))
        self.best_matches_list.setStyleSheet("""
            QListWidget {
                background-color: transparent;
                border: 1px solid #28a745;
                border-radius: 5px;
                padding: 5px;
            }
            QListWidget::item {
                padding: 8px;
                border-bottom: 1px solid #333;
                color: #28a745;
            }
            QListWidget::item:hover {
                background-color: rgba(40, 167, 69, 0.2);
            }
        """)
        best_layout.addWidget(self.best_matches_list)
        
        matches_layout.addWidget(best_widget)
        
        # Worst defeats
        worst_widget = QWidget()
        worst_layout = QVBoxLayout(worst_widget)
        worst_title = QLabel("âŒ Worst Defeats")
        worst_title.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        worst_title.setStyleSheet("color: #dc3545; padding: 5px; background-color: #f8d7da; border-radius: 3px;")
        worst_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        worst_layout.addWidget(worst_title)
        
        self.worst_matches_list = QListWidget()
        self.worst_matches_list.setMinimumHeight(180)
        self.worst_matches_list.setFont(QFont("Arial", 10))
        self.worst_matches_list.setStyleSheet("""
            QListWidget {
                background-color: transparent;
                border: 1px solid #dc3545;
                border-radius: 5px;
                padding: 5px;
            }
            QListWidget::item {
                padding: 8px;
                border-bottom: 1px solid #333;
                color: #dc3545;
            }
            QListWidget::item:hover {
                background-color: rgba(220, 53, 69, 0.2);
            }
        """)
        worst_layout.addWidget(self.worst_matches_list)
        
        matches_layout.addWidget(worst_widget)
        
        matches_group.setLayout(matches_layout)
        layout.addWidget(matches_group)
        
        # Add bottom padding
        layout.addSpacing(30)
        
        scroll.setWidget(content)
        return scroll
    
    def create_issues_tab(self):
        """Create issues selection tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Selection method
        method_group = QGroupBox("Issue Selection Method")
        method_layout = QVBoxLayout()
        
        self.issue_method_group = QButtonGroup()
        self.rb_manual = QRadioButton("Manually Select")
        self.rb_severity = QRadioButton("Select by Severity")
        self.rb_category = QRadioButton("Select by Category")
        self.rb_all = QRadioButton("Select All")
        
        self.issue_method_group.addButton(self.rb_manual, 0)
        self.issue_method_group.addButton(self.rb_severity, 1)
        self.issue_method_group.addButton(self.rb_category, 2)
        self.issue_method_group.addButton(self.rb_all, 3)
        
        self.rb_manual.setChecked(True)
        self.issue_method_group.buttonClicked.connect(self.update_issue_selection)
        
        method_layout.addWidget(self.rb_manual)
        method_layout.addWidget(self.rb_severity)
        method_layout.addWidget(self.rb_category)
        method_layout.addWidget(self.rb_all)
        method_group.setLayout(method_layout)
        
        layout.addWidget(method_group)
        
        # Selection options
        self.severity_combo = QComboBox()
        self.severity_combo.addItems(["high", "medium", "low"])
        self.severity_combo.hide()
        self.severity_combo.currentTextChanged.connect(self.filter_issues_by_severity)
        
        self.category_combo = QComboBox()
        self.category_combo.hide()
        self.category_combo.currentTextChanged.connect(self.filter_issues_by_category)
        
        layout.addWidget(self.severity_combo)
        layout.addWidget(self.category_combo)
        
        # Issues list
        self.issues_list = QListWidget()
        self.issues_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        layout.addWidget(QLabel("Select Issues to Address:"))
        layout.addWidget(self.issues_list)
        
        # Generate recommendations button
        self.btn_recommend = QPushButton("🎯 Generate Recommendations")
        self.btn_recommend.setEnabled(False)
        self.btn_recommend.clicked.connect(self.generate_recommendations)
        self.btn_recommend.setStyleSheet("QPushButton { background-color: #28a745; color: white; font-weight: bold; padding: 10px; }")
        layout.addWidget(self.btn_recommend)
        
        return widget
    
    def create_recommendations_tab(self):
        """Create recommendations display tab with position filter"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Title
        title = QLabel("🎯 Player Recommendations")
        title.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("color: #0066cc; padding: 10px;")
        layout.addWidget(title)
        
        # === ARCHETYPE FILTER SECTION ===
        # Replaces the old position-checkbox filter.
        # Populated with needed archetypes (ordered by weight) after recommendations run.
        filter_group = QGroupBox("🏷️ Filter by Archetype")
        filter_group.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        filter_layout = QHBoxLayout()
        filter_layout.setContentsMargins(10, 8, 10, 8)

        filter_label = QLabel("Archetype:")
        filter_label.setFont(QFont("Arial", 10))
        filter_layout.addWidget(filter_label)

        self.archetype_combo = QComboBox()
        self.archetype_combo.addItem("All archetypes")
        self.archetype_combo.setFont(QFont("Arial", 10))
        self.archetype_combo.setMinimumWidth(220)
        self.archetype_combo.currentTextChanged.connect(self.on_archetype_filter_changed)
        filter_layout.addWidget(self.archetype_combo, stretch=1)

        filter_hint = QLabel("Needed archetypes listed first (% = team priority weight)")
        filter_hint.setFont(QFont("Arial", 9))
        filter_hint.setStyleSheet("color: #6c757d; padding-left: 8px;")
        filter_layout.addWidget(filter_hint)

        filter_group.setLayout(filter_layout)
        layout.addWidget(filter_group)

        # Storage for archetype_needs dict populated after recommendations run
        self.archetype_needs = {}
        
        # Info label
        self.recs_info_label = QLabel("Generate recommendations to see results...")
        self.recs_info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.recs_info_label.setFont(QFont("Arial", 10))
        self.recs_info_label.setStyleSheet("color: #6c757d; padding: 10px;")
        layout.addWidget(self.recs_info_label)
        
        # Scroll area for player cards (replacing table)
        self.recs_scroll = QScrollArea()
        self.recs_scroll.setWidgetResizable(True)
        self.recs_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.recs_scroll.setStyleSheet("""
            QScrollArea {
                border: 1px solid #444;
                background-color: transparent;
            }
        """)
        
        # Container for player cards
        self.recs_container = QWidget()
        self.recs_layout = QVBoxLayout(self.recs_container)
        self.recs_layout.setSpacing(10)
        self.recs_layout.setContentsMargins(10, 10, 10, 10)
        self.recs_layout.addStretch()  # Push cards to top
        
        self.recs_scroll.setWidget(self.recs_container)
        layout.addWidget(self.recs_scroll)
        
        # Export buttons
        export_layout = QHBoxLayout()
        
        self.btn_export_csv = QPushButton("💾 Export to CSV")
        self.btn_export_csv.setEnabled(False)
        self.btn_export_csv.clicked.connect(self.export_recommendations)
        self.btn_export_csv.setFont(QFont("Arial", 10))
        self.btn_export_csv.setStyleSheet("""
            QPushButton {
                background-color: #28a745;
                color: white;
                padding: 8px 15px;
                border-radius: 5px;
                border: none;
            }
            QPushButton:hover {
                background-color: #218838;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        export_layout.addWidget(self.btn_export_csv)
        
        self.btn_export_pdf = QPushButton("📄 Export Scouting Report to PDF")
        self.btn_export_pdf.setEnabled(False)
        self.btn_export_pdf.clicked.connect(self.export_full_report_pdf)
        self.btn_export_pdf.setFont(QFont("Arial", 10))
        self.btn_export_pdf.setStyleSheet("""
            QPushButton {
                background-color: #dc3545;
                color: white;
                padding: 8px 15px;
                border-radius: 5px;
                border: none;
            }
            QPushButton:hover {
                background-color: #c82333;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        export_layout.addWidget(self.btn_export_pdf)
        
        layout.addLayout(export_layout)
        
        # Store original recommendations for filtering
        self.all_recommendations = []
        
        return widget
    
    def update_budget_label(self):
        """Update budget label when slider changes"""
        value = self.budget_slider.value()
        if value >= 200:
            self.budget_value_label.setText("No limit")
        else:
            self.budget_value_label.setText(f"â‚¬{value}M")
    
    # ===================== EVENT HANDLERS =====================
    
    def load_matches_file(self):
        """Load matches data file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Matches File",
            "",
            "Data Files (*.csv *.xlsx);;CSV Files (*.csv);;Excel Files (*.xlsx);;All Files (*)"
        )
        
        if file_path:
            try:
                # Create a file-like object
                with open(file_path, 'rb') as f:
                    class FileObj:
                        def __init__(self, path, content):
                            self.name = path
                            self.content = content
                            self.pos = 0
                        
                        def read(self, size=-1):
                            if size == -1:
                                result = self.content[self.pos:]
                                self.pos = len(self.content)
                            else:
                                result = self.content[self.pos:self.pos + size]
                                self.pos += size
                            return result
                        
                        def seek(self, pos):
                            self.pos = pos
                    
                    content = f.read()
                    file_obj = FileObj(file_path, content)
                
                self.matches_df = load_matches_data(file_obj)
                
                # Validate that it has required columns
                required_cols = ['HomeTeam', 'AwayTeam']
                missing_cols = [col for col in required_cols if col not in self.matches_df.columns]
                
                if missing_cols:
                    QMessageBox.warning(
                        self,
                        "Missing Columns",
                        f"Matches file is missing required columns: {', '.join(missing_cols)}\n\n"
                        f"Required columns: HomeTeam, AwayTeam\n"
                        f"Found columns: {', '.join(self.matches_df.columns[:10].tolist())}..."
                    )
                    self.matches_df = None
                    return
                
                self.matches_file_label.setText(f"âœ… {file_path.split('/')[-1]}")
                self.statusBar().showMessage(f"Loaded {len(self.matches_df)} matches")
                
                self.check_data_loaded()
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load matches file:\n{str(e)}")
    
    def load_players_file(self):
        """Load players data file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Players File",
            "",
            "Data Files (*.csv *.xlsx);;CSV Files (*.csv);;Excel Files (*.xlsx);;All Files (*)"
        )
        
        if file_path:
            try:
                # Create a file-like object
                with open(file_path, 'rb') as f:
                    class FileObj:
                        def __init__(self, path, content):
                            self.name = path
                            self.content = content
                            self.pos = 0
                        
                        def read(self, size=-1):
                            if size == -1:
                                result = self.content[self.pos:]
                                self.pos = len(self.content)
                            else:
                                result = self.content[self.pos:self.pos + size]
                                self.pos += size
                            return result
                        
                        def seek(self, pos):
                            self.pos = pos
                    
                    content = f.read()
                    file_obj = FileObj(file_path, content)
                
                self.players_df, messages = load_players_data(file_obj)
                
                # Validate that it has required columns
                required_cols = ['Player', 'Pos', 'Squad']
                missing_cols = [col for col in required_cols if col not in self.players_df.columns]
                
                if missing_cols:
                    QMessageBox.warning(
                        self,
                        "Missing Columns",
                        f"Players file is missing recommended columns: {', '.join(missing_cols)}\n\n"
                        f"Recommended columns: Player, Pos, Squad, Age, 90s\n"
                        f"Found columns: {', '.join(self.players_df.columns[:10].tolist())}...\n\n"
                        f"Analysis will continue but may have limited functionality."
                    )
                
                self.players_file_label.setText(f"âœ… {file_path.split('/')[-1]}")
                self.statusBar().showMessage(f"Loaded {len(self.players_df)} players")
                
                self.check_data_loaded()
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load players file:\n{str(e)}")
    
    # ------------------------------------------------------------------ #
    # Season helpers                                                       #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _date_to_season(date_str):
        """Convert a DD/MM/YYYY date string to a football-season label.

        A season runs from ~July of year Y to ~June of year Y+1, so:
          months 7-12  →  "Y/YY+1"  (e.g. Aug 2023 → "2023/24")
          months 1-6   →  "Y-1/YY"  (e.g. Mar 2024 → "2023/24")
        """
        try:
            parts = str(date_str).strip().split('/')
            month = int(parts[1])
            year = int(parts[2])
        except (IndexError, ValueError):
            return None
        if month >= 7:
            return f"{year}/{str(year + 1)[-2:]}"
        else:
            return f"{year - 1}/{str(year)[-2:]}"

    def get_last_four_seasons(self, matches):
        """Return a list of the four most recent season labels derived from MatchDate."""
        if 'MatchDate' not in matches.columns:
            return []
        seasons = matches['MatchDate'].apply(self._date_to_season).dropna().unique()
        seasons_sorted = sorted(seasons)
        return seasons_sorted[-4:]

    def filter_matches_by_season(self, matches, season):
        """Return *matches* filtered to *season*.

        If *season* is "All seasons" the trimmed dataset (last 4 seasons) is
        returned unchanged.  Otherwise only rows whose MatchDate maps to the
        selected season are included.
        """
        if season == "All seasons" or 'MatchDate' not in matches.columns:
            return matches
        mask = matches['MatchDate'].apply(self._date_to_season) == season
        return matches[mask].copy()

    def on_season_changed(self, season_text):
        """Handle season combo-box selection change."""
        self.selected_season = season_text

    # ------------------------------------------------------------------ #

    def check_data_loaded(self):
        """Check if both data files are loaded and enable team selection"""
        if self.matches_df is not None and self.players_df is not None:
            # Clean data
            self.matches_df, self.players_df = clean_data(self.matches_df, self.players_df)

            # Trim to last 4 seasons (done once after loading)
            last_four = self.get_last_four_seasons(self.matches_df)
            if last_four:
                mask = self.matches_df['MatchDate'].apply(self._date_to_season).isin(last_four)
                self.matches_trimmed_df = self.matches_df[mask].copy()
            else:
                self.matches_trimmed_df = self.matches_df.copy()

            # Populate season combo box
            self.season_combo.blockSignals(True)
            self.season_combo.clear()
            self.season_combo.addItem("All seasons")
            for s in last_four:
                self.season_combo.addItem(s)
            self.season_combo.setCurrentIndex(0)
            self.season_combo.setEnabled(True)
            self.season_combo.blockSignals(False)
            self.selected_season = "All seasons"

            # Populate team combo from MATCHES dataframe, not players
            try:
                # Check if required columns exist in matches dataframe
                if 'HomeTeam' not in self.matches_df.columns or 'AwayTeam' not in self.matches_df.columns:
                    QMessageBox.critical(
                        self, 
                        "Error", 
                        "Matches file must contain 'HomeTeam' and 'AwayTeam' columns."
                    )
                    return
                
                all_teams = sorted(pd.concat([
                    self.matches_df["HomeTeam"],
                    self.matches_df["AwayTeam"]
                ]).unique())
                
                if len(all_teams) == 0:
                    QMessageBox.warning(
                        self,
                        "Warning",
                        "No teams found in the matches file."
                    )
                    return
                
                self.team_combo.clear()
                self.team_combo.addItems(all_teams)
                self.team_combo.setEnabled(True)
                self.btn_analyze.setEnabled(True)
                
                self.statusBar().showMessage(f"Data loaded! {len(all_teams)} teams available for analysis.")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to process data:\n{str(e)}")
    
    def run_analysis(self):
        """Run team analysis"""
        self.current_team = self.team_combo.currentText()
        
        if not self.current_team:
            QMessageBox.warning(self, "Warning", "Please select a team first.")
            return
        
        # Get scoring method selection
        use_percentiles = self.rb_percentile.isChecked()
        
        # Show progress dialog
        progress = QProgressDialog(f"Analyzing {self.current_team}...", None, 0, 0, self)
        progress.setWindowTitle("Analysis in Progress")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.show()
        
        # Apply season filter before analysis
        base_matches = self.matches_trimmed_df if self.matches_trimmed_df is not None else self.matches_df
        matches_for_analysis = self.filter_matches_by_season(base_matches, self.selected_season)

        # Run analysis in worker thread
        self.analysis_worker = AnalysisWorker(
            matches_for_analysis.copy(),
            self.players_df.copy(),
            self.current_team,
            use_percentiles=use_percentiles
        )
        self.analysis_worker.finished.connect(lambda *args: self.on_analysis_complete(*args, progress))
        self.analysis_worker.error.connect(lambda msg: self.on_analysis_error(msg, progress))
        self.analysis_worker.start()
    
    def on_analysis_complete(self, df_team, patterns, evaluated_players, progress):
        """Handle analysis completion"""
        progress.close()
        
        self.df_team = df_team
        self.patterns = patterns
        self.evaluated_players = evaluated_players
        
        # Update statistics tab
        self.update_statistics_tab()
        
        # Update issues tab
        self.update_issues_tab()
        
        # Switch to statistics tab
        self.tabs.setCurrentIndex(0)
        
        self.statusBar().showMessage(f"Analysis complete for {self.current_team}")
        QMessageBox.information(self, "Success", f"Analysis complete for {self.current_team}!")
    
    def on_analysis_error(self, error_msg, progress):
        """Handle analysis error"""
        progress.close()
        QMessageBox.critical(self, "Analysis Error", error_msg)
        self.statusBar().showMessage("Analysis failed")
    
    def update_statistics_tab(self):
        """Update team statistics display with enhanced metrics and charts"""
        # Update title
        self.stats_title.setText(f"Team Statistics: {self.current_team}")
        
        # Get basic statistics
        stats = get_team_statistics(self.df_team)
        
        # === UPDATE OVERVIEW METRICS ===
        self.overview_labels['Games'].setText(str(stats['total_games']))
        self.overview_labels['Wins'].setText(str(stats['wins']))
        self.overview_labels['Draws'].setText(str(stats['draws']))
        self.overview_labels['Losses'].setText(str(stats['losses']))
        self.overview_labels['Win Rate'].setText(f"{stats['win_rate']:.1%}")
        
        # === CREATE ADVANCED CHARTS ===
        
        # 1. Form & Elo Timeline
        for i in reversed(range(self.form_elo_container.layout().count())): 
            self.form_elo_container.layout().itemAt(i).widget().setParent(None)
        
        form_elo_chart = FormEloTimelineWidget(self.df_team)
        self.form_elo_container.layout().addWidget(form_elo_chart)
        
        # 2. Half-Time to Full-Time Analysis
        for i in reversed(range(self.ht_ft_container.layout().count())): 
            self.ht_ft_container.layout().itemAt(i).widget().setParent(None)
        
        ht_ft_chart = HalfTimeFullTimeWidget(self.df_team)
        self.ht_ft_container.layout().addWidget(ht_ft_chart)
        
        # 3. Home vs Away Radar
        home_stats, away_stats = calculate_home_away_stats(self.df_team)
        
        for i in reversed(range(self.radar_container.layout().count())): 
            self.radar_container.layout().itemAt(i).widget().setParent(None)
        
        radar_chart = HomeAwayRadarWidget(home_stats, away_stats)
        self.radar_container.layout().addWidget(radar_chart)
        
        # 4. Shot Efficiency Quadrant
        for i in reversed(range(self.shot_efficiency_container.layout().count())): 
            self.shot_efficiency_container.layout().itemAt(i).widget().setParent(None)
        
        shot_chart = ShotEfficiencyQuadrantWidget(self.df_team)
        self.shot_efficiency_container.layout().addWidget(shot_chart)
        
        # 5. Odds vs Reality
        for i in reversed(range(self.odds_reality_container.layout().count())): 
            self.odds_reality_container.layout().itemAt(i).widget().setParent(None)
        
        odds_reality_chart = OddsVsRealityWidget(self.df_team)
        self.odds_reality_container.layout().addWidget(odds_reality_chart)
        
        # === UPDATE HOME/AWAY TABLE ===
        table_data = [
            ('Games Played', str(home_stats['games']), str(away_stats['games'])),
            ('Wins', str(home_stats['wins']), str(away_stats['wins'])),
            ('Draws', str(home_stats['draws']), str(away_stats['draws'])),
            ('Losses', str(home_stats['losses']), str(away_stats['losses'])),
            ('Win Rate', f"{home_stats['win_rate']:.1f}%", f"{away_stats['win_rate']:.1f}%"),
            ('Goals For/Game', f"{home_stats['goals_for']:.2f}", f"{away_stats['goals_for']:.2f}"),
            ('Goals Against/Game', f"{home_stats['goals_against']:.2f}", f"{away_stats['goals_against']:.2f}"),
        ]
        
        self.homeaway_table.setRowCount(len(table_data))
        for row, (metric, home_val, away_val) in enumerate(table_data):
            self.homeaway_table.setItem(row, 0, QTableWidgetItem(metric))
            
            home_item = QTableWidgetItem(home_val)
            home_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.homeaway_table.setItem(row, 1, home_item)
            
            away_item = QTableWidgetItem(away_val)
            away_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.homeaway_table.setItem(row, 2, away_item)
        
        # === UPDATE OFFENSIVE/DEFENSIVE METRICS ===
        metrics = calculate_offensive_defensive_metrics(self.df_team)
        
        # Offensive table
        offensive_data = [
            ('Goals/Game', f"{metrics['goals_per_game']:.2f}"),
            ('Shots/Game', f"{metrics['shots_per_game']:.2f}"),
            ('Shots on Target/Game', f"{metrics['shots_on_target_per_game']:.2f}"),
            ('Shot Accuracy', f"{metrics['shot_accuracy']:.1f}%"),
            ('Conversion Rate', f"{metrics['conversion_rate']:.1f}%"),
        ]
        
        self.offensive_table.setRowCount(len(offensive_data))
        for row, (metric, value) in enumerate(offensive_data):
            self.offensive_table.setItem(row, 0, QTableWidgetItem(metric))
            value_item = QTableWidgetItem(value)
            value_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            value_item.setFont(QFont("Arial", 10, QFont.Weight.Bold))
            self.offensive_table.setItem(row, 1, value_item)
        
        # Defensive table
        defensive_data = [
            ('Goals Conceded/Game', f"{metrics['goals_conceded_per_game']:.2f}"),
            ('Clean Sheets', str(metrics['clean_sheets'])),
            ('Shots Allowed/Game', f"{metrics['shots_allowed_per_game']:.2f}"),
            ('Yellow Cards/Game', f"{metrics['yellow_cards_per_game']:.2f}"),
            ('Red Cards (Total)', str(int(metrics['red_cards_total']))),
        ]
        
        self.defensive_table.setRowCount(len(defensive_data))
        for row, (metric, value) in enumerate(defensive_data):
            self.defensive_table.setItem(row, 0, QTableWidgetItem(metric))
            value_item = QTableWidgetItem(value)
            value_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            value_item.setFont(QFont("Arial", 10, QFont.Weight.Bold))
            self.defensive_table.setItem(row, 1, value_item)
        
        # === UPDATE PREDICTIVE METRICS ===
        pred_metrics = calculate_predictive_metrics(self.df_team)
        
        self.predictive_labels['Form Trend'].setText(pred_metrics['form_trend'])
        self.predictive_labels['Momentum Score'].setText(f"{pred_metrics['momentum_score']:.1f}/10")
        self.predictive_labels['Expected Points'].setText(str(pred_metrics['expected_points']))
        self.predictive_labels['Actual Points'].setText(str(pred_metrics['actual_points']))
        
        points_diff = pred_metrics['points_difference']
        points_diff_text = f"{points_diff:+.1f}"
        self.predictive_labels['Points Diff'].setText(points_diff_text)
        
        # Color code points difference
        if points_diff > 3:
            self.predictive_labels['Points Diff'].setStyleSheet("color: #28a745;")
        elif points_diff < -3:
            self.predictive_labels['Points Diff'].setStyleSheet("color: #dc3545;")
        else:
            self.predictive_labels['Points Diff'].setStyleSheet("color: #6c757d;")
        
        self.predictive_labels['Status'].setText(pred_metrics['performance_status'])
        
        # === UPDATE BETTING ODDS ANALYSIS ===
        odds_analysis = calculate_betting_odds_analysis(self.df_team)
        
        if odds_analysis['has_odds_data']:
            # Show chart and hide info label
            self.odds_info_label.hide()
            self.odds_chart_container.show()
            self.odds_performance_label.show()
            
            # Clear existing chart
            for i in reversed(range(self.odds_chart_container.layout().count())): 
                self.odds_chart_container.layout().itemAt(i).widget().setParent(None)
            
            # Create chart data (now integers, not floats)
            expected_data = [
                odds_analysis['expected_wins'],
                odds_analysis['expected_draws'],
                odds_analysis['expected_losses']
            ]
            actual_data = [
                odds_analysis['actual_wins'],
                odds_analysis['actual_draws'],
                odds_analysis['actual_losses']
            ]
            
            # Create and add chart
            odds_chart = BettingOddsChartWidget(expected_data, actual_data)
            self.odds_chart_container.layout().addWidget(odds_chart)
            
            # Update performance label
            performance_text = f"{odds_analysis['odds_performance']}\n"
            performance_text += f"Wins: {odds_analysis['wins_diff']:+.1f} | "
            performance_text += f"Draws: {odds_analysis['draws_diff']:+.1f} | "
            performance_text += f"Losses: {odds_analysis['losses_diff']:+.1f}"
            
            self.odds_performance_label.setText(performance_text)
            
            # Color code based on performance
            if "Beating" in odds_analysis['odds_performance']:
                self.odds_performance_label.setStyleSheet(
                    "color: #28a745; background-color: rgba(40, 167, 69, 0.1); "
                    "padding: 10px; border-radius: 5px; border: 2px solid #28a745;"
                )
            elif "Below" in odds_analysis['odds_performance']:
                self.odds_performance_label.setStyleSheet(
                    "color: #dc3545; background-color: rgba(220, 53, 69, 0.1); "
                    "padding: 10px; border-radius: 5px; border: 2px solid #dc3545;"
                )
            else:
                self.odds_performance_label.setStyleSheet(
                    "color: #0066cc; background-color: rgba(0, 102, 204, 0.1); "
                    "padding: 10px; border-radius: 5px; border: 2px solid #0066cc;"
                )
        else:
            # No odds data available
            self.odds_info_label.show()
            self.odds_chart_container.hide()
            self.odds_performance_label.hide()
        
        # === UPDATE BEST/WORST MATCHES ===
        best_matches, worst_matches = get_best_worst_matches(self.df_team)
        
        self.best_matches_list.clear()
        if best_matches:
            for i, match in enumerate(best_matches, 1):
                text = f"{i}. vs {match['opponent']} - {match['score']} ({match['location']})"
                self.best_matches_list.addItem(text)
        else:
            self.best_matches_list.addItem("No victories recorded")
        
        self.worst_matches_list.clear()
        if worst_matches:
            for i, match in enumerate(worst_matches, 1):
                text = f"{i}. vs {match['opponent']} - {match['score']} ({match['location']})"
                self.worst_matches_list.addItem(text)
        else:
            self.worst_matches_list.addItem("No defeats recorded")
    
    def update_issues_tab(self):
        """Update issues list"""
        self.all_issues = get_all_issues(self.patterns)
        
        if not self.all_issues:
            self.issues_list.addItem("No significant weaknesses detected! Team is well-balanced. 👍")
            self.btn_recommend.setEnabled(False)
            return
        
        # Populate issues list
        self.issues_list.clear()
        for issue in self.all_issues:
            item_text = f"{issue['number']}. [{issue['category'].upper()}] {issue['description']} (Severity: {issue['severity']})"
            self.issues_list.addItem(item_text)
        
        # Populate category combo
        categories = sorted(list(set(i['category'] for i in self.all_issues)))
        self.category_combo.clear()
        self.category_combo.addItems(categories)
        
        self.btn_recommend.setEnabled(True)
        
        # Switch to issues tab
        self.tabs.setCurrentIndex(1)
    
    def update_issue_selection(self):
        """Update issue selection based on radio button"""
        selected = self.issue_method_group.checkedId()
        
        # Hide all selection widgets
        self.severity_combo.hide()
        self.category_combo.hide()
        self.issues_list.clearSelection()
        
        if selected == 0:  # Manual
            pass
        elif selected == 1:  # Severity
            self.severity_combo.show()
            self.filter_issues_by_severity()
        elif selected == 2:  # Category
            self.category_combo.show()
            self.filter_issues_by_category()
        elif selected == 3:  # All
            for i in range(self.issues_list.count()):
                self.issues_list.item(i).setSelected(True)
    
    def filter_issues_by_severity(self):
        """Filter issues by severity"""
        severity = self.severity_combo.currentText()
        self.issues_list.clearSelection()
        
        for i, issue in enumerate(self.all_issues):
            if issue['severity'] == severity:
                self.issues_list.item(i).setSelected(True)
    
    def filter_issues_by_category(self):
        """Filter issues by category"""
        category = self.category_combo.currentText()
        self.issues_list.clearSelection()
        
        for i, issue in enumerate(self.all_issues):
            if issue['category'] == category:
                self.issues_list.item(i).setSelected(True)
    
    def generate_recommendations(self):
        """Generate player recommendations"""
        # Get selected issues
        selected_items = self.issues_list.selectedItems()
        
        if not selected_items:
            QMessageBox.warning(self, "Warning", "Please select at least one issue to address.")
            return
        
        # Build custom patterns from selected issues
        selected_indices = [self.issues_list.row(item) for item in selected_items]
        selected_issues = [self.all_issues[i] for i in selected_indices]
        
        custom_patterns = {}
        for issue in selected_issues:
            category = issue['category']
            if category not in custom_patterns:
                custom_patterns[category] = {'issues': []}
            custom_patterns[category]['issues'].append(issue['full_issue'])
        
        # Get filters
        age_input = self.age_input.text()
        age_min, age_max = parse_age_range(age_input)
        
        foot_choice = self.foot_combo.currentText()
        if foot_choice == "Right-footed":
            preferred_foot = "right"
        elif foot_choice == "Left-footed":
            preferred_foot = "left"
        else:
            preferred_foot = None
        
        # Free agents filter
        free_agents_only = self.free_agents_checkbox.isChecked()
        
        # Budget filter
        budget_value = self.budget_slider.value()
        max_budget = budget_value * 1_000_000 if budget_value < 200 else None  # Convert to actual value
        
        top_n = self.top_n_spin.value()
        
        # Get scoring method
        scoring_method = "percentile" if self.rb_percentile.isChecked() else "min-max"
        
        # Show progress dialog
        progress = QProgressDialog("Generating recommendations...", None, 0, 0, self)
        progress.setWindowTitle("Finding Players")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.show()
        
        # Run recommendation in worker thread
        self.rec_worker = RecommendationWorker(
            custom_patterns,
            self.evaluated_players.copy(),
            self.current_team,
            top_n,
            age_min,
            age_max,
            preferred_foot,
            free_agents_only,
            max_budget,
            scoring_method=scoring_method
        )
        self.rec_worker.finished.connect(lambda df, an: self.on_recommendations_complete(df, an, progress))
        self.rec_worker.error.connect(lambda msg: self.on_recommendations_error(msg, progress))
        self.rec_worker.start()
    
    def on_recommendations_complete(self, recommendations_df, archetype_needs, progress):
        """Handle recommendations completion — now receives archetype_needs alongside df."""
        progress.close()

        self.recommendations_df = recommendations_df
        self.all_recommendations = recommendations_df.copy()
        self.archetype_needs = archetype_needs  # {archetype_name: priority_weight}

        # Repopulate the archetype filter combo.
        # Needed archetypes come first (sorted by weight desc), then remaining known archetypes.
        self.archetype_combo.blockSignals(True)
        self.archetype_combo.clear()
        self.archetype_combo.addItem("All archetypes")
        for arch_name, weight in sorted(archetype_needs.items(), key=lambda x: -x[1]):
            self.archetype_combo.addItem(f"{arch_name}  ({weight * 100:.0f}%)")
        # Add any known archetypes not already in archetype_needs (greyed out hint)
        for arch_name in ARCHETYPE_DEFINITIONS:
            if arch_name not in archetype_needs:
                self.archetype_combo.addItem(arch_name)
        self.archetype_combo.blockSignals(False)

        if recommendations_df.empty:
            QMessageBox.information(self, "No Results", "No suitable players found for the selected criteria and filters.")
            return

        # Show grouped-by-archetype view for "All archetypes" on first load
        self.display_recommendations(recommendations_df, grouped=True)

        self.btn_export_csv.setEnabled(True)
        self.btn_export_pdf.setEnabled(True)
        self.tabs.setCurrentIndex(2)
        self.statusBar().showMessage(f"Generated {len(recommendations_df)} recommendations")
        QMessageBox.information(self, "Success", f"Found {len(recommendations_df)} suitable players!")

    def on_archetype_filter_changed(self, text: str):
        """Handle archetype ComboBox selection changes.

        'All archetypes' → grouped view.
        Specific archetype  → flat list filtered to that archetype column value.
        """
        if not hasattr(self, 'all_recommendations') or self.all_recommendations is None:
            return
        if not len(self.all_recommendations):
            return

        if not text or text == "All archetypes":
            self.display_recommendations(self.all_recommendations, grouped=True)
            return

        # Strip the weight suffix added in on_recommendations_complete ("Name  (35%)")
        arch_name = text.split("  (")[0].strip()

        # Filter by the Archetype column that _add_recommendation_reasoning populated
        if 'Archetype' in self.all_recommendations.columns:
            filtered = self.all_recommendations[
                self.all_recommendations['Archetype'] == arch_name
            ].copy()
        else:
            # Fallback if Archetype column is absent
            filtered = self.all_recommendations

        if filtered.empty:
            self.recs_info_label.setText(f"No recommendations tagged as '{arch_name}'")
            while self.recs_layout.count() > 1:
                item = self.recs_layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
            return

        self.display_recommendations(filtered, grouped=False)
        self.recs_info_label.setText(
            f"Showing {len(filtered)} players for archetype: {arch_name}"
        )

    def _resolve_column_aliases(self, df: "pd.DataFrame") -> dict:
        """Return {standard_name: actual_col} for optional card columns."""
        column_aliases = {
            'foot': ['foot', 'Foot', 'preferred_foot', 'Preferred Foot', 'Pref Foot'],
            'contract_expires': ['contract_expires', 'Contract', 'Contract Expires', 'contract', 'Contract Expiry'],
            'market_value': ['market_value', 'Market Value', 'Value', 'Transfer Value', 'market_val', 'value'],
        }
        actual = {}
        for std, aliases in column_aliases.items():
            for alias in aliases:
                if alias in df.columns:
                    actual[std] = alias
                    break
        return actual

    def display_recommendations(self, recommendations_df, grouped: bool = False):
        """Display recommendations as clickable player cards.

        grouped=True  → section headers per archetype (used for 'All archetypes' view).
        grouped=False → flat ranked list (used when a specific archetype is selected).
        """
        # Clear existing cards/headers
        while self.recs_layout.count() > 1:  # Keep the trailing stretch
            item = self.recs_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        actual_columns = self._resolve_column_aliases(recommendations_df)
        has_archetype_col = 'Archetype' in recommendations_df.columns

        if grouped and has_archetype_col and self.archetype_needs:
            # ── Grouped view: one section per needed archetype, ordered by weight ──
            rank = 1
            shown_players: set = set()
            arch_order = sorted(self.archetype_needs, key=lambda x: -self.archetype_needs[x])
            for arch_name in arch_order:
                group = recommendations_df[recommendations_df['Archetype'] == arch_name]
                if group.empty:
                    continue
                # Section header label
                weight_pct = int(self.archetype_needs.get(arch_name, 0) * 100)
                header = QLabel(f"  {arch_name}  ·  {weight_pct}% team need")
                header.setFont(QFont("Arial", 11, QFont.Weight.Bold))
                header.setStyleSheet(
                    "color: #ffd700; background-color: #1e1e2e; "
                    "padding: 6px 10px; border-radius: 4px; margin-top: 6px;"
                )
                self.recs_layout.insertWidget(self.recs_layout.count() - 1, header)
                for _, row in group.iterrows():
                    player_name = row.get('Player', '')
                    if player_name in shown_players:
                        continue
                    shown_players.add(player_name)
                    card = self.create_player_card(row, rank, actual_columns)
                    self.recs_layout.insertWidget(self.recs_layout.count() - 1, card)
                    rank += 1
            self.recs_info_label.setText(
                f"Showing {len(recommendations_df)} players grouped by archetype"
            )
        else:
            # ── Flat ranked view ──
            for idx, (_, row) in enumerate(recommendations_df.iterrows(), 1):
                card = self.create_player_card(row, idx, actual_columns)
                self.recs_layout.insertWidget(self.recs_layout.count() - 1, card)
            self.recs_info_label.setText(f"Found {len(recommendations_df)} suitable players")
    
    def create_player_card(self, player_row, rank, actual_columns):
        """Create a single player card widget"""
        card = QFrame()
        card.setFrameShape(QFrame.Shape.StyledPanel)
        card.setStyleSheet("""
            QFrame {
                background-color: #2b2b2b;
                border: 2px solid #444;
                border-radius: 8px;
                padding: 15px;
            }
            QFrame:hover {
                border-color: #0066cc;
                background-color: #323232;
            }
        """)
        
        layout = QHBoxLayout(card)
        layout.setSpacing(15)
        
        # Rank badge
        rank_label = QLabel(str(rank))
        rank_label.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        rank_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        rank_label.setFixedSize(40, 40)
        if rank <= 3:
            rank_label.setStyleSheet("background-color: #ffd700; color: #000; border-radius: 20px; font-weight: bold;")
        elif rank <= 10:
            rank_label.setStyleSheet("background-color: #c0c0c0; color: #000; border-radius: 20px; font-weight: bold;")
        else:
            rank_label.setStyleSheet("background-color: #555; color: #fff; border-radius: 20px;")
        layout.addWidget(rank_label)
        
        # Player image (rectangular thumbnail)
        image_label = QLabel()
        # No size constraints - let the pixmap determine size
        image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        image_label.setStyleSheet("""
            QLabel {
                border: 2px solid #666;
                border-radius: 5px;
                background-color: #444;
                min-width: 50px;
                min-height: 50px;
                max-width: 70px;
                max-height: 70px;
            }
        """)
        
        # Load player image if available
        image_loader = ImageLoader()
        if 'player_image_url' in player_row and pd.notna(player_row['player_image_url']):
            def set_card_image(pixmap):
                if pixmap and not pixmap.isNull():
                    # Scale to fit the label exactly (60x60)
                    scaled = pixmap.scaled(60, 60, 
                                          Qt.AspectRatioMode.KeepAspectRatio,
                                          Qt.TransformationMode.SmoothTransformation)
                    
                    # Ensure the pixmap fits properly
                    image_label.setPixmap(scaled)
                    image_label.adjustSize()  # Adjust label to pixmap size
                else:
                    image_label.setText("👤")
                    image_label.setStyleSheet("""
                        QLabel {
                            border: 2px solid #666;
                            border-radius: 5px;
                            background-color: #444;
                            font-size: 24px;
                        }
                    """)
            
            image_loader.load_image(player_row['player_image_url'], set_card_image, size=(60, 60))
        else:
            # Placeholder
            image_label.setText("👤")
            image_label.setStyleSheet("""
                QLabel {
                    border: 2px solid #666;
                    border-radius: 30px;
                    background-color: #444;
                    font-size: 24px;
                }
            """)
        
        layout.addWidget(image_label)
        
        # Player info section
        info_layout = QVBoxLayout()
        info_layout.setSpacing(5)
        
        # Player name (clickable)
        player_name = player_row.get('Player', 'Unknown')
        name_btn = QPushButton(f"👤 {player_name}")
        name_btn.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        name_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        name_btn.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                border: none;
                color: #4da6ff;
                text-align: left;
                padding: 0px;
                text-decoration: underline;
            }
            QPushButton:hover {
                color: #66b3ff;
            }
        """)
        name_btn.clicked.connect(lambda checked, r=player_row: self.show_player_profile(r))
        info_layout.addWidget(name_btn)
        
        # Basic details line
        details = []
        if 'Pos' in player_row and pd.notna(player_row['Pos']):
            details.append(f"⚽ {player_row['Pos']}")
        if 'Squad' in player_row and pd.notna(player_row['Squad']):
            squad = str(player_row['Squad'])
            if len(squad) > 20:
                squad = squad[:17] + "..."
            details.append(f"🏆 {squad}")
        if 'Age' in player_row and pd.notna(player_row['Age']):
            details.append(f"📅 {player_row['Age']}")

        if details:
            details_label = QLabel(" | ".join(details))
            details_label.setFont(QFont("Arial", 10))
            details_label.setStyleSheet("color: #bbb;")
            info_layout.addWidget(details_label)

        # Archetype tag badge
        arch_val = player_row.get('Archetype', '') if hasattr(player_row, 'get') else ''
        if arch_val and str(arch_val) not in ('—', '', 'nan'):
            arch_label = QLabel(f"🏷️ {arch_val}")
            arch_label.setFont(QFont("Arial", 9))
            arch_label.setStyleSheet(
                "color: #ffd700; background-color: #2a2a3e; "
                "padding: 2px 6px; border-radius: 3px;"
            )
            info_layout.addWidget(arch_label)
        
        # Additional info line (foot, contract, value)
        extra_details = []
        
        # Foot
        foot_col = actual_columns.get('foot')
        if foot_col and pd.notna(player_row.get(foot_col)):
            extra_details.append(f"👟 {str(player_row[foot_col]).capitalize()}")
        
        # Contract
        contract_col = actual_columns.get('contract_expires')
        if contract_col and pd.notna(player_row.get(contract_col)):
            contract = str(player_row[contract_col])
            if '30/06/2026' in contract:
                extra_details.append(f"📝 {contract} ⭐ FREE AGENT")
            else:
                extra_details.append(f"📝 {contract}")
        
        # Value
        value_col = actual_columns.get('market_value')
        if value_col and pd.notna(player_row.get(value_col)):
            try:
                val_num = float(player_row[value_col])
                if val_num >= 1_000_000:
                    value_str = f"€{val_num/1_000_000:.1f}M"
                elif val_num >= 1_000:
                    value_str = f"€{val_num/1_000:.0f}K"
                else:
                    value_str = f"€{val_num:.0f}"
                extra_details.append(f"💰 {value_str}")
            except:
                pass
        
        if extra_details:
            extra_label = QLabel(" | ".join(extra_details))
            extra_label.setFont(QFont("Arial", 9))
            extra_label.setStyleSheet("color: #999;")
            info_layout.addWidget(extra_label)
        
        layout.addLayout(info_layout, stretch=3)
        
        # Score section
        score_layout = QVBoxLayout()
        score_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        if 'Score' in player_row and pd.notna(player_row['Score']):
            try:
                score = float(player_row['Score'])
                score_pct = int(score * 100)
                
                score_label = QLabel(f"{score_pct}%")
                score_label.setFont(QFont("Arial", 20, QFont.Weight.Bold))
                score_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                
                if score > 0.7:
                    score_label.setStyleSheet("color: #28a745;")
                    rating = "⭐⭐⭐"
                elif score > 0.5:
                    score_label.setStyleSheet("color: #ffc107;")
                    rating = "⭐⭐"
                else:
                    score_label.setStyleSheet("color: #6c757d;")
                    rating = "⭐"
                
                score_layout.addWidget(score_label)
                rating_label = QLabel(rating)
                rating_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                score_layout.addWidget(rating_label)
            except:
                score_layout.addWidget(QLabel("N/A"))
        
        layout.addLayout(score_layout, stretch=1)
        
        # Addresses (abbreviated)
        if 'Addresses' in player_row and pd.notna(player_row['Addresses']):
            addresses_layout = QVBoxLayout()
            addresses_text = str(player_row['Addresses'])
            needs = addresses_text.split(',')[:2]  # Show first 2 needs
            
            for need in needs:
                need = need.strip()
                if need:
                    need_label = QLabel(f"✅ {need}")
                    need_label.setFont(QFont("Arial", 8))
                    need_label.setStyleSheet("color: #4da6ff;")
                    addresses_layout.addWidget(need_label)
            
            if len(addresses_text.split(',')) > 2:
                more_label = QLabel(f"+{len(addresses_text.split(',')) - 2} more...")
                more_label.setFont(QFont("Arial", 7))
                more_label.setStyleSheet("color: #666; font-style: italic;")
                addresses_layout.addWidget(more_label)
            
            layout.addLayout(addresses_layout, stretch=2)
        
        return card
    
    def show_player_profile(self, player_row):
        """Show detailed player profile dialog.

        Always uses evaluated_players as the KNN candidate pool because
        recommendations_df has its per-90 columns stripped by
        RecommendationEngine._add_recommendation_reasoning() before being
        returned — leaving no features for the similarity model to work with.
        evaluated_players is the direct output of PlayerEvaluator.evaluate_all_players()
        and retains every _p90 column.
        """
        dialog = PlayerProfileDialog(
            player_row, self, candidates_df=self.evaluated_players
        )
        dialog.exec()
    
    def on_recommendations_error(self, error_msg, progress):
        """Handle recommendations error"""
        progress.close()
        QMessageBox.critical(self, "Recommendation Error", error_msg)
        self.statusBar().showMessage("Recommendation generation failed")
    
    def export_recommendations(self):
        """Export recommendations to CSV"""
        if not hasattr(self, 'recommendations_df') or self.recommendations_df.empty:
            QMessageBox.warning(self, "Warning", "No recommendations to export.")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Recommendations",
            f"recommendations_{self.current_team.replace(' ', '_')}.csv",
            "CSV Files (*.csv);;All Files (*)"
        )
        
        if file_path:
            try:
                self.recommendations_df.to_csv(file_path, index=False)
                QMessageBox.information(self, "Success", f"Recommendations exported to:\n{file_path}")
                self.statusBar().showMessage(f"Exported to {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to export:\n{str(e)}")
    
    def export_full_report_pdf(self):
        """Export comprehensive analysis report to PDF"""
        if not hasattr(self, 'current_team') or self.current_team is None:
            QMessageBox.warning(self, "Warning", "No analysis to export.")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Full Report",
            f"analysis_report_{self.current_team.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.pdf",
            "PDF Files (*.pdf);;All Files (*)"
        )
        
        if not file_path:
            return
        
        try:
            # Show progress
            progress = QProgressDialog("Generating PDF report...", None, 0, 0, self)
            progress.setWindowTitle("Creating PDF")
            progress.setWindowModality(Qt.WindowModality.WindowModal)
            progress.show()
            
            # Create PDF document
            doc = SimpleDocTemplate(file_path, pagesize=letter,
                                   rightMargin=0.75*inch, leftMargin=0.75*inch,
                                   topMargin=1*inch, bottomMargin=0.75*inch)
            
            # Container for PDF elements
            story = []
            styles = getSampleStyleSheet()
            
            # Custom styles
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=24,
                textColor=colors.HexColor('#0066cc'),
                spaceAfter=30,
                alignment=TA_CENTER
            )
            
            heading_style = ParagraphStyle(
                'CustomHeading',
                parent=styles['Heading2'],
                fontSize=16,
                textColor=colors.HexColor('#0066cc'),
                spaceAfter=12,
                spaceBefore=12
            )
            
            # Title
            story.append(Paragraph(f"Football Analysis Report: {self.current_team}", title_style))
            story.append(Paragraph(f"Generated on {datetime.now().strftime('%B %d, %Y at %H:%M')}", styles['Normal']))
            story.append(Spacer(1, 0.3*inch))
            
            # === SECTION 1: OVERVIEW STATISTICS ===
            story.append(Paragraph("📊 Team Overview", heading_style))
            
            # Build comprehensive stats
            if hasattr(self, 'df_team') and self.df_team is not None and len(self.df_team) > 0:
                try:
                    # Get basic stats
                    basic_stats = get_team_statistics(self.df_team)
                    
                    # Calculate goals
                    total_gf = self.df_team['GF'].sum() if 'GF' in self.df_team.columns else 0
                    total_ga = self.df_team['GA'].sum() if 'GA' in self.df_team.columns else 0
                    avg_gf = self.df_team['GF'].mean() if 'GF' in self.df_team.columns else 0
                    avg_ga = self.df_team['GA'].mean() if 'GA' in self.df_team.columns else 0
                    goal_diff = total_gf - total_ga
                    
                    print(f"PDF Export - Overview Stats: Games={basic_stats.get('total_games', 0)}, Wins={basic_stats.get('wins', 0)}")
                    
                    overview_data = [
                        ['Metric', 'Value'],
                        ['Games Played', str(basic_stats.get('total_games', 0))],
                        ['Wins', f"{basic_stats.get('wins', 0)} ({basic_stats.get('win_rate', 0)*100:.1f}%)"],
                        ['Draws', str(basic_stats.get('draws', 0))],
                        ['Losses', str(basic_stats.get('losses', 0))],
                        ['Goals For', f"{int(total_gf)} (avg {avg_gf:.2f}/game)"],
                        ['Goals Against', f"{int(total_ga)} (avg {avg_ga:.2f}/game)"],
                        ['Goal Difference', f"{int(goal_diff):+d}"],
                    ]
                except Exception as e:
                    print(f"Error building overview stats: {e}")
                    import traceback
                    traceback.print_exc()
                    overview_data = [
                        ['Metric', 'Value'],
                        ['Status', 'Statistics not available'],
                    ]
            else:
                print(f"PDF Export - No team data: has_df_team={hasattr(self, 'df_team')}, df_team={self.df_team is not None if hasattr(self, 'df_team') else 'N/A'}")
                overview_data = [
                    ['Metric', 'Value'],
                    ['Status', 'No team data available'],
                ]
            
            # ALWAYS create and add the table
            overview_table = Table(overview_data, colWidths=[3*inch, 3.5*inch])
            overview_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0066cc')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('FONTSIZE', (0, 1), (-1, -1), 10),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
            ]))
            
            story.append(overview_table)
            story.append(Spacer(1, 0.3*inch))
            
            # === SECTION 2: HOME VS AWAY ===
            story.append(Paragraph("🏠 Home vs Away Performance", heading_style))
            
            if hasattr(self, 'df_team') and self.df_team is not None and len(self.df_team) > 0:
                try:
                    home_stats, away_stats = calculate_home_away_stats(self.df_team)
                    
                    print(f"PDF Export - Home/Away: Home games={home_stats.get('games', 0)}, Away games={away_stats.get('games', 0)}")
                    
                    homeaway_data = [
                        ['Metric', 'Home', 'Away'],
                        ['Games', str(home_stats.get('games', 0)), str(away_stats.get('games', 0))],
                        ['Wins', str(home_stats.get('wins', 0)), str(away_stats.get('wins', 0))],
                        ['Win Rate', f"{home_stats.get('win_rate', 0):.1f}%", f"{away_stats.get('win_rate', 0):.1f}%"],
                        ['Goals For/Game', f"{home_stats.get('goals_for', 0):.2f}", f"{away_stats.get('goals_for', 0):.2f}"],
                        ['Goals Against/Game', f"{home_stats.get('goals_against', 0):.2f}", f"{away_stats.get('goals_against', 0):.2f}"],
                    ]
                except Exception as e:
                    print(f"Error building home/away stats: {e}")
                    import traceback
                    traceback.print_exc()
                    homeaway_data = [
                        ['Metric', 'Home', 'Away'],
                        ['Status', 'N/A', 'N/A'],
                    ]
            else:
                homeaway_data = [
                    ['Metric', 'Home', 'Away'],
                    ['Status', 'No data', 'No data'],
                ]
            
            # ALWAYS create and add the table
            homeaway_table = Table(homeaway_data, colWidths=[2.5*inch, 2*inch, 2*inch])
            homeaway_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0066cc')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 11),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('FONTSIZE', (0, 1), (-1, -1), 10),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
            ]))
            
            story.append(homeaway_table)
            story.append(Spacer(1, 0.3*inch))
            
            # === SECTION 2.5: VISUAL CHARTS ===
            story.append(Paragraph("📊 Visual Analysis", heading_style))
            
            if hasattr(self, 'df_team') and self.df_team is not None and len(self.df_team) > 0:
                try:
                    # Create charts and save to BytesIO
                    from io import BytesIO
                    
                    # PIE CHART - Win/Draw/Loss distribution
                    basic_stats = get_team_statistics(self.df_team)
                    
                    fig1 = Figure(figsize=(3.5, 3.5), facecolor='white')
                    ax1 = fig1.add_subplot(111)
                    
                    sizes = [basic_stats['wins'], basic_stats['draws'], basic_stats['losses']]
                    labels = ['Wins', 'Draws', 'Losses']
                    colors_chart = ['#28a745', '#ffc107', '#dc3545']
                    explode = (0.05, 0, 0)
                    
                    ax1.pie(sizes, explode=explode, labels=labels, colors=colors_chart,
                           autopct='%1.1f%%', shadow=True, startangle=90)
                    ax1.set_title('Results Distribution', fontsize=12, weight='bold', pad=10)
                    
                    # Save pie chart to BytesIO
                    pie_buffer = BytesIO()
                    fig1.savefig(pie_buffer, format='png', dpi=150, bbox_inches='tight')
                    pie_buffer.seek(0)
                    
                    # Add pie chart to PDF
                    from reportlab.platypus import Image as RLImage
                    pie_img = RLImage(pie_buffer, width=3*inch, height=3*inch)
                    story.append(pie_img)
                    story.append(Spacer(1, 0.2*inch))
                    
                    # BAR CHART - Home vs Away
                    home_stats, away_stats = calculate_home_away_stats(self.df_team)
                    
                    fig2 = Figure(figsize=(6, 3.5), facecolor='white')
                    ax2 = fig2.add_subplot(111)
                    
                    categories = ['Win Rate %', 'Goals For', 'Goals Against']
                    x = np.arange(len(categories))
                    width = 0.35
                    
                    home_data = [home_stats['win_rate'], home_stats['goals_for'], home_stats['goals_against']]
                    away_data = [away_stats['win_rate'], away_stats['goals_for'], away_stats['goals_against']]
                    
                    bars1 = ax2.bar(x - width/2, home_data, width, label='Home', color='#0066cc', alpha=0.8)
                    bars2 = ax2.bar(x + width/2, away_data, width, label='Away', color='#ff6600', alpha=0.8)
                    
                    # Add value labels
                    for bars in [bars1, bars2]:
                        for bar in bars:
                            height = bar.get_height()
                            ax2.text(bar.get_x() + bar.get_width()/2., height,
                                   f'{height:.1f}', ha='center', va='bottom', fontsize=8)
                    
                    ax2.set_ylabel('Value', fontsize=10, weight='bold')
                    ax2.set_title('Home vs Away Comparison', fontsize=12, weight='bold', pad=10)
                    ax2.set_xticks(x)
                    ax2.set_xticklabels(categories, fontsize=9)
                    ax2.legend(fontsize=9)
                    ax2.grid(axis='y', alpha=0.3, linestyle='--')
                    
                    fig2.tight_layout()
                    
                    # Save bar chart to BytesIO
                    bar_buffer = BytesIO()
                    fig2.savefig(bar_buffer, format='png', dpi=150, bbox_inches='tight')
                    bar_buffer.seek(0)
                    
                    # Add bar chart to PDF
                    bar_img = RLImage(bar_buffer, width=5*inch, height=2.5*inch)
                    story.append(bar_img)
                    story.append(Spacer(1, 0.3*inch))
                    
                    print("PDF Export - Charts added successfully")
                    
                except Exception as e:
                    print(f"Error adding charts to PDF: {e}")
                    import traceback
                    traceback.print_exc()
                    story.append(Paragraph("Charts could not be generated", styles['Normal']))
                    story.append(Spacer(1, 0.2*inch))
            
            # === SECTION 2.6: BETTING ODDS CHART ===
            if hasattr(self, 'df_team') and self.df_team is not None and len(self.df_team) > 0:
                try:
                    odds_analysis = calculate_betting_odds_analysis(self.df_team)
                    
                    if odds_analysis['has_odds_data']:
                        story.append(Paragraph("🎲 Betting Odds Analysis", heading_style))
                        
                        fig3 = Figure(figsize=(6, 3.5), facecolor='white')
                        ax3 = fig3.add_subplot(111)
                        
                        categories = ['Wins', 'Draws', 'Losses']
                        x = np.arange(len(categories))
                        width = 0.35
                        
                        expected_data = [
                            odds_analysis['expected_wins'],
                            odds_analysis['expected_draws'],
                            odds_analysis['expected_losses']
                        ]
                        actual_data = [
                            odds_analysis['actual_wins'],
                            odds_analysis['actual_draws'],
                            odds_analysis['actual_losses']
                        ]
                        
                        bars1 = ax3.bar(x - width/2, expected_data, width, label='Expected (Odds)', 
                                       color='#6c757d', alpha=0.7)
                        bars2 = ax3.bar(x + width/2, actual_data, width, label='Actual', 
                                       color='#0066cc', alpha=0.8)
                        
                        # Add value labels - decimals for expected, integers for actual
                        for bar in bars1:
                            height = bar.get_height()
                            ax3.text(bar.get_x() + bar.get_width()/2., height,
                                   f'{height:.1f}', ha='center', va='bottom', fontsize=8)
                        
                        for bar in bars2:
                            height = bar.get_height()
                            ax3.text(bar.get_x() + bar.get_width()/2., height,
                                   f'{int(height)}', ha='center', va='bottom', fontsize=8, weight='bold')
                        
                        ax3.set_ylabel('Number of Games', fontsize=10, weight='bold')
                        ax3.set_title('Expected vs Actual Results (Betting Odds)', fontsize=11, weight='bold', pad=10)
                        ax3.set_xticks(x)
                        ax3.set_xticklabels(categories, fontsize=9)
                        ax3.legend(fontsize=9)
                        ax3.grid(axis='y', alpha=0.3, linestyle='--')
                        ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
                        
                        fig3.tight_layout()
                        
                        # Save odds chart to BytesIO
                        odds_buffer = BytesIO()
                        fig3.savefig(odds_buffer, format='png', dpi=150, bbox_inches='tight')
                        odds_buffer.seek(0)
                        
                        # Add odds chart to PDF
                        odds_img = RLImage(odds_buffer, width=5*inch, height=2.5*inch)
                        story.append(odds_img)
                        
                        # Add performance assessment
                        perf_text = f"<b>{odds_analysis['odds_performance']}</b><br/>"
                        perf_text += f"Wins: {odds_analysis['wins_diff']:+d} | "
                        perf_text += f"Draws: {odds_analysis['draws_diff']:+d} | "
                        perf_text += f"Losses: {odds_analysis['losses_diff']:+d}"
                        
                        story.append(Spacer(1, 0.1*inch))
                        story.append(Paragraph(perf_text, styles['Normal']))
                        story.append(Spacer(1, 0.3*inch))
                        
                        print("PDF Export - Betting odds chart added successfully")
                        
                except Exception as e:
                    print(f"Error adding betting odds chart to PDF: {e}")
                    import traceback
                    traceback.print_exc()
            
            # === SECTION 3: DETECTED ISSUES ===
            if hasattr(self, 'all_issues') and len(self.all_issues) > 0:
                story.append(Paragraph(f"⚠️ Detected Issues ({len(self.all_issues)})", heading_style))
                
                issues_data = [['#', 'Category', 'Severity', 'Description']]
                for issue in self.all_issues[:15]:  # Top 15 issues
                    issues_data.append([
                        str(issue['number']),
                        issue['category'].replace('_', ' ').title(),
                        issue['severity'].upper(),
                        issue['description'][:80] + ('...' if len(issue['description']) > 80 else '')
                    ])
                
                issues_table = Table(issues_data, colWidths=[0.4*inch, 1.5*inch, 0.8*inch, 3.8*inch])
                issues_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#dc3545')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
                    ('FONTSIZE', (0, 1), (-1, -1), 8),
                    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
                    ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ]))
                
                story.append(issues_table)
                story.append(Spacer(1, 0.2*inch))
            
            # New page for recommendations
            if len(self.all_issues) > 0:
                story.append(PageBreak())
            
            # === SECTION 4: TOP RECOMMENDATIONS ===
            if hasattr(self, 'recommendations_df') and not self.recommendations_df.empty:
                story.append(Paragraph(f"🎯 Top Player Recommendations ({len(self.recommendations_df)})", heading_style))
                
                # Top 20 recommendations
                top_recs = self.recommendations_df.head(20)
                
                # Check which optional columns are available
                has_foot = any(col in top_recs.columns for col in ['foot', 'Foot', 'preferred_foot'])
                has_contract = any(col in top_recs.columns for col in ['contract_expires', 'Contract', 'Contract Expires'])
                has_value = any(col in top_recs.columns for col in ['market_value', 'Market Value', 'Value'])
                
                # Get actual column names
                foot_col = next((col for col in ['foot', 'Foot', 'preferred_foot'] if col in top_recs.columns), None)
                contract_col = next((col for col in ['contract_expires', 'Contract', 'Contract Expires'] if col in top_recs.columns), None)
                value_col = next((col for col in ['market_value', 'Market Value', 'Value'] if col in top_recs.columns), None)
                
                # Build header dynamically
                header = ['#', 'Player', 'Pos', 'Team', 'Age']
                if has_foot:
                    header.append('Foot')
                if has_contract:
                    header.append('Contract')
                if has_value:
                    header.append('Value (â‚¬)')
                header.extend(['Score', 'Addresses'])
                
                recs_data = [header]
                
                for idx, (_, row) in enumerate(top_recs.iterrows(), 1):
                    addresses = row.get('Addresses', '')
                    if len(str(addresses)) > 50:
                        addresses = str(addresses)[:47] + '...'
                    
                    row_data = [
                        str(idx),
                        str(row.get('Player', '')),
                        str(row.get('Pos', '')),
                        str(row.get('Squad', ''))[:15],
                        str(row.get('Age', '')),
                    ]
                    
                    # Add optional columns
                    if has_foot:
                        foot_val = str(row.get(foot_col, '')).capitalize()
                        row_data.append(foot_val[:1])  # Just 'R' or 'L' to save space
                    
                    if has_contract:
                        contract_val = str(row.get(contract_col, ''))
                        # Shorten date format: 30/06/2026 -> 06/26
                        if '/' in contract_val:
                            parts = contract_val.split('/')
                            if len(parts) == 3:
                                contract_val = f"{parts[1]}/{parts[2][-2:]}"
                        row_data.append(contract_val)
                    
                    if has_value:
                        value_val = row.get(value_col, '')
                        try:
                            val_num = float(value_val)
                            if val_num >= 1_000_000:
                                value_val = f"â‚¬{val_num/1_000_000:.1f}M"
                            elif val_num >= 1_000:
                                value_val = f"â‚¬{val_num/1_000:.0f}K"
                            else:
                                value_val = f"â‚¬{val_num:.0f}"
                        except:
                            value_val = str(value_val)
                        row_data.append(value_val)
                    
                    row_data.extend([
                        f"{float(row.get('Score', 0)):.2f}",
                        str(addresses)
                    ])
                    
                    recs_data.append(row_data)
                
                # Adjust column widths based on what's included
                base_widths = [0.3*inch, 1.4*inch, 0.5*inch, 1.0*inch, 0.4*inch]
                optional_widths = []
                if has_foot:
                    optional_widths.append(0.3*inch)  # Foot
                if has_contract:
                    optional_widths.append(0.6*inch)  # Contract
                if has_value:
                    optional_widths.append(0.7*inch)  # Value
                
                col_widths = base_widths + optional_widths + [0.5*inch, 1.8*inch]  # Score + Addresses
                
                recs_table = Table(recs_data, colWidths=col_widths)
                recs_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#28a745')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 8),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
                    ('FONTSIZE', (0, 1), (-1, -1), 7),
                    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
                    ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ]))
                
                story.append(recs_table)
                story.append(Spacer(1, 0.2*inch))
            
            # === FOOTER ===
            story.append(Spacer(1, 0.5*inch))
            footer_text = f"Report generated by Football Analyzer | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            story.append(Paragraph(footer_text, styles['Normal']))
            
            # Build PDF
            doc.build(story)
            
            progress.close()
            
            QMessageBox.information(self, "Success", f"Full report exported to:\n{file_path}")
            self.statusBar().showMessage(f"PDF report exported to {file_path}")
            
        except Exception as e:
            if 'progress' in locals():
                progress.close()
            
            import traceback
            error_details = traceback.format_exc()
            print(f"PDF Export Error:\n{error_details}")  # Print to console for debugging
            
            QMessageBox.critical(self, "Error", f"Failed to generate PDF:\n{str(e)}\n\nCheck console for details.")

    
    def reset_app(self):
        """Reset application state"""
        reply = QMessageBox.question(
            self,
            "Reset Application",
            "Are you sure you want to reset? All data and analysis will be cleared.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            # Clear all data
            self.matches_df = None
            self.players_df = None
            self.df_team = None
            self.patterns = None
            self.evaluated_players = None
            self.current_team = None
            self.all_issues = []
            
            # Reset UI
            self.matches_file_label.setText("No matches file loaded")
            self.players_file_label.setText("No players file loaded")
            self.team_combo.clear()
            self.team_combo.setEnabled(False)
            self.btn_analyze.setEnabled(False)
            self.stats_text.clear()
            self.issues_list.clear()
            self.recs_table.setRowCount(0)
            self.btn_recommend.setEnabled(False)
            self.btn_export_csv.setEnabled(False)
            self.btn_export_pdf.setEnabled(False)
            self.recs_info_label.setText("Generate recommendations to see results...")
            
            self.statusBar().showMessage("Application reset. Please load data files to begin.")


# ===================== Main Entry Point =====================
def main():
    app = QApplication(sys.argv)
    

    # Set application style
    app.setStyle('Fusion')
    
    # Create and show main window
    window = FootballAnalyzerApp()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()