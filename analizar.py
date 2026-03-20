"""
analyzer.py
===========
Enhanced Football Performance Analyzer
- Comprehensive pattern recognition across all game aspects
- Multi-dimensional player evaluation system
- Context-aware recommendations based on detected weaknesses
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
import re
import unicodedata

warnings.filterwarnings('ignore')


# ---------------------- Data Loading ---------------------- #

def load_matches_data(file_obj):
    """Load match data from file object with automatic delimiter detection"""
    try:
        if file_obj.name.endswith('.xlsx'):
            matches = pd.read_excel(file_obj)
        else:
            # Try with automatic delimiter detection using python engine
            file_obj.seek(0)
            try:
                # Let pandas detect the delimiter automatically
                matches = pd.read_csv(file_obj, sep=None, engine='python', low_memory=False)
            except Exception:
                # If that fails, manually try different delimiters
                file_obj.seek(0)
                first_line = file_obj.read(1000)
                file_obj.seek(0)
                
                # Decode if bytes
                if isinstance(first_line, bytes):
                    first_line = first_line.decode('utf-8', errors='ignore')
                
                # Count delimiters
                comma_count = first_line.count(',')
                semicolon_count = first_line.count(';')
                tab_count = first_line.count('\t')
                
                # Choose delimiter with highest count
                if semicolon_count > comma_count and semicolon_count > tab_count:
                    sep = ';'
                elif tab_count > comma_count:
                    sep = '\t'
                else:
                    sep = ','
                
                file_obj.seek(0)
                matches = pd.read_csv(file_obj, sep=sep, low_memory=False)
        
        # Validate that it loaded properly
        if len(matches.columns) < 3:
            raise Exception(f"CSV appears to be incorrectly parsed. Only {len(matches.columns)} columns detected.")
        
        return matches
        
    except Exception as e:
        raise Exception(f"Error loading matches file: {e}")


def load_players_data(file_obj):
    """Load players data from file object with multiple fallback attempts"""
    players = None
    
    if file_obj.name.endswith('.xlsx'):
        try:
            players = pd.read_excel(file_obj)
            return players, f"Successfully loaded players data from Excel file"
        except Exception as e:
            raise Exception(f"Error loading players Excel file: {e}")
    
    # Try various CSV reading methods
    attempts = [
        {"sep": ",", "encoding": "utf-8", "on_bad_lines": "skip"},
        {"sep": ";", "encoding": "utf-8", "on_bad_lines": "skip"},
        {"sep": "\t", "encoding": "utf-8", "on_bad_lines": "skip"},
        {"sep": ",", "encoding": "latin-1", "on_bad_lines": "skip"},
        {"sep": ";", "encoding": "latin-1", "on_bad_lines": "skip"},
        {"sep": None, "encoding": "utf-8", "engine": "python", "on_bad_lines": "skip"},
    ]
    
    messages = []
    for i, params in enumerate(attempts, 1):
        try:
            file_obj.seek(0)
            messages.append(f"Attempting to read players CSV (method {i}/{len(attempts)})...")
            try:
                players = pd.read_csv(file_obj, low_memory=False, **params)
            except TypeError:
                # Fallback for older pandas versions
                params_old = params.copy()
                if "on_bad_lines" in params_old:
                    params_old.pop("on_bad_lines")
                    params_old["error_bad_lines"] = False
                file_obj.seek(0)
                players = pd.read_csv(file_obj, low_memory=False, **params_old)
            
            if len(players.columns) > 5 and len(players) > 10:
                messages.append(f"Successfully loaded players data with {len(players)} rows and {len(players.columns)} columns")
                return players, "\n".join(messages)
            else:
                messages.append(f"Attempt {i} resulted in too few columns or rows, trying next method...")
                players = None
        except Exception as e:
            messages.append(f"Attempt {i} failed: {type(e).__name__}: {str(e)[:100]}")
            players = None
            continue
    
    if players is None:
        raise Exception("Could not load players data file. Please ensure it's a valid CSV or Excel file.")
    
    return players, "\n".join(messages)


def clean_data(matches, players):
    """Clean up column names and handle missing values"""
    # Clean column names
    if players is not None and not players.empty:
        players.columns = players.columns.str.strip()
        # Normalise market-value column: rename lowercase 'value' → 'market_value'
        # so all downstream lookups that check for 'market_value' find it.
        if 'value' in players.columns and 'market_value' not in players.columns:
            players = players.rename(columns={'value': 'market_value'})
    if matches is not None and not matches.empty:
        matches.columns = matches.columns.str.strip()
    
    # Handle missing values in numeric columns for matches
    numeric_cols_matches = ['HomeElo', 'AwayElo', 'Form3Home', 'Form5Home', 'Form3Away', 'Form5Away',
                            'FTHome', 'FTAway', 'HTHome', 'HTAway', 'HomeShots', 'AwayShots',
                            'HomeTarget', 'AwayTarget', 'HomeFouls', 'AwayFouls', 'HomeCorners',
                            'AwayCorners', 'HomeYellow', 'AwayYellow', 'HomeRed', 'AwayRed']
    
    for col in numeric_cols_matches:
        if col in matches.columns:
            matches[col] = pd.to_numeric(matches[col], errors='coerce')
    
    return matches, players


# ---------------------- Team Table Construction ---------------------- #

def compute_team_table(matches: pd.DataFrame, team: str) -> pd.DataFrame:
    """Build comprehensive team-perspective match table with advanced metrics"""
    df = matches[(matches["HomeTeam"] == team) | (matches["AwayTeam"] == team)].copy()
    if df.empty:
        return df

    df["is_home"] = (df["HomeTeam"] == team)
    df["opponent"] = np.where(df["is_home"], df["AwayTeam"], df["HomeTeam"])
    
    # Basic metrics from team perspective - handle missing values
    df["GF"] = pd.to_numeric(np.where(df["is_home"], df["FTHome"], df["FTAway"]), errors='coerce')
    df["GA"] = pd.to_numeric(np.where(df["is_home"], df["FTAway"], df["FTHome"]), errors='coerce')
    
    # Half-time goals - handle missing values
    if "HTHome" in df.columns and "HTAway" in df.columns:
        df["HTGoalsFor"] = pd.to_numeric(np.where(df["is_home"], df["HTHome"], df["HTAway"]), errors='coerce')
        df["HTGoalsAgainst"] = pd.to_numeric(np.where(df["is_home"], df["HTAway"], df["HTHome"]), errors='coerce')
    
    # Shot metrics - handle missing values
    if "HomeShots" in df.columns and "AwayShots" in df.columns:
        df["ShotsFor"] = pd.to_numeric(np.where(df["is_home"], df["HomeShots"], df["AwayShots"]), errors='coerce')
        df["ShotsAgainst"] = pd.to_numeric(np.where(df["is_home"], df["AwayShots"], df["HomeShots"]), errors='coerce')
    
    if "HomeTarget" in df.columns and "AwayTarget" in df.columns:
        df["ShotsOnTargetFor"] = pd.to_numeric(np.where(df["is_home"], df["HomeTarget"], df["AwayTarget"]), errors='coerce')
        df["ShotsOnTargetAgainst"] = pd.to_numeric(np.where(df["is_home"], df["AwayTarget"], df["HomeTarget"]), errors='coerce')
    
    # Discipline and set pieces - handle missing values
    if "HomeFouls" in df.columns and "AwayFouls" in df.columns:
        df["FoulsFor"] = pd.to_numeric(np.where(df["is_home"], df["HomeFouls"], df["AwayFouls"]), errors='coerce')
        df["FoulsAgainst"] = pd.to_numeric(np.where(df["is_home"], df["AwayFouls"], df["HomeFouls"]), errors='coerce')
    
    if "HomeCorners" in df.columns and "AwayCorners" in df.columns:
        df["CornersFor"] = pd.to_numeric(np.where(df["is_home"], df["HomeCorners"], df["AwayCorners"]), errors='coerce')
        df["CornersAgainst"] = pd.to_numeric(np.where(df["is_home"], df["AwayCorners"], df["HomeCorners"]), errors='coerce')
    
    if "HomeYellow" in df.columns and "AwayYellow" in df.columns:
        df["YellowCards"] = pd.to_numeric(np.where(df["is_home"], df["HomeYellow"], df["AwayYellow"]), errors='coerce')
    
    if "HomeRed" in df.columns and "AwayRed" in df.columns:
        df["RedCards"] = pd.to_numeric(np.where(df["is_home"], df["HomeRed"], df["AwayRed"]), errors='coerce')
    
    # Result mapping
    def map_result(row):
        r = str(row.get("FTResult", "")).upper()
        if r == "H":
            return "W" if row["is_home"] else "L"
        elif r == "A":
            return "L" if row["is_home"] else "W"
        elif r == "D":
            return "D"
        # Fallback to goals
        gf, ga = row.get("GF", np.nan), row.get("GA", np.nan)
        if pd.notna(gf) and pd.notna(ga):
            return "W" if gf > ga else ("L" if gf < ga else "D")
        return "D"
    
    df["Result"] = df.apply(map_result, axis=1)
    
    # ELO features - handle missing values
    if {"HomeElo", "AwayElo"}.issubset(df.columns):
        df["TeamElo"] = pd.to_numeric(np.where(df["is_home"], df["HomeElo"], df["AwayElo"]), errors='coerce')
        df["OppElo"] = pd.to_numeric(np.where(df["is_home"], df["AwayElo"], df["HomeElo"]), errors='coerce')
        df["EloDiff"] = df["OppElo"] - df["TeamElo"]
    
    # Form features - handle missing values
    if {"Form3Home", "Form3Away", "Form5Home", "Form5Away"}.issubset(df.columns):
        df["TeamForm3"] = pd.to_numeric(np.where(df["is_home"], df["Form3Home"], df["Form3Away"]), errors='coerce')
        df["TeamForm5"] = pd.to_numeric(np.where(df["is_home"], df["Form5Home"], df["Form5Away"]), errors='coerce')
        df["OppForm3"] = pd.to_numeric(np.where(df["is_home"], df["Form3Away"], df["Form3Home"]), errors='coerce')
        df["OppForm5"] = pd.to_numeric(np.where(df["is_home"], df["Form5Away"], df["Form5Home"]), errors='coerce')
    
    return df


# ---------------------- Advanced Pattern Recognition ---------------------- #

class PatternAnalyzer:
    """Comprehensive pattern recognition for football matches"""
    
    def __init__(self, df_team: pd.DataFrame, matches_full: pd.DataFrame):
        self.df = df_team
        self.matches_full = matches_full
        self.losses = df_team[df_team["Result"] == "L"]
        self.wins = df_team[df_team["Result"] == "W"]
        self.draws = df_team[df_team["Result"] == "D"]
        
    def analyze_all_patterns(self) -> Dict:
        """Run comprehensive analysis across all game aspects"""
        patterns = {
            "defensive": self._analyze_defensive_patterns(),
            "offensive": self._analyze_offensive_patterns(),
            "tactical": self._analyze_tactical_patterns(),
            "physical": self._analyze_physical_patterns(),
            "mental": self._analyze_mental_patterns(),
            "set_pieces": self._analyze_set_piece_patterns(),
            "game_management": self._analyze_game_management(),
            "opponent_specific": self._analyze_opponent_patterns()
        }
        return patterns
    
    def _analyze_defensive_patterns(self) -> Dict:
        analysis = {"issues": [], "metrics": {}}
        if len(self.losses) == 0:
            return analysis
        
        ga_losses = self.losses["GA"].mean()
        ga_overall = self.df["GA"].mean()
        analysis["metrics"]["ga_losses"] = ga_losses
        analysis["metrics"]["ga_overall"] = ga_overall
        
        if ga_losses > 2.5:
            analysis["issues"].append({
                "type": "high_line_vulnerability",
                "severity": "high",
                "description": f"Conceding {ga_losses:.1f} goals per loss - defensive line too high or poor recovery",
                "positions_needed": ["CB_pace", "DM_defensive", "FB_recovery"]
            })
        
        if "ShotsAgainst" in self.losses.columns and not self.losses["ShotsAgainst"].isna().all():
            shots_against_losses = self.losses["ShotsAgainst"].mean()
            if pd.notna(shots_against_losses) and shots_against_losses > 15:
                analysis["issues"].append({
                    "type": "pressure_failure",
                    "severity": "high",
                    "description": f"Allowing {shots_against_losses:.1f} shots in losses - lacking defensive pressure",
                    "positions_needed": ["DM_pressing", "CM_workrate", "FB_defensive"]
                })
            if "ShotsOnTargetAgainst" in self.losses.columns and not self.losses["ShotsOnTargetAgainst"].isna().all():
                shots_on_target_against = self.losses["ShotsOnTargetAgainst"].mean()
                if pd.notna(shots_on_target_against) and shots_on_target_against > 0:
                    conversion_rate = ga_losses / shots_on_target_against
                    if conversion_rate > 0.4:
                        analysis["issues"].append({
                            "type": "goalkeeper_issues",
                            "severity": "high",
                            "description": f"High shot conversion rate against ({conversion_rate:.1%}) - GK or last-ditch defending issues",
                            "positions_needed": ["GK", "CB_blocking"]
                        })
        
        home_losses = self.losses[self.losses["is_home"]]
        away_losses = self.losses[~self.losses["is_home"]]
        if len(home_losses) > 0 and len(away_losses) > 0:
            ga_home = home_losses["GA"].mean()
            ga_away = away_losses["GA"].mean()
            if pd.notna(ga_home) and pd.notna(ga_away) and ga_away > ga_home * 1.3:
                analysis["issues"].append({
                    "type": "away_defensive_fragility",
                    "severity": "medium",
                    "description": f"Defensive structure breaks down away ({ga_away:.1f} vs {ga_home:.1f} GA)",
                    "positions_needed": ["CB_leadership", "DM_experienced", "GK_commanding"]
                })
        
        if "HTGoalsAgainst" in self.losses.columns and not self.losses["HTGoalsAgainst"].isna().all():
            early_goals = self.losses["HTGoalsAgainst"].mean()
            if pd.notna(early_goals) and early_goals > 1.0:
                analysis["issues"].append({
                    "type": "slow_start",
                    "severity": "medium",
                    "description": f"Conceding {early_goals:.1f} first-half goals in losses - concentration issues",
                    "positions_needed": ["CB_concentration", "DM_tactical"]
                })
        
        return analysis
    
    def _analyze_offensive_patterns(self) -> Dict:
        analysis = {"issues": [], "metrics": {}}
        if len(self.losses) == 0:
            return analysis
        
        gf_losses = self.losses["GF"].mean()
        gf_overall = self.df["GF"].mean()
        analysis["metrics"]["gf_losses"] = gf_losses
        analysis["metrics"]["gf_overall"] = gf_overall
        
        if pd.notna(gf_losses) and gf_losses < 0.8:
            analysis["issues"].append({
                "type": "scoring_drought",
                "severity": "high",
                "description": f"Only {gf_losses:.1f} goals per loss - lacking clinical finishing",
                "positions_needed": ["ST_finishing", "W_goals", "AM_creativity"]
            })
        
        if "ShotsFor" in self.losses.columns and not self.losses["ShotsFor"].isna().all():
            shots_losses = self.losses["ShotsFor"].mean()
            if pd.notna(shots_losses) and shots_losses < 10:
                analysis["issues"].append({
                    "type": "chance_creation",
                    "severity": "high",
                    "description": f"Only {shots_losses:.1f} shots in losses - lacking creativity",
                    "positions_needed": ["AM_creative", "W_dribbling", "CM_passing"]
                })
            if "ShotsOnTargetFor" in self.losses.columns and not self.losses["ShotsOnTargetFor"].isna().all():
                shots_on_target = self.losses["ShotsOnTargetFor"].mean()
                if pd.notna(shots_losses) and pd.notna(shots_on_target) and shots_losses > 0:
                    accuracy = shots_on_target / shots_losses
                    if accuracy < 0.3:
                        analysis["issues"].append({
                            "type": "poor_finishing",
                            "severity": "medium",
                            "description": f"Shot accuracy only {accuracy:.1%} - need better finishers",
                            "positions_needed": ["ST_finishing", "W_shooting", "AM_longshots"]
                        })
        
        if "CornersFor" in self.losses.columns and not self.losses["CornersFor"].isna().all():
            corners_losses = self.losses["CornersFor"].mean()
            if pd.notna(corners_losses) and corners_losses < 3:
                analysis["issues"].append({
                    "type": "low_attacking_pressure",
                    "severity": "medium",
                    "description": f"Only {corners_losses:.1f} corners won - lacking attacking penetration",
                    "positions_needed": ["W_pace", "FB_attacking", "ST_holdup"]
                })
        
        return analysis
    
    def _analyze_tactical_patterns(self) -> Dict:
        analysis = {"issues": [], "metrics": {}}
        if len(self.losses) == 0:
            return analysis
        
        if ("ShotsFor" in self.losses.columns and "ShotsAgainst" in self.losses.columns and 
            not self.losses["ShotsFor"].isna().all() and not self.losses["ShotsAgainst"].isna().all()):
            shots_for = self.losses["ShotsFor"].mean()
            shots_against = self.losses["ShotsAgainst"].mean()
            if pd.notna(shots_for) and pd.notna(shots_against):
                possession_proxy = shots_for / (shots_for + shots_against) if (shots_for + shots_against) > 0 else 0.5
                if possession_proxy < 0.35:
                    analysis["issues"].append({
                        "type": "possession_loss",
                        "severity": "high",
                        "description": f"Dominated in possession (proxy: {possession_proxy:.1%}) - need ball retention",
                        "positions_needed": ["CM_passing", "DM_progressive", "CB_ballplaying"]
                    })
        
        if ("HTGoalsFor" in self.losses.columns and "HTGoalsAgainst" in self.losses.columns and
            not self.losses["HTGoalsFor"].isna().all() and not self.losses["HTGoalsAgainst"].isna().all()):
            ht_gf = self.losses["HTGoalsFor"].mean()
            ht_ga = self.losses["HTGoalsAgainst"].mean()
            if pd.notna(ht_gf) and pd.notna(ht_ga):
                second_half_gf = self.losses["GF"].mean() - ht_gf
                second_half_ga = self.losses["GA"].mean() - ht_ga
                if second_half_ga > second_half_gf * 1.5 and second_half_ga > 1:
                    analysis["issues"].append({
                        "type": "second_half_collapse",
                        "severity": "high",
                        "description": "Tactical structure breaks down in second half",
                        "positions_needed": ["CM_stamina", "DM_tactical", "CB_concentration"]
                    })
        
        if "EloDiff" in self.losses.columns and not self.losses["EloDiff"].isna().all():
            strong_opp_losses = self.losses[self.losses["EloDiff"] > 50]
            weak_opp_losses = self.losses[self.losses["EloDiff"] <= -50]
            if len(strong_opp_losses) > 2:
                ga_strong = strong_opp_losses["GA"].mean()
                if pd.notna(ga_strong) and ga_strong > 2.5:
                    analysis["issues"].append({
                        "type": "big_game_mentality",
                        "severity": "high",
                        "description": f"Struggle against stronger teams ({ga_strong:.1f} GA)",
                        "positions_needed": ["CB_experienced", "CM_bigmatch", "GK_commanding"]
                    })
            if len(weak_opp_losses) > 1:
                analysis["issues"].append({
                    "type": "complacency",
                    "severity": "medium",
                    "description": "Dropping points against weaker teams",
                    "positions_needed": ["CM_leadership", "ST_consistent", "DM_workrate"]
                })

        # High-Scoring Game Vulnerability: losses in games likely to produce 2+ goals (Over25 < 1.75)
        if "Over25" in self.losses.columns and not self.losses["Over25"].isna().all():
            high_scoring_losses = self.losses[self.losses["Over25"] < 1.75]
            if len(high_scoring_losses) > len(self.losses) * 0.35:
                avg_ga = high_scoring_losses["GA"].mean()
                avg_gf = high_scoring_losses["GF"].mean()
                if pd.notna(avg_ga) and pd.notna(avg_gf):
                    if avg_ga > avg_gf * 1.5:
                        analysis["issues"].append({
                            "type": "high_scoring_defensive_vulnerability",
                            "severity": "high",
                            "description": (
                                f"Defensively exposed in high-scoring games "
                                f"(avg GA: {avg_ga:.1f}, GF: {avg_gf:.1f})"
                            ),
                            "positions_needed": ["DM_defensive", "CB_pace", "GK_shotstopper"]
                        })
                    elif avg_gf < 1.0:
                        analysis["issues"].append({
                            "type": "high_scoring_offensive_absence",
                            "severity": "medium",
                            "description": (
                                f"Failing to contribute in high-scoring games "
                                f"(avg GF: {avg_gf:.1f}) — poor attacking output"
                            ),
                            "positions_needed": ["ST_pressing", "W_pace", "CM_workrate"]
                        })

        return analysis
    
    def _analyze_physical_patterns(self) -> Dict:
        analysis = {"issues": [], "metrics": {}}
        if len(self.losses) == 0:
            return analysis
        
        if ("FoulsFor" in self.losses.columns and "FoulsAgainst" in self.losses.columns and
            not self.losses["FoulsFor"].isna().all() and not self.losses["FoulsAgainst"].isna().all()):
            fouls_for = self.losses["FoulsFor"].mean()
            fouls_against = self.losses["FoulsAgainst"].mean()
            if pd.notna(fouls_for) and pd.notna(fouls_against) and fouls_against > fouls_for * 1.3:
                analysis["issues"].append({
                    "type": "physical_dominance_lost",
                    "severity": "medium",
                    "description": f"Being outmuscled ({fouls_against:.1f} fouls suffered vs {fouls_for:.1f} committed)",
                    "positions_needed": ["CB_physical", "DM_strength", "ST_holdup"]
                })
        
        if "YellowCards" in self.losses.columns and not self.losses["YellowCards"].isna().all():
            yellows = self.losses["YellowCards"].mean()
            if pd.notna(yellows) and yellows > 2.5:
                analysis["issues"].append({
                    "type": "discipline_problems",
                    "severity": "medium",
                    "description": f"Averaging {yellows:.1f} yellows in losses - discipline issues",
                    "positions_needed": ["DM_disciplined", "CB_calm", "FB_tactical"]
                })
        
        if "RedCards" in self.losses.columns and not self.losses["RedCards"].isna().all():
            red_games = self.losses[self.losses["RedCards"] > 0]
            if len(red_games) > len(self.losses) * 0.15:
                analysis["issues"].append({
                    "type": "red_card_issues",
                    "severity": "high",
                    "description": "Frequent red cards disrupting game plan",
                    "positions_needed": ["CB_disciplined", "DM_calm", "FB_composed"]
                })

        # Fixture Congestion Fatigue: losses cluster in periods with 3+ games in 10 days
        if "MatchDate" in self.df.columns:
            try:
                df_dated = self.df.copy()
                df_dated["_date"] = pd.to_datetime(df_dated["MatchDate"], errors="coerce")
                df_dated = df_dated.dropna(subset=["_date"]).sort_values("_date")
                dates_list = df_dated["_date"].tolist()

                # Mark dates that fall within a congested window (3+ games in 10 days)
                congested_dates = set()
                for i, d in enumerate(dates_list):
                    window = [x for x in dates_list if abs((x - d).days) <= 10]
                    if len(window) >= 3:
                        congested_dates.add(d)

                if congested_dates and "MatchDate" in self.losses.columns:
                    losses_dated = self.losses.copy()
                    losses_dated["_date"] = pd.to_datetime(losses_dated["MatchDate"], errors="coerce")
                    congested_losses = losses_dated[losses_dated["_date"].isin(congested_dates)]
                    congested_loss_rate = len(congested_losses) / len(self.losses)
                    if congested_loss_rate > 0.40 and len(congested_losses) >= 3:
                        severity = "high" if congested_loss_rate > 0.60 else "medium"
                        analysis["issues"].append({
                            "type": "fixture_congestion_fatigue",
                            "severity": severity,
                            "description": (
                                f"{congested_loss_rate:.0%} of losses occur during congested fixture periods "
                                f"(3+ games in 10 days) — squad depth and fitness issue"
                            ),
                            "positions_needed": ["CM_stamina", "FB_fitness", "ST_pressing"]
                        })
            except Exception:
                pass

        return analysis
    
    def _analyze_mental_patterns(self) -> Dict:
        analysis = {"issues": [], "metrics": {}}
        if len(self.losses) == 0:
            return analysis
        
        if "HTGoalsFor" in self.losses.columns and "HTGoalsAgainst" in self.losses.columns:
            losing_at_ht = self.losses[self.losses["HTGoalsFor"] < self.losses["HTGoalsAgainst"]]
            if len(losing_at_ht) > len(self.losses) * 0.6:
                analysis["issues"].append({
                    "type": "poor_comeback_ability",
                    "severity": "high",
                    "description": "Rarely recover from losing positions",
                    "positions_needed": ["ST_impactful", "CM_leadership", "W_gamechanging"]
                })
        
        if "TeamForm3" in self.losses.columns:
            poor_form_losses = self.losses[self.losses["TeamForm3"] <= 3]
            if len(poor_form_losses) > len(self.losses) * 0.7:
                analysis["issues"].append({
                    "type": "form_dependency",
                    "severity": "medium",
                    "description": "Team confidence collapses during poor runs",
                    "positions_needed": ["CM_experienced", "CB_leadership", "GK_presence"]
                })
        
        if "OddHome" in self.losses.columns and "OddAway" in self.losses.columns:
            home_favored_losses = self.losses[(self.losses["is_home"]) & (self.losses["OddHome"] < 2.0)]
            away_favored_losses = self.losses[(~self.losses["is_home"]) & (self.losses["OddAway"] < 2.0)]
            if len(home_favored_losses) + len(away_favored_losses) > 2:
                analysis["issues"].append({
                    "type": "pressure_handling",
                    "severity": "medium",
                    "description": "Struggle when expected to win",
                    "positions_needed": ["CM_composed", "ST_clinical", "CB_reliable"]
                })

        # Long-Form Decay: Form5 in losses far below overall Form5 → confidence spiral
        if "TeamForm5" in self.losses.columns and not self.losses["TeamForm5"].isna().all():
            loss_form5 = self.losses["TeamForm5"].mean()
            all_form5 = self.df["TeamForm5"].mean() if "TeamForm5" in self.df.columns else None
            if pd.notna(loss_form5) and pd.notna(all_form5) and all_form5 > 0:
                decay_ratio = loss_form5 / all_form5
                if decay_ratio < 0.70:
                    severity = "high" if decay_ratio < 0.55 else "medium"
                    analysis["issues"].append({
                        "type": "long_form_decay",
                        "severity": severity,
                        "description": (
                            f"Form5 drops to {loss_form5:.1f} in losses vs {all_form5:.1f} overall "
                            f"— team enters losses already in a confidence spiral"
                        ),
                        "positions_needed": ["CM_experienced", "CB_leadership", "GK_presence"]
                    })

        # Heavy Favorite Collapse: losing as a very heavy favourite (odds < 1.45)
        if "OddHome" in self.losses.columns and "OddAway" in self.losses.columns:
            heavy_fav_losses = self.losses[
                ((self.losses["is_home"]) & (self.losses["OddHome"] < 1.45)) |
                ((~self.losses["is_home"]) & (self.losses["OddAway"] < 1.45))
            ]
            if len(heavy_fav_losses) >= 2:
                analysis["issues"].append({
                    "type": "heavy_favorite_collapse",
                    "severity": "high",
                    "description": (
                        f"Lost {len(heavy_fav_losses)} game(s) as heavy favourites (odds < 1.45) "
                        f"— severe mentality issue under expectation"
                    ),
                    "positions_needed": ["CM_composed", "ST_clinical", "CB_reliable", "DM_calm"]
                })

        return analysis
    
    def _analyze_set_piece_patterns(self) -> Dict:
        analysis = {"issues": [], "metrics": {}}
        if len(self.losses) == 0:
            return analysis
        
        if "CornersAgainst" in self.losses.columns and not self.losses["CornersAgainst"].isna().all():
            corners_against = self.losses["CornersAgainst"].mean()
            ga = self.losses["GA"].mean()
            if pd.notna(corners_against) and corners_against > 0 and pd.notna(ga) and ga / corners_against > 0.2:
                analysis["issues"].append({
                    "type": "set_piece_defending",
                    "severity": "high",
                    "description": "Vulnerable to set pieces defensively",
                    "positions_needed": ["CB_aerial", "GK_commanding", "DM_height"]
                })
        
        if "CornersFor" in self.losses.columns and not self.losses["CornersFor"].isna().all():
            corners_for = self.losses["CornersFor"].mean()
            gf = self.losses["GF"].mean()
            if pd.notna(corners_for) and corners_for > 4 and pd.notna(gf) and gf < 1:
                analysis["issues"].append({
                    "type": "set_piece_inefficiency",
                    "severity": "medium",
                    "description": "Not converting set piece opportunities",
                    "positions_needed": ["CB_setpiece", "ST_aerial", "CM_delivery"]
                })
        return analysis
    
    def _analyze_game_management(self) -> Dict:
        analysis = {"issues": [], "metrics": {}}
        if len(self.losses) == 0:
            return analysis
        
        if "GA" in self.losses.columns and "HTGoalsAgainst" in self.losses.columns:
            late_goals = self.losses["GA"].mean() - self.losses["HTGoalsAgainst"].mean()
            if late_goals > 1.2:
                analysis["issues"].append({
                    "type": "late_collapse",
                    "severity": "high",
                    "description": f"Conceding {late_goals:.1f} goals in second half of losses",
                    "positions_needed": ["CM_stamina", "FB_fitness", "DM_defensive"]
                })
        
        narrow_losses = self.losses[(self.losses["GA"] - self.losses["GF"]) == 1]
        if len(narrow_losses) > len(self.losses) * 0.5:
            analysis["issues"].append({
                "type": "game_management",
                "severity": "medium",
                "description": "Many narrow losses - poor game management",
                "positions_needed": ["DM_tactical", "CM_experienced", "CB_leadership"]
            })

        # Half-Time Lead Squander: leading at HT but conceding the game
        if ("HTGoalsFor" in self.losses.columns and "HTGoalsAgainst" in self.losses.columns and
                not self.losses["HTGoalsFor"].isna().all() and not self.losses["HTGoalsAgainst"].isna().all()):
            ht_leads_lost = self.losses[self.losses["HTGoalsFor"] > self.losses["HTGoalsAgainst"]]
            ht_lead_rate = len(ht_leads_lost) / len(self.losses)
            if ht_lead_rate > 0.20:
                severity = "high" if ht_lead_rate > 0.25 else "medium"
                analysis["issues"].append({
                    "type": "ht_lead_squander",
                    "severity": severity,
                    "description": f"Threw away half-time lead in {ht_lead_rate:.0%} of losses — second-half collapse",
                    "positions_needed": ["DM_tactical", "CB_concentration", "CM_stamina", "FB_fitness"]
                })

        return analysis
    
    def _analyze_opponent_patterns(self) -> Dict:
        analysis = {"issues": [], "metrics": {}}
        if len(self.losses) == 0:
            return analysis
        
        if "OppElo" in self.losses.columns and not self.losses["OppElo"].isna().all():
            losses_with_elo = self.losses[self.losses["OppElo"].notna()]
            if len(losses_with_elo) > 0:
                try:
                    q33, q66 = losses_with_elo["OppElo"].quantile([0.33, 0.66])
                    weak_opp = losses_with_elo[losses_with_elo["OppElo"] <= q33]
                    mid_opp = losses_with_elo[(losses_with_elo["OppElo"] > q33) & (losses_with_elo["OppElo"] <= q66)]
                    strong_opp = losses_with_elo[losses_with_elo["OppElo"] > q66]
                    for opp_df, label in [(weak_opp, "weak"), (mid_opp, "mid"), (strong_opp, "strong")]:
                        if len(opp_df) > 1:
                            ga_avg = opp_df["GA"].mean()
                            gf_avg = opp_df["GF"].mean()
                            if label == "weak" and len(opp_df) > len(losses_with_elo) * 0.2:
                                analysis["issues"].append({
                                    "type": f"struggle_vs_{label}",
                                    "severity": "high",
                                    "description": f"Losing to weaker teams (GA: {ga_avg:.1f}, GF: {gf_avg:.1f})",
                                    "positions_needed": ["CM_consistent", "ST_clinical", "DM_workrate"]
                                })
                            elif label == "strong" and ga_avg > 3:
                                analysis["issues"].append({
                                    "type": f"struggle_vs_{label}",
                                    "severity": "medium",
                                    "description": f"Heavy defeats vs strong teams (GA: {ga_avg:.1f})",
                                    "positions_needed": ["CB_elite", "DM_defensive", "GK_shotstopper"]
                                })
                except Exception:
                    pass

        # In-Form Opponent Capitulation: losing disproportionately to opponents in good form
        if "OppForm3" in self.losses.columns and not self.losses["OppForm3"].isna().all():
            high_form_losses = self.losses[self.losses["OppForm3"] > 7]
            high_form_rate = len(high_form_losses) / len(self.losses)
            if high_form_rate > 0.40:
                analysis["issues"].append({
                    "type": "in_form_opponent_capitulation",
                    "severity": "medium",
                    "description": f"{high_form_rate:.0%} of losses against in-form opponents (OppForm3 > 7) — struggle vs momentum teams",
                    "positions_needed": ["CB_experienced", "CM_bigmatch", "GK_commanding"]
                })

        return analysis


# ---------------------- Enhanced Player Evaluation System ---------------------- #

class PlayerEvaluator:
    """Comprehensive player evaluation system"""
    
    def __init__(self, players_df: pd.DataFrame, use_percentiles: bool = False):
        self.df = players_df.copy()
        self.use_percentiles = use_percentiles  # Toggle between min-max and percentile normalization
        self._prepare_data()
        
    def _prepare_data(self):
        """Prepare and clean player data"""
        
        # === STEP 1: DELETE EXACT DUPLICATES (same Player + same Squad) ===
        if 'Player' in self.df.columns and 'Squad' in self.df.columns:
            initial_count = len(self.df)
            self.df = self.df.drop_duplicates(subset=['Player', 'Squad'], keep='first')
            duplicates_removed = initial_count - len(self.df)
            if duplicates_removed > 0:
                print(f"🗑️  Removed {duplicates_removed} exact duplicates (same player, same team)")
        
        # === STEP 2: AGGREGATE PLAYERS WITH DIFFERENT SQUADS (mid-season transfers) ===
        if 'Player' in self.df.columns and 'Squad' in self.df.columns:
            # Check for players appearing with multiple squads
            player_squad_counts = self.df.groupby('Player')['Squad'].nunique()
            multi_squad_players = player_squad_counts[player_squad_counts > 1].index.tolist()
            
            if len(multi_squad_players) > 0:
                print(f"🔄 Found {len(multi_squad_players)} players with multiple teams (transfers)")
                print(f"   Examples: {', '.join(multi_squad_players[:3])}{'...' if len(multi_squad_players) > 3 else ''}")
                
                # Columns to SUM (counting stats - season totals)
                sum_cols = ['Gls', 'Ast', 'Sh', 'SoT', 'xG', 'npxG', 'xAG', 'KP', 'Succ', 
                           'Tkl', 'TklW', 'Int', 'Blocks', 'Clr', 'SCA', 'GCA', 'Touches', 
                           'Carries', 'PrgC', 'PrgP', 'PrgR', 'Fls', 'Recov', 'Won', 
                           'Lost_stats_misc', 'CrdY', 'CrdR', 'Min', 'MP',
                           'Saves', 'GA', 'CS', 'PKatt', 'PKsv', 'PKm',
                           'Blocks_stats_defense', 'Sh_Att_Pen', 'Def_Pen', 'Mid_3rd', 
                           'Att_3rd', 'Def_3rd', 'Att_Pen', 'Live', 'Dead', 'FK', 'TB',
                           'Sw', 'Crs', 'TI', 'CK', 'In', 'Out', 'Str', 'Cmp', 'Off',
                           'Pressures', 'Tkl_Att_3rd', 'Tkl_Mid_3rd', 'Tkl_Def_3rd',
                           'Att', 'Succ_stats_take_ons', 'Tkld', 'Carries_stats_possession',
                           'TotDist_stats_possession', 'PrgDist_stats_possession', 'CrsPA',
                           '1/3', 'Mis', 'Dis', 'Rec', 'PrgR_stats_possession',
                           'Long', 'Cmp_stats_passing', 'TotDist', 'PrgDist_stats_passing']
                
                # Columns to AVERAGE (percentages, rates, per-90 metrics)
                avg_cols = ['Cmp%', 'PassCmp%', 'Save%', 'CS%', 'Won%', 'Succ%', 'SoT%', 
                           'G/Sh', 'AerialWon%', 'Min%', 'PPM', 'G/SoT', 'Sh/90', 'SoT/90',
                           'G/90', 'Ast/90', 'G+A', 'G+A-PK', 'xG/90', 'xAG/90', 'xG+xAG',
                           'npxG/90', 'npxG+xAG', 'xA', 'A-xAG', 'KP/90', 'Cmp%_x',
                           'Tkl%', 'Int/90', 'Blocks/90', 'Clr/90', 'Err', 'Touches/90',
                           'Def Pen_x', 'Att Pen_x', 'Live/90', 'Dead/90', 'FK/90',
                           'Press%', 'Succ%_x', 'Tkld%', 'Carries/90', 'TotDist/90',
                           'PrgDist/90', 'PrgC/90', 'Rec/90', 'PrgR/90', 'Att_GCA', 'PassLive_GCA',
                           'GA90', 'SoTA', 'PKsv%', 'PSxG', 'PSxG/SoT', 'PSxG+/-', '/90',
                           'Launch%', 'AvgLen', 'Launch%_x', 'AvgLen_x', 'Opp', 'Stp', 'Stp%',
                           '#OPA', 'AvgDist', '#OPA/90']
                
                # Columns to take LAST value (descriptive - most recent is most relevant)
                last_cols = ['Nation', 'Pos', 'Comp', 'League']
                
                # Columns to NEVER aggregate (makes no sense or invalid to combine)
                skip_cols = ['Age', 'Born', 'Rk', '90s', 'Matches']  # Age/Born don't aggregate
                
                # Build aggregation dictionary
                agg_dict = {}
                
                # Sum columns
                for col in sum_cols:
                    if col in self.df.columns and col not in skip_cols:
                        agg_dict[col] = 'sum'
                
                # Average columns
                for col in avg_cols:
                    if col in self.df.columns and col not in skip_cols:
                        agg_dict[col] = 'mean'
                
                # Last value columns
                for col in last_cols:
                    if col in self.df.columns and col not in skip_cols:
                        agg_dict[col] = 'last'
                
                # Special: Age - take the most recent (last) age
                if 'Age' in self.df.columns:
                    agg_dict['Age'] = 'last'
                
                # Special: Born - keep first (birth year doesn't change)
                if 'Born' in self.df.columns:
                    agg_dict['Born'] = 'first'
                
                # Special: Squad - combine all teams
                agg_dict['Squad'] = lambda x: ' → '.join(x.unique())
                
                # Aggregate only the multi-squad players
                multi_squad_df = self.df[self.df['Player'].isin(multi_squad_players)].copy()
                single_squad_df = self.df[~self.df['Player'].isin(multi_squad_players)].copy()
                
                aggregated = multi_squad_df.groupby('Player', as_index=False).agg(agg_dict)
                
                # Combine back
                self.df = pd.concat([single_squad_df, aggregated], ignore_index=True)
                print(f"   ✅ Combined stats for {len(multi_squad_players)} transferred players")
        
        # Ensure 90s column
        if "90s" not in self.df.columns:
            if "Min" in self.df.columns:
                self.df["90s"] = pd.to_numeric(self.df["Min"], errors='coerce') / 90.0
            else:
                self.df["90s"] = 0
        else:
            self.df["90s"] = pd.to_numeric(self.df["90s"], errors='coerce')
        
        # Filter for minimum playing time
        self.df = self.df[self.df["90s"] >= 2.0].copy()
        
        # Position parsing
        self.df["primary_pos"] = self.df["Pos"].fillna("").str.split(",").str[0].str.strip()
        
        # Create position categories
        self.df["is_gk"] = self.df["primary_pos"].str.contains("GK", na=False)
        self.df["is_def"] = self.df["primary_pos"].str.contains("DF|WB", na=False)
        self.df["is_mid"] = self.df["primary_pos"].str.contains("MF|DM|CM|AM", na=False)
        self.df["is_att"] = self.df["primary_pos"].str.contains("FW|CF|LW|RW|ST", na=False)
        
        # Specific position flags
        self.df["is_cb"] = self.df["Pos"].fillna("").str.contains("CB", na=False) | \
                           (self.df["is_def"] & ~self.df["Pos"].fillna("").str.contains("FB|WB|LB|RB", na=False))
        self.df["is_fb"] = self.df["Pos"].fillna("").str.contains("FB|WB|LB|RB", na=False) | \
                           (self.df["is_def"] & self.df["Pos"].fillna("").str.contains("B", na=False))
        self.df["is_dm"] = (self.df["Pos"].fillna("").str.contains("DM", na=False)) | \
                           (self.df["is_mid"] & self.df.get("Tkl", 0) > self.df.get("Ast", 0))
        self.df["is_cm"] = self.df["Pos"].fillna("").str.contains("CM|MF", na=False) & self.df["is_mid"]
        self.df["is_am"] = self.df["Pos"].fillna("").str.contains("AM|CAM", na=False)
        self.df["is_winger"] = self.df["Pos"].fillna("").str.contains("LW|RW|W", na=False)
        self.df["is_striker"] = self.df["Pos"].fillna("").str.contains("ST|CF|FW", na=False)
    
    def _safe_get(self, col_name, default=0):
        """Safely get column values with default fallback"""
        if col_name in self.df.columns:
            return pd.to_numeric(self.df[col_name], errors='coerce').fillna(default)
        return pd.Series(default, index=self.df.index)
        
    def evaluate_all_players(self) -> pd.DataFrame:
        """Evaluate all players across multiple dimensions"""
        self._calculate_per90_metrics()
        
        # Indicate which normalization method is being used
        if self.use_percentiles:
            print(f"📊 Using PERCENTILE-based normalization (position-specific ranking)")
        else:
            print(f"📊 Using MIN-MAX normalization (absolute performance scaling)")
        
        self._calculate_defensive_scores()
        self._calculate_offensive_scores()
        self._calculate_playmaking_scores()
        self._calculate_physical_scores()
        self._calculate_mental_scores()
        self._calculate_specialized_role_scores()
        self._calculate_comprehensive_scores()  # NEW: Enhanced scoring
        return self.df
    
    def _calculate_comprehensive_scores(self):
        """
        ULTRA-COMPREHENSIVE scoring using MAXIMUM available columns for realistic differentiation.
        Each archetype uses DISTINCT stat combinations to minimize repetitive recommendations.
        19 specialized archetypes + 3 broad categories = 22 total scoring dimensions
        """
        
        # === SPECIALIZED DEFENDER ARCHETYPES (5 types) ===
        
        # 1. STOPPER CB (Pure defending, blocking, clearances, aerial dominance)
        # 1. STOPPER CB (Pure defending, blocking, clearances, aerial dominance)
        stopper_components = []
        stopper_components.append(0.25 * self._norm(self._safe_get('Clr_p90')))
        stopper_components.append(0.20 * self._norm(self._safe_get('Blocks_stats_defense') / self.df['90s'].replace(0, np.nan)))
        stopper_components.append(0.15 * self._norm(self._safe_get('Tkl_p90') + self._safe_get('Int_p90')))
        stopper_components.append(0.15 * self._norm(self._safe_get('Won') / self.df['90s'].replace(0, np.nan)))
        if 'Won%' in self.df.columns:
            stopper_components.append(0.10 * self._norm(self._safe_get('Won%') / 100))
        if 'Def 3rd' in self.df.columns:
            stopper_components.append(0.10 * self._norm(self._safe_get('Def 3rd') / self.df['90s'].replace(0, np.nan)))
        stopper_components.append(0.05 * (1 - self._norm(self._safe_get('Err') / self.df['90s'].replace(0, np.nan))))
        self.df['stopper_cb_score'] = sum(stopper_components) if stopper_components else 0
        
        # 2. BALL-PLAYING CB (Passing accuracy, progressive passes, switches, build-up)
        ballplaying_components = []
        ballplaying_components.append(0.25 * self._norm(self._safe_get('Cmp%') / 100))
        ballplaying_components.append(0.20 * self._norm(self._safe_get('PrgP_stats_passing') / self.df['90s'].replace(0, np.nan)))
        ballplaying_components.append(0.15 * self._norm(self._safe_get('PrgC_stats_possession') / self.df['90s'].replace(0, np.nan)))
        if 'TotDist' in self.df.columns:
            ballplaying_components.append(0.15 * self._norm(self._safe_get('TotDist') / self.df['90s'].replace(0, np.nan)))
        ballplaying_components.append(0.10 * self._norm(self._safe_get('Tkl_p90') + self._safe_get('Int_p90')))
        if 'Sw' in self.df.columns:
            ballplaying_components.append(0.10 * self._norm(self._safe_get('Sw') / self.df['90s'].replace(0, np.nan)))
        ballplaying_components.append(0.05 * (1 - self._norm(self._safe_get('Err') / self.df['90s'].replace(0, np.nan))))
        self.df['ballplaying_cb_score'] = sum(ballplaying_components) if ballplaying_components else 0
        
        # 3. AGGRESSIVE CB (High tackles, recoveries, pressing)
        aggressive_cb_components = []
        aggressive_cb_components.append(0.30 * self._norm(self._safe_get('Tkl_p90')))
        aggressive_cb_components.append(0.25 * self._norm(self._safe_get('Recov_p90')))
        if 'Att 3rd' in self.df.columns:
            aggressive_cb_components.append(0.20 * self._norm(self._safe_get('Att 3rd') / self.df['90s'].replace(0, np.nan)))
        aggressive_cb_components.append(0.15 * self._norm(self._safe_get('Won') / self.df['90s'].replace(0, np.nan)))
        aggressive_cb_components.append(0.10 * self._norm(self._safe_get('Int_p90')))
        self.df['aggressive_cb_score'] = sum(aggressive_cb_components) if aggressive_cb_components else 0
        
        # 4. ATTACKING FULLBACK (Crosses, progressive carries, attacking third)
        attacking_fb_components = []
        if 'Crs' in self.df.columns:
            attacking_fb_components.append(0.25 * self._norm(self._safe_get('Crs') / self.df['90s'].replace(0, np.nan)))
        if 'CrsPA' in self.df.columns:
            attacking_fb_components.append(0.20 * self._norm(self._safe_get('CrsPA') / self.df['90s'].replace(0, np.nan)))
        attacking_fb_components.append(0.20 * self._norm(self._safe_get('PrgC_stats_possession') / self.df['90s'].replace(0, np.nan)))
        if 'Att 3rd_stats_possession' in self.df.columns:
            attacking_fb_components.append(0.15 * self._norm(self._safe_get('Att 3rd_stats_possession') / self.df['90s'].replace(0, np.nan)))
        attacking_fb_components.append(0.10 * self._norm(self._safe_get('xAG_p90')))
        attacking_fb_components.append(0.10 * self._norm(self._safe_get('Succ%') / 100 if 'Succ%' in self.df.columns else 0))
        self.df['attacking_fb_score'] = sum(attacking_fb_components) if attacking_fb_components else 0
        
        # 5. DEFENSIVE FULLBACK (Tackles, interceptions, defensive positioning)
        defensive_fb_components = []
        defensive_fb_components.append(0.30 * self._norm(self._safe_get('Tkl_p90')))
        defensive_fb_components.append(0.25 * self._norm(self._safe_get('Int_p90')))
        defensive_fb_components.append(0.20 * self._norm(self._safe_get('Recov_p90')))
        if 'Def 3rd' in self.df.columns:
            defensive_fb_components.append(0.15 * self._norm(self._safe_get('Def 3rd') / self.df['90s'].replace(0, np.nan)))
        defensive_fb_components.append(0.10 * (1 - self._norm(self._safe_get('Tkld%') / 100 if 'Tkld%' in self.df.columns else 0)))
        self.df['defensive_fb_score'] = sum(defensive_fb_components) if defensive_fb_components else 0
        
        # === SPECIALIZED MIDFIELDER ARCHETYPES ===
        
        # 6. DESTROYER DM (Pure defensive midfielder - tackles, blocks, interceptions)
        destroyer_components = []
        destroyer_components.append(0.30 * self._norm(self._safe_get('Tkl_p90') + self._safe_get('Int_p90')))
        destroyer_components.append(0.20 * self._norm(self._safe_get('Blocks_stats_defense') / self.df['90s'].replace(0, np.nan)))
        destroyer_components.append(0.20 * self._norm(self._safe_get('Recov_p90')))
        if 'Def 3rd' in self.df.columns and 'Mid 3rd' in self.df.columns:
            destroyer_components.append(0.15 * self._norm((self._safe_get('Def 3rd') + self._safe_get('Mid 3rd')) / self.df['90s'].replace(0, np.nan)))
        destroyer_components.append(0.10 * self._norm(self._safe_get('Won') / self.df['90s'].replace(0, np.nan)))
        destroyer_components.append(0.05 * self._norm(self._safe_get('Cmp%') / 100))
        self.df['destroyer_dm_score'] = sum(destroyer_components) if destroyer_components else 0
        
        # 7. DEEP-LYING PLAYMAKER (Passing range, switches, progressive passes)
        dlp_components = []
        dlp_components.append(0.30 * self._norm(self._safe_get('Cmp%') / 100))
        if 'TotDist' in self.df.columns:
            dlp_components.append(0.25 * self._norm(self._safe_get('TotDist') / self.df['90s'].replace(0, np.nan)))
        dlp_components.append(0.20 * self._norm(self._safe_get('PrgP_stats_passing') / self.df['90s'].replace(0, np.nan)))
        if 'Sw' in self.df.columns:
            dlp_components.append(0.15 * self._norm(self._safe_get('Sw') / self.df['90s'].replace(0, np.nan)))
        dlp_components.append(0.10 * self._norm(self._safe_get('KP') / self.df['90s'].replace(0, np.nan)))
        self.df['dlp_score'] = sum(dlp_components) if dlp_components else 0
        
        # 8. BOX-TO-BOX MIDFIELDER (Goals, distance covered, all-round contribution)
        b2b_components = []
        b2b_components.append(0.25 * self._norm(self._safe_get('Gls_p90') + self._safe_get('Ast_p90')))
        b2b_components.append(0.20 * self._norm(self._safe_get('Tkl_p90') + self._safe_get('Int_p90')))
        b2b_components.append(0.20 * self._norm(self._safe_get('PrgC_stats_possession') / self.df['90s'].replace(0, np.nan)))
        if 'TotDist_stats_possession' in self.df.columns:
            b2b_components.append(0.15 * self._norm(self._safe_get('TotDist_stats_possession') / self.df['90s'].replace(0, np.nan)))
        b2b_components.append(0.10 * self._norm(self._safe_get('Recov_p90')))
        b2b_components.append(0.10 * self._norm(self._safe_get('Touches_p90')))
        self.df['b2b_cm_score'] = sum(b2b_components) if b2b_components else 0
        
        # 9. CREATIVE CM (Key passes, through balls, shot creation)
        creative_cm_components = []
        creative_cm_components.append(0.30 * self._norm(self._safe_get('KP') / self.df['90s'].replace(0, np.nan)))
        creative_cm_components.append(0.25 * self._norm(self._safe_get('xAG_p90')))
        if 'SCA' in self.df.columns:
            creative_cm_components.append(0.20 * self._norm(self._safe_get('SCA') / self.df['90s'].replace(0, np.nan)))
        if 'TB' in self.df.columns:
            creative_cm_components.append(0.10 * self._norm(self._safe_get('TB') / self.df['90s'].replace(0, np.nan)))
        if 'PPA' in self.df.columns:
            creative_cm_components.append(0.10 * self._norm(self._safe_get('PPA') / self.df['90s'].replace(0, np.nan)))
        creative_cm_components.append(0.05 * self._norm(self._safe_get('PrgP_stats_passing') / self.df['90s'].replace(0, np.nan)))
        self.df['creative_cm_score'] = sum(creative_cm_components) if creative_cm_components else 0
        
        # 10. ADVANCED PLAYMAKER / #10 (Final third creativity, assists, through balls)
        playmaker_components = []
        playmaker_components.append(0.30 * self._norm(self._safe_get('xAG_p90')))
        playmaker_components.append(0.25 * self._norm(self._safe_get('KP') / self.df['90s'].replace(0, np.nan)))
        if 'PPA' in self.df.columns:
            playmaker_components.append(0.15 * self._norm(self._safe_get('PPA') / self.df['90s'].replace(0, np.nan)))
        if 'GCA' in self.df.columns:
            playmaker_components.append(0.15 * self._norm(self._safe_get('GCA') / self.df['90s'].replace(0, np.nan)))
        playmaker_components.append(0.10 * self._norm(self._safe_get('xG_p90')))
        if 'TB' in self.df.columns:
            playmaker_components.append(0.05 * self._norm(self._safe_get('TB') / self.df['90s'].replace(0, np.nan)))
        self.df['playmaker_am_score'] = sum(playmaker_components) if playmaker_components else 0
        
        # === SPECIALIZED ATTACKER ARCHETYPES ===
        
        # 11. POACHER / PENALTY BOX STRIKER (Pure goal scoring, conversion)
        poacher_components = []
        poacher_components.append(0.35 * self._norm(self._safe_get('Gls_p90')))
        if 'G/Sh' in self.df.columns:
            poacher_components.append(0.25 * self._norm(self._safe_get('G/Sh')))
        poacher_components.append(0.20 * self._norm(self._safe_get('npxG_p90')))
        if 'Att Pen' in self.df.columns:
            poacher_components.append(0.10 * self._norm(self._safe_get('Att Pen') / self.df['90s'].replace(0, np.nan)))
        if 'G-xG' in self.df.columns:
            poacher_components.append(0.10 * self._norm(self._safe_get('G-xG') + 1))
        self.df['poacher_st_score'] = sum(poacher_components) if poacher_components else 0
        
        # 12. TARGET MAN (Aerial duels, hold-up play, physical presence)
        targetman_components = []
        targetman_components.append(0.35 * self._norm(self._safe_get('Won') / self.df['90s'].replace(0, np.nan)))
        if 'Won%' in self.df.columns:
            targetman_components.append(0.25 * self._norm(self._safe_get('Won%') / 100))
        targetman_components.append(0.15 * self._norm(self._safe_get('Touches_p90')))
        targetman_components.append(0.10 * self._norm(self._safe_get('Gls_p90')))
        if 'Fld' in self.df.columns:
            targetman_components.append(0.10 * self._norm(self._safe_get('Fld') / self.df['90s'].replace(0, np.nan)))
        targetman_components.append(0.05 * self._norm(self._safe_get('Ast_p90')))
        self.df['targetman_st_score'] = sum(targetman_components) if targetman_components else 0
        
        # 13. PRESSING FORWARD (Pressures, defensive contribution, work rate)
        pressing_fw_components = []
        if 'Pressures' in self.df.columns:
            pressing_fw_components.append(0.35 * self._norm(self._safe_get('Pressures') / self.df['90s'].replace(0, np.nan)))
        pressing_fw_components.append(0.20 * self._norm(self._safe_get('Recov_p90')))
        pressing_fw_components.append(0.15 * self._norm(self._safe_get('Gls_p90')))
        if 'TotDist_stats_possession' in self.df.columns:
            pressing_fw_components.append(0.15 * self._norm(self._safe_get('TotDist_stats_possession') / self.df['90s'].replace(0, np.nan)))
        pressing_fw_components.append(0.10 * self._norm(self._safe_get('Tkl_p90')))
        pressing_fw_components.append(0.05 * self._norm(self._safe_get('Blocks_stats_defense') / self.df['90s'].replace(0, np.nan)))
        self.df['pressing_fw_score'] = sum(pressing_fw_components) if pressing_fw_components else 0
        
        # 14. COMPLETE FORWARD (Goals, assists, dribbling, all-round)
        complete_fw_components = []
        complete_fw_components.append(0.25 * self._norm(self._safe_get('Gls_p90')))
        complete_fw_components.append(0.20 * self._norm(self._safe_get('Ast_p90')))
        complete_fw_components.append(0.20 * self._norm(self._safe_get('xG_p90') + self._safe_get('xAG_p90')))
        if 'SCA' in self.df.columns:
            complete_fw_components.append(0.15 * self._norm(self._safe_get('SCA') / self.df['90s'].replace(0, np.nan)))
        complete_fw_components.append(0.10 * self._norm(self._safe_get('Succ') / self.df['90s'].replace(0, np.nan)))
        complete_fw_components.append(0.10 * self._norm(self._safe_get('PrgC_stats_possession') / self.df['90s'].replace(0, np.nan)))
        self.df['complete_fw_score'] = sum(complete_fw_components) if complete_fw_components else 0
        
        # 15. DRIBBLING WINGER (Successful dribbles, progressive carries, 1v1s)
        dribbling_w_components = []
        dribbling_w_components.append(0.35 * self._norm(self._safe_get('Succ') / self.df['90s'].replace(0, np.nan)))
        if 'Succ%' in self.df.columns:
            dribbling_w_components.append(0.20 * self._norm(self._safe_get('Succ%') / 100))
        dribbling_w_components.append(0.20 * self._norm(self._safe_get('PrgC_stats_possession') / self.df['90s'].replace(0, np.nan)))
        if 'Att Pen' in self.df.columns:
            dribbling_w_components.append(0.10 * self._norm(self._safe_get('Att Pen') / self.df['90s'].replace(0, np.nan)))
        if 'TO' in self.df.columns:
            dribbling_w_components.append(0.10 * self._norm(self._safe_get('TO') / self.df['90s'].replace(0, np.nan)))
        dribbling_w_components.append(0.05 * self._norm(self._safe_get('Gls_p90')))
        self.df['dribbling_w_score'] = sum(dribbling_w_components) if dribbling_w_components else 0
        
        # 16. CROSSING WINGER (Crosses, assists, wide play)
        crossing_w_components = []
        if 'Crs' in self.df.columns:
            crossing_w_components.append(0.35 * self._norm(self._safe_get('Crs') / self.df['90s'].replace(0, np.nan)))
        if 'CrsPA' in self.df.columns:
            crossing_w_components.append(0.25 * self._norm(self._safe_get('CrsPA') / self.df['90s'].replace(0, np.nan)))
        crossing_w_components.append(0.20 * self._norm(self._safe_get('xAG_p90')))
        crossing_w_components.append(0.10 * self._norm(self._safe_get('Ast_p90')))
        crossing_w_components.append(0.10 * self._norm(self._safe_get('KP') / self.df['90s'].replace(0, np.nan)))
        self.df['crossing_w_score'] = sum(crossing_w_components) if crossing_w_components else 0
        
        # 17. INSIDE FORWARD / INVERTED WINGER (Goals, shots, cutting inside)
        inside_fw_components = []
        inside_fw_components.append(0.30 * self._norm(self._safe_get('Gls_p90')))
        inside_fw_components.append(0.25 * self._norm(self._safe_get('npxG_p90')))
        inside_fw_components.append(0.20 * self._norm(self._safe_get('Sh') / self.df['90s'].replace(0, np.nan)))
        if 'SoT%' in self.df.columns:
            inside_fw_components.append(0.10 * self._norm(self._safe_get('SoT%') / 100))
        inside_fw_components.append(0.10 * self._norm(self._safe_get('PrgC_stats_possession') / self.df['90s'].replace(0, np.nan)))
        if 'G/Sh' in self.df.columns:
            inside_fw_components.append(0.05 * self._norm(self._safe_get('G/Sh')))
        self.df['inside_fw_score'] = sum(inside_fw_components) if inside_fw_components else 0
        
        # === GOALKEEPER ARCHETYPES ===
        
        # 18. SHOT-STOPPING GK (Save percentage, post-shot xG)
        if 'Save%' in self.df.columns:
            shotstopper_components = []
            shotstopper_components.append(0.40 * self._norm(self._safe_get('Save%') / 100))
            if 'PSxG+/-' in self.df.columns:
                shotstopper_components.append(0.30 * self._norm(self._safe_get('PSxG+/-') + 5))
            if 'Saves' in self.df.columns:
                shotstopper_components.append(0.20 * self._norm(self._safe_get('Saves') / self.df['90s'].replace(0, np.nan)))
            if 'CS%' in self.df.columns:
                shotstopper_components.append(0.10 * self._norm(self._safe_get('CS%') / 100))
            self.df['shotstopper_gk_score'] = (sum(shotstopper_components) if shotstopper_components else 0) * self.df['is_gk'].astype(int)
        else:
            self.df['shotstopper_gk_score'] = 0
        
        # 19. SWEEPER-KEEPER (Actions outside box, passing, distribution)
        if 'AvgDist' in self.df.columns:
            sweeper_components = []
            sweeper_components.append(0.30 * self._norm(self._safe_get('AvgDist')))
            if '#OPA/90' in self.df.columns:
                sweeper_components.append(0.25 * self._norm(self._safe_get('#OPA/90')))
            if 'Cmp%_stats_keeper_adv' in self.df.columns:
                sweeper_components.append(0.20 * self._norm(self._safe_get('Cmp%_stats_keeper_adv') / 100))
            if 'Launch%' in self.df.columns:
                sweeper_components.append(0.15 * self._norm(1 - self._safe_get('Launch%') / 100))
            if 'Save%' in self.df.columns:
                sweeper_components.append(0.10 * self._norm(self._safe_get('Save%') / 100))
            self.df['sweeper_gk_score'] = (sum(sweeper_components) if sweeper_components else 0) * self.df['is_gk'].astype(int)
        else:
            self.df['sweeper_gk_score'] = self.df.get('gk_score', 0) * self.df['is_gk'].astype(int)
        
        # === FALLBACK BROAD CATEGORY SCORES ===
        
        # General defender score (for when no specific archetype matches)
        self.df['defender_comprehensive'] = (
            0.30 * self.df['stopper_cb_score'] +
            0.30 * self.df['ballplaying_cb_score'] +
            0.20 * self.df['defensive_fb_score'] +
            0.20 * self.df['aggressive_cb_score']
        )
        
        # General midfielder score
        self.df['midfielder_comprehensive'] = (
            0.25 * self.df['b2b_cm_score'] +
            0.25 * self.df['creative_cm_score'] +
            0.25 * self.df['dlp_score'] +
            0.15 * self.df['destroyer_dm_score'] +
            0.10 * self.df['playmaker_am_score']
        )
        
        # General attacker score
        self.df['attacker_comprehensive'] = (
            0.25 * self.df['complete_fw_score'] +
            0.25 * self.df['poacher_st_score'] +
            0.20 * self.df['inside_fw_score'] +
            0.15 * self.df['dribbling_w_score'] +
            0.15 * self.df['crossing_w_score']
        )
        
        # Versatile all-rounder
        self.df['versatile_score'] = (
            0.33 * self.df['defender_comprehensive'] +
            0.34 * self.df['midfielder_comprehensive'] +
            0.33 * self.df['attacker_comprehensive']
        )
        
        print(f"Enhanced archetype scoring: {len(self.df)} players evaluated across 19 distinct archetypes")
        print(f"  - Defenders: {(self.df['defender_comprehensive'] > 0.3).sum()} strong")
        print(f"  - Midfielders: {(self.df['midfielder_comprehensive'] > 0.3).sum()} strong")
        print(f"  - Attackers: {(self.df['attacker_comprehensive'] > 0.3).sum()} strong")

    def _calculate_per90_metrics(self):
        per90_cols = [
            'Gls', 'Ast', 'Sh', 'SoT', 'xG', 'npxG', 'xAG',
            'PrgC', 'PrgP', 'PrgR', 'Tkl', 'TklW', 'Int', 'Blocks', 'Clr',
            'SCA', 'GCA', 'Touches', 'Carries', 'PrgDist_stats_possession',
            'Fls', 'Recov', 'Won', 'Lost_stats_misc'
        ]
        for col in per90_cols:
            if col in self.df.columns:
                try:
                    self.df[f"{col}_p90"] = pd.to_numeric(self.df[col], errors='coerce') / self.df["90s"].replace(0, np.nan)
                except Exception:
                    self.df[f"{col}_p90"] = 0
    
    def _calculate_defensive_scores(self):
        # CB Score - NO LONGER multiplied by is_cb flag!
        self.df["cb_score"] = (
            0.25 * self._normalize(self._safe_get("Blocks_p90")) +
            0.25 * self._normalize(self._safe_get("Clr_p90")) +
            0.15 * self._normalize(self._safe_get("Won") / self.df["90s"].replace(0, np.nan)) +
            0.15 * self._normalize(self._safe_get("Int_p90")) +
            0.10 * self._normalize(self._safe_get("TklW") / self._safe_get("Tkl", 1).replace(0, 1)) +
            0.10 * (1 - self._normalize(self._safe_get("Err") / self.df["90s"].replace(0, np.nan)))
        )
        
        # FB Score - NO LONGER multiplied by is_fb flag!
        self.df["fb_score"] = (
            0.25 * self._normalize(self._safe_get("Tkl_p90")) +
            0.20 * self._normalize(self._safe_get("Int_p90")) +
            0.20 * self._normalize(self._safe_get("Recov_p90")) +
            0.15 * self._normalize(self._safe_get("PrgC_p90")) +
            0.10 * self._normalize(self._safe_get("Succ%") / 100) +
            0.10 * self._normalize(self._safe_get("CrsPA") / self.df["90s"].replace(0, np.nan))
        )
        
        # DM Score - NO LONGER multiplied by is_dm flag!
        tkl_int = self._safe_get("Tkl_p90") + self._safe_get("Int_p90")
        self.df["dm_score"] = (
            0.30 * self._normalize(tkl_int) +
            0.20 * self._normalize(self._safe_get("Blocks_p90")) +
            0.15 * self._normalize(self._safe_get("Recov_p90")) +
            0.15 * self._normalize(self._safe_get("PrgP_p90")) +
            0.10 * self._normalize(self._safe_get("PassCmp%") / 100 if "PassCmp%" in self.df.columns else self._safe_get("Cmp%") / 100) +
            0.10 * self._normalize(self._safe_get("Won") / self.df["90s"].replace(0, np.nan))
        )
        
        # GK Score - Keep position flag for goalkeepers (they're truly unique)
        if "Save%" in self.df.columns or "GA90" in self.df.columns:
            save_pct = self._safe_get("Save%") / 100
            ga90 = self._safe_get("GA90")
            cs_pct = self._safe_get("CS%") / 100
            psxg = self._safe_get("PSxG+/-")
            self.df["gk_score"] = (
                0.40 * self._normalize(save_pct) +
                0.30 * self._normalize(2 - ga90) +
                0.15 * self._normalize(cs_pct) +
                0.15 * self._normalize(psxg + 1)
            ) * self.df["is_gk"].astype(int)  # Keep flag for GK
        else:
            self.df["gk_score"] = 0
    
    def _calculate_offensive_scores(self):
        g_sh = self._safe_get("G/Sh") if "G/Sh" in self.df.columns else (self._safe_get("Gls") / self._safe_get("Sh", 1).replace(0, 1))
        g_xg = self._safe_get("G-xG") if "G-xG" in self.df.columns else (self._safe_get("Gls") - self._safe_get("xG"))
        sot_pct = self._safe_get("SoT%") / 100 if "SoT%" in self.df.columns else (self._safe_get("SoT") / self._safe_get("Sh", 1).replace(0, 1))
        
        # ST Score - NO LONGER multiplied by is_striker flag!
        self.df["st_score"] = (
            0.35 * self._normalize(self._safe_get("Gls_p90")) +
            0.20 * self._normalize(g_sh) +
            0.15 * self._normalize(self._safe_get("npxG_p90")) +
            0.15 * self._normalize(sot_pct) +
            0.10 * self._normalize(self._safe_get("Won") / self.df["90s"].replace(0, np.nan)) +
            0.05 * self._normalize(g_xg)
        )
        
        # W Score - NO LONGER multiplied by is_winger flag!
        succ_dribbles = self._safe_get("Succ") if "Succ" in self.df.columns else 0
        crosses = self._safe_get("Crs") if "Crs" in self.df.columns else 0
        sca90 = self._safe_get("SCA90") if "SCA90" in self.df.columns else self._safe_get("SCA_p90")
        self.df["w_score"] = (
            0.20 * self._normalize(self._safe_get("Gls_p90")) +
            0.20 * self._normalize(self._safe_get("Ast_p90")) +
            0.20 * self._normalize(succ_dribbles / self.df["90s"].replace(0, np.nan)) +
            0.15 * self._normalize(self._safe_get("PrgC_p90")) +
            0.15 * self._normalize(sca90) +
            0.10 * self._normalize(crosses / self.df["90s"].replace(0, np.nan))
        )
        
        # AM Score - NO LONGER multiplied by is_am flag!
        kp = self._safe_get("KP") if "KP" in self.df.columns else 0
        ppa = self._safe_get("PPA") if "PPA" in self.df.columns else 0
        self.df["am_score"] = (
            0.25 * self._normalize(self._safe_get("xAG_p90")) +
            0.20 * self._normalize(kp / self.df["90s"].replace(0, np.nan)) +
            0.20 * self._normalize(sca90) +
            0.15 * self._normalize(self._safe_get("Gls_p90")) +
            0.10 * self._normalize(ppa / self.df["90s"].replace(0, np.nan)) +
            0.10 * self._normalize(self._safe_get("PrgP_p90"))
        )
    
    def _calculate_playmaking_scores(self):
        pass_cmp = self._safe_get("PassCmp%") / 100 if "PassCmp%" in self.df.columns else self._safe_get("Cmp%") / 100
        tkl_int = self._safe_get("Tkl_p90") + self._safe_get("Int_p90")
        
        # CM Score - NO LONGER multiplied by is_cm flag!
        self.df["cm_score"] = (
            0.20 * self._normalize(pass_cmp) +
            0.15 * self._normalize(self._safe_get("PrgP_p90")) +
            0.15 * self._normalize(self._safe_get("Touches_p90")) +
            0.15 * self._normalize(self._safe_get("Carries_p90")) +
            0.10 * self._normalize(tkl_int) +
            0.10 * self._normalize(self._safe_get("xAG_p90")) +
            0.10 * self._normalize(self._safe_get("PrgDist_stats_possession_p90")) +
            0.05 * self._normalize(self._safe_get("Won") / self.df["90s"].replace(0, np.nan))
        )
        
        # DLP Score - slightly more flexible flag check
        tot_dist = self._safe_get("TotDist") if "TotDist" in self.df.columns else 0
        switches = self._safe_get("Sw") if "Sw" in self.df.columns else 0
        long_passes = self._safe_get("Long") if "Long" in self.df.columns else 0
        self.df["dlp_score"] = (
            0.30 * self._normalize(pass_cmp) +
            0.25 * self._normalize(self._safe_get("PrgP_p90")) +
            0.20 * self._normalize(tot_dist / self.df["90s"].replace(0, np.nan)) +
            0.15 * self._normalize(switches / self.df["90s"].replace(0, np.nan)) +
            0.10 * self._normalize(long_passes / self.df["90s"].replace(0, np.nan))
        )
    
    def _calculate_physical_scores(self):
        pressures = self._safe_get("Pressures") / self.df["90s"].replace(0, np.nan) if "Pressures" in self.df.columns else 0
        tot_dist_poss = self._safe_get("TotDist_stats_possession") if "TotDist_stats_possession" in self.df.columns else 0
        self.df["workrate_score"] = (
            0.30 * self._normalize(self._safe_get("Touches_p90")) +
            0.25 * self._normalize(pressures) +
            0.25 * self._normalize(self._safe_get("Recov_p90")) +
            0.20 * self._normalize(tot_dist_poss / self.df["90s"].replace(0, np.nan))
        )
        
        aerial_won_pct = self._safe_get("AerialWon%") / 100 if "AerialWon%" in self.df.columns else 0
        self.df["aerial_score"] = (
            0.60 * self._normalize(self._safe_get("Won") / self.df["90s"].replace(0, np.nan)) +
            0.40 * self._normalize(aerial_won_pct)
        )
        
        min_pct = self._safe_get("Min%") / 100 if "Min%" in self.df.columns else (self._safe_get("Min") / 90)
        mp = self._safe_get("MP") if "MP" in self.df.columns else self._safe_get("MP_stats_playing_time", 1)
        compl = self._safe_get("Compl") if "Compl" in self.df.columns else 0
        self.df["stamina_score"] = (
            0.50 * self._normalize(min_pct) +
            0.30 * self._normalize(self.df["90s"] / mp.replace(0, 1)) +
            0.20 * self._normalize(compl / mp.replace(0, 1))
        )
    
    def _calculate_mental_scores(self):
        crdy = self._safe_get("CrdY") if "CrdY" in self.df.columns else self._safe_get("CrdY_stats_misc")
        crdr = self._safe_get("CrdR") if "CrdR" in self.df.columns else self._safe_get("CrdR_stats_misc")
        self.df["discipline_score"] = 1 - (
            0.60 * self._normalize(crdy / self.df["90s"].replace(0, np.nan)) +
            0.30 * self._normalize(crdr * 3 / self.df["90s"].replace(0, np.nan)) +
            0.10 * self._normalize(self._safe_get("Fls_p90"))
        )
        
        age = self._safe_get("Age", 25)
        mp = self._safe_get("MP") if "MP" in self.df.columns else self._safe_get("MP_stats_playing_time", 0)
        self.df["experience_score"] = (
            0.40 * np.where((age >= 24) & (age <= 32), 1, np.where(age < 24, age/24, 32/age)) +
            0.30 * self._normalize(self.df["90s"]) +
            0.30 * self._normalize(mp)
        )
        
        ppm = self._safe_get("PPM") if "PPM" in self.df.columns else 0
        xg_diff = self._safe_get("xG+/-90") if "xG+/-90" in self.df.columns else 0
        min_pct = self._safe_get("Min%") / 100 if "Min%" in self.df.columns else (self._safe_get("Min") / 90)
        if "PPM" in self.df.columns:
            self.df["consistency_score"] = (
                0.50 * self._normalize(ppm / 3) +
                0.30 * self._normalize(1 - abs(xg_diff)) +
                0.20 * self._normalize(min_pct)
            )
        else:
            self.df["consistency_score"] = self._normalize(min_pct)
    
    def _calculate_specialized_role_scores(self):
        pass_cmp = self._safe_get("PassCmp%") / 100 if "PassCmp%" in self.df.columns else self._safe_get("Cmp%") / 100
        self.df["cb_ballplaying"] = (
            0.40 * self.df["cb_score"] +
            0.30 * self._normalize(pass_cmp) +
            0.20 * self._normalize(self._safe_get("PrgP_p90")) +
            0.10 * self._normalize(self._safe_get("PrgC_p90"))
        ) * self.df["is_cb"].astype(int)
        
        pressures = self._safe_get("Pressures") / self.df["90s"].replace(0, np.nan) if "Pressures" in self.df.columns else 0
        self.df["st_pressing"] = (
            0.40 * self.df["st_score"] +
            0.30 * self._normalize(pressures) +
            0.30 * self.df["workrate_score"]
        ) * self.df["is_striker"].astype(int)
        
        tkl_int = self._safe_get("Tkl_p90") + self._safe_get("Int_p90")
        goals_assists = self._safe_get("Gls_p90") + self._safe_get("Ast_p90")
        self.df["cm_b2b"] = (
            0.30 * self.df["cm_score"] +
            0.25 * self.df["workrate_score"] +
            0.25 * self._normalize(goals_assists) +
            0.20 * self._normalize(tkl_int)
        ) * self.df["is_cm"].astype(int)
        
        touches = self._safe_get("Touches") if "Touches" in self.df.columns else 0
        self.df["st_targetman"] = (
            0.35 * self.df["st_score"] +
            0.35 * self.df["aerial_score"] +
            0.30 * self._normalize(touches / self.df["90s"].replace(0, np.nan))
        ) * self.df["is_striker"].astype(int)
        
        if "AvgDist" in self.df.columns:
            avg_dist = self._safe_get("AvgDist")
            opa_per90 = self._safe_get("#OPA/90") if "#OPA/90" in self.df.columns else self._safe_get("#OPA") / self.df["90s"].replace(0, np.nan)
            self.df["gk_sweeper"] = (
                0.50 * self.df.get("gk_score", 0) +
                0.30 * self._normalize(avg_dist) +
                0.20 * self._normalize(opa_per90)
            ) * self.df["is_gk"].astype(int)
        else:
            self.df["gk_sweeper"] = self.df.get("gk_score", 0)
    
    def _normalize(self, series):
        """Normalize a series to 0-1 range"""
        if isinstance(series, (int, float)):
            series = pd.Series([series])
        elif not isinstance(series, pd.Series):
            series = pd.Series(series)
        
        series = series.replace([np.inf, -np.inf], np.nan)
        min_val = series.min(skipna=True)
        max_val = series.max(skipna=True)
        if pd.isna(min_val) or pd.isna(max_val) or min_val == max_val:
            return pd.Series(0.5, index=series.index)
        normalized = (series - min_val) / (max_val - min_val)
        return normalized.fillna(0.5)

    def _normalize_percentile(self, series, position_mask=None):
        """
        Normalize using percentile ranking instead of min-max.
        
        This method calculates where each value ranks within the dataset as a percentile.
        A score of 0.7 means the player is better than 70% of comparable players.
        
        Args:
            series: The stat series to normalize (can be Series, int, float, or array-like)
            position_mask: Optional boolean mask/series to calculate percentiles 
                          within a specific position group only (e.g., only CBs)
        
        Returns:
            pd.Series: Percentile scores in 0-1 range
            
        Example:
            # Global percentiles (compare against all players)
            scores = self._normalize_percentile(self.df['Gls_p90'])
            
            # Position-specific percentiles (compare CBs only to other CBs)
            cb_mask = self.df['is_cb']
            scores = self._normalize_percentile(self.df['Tkl_p90'], position_mask=cb_mask)
        
        Notes:
            - More robust to outliers than min-max normalization
            - Provides intuitive interpretation (percentile rank)
            - Requires at least 3 valid data points to calculate
            - Returns 0.5 (median) for players with missing data or insufficient sample
        """
        # Convert to Series if needed
        if isinstance(series, (int, float)):
            series = pd.Series([series])
        elif not isinstance(series, pd.Series):
            series = pd.Series(series)
        
        # Clean infinite values
        series = series.replace([np.inf, -np.inf], np.nan)
        
        # If position mask provided, calculate percentiles only within that group
        if position_mask is not None:
            # Ensure mask is boolean Series aligned with series
            if not isinstance(position_mask, pd.Series):
                position_mask = pd.Series(position_mask, index=series.index)
            
            # Get valid data within position group
            valid_mask = position_mask & series.notna()
            valid_data = series[valid_mask]
            
            if len(valid_data) < 3:
                # Not enough data for meaningful percentiles
                return pd.Series(0.5, index=series.index)
            
            # Calculate percentile rank only for this position
            # Use 'average' method to handle ties fairly
            percentiles = pd.Series(0.5, index=series.index)
            percentiles[valid_mask] = valid_data.rank(pct=True, method='average')
            
        else:
            # Global percentiles across all players
            valid_data = series[series.notna()]
            
            if len(valid_data) < 3:
                return pd.Series(0.5, index=series.index)
            
            # Calculate percentile rank across entire dataset
            percentiles = series.rank(pct=True, method='average')
        
        # Fill any remaining NaN values with median (0.5)
        return percentiles.fillna(0.5)


    
    def _norm(self, series, position_mask=None):
        """
        Smart normalization wrapper that chooses method based on use_percentiles flag.
        
        This method automatically uses either min-max or percentile normalization
        depending on the initialization setting. Use this in scoring methods for
        automatic switching between normalization approaches.
        
        Args:
            series: The stat series to normalize
            position_mask: Optional position mask (only used for percentile method)
        
        Returns:
            Normalized series using the chosen method
            
        Example:
            # In a scoring method, use _norm instead of _normalize:
            score = 0.3 * self._norm(self._safe_get('Gls_p90'), position_mask=self.df['is_striker'])
        """
        if self.use_percentiles:
            return self._normalize_percentile(series, position_mask=position_mask)
        else:
            return self._normalize(series)

# ---------------------- Utility Functions ---------------------- #

def parse_age_range(text: str):
    """
    Parse user input like:
      '', '18-25', '>=30', '<=23', '30+', 'u23', 'U21'
    Returns (age_min, age_max) as ints or (None, None) if no filter.
    """
    if not text or not text.strip():
        return None, None
    s = text.strip().lower().replace(' ', '')

    m = re.fullmatch(r'u(\d+)', s)
    if m:
        return None, int(m.group(1))

    m = re.fullmatch(r'(?:>=|)(\d+)\+', s)
    if m:
        return int(m.group(1)), None
    m = re.fullmatch(r'>=\s*(\d+)', s)
    if m:
        return int(m.group(1)), None

    m = re.fullmatch(r'<=\s*(\d+)', s)
    if m:
        return None, int(m.group(1))

    m = re.fullmatch(r'(\d+)\s*-\s*(\d+)', s)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        if a > b: a, b = b, a
        return a, b

    m = re.fullmatch(r'(\d+)', s)
    if m:
        x = int(m.group(1))
        return x, x

    return None, None


def _normalize_string_for_match(s: str) -> str:
    """
    Normalize a string for fuzzy matching:
    - lowercases
    - strips accents (ÃƒÂ§ -> c, ÃƒÂ© -> e, etc.)
    - collapses whitespace
    """
    if s is None:
        return ""
    # Ensure it's a string
    s = str(s)

    # Normalize accents
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))

    # Lowercase, strip, collapse spaces
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


def merge_preferred_foot(
    players_df: pd.DataFrame,
    sofascore_df: pd.DataFrame,
    fb_player_col: str = "Player",
    fb_team_col: str = "Squad",
    ss_player_col: str = "name",
    ss_team_col: str = "team",
    ss_foot_col: str = "preferred_foot",
) -> pd.DataFrame:
    """
    Merge preferred foot from a SofaScore dataset into your players_df.
    Matching is accent-insensitive on player+team (e.g., c == ÃƒÂ§, ÃƒÂ¡ == a).
    """

    df_fb = players_df.copy()
    df_ss = sofascore_df.copy()

    # Normalize names & team names for both datasets
    df_fb["name_key"] = df_fb[fb_player_col].apply(_normalize_string_for_match)
    df_fb["team_key"] = df_fb[fb_team_col].apply(_normalize_string_for_match) if fb_team_col in df_fb.columns else ""

    df_ss["name_key"] = df_ss[ss_player_col].apply(_normalize_string_for_match)
    df_ss["team_key"] = df_ss[ss_team_col].apply(_normalize_string_for_match) if ss_team_col in df_ss.columns else ""

    # Keep only the columns we need from SofaScore side
    ss_keep_cols = ["name_key", "team_key", ss_foot_col]
    ss_keep_cols = [c for c in ss_keep_cols if c in df_ss.columns]
    df_ss_small = df_ss[ss_keep_cols].drop_duplicates(subset=["name_key", "team_key"])

    # Merge
    merged = df_fb.merge(
        df_ss_small,
        on=["name_key", "team_key"],
        how="left",
        suffixes=("", "_sofa")
    )

    # Make sure new column is nicely named "preferred_foot"
    if ss_foot_col in merged.columns:
        merged.rename(columns={ss_foot_col: "preferred_foot"}, inplace=True)

    # Clean up helper keys if you don't want to see them later
    merged.drop(columns=["name_key", "team_key"], inplace=True, errors="ignore")

    return merged


# ---------------------- Archetype Definitions ---------------------- #
# Single source-of-truth mapping: human-readable archetype name →
#   score_col   : column produced by PlayerEvaluator (already on evaluated_players)
#   position_hints : Pos substrings that typically fill this role (soft signal only)
#   key_stats   : list of (per90_col, higher_is_better) for display purposes
# Extend this dict to add new archetypes; scoring code reads it dynamically.

ARCHETYPE_DEFINITIONS: Dict[str, Dict] = {
    # ── Defenders ─────────────────────────────────────────────────────────
    "Stopper CB": {
        "score_col": "stopper_cb_score",
        "position_hints": ["CB", "DF"],
        "key_stats": [("Clr_p90", True), ("Tkl_p90", True), ("Int_p90", True)],
    },
    "Ballplaying CB": {
        "score_col": "ballplaying_cb_score",
        "position_hints": ["CB", "DF"],
        "key_stats": [("PrgP_p90", True), ("Cmp%", True), ("PrgC_p90", True)],
    },
    "Aggressive CB": {
        "score_col": "aggressive_cb_score",
        "position_hints": ["CB", "DF"],
        "key_stats": [("Tkl_p90", True), ("Recov_p90", True)],
    },
    "Attacking FB": {
        "score_col": "attacking_fb_score",
        "position_hints": ["FB", "WB", "DF"],
        "key_stats": [("PrgC_p90", True), ("xAG_p90", True)],
    },
    "Defensive FB": {
        "score_col": "defensive_fb_score",
        "position_hints": ["FB", "WB", "DF"],
        "key_stats": [("Tkl_p90", True), ("Int_p90", True)],
    },
    # ── Midfielders ───────────────────────────────────────────────────────
    "Destroyer DM": {
        "score_col": "destroyer_dm_score",
        "position_hints": ["DM", "MF"],
        "key_stats": [("Tkl_p90", True), ("Int_p90", True), ("Recov_p90", True)],
    },
    "Deep-Lying Playmaker": {
        "score_col": "dlp_score",
        "position_hints": ["DM", "MF"],
        "key_stats": [("PrgP_p90", True), ("Cmp%", True)],
    },
    "Box-to-Box CM": {
        "score_col": "b2b_cm_score",
        "position_hints": ["MF", "CM"],
        "key_stats": [("PrgC_p90", True), ("Tkl_p90", True), ("Gls_p90", True)],
    },
    "Creative CM": {
        "score_col": "creative_cm_score",
        "position_hints": ["MF", "CM", "AM"],
        "key_stats": [("xAG_p90", True), ("SCA_p90", True)],
    },
    "Playmaker AM": {
        "score_col": "playmaker_am_score",
        "position_hints": ["AM", "MF"],
        "key_stats": [("xAG_p90", True), ("GCA_p90", True)],
    },
    # ── Attackers ─────────────────────────────────────────────────────────
    "Poacher ST": {
        "score_col": "poacher_st_score",
        "position_hints": ["FW", "ST"],
        "key_stats": [("Gls_p90", True), ("SoT_p90", True)],
    },
    "Target Man ST": {
        "score_col": "targetman_st_score",
        "position_hints": ["FW", "ST"],
        "key_stats": [("Gls_p90", True)],
    },
    "Pressing Forward": {
        "score_col": "pressing_fw_score",
        "position_hints": ["FW", "MF"],
        "key_stats": [("Recov_p90", True), ("Gls_p90", True)],
    },
    "Complete Forward": {
        "score_col": "complete_fw_score",
        "position_hints": ["FW", "ST"],
        "key_stats": [("Gls_p90", True), ("Ast_p90", True), ("xG_p90", True)],
    },
    "Dribbling Winger": {
        "score_col": "dribbling_w_score",
        "position_hints": ["FW", "MF"],
        "key_stats": [("PrgC_p90", True), ("Ast_p90", True)],
    },
    "Crossing Winger": {
        "score_col": "crossing_w_score",
        "position_hints": ["FW", "MF"],
        "key_stats": [("xAG_p90", True), ("Ast_p90", True)],
    },
    "Inside Forward": {
        "score_col": "inside_fw_score",
        "position_hints": ["FW", "MF"],
        "key_stats": [("Gls_p90", True), ("SoT_p90", True), ("npxG_p90", True)],
    },
    # ── Goalkeepers ───────────────────────────────────────────────────────
    "Shot-Stopper GK": {
        "score_col": "shotstopper_gk_score",
        "position_hints": ["GK"],
        "key_stats": [],
    },
    "Sweeper Keeper": {
        "score_col": "sweeper_gk_score",
        "position_hints": ["GK"],
        "key_stats": [],
    },
}

# Reverse lookup: score column → archetype display name
_SCORE_COL_TO_ARCHETYPE: Dict[str, str] = {
    defn["score_col"]: name for name, defn in ARCHETYPE_DEFINITIONS.items()
}


# ---------------------- Recommendation Engine ---------------------- #

class RecommendationEngine:
    """Generate player recommendations based on detected patterns"""
    
    def __init__(self, patterns: Dict, evaluated_players: pd.DataFrame, team_name: str, scoring_method: str = "min-max"):
        self.patterns = patterns
        self.players = evaluated_players
        self.team_name = team_name
        self.scoring_method = scoring_method  # "min-max" or "percentile"
        self.position_mapping = self._create_position_mapping()
        # Populated by generate_recommendations(); maps archetype name → priority weight
        self.archetype_needs: Dict[str, float] = {}
        
    def _create_position_mapping(self) -> Dict:
        """Map pattern needs to player evaluation scores - now with 19 specialized archetypes"""
        return {
            # === DEFENDER MAPPING (using specialized archetypes) ===
            "CB_pace": ["aggressive_cb_score", "stopper_cb_score", "defender_comprehensive", "stamina_score"],
            "CB_physical": ["stopper_cb_score", "aggressive_cb_score", "defender_comprehensive", "aerial_score"],
            "CB_leadership": ["stopper_cb_score", "ballplaying_cb_score", "defender_comprehensive", "experience_score"],
            "CB_ballplaying": ["ballplaying_cb_score", "defender_comprehensive"],
            "CB_aerial": ["stopper_cb_score", "defender_comprehensive", "aerial_score"],
            "CB_concentration": ["stopper_cb_score", "ballplaying_cb_score", "defender_comprehensive", "discipline_score"],
            "CB_experienced": ["ballplaying_cb_score", "stopper_cb_score", "defender_comprehensive", "experience_score"],
            "CB_calm": ["ballplaying_cb_score", "stopper_cb_score", "defender_comprehensive", "discipline_score"],
            "CB_blocking": ["stopper_cb_score", "aggressive_cb_score", "defender_comprehensive"],
            "CB_setpiece": ["stopper_cb_score", "defender_comprehensive", "aerial_score"],
            "CB_elite": ["ballplaying_cb_score", "stopper_cb_score", "defender_comprehensive", "experience_score"],
            "CB_disciplined": ["stopper_cb_score", "ballplaying_cb_score", "defender_comprehensive", "discipline_score"],
            "CB_reliable": ["stopper_cb_score", "ballplaying_cb_score", "defender_comprehensive", "consistency_score"],
            
            # Fullback positions (specialized)
            "FB_recovery": ["defensive_fb_score", "attacking_fb_score", "defender_comprehensive", "stamina_score"],
            "FB_defensive": ["defensive_fb_score", "defender_comprehensive", "discipline_score"],
            "FB_attacking": ["attacking_fb_score", "midfielder_comprehensive", "crossing_w_score"],
            "FB_fitness": ["attacking_fb_score", "defensive_fb_score", "defender_comprehensive", "stamina_score"],
            "FB_tactical": ["defensive_fb_score", "attacking_fb_score", "defender_comprehensive", "discipline_score"],
            "FB_composed": ["defensive_fb_score", "attacking_fb_score", "defender_comprehensive", "discipline_score"],
            
            # === MIDFIELDER MAPPING (using specialized archetypes) ===
            "DM_defensive": ["destroyer_dm_score", "dlp_score", "midfielder_comprehensive"],
            "DM_pressing": ["destroyer_dm_score", "aggressive_cb_score", "midfielder_comprehensive"],
            "DM_progressive": ["dlp_score", "b2b_cm_score", "midfielder_comprehensive"],
            "DM_tactical": ["destroyer_dm_score", "dlp_score", "midfielder_comprehensive", "experience_score"],
            "DM_experienced": ["dlp_score", "destroyer_dm_score", "midfielder_comprehensive", "experience_score"],
            "DM_height": ["destroyer_dm_score", "midfielder_comprehensive", "aerial_score"],
            "DM_workrate": ["destroyer_dm_score", "b2b_cm_score", "midfielder_comprehensive"],
            "DM_disciplined": ["destroyer_dm_score", "dlp_score", "midfielder_comprehensive", "discipline_score"],
            "DM_calm": ["dlp_score", "destroyer_dm_score", "midfielder_comprehensive", "discipline_score"],
            "DM_strength": ["destroyer_dm_score", "midfielder_comprehensive", "aerial_score"],
            
            # Central midfielder positions (specialized)
            "CM_workrate": ["b2b_cm_score", "midfielder_comprehensive", "workrate_score"],
            "CM_passing": ["dlp_score", "creative_cm_score", "midfielder_comprehensive"],
            "CM_leadership": ["b2b_cm_score", "creative_cm_score", "midfielder_comprehensive", "experience_score"],
            "CM_stamina": ["b2b_cm_score", "midfielder_comprehensive", "stamina_score"],
            "CM_bigmatch": ["b2b_cm_score", "creative_cm_score", "midfielder_comprehensive", "experience_score"],
            "CM_experienced": ["dlp_score", "b2b_cm_score", "midfielder_comprehensive", "experience_score"],
            "CM_composed": ["creative_cm_score", "b2b_cm_score", "midfielder_comprehensive", "discipline_score"],
            "CM_consistent": ["b2b_cm_score", "creative_cm_score", "midfielder_comprehensive", "consistency_score"],
            "CM_delivery": ["creative_cm_score", "dlp_score", "midfielder_comprehensive"],
            
            # Attacking midfielder positions (specialized)
            "AM_creativity": ["playmaker_am_score", "creative_cm_score", "midfielder_comprehensive"],
            "AM_creative": ["playmaker_am_score", "creative_cm_score", "midfielder_comprehensive"],
            "AM_longshots": ["playmaker_am_score", "inside_fw_score", "attacker_comprehensive"],
            
            # === ATTACKER MAPPING (using specialized archetypes) ===
            "W_goals": ["inside_fw_score", "attacker_comprehensive", "complete_fw_score"],
            "W_dribbling": ["dribbling_w_score", "attacker_comprehensive"],
            "W_shooting": ["inside_fw_score", "complete_fw_score", "attacker_comprehensive"],
            "W_pace": ["dribbling_w_score", "inside_fw_score", "attacker_comprehensive"],
            "W_gamechanging": ["inside_fw_score", "complete_fw_score", "dribbling_w_score", "attacker_comprehensive"],
            
            # Striker positions (specialized)
            "ST_finishing": ["poacher_st_score", "complete_fw_score", "attacker_comprehensive"],
            "ST_clinical": ["poacher_st_score", "inside_fw_score", "attacker_comprehensive", "consistency_score"],
            "ST_holdup": ["targetman_st_score", "complete_fw_score", "attacker_comprehensive"],
            "ST_consistent": ["poacher_st_score", "complete_fw_score", "attacker_comprehensive", "consistency_score"],
            "ST_aerial": ["targetman_st_score", "poacher_st_score", "attacker_comprehensive", "aerial_score"],
            "ST_impactful": ["complete_fw_score", "poacher_st_score", "pressing_fw_score", "attacker_comprehensive"],
            "ST_pressing": ["pressing_fw_score", "complete_fw_score", "attacker_comprehensive"],
            
            # === GOALKEEPER (specialized) ===
            "GK": ["shotstopper_gk_score", "sweeper_gk_score"],
            "GK_commanding": ["shotstopper_gk_score", "sweeper_gk_score", "experience_score"],
            "GK_shotstopper": ["shotstopper_gk_score"],
            "GK_presence": ["sweeper_gk_score", "shotstopper_gk_score", "experience_score"],
            
            # === BROAD FALLBACKS ===
            "Defender": ["defender_comprehensive", "versatile_score"],
            "Midfielder": ["midfielder_comprehensive", "versatile_score"],
            "Attacker": ["attacker_comprehensive", "versatile_score"],
            "Versatile": ["versatile_score"],
        }

    def _need_prefix_to_flag(self, need_key: str) -> str:
        prefix = need_key.split("_", 1)[0] if "_" in need_key else need_key
        mapping = {
            "GK": "is_gk",
            "CB": "is_cb",
            "FB": "is_fb",
            "DM": "is_dm",
            "CM": "is_cm",
            "AM": "is_am",
            "W":  "is_winger",
            "ST": "is_striker",
        }
        return mapping.get(prefix, "")

    def _apply_age_filter(
        self, df: pd.DataFrame, age_min: Optional[int], age_max: Optional[int]
    ) -> pd.DataFrame:
        """Filter players by age range if provided."""
        if (age_min is None) and (age_max is None):
            return df
        if "Age" not in df.columns:
            return df

        age_num = pd.to_numeric(df["Age"].astype(str).str.extract(r'(\d+)', expand=False), errors='coerce')
        mask = pd.Series(True, index=df.index)
        if age_min is not None:
            mask &= age_num >= age_min
        if age_max is not None:
            mask &= age_num <= age_max
        return df[mask].copy()

    def _collect_needed_positions(self) -> Dict[str, float]:
        """Collect and weight all needed positions from patterns."""
        position_weights: Dict[str, float] = {}
        for category, analysis in self.patterns.items():
            if "issues" in analysis:
                for issue in analysis["issues"]:
                    severity_weight = {"high": 1.0, "medium": 0.6, "low": 0.3}.get(
                        issue.get("severity", "medium"), 0.5
                    )
                    for pos in issue.get("positions_needed", []):
                        position_weights[pos] = position_weights.get(pos, 0.0) + severity_weight
        total = sum(position_weights.values())
        if total > 0:
            for pos in position_weights:
                position_weights[pos] /= total
        return position_weights

    def _collect_archetype_needs(self, position_weights: Dict[str, float]) -> Dict[str, float]:
        """Derive archetype priority weights from position_weights.

        For each need_key (e.g. "CB_pace") its mapped score columns are looked up in
        ARCHETYPE_DEFINITIONS.  The accumulated weight is spread across all
        matching archetypes, then normalised to sum to 1.0.
        """
        archetype_weights: Dict[str, float] = {}
        for need_key, weight in position_weights.items():
            score_cols = self.position_mapping.get(need_key)
            if not isinstance(score_cols, list):
                continue
            for score_col in score_cols:
                arch_name = _SCORE_COL_TO_ARCHETYPE.get(score_col)
                if arch_name:
                    archetype_weights[arch_name] = archetype_weights.get(arch_name, 0.0) + weight

        total = sum(archetype_weights.values())
        if total > 0:
            for k in archetype_weights:
                archetype_weights[k] /= total
        return archetype_weights

    @staticmethod
    def _pos_matches_hints(pos_str: str, position_hints: List[str]) -> bool:
        """Return True if the player's Pos value contains any of the position_hints substrings.

        Examples
        --------
        "DF,MF" matches hints ["MF", "CM"] → True
        "DF"    matches hints ["MF", "CM"] → False  (blocks a CB from a CM archetype slot)
        No position data → True (don't exclude unknowns).
        """
        if not position_hints:
            return True
        pos_upper = str(pos_str).upper()
        if not pos_upper or pos_upper == "NAN":
            return True  # unknown position — don't exclude
        return any(hint.upper() in pos_upper for hint in position_hints)

    def _get_best_archetype_for_player(self, player: "pd.Series") -> str:
        """Return the archetype name (from archetype_needs) where this player
        scores highest, restricted to archetypes compatible with the player's
        actual position.  Falls back to '—' when nothing matches.
        """
        pos_str = str(player.get("Pos", ""))
        best_name, best_val = "—", -1.0
        for arch_name, weight in self.archetype_needs.items():
            if weight < 0.01:
                continue
            defn = ARCHETYPE_DEFINITIONS.get(arch_name)
            if not defn:
                continue
            # Hard gate: skip archetypes incompatible with this player's position
            if not self._pos_matches_hints(pos_str, defn.get("position_hints", [])):
                continue
            score_col = defn["score_col"]
            val = float(player.get(score_col, 0) or 0)
            if val > best_val:
                best_val, best_name = val, arch_name
        return best_name

    def get_candidates_for_archetype(
        self, archetype: str, player_pool_df: "pd.DataFrame", top_n: int = 20
    ) -> "pd.DataFrame":
        """Return players from *player_pool_df* sorted by the archetype's score column,
        restricted to players whose Pos field matches the archetype's position_hints.
        Falls back to the full pool if no position-compatible players are found.
        """
        defn = ARCHETYPE_DEFINITIONS.get(archetype)
        if not defn:
            print(f"⚠️  get_candidates_for_archetype: unknown archetype '{archetype}'")
            return player_pool_df.head(top_n)
        score_col = defn["score_col"]
        if score_col not in player_pool_df.columns:
            print(f"⚠️  get_candidates_for_archetype: score column '{score_col}' not found")
            return player_pool_df.head(top_n)

        position_hints = defn.get("position_hints", [])
        if position_hints and "Pos" in player_pool_df.columns:
            pos_mask = player_pool_df["Pos"].fillna("").apply(
                lambda p: self._pos_matches_hints(p, position_hints)
            )
            pool = player_pool_df[pos_mask] if pos_mask.any() else player_pool_df
        else:
            pool = player_pool_df

        return pool.sort_values(score_col, ascending=False).head(top_n).copy()

    def get_top_candidates_by_archetype(
        self,
        archetypes: List[str],
        player_pool_df: "pd.DataFrame",
        top_n: int = 15,
    ) -> Dict[str, "pd.DataFrame"]:
        """Return {archetype_name: top_candidates_df} for each requested archetype.

        Designed for the 'All archetypes' grouped view so no candidates are
        missed due to position-weight dilution.
        """
        result: Dict[str, "pd.DataFrame"] = {}
        for arch in archetypes:
            candidates = self.get_candidates_for_archetype(arch, player_pool_df, top_n=top_n)
            if not candidates.empty:
                result[arch] = candidates
        return result

    def _score_players_for_needs(
        self,
        needed_positions: Dict[str, float],
        players_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """Score each player against the weighted team needs."""
        player_scores = (players_df if players_df is not None else self.players).copy()
        if player_scores.empty:
            player_scores["recommendation_score"] = 0.0
            player_scores["matched_needs"] = ""
            return player_scores

        player_scores["recommendation_score"] = 0.0
        player_scores["matched_needs"] = ""

        DEFAULT_FIELDS = {
            "GK": ["gk_score"],
            "CB": ["cb_score"],
            "FB": ["fb_score"],
            "DM": ["dm_score"],
            "CM": ["cm_score"],
            "AM": ["am_score"],
            "W":  ["w_score"],
            "ST": ["st_score"],
        }

        for idx, player in player_scores.iterrows():
            score = 0.0
            matched: List[str] = []

            for position, weight in needed_positions.items():
                # Soft position signal: players who naturally play the role get a
                # small bonus, but nobody is excluded purely on position.
                # Archetype scores (e.g. stopper_cb_score) already encode positional
                # fitness — a striker will naturally score near-0 on CB archetypes.
                flag_col = self._need_prefix_to_flag(position)
                position_bonus = 0.05 if (flag_col and bool(player.get(flag_col, False))) else 0.0

                entry = self.position_mapping.get(position)
                if isinstance(entry, list):
                    fields = entry
                elif isinstance(entry, dict):
                    fields = entry.get("scores", [])
                else:
                    prefix = position.split("_", 1)[0] if "_" in position else position
                    fields = DEFAULT_FIELDS.get(prefix, [])

                if not fields:
                    continue

                vals = [player.get(f, 0) for f in fields]
                position_score = float(np.nanmean(vals)) + position_bonus if len(vals) else 0.0
                # Threshold: only meaningful archetype scores contribute.
                # Near-0 scores (e.g. a striker on CB metrics) are naturally filtered out.
                if position_score > 0.45:
                    score += position_score * weight
                    matched.append(position.replace("_", " "))

            player_scores.at[idx, "recommendation_score"] = score
            player_scores.at[idx, "matched_needs"] = ", ".join(matched[:3])

        np.random.seed(abs(hash(self.team_name)) % (2**32))
        player_scores["recommendation_score"] += np.random.normal(0, 0.02, len(player_scores))
        return player_scores

    def _select_best_players(self, player_scores: pd.DataFrame, needed_positions: Dict[str, float], top_n: int) -> pd.DataFrame:
        """Select best players ensuring position diversity."""
        position_groups = {
            "GK": ["GK"],
            "CB": ["CB_"],
            "FB": ["FB_"],
            "DM": ["DM_"],
            "CM": ["CM_"],
            "AM": ["AM_"],
            "W":  ["W_"],
            "ST": ["ST_"],
        }

        quotas = {}
        for group, prefixes in position_groups.items():
            group_weight = sum(
                weight for pos, weight in needed_positions.items()
                if any(pos.startswith(prefix) for prefix in prefixes)
            )
            quotas[group] = max(1, int(top_n * group_weight)) if group_weight > 0 else 0

        quotas["GK"] = max(quotas.get("GK", 0), 2)
        quotas["CB"] = max(quotas.get("CB", 0), 3)
        quotas["FB"] = max(quotas.get("FB", 0), 2)
        quotas["DM"] = max(quotas.get("DM", 0), 2)

        selected = []
        for group, quota in quotas.items():
            if quota <= 0:
                continue
            if group == "GK":
                candidates = player_scores[player_scores["is_gk"]]
            elif group == "CB":
                candidates = player_scores[player_scores["is_cb"]]
            elif group == "FB":
                candidates = player_scores[player_scores["is_fb"]]
            elif group == "DM":
                candidates = player_scores[player_scores["is_dm"]]
            elif group == "CM":
                candidates = player_scores[player_scores["is_cm"]]
            elif group == "AM":
                candidates = player_scores[player_scores["is_am"]]
            elif group == "W":
                candidates = player_scores[player_scores["is_winger"]]
            elif group == "ST":
                candidates = player_scores[player_scores["is_striker"]]
            else:
                continue

            candidates = candidates.sort_values("recommendation_score", ascending=False).head(quota)
            selected.append(candidates)

        if selected:
            recommendations = pd.concat(selected, ignore_index=True)
            key_cols = ["Player"] if "Player" in recommendations.columns else [recommendations.columns[0]]
            recommendations = recommendations.drop_duplicates(subset=key_cols)
            recommendations = recommendations.sort_values("recommendation_score", ascending=False).head(top_n)
        else:
            recommendations = player_scores.sort_values("recommendation_score", ascending=False).head(top_n)

        return recommendations

    def _select_best_players_by_archetype(
        self, player_scores: pd.DataFrame, top_n: int
    ) -> pd.DataFrame:
        """Select best players guaranteeing at least one per needed archetype.

        Two-phase selection:

        Phase 1 – Guaranteed slots (1 per archetype, processed high-weight first).
            For each archetype in archetype_needs, pick the highest scorer on that
            archetype's score column who has not yet been reserved.  This ensures
            every archetype gets a representative regardless of top_n or weight.

        Phase 2 – Fill remaining slots.
            From the players not already guaranteed, take the highest-scoring by
            recommendation_score until the total reaches top_n.

        The two pools are concatenated and sorted by recommendation_score for display.
        """
        if not self.archetype_needs:
            return player_scores.sort_values("recommendation_score", ascending=False).head(top_n)

        key_col = "Player" if "Player" in player_scores.columns else player_scores.columns[0]

        # ── Phase 1: one guaranteed pick per archetype ──────────────────────
        guaranteed_indices: List = []      # positional indices into player_scores
        used_player_names: set = set()

        for arch_name, _weight in sorted(self.archetype_needs.items(), key=lambda x: -x[1]):
            defn = ARCHETYPE_DEFINITIONS.get(arch_name)
            if not defn:
                print(f"⚠️  [RecommendationEngine] Unknown archetype '{arch_name}' – skipping.")
                continue
            score_col = defn["score_col"]
            if score_col not in player_scores.columns:
                continue

            # Restrict to position-compatible players for this archetype's guaranteed slot.
            # Fall back to the full pool only if no position-compatible players exist.
            position_hints = defn.get("position_hints", [])
            if position_hints and "Pos" in player_scores.columns:
                pos_mask = player_scores["Pos"].fillna("").apply(
                    lambda p: self._pos_matches_hints(p, position_hints)
                )
                ranked_pool = player_scores[pos_mask] if pos_mask.any() else player_scores
            else:
                ranked_pool = player_scores

            ranked = ranked_pool.sort_values(score_col, ascending=False)
            for idx, row in ranked.iterrows():
                name = row.get(key_col, idx)
                if name not in used_player_names:
                    guaranteed_indices.append(idx)
                    used_player_names.add(name)
                    break   # one guaranteed pick per archetype

        guaranteed_df = player_scores.loc[guaranteed_indices] if guaranteed_indices else pd.DataFrame(columns=player_scores.columns)

        # ── Phase 2: fill remaining slots by recommendation_score ───────────
        remaining_budget = max(0, top_n - len(guaranteed_indices))
        if remaining_budget > 0:
            fill_pool = player_scores[~player_scores[key_col].isin(used_player_names)]
            fill_df = fill_pool.sort_values("recommendation_score", ascending=False).head(remaining_budget)
        else:
            fill_df = pd.DataFrame(columns=player_scores.columns)

        # ── Combine and sort ────────────────────────────────────────────────
        result = pd.concat([guaranteed_df, fill_df], ignore_index=True)
        result = result.drop_duplicates(subset=[key_col])
        result = result.sort_values("recommendation_score", ascending=False)
        return result

    def generate_recommendations(
        self,
        top_n: int = 20,
        age_min: Optional[int] = None,
        age_max: Optional[int] = None,
        preferred_foot: Optional[str] = None,
        free_agents_only: bool = False,
        max_budget: Optional[float] = None,
    ) -> pd.DataFrame:
        """Generate player recommendations based on patterns (optionally age-filtered)."""
        needed_positions = self._collect_needed_positions()
        # Derive archetype priorities; stored on self so the UI can access them
        self.archetype_needs = self._collect_archetype_needs(needed_positions)
        
        # Filter out current team's players if 'Squad' column exists
        candidates = self.players
        if 'Squad' in candidates.columns:
            candidates = candidates[candidates['Squad'] != self.team_name]
        
        # Age filter
        candidates = self._apply_age_filter(candidates, age_min, age_max)
        
        # Free agents filter (contract expires 30/06/2026)
        if free_agents_only and 'contract_expires' in candidates.columns:
            candidates = candidates[
                candidates['contract_expires'].astype(str).str.contains('30/06/2026', na=False)
            ].copy()
        
        # Budget filter (market value) — accept both canonical and lowercase column names
        _mv_col = next((c for c in ('market_value', 'Market Value', 'Value', 'value')
                        if c in candidates.columns), None)
        if max_budget is not None and max_budget > 0 and _mv_col is not None:
            market_values = pd.to_numeric(candidates[_mv_col], errors='coerce')
            candidates = candidates[market_values <= max_budget].copy()
        
        player_scores = self._score_players_for_needs(needed_positions, players_df=candidates)
        
        # Preferred foot boost (+0.10)
        if preferred_foot is not None:
            pf = preferred_foot.strip().lower()
            if pf in ["right", "left"]:
                if "foot" in player_scores.columns:
                    foot_series = (
                        player_scores["foot"]
                        .astype(str)
                        .str.lower()
                        .fillna("")
                    )
                    if pf == "right":
                        mask = foot_series.str.contains("right", na=False)
                    else:
                        mask = foot_series.str.contains("left", na=False)

                    player_scores.loc[mask, "recommendation_score"] += 0.10

        # Archetype-first selection: quota per archetype replaces position buckets
        recommendations = self._select_best_players_by_archetype(player_scores, top_n)
        recommendations = self._add_recommendation_reasoning(recommendations)
        return recommendations
        
    def _add_recommendation_reasoning(self, recommendations: pd.DataFrame) -> pd.DataFrame:
        """Add reasoning for each recommendation."""
        issue_descriptions: List[str] = []
        for category, analysis in self.patterns.items():
            if "issues" in analysis:
                for issue in analysis["issues"]:
                    if issue.get("severity") == "high":
                        issue_descriptions.append(issue.get("description", ""))

        recommendations["team_issues"] = (
            "; ".join(issue_descriptions[:3]) if issue_descriptions else "General team improvement"
        )

        # Tag each player with their best-matching archetype from archetype_needs
        recommendations["Archetype"] = recommendations.apply(
            self._get_best_archetype_for_player, axis=1
        )

        # Core columns to always include
        columns_to_keep = [
            "Player", "Nation", "Pos", "Squad", "Comp", "Age", "90s",
            "Archetype", "matched_needs", "recommendation_score", "team_issues"
        ]
        
        # Add optional columns if they exist (foot, contract, value)
        optional_columns = [
            "foot", "Foot", "preferred_foot",  # Foot variations
            "contract_expires", "Contract", "Contract Expires", "contract",  # Contract variations
            "market_value", "Market Value", "Value", "Transfer Value", "value",  # Value variations
            "player_image_url", "image_url", "photo_url", "image"  # Image URL variations
        ]
        
        for col in optional_columns:
            if col in recommendations.columns and col not in columns_to_keep:
                columns_to_keep.append(col)
        
        available = [c for c in columns_to_keep if c in recommendations.columns]
        recommendations = recommendations[available].copy()

        recommendations["recommendation_score"] = recommendations["recommendation_score"].round(3)
        recommendations = recommendations.rename(columns={
            "matched_needs": "Addresses",
            "recommendation_score": "Score",
            "team_issues": "Team_Weaknesses",
        })
        return recommendations


def get_team_statistics(df_team: pd.DataFrame) -> Dict:
    """Calculate basic team statistics"""
    total_games = len(df_team)
    wins = len(df_team[df_team["Result"] == "W"])
    draws = len(df_team[df_team["Result"] == "D"])
    losses = len(df_team[df_team["Result"] == "L"])
    
    return {
        "total_games": total_games,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "win_rate": wins / total_games if total_games > 0 else 0
    }


def get_all_issues(patterns: Dict) -> List[Dict]:
    """Collect and number all issues from the analysis."""
    all_issues = []
    issue_count = 0
    for category, analysis in patterns.items():
        if "issues" in analysis and analysis["issues"]:
            for issue in analysis["issues"]:
                issue_count += 1
                all_issues.append({
                    'number': issue_count,
                    'category': category,
                    'severity': issue.get("severity", "medium"),
                    'description': issue.get('description', 'Unknown issue'),
                    'positions_needed': issue.get('positions_needed', []),
                    'full_issue': issue
                })
    return all_issues


def calculate_betting_odds_analysis(df_team):
    """
    Calculate expected vs actual results based on betting odds
    Uses probability method (1/odds) to get fair expected values
    """
    analysis = {
        'expected_wins': 0,
        'expected_draws': 0,
        'expected_losses': 0,
        'actual_wins': 0,
        'actual_draws': 0,
        'actual_losses': 0,
        'has_odds_data': False
    }
    
    # Check if odds columns exist
    if not {'OddHome', 'OddDraw', 'OddAway'}.issubset(df_team.columns):
        return analysis
    
    # Filter games with valid odds data
    df_with_odds = df_team[
        df_team['OddHome'].notna() & 
        df_team['OddDraw'].notna() & 
        df_team['OddAway'].notna()
    ].copy()
    
    if len(df_with_odds) == 0:
        return analysis
    
    analysis['has_odds_data'] = True
    
    # Convert odds to probabilities and calculate expected results
    for idx, row in df_with_odds.iterrows():
        is_home = row.get('is_home', False)
        
        # Get odds
        odd_home = float(row['OddHome'])
        odd_draw = float(row['OddDraw'])
        odd_away = float(row['OddAway'])
        
        # Convert to probabilities (implied probability = 1 / odds)
        prob_home = 1 / odd_home if odd_home > 0 else 0
        prob_draw = 1 / odd_draw if odd_draw > 0 else 0
        prob_away = 1 / odd_away if odd_away > 0 else 0
        
        # Normalize probabilities (remove bookmaker margin)
        total = prob_home + prob_draw + prob_away
        if total > 0:
            prob_home /= total
            prob_draw /= total
            prob_away /= total
        
        # Add to expected results based on team perspective
        if is_home:
            analysis['expected_wins'] += prob_home
            analysis['expected_draws'] += prob_draw
            analysis['expected_losses'] += prob_away
        else:
            analysis['expected_wins'] += prob_away
            analysis['expected_draws'] += prob_draw
            analysis['expected_losses'] += prob_home
        
        # Count actual results
        result = row['Result']
        if result == 'W':
            analysis['actual_wins'] += 1
        elif result == 'D':
            analysis['actual_draws'] += 1
        elif result == 'L':
            analysis['actual_losses'] += 1
    
    # Keep as float for more precise comparison
    # But round for display
    analysis['expected_wins'] = round(analysis['expected_wins'], 1)
    analysis['expected_draws'] = round(analysis['expected_draws'], 1)
    analysis['expected_losses'] = round(analysis['expected_losses'], 1)
    
    # Calculate differences
    analysis['wins_diff'] = analysis['actual_wins'] - analysis['expected_wins']
    analysis['draws_diff'] = analysis['actual_draws'] - analysis['expected_draws']
    analysis['losses_diff'] = analysis['actual_losses'] - analysis['expected_losses']
    
    # Overall performance assessment - STRICTER THRESHOLDS
    # Use percentage-based assessment for fairness
    total_games = len(df_with_odds)
    win_diff_pct = (analysis['wins_diff'] / total_games * 100) if total_games > 0 else 0
    
    if win_diff_pct > 8:  # More than 8% better than expected (e.g., 3+ wins in 38 games)
        analysis['odds_performance'] = "Beating expectations!"
    elif win_diff_pct < -8:  # More than 8% worse than expected
        analysis['odds_performance'] = "Below expectations"
    else:
        analysis['odds_performance'] = "Meeting expectationsâ€œ"
    
    return analysis