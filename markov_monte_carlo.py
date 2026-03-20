"""
markov_monte_carlo_enhanced.py
================================
TRUE MARKOV CHAIN MONTE CARLO Simulation for Football Player Performance

WHAT'S NEW (TRUE MARKOV FEATURES):
-----------------------------------
1. ✅ STATE MEMORY: Tracks consecutive games in each state (momentum)
2. ✅ ADAPTIVE TRANSITIONS: Matrix changes based on streaks and fatigue
3. ✅ SMART INJURY SYSTEM: Age/position/fatigue/state-dependent injury risk
4. ✅ INJURY DURATION: Realistic recovery times based on age and injury type
5. ✅ INITIAL STATE ESTIMATION: Uses real season stats to set starting form
6. ✅ FATIGUE ACCUMULATION: Late-season injury risk increases
7. ✅ RECOVERY DYNAMICS: Multi-game recovery with re-injury risk
8. ✅ SUSPENSION TRACKING: Card accumulation → automatic bans

This is NOW a true Markov Chain where:
- Game N+1 performance DEPENDS ON Game N (memory)
- Transitions ADAPT based on recent history (momentum)
- Injury risk ACCUMULATES over season (fatigue)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from enum import Enum
from dataclasses import dataclass
from scipy.stats import norm, nbinom, poisson


# ======================== IMPORTANCE / ROLE PROFILES ========================

IMPORTANCE_PROFILES: Dict[str, Dict] = {
    "Starter — Full 90": {
        "avg_minutes": 90,
        "minutes_std":  5,
        "selection_prob": 1.00,
        "perf_multiplier": 1.00,
        "description": "Starts and plays the full game every match",
    },
    "Starter — Subbed Off": {
        "avg_minutes": 75,
        "minutes_std": 10,
        "selection_prob": 1.00,
        "perf_multiplier": 1.00,
        "description": "Regular starter, withdrawn around 70-80 min",
    },
    "Rotation Player": {
        "avg_minutes": 60,
        "minutes_std": 18,
        "selection_prob": 0.80,
        "perf_multiplier": 0.95,
        "description": "Competes for a starting spot; benched ~20% of games",
    },
    "Super-Sub": {
        "avg_minutes": 30,
        "minutes_std": 12,
        "selection_prob": 0.75,
        "perf_multiplier": 1.20,
        "description": "Impact sub — comes on in ~75% of games (+20% per-90 boost)",
    },
    "Fringe / Cup Player": {
        "avg_minutes": 20,
        "minutes_std":  8,
        "selection_prob": 0.45,
        "perf_multiplier": 1.05,
        "description": "Rare appearances; motivated when selected",
    },
}


# ======================== STATE DEFINITIONS ========================

class PerformanceState(Enum):
    """Player's performance state (Markov states)"""
    HOT = "hot"              # Excellent recent form
    GOOD = "good"            # Normal performance
    COLD = "cold"            # Poor recent form
    INJURED = "injured"      # Cannot play - injury
    SUSPENDED = "suspended"  # Cannot play - cards
    RECOVERY = "recovery"    # Just returned from injury


class Position(Enum):
    """Player positions"""
    GK = "Goalkeeper"
    CB = "Center Back"
    FB = "Fullback"
    DM = "Defensive Mid"
    CM = "Central Mid"
    AM = "Attacking Mid"
    W = "Winger"
    ST = "Striker"


@dataclass
class StateMultipliers:
    """Performance multipliers for each state"""
    goals: float
    assists: float
    defensive: float
    progressive: float  # Progressive carries / passes
    injury_risk: float
    card_risk: float


# State effect multipliers
STATE_EFFECTS = {
    PerformanceState.HOT: StateMultipliers(
        goals=1.35,        # 35% boost to scoring
        assists=1.30,      # 30% boost to assists
        defensive=1.20,    # 20% boost to defensive actions
        progressive=1.25,  # 25% boost to progressive actions
        injury_risk=1.15,  # Slight increase (playing more minutes)
        card_risk=0.90     # Better discipline when confident
    ),
    PerformanceState.GOOD: StateMultipliers(
        goals=1.00,        # Baseline
        assists=1.00,
        defensive=1.00,
        progressive=1.00,
        injury_risk=1.00,
        card_risk=1.00
    ),
    PerformanceState.COLD: StateMultipliers(
        goals=0.75,        # 25% penalty to scoring
        assists=0.80,      # 20% penalty to assists
        defensive=0.85,    # 15% penalty to defensive actions
        progressive=0.80,  # 20% fewer progressive actions
        injury_risk=1.15,
        card_risk=1.15
    ),
    PerformanceState.RECOVERY: StateMultipliers(
        goals=0.85,        # Still recovering fitness
        assists=0.90,
        defensive=0.90,
        progressive=0.85,  # Cautious in possession
        injury_risk=1.30,  # High re-injury risk
        card_risk=1.00
    ),
    PerformanceState.INJURED: StateMultipliers(
        goals=0.0, assists=0.0, defensive=0.0, progressive=0.0,
        injury_risk=0.0, card_risk=0.0
    ),
    PerformanceState.SUSPENDED: StateMultipliers(
        goals=0.0, assists=0.0, defensive=0.0, progressive=0.0,
        injury_risk=0.0, card_risk=0.0
    ),
}


# ======================== MARKOV CHAIN SIMULATOR ========================

class MonteCarloSimulator:
    """
    TRUE Markov Chain Monte Carlo simulator for football player performance.
    
    KEY INNOVATIONS:
    ----------------
    1. State-dependent transitions (form momentum)
    2. Adaptive transition probabilities (streaks matter)
    3. Smart injury system (age/position/fatigue)
    4. Memory-based initial states (real data informs start)
    5. Fatigue accumulation (late season effects)
    """
    
    def __init__(self, player_data: pd.Series, n_simulations: int = 1000,
                 importance: str = "Starter — Full 90"):
        """
        Initialize TRUE Markov Chain simulator.

        Args:
            player_data:   Player's stats from DataFrame row
            n_simulations: Number of seasons to simulate
            importance:    Key from IMPORTANCE_PROFILES describing the player's role
        """
        self.player = player_data
        self.n_sims = n_simulations
        self.position = self._determine_position()
        self.age = int(player_data.get('Age', 25))

        # Role / importance profile
        profile = IMPORTANCE_PROFILES.get(importance, IMPORTANCE_PROFILES["Starter — Full 90"])
        self.avg_minutes    = float(profile["avg_minutes"])
        self.minutes_std    = float(profile["minutes_std"])
        self.selection_prob = float(profile["selection_prob"])
        self.perf_multiplier = float(profile["perf_multiplier"])
        self.importance_label = importance

        # Extract player's real per-game statistics (with role boost already applied)
        self.real_stats = self._extract_real_stats()

        # Calculate player durability from real data
        self.durability = self._calculate_durability()

        # Build position-specific BASE transition matrix
        self.base_transition_matrix = self._build_base_transition_matrix()

        # Initial state distribution (SMART - uses real data!)
        self.initial_state_probs = self._get_initial_state_distribution()
    
    def _determine_position(self) -> Position:
        """Determine player's primary position"""
        if self.player.get('is_gk', False):
            return Position.GK
        elif self.player.get('is_cb', False):
            return Position.CB
        elif self.player.get('is_fb', False):
            return Position.FB
        elif self.player.get('is_dm', False):
            return Position.DM
        elif self.player.get('is_cm', False):
            return Position.CM
        elif self.player.get('is_am', False):
            return Position.AM
        elif self.player.get('is_winger', False):
            return Position.W
        elif self.player.get('is_striker', False):
            return Position.ST
        else:
            return Position.CM  # Default
    
    def _extract_real_stats(self) -> Dict:
        """Extract player's real per-90 statistics, adjusted by role perf_multiplier."""
        ninety_min_games = float(self.player.get('90s', 1))
        if ninety_min_games < 0.1:
            ninety_min_games = 1.0

        m = self.perf_multiplier  # role-based multiplier (e.g. 1.20 for Super-Sub)

        return {
            'goals_per90':        float(self.player.get('Gls', 0)) / ninety_min_games * m,
            'assists_per90':      float(self.player.get('Ast', 0)) / ninety_min_games * m,
            'shots_per90':        float(self.player.get('Sh',  0)) / ninety_min_games * m,
            'tackles_per90':      float(self.player.get('Tkl', 0)) / ninety_min_games * m,
            'interceptions_per90':float(self.player.get('Int', 0)) / ninety_min_games * m,
            'clearances_per90':   float(self.player.get('Clr', 0)) / ninety_min_games * m,
            'blocks_per90':       float(self.player.get('Blocks', self.player.get('Blocks_stats_defense', 0))) / ninety_min_games * m,
            'prgc_per90':         float(self.player.get('PrgC', 0)) / ninety_min_games * m,
            'prgp_per90':         float(self.player.get('PrgP', 0)) / ninety_min_games * m,
            # Cards scale with minutes, not intensity — no perf_multiplier applied
            'yellows_per90':      float(self.player.get('CrdY', 0)) / ninety_min_games,
            'reds_per90':         float(self.player.get('CrdR', 0)) / ninety_min_games,
            'saves_per90':        float(self.player.get('Saves', 0)) / ninety_min_games * m if self.position == Position.GK else 0,
            'ga_per90':           float(self.player.get('GA',    0)) / ninety_min_games     if self.position == Position.GK else 0,
            'games_played': ninety_min_games,
        }
    
    def _calculate_durability(self) -> float:
        """
        Calculate player's durability score from real data.

        Normalises raw '90s' back to estimated appearances so that a Super-Sub
        with only 8 raw '90s' (= 24 × 30-min appearances) is not misread as
        injury-prone.

        Returns: 0.0 (injury-prone) to 1.0 (iron man)
        """
        raw_90s = self.real_stats['games_played']
        # Convert 90s back to approximate number of appearances for this role
        minutes_fraction = max(self.avg_minutes / 90.0, 0.10)
        estimated_appearances = raw_90s / minutes_fraction

        if estimated_appearances >= 32:
            return 1.0
        elif estimated_appearances >= 25:
            return 0.85
        elif estimated_appearances >= 20:
            return 0.70
        elif estimated_appearances >= 15:
            return 0.55
        else:
            return 0.40
    
    def _build_base_transition_matrix(self) -> Dict[PerformanceState, Dict[PerformanceState, float]]:
        """
        Build BASE Markov transition matrix.
        
        This will be ADAPTED dynamically during simulation!
        """
        # Base transition probabilities.
        # INJURED and SUSPENDED are NOT destinations here — they are handled
        # exclusively by the injury-tracking and card-accumulation systems.
        base_matrix = {
            PerformanceState.HOT: {
                PerformanceState.HOT: 0.52,
                PerformanceState.GOOD: 0.43,
                PerformanceState.COLD: 0.05,
                PerformanceState.RECOVERY: 0.00,
            },
            PerformanceState.GOOD: {
                PerformanceState.HOT: 0.16,
                PerformanceState.GOOD: 0.72,
                PerformanceState.COLD: 0.12,
                PerformanceState.RECOVERY: 0.00,
            },
            PerformanceState.COLD: {
                PerformanceState.HOT: 0.05,
                PerformanceState.GOOD: 0.48,
                PerformanceState.COLD: 0.47,
                PerformanceState.RECOVERY: 0.00,
            },
            # INJURED and SUSPENDED rows are dead code (handled by counters),
            # but kept for completeness so the dict is always fully populated.
            PerformanceState.INJURED: {
                PerformanceState.HOT: 0.00,
                PerformanceState.GOOD: 1.00,
                PerformanceState.COLD: 0.00,
                PerformanceState.RECOVERY: 0.00,
            },
            PerformanceState.SUSPENDED: {
                PerformanceState.HOT: 0.00,
                PerformanceState.GOOD: 1.00,
                PerformanceState.COLD: 0.00,
                PerformanceState.RECOVERY: 0.00,
            },
            PerformanceState.RECOVERY: {
                PerformanceState.HOT: 0.05,
                PerformanceState.GOOD: 0.68,
                PerformanceState.COLD: 0.27,
                PerformanceState.RECOVERY: 0.00,
            },
        }
        
        # Adjust based on player characteristics
        adjusted_matrix = self._adjust_base_transition_matrix(base_matrix)
        
        return adjusted_matrix
    
    def _adjust_base_transition_matrix(self, base_matrix: Dict) -> Dict:
        """
        Adjust BASE transition probabilities based on player characteristics.
        
        Factors:
        1. Age: Young = volatile, Old = injury-prone
        2. Position: Attackers = streaky, Defenders = consistent
        3. Discipline: High cards = suspension risk
        4. Durability: Injury history = injury risk
        """
        adjusted = {state: probs.copy() for state, probs in base_matrix.items()}
        
        # 1. AGE ADJUSTMENT (form volatility only — injury risk lives in _check_for_injury)
        if self.age < 23:
            # Young players: More volatile (bigger swings)
            for from_state in [PerformanceState.HOT, PerformanceState.GOOD, PerformanceState.COLD]:
                adjusted[from_state][PerformanceState.HOT] *= 1.20
                adjusted[from_state][PerformanceState.COLD] *= 1.20
                adjusted[from_state][PerformanceState.GOOD] *= 0.80

        # 2. POSITION ADJUSTMENT (streakiness)
        if self.position in [Position.ST, Position.W, Position.AM]:
            # Attackers: Streaky (longer hot/cold runs)
            adjusted[PerformanceState.HOT][PerformanceState.HOT] *= 1.15
            adjusted[PerformanceState.COLD][PerformanceState.COLD] *= 1.15

        elif self.position in [Position.CB, Position.DM]:
            # Defenders/DMs: Consistent
            adjusted[PerformanceState.GOOD][PerformanceState.GOOD] *= 1.20

        # 3. NORMALIZE (ensure rows sum to 1.0)
        for from_state in adjusted:
            total = sum(adjusted[from_state].values())
            if total > 0:
                for to_state in adjusted[from_state]:
                    adjusted[from_state][to_state] /= total
        
        return adjusted
    
    def _get_initial_state_distribution(self) -> Dict[PerformanceState, float]:
        """
        SMART initial state estimation based on player's REAL season stats.
        
        Uses actual performance to set realistic starting point!
        """
        games_played = self.real_stats['games_played']
        goals_per90  = self.real_stats['goals_per90']

        # Position benchmarks (what's "good" for this position)
        position_benchmarks = {
            Position.ST: {'goals': 0.60, 'assists': 0.25},
            Position.W: {'goals': 0.40, 'assists': 0.35},
            Position.AM: {'goals': 0.35, 'assists': 0.40},
            Position.CM: {'goals': 0.15, 'assists': 0.25},
            Position.DM: {'goals': 0.05, 'assists': 0.15},
            Position.FB: {'goals': 0.08, 'assists': 0.20},
            Position.CB: {'goals': 0.05, 'assists': 0.05},
            Position.GK: {'goals': 0.00, 'assists': 0.00},
        }
        
        benchmark = position_benchmarks.get(self.position, {'goals': 0.30, 'assists': 0.25})
        
        # DURABILITY CHECK: Missed lots of games = likely recovering
        if games_played < 15:
            return {
                PerformanceState.GOOD: 0.40,
                PerformanceState.COLD: 0.25,
                PerformanceState.RECOVERY: 0.25,  # Coming back from injury
                PerformanceState.HOT: 0.10,
                PerformanceState.INJURED: 0.00,
                PerformanceState.SUSPENDED: 0.00,
            }
        
        # PERFORMANCE-BASED: Compare to position benchmark
        if goals_per90 > benchmark['goals'] * 1.35:
            # Overperforming → start HOT
            return {
                PerformanceState.HOT: 0.60,
                PerformanceState.GOOD: 0.35,
                PerformanceState.COLD: 0.05,
                PerformanceState.INJURED: 0.00,
                PerformanceState.SUSPENDED: 0.00,
                PerformanceState.RECOVERY: 0.00,
            }
        elif goals_per90 < benchmark['goals'] * 0.65:
            # Underperforming → start COLD
            return {
                PerformanceState.COLD: 0.50,
                PerformanceState.GOOD: 0.45,
                PerformanceState.HOT: 0.05,
                PerformanceState.INJURED: 0.00,
                PerformanceState.SUSPENDED: 0.00,
                PerformanceState.RECOVERY: 0.00,
            }
        else:
            # Normal → start GOOD
            return {
                PerformanceState.GOOD: 0.75,
                PerformanceState.HOT: 0.15,
                PerformanceState.COLD: 0.10,
                PerformanceState.INJURED: 0.00,
                PerformanceState.SUSPENDED: 0.00,
                PerformanceState.RECOVERY: 0.00,
            }
    
    def run_simulations(self) -> Dict:
        """
        Run TRUE Markov Chain Monte Carlo simulations.
        
        Each simulation now tracks:
        - State history (momentum)
        - Consecutive games in each state
        - Injury durations
        - Card accumulation
        """
        all_simulations = []
        
        for _ in range(self.n_sims):
            season = self._simulate_season_markov()
            all_simulations.append(season)
        
        # Calculate percentile statistics
        results = self._calculate_statistics(all_simulations)
        results['simulations'] = all_simulations
        results['player_name'] = str(self.player.get('Player', 'Unknown'))
        results['position'] = self.position.value
        results['age'] = self.age
        results['durability'] = round(self.durability, 2)
        results['importance'] = self.importance_label
        results['avg_minutes'] = self.avg_minutes
        
        return results
    
    def _simulate_season_markov(self) -> Dict:
        """
        Simulate one season using TRUE Markov Chain.
        
        KEY FEATURES:
        - Adaptive transitions (momentum-based)
        - Smart injury system (age/position/fatigue)
        - State memory (tracks consecutive games)
        """
        # Initialize season
        season = {
            'goals': 0, 'assists': 0, 'shots': 0,
            'tackles': 0, 'interceptions': 0, 'clearances': 0,
            'prog_carries': 0, 'prog_passes': 0,
            'saves': 0, 'goals_conceded': 0, 'clean_sheets': 0,
            'yellow_cards': 0, 'red_cards': 0,
            'games_played': 0, 'games_injured': 0, 'games_suspended': 0,
            'games_not_selected': 0,
            'state_history': [],
            'performance_history': [],
            'injuries': [],  # Track injury events
        }
        
        # Sample initial state (SMART - uses real data!)
        states = list(self.initial_state_probs.keys())
        probs = list(self.initial_state_probs.values())
        current_state = np.random.choice(states, p=probs)
        
        # State memory tracking
        consecutive_in_state = 0
        last_state = None
        
        # Injury tracking
        injury_games_remaining = 0
        recovery_games_since_injury = 0
        
        # Card tracking
        accumulated_yellows = 0
        suspension_games_remaining = 0
        
        # Simulate 38-game season
        for game_num in range(38):
            # SELECTION CHECK: rotation/fringe players may not feature at all
            if self.selection_prob < 1.0 and np.random.random() > self.selection_prob:
                season['games_not_selected'] += 1
                continue

            # SAMPLE ACTUAL MINUTES for this appearance (clipped to [5, 90])
            actual_minutes = float(np.clip(
                np.random.normal(self.avg_minutes, self.minutes_std), 5.0, 90.0
            ))
            minutes_fraction = actual_minutes / 90.0

            # Update consecutive state counter
            if current_state == last_state and current_state in [PerformanceState.HOT, PerformanceState.GOOD, PerformanceState.COLD]:
                consecutive_in_state += 1
            else:
                consecutive_in_state = 1
                last_state = current_state

            # HANDLE INJURIES (fixed duration)
            if injury_games_remaining > 0:
                injury_games_remaining -= 1
                season['games_injured'] += 1
                current_state = PerformanceState.INJURED
                season['state_history'].append(current_state)
                
                # Check if healed
                if injury_games_remaining == 0:
                    current_state = PerformanceState.RECOVERY
                    recovery_games_since_injury = 0
                
                continue
            
            # HANDLE SUSPENSIONS
            if suspension_games_remaining > 0:
                suspension_games_remaining -= 1
                season['games_suspended'] += 1
                current_state = PerformanceState.SUSPENDED
                season['state_history'].append(current_state)
                
                # After suspension, return to GOOD
                if suspension_games_remaining == 0:
                    current_state = PerformanceState.GOOD
                    consecutive_in_state = 0
                
                continue
            
            # TRANSITION TO NEW STATE (with ADAPTIVE probabilities!)
            if game_num > 0 and current_state not in [PerformanceState.INJURED, PerformanceState.SUSPENDED]:
                current_state = self._adaptive_transition_state(
                    current_state,
                    consecutive_in_state,
                    game_num,
                    recovery_games_since_injury
                )
            
            season['state_history'].append(current_state)
            
            # Track recovery progress
            if current_state == PerformanceState.RECOVERY:
                recovery_games_since_injury += 1
            else:
                recovery_games_since_injury = 0
            
            # Simulate game stats (scaled to actual minutes played)
            game_stats = self._simulate_game_in_state(current_state, minutes_fraction)
            season['performance_history'].append({
                'game': game_num + 1,
                'state': current_state.value,
                'minutes': round(actual_minutes),
                'stats': game_stats
            })
            
            # Accumulate stats (only counted if we didn't continue above!)
            season['goals'] += game_stats['goals']
            season['assists'] += game_stats['assists']
            season['shots'] += game_stats['shots']
            season['tackles'] += game_stats['tackles']
            season['interceptions'] += game_stats['interceptions']
            season['clearances'] += game_stats['clearances']
            season['prog_carries'] += game_stats['prog_carries']
            season['prog_passes'] += game_stats['prog_passes']
            season['saves'] += game_stats['saves']
            season['goals_conceded'] += game_stats['goals_conceded']
            season['clean_sheets'] += game_stats['clean_sheet']
            season['games_played'] += 1  # This only runs if not injured/suspended!
            
            # Handle cards
            accumulated_yellows += game_stats['yellow']
            season['yellow_cards'] += game_stats['yellow']
            season['red_cards'] += game_stats['red']
            
            # Check for suspension
            if game_stats['red'] > 0:
                ban_length = np.random.choice([1, 2, 3], p=[0.60, 0.30, 0.10])
                suspension_games_remaining = ban_length
                accumulated_yellows = 0
            elif accumulated_yellows >= 10:
                suspension_games_remaining = 2
                accumulated_yellows = 0
            elif accumulated_yellows >= 5:
                suspension_games_remaining = 1
                accumulated_yellows = 0
            
            # CHECK FOR NEW INJURY (SMART SYSTEM!)
            if current_state in [PerformanceState.HOT, PerformanceState.GOOD, PerformanceState.COLD, PerformanceState.RECOVERY]:
                injury_occurred, injury_duration, injury_type = self._check_for_injury(
                    current_state,
                    game_num,
                    recovery_games_since_injury
                )
                
                if injury_occurred:
                    injury_games_remaining = injury_duration
                    season['injuries'].append({
                        'game': game_num + 1,
                        'type': injury_type,
                        'duration': injury_duration,
                        'state_when_injured': current_state.value
                    })
        
        season['total_injuries'] = len(season['injuries'])
        
        return season
    
    def _adaptive_transition_state(self, 
                                   current_state: PerformanceState,
                                   consecutive_games: int,
                                   game_num: int,
                                   recovery_games: int) -> PerformanceState:
        """
        TRUE MARKOV: Adaptive state transition based on MEMORY!
        
        Factors:
        - Consecutive games in state (momentum)
        - Season fatigue (late games)
        - Recovery progress (if just back from injury)
        """
        # Get base transition probabilities
        base_probs = self.base_transition_matrix[current_state].copy()
        
        # MOMENTUM EFFECT: Longer in state → harder to leave
        if consecutive_games >= 4:
            if current_state == PerformanceState.HOT:
                base_probs[PerformanceState.HOT] *= 1.35
            elif current_state == PerformanceState.COLD:
                base_probs[PerformanceState.COLD] *= 1.30

        # RECOVERY DYNAMICS: Gradual return to normal form
        if current_state == PerformanceState.RECOVERY:
            if recovery_games >= 3:
                # After 3 games back, drift toward GOOD
                base_probs[PerformanceState.GOOD] *= 1.50
        
        # Normalize
        total = sum(base_probs.values())
        if total > 0:
            for state in base_probs:
                base_probs[state] /= total
        
        # Sample next state
        states = list(base_probs.keys())
        probs = list(base_probs.values())
        next_state = np.random.choice(states, p=probs)
        
        return next_state
    
    def _check_for_injury(self, 
                         current_state: PerformanceState,
                         game_num: int,
                         recovery_games: int) -> Tuple[bool, int, str]:
        """
        SMART INJURY SYSTEM based on:
        - Age (CRITICAL - older = muscular injuries)
        - Position (fullbacks/midfielders = high workload)
        - Current state (COLD/RECOVERY = higher risk)
        - Season fatigue (late games = breakdown)
        - Durability (historical injury-proneness)
        
        Returns:
            (injury_occurred, duration_games, injury_type)
        """
        # Base injury probability per game.
        # With INJURED removed from the Markov matrix, this is the sole source
        # of injuries. 1.8% baseline gives ~1 injury/season for a typical player.
        base_risk = 0.018
        
        # AGE FACTOR (BIGGEST!)
        if self.age < 23:
            age_multiplier = 0.80  # Young: -20% (was -30%)
        elif self.age < 27:
            age_multiplier = 0.90  # Prime: -10% (was -15%)
        elif self.age < 31:
            age_multiplier = 1.00  # Normal
        elif self.age < 34:
            age_multiplier = 1.30  # Aging: +30% (was +50%)
        else:
            age_multiplier = 1.60  # Veterans: +60% (was +100%)
        
        # POSITION FACTOR
        position_risk = {
            Position.GK: 0.60,   # Goalkeepers rarely injured
            Position.ST: 0.85,   # Strikers less contact
            Position.W: 0.95,    # Wingers sprint a lot
            Position.AM: 1.00,
            Position.CM: 1.15,   # Box-to-box = high workload (was 1.20)
            Position.DM: 1.10,   # Physical battles (was 1.15)
            Position.FB: 1.20,   # Fullbacks run a lot (was 1.35!)
            Position.CB: 1.05,   # Physical contact (was 1.10)
        }
        position_multiplier = position_risk.get(self.position, 1.0)
        
        # DURABILITY (from real data!)
        durability_multiplier = 1.5 - (self.durability * 0.5)  # 0.4 durability → 1.3x (was 1.6x!)
        
        # STATE EFFECTS
        state_effects = STATE_EFFECTS[current_state]
        state_multiplier = state_effects.injury_risk
        
        # RECOVERY RE-INJURY RISK
        if current_state == PerformanceState.RECOVERY:
            if recovery_games == 0:
                state_multiplier *= 1.40  # First game back: higher risk (was 2.0!)
            elif recovery_games == 1:
                state_multiplier *= 1.20  # Second game: still risky (was 1.5)
        
        # FATIGUE ACCUMULATION
        fatigue_multiplier = 1.0
        if game_num > 32:
            fatigue_multiplier = 1.20  # End of season (was 1.40)
        elif game_num > 25:
            fatigue_multiplier = 1.10  # Late season (was 1.20)
        
        # CALCULATE FINAL RISK
        injury_risk = (base_risk * age_multiplier * position_multiplier * 
                      durability_multiplier * state_multiplier * fatigue_multiplier)
        
        # Cap at 15% per game (was 25%!)
        injury_risk = min(injury_risk, 0.15)
        
        # Roll for injury
        if np.random.random() < injury_risk:
            # Injury occurred! Determine duration and type
            duration, injury_type = self._determine_injury_duration()
            return True, duration, injury_type
        
        return False, 0, "none"
    
    def _determine_injury_duration(self) -> Tuple[int, str]:
        """
        Determine injury duration and type based on age.
        
        OLDER PLAYERS: More muscular injuries (longer recovery)
        YOUNG PLAYERS: More impact injuries (shorter recovery)
        
        Returns:
            (games_missed, injury_type)
        """
        # AGE-DEPENDENT INJURY TYPES
        if self.age < 25:
            # Young: mostly impact injuries
            injury_types = {
                'minor_knock': 0.50,      # 1-2 games
                'muscle_strain': 0.30,    # 2-4 games
                'ligament': 0.15,         # 4-7 games
                'serious': 0.05           # 8-12 games
            }
        elif self.age < 32:
            # Prime: balanced
            injury_types = {
                'minor_knock': 0.40,
                'muscle_strain': 0.35,
                'ligament': 0.20,
                'serious': 0.05
            }
        else:
            # Older: MUSCULAR DOMINATE!
            injury_types = {
                'minor_knock': 0.25,
                'muscle_strain': 0.50,    # HAMSTRINGS, CALVES!
                'ligament': 0.20,
                'serious': 0.05
            }
        
        # Sample injury type
        injury = np.random.choice(
            list(injury_types.keys()),
            p=list(injury_types.values())
        )
        
        # Duration by type
        if injury == 'minor_knock':
            duration = np.random.randint(1, 3)  # 1-2 games
        elif injury == 'muscle_strain':
            duration = np.random.randint(2, 5)  # 2-4 games
        elif injury == 'ligament':
            duration = np.random.randint(4, 8)  # 4-7 games
        else:  # serious
            duration = np.random.randint(8, 13)  # 8-12 games
        
        # Older players take LONGER to recover
        if self.age > 32:
            duration = int(duration * 1.35)  # +35% recovery time
        elif self.age > 29:
            duration = int(duration * 1.15)  # +15% recovery time
        
        return duration, injury
    
    def _simulate_game_in_state(self, state: PerformanceState,
                                minutes_fraction: float = 1.0) -> Dict:
        """Simulate a single game given the player's current state and minutes played.

        All expected per-game values are scaled by ``minutes_fraction`` (0–1) so
        that a 30-minute appearance produces proportionally fewer events than a
        full 90.  The perf_multiplier is already baked into real_stats; only the
        time-on-pitch fraction is applied here.
        """
        # Get state multipliers
        multipliers = STATE_EFFECTS[state]

        mf = minutes_fraction  # shorthand

        # Position-specific per-game floors (scaled by minutes fraction)
        min_goals_by_pos = {
            Position.ST: 0.35, Position.W: 0.20, Position.AM: 0.16,
            Position.CM: 0.06, Position.DM: 0.03, Position.FB: 0.04,
            Position.CB: 0.02, Position.GK: 0.00
        }
        min_assists_by_pos = {
            Position.ST: 0.12, Position.W: 0.18, Position.AM: 0.22,
            Position.CM: 0.12, Position.DM: 0.07, Position.FB: 0.10,
            Position.CB: 0.03, Position.GK: 0.00
        }
        min_tackles_by_pos = {
            Position.ST: 0.30, Position.W: 0.50, Position.AM: 0.50,
            Position.CM: 1.00, Position.DM: 1.50, Position.FB: 1.00,
            Position.CB: 0.80, Position.GK: 0.10
        }
        min_int_by_pos = {
            Position.ST: 0.20, Position.W: 0.30, Position.AM: 0.30,
            Position.CM: 0.60, Position.DM: 1.00, Position.FB: 0.60,
            Position.CB: 0.60, Position.GK: 0.10
        }
        min_prgc_by_pos = {
            Position.ST: 1.00, Position.W: 2.50, Position.AM: 2.00,
            Position.CM: 1.50, Position.DM: 0.80, Position.FB: 2.00,
            Position.CB: 0.50, Position.GK: 0.10
        }
        min_prgp_by_pos = {
            Position.ST: 0.50, Position.W: 1.00, Position.AM: 2.50,
            Position.CM: 3.00, Position.DM: 3.50, Position.FB: 2.50,
            Position.CB: 3.00, Position.GK: 0.50
        }

        base_goals   = max(self.real_stats['goals_per90'],
                           min_goals_by_pos.get(self.position, 0.05) * mf)
        base_assists = max(self.real_stats['assists_per90'],
                           min_assists_by_pos.get(self.position, 0.05) * mf)

        exp_goals   = base_goals   * multipliers.goals     * mf
        exp_assists = base_assists * multipliers.assists   * mf
        exp_shots   = max(self.real_stats['shots_per90'],   0.8) * multipliers.goals     * mf
        exp_tackles = max(self.real_stats['tackles_per90'],
                          min_tackles_by_pos.get(self.position, 0.5)) * multipliers.defensive * mf
        exp_int     = max(self.real_stats['interceptions_per90'],
                          min_int_by_pos.get(self.position, 0.3))     * multipliers.defensive * mf
        exp_clr     = max(self.real_stats['clearances_per90'],    0.3) * multipliers.defensive * mf
        exp_prgc    = max(self.real_stats['prgc_per90'],
                          min_prgc_by_pos.get(self.position, 0.5))    * multipliers.progressive * mf
        exp_prgp    = max(self.real_stats['prgp_per90'],
                          min_prgp_by_pos.get(self.position, 1.0))    * multipliers.progressive * mf
        # Cards scale with minutes only — card_risk already in multipliers
        exp_yellows = max(self.real_stats['yellows_per90'], 0.03) * multipliers.card_risk * mf
        exp_reds    = max(self.real_stats['reds_per90'],  0.001)  * multipliers.card_risk * mf

        # Sample stats
        goals, assists = self._sample_correlated_goals_assists(exp_goals, exp_assists)

        shots = int(np.random.gamma(max(1, exp_shots), 1.2)) if exp_shots > 0 else 0
        shots = max(shots, goals)

        tackles       = int(np.random.gamma(max(0.5, exp_tackles), 1.3)) if exp_tackles > 0 else 0
        interceptions = int(np.random.gamma(max(0.5, exp_int),     1.3)) if exp_int     > 0 else 0
        clearances    = int(np.random.gamma(max(0.5, exp_clr),     1.3)) if exp_clr     > 0 else 0
        prog_carries  = int(np.random.gamma(max(0.5, exp_prgc),    1.3)) if exp_prgc    > 0 else 0
        prog_passes   = int(np.random.gamma(max(0.5, exp_prgp),    1.3)) if exp_prgp    > 0 else 0

        yellow = min(np.random.poisson(exp_yellows), 2) if exp_yellows > 0 else 0
        red    = min(np.random.poisson(exp_reds),    1) if exp_reds    > 0 else 0

        # GK stats (also scaled by minutes fraction)
        saves = 0
        goals_conceded = 0
        clean_sheet = 0
        if self.position == Position.GK:
            exp_saves = self.real_stats['saves_per90'] * multipliers.defensive * mf
            exp_ga    = self.real_stats['ga_per90'] * mf

            saves = int(np.random.gamma(max(1, exp_saves), 1.1)) if exp_saves > 0 else 0
            goals_conceded = np.random.poisson(exp_ga) if exp_ga > 0 else 0
            clean_sheet = 1 if goals_conceded == 0 else 0

        return {
            'goals': goals, 'assists': assists, 'shots': shots,
            'tackles': tackles, 'interceptions': interceptions, 'clearances': clearances,
            'prog_carries': prog_carries, 'prog_passes': prog_passes,
            'saves': saves, 'goals_conceded': goals_conceded, 'clean_sheet': clean_sheet,
            'yellow': yellow, 'red': red,
            'state': state.value
        }
    
    def _sample_correlated_goals_assists(self, exp_goals: float, exp_assists: float) -> Tuple[int, int]:
        """Sample correlated goals and assists using Gaussian copula."""
        correlation_map = {
            Position.ST: 0.50, Position.W: 0.55, Position.AM: 0.60,
            Position.CM: 0.30, Position.DM: 0.20, Position.FB: 0.25,
            Position.CB: 0.15, Position.GK: 0.00
        }
        correlation = correlation_map.get(self.position, 0.30)
        
        if exp_goals < 0.01 and exp_assists < 0.01:
            return (0, 0)
        if exp_goals < 0.01:
            assists = np.random.poisson(exp_assists) if exp_assists > 0 else 0
            return (0, assists)
        if exp_assists < 0.01:
            if exp_goals > 0.1:
                n = max(1.5, exp_goals * 2.5)
                p = n / (n + exp_goals)
                goals = np.random.negative_binomial(int(round(n)), p)
            else:
                goals = 1 if np.random.random() < exp_goals else 0
            return (goals, 0)
        
        mean = [0, 0]
        cov = [[1, correlation], [correlation, 1]]
        z1, z2 = np.random.multivariate_normal(mean, cov)
        
        u1 = norm.cdf(z1)
        u2 = norm.cdf(z2)
        
        if exp_goals > 0.1:
            n = max(1.5, exp_goals * 2.5)
            p = n / (n + exp_goals)
            goals = nbinom.ppf(u1, n, p)
        else:
            goals = 1 if u1 > (1 - exp_goals) else 0
        
        assists = poisson.ppf(u2, exp_assists)
        
        return (int(max(0, goals)), int(max(0, assists)))
    
    def _calculate_statistics(self, simulations: List[Dict]) -> Dict:
        """Calculate percentile statistics from all simulations"""
        stats = {}
        metrics = [
            'games_played', 'games_not_selected', 'goals', 'assists', 'shots',
            'tackles', 'interceptions', 'clearances',
            'prog_carries', 'prog_passes',
            'saves', 'clean_sheets',
        ]

        for metric in metrics:
            values = [s[metric] for s in simulations]
            stats[metric] = {
                'p5': int(np.percentile(values, 5)),
                'p10': int(np.percentile(values, 10)),
                'p25': int(np.percentile(values, 25)),
                'median': int(np.percentile(values, 50)),
                'p75': int(np.percentile(values, 75)),
                'p90': int(np.percentile(values, 90)),
                'p95': int(np.percentile(values, 95)),
                'mean': float(np.mean(values)),
            }
        
        # Discipline stats
        stats['discipline'] = {
            'yellow_cards': {
                'p10': int(np.percentile([s['yellow_cards'] for s in simulations], 10)),
                'median': int(np.percentile([s['yellow_cards'] for s in simulations], 50)),
                'p90': int(np.percentile([s['yellow_cards'] for s in simulations], 90)),
            },
            'red_cards': {
                'p10': int(np.percentile([s['red_cards'] for s in simulations], 10)),
                'median': int(np.percentile([s['red_cards'] for s in simulations], 50)),
                'p90': int(np.percentile([s['red_cards'] for s in simulations], 90)),
            }
        }
        
        # State statistics
        stats['state_statistics'] = self._calculate_state_statistics(simulations)
        
        # INJURY STATISTICS (NEW!)
        stats['injury_statistics'] = {
            'total_injuries': {
                'p10': int(np.percentile([s['total_injuries'] for s in simulations], 10)),
                'median': int(np.percentile([s['total_injuries'] for s in simulations], 50)),
                'p90': int(np.percentile([s['total_injuries'] for s in simulations], 90)),
                'mean': round(np.mean([s['total_injuries'] for s in simulations]), 1),
            },
            'games_injured': {
                'p10': int(np.percentile([s['games_injured'] for s in simulations], 10)),
                'median': int(np.percentile([s['games_injured'] for s in simulations], 50)),
                'p90': int(np.percentile([s['games_injured'] for s in simulations], 90)),
                'mean': round(np.mean([s['games_injured'] for s in simulations]), 1),
            }
        }
        
        return stats
    
    def _calculate_state_statistics(self, simulations: List[Dict]) -> Dict:
        """Calculate time spent in each state across simulations."""
        state_counts = {state: [] for state in PerformanceState}
        
        for sim in simulations:
            state_count = {state: 0 for state in PerformanceState}
            for state in sim['state_history']:
                state_count[state] += 1
            
            for state in PerformanceState:
                state_counts[state].append(state_count[state])
        
        stats = {}
        for state in PerformanceState:
            values = state_counts[state]
            stats[state.value] = {
                'median': int(np.percentile(values, 50)),
                'mean': round(np.mean(values), 1),
                'p25': int(np.percentile(values, 25)),
                'p75': int(np.percentile(values, 75)),
            }
        
        return stats


# ======================== USAGE EXAMPLE ========================

if __name__ == "__main__":
    # Example: 34-year-old striker (injury-prone)
    old_striker = pd.Series({
        'Player': 'Veteran Striker',
        'Pos': 'FW',
        'Age': 34,  # OLD!
        '90s': 18,  # Missed lots of games
        'Gls': 12,
        'Ast': 4,
        'Sh': 60,
        'CrdY': 2,
        'CrdR': 0,
        'Tkl': 8,
        'Int': 5,
        'Clr': 2,
        'Blocks': 1,
        'is_striker': True,
        'is_gk': False, 'is_cb': False, 'is_fb': False,
        'is_dm': False, 'is_cm': False, 'is_am': False, 'is_winger': False,
    })
    
    print("=" * 80)
    print("TRUE MARKOV CHAIN MONTE CARLO SIMULATION")
    print("With Smart Injury System & Adaptive Transitions")
    print("=" * 80)
    
    sim = MonteCarloSimulator(old_striker, n_simulations=500)
    results = sim.run_simulations()
    
    print(f"\nPlayer: {results['player_name']} (Age: {results['age']})")
    print(f"Position: {results['position']}")
    print(f"Durability: {results['durability']:.2f} (0.0=injury-prone, 1.0=iron man)")
    
    print("\n--- Season Projections (p25 - median - p75) ---")
    print(f"Games Played:     {results['games_played']['p25']:3d} - {results['games_played']['median']:3d} - {results['games_played']['p75']:3d}")
    print(f"Goals:            {results['goals']['p25']:3d} - {results['goals']['median']:3d} - {results['goals']['p75']:3d}")
    print(f"Assists:          {results['assists']['p25']:3d} - {results['assists']['median']:3d} - {results['assists']['p75']:3d}")
    print(f"Tackles:          {results['tackles']['p25']:3d} - {results['tackles']['median']:3d} - {results['tackles']['p75']:3d}")
    print(f"Interceptions:    {results['interceptions']['p25']:3d} - {results['interceptions']['median']:3d} - {results['interceptions']['p75']:3d}")
    print(f"Prog. Carries:    {results['prog_carries']['p25']:3d} - {results['prog_carries']['median']:3d} - {results['prog_carries']['p75']:3d}")
    print(f"Prog. Passes:     {results['prog_passes']['p25']:3d} - {results['prog_passes']['median']:3d} - {results['prog_passes']['p75']:3d}")
    
    print("\n--- State Distribution ---")
    for state, stats in results['state_statistics'].items():
        print(f"{state.upper():12s}: {stats['median']:2d} games (avg {stats['mean']:.1f})")
    
    print("\n--- Injury Statistics (NEW!) ---")
    inj_stats = results['injury_statistics']
    print(f"Total Injuries:  {inj_stats['total_injuries']['median']} (range: {inj_stats['total_injuries']['p10']}-{inj_stats['total_injuries']['p90']})")
    print(f"Games Missed:    {inj_stats['games_injured']['median']} (range: {inj_stats['games_injured']['p10']}-{inj_stats['games_injured']['p90']})")
    
    print("\n✅ TRUE Markov Chain simulation complete!")
    print("Features: State memory, adaptive transitions, smart injuries!")