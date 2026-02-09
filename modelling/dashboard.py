import pandas as pd
import numpy as np
import os
import sys
import tkinter as tk
from tkinter import ttk, messagebox
import threading
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor,
    RandomForestClassifier, GradientBoostingClassifier,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

# ============================================================
# Teams Dictionary
# ============================================================
TEAMS = {
    1610612737: "Atlanta_Hawks",
    1610612738: "Boston_Celtics",
    1610612751: "Brooklyn_Nets",
    1610612766: "Charlotte_Hornets",
    1610612741: "Chicago_Bulls",
    1610612739: "Cleveland_Cavaliers",
    1610612742: "Dallas_Mavericks",
    1610612743: "Denver_Nuggets",
    1610612765: "Detroit_Pistons",
    1610612744: "Golden_State_Warriors",
    1610612745: "Houston_Rockets",
    1610612754: "Indiana_Pacers",
    1610612746: "Los_Angeles_Clippers",
    1610612747: "Los_Angeles_Lakers",
    1610612763: "Memphis_Grizzlies",
    1610612748: "Miami_Heat",
    1610612749: "Milwaukee_Bucks",
    1610612750: "Minnesota_Timberwolves",
    1610612740: "New_Orleans_Pelicans",
    1610612752: "New_York_Knicks",
    1610612760: "Oklahoma_City_Thunder",
    1610612753: "Orlando_Magic",
    1610612755: "Philadelphia_76ers",
    1610612756: "Phoenix_Suns",
    1610612757: "Portland_Trail_Blazers",
    1610612758: "Sacramento_Kings",
    1610612759: "San_Antonio_Spurs",
    1610612761: "Toronto_Raptors",
    1610612762: "Utah_Jazz",
    1610612764: "Washington_Wizards",
}

TEAM_NAME_TO_ID = {v.replace("_", " "): k for k, v in TEAMS.items()}

ROLLING_WINDOW = 5
ROLLING_COLS = [
    'PTS', 'FG_PCT', 'FG3_PCT', 'FTM', 'FTA',
    'OREB', 'DREB', 'AST', 'STL', 'BLK', 'TOV',
    'offensiveRating', 'defensiveRating', 'pace',
    'effectiveFieldGoalPercentage', 'trueShootingPercentage',
    'PLUS_MINUS',
]


# ============================================================
# Data Loading & Feature Engineering
# ============================================================
def load_and_prepare_data():
    base = Path(__file__).parent.parent / "collected data"
    folder = base / "2025-26 season"

    all_teams = []
    for team_name in TEAMS.values():
        fp = folder / f"{team_name}_2025-26.csv"
        if fp.exists():
            all_teams.append(pd.read_csv(fp))
    df = pd.concat(all_teams, ignore_index=True)

    df = df[df['startersBench'] == 'Starters'].reset_index(drop=True)
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    df = df.sort_values(['TEAM_ID', 'GAME_DATE']).reset_index(drop=True)

    df['is_home'] = df['MATCHUP'].str.contains('vs.').astype(int)
    df['WIN'] = (df['WL'] == 'W').astype(int)
    df['OPP_PTS'] = (df['PTS'] - df['PLUS_MINUS']).astype(int)
    df['GAME_TOTAL'] = df['PTS'] + df['OPP_PTS']

    for col in ROLLING_COLS:
        df[f'avg_{col}'] = df.groupby('TEAM_ID')[col].transform(
            lambda x: x.shift(1).rolling(window=ROLLING_WINDOW, min_periods=3).mean()
        )

    avg_cols = [f'avg_{col}' for col in ROLLING_COLS]

    opp = df[['GAME_ID', 'TEAM_ID'] + avg_cols].copy()
    opp.columns = ['GAME_ID', 'OPP_TEAM_ID'] + [f'opp_{c}' for c in avg_cols]
    df = df.merge(opp, on='GAME_ID')
    df = df[df['TEAM_ID'] != df['OPP_TEAM_ID']].reset_index(drop=True)

    opp_avg_cols = [f'opp_{c}' for c in avg_cols]
    df = df.dropna(subset=avg_cols + opp_avg_cols).reset_index(drop=True)
    return df


def compute_latest_averages(df):
    avg_cols = [f'avg_{col}' for col in ROLLING_COLS]
    latest = {}
    for team_id in df['TEAM_ID'].unique():
        team_df = df[df['TEAM_ID'] == team_id].sort_values('GAME_DATE')
        row = {}
        for col in ROLLING_COLS:
            vals = team_df[col].values
            if len(vals) >= 3:
                row[f'avg_{col}'] = float(np.mean(vals[-ROLLING_WINDOW:]))
            else:
                row[f'avg_{col}'] = float(np.mean(vals))
        latest[team_id] = row
    return latest


# ============================================================
# Model Training
# ============================================================
def train_models(df):
    avg_cols = [f'avg_{col}' for col in ROLLING_COLS]
    feature_cols = ['is_home'] + avg_cols + [f'opp_{c}' for c in avg_cols]

    X = df[feature_cols]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    spread_models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, max_depth=4, random_state=42),
    }
    for m in spread_models.values():
        m.fit(X_scaled, df['PLUS_MINUS'])

    ml_models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, max_depth=4, random_state=42),
    }
    for m in ml_models.values():
        m.fit(X_scaled, df['WIN'])

    pts_models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, max_depth=4, random_state=42),
    }
    for m in pts_models.values():
        m.fit(X_scaled, df['PTS'])

    total_models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, max_depth=4, random_state=42),
    }
    for m in total_models.values():
        m.fit(X_scaled, df['GAME_TOTAL'])

    training_data = {'X_scaled': X_scaled, 'GAME_TOTAL': df['GAME_TOTAL'].values}

    return scaler, feature_cols, spread_models, ml_models, pts_models, total_models, training_data


# ============================================================
# Prediction
# ============================================================
def build_feature_vector(team_id, opp_id, is_home, latest_avgs):
    avg_cols = [f'avg_{col}' for col in ROLLING_COLS]
    team_avgs = latest_avgs[team_id]
    opp_avgs = latest_avgs[opp_id]

    row = [is_home]
    for c in avg_cols:
        row.append(team_avgs[c])
    for c in avg_cols:
        row.append(opp_avgs[c])
    return np.array(row).reshape(1, -1)


def predict_game(home_id, away_id, home_spread, ou_line, home_name, away_name,
                 scaler, spread_models, ml_models, pts_models, total_models,
                 training_data, latest_avgs):
    home_vec = scaler.transform(build_feature_vector(home_id, away_id, 1, latest_avgs))
    away_vec = scaler.transform(build_feature_vector(away_id, home_id, 0, latest_avgs))

    away_spread = -home_spread
    home_sign = "+" if home_spread > 0 else ""
    away_sign = "+" if away_spread > 0 else ""

    lines = []
    lines.append(f"  {away_name}  @  {home_name}")
    lines.append("")

    # --- Spread ---
    lines.append("=" * 58)
    lines.append("  SPREAD")
    lines.append(f"  Line: {home_name} {home_sign}{home_spread}"
                 f"  |  {away_name} {away_sign}{away_spread}")
    lines.append("=" * 58)
    home_picks = 0
    preds = []
    for label, model in spread_models.items():
        pred = model.predict(home_vec)[0]
        preds.append(pred)
        # pred is home margin: positive = home wins, negative = home loses
        # home covers if predicted margin beats the spread
        #   home_spread negative (favorite): must win by more than abs(spread)
        #   home_spread positive (underdog): can lose by up to spread points
        home_covers = pred > -home_spread
        if home_covers:
            home_picks += 1
            pick = home_name
            if pred > 0:
                reason = f"predicted to win by {pred:.1f}"
            else:
                reason = f"predicted to lose by {abs(pred):.1f}, within {abs(home_spread)} pts"
        else:
            pick = away_name
            if pred < 0:
                reason = f"predicted to lose by {abs(pred):.1f}, exceeds {abs(home_spread)} pts"
            else:
                reason = f"predicted to win by {pred:.1f}, doesn't cover {abs(home_spread)}"
        lines.append(f"  {label:25s}  {pick:20s}  ({reason})")

    if home_picks >= 2:
        rec_team = home_name
        rec_spread = home_spread
    else:
        rec_team = away_name
        rec_spread = -home_spread
    rec_sign = "+" if rec_spread > 0 else ""
    votes = max(home_picks, len(spread_models) - home_picks)
    if rec_spread < 0:
        rec_meaning = f"must win by more than {abs(rec_spread)}"
    else:
        rec_meaning = f"must stay within {abs(rec_spread)} pts"
    lines.append("  " + "-" * 54)
    lines.append(f"  >>> PLACE: {rec_team} {rec_sign}{rec_spread}"
                 f"  ({votes}/{len(spread_models)} models agree)")
    lines.append(f"      ({rec_meaning})")

    # --- Moneyline ---
    lines.append("")
    lines.append("=" * 58)
    lines.append("  MONEYLINE (Win Probability)")
    lines.append("=" * 58)
    for label, model in ml_models.items():
        home_prob = model.predict_proba(home_vec)[0][1]
        away_prob = model.predict_proba(away_vec)[0][1]
        lines.append(f"  {label:25s}  Home: {home_prob:.1%}  |  Away: {away_prob:.1%}")

    # --- Team Totals ---
    lines.append("")
    lines.append("=" * 58)
    lines.append("  TEAM TOTALS (Predicted Points)")
    lines.append("=" * 58)
    for label, model in pts_models.items():
        home_pts = model.predict(home_vec)[0]
        away_pts = model.predict(away_vec)[0]
        lines.append(f"  {label:25s}  Home: {home_pts:.1f}  |  Away: {away_pts:.1f}")

    # --- Game Total O/U ---
    lines.append("")
    lines.append("=" * 58)
    lines.append("  GAME TOTAL OVER/UNDER")
    lines.append(f"  Line: {ou_line}")
    lines.append("=" * 58)
    for label, model in total_models.items():
        pred = model.predict(home_vec)[0]
        call = "OVER" if pred > ou_line else "UNDER"
        lines.append(f"  {label:25s}  Predicted: {pred:.1f}  ->  {call}")

    ou_target = (training_data['GAME_TOTAL'] > ou_line).astype(int)
    ou_log_model = LogisticRegression(max_iter=1000, random_state=42)
    ou_log_model.fit(training_data['X_scaled'], ou_target)
    over_prob = ou_log_model.predict_proba(home_vec)[0][1]
    call = "OVER" if over_prob > 0.5 else "UNDER"
    lines.append(f"  {'Logistic Regression':25s}  Over: {over_prob:.1%}  ->  {call}")

    return "\n".join(lines)


# ============================================================
# Top Picks — scan all games on a date, rank best bets
# ============================================================
def get_top_picks(games_df, scaler, spread_models, ml_models, pts_models,
                  total_models, latest_avgs):
    """Analyse every game on a date and return the top 3 most confident bets.

    Bet types considered:
      - Moneyline: avg win-probability across 3 classifiers (pick the favoured side)
      - Spread:    avg predicted margin across 3 regressors (confidence = magnitude)
      - Game Total: avg predicted total across 3 regressors
    """
    bets = []  # (confidence, description_string)

    for _, row in games_df.iterrows():
        home_name = row['Home Team']
        away_name = row['Away Team']
        home_id = TEAM_NAME_TO_ID.get(home_name)
        away_id = TEAM_NAME_TO_ID.get(away_name)
        if home_id is None or away_id is None:
            continue
        if home_id not in latest_avgs or away_id not in latest_avgs:
            continue

        home_vec = scaler.transform(build_feature_vector(home_id, away_id, 1, latest_avgs))
        away_vec = scaler.transform(build_feature_vector(away_id, home_id, 0, latest_avgs))

        # --- Moneyline confidence ---
        home_probs = []
        for model in ml_models.values():
            home_probs.append(model.predict_proba(home_vec)[0][1])
        avg_home_prob = np.mean(home_probs)
        # Pick the stronger side
        if avg_home_prob >= 0.5:
            ml_conf = avg_home_prob
            ml_pick = home_name
        else:
            ml_conf = 1 - avg_home_prob
            ml_pick = away_name
        bets.append((
            ml_conf,
            f"Moneyline:  {ml_pick} to WIN\n"
            f"              {away_name} @ {home_name}\n"
            f"              Avg Win Prob: {ml_conf:.1%}"
        ))

        # --- Spread confidence (no line needed — just magnitude) ---
        margins = []
        for model in spread_models.values():
            margins.append(model.predict(home_vec)[0])
        avg_margin = np.mean(margins)
        if avg_margin >= 0:
            sp_pick = home_name
            sp_sign = "+"
        else:
            sp_pick = away_name
            sp_sign = ""
        sp_conf = abs(avg_margin)
        bets.append((
            sp_conf,
            f"Spread:     {sp_pick} favoured\n"
            f"              {away_name} @ {home_name}\n"
            f"              Avg Predicted Margin: {sp_sign}{avg_margin:.1f}"
        ))

        # --- Game total direction confidence ---
        totals = []
        for model in total_models.values():
            totals.append(model.predict(home_vec)[0])
        avg_total = np.mean(totals)
        # Confidence = how far the predicted total deviates from league average (~224)
        league_avg = 224.0
        total_dev = abs(avg_total - league_avg)
        if avg_total >= league_avg:
            gt_call = "HIGH-scoring"
        else:
            gt_call = "LOW-scoring"
        bets.append((
            total_dev,
            f"Game Total: {gt_call} game\n"
            f"              {away_name} @ {home_name}\n"
            f"              Avg Predicted Total: {avg_total:.1f}"
        ))

    # Sort by confidence descending and take top 3
    bets.sort(key=lambda x: x[0], reverse=True)
    top = bets[:3]

    lines = []
    lines.append("=" * 58)
    lines.append("  TOP 3 PICKS OF THE DAY")
    lines.append("=" * 58)
    for rank, (conf, desc) in enumerate(top, 1):
        lines.append(f"\n  #{rank}")
        for d in desc.split("\n"):
            lines.append(f"  {d}")
    lines.append("")
    return "\n".join(lines)


# ============================================================
# Schedule Parsing
# ============================================================
def load_schedule():
    path = Path(__file__).parent.parent / "collected data" / "nba_schedule_2025-2026.csv"
    sched = pd.read_csv(path)
    sched['parsed_date'] = pd.to_datetime(sched['Date'], format='mixed')
    return sched


def games_on_date(sched, date_str):
    target = pd.to_datetime(date_str).date()
    mask = sched['parsed_date'].dt.date == target
    return sched[mask].reset_index(drop=True)


# ============================================================
# Tkinter GUI
# ============================================================
class DashboardApp:
    def __init__(self, root):
        self.root = root
        self.root.title("NBA Game Prediction Dashboard")
        self.root.geometry("750x700")
        self.root.resizable(True, True)

        # State
        self.games_df = None
        self.scaler = None
        self.spread_models = None
        self.ml_models = None
        self.pts_models = None
        self.total_models = None
        self.training_data = None
        self.latest_avgs = None
        self.sched = None
        self.models_ready = False

        self._build_ui()
        self._set_loading_state(True)

        # Train models in a background thread so the window doesn't freeze
        threading.Thread(target=self._load_models, daemon=True).start()

    # ---- UI construction ----
    def _build_ui(self):
        pad = {'padx': 8, 'pady': 4}

        # --- Top: Date search ---
        date_frame = ttk.LabelFrame(self.root, text="1. Select Date")
        date_frame.pack(fill='x', **pad)

        ttk.Label(date_frame, text="Date (YYYY-MM-DD):").pack(side='left', **pad)
        self.date_var = tk.StringVar()
        self.date_entry = ttk.Entry(date_frame, textvariable=self.date_var, width=14)
        self.date_entry.pack(side='left', **pad)
        self.search_btn = ttk.Button(date_frame, text="Search", command=self._on_search)
        self.search_btn.pack(side='left', **pad)
        self.top_picks_btn = ttk.Button(date_frame, text="Top Picks", command=self._on_top_picks)
        self.top_picks_btn.pack(side='left', **pad)

        self.status_label = ttk.Label(date_frame, text="Loading models...", foreground="gray")
        self.status_label.pack(side='right', **pad)

        # --- Games list ---
        games_frame = ttk.LabelFrame(self.root, text="2. Select Game")
        games_frame.pack(fill='x', **pad)

        self.games_listbox = tk.Listbox(games_frame, height=8, font=("Consolas", 10))
        self.games_listbox.pack(fill='x', **pad)

        # --- Inputs: Spread + O/U ---
        input_frame = ttk.LabelFrame(self.root, text="3. Enter Lines")
        input_frame.pack(fill='x', **pad)

        ttk.Label(input_frame, text="Home Spread:").grid(row=0, column=0, **pad, sticky='e')
        self.spread_var = tk.StringVar()
        self.spread_entry = ttk.Entry(input_frame, textvariable=self.spread_var, width=10)
        self.spread_entry.grid(row=0, column=1, **pad, sticky='w')
        ttk.Label(input_frame, text="(e.g. -1.5 or +3)").grid(row=0, column=2, **pad, sticky='w')

        ttk.Label(input_frame, text="Over/Under:").grid(row=1, column=0, **pad, sticky='e')
        self.ou_var = tk.StringVar()
        self.ou_entry = ttk.Entry(input_frame, textvariable=self.ou_var, width=10)
        self.ou_entry.grid(row=1, column=1, **pad, sticky='w')
        ttk.Label(input_frame, text="(e.g. 220.5)").grid(row=1, column=2, **pad, sticky='w')

        self.predict_btn = ttk.Button(input_frame, text="Predict", command=self._on_predict)
        self.predict_btn.grid(row=0, column=3, rowspan=2, **pad)

        # --- Results ---
        results_frame = ttk.LabelFrame(self.root, text="Predictions")
        results_frame.pack(fill='both', expand=True, **pad)

        self.results_text = tk.Text(results_frame, font=("Consolas", 10), wrap='none',
                                    state='disabled', bg="#1e1e1e", fg="#d4d4d4",
                                    insertbackground="white")
        scrollbar_y = ttk.Scrollbar(results_frame, orient='vertical', command=self.results_text.yview)
        scrollbar_x = ttk.Scrollbar(results_frame, orient='horizontal', command=self.results_text.xview)
        self.results_text.configure(yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set)

        scrollbar_y.pack(side='right', fill='y')
        scrollbar_x.pack(side='bottom', fill='x')
        self.results_text.pack(fill='both', expand=True)

    def _set_loading_state(self, loading):
        state = 'disabled' if loading else 'normal'
        self.search_btn.configure(state=state)
        self.predict_btn.configure(state=state)
        self.top_picks_btn.configure(state=state)

    # ---- Background model loading ----
    def _load_models(self):
        df = load_and_prepare_data()
        self.latest_avgs = compute_latest_averages(df)
        (self.scaler, _, self.spread_models, self.ml_models,
         self.pts_models, self.total_models, self.training_data) = train_models(df)
        self.sched = load_schedule()
        self.models_ready = True
        self.root.after(0, self._on_models_ready)

    def _on_models_ready(self):
        self.status_label.configure(text="Models ready", foreground="green")
        self._set_loading_state(False)

    # ---- Callbacks ----
    def _on_search(self):
        date_str = self.date_var.get().strip()
        if not date_str:
            messagebox.showwarning("Input", "Please enter a date.")
            return

        try:
            self.games_df = games_on_date(self.sched, date_str)
        except Exception:
            messagebox.showerror("Error", "Could not parse date. Use YYYY-MM-DD format.")
            return

        self.games_listbox.delete(0, tk.END)
        if self.games_df.empty:
            self.games_listbox.insert(tk.END, "No games found for this date.")
            return

        for _, row in self.games_df.iterrows():
            self.games_listbox.insert(
                tk.END,
                f"{row['Away Team']}  @  {row['Home Team']}    ({row['Time (ET)']})"
            )

    def _on_predict(self):
        # Validate game selection
        sel = self.games_listbox.curselection()
        if not sel:
            messagebox.showwarning("Input", "Please select a game from the list.")
            return
        if self.games_df is None or self.games_df.empty:
            messagebox.showwarning("Input", "Search for a date first.")
            return

        idx = sel[0]
        game_row = self.games_df.iloc[idx]
        home_name = game_row['Home Team']
        away_name = game_row['Away Team']

        home_id = TEAM_NAME_TO_ID.get(home_name)
        away_id = TEAM_NAME_TO_ID.get(away_name)
        if home_id is None or away_id is None:
            messagebox.showerror("Error", f"Could not map team names: {home_name}, {away_name}")
            return
        if home_id not in self.latest_avgs or away_id not in self.latest_avgs:
            messagebox.showerror("Error", "Not enough historical data for one of these teams.")
            return

        # Validate spread
        try:
            home_spread = float(self.spread_var.get().strip())
        except ValueError:
            messagebox.showwarning("Input", "Enter a valid number for the home spread.")
            return

        # Validate O/U
        try:
            ou_line = float(self.ou_var.get().strip())
        except ValueError:
            messagebox.showwarning("Input", "Enter a valid number for the O/U line.")
            return

        # Run prediction
        result = predict_game(
            home_id, away_id, home_spread, ou_line, home_name, away_name,
            self.scaler, self.spread_models, self.ml_models,
            self.pts_models, self.total_models, self.training_data,
            self.latest_avgs,
        )

        self._show_result(result)

    def _on_top_picks(self):
        if self.games_df is None or self.games_df.empty:
            messagebox.showwarning("Input", "Search for a date first so there are games to analyse.")
            return

        result = get_top_picks(
            self.games_df, self.scaler, self.spread_models, self.ml_models,
            self.pts_models, self.total_models, self.latest_avgs,
        )
        self._show_result(result)

    def _show_result(self, text):
        self.results_text.configure(state='normal')
        self.results_text.delete('1.0', tk.END)
        self.results_text.insert(tk.END, text)
        self.results_text.configure(state='disabled')


def main():
    root = tk.Tk()
    DashboardApp(root)
    root.mainloop()


if __name__ == '__main__':
    main()
