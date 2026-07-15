"""
cricket_engine.py
-----------------
Simulates a live T20 cricket match ball-by-ball in a background thread.
Generates realistic delivery data (length, line, speed, bounce coordinates,
seam/spin deviation) and match statistics (CRR, RRR, pressure index,
partnership, fall of wickets).

No API key required — runs in demo simulation mode by default.
"""

import time
import random
import threading


class CricketEngine:
    TEAMS = [
        {
            "name": "India", "abbr": "IND", "color": "#0066B4",
            "batsmen": ["Rohit Sharma", "Virat Kohli", "Shubman Gill",
                        "Suryakumar Yadav", "Hardik Pandya", "Rishabh Pant",
                        "Axar Patel", "Ravindra Jadeja", "Jasprit Bumrah",
                        "Mohammed Shami", "Arshdeep Singh"],
            "bowlers": ["Jasprit Bumrah", "Mohammed Shami", "Arshdeep Singh",
                        "Axar Patel", "Ravindra Jadeja"],
        },
        {
            "name": "Australia", "abbr": "AUS", "color": "#FFCD00",
            "batsmen": ["David Warner", "Travis Head", "Steve Smith",
                        "Glenn Maxwell", "Cameron Green", "Matthew Wade",
                        "Pat Cummins", "Mitchell Starc", "Josh Hazlewood",
                        "Adam Zampa", "Nathan Ellis"],
            "bowlers": ["Pat Cummins", "Mitchell Starc", "Josh Hazlewood",
                        "Adam Zampa", "Nathan Ellis"],
        },
        {
            "name": "England", "abbr": "ENG", "color": "#003366",
            "batsmen": ["Jos Buttler", "Phil Salt", "Dawid Malan",
                        "Joe Root", "Ben Stokes", "Liam Livingstone",
                        "Sam Curran", "Adil Rashid", "Mark Wood",
                        "Jofra Archer", "Harry Brook"],
            "bowlers": ["Mark Wood", "Jofra Archer", "Sam Curran",
                        "Adil Rashid", "Chris Woakes"],
        },
        {
            "name": "South Africa", "abbr": "SA", "color": "#007749",
            "batsmen": ["Quinton de Kock", "Temba Bavuma", "Rassie van der Dussen",
                        "David Miller", "Heinrich Klaasen", "Aiden Markram",
                        "Marco Jansen", "Kagiso Rabada", "Tabraiz Shamsi",
                        "Anrich Nortje", "Gerald Coetzee"],
            "bowlers": ["Kagiso Rabada", "Anrich Nortje", "Marco Jansen",
                        "Tabraiz Shamsi", "Gerald Coetzee"],
        },
        {
            "name": "New Zealand", "abbr": "NZ", "color": "#000000",
            "batsmen": ["Devon Conway", "Finn Allen", "Kane Williamson",
                        "Glenn Phillips", "Daryl Mitchell", "James Neesham",
                        "Mitchell Santner", "Tim Southee", "Trent Boult",
                        "Matt Henry", "Lockie Ferguson"],
            "bowlers": ["Tim Southee", "Trent Boult", "Matt Henry",
                        "Mitchell Santner", "Lockie Ferguson"],
        },
        {
            "name": "Pakistan", "abbr": "PAK", "color": "#01411C",
            "batsmen": ["Babar Azam", "Mohammad Rizwan", "Fakhar Zaman",
                        "Mohammad Haris", "Shadab Khan", "Iftikhar Ahmed",
                        "Imad Wasim", "Shaheen Afridi", "Naseem Shah",
                        "Haris Rauf", "Mohammad Nawaz"],
            "bowlers": ["Shaheen Afridi", "Naseem Shah", "Haris Rauf",
                        "Shadab Khan", "Imad Wasim"],
        },
    ]

    VENUES = [
        "Wankhede Stadium, Mumbai",
        "Eden Gardens, Kolkata",
        "M. Chinnaswamy Stadium, Bengaluru",
        "MCG, Melbourne",
        "Lord's Cricket Ground, London",
        "Sydney Cricket Ground",
        "Newlands, Cape Town",
        "National Stadium, Karachi",
    ]

    PITCH_TYPES = ["flat", "seaming", "turning", "bouncy"]

    def __init__(self, api_key=None, ball_interval=2.0):
        self.api_key = api_key
        self.demo_mode = True  # always demo until real API integrated
        self.ball_interval = ball_interval   # seconds between simulated balls

        self._lock = threading.Lock()
        self._running = False
        self._thread = None
        self._on_match_complete = None

        self.match_state = self._empty_state()
        self.ball_log = []

        self._start()

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------

    def _empty_state(self):
        return {
            "status": "loading",
            "innings": 1,
            "batting_team": "",
            "batting_team_abbr": "",
            "batting_team_color": "#FFFFFF",
            "bowling_team": "",
            "bowling_team_abbr": "",
            "bowling_team_color": "#AAAAAA",
            "score": 0,
            "wickets": 0,
            "overs": "0.0",
            "crr": 0.0,
            "target": None,
            "rrr": None,
            "phase": "powerplay",
            "over_scores": [],
            "recent_balls": [],
            "pressure_index": 0.0,
            "partnership_runs": 0,
            "partnership_balls": 0,
            "last3_runrate": 0.0,
            "pitch_type": "flat",
            "match_name": "",
            "venue": "",
            "updated_at": time.time(),
            "over_just_completed": False,
            "current_batsmen": ["", ""],
            "current_bowler": "",
            "fall_of_wickets": [],
            "innings1_score": None,
            "innings1_wickets": None,
        }

    # ------------------------------------------------------------------
    # Delivery generator
    # ------------------------------------------------------------------

    def _generate_delivery(self, over, ball_in_over, score, wickets,
                           batting_strength=0.75, pitch_type="flat"):
        """Generate a single realistic delivery with all metadata."""
        # ---- Phase ----
        if over < 6:
            phase = "powerplay"
        elif over < 15:
            phase = "middle"
        else:
            phase = "death"

        # ---- Bowler type ----
        if over in (0, 1, 2, 3, 18, 19):
            bowler_type = "pace"
        elif 6 <= over <= 11:
            bowler_type = random.choice(["pace", "spin", "spin"])
        else:
            bowler_type = random.choice(["pace", "spin"])

        # ---- Length ----
        if phase == "death":
            lw = {"short": 0.10, "good": 0.35, "full": 0.35, "yorker": 0.20}
        elif phase == "powerplay":
            lw = {"short": 0.18, "good": 0.45, "full": 0.28, "yorker": 0.09}
        else:
            lw = {"short": 0.15, "good": 0.48, "full": 0.28, "yorker": 0.09}

        length = random.choices(list(lw), weights=list(lw.values()))[0]

        # ---- Line ----
        line = random.choices(
            ["off", "middle", "leg", "wide_off", "wide_leg"],
            weights=[0.34, 0.30, 0.22, 0.09, 0.05],
        )[0]

        # ---- Speed / spin ----
        if bowler_type == "pace":
            if pitch_type == "bouncy":
                speed = round(random.gauss(143, 7), 1)
            else:
                speed = round(random.gauss(138, 8), 1)
            spin_rpm = 0
        else:
            speed = round(random.gauss(84, 5), 1)
            spin_rpm = int(random.gauss(2200, 300))

        # ---- Extras ----
        is_wide = (line in ("wide_off", "wide_leg")) and random.random() < 0.65
        is_noball = random.random() < 0.018

        # ---- Wicket probability ----
        wp = 0.055 - (batting_strength * 0.02)
        if length == "good":
            wp += 0.012
        if pitch_type in ("seaming", "turning"):
            wp += 0.008
        if phase == "death" and batting_strength < 0.55:
            wp += 0.015
        if wickets >= 7:
            wp += 0.025
        is_wicket = (not is_wide) and (not is_noball) and (random.random() < max(0, wp))

        # ---- Runs ----
        if is_wide:
            runs = 1
        elif is_noball:
            runs = random.choices([0, 1, 2, 4, 6], weights=[25, 35, 15, 15, 10])[0]
        elif is_wicket:
            runs = 0
        else:
            if phase == "powerplay":
                wts = [28, 30, 10, 4, 15, 13]
            elif phase == "death":
                wts = [17, 21, 10, 4, 24, 24]
            else:
                wts = [36, 31, 12, 4, 12, 5]
            runs = random.choices([0, 1, 2, 3, 4, 6], weights=wts)[0]

        # ---- Bounce coordinates (top-down, y=0 near batsman, y=1 near bowler) ----
        by_map = {
            "yorker": (0.07, 0.18),
            "full":   (0.22, 0.38),
            "good":   (0.40, 0.58),
            "short":  (0.60, 0.76),
        }
        bx_map = {
            "off":      (0.58, 0.72),
            "middle":   (0.44, 0.56),
            "leg":      (0.28, 0.42),
            "wide_off": (0.74, 0.92),
            "wide_leg": (0.08, 0.26),
        }
        bounce_y = round(random.uniform(*by_map[length]), 3)
        bounce_x = round(random.uniform(*bx_map[line]), 3)

        # ---- Deviation (seam / spin) ----
        if bowler_type == "pace":
            dev = round(abs(random.gauss(1.8, 1.3)), 2)
            dev_dir = random.choice(["in", "out"])
        else:
            dev = round(abs(random.gauss(3.8, 1.8)), 2)
            dev_dir = random.choice(["off", "leg"])

        return {
            "over": over,
            "ball": ball_in_over + 1,
            "runs": runs,
            "wicket": is_wicket,
            "wide": is_wide,
            "noball": is_noball,
            "length": length,
            "line": line,
            "speed": speed,
            "bowler_type": bowler_type,
            "spin_rpm": spin_rpm,
            "bounce_x": bounce_x,
            "bounce_y": bounce_y,
            "deviation": dev,
            "deviation_dir": dev_dir,
            "phase": phase,
        }

    # ------------------------------------------------------------------
    # Innings simulator
    # ------------------------------------------------------------------

    def _simulate_innings(self, batting_team, bowling_team, pitch_type,
                          venue, innings_num, target):
        score = 0
        wickets = 0
        over_scores = []
        ball_log = []
        batsmen_idx = [0, 1]
        next_idx = 2
        partnership_runs = 0
        partnership_balls = 0
        fall_of_wickets = []

        with self._lock:
            self.ball_log = []
            self.match_state = self._empty_state()
            self.match_state.update({
                "status": "live",
                "innings": innings_num,
                "batting_team": batting_team["name"],
                "batting_team_abbr": batting_team["abbr"],
                "batting_team_color": batting_team["color"],
                "bowling_team": bowling_team["name"],
                "bowling_team_abbr": bowling_team["abbr"],
                "bowling_team_color": bowling_team["color"],
                "target": target,
                "pitch_type": pitch_type,
                "match_name": f"{batting_team['name']} vs {bowling_team['name']} — T20",
                "venue": venue,
                "current_batsmen": [batting_team["batsmen"][0],
                                    batting_team["batsmen"][1]],
                "current_bowler": random.choice(bowling_team["bowlers"]),
                "updated_at": time.time(),
            })

        for over in range(20):
            if wickets >= 10 or not self._running:
                break
            if target and score >= target:
                break

            bowler = random.choice(bowling_team["bowlers"])
            current_over_runs = 0
            ball_in_over = 0

            with self._lock:
                self.match_state["over_just_completed"] = False

            while ball_in_over < 6 and wickets < 10 and self._running:
                if target and score >= target:
                    break

                batting_strength = max(0.3, 0.88 - wickets * 0.08)
                d = self._generate_delivery(over, ball_in_over, score,
                                            wickets, batting_strength, pitch_type)

                score += d["runs"]
                current_over_runs += d["runs"]

                if d["wicket"]:
                    wickets += 1
                    fall_of_wickets.append({
                        "score": score, "wicket": wickets,
                        "over": f"{over}.{ball_in_over + 1}",
                    })
                    if next_idx < len(batting_team["batsmen"]):
                        batsmen_idx[0] = next_idx
                        next_idx += 1
                    partnership_runs = 0
                    partnership_balls = 0
                else:
                    partnership_runs += d["runs"]
                    if not d["wide"]:
                        partnership_balls += 1

                ball_in_over += 1
                balls_done = over * 6 + ball_in_over
                overs_done = over + ball_in_over / 6
                crr = round(score / overs_done, 2) if overs_done > 0 else 0.0

                rrr = None
                if target:
                    remaining_balls = 120 - balls_done
                    runs_needed = target - score
                    if remaining_balls > 0 and runs_needed > 0:
                        rrr = round(runs_needed / (remaining_balls / 6), 2)
                    elif runs_needed <= 0:
                        rrr = 0.0

                # Pressure index
                if rrr is not None and crr > 0:
                    pressure = round(min(5.0, (rrr / crr) * (1 + wickets * 0.07)), 2)
                else:
                    pressure = round(min(5.0, wickets * 0.12), 2)

                # Last-3-overs run rate
                recent_balls = ball_log[-18:] if len(ball_log) >= 18 else ball_log
                last3_rr = round(
                    sum(b["runs"] for b in recent_balls) / len(recent_balls) * 6, 2
                ) if recent_balls else crr

                d.update({
                    "cumulative_score": score,
                    "cumulative_wickets": wickets,
                    "cumulative_balls": balls_done,
                })
                ball_log.append(d)

                phase = d["phase"]
                b0 = batting_team["batsmen"][batsmen_idx[0]] if batsmen_idx[0] < len(batting_team["batsmen"]) else "Unknown"
                b1 = batting_team["batsmen"][batsmen_idx[1]] if batsmen_idx[1] < len(batting_team["batsmen"]) else "Unknown"

                with self._lock:
                    self.ball_log = ball_log[:]
                    self.match_state.update({
                        "score": score,
                        "wickets": wickets,
                        "overs": f"{over}.{ball_in_over}",
                        "crr": crr,
                        "rrr": rrr,
                        "phase": phase,
                        "recent_balls": ball_log[-18:],
                        "last3_runrate": last3_rr,
                        "partnership_runs": partnership_runs,
                        "partnership_balls": partnership_balls,
                        "pressure_index": pressure,
                        "fall_of_wickets": fall_of_wickets,
                        "current_batsmen": [b0, b1],
                        "current_bowler": bowler,
                        "over_just_completed": False,
                        "updated_at": time.time(),
                    })

                time.sleep(self.ball_interval)

            over_scores.append(current_over_runs)
            with self._lock:
                self.match_state["over_scores"] = over_scores
                self.match_state["overs"] = f"{over + 1}.0"
                self.match_state["over_just_completed"] = True
                self.match_state["current_bowler"] = random.choice(bowling_team["bowlers"])

        with self._lock:
            self.match_state["final_score"] = score
            self.match_state["final_wickets"] = wickets

        return score, wickets, ball_log

    # ------------------------------------------------------------------
    # Full match runner
    # ------------------------------------------------------------------

    def _simulate_match(self):
        teams = random.sample(self.TEAMS, 2)
        bat1, bat2 = teams[0], teams[1]
        pitch_type = random.choice(self.PITCH_TYPES)
        venue = random.choice(self.VENUES)

        s1, w1, log1 = self._simulate_innings(bat1, bat2, pitch_type,
                                               venue, 1, target=None)
        if not self._running:
            return

        # Innings break
        with self._lock:
            self.match_state["status"] = "innings_break"
            self.match_state["innings1_score"] = s1
            self.match_state["innings1_wickets"] = w1
        time.sleep(6)

        if not self._running:
            return

        s2, w2, log2 = self._simulate_innings(bat2, bat1, pitch_type,
                                               venue, 2, target=s1 + 1)

        # Determine winner
        if s2 > s1:
            winner = bat2["name"]
        elif s1 > s2:
            winner = bat1["name"]
        else:
            winner = "Tie"

        with self._lock:
            self.match_state["status"] = "completed"
            self.match_state["winner"] = winner

        match_record = {
            "match_name": f"{bat1['name']} vs {bat2['name']}",
            "venue": venue,
            "pitch_type": pitch_type,
            "innings1": {"team": bat1["name"], "score": s1, "wickets": w1},
            "innings2": {"team": bat2["name"], "score": s2, "wickets": w2},
            "ball_log": log1,
            "final_score": s1,
            "timestamp": time.time(),
        }

        if self._on_match_complete:
            try:
                self._on_match_complete(match_record)
            except Exception:
                pass

    def _run(self):
        while self._running:
            self._simulate_match()
            if self._running:
                time.sleep(5)  # pause between matches

    def _start(self):
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_match_complete_callback(self, cb):
        self._on_match_complete = cb

    def get_live_state(self):
        with self._lock:
            return dict(self.match_state)

    def get_pitch_data(self):
        with self._lock:
            return list(self.ball_log)

    def stop(self):
        self._running = False
