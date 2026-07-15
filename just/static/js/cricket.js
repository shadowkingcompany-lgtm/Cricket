/**
 * cricket.js
 * ----------
 * Drives the Live Cricket Intelligence Dashboard.
 *
 * • Polls /api/cricket/live every 3 seconds
 * • Renders the Hawkeye pitch map on <canvas>
 * • Updates score, ball feed, stats, match situation
 * • Draws Chart.js score-progression + prediction band
 * • Shows adaptive ML prediction with feature-importance bars
 */

'use strict';

// ─── State ────────────────────────────────────────────────────────────────────
let lastState       = null;
let lastPrediction  = null;
let scoreChart      = null;
let prevScore       = 0;
let pollTimer       = null;
const POLL_MS       = 3000;

// ─── Canvas setup ─────────────────────────────────────────────────────────────
const canvas    = document.getElementById('hawkeyeCanvas');
const ctx       = canvas ? canvas.getContext('2d') : null;

// Pitch geometry (normalised coords → pixel coords)
const PITCH = {
  get w()      { return canvas ? canvas.width : 260; },
  get h()      { return canvas ? canvas.height : 480; },
  get padX()   { return Math.round(this.w * 0.22); },
  get padY()   { return Math.round(this.h * 0.04); },
  get pitchW() { return this.w - this.padX * 2; },
  get pitchH() { return this.h - this.padY * 2; },
  toX(nx) { return this.padX + nx * this.pitchW; },
  toY(ny) { return this.padY + ny * this.pitchH; },
};

// ─── Colour helpers ───────────────────────────────────────────────────────────
function ballColour(d) {
  if (d.wicket)        return { fill: '#EF4444', label: 'W',  cls: 'wkt'  };
  if (d.wide)          return { fill: '#F97316', label: 'Wd', cls: 'wide' };
  if (d.noball)        return { fill: '#A855F7', label: 'NB', cls: 'nb'   };
  if (d.runs === 6)    return { fill: '#FFD700', label: '6',  cls: 'six'  };
  if (d.runs === 4)    return { fill: '#10B981', label: '4',  cls: 'four' };
  if (d.runs === 3)    return { fill: '#FBBF24', label: '3',  cls: 'two'  };
  if (d.runs === 2)    return { fill: '#FBBF24', label: '2',  cls: 'two'  };
  if (d.runs === 1)    return { fill: '#F59E0B', label: '1',  cls: 'one'  };
  return                      { fill: '#3A5080', label: '·',  cls: 'dot'  };
}

// ─── Hawkeye canvas renderer ──────────────────────────────────────────────────
function drawPitch(deliveries) {
  if (!ctx) return;

  const W = canvas.width;
  const H = canvas.height;
  ctx.clearRect(0, 0, W, H);

  // ── Outfield (dark) ──
  ctx.fillStyle = '#0A1420';
  ctx.fillRect(0, 0, W, H);

  // ── Pitch strip ──
  const px = PITCH.padX, py = PITCH.padY;
  const pw = PITCH.pitchW, ph = PITCH.pitchH;

  const pitchGrad = ctx.createLinearGradient(px, py, px, py + ph);
  pitchGrad.addColorStop(0,   '#7A6845');
  pitchGrad.addColorStop(0.5, '#8B7A52');
  pitchGrad.addColorStop(1,   '#7A6845');
  ctx.fillStyle = pitchGrad;
  roundRect(ctx, px, py, pw, ph, 6);
  ctx.fill();

  // Pitch border
  ctx.strokeStyle = 'rgba(255,255,255,0.08)';
  ctx.lineWidth = 1;
  roundRect(ctx, px, py, pw, ph, 6);
  ctx.stroke();

  // ── Length zone overlays (subtle) ──
  const zones = [
    { label: 'YORKER',      y: 0.07, h: 0.11, alpha: 0.06 },
    { label: 'FULL',        y: 0.22, h: 0.16, alpha: 0.05 },
    { label: 'GOOD LENGTH', y: 0.40, h: 0.18, alpha: 0.04 },
    { label: 'SHORT',       y: 0.60, h: 0.16, alpha: 0.05 },
  ];
  zones.forEach(z => {
    ctx.fillStyle = `rgba(0,212,255,${z.alpha})`;
    ctx.fillRect(px, PITCH.toY(z.y), pw, ph * z.h);

    ctx.fillStyle = 'rgba(255,255,255,0.18)';
    ctx.font = `bold ${Math.max(8, pw * 0.08)}px Inter`;
    ctx.textAlign = 'left';
    ctx.fillText(z.label, px + 4, PITCH.toY(z.y) + 11);
  });
  ctx.textAlign = 'center';

  // ── Crease lines ──
  ctx.strokeStyle = 'rgba(255,255,255,0.55)';
  ctx.lineWidth = 1.2;

  const batCreaseY  = PITCH.toY(0.08);
  const bowlCreaseY = PITCH.toY(0.88);

  // Batting crease
  hLine(ctx, px - 8, batCreaseY, pw + 16);
  // Popping crease
  ctx.setLineDash([4, 4]);
  hLine(ctx, px - 8, PITCH.toY(0.10), pw + 16);
  ctx.setLineDash([]);
  // Bowling crease
  hLine(ctx, px - 8, bowlCreaseY, pw + 16);
  ctx.setLineDash([4, 4]);
  hLine(ctx, px - 8, PITCH.toY(0.86), pw + 16);
  ctx.setLineDash([]);

  // Return creases
  ctx.strokeStyle = 'rgba(255,255,255,0.35)';
  // Batsman end
  vLine(ctx, px + pw * 0.3, batCreaseY - 6, 18);
  vLine(ctx, px + pw * 0.7, batCreaseY - 6, 18);
  // Bowler end
  vLine(ctx, px + pw * 0.3, bowlCreaseY - 6, 18);
  vLine(ctx, px + pw * 0.7, bowlCreaseY - 6, 18);

  // ── Wickets (3 stumps each end) ──
  drawStumps(ctx, px + pw * 0.5, batCreaseY,  pw);
  drawStumps(ctx, px + pw * 0.5, bowlCreaseY, pw);

  // ── Deliveries (heat-map accumulation) ──
  const total = deliveries.length;
  deliveries.forEach((d, i) => {
    const cx = PITCH.toX(d.bounce_x);
    const cy = PITCH.toY(d.bounce_y);
    const age = (i + 1) / total;            // newer = closer to 1
    const c   = ballColour(d);

    // Release point (bowling crease, middle stump)
    const releaseX = px + pw * 0.5;
    const releaseY = bowlCreaseY;

    // ── Delivery path line ──
    if (age > 0.92) {                       // only last few balls get arrow
      ctx.save();
      ctx.globalAlpha = 0.35;
      ctx.strokeStyle = c.fill;
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(releaseX, releaseY);
      ctx.lineTo(cx, cy);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.restore();
    }

    // ── Glow ──
    const r = 12 * age;
    const grd = ctx.createRadialGradient(cx, cy, 0, cx, cy, r + 4);
    grd.addColorStop(0, hexAlpha(c.fill, 0.5 * age));
    grd.addColorStop(1, hexAlpha(c.fill, 0));
    ctx.fillStyle = grd;
    ctx.beginPath();
    ctx.arc(cx, cy, r + 4, 0, Math.PI * 2);
    ctx.fill();

    // ── Core dot ──
    ctx.fillStyle = hexAlpha(c.fill, 0.85 * age + 0.15);
    ctx.beginPath();
    ctx.arc(cx, cy, Math.max(2.5, 4 * age), 0, Math.PI * 2);
    ctx.fill();
  });

  // ── Labels (line side) ──
  ctx.font = '9px Inter';
  ctx.fillStyle = 'rgba(255,255,255,0.25)';
  ctx.textAlign = 'right';
  ctx.fillText('LEG', px - 3, H * 0.5);
  ctx.textAlign = 'left';
  ctx.fillText('OFF', px + pw + 3, H * 0.5);
  ctx.textAlign = 'center';

  // Bat end / bowl end labels
  ctx.fillStyle = 'rgba(255,255,255,0.22)';
  ctx.font = '9px Inter';
  ctx.fillText('BATTING END', px + pw / 2, py - 4);
  ctx.fillText('BOWLING END', px + pw / 2, py + ph + 10);
}

// ─── Canvas helpers ───────────────────────────────────────────────────────────
function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + w - r, y);
  ctx.quadraticCurveTo(x + w, y, x + w, y + r);
  ctx.lineTo(x + w, y + h - r);
  ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
  ctx.lineTo(x + r, y + h);
  ctx.quadraticCurveTo(x, y + h, x, y + h - r);
  ctx.lineTo(x, y + r);
  ctx.quadraticCurveTo(x, y, x + r, y);
  ctx.closePath();
}

function hLine(ctx, x, y, w) {
  ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x + w, y); ctx.stroke();
}
function vLine(ctx, x, y, h) {
  ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x, y + h); ctx.stroke();
}

function drawStumps(ctx, cx, cy, pitchWidth) {
  const gap  = pitchWidth * 0.038;
  const h    = 10;
  const xs   = [cx - gap, cx, cx + gap];
  ctx.strokeStyle = '#E8C87A';
  ctx.lineWidth   = 2;
  xs.forEach(x => {
    ctx.beginPath();
    ctx.moveTo(x, cy - h);
    ctx.lineTo(x, cy + 3);
    ctx.stroke();
  });
  // Bails
  ctx.strokeStyle = '#E8C87A';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(xs[0] - 1, cy - h);
  ctx.lineTo(xs[2] + 1, cy - h);
  ctx.stroke();
}

function hexAlpha(hex, alpha) {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${r},${g},${b},${alpha.toFixed(2)})`;
}

// ─── Chart.js – score progression ────────────────────────────────────────────
function initChart() {
  const el = document.getElementById('scoreChart');
  if (!el) return;
  const chartCtx = el.getContext('2d');

  scoreChart = new Chart(chartCtx, {
    type: 'bar',
    data: {
      labels:   [],
      datasets: [
        {
          type: 'bar',
          label: 'Runs / Over',
          data: [],
          backgroundColor: 'rgba(0,212,255,0.25)',
          borderColor:     'rgba(0,212,255,0.6)',
          borderWidth: 1,
          borderRadius: 3,
          yAxisID: 'y',
        },
        {
          type: 'line',
          label: 'Cumulative Score',
          data: [],
          borderColor: '#00FF88',
          backgroundColor: 'rgba(0,255,136,0.06)',
          borderWidth: 2,
          pointRadius: 2,
          tension: 0.35,
          fill: true,
          yAxisID: 'y2',
        },
        {
          type: 'line',
          label: 'Predicted Final',
          data: [],
          borderColor: '#FFD700',
          borderDash: [6, 4],
          borderWidth: 1.5,
          pointRadius: 0,
          fill: false,
          yAxisID: 'y2',
        },
        {
          type: 'line',
          label: 'Upper Bound',
          data: [],
          borderColor: 'rgba(255,215,0,0)',
          backgroundColor: 'rgba(255,215,0,0.06)',
          borderWidth: 0,
          pointRadius: 0,
          fill: '+1',
          yAxisID: 'y2',
        },
        {
          type: 'line',
          label: 'Lower Bound',
          data: [],
          borderColor: 'rgba(255,215,0,0)',
          borderWidth: 0,
          pointRadius: 0,
          fill: false,
          yAxisID: 'y2',
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: 'rgba(10,14,26,0.95)',
          borderColor: 'rgba(0,212,255,0.2)',
          borderWidth: 1,
          titleColor: '#8899BB',
          bodyColor: '#E8EDF7',
          callbacks: {
            label: ctx => {
              if (ctx.datasetIndex === 0) return `  Runs: ${ctx.parsed.y}`;
              if (ctx.datasetIndex === 1) return `  Cumulative: ${ctx.parsed.y}`;
              if (ctx.datasetIndex === 2) return `  Prediction: ${ctx.parsed.y}`;
              return null;
            },
          },
        },
      },
      scales: {
        x: {
          grid: { color: 'rgba(255,255,255,0.04)' },
          ticks: { color: '#4D5F80', font: { size: 10 } },
        },
        y: {
          position: 'right',
          grid: { display: false },
          ticks: { color: '#4D5F80', font: { size: 10 } },
          title: { display: true, text: 'Per Over', color: '#4D5F80', font: { size: 9 } },
        },
        y2: {
          grid: { color: 'rgba(255,255,255,0.04)' },
          ticks: { color: '#4D5F80', font: { size: 10 } },
          title: { display: true, text: 'Score', color: '#4D5F80', font: { size: 9 } },
        },
      },
    },
  });
}

function updateChart(state, prediction) {
  if (!scoreChart) return;
  const overScores = state.over_scores || [];
  const labels     = overScores.map((_, i) => `Ov ${i + 1}`);
  const cumulative = overScores.reduce((acc, v, i) => {
    acc.push((acc[i - 1] || 0) + v); return acc;
  }, []);
  const predFinal = prediction ? prediction.predicted_score : null;
  const upper     = prediction ? prediction.upper           : null;
  const lower     = prediction ? prediction.lower           : null;
  const maxOvers  = 20;
  const allLabels = Array.from({ length: maxOvers }, (_, i) => `Ov ${i + 1}`);

  scoreChart.data.labels                 = allLabels;
  scoreChart.data.datasets[0].data       = [...overScores, ...Array(maxOvers - overScores.length).fill(null)];
  scoreChart.data.datasets[1].data       = [...cumulative,  ...Array(maxOvers - cumulative.length).fill(null)];
  scoreChart.data.datasets[2].data       = predFinal ? Array(maxOvers).fill(predFinal) : [];
  scoreChart.data.datasets[3].data       = upper     ? Array(maxOvers).fill(upper)     : [];
  scoreChart.data.datasets[4].data       = lower     ? Array(maxOvers).fill(lower)     : [];
  scoreChart.update('none');
}

// ─── DOM helpers ──────────────────────────────────────────────────────────────
function el(id)          { return document.getElementById(id); }
function setText(id, v)  { const e = el(id); if (e) e.textContent = v ?? '—'; }
function setHTML(id, v)  { const e = el(id); if (e) e.innerHTML = v ?? ''; }

// ─── State → DOM update ───────────────────────────────────────────────────────
function updateUI(data) {
  const s    = data.match_state;
  const pred = data.prediction;
  const ms   = data.model_stats;

  // ── Loading overlay ──
  if (s.status !== 'loading') {
    const ov = el('loadingOverlay');
    if (ov && !ov.classList.contains('hidden')) ov.classList.add('hidden');
  }

  // ── Header ──
  setText('matchName',   s.match_name  || 'Loading…');
  setText('matchVenue',  s.venue       || '');
  updateStatusBadge(s.status);
  updatePhaseIndicator(s.phase);
  updateLiveBadge(s.status);

  // ── Score banner ──
  setText('batTeamName',   s.batting_team  || '—');
  setText('batTeamAbbr',   s.batting_team_abbr || '—');
  setText('bowlTeamName',  s.bowling_team  || '—');
  setText('bowlTeamAbbr',  s.bowling_team_abbr || '—');

  const badgeB = el('batTeamBadge');
  const badgeW = el('bowlTeamBadge');
  if (badgeB) badgeB.style.background = s.batting_team_color || '#0066B4';
  if (badgeW) badgeW.style.background = s.bowling_team_color || '#AAAAAA';
  if (badgeB) badgeB.textContent = s.batting_team_abbr || '?';
  if (badgeW) badgeW.textContent = s.bowling_team_abbr || '?';

  // Score with flash
  const scoreEl = el('currentScore');
  if (scoreEl && s.score !== prevScore) {
    scoreEl.classList.remove('score-flash');
    void scoreEl.offsetWidth;
    scoreEl.classList.add('score-flash');
    prevScore = s.score;
  }
  setHTML('currentScore', `${s.score}<span class="wkts">/${s.wickets}</span>`);
  setText('currentOvers', `${s.overs} overs`);
  setText('currentCRR',   `CRR: ${(s.crr || 0).toFixed(2)}`);

  // Target / RRR
  const targetEl = el('targetPill');
  if (s.target && s.innings === 2) {
    if (targetEl) targetEl.style.display = '';
    setText('targetScore', s.target);
    setText('rrr', s.rrr ? s.rrr.toFixed(2) : '—');
  } else {
    if (targetEl) targetEl.style.display = 'none';
  }

  // Stat chips
  setText('chipCRR',  (s.crr  || 0).toFixed(2));
  setText('chipRRR',  s.rrr ? s.rrr.toFixed(2) : '—');
  setText('chipL3RR', (s.last3_runrate || 0).toFixed(2));

  const wih = 10 - (s.wickets || 0);
  setText('chipWIH', wih);
  const wihChip = el('chipWIH')?.parentElement;
  if (wihChip) {
    wihChip.className = 'stat-chip ' + (wih <= 3 ? 'danger' : wih <= 6 ? 'highlight' : 'good');
  }

  // ── Left panel ──
  updateBatsmen(s);
  updateBallFeed(s.recent_balls || []);
  updatePartnership(s);
  updateFOW(s.fall_of_wickets || []);
  updateOverBars(s.over_scores || []);

  // ── Centre panel ──
  drawPitch(data.pitch_data || []);
  updatePitchAnalytics(data.pitch_data || [], s);
  updateChart(s, pred);

  // ── Right panel ──
  if (pred) updatePrediction(pred);
  if (ms)   updateModelInfo(ms);
  updatePressure(s.pressure_index || 0);

  // Timestamp
  setText('lastUpdate', `Updated ${new Date().toLocaleTimeString()}`);

  lastState      = s;
  lastPrediction = pred;
}

// ── Status badge ──
function updateStatusBadge(status) {
  const el = document.getElementById('statusBadge');
  if (!el) return;
  const map = {
    live:          ['live',      'LIVE'],
    innings_break: ['break',     'BREAK'],
    completed:     ['completed', 'FULL TIME'],
    loading:       ['loading',   'LOADING'],
  };
  const [cls, label] = map[status] || map.loading;
  el.className = `status-badge ${cls}`;
  el.textContent = label;
}

function updateLiveBadge(status) {
  const el = document.getElementById('liveBadge');
  if (!el) return;
  el.style.display = status === 'live' ? '' : 'none';
}

function updatePhaseIndicator(phase) {
  const el = document.getElementById('phaseIndicator');
  if (!el) return;
  const labels = { powerplay: 'Powerplay', middle: 'Middle Overs', death: 'Death Overs' };
  el.className = `phase-indicator ${phase || 'middle'}`;
  el.innerHTML = `<span class="ph-dot"></span>${labels[phase] || phase}`;
}

// ── Batsmen ──
function updateBatsmen(s) {
  const batsmen = s.current_batsmen || ['', ''];
  const bowler  = s.current_bowler  || '';
  setHTML('batsmenTable', `
    <table class="batsmen-table">
      <thead>
        <tr><th>Batsman</th><th>R</th><th>B</th><th>SR</th></tr>
      </thead>
      <tbody>
        <tr class="on-strike">
          <td class="bat-name">${esc(batsmen[0])}</td>
          <td>—</td><td>—</td><td>—</td>
        </tr>
        <tr>
          <td class="bat-name">${esc(batsmen[1])}</td>
          <td>—</td><td>—</td><td>—</td>
        </tr>
      </tbody>
    </table>
    <div class="bowler-row">Bowling: <span>${esc(bowler)}</span></div>
  `);
}

// ── Ball feed ──
function updateBallFeed(balls) {
  const container = el('ballFeed');
  if (!container) return;
  const prev = container.innerHTML;
  const html = balls.map((b, i) => {
    const c = ballColour(b);
    const isNew = (i === balls.length - 1) && prev !== '';
    return `<div class="ball-dot ${c.cls} ${isNew ? 'new-ball' : ''}" title="Ov ${b.over+1}.${b.ball} | ${b.length} | ${b.line} | ${b.speed}kph">${c.label}</div>`;
  }).join('');
  container.innerHTML = html;
}

// ── Partnership ──
function updatePartnership(s) {
  setText('pshipRuns',  s.partnership_runs  || 0);
  setText('pshipBalls', s.partnership_balls || 0);
}

// ── Fall of wickets ──
function updateFOW(fow) {
  if (!fow.length) {
    setHTML('fowList', '<div class="empty-state" style="padding:12px">No wickets yet</div>');
    return;
  }
  setHTML('fowList', fow.map(f =>
    `<div class="fow-item">
       <span class="fow-score">${f.wicket}-${f.score}</span>
       <span class="fow-over">Ov ${f.over}</span>
     </div>`
  ).join(''));
}

// ── Over bars ──
function updateOverBars(overScores) {
  const container = el('overBars');
  if (!container) return;
  const max = Math.max(...overScores, 1);
  container.innerHTML = overScores.map((r, i) => {
    const pct = Math.round((r / max) * 100);
    const col = r >= 15 ? '#00FF88' : r >= 10 ? '#00D4FF' : r >= 6 ? '#FFD700' : '#FF4455';
    return `<div class="over-bar" data-runs="${r} runs (Ov ${i+1})"
               style="height:${pct}%;background:${col};flex:1 0 auto"></div>`;
  }).join('');
}

// ── Pitch analytics ──
function updatePitchAnalytics(pitchData, state) {
  const total = pitchData.length || 1;
  const dots    = pitchData.filter(b => !b.wicket && !b.wide && b.runs === 0).length;
  const fours   = pitchData.filter(b => b.runs === 4).length;
  const sixes   = pitchData.filter(b => b.runs === 6).length;
  const wickets = pitchData.filter(b => b.wicket).length;
  const spin    = pitchData.filter(b => b.bowler_type === 'spin').length;
  const pace    = pitchData.filter(b => b.bowler_type === 'pace').length;
  const avgDev  = pitchData.reduce((s, b) => s + (b.deviation || 0), 0) / total;
  const avgSpd  = pitchData.filter(b => b.bowler_type === 'pace')
                            .reduce((s, b) => s + b.speed, 0) / (pace || 1);

  setText('statDotPct',  `${Math.round(dots   / total * 100)}%`);
  setText('statBndyPct', `${Math.round((fours + sixes) / total * 100)}%`);
  setText('statSpinPct', `${Math.round(spin   / total * 100)}%`);
  setText('statAvgDev',  `${avgDev.toFixed(1)}°`);
  setText('statAvgSpd',  `${Math.round(avgSpd)} kph`);
  setText('statPitch',   state.pitch_type ? state.pitch_type.charAt(0).toUpperCase() + state.pitch_type.slice(1) : '—');

  const rpo = (state.score || 0) / (Math.max(0.1, parseFloat(state.overs || '0')));
  setText('statRPO', rpo.toFixed(2));
  setText('totalBalls', total);
}

// ── Prediction panel ──
function updatePrediction(pred) {
  const scoreEl = el('predScore');
  if (scoreEl) {
    const old = parseInt(scoreEl.textContent) || 0;
    if (old !== pred.predicted_score) {
      scoreEl.style.transform = 'scale(1.08)';
      setTimeout(() => { scoreEl.style.transform = 'scale(1)'; }, 300);
    }
    scoreEl.textContent = pred.predicted_score;
  }
  setText('predLower', pred.lower);
  setText('predUpper', pred.upper);
  setText('predModel', pred.model_type || '—');

  // Confidence bar
  const pct = Math.round((pred.confidence || 0) * 100);
  setText('confPct', `${pct}%`);
  const bar = el('confBar');
  if (bar) bar.style.width = `${pct}%`;

  // Feature importance
  const fi    = pred.feature_importance || {};
  const fiEl  = el('featureImportance');
  if (!fiEl) return;
  const sorted = Object.entries(fi).sort((a, b) => b[1] - a[1]);
  const max = sorted[0]?.[1] || 1;
  fiEl.innerHTML = sorted.slice(0, 6).map(([name, val]) => {
    const pct = Math.round((val / max) * 100);
    return `
      <div class="fi-item">
        <span class="fi-name">${name}</span>
        <div class="fi-bar-bg"><div class="fi-bar-fill" style="width:${pct}%"></div></div>
        <span class="fi-pct">${(val * 100).toFixed(1)}%</span>
      </div>`;
  }).join('');
}

// ── Model info ──
function updateModelInfo(ms) {
  setText('miSamples',  ms.total_samples  || 0);
  setText('miMatches',  ms.completed_matches || 0);
  setText('miType',     ms.model_type || '—');
}

// ── Pressure gauge ──
function updatePressure(pressure) {
  const pct = Math.round(Math.min(1, pressure / 5) * 100);
  const bar = el('pressureBar');
  if (bar) {
    bar.style.width = `${pct}%`;
    if (pressure >= 3.5) bar.style.background = 'linear-gradient(90deg,#FF7700,#FF4455)';
    else if (pressure >= 2) bar.style.background = 'linear-gradient(90deg,#FFD700,#FF7700)';
    else bar.style.background = 'linear-gradient(90deg,#00FF88,#00D4FF)';
  }
  setText('pressureVal', pressure.toFixed(2));
}

// ─── Escape HTML ──────────────────────────────────────────────────────────────
function esc(str) {
  return String(str || '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}

// ─── API poll ─────────────────────────────────────────────────────────────────
async function poll() {
  try {
    const res  = await fetch('/api/cricket/live');
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    updateUI(data);
  } catch (err) {
    console.warn('Poll error:', err.message);
  } finally {
    pollTimer = setTimeout(poll, POLL_MS);
  }
}

// ─── Canvas resize ────────────────────────────────────────────────────────────
function resizeCanvas() {
  if (!canvas) return;
  const wrap = canvas.parentElement;
  if (!wrap) return;
  const w = Math.min(280, wrap.clientWidth - 20);
  canvas.width  = w;
  canvas.height = Math.round(w * 1.75);
  if (lastState) {
    fetch('/api/cricket/live')
      .then(r => r.json())
      .then(d => drawPitch(d.pitch_data || []))
      .catch(() => {});
  }
}

// ─── Boot ─────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  resizeCanvas();
  window.addEventListener('resize', resizeCanvas);
  initChart();
  poll();               // immediate first poll
});
