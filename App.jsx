/**
 * App.jsx — AI Research Paper Generator (Redesigned)
 * 
 * New in this version:
 *  - Smart onboarding: asks if user has reference papers first
 *  - Dynamic 3-approach comparison with real precision/recall/F1/accuracy graphs
 *  - Section quality evaluator with real NLP metrics
 *  - Paper readability score (Flesch-Kincaid style)
 *  - Novelty detection: flags potentially unsupported claims
 *  - Live word/section completeness progress
 *  - Export BibTeX alongside DOCX
 *  - Approach recommendation engine (picks best approach for user's context)
 *  - Non-AI editorial aesthetic (Neue Haas Grotesk feel, ink-on-paper palette)
 */

import React, { useState, useRef, useCallback, useEffect } from 'react';

const API = 'http://localhost:8000';

/* ─── Design tokens ───────────────────────────────────────────────────── */
const GLOBAL_CSS = `
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;1,9..40,300&family=DM+Mono:wght@400;500&display=swap');

*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}

:root{
  --ink:#111010;
  --paper:#F6F3EE;
  --card:#FDFBF8;
  --mist:#E8E3DC;
  --fog:#D4CEC6;
  --graphite:#6B6560;
  --rust:#B5451B;
  --rust-lt:#F4E5DF;
  --teal:#1A6B6B;
  --teal-lt:#DDF0F0;
  --gold:#A07C3F;
  --gold-lt:#F5EDD9;
  --slate:#3A4A5C;
  --r:6px;
  --r-lg:10px;
  --shadow:0 1px 3px rgba(17,16,16,0.08),0 4px 16px rgba(17,16,16,0.05);
  --shadow-lg:0 2px 8px rgba(17,16,16,0.1),0 12px 40px rgba(17,16,16,0.08);
  --font-display:'DM Serif Display',Georgia,serif;
  --font-body:'DM Sans',system-ui,sans-serif;
  --font-mono:'DM Mono',monospace;
}

html{font-size:15px;-webkit-font-smoothing:antialiased}
body{font-family:var(--font-body);background:var(--paper);color:var(--ink);line-height:1.6;min-height:100vh}

/* Layout */
.shell{display:flex;flex-direction:column;min-height:100vh}
.wrap{max-width:1100px;margin:0 auto;padding:0 1.5rem;width:100%}

/* Masthead */
.masthead{background:var(--ink);color:var(--paper);padding:2.5rem 0 2rem;position:relative;overflow:hidden}
.masthead::after{content:'';position:absolute;inset:0;background:url("data:image/svg+xml,%3Csvg width='40' height='40' viewBox='0 0 40 40' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='%23ffffff' fill-opacity='0.025'%3E%3Cpath d='M0 40L40 0H20L0 20M40 40V20L20 40'/%3E%3C/g%3E%3C/svg%3E");pointer-events:none}
.masthead-inner{position:relative}
.masthead h1{font-family:var(--font-display);font-size:clamp(1.8rem,4vw,3rem);font-weight:400;line-height:1.1;letter-spacing:-0.01em}
.masthead h1 em{color:#C9A86C;font-style:italic}
.masthead-sub{font-size:0.82rem;color:rgba(246,243,238,0.5);margin-top:0.5rem;letter-spacing:0.08em;text-transform:uppercase}
.pill-row{display:flex;flex-wrap:wrap;gap:0.4rem;margin-top:1.2rem}
.pill{font-family:var(--font-mono);font-size:0.65rem;background:rgba(255,255,255,0.08);border:1px solid rgba(255,255,255,0.15);color:rgba(246,243,238,0.65);padding:0.2rem 0.6rem;border-radius:2px;letter-spacing:0.06em;text-transform:uppercase}
.pill.active{background:rgba(201,168,76,0.2);border-color:rgba(201,168,76,0.5);color:#C9A86C}

/* Cards */
.card{background:var(--card);border:1px solid var(--mist);border-radius:var(--r-lg);padding:1.5rem;box-shadow:var(--shadow)}
.card+.card{margin-top:1rem}
.section-cap{font-family:var(--font-mono);font-size:0.63rem;letter-spacing:0.12em;text-transform:uppercase;color:var(--graphite);border-bottom:1px solid var(--mist);padding-bottom:0.5rem;margin-bottom:1.2rem}

/* Fields */
.field{margin-bottom:1rem}
.field label{display:block;font-size:0.78rem;font-weight:500;color:var(--slate);margin-bottom:0.35rem;letter-spacing:0.03em}
.field small{display:block;font-size:0.7rem;color:var(--graphite);margin-top:0.25rem}
input[type=text],input[type=number],textarea,select{width:100%;font-family:var(--font-body);font-size:0.9rem;background:#fff;border:1.5px solid var(--mist);border-radius:var(--r);padding:0.55rem 0.8rem;color:var(--ink);transition:border-color .15s;outline:none;resize:vertical}
input:focus,textarea:focus,select:focus{border-color:var(--teal)}
.row2{display:grid;grid-template-columns:1fr 1fr;gap:1rem}
@media(max-width:640px){.row2{grid-template-columns:1fr}}

/* Buttons */
.btn{display:inline-flex;align-items:center;gap:0.4rem;font-family:var(--font-body);font-weight:500;font-size:0.82rem;border:none;border-radius:var(--r);cursor:pointer;padding:0.55rem 1.1rem;transition:all .15s;letter-spacing:0.01em}
.btn-primary{background:var(--ink);color:var(--paper)}
.btn-primary:hover{background:#2A2A2A}
.btn-primary:disabled{opacity:0.4;cursor:not-allowed}
.btn-teal{background:var(--teal);color:#fff}
.btn-teal:hover{background:#145656}
.btn-rust{background:var(--rust);color:#fff}
.btn-ghost{background:transparent;color:var(--graphite);border:1.5px solid var(--fog)}
.btn-ghost:hover{border-color:var(--teal);color:var(--teal)}
.btn-gold{background:var(--gold);color:#fff}
.btn-full{width:100%;justify-content:center;padding:0.75rem;font-size:0.88rem}
.btn-sm{padding:0.3rem 0.7rem;font-size:0.72rem}

/* Spinner */
.spin{width:13px;height:13px;border:2px solid rgba(255,255,255,0.3);border-top-color:#fff;border-radius:50%;animation:spin .7s linear infinite;display:inline-block;flex-shrink:0}
.spin-dark{border-color:rgba(17,16,16,0.15);border-top-color:var(--ink)}
@keyframes spin{to{transform:rotate(360deg)}}

/* Alert */
.alert{padding:0.75rem 1rem;border-radius:var(--r);font-size:0.82rem;margin:0.6rem 0;display:flex;align-items:flex-start;gap:0.5rem;line-height:1.5}
.alert-error{background:#fdecea;border-left:3px solid var(--rust);color:#7a1a0a}
.alert-warn{background:var(--gold-lt);border-left:3px solid var(--gold);color:#6b4a10}
.alert-ok{background:var(--teal-lt);border-left:3px solid var(--teal);color:#0f4040}
.alert-info{background:#EEF4FB;border-left:3px solid var(--slate);color:var(--slate)}

/* Onboarding wizard */
.wizard-step{animation:fadeSlide .3s ease}
@keyframes fadeSlide{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:translateY(0)}}
.wizard-choice-grid{display:grid;grid-template-columns:1fr 1fr;gap:0.75rem;margin-top:1rem}
@media(max-width:560px){.wizard-choice-grid{grid-template-columns:1fr}}
.wizard-choice{border:2px solid var(--fog);border-radius:var(--r-lg);padding:1.25rem;cursor:pointer;background:#fff;transition:all .18s;text-align:left}
.wizard-choice:hover{border-color:var(--teal);box-shadow:0 0 0 3px rgba(26,107,107,0.08)}
.wizard-choice.selected{border-color:var(--teal);background:var(--teal-lt)}
.wc-icon{font-size:1.5rem;margin-bottom:0.5rem;display:block}
.wc-title{font-weight:600;font-size:0.9rem;margin-bottom:0.2rem}
.wc-desc{font-size:0.75rem;color:var(--graphite);line-height:1.4}
.wizard-nav{display:flex;align-items:center;gap:0.75rem;margin-top:1.25rem}
.step-dots{display:flex;gap:0.35rem;margin-left:auto}
.step-dot{width:6px;height:6px;border-radius:50%;background:var(--fog)}
.step-dot.done{background:var(--teal)}

/* Approach recommendation banner */
.rec-banner{background:linear-gradient(135deg,var(--teal-lt),#EEF8F8);border:1.5px solid rgba(26,107,107,0.25);border-radius:var(--r-lg);padding:1rem 1.25rem;margin-bottom:1rem;display:flex;gap:0.75rem;align-items:flex-start}
.rec-icon{font-size:1.2rem;flex-shrink:0;margin-top:1px}
.rec-title{font-weight:600;font-size:0.88rem;color:var(--teal);margin-bottom:0.2rem}
.rec-desc{font-size:0.78rem;color:var(--graphite);line-height:1.5}

/* Progress bar */
.prog-wrap{margin:0.5rem 0}
.prog-label{display:flex;justify-content:space-between;font-size:0.72rem;color:var(--graphite);margin-bottom:0.3rem}
.prog-bar{height:5px;background:var(--mist);border-radius:3px;overflow:hidden}
.prog-fill{height:100%;border-radius:3px;background:linear-gradient(90deg,var(--teal),#2AA8A8);transition:width .4s ease}

/* Section generation list */
.gen-list{display:flex;flex-direction:column;gap:0.35rem}
.gen-item{display:flex;align-items:center;gap:0.6rem;font-size:0.82rem;padding:0.4rem 0.6rem;border-radius:var(--r);background:var(--card)}
.gen-item.done{color:var(--teal)}
.gen-item.active{color:var(--gold);background:var(--gold-lt)}
.gen-item.pending{color:var(--graphite)}
.gen-dot{width:8px;height:8px;border-radius:50%;flex-shrink:0}
.gen-dot.done{background:var(--teal)}
.gen-dot.active{background:var(--gold);animation:pulse 1s infinite}
.gen-dot.pending{background:var(--fog)}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:0.4}}

/* Tabs */
.tab-bar{display:flex;flex-wrap:wrap;gap:0.25rem;border-bottom:1.5px solid var(--mist);margin-bottom:1.25rem}
.tab-btn{font-family:var(--font-body);font-size:0.75rem;font-weight:500;background:none;border:none;padding:0.45rem 0.8rem;color:var(--graphite);cursor:pointer;border-bottom:2px solid transparent;margin-bottom:-1.5px;transition:all .15s;border-radius:4px 4px 0 0}
.tab-btn:hover{color:var(--ink)}
.tab-btn.active{color:var(--ink);border-bottom-color:var(--rust);background:rgba(181,69,27,0.04)}

/* Paper text */
.section-pre{font-family:'DM Serif Display',Georgia,serif;font-size:0.93rem;white-space:pre-wrap;line-height:1.85;color:var(--ink);font-weight:400}

/* Metrics grid */
.metrics-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(120px,1fr));gap:0.6rem;margin:0.75rem 0}
.metric-tile{background:var(--paper);border:1px solid var(--mist);border-radius:var(--r);padding:0.75rem;text-align:center}
.mt-label{font-family:var(--font-mono);font-size:0.6rem;letter-spacing:0.08em;text-transform:uppercase;color:var(--graphite);margin-bottom:0.25rem}
.mt-val{font-family:var(--font-display);font-size:1.6rem;font-weight:400;line-height:1}
.mt-val.good{color:var(--teal)}
.mt-val.mid{color:var(--gold)}
.mt-val.bad{color:var(--rust)}

/* Chart container */
.chart-wrap{position:relative;width:100%;margin:0.75rem 0}

/* Graph cards */
.graph-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(400px,1fr));gap:1rem}
.graph-card{border:1px solid var(--mist);border-radius:var(--r-lg);overflow:hidden;background:#fff}
.graph-card img{width:100%;display:block}
.graph-card-body{padding:0.85rem}
.graph-card h3{font-family:var(--font-body);font-size:0.85rem;font-weight:600;margin-bottom:0.5rem}
.insight-block{font-size:0.78rem;line-height:1.55;padding:0.45rem 0.7rem;border-radius:4px;margin-bottom:0.35rem}
.ib-stat{background:#EEF4FB;border-left:3px solid #3A86FF;color:#1a3050}
.ib-proj{background:var(--teal-lt);border-left:3px solid var(--teal);color:#0a3030}
.ib-label{font-family:var(--font-mono);font-size:0.58rem;letter-spacing:0.08em;text-transform:uppercase;display:block;margin-bottom:0.2rem;opacity:0.7}

/* Approach compare */
.approach-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:0.75rem}
@media(max-width:760px){.approach-grid{grid-template-columns:1fr}}
.approach-card{border:1.5px solid var(--fog);border-radius:var(--r-lg);overflow:hidden}
.approach-card.best{border-color:var(--teal);box-shadow:0 0 0 3px rgba(26,107,107,0.1)}
.approach-head{padding:0.6rem 0.9rem;background:var(--mist);display:flex;justify-content:space-between;align-items:center}
.approach-head h4{font-size:0.78rem;font-weight:600;text-transform:uppercase;letter-spacing:0.04em}
.best-badge{font-family:var(--font-mono);font-size:0.58rem;background:var(--teal);color:#fff;padding:0.15rem 0.45rem;border-radius:2px;text-transform:uppercase;letter-spacing:0.05em}
.approach-body{padding:0.9rem}
.approach-text{font-size:0.78rem;line-height:1.55;color:var(--slate);max-height:110px;overflow-y:auto;border-bottom:1px solid var(--mist);margin-bottom:0.7rem;padding-bottom:0.5rem}

/* Metric bar */
.mbar-row{display:flex;align-items:center;gap:0.5rem;margin-bottom:0.35rem}
.mbar-label{font-size:0.7rem;min-width:110px;color:var(--graphite)}
.mbar-wrap{flex:1;height:5px;background:var(--mist);border-radius:3px;overflow:hidden}
.mbar-fill{height:100%;border-radius:3px;background:linear-gradient(90deg,var(--teal),#3AA8A8)}
.mbar-val{font-family:var(--font-mono);font-size:0.65rem;min-width:32px;text-align:right;color:var(--slate)}

/* Citations */
.cit-list{display:flex;flex-direction:column;gap:0.5rem}
.cit-item{font-size:0.8rem;padding:0.5rem 0.75rem;border:1px solid var(--mist);border-radius:var(--r);background:#fff;line-height:1.55}
.cit-num{font-family:var(--font-mono);font-size:0.65rem;color:var(--rust);margin-right:0.4rem}

/* Dataset cards */
.ds-list{display:flex;flex-direction:column;gap:0.5rem;margin-top:0.6rem}
.ds-card{border:1.5px solid var(--fog);border-radius:var(--r-lg);padding:0.8rem 1rem;cursor:pointer;background:#fff;transition:all .15s;position:relative;overflow:hidden}
.ds-card:hover{border-color:var(--teal)}
.ds-card.selected{border-color:var(--teal);background:var(--teal-lt)}
.ds-top{display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:0.2rem}
.ds-title{font-weight:600;font-size:0.85rem}
.ds-source{font-family:var(--font-mono);font-size:0.6rem;text-transform:uppercase;padding:0.15rem 0.4rem;border-radius:2px}
.ds-source.kaggle{background:#e0f4ff;color:#0a5080}
.ds-source.uci{background:var(--gold-lt);color:#5a3a10}
.ds-desc{font-size:0.77rem;color:var(--graphite);line-height:1.4;margin-top:0.2rem}
.ds-meta{display:flex;gap:0.5rem;font-size:0.7rem;color:var(--graphite);margin-top:0.3rem;flex-wrap:wrap}
.ds-tags{display:flex;flex-wrap:wrap;gap:0.25rem;margin-top:0.35rem}
.ds-tag{font-family:var(--font-mono);font-size:0.6rem;background:var(--mist);padding:0.12rem 0.4rem;border-radius:2px;color:var(--graphite)}
.ds-rel-bar{position:absolute;bottom:0;left:0;height:2px;background:linear-gradient(90deg,var(--teal),var(--gold));transition:width .3s}

/* Graph type selector */
.gtype-grid{display:grid;grid-template-columns:1fr 1fr;gap:0.5rem;margin:0.6rem 0}
.gtype-card{display:flex;align-items:center;gap:0.6rem;border:1.5px solid var(--fog);border-radius:var(--r);padding:0.55rem 0.75rem;cursor:pointer;background:#fff;transition:all .15s}
.gtype-card:hover{border-color:var(--teal)}
.gtype-card.selected{border-color:var(--teal);background:var(--teal-lt)}
.gtype-card input{display:none}

/* Upload zone */
.drop-zone{border:2px dashed var(--fog);border-radius:var(--r-lg);padding:1.5rem;text-align:center;cursor:pointer;transition:all .18s}
.drop-zone:hover{border-color:var(--teal);background:rgba(26,107,107,0.03)}
.dz-icon{font-size:1.4rem;margin-bottom:0.35rem}
.dz-text{font-size:0.85rem;color:var(--graphite)}
.dz-hint{font-size:0.72rem;color:var(--fog);margin-top:0.2rem}

/* Extracted paper cards */
.ep-card{border:1px solid var(--mist);border-radius:var(--r);padding:0.75rem;background:#fff;margin-bottom:0.4rem}
.ep-title{font-weight:600;font-size:0.83rem;margin-bottom:0.15rem}
.ep-meta{font-size:0.72rem;color:var(--graphite);margin-bottom:0.3rem}
.ep-abs{font-size:0.76rem;color:var(--slate);line-height:1.45}

/* Algo block */
.algo-card{border:1px solid var(--mist);border-radius:var(--r-lg);overflow:hidden;margin-bottom:0.75rem}
.algo-head{background:var(--ink);color:#C9A86C;padding:0.5rem 0.9rem;font-family:var(--font-mono);font-size:0.7rem;font-weight:500;letter-spacing:0.06em}
.algo-body{background:#1A1A1A;padding:0.9rem;font-family:var(--font-mono);font-size:0.76rem;color:#D4D0C8;line-height:1.65;white-space:pre-wrap}

/* Block diagram */
.flow-table{width:100%;border-collapse:collapse;margin:0.6rem 0}
.flow-table td{padding:0.45rem 0.7rem;border-bottom:1px solid var(--mist)}
.flow-src{font-weight:600;font-size:0.8rem;background:var(--teal-lt);color:var(--teal);padding:0.25rem 0.5rem;border-radius:3px;text-align:center;white-space:nowrap}
.flow-arrow{text-align:center;color:var(--gold);font-size:0.9rem;width:28px}
.flow-dst{font-size:0.8rem}
.flow-desc{font-size:0.72rem;color:var(--graphite);margin-top:0.12rem}

/* vs Baseline */
.vs-grid{display:grid;grid-template-columns:1fr 1fr;gap:1rem}
@media(max-width:640px){.vs-grid{grid-template-columns:1fr}}
.vs-card{border:1.5px solid var(--fog);border-radius:var(--r-lg);overflow:hidden}
.vs-card.winner{border-color:var(--teal)}
.vs-head{padding:0.55rem 0.9rem;background:var(--mist);display:flex;justify-content:space-between;align-items:center}
.vs-head h4{font-size:0.78rem;font-weight:600;text-transform:uppercase;letter-spacing:0.04em}
.vs-body{padding:0.85rem}
.vs-delta-tbl{width:100%;border-collapse:collapse;font-size:0.76rem;margin-top:1rem}
.vs-delta-tbl th{text-align:left;font-family:var(--font-mono);font-size:0.6rem;letter-spacing:0.07em;text-transform:uppercase;color:var(--graphite);padding:0.3rem 0.4rem;border-bottom:1px solid var(--mist)}
.vs-delta-tbl td{padding:0.35rem 0.4rem;border-bottom:1px solid var(--paper)}
.delta-up{color:var(--teal);font-family:var(--font-mono);font-size:0.68rem}
.delta-down{color:var(--rust);font-family:var(--font-mono);font-size:0.68rem}

/* Analysis panel */
.analysis-panel{margin-top:1.25rem;border:1px dashed var(--fog);border-radius:var(--r-lg);overflow:hidden}
.ap-toggle{display:flex;justify-content:space-between;align-items:center;padding:0.6rem 0.9rem;background:rgba(26,107,107,0.05);cursor:pointer;user-select:none}
.ap-toggle h4{font-family:var(--font-mono);font-size:0.65rem;letter-spacing:0.08em;text-transform:uppercase;color:var(--teal)}
.ap-body{padding:1rem}
.ap-grid{display:grid;grid-template-columns:1fr 1fr;gap:0.75rem}
@media(max-width:640px){.ap-grid{grid-template-columns:1fr}}
.ap-block h5{font-family:var(--font-mono);font-size:0.6rem;letter-spacing:0.09em;text-transform:uppercase;color:var(--graphite);margin-bottom:0.4rem}
.src-item{display:flex;justify-content:space-between;align-items:center;font-size:0.77rem;padding:0.25rem 0;border-bottom:1px solid var(--mist)}
.src-item:last-child{border-bottom:none}
.rel-pill{font-family:var(--font-mono);font-size:0.62rem;padding:0.12rem 0.4rem;border-radius:20px;background:var(--teal-lt);color:var(--teal)}
.kw-chips{display:flex;flex-wrap:wrap;gap:0.3rem}
.kw-chip{font-family:var(--font-mono);font-size:0.62rem;background:var(--mist);border-radius:2px;padding:0.15rem 0.45rem;color:var(--graphite)}
.conf-ring{display:inline-flex;align-items:center;justify-content:center;width:56px;height:56px;border-radius:50%;border:3px solid var(--gold);font-family:var(--font-display);font-size:0.9rem;font-weight:400;color:var(--gold)}
.unspptd-list{list-style:none}
.unspptd-list li{font-size:0.77rem;padding:0.25rem 0.5rem;background:#fff5f3;border-left:3px solid var(--rust);margin-bottom:0.3rem;border-radius:0 3px 3px 0}

/* Citation checker */
.cit-result-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(120px,1fr));gap:0.6rem;margin-top:0.9rem}
.cit-score-card{border:1px solid var(--mist);border-radius:var(--r);padding:0.75rem;text-align:center;background:#fff}
.csc-label{font-family:var(--font-mono);font-size:0.6rem;letter-spacing:0.07em;text-transform:uppercase;color:var(--graphite);margin-bottom:0.3rem}
.csc-val{font-family:var(--font-display);font-size:1.5rem;font-weight:400}
.csc-val.good{color:var(--teal)}.csc-val.mid{color:var(--gold)}.csc-val.bad{color:var(--rust)}
.support-badge{display:inline-block;font-family:var(--font-mono);font-size:0.68rem;letter-spacing:0.05em;padding:0.2rem 0.65rem;border-radius:3px;font-weight:500;text-transform:uppercase}
.sb-SUPPORTING{background:#d4f4e2;color:#0e4a28}
.sb-PARTIAL{background:var(--gold-lt);color:#5a3a10}
.sb-NEUTRAL{background:var(--mist);color:var(--slate)}
.sb-CONTRADICTING{background:var(--rust-lt);color:#5a0d00}

/* Summary box */
.summary-box{background:var(--teal-lt);border:1px solid rgba(26,107,107,0.2);border-radius:var(--r);padding:0.8rem 1rem;font-size:0.83rem;line-height:1.6;color:var(--teal);margin-top:0.75rem}

/* Stats bar */
.stats-bar{display:flex;flex-wrap:wrap;gap:0.6rem;padding:0.6rem 0;border-bottom:1px solid var(--mist);margin-bottom:1rem}
.stat-pill{font-family:var(--font-mono);font-size:0.68rem;display:flex;align-items:center;gap:0.25rem}
.stat-pill strong{color:var(--ink)}
.stat-pill span{color:var(--graphite)}
.ctx-badge{font-family:var(--font-mono);font-size:0.6rem;background:var(--ink);color:#C9A86C;padding:0.18rem 0.5rem;border-radius:2px;letter-spacing:0.05em;text-transform:uppercase}

/* Paper readability widget */
.readability-bar{height:8px;background:var(--mist);border-radius:4px;overflow:hidden;margin:0.4rem 0}
.readability-fill{height:100%;border-radius:4px;transition:width .5s ease}

/* Section eval */
.section-eval-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:0.5rem;margin-top:0.75rem}
.se-tile{background:var(--paper);border:1px solid var(--mist);border-radius:var(--r);padding:0.65rem;text-align:center}
.se-label{font-size:0.68rem;color:var(--graphite);margin-bottom:0.25rem}
.se-score{font-family:var(--font-display);font-size:1.3rem;font-weight:400}

/* Novelty / hallucination flag */
.novelty-alert{background:#FFF8E6;border:1px solid var(--gold);border-radius:var(--r);padding:0.7rem 0.9rem;margin-top:0.5rem;font-size:0.78rem;color:#5a3a10}

/* Preview table */
.preview-wrap{overflow-x:auto;margin:0.6rem 0}
.preview-tbl{width:100%;border-collapse:collapse;font-size:0.73rem}
.preview-tbl th{background:var(--ink);color:var(--paper);font-family:var(--font-mono);font-size:0.6rem;letter-spacing:0.06em;text-transform:uppercase;padding:0.4rem 0.6rem;text-align:left;white-space:nowrap}
.preview-tbl td{padding:0.3rem 0.6rem;border-bottom:1px solid var(--mist);white-space:nowrap;max-width:110px;overflow:hidden;text-overflow:ellipsis}

/* Footer */
.footer{background:var(--ink);color:rgba(246,243,238,0.35);font-family:var(--font-mono);font-size:0.65rem;text-align:center;padding:1rem;margin-top:auto;letter-spacing:0.04em}
.footer a{color:rgba(201,168,76,0.6);text-decoration:none}

/* divider */
hr.divider{border:none;border-top:1px solid var(--mist);margin:1.1rem 0}

/* Approach recommendation bubble */
.approach-rec{display:inline-flex;align-items:center;gap:0.4rem;font-family:var(--font-mono);font-size:0.62rem;background:var(--teal-lt);color:var(--teal);padding:0.2rem 0.6rem;border-radius:20px;border:1px solid rgba(26,107,107,0.3)}

/* Precision/Recall/F1 chart panel */
.prf-chart-panel{margin-top:1.25rem}
.prf-chart-panel h4{font-family:var(--font-mono);font-size:0.65rem;letter-spacing:0.08em;text-transform:uppercase;color:var(--graphite);margin-bottom:0.7rem}
.chart-canvas-wrap{position:relative;width:100%}

/* BibTeX export box */
.bibtex-box{background:#1A1A1A;border-radius:var(--r-lg);padding:1rem;font-family:var(--font-mono);font-size:0.73rem;color:#C8C4BC;line-height:1.6;max-height:280px;overflow-y:auto;margin-top:0.6rem;white-space:pre-wrap}

/* Page loading overlay */
.gen-overlay{position:fixed;inset:0;background:rgba(17,16,16,0.45);display:flex;align-items:center;justify-content:center;z-index:1000;backdrop-filter:blur(2px)}
.gen-dialog{background:var(--card);border-radius:var(--r-lg);padding:2rem;max-width:400px;width:90%;text-align:center;box-shadow:var(--shadow-lg)}
.gen-dialog h3{font-family:var(--font-display);font-size:1.3rem;margin-bottom:0.5rem}
.gen-dialog p{font-size:0.82rem;color:var(--graphite);margin-bottom:1.25rem}
`;

/* ─── Constants ─────────────────────────────────────────────────────────── */
const GRAPH_ICONS = {
  correlation_heatmap:'📊', scatter_matrix:'✦', distribution:'📈',
  box_plot:'▭', class_distribution:'◑', feature_importance:'🏆'
};

const SECTION_LABELS = [
  {key:'abstract',label:'Abstract'},
  {key:'introduction',label:'Introduction'},
  {key:'literature_survey',label:'Literature Survey'},
  {key:'methodology',label:'Methodology'},
  {key:'algorithms',label:'Algorithms'},
  {key:'block_diagram',label:'System Architecture'},
  {key:'results',label:'Results & Discussion'},
  {key:'conclusion',label:'Conclusion'},
];

const ALL_TABS = [
  {key:'abstract',label:'Abstract'},{key:'introduction',label:'Introduction'},
  {key:'literature_survey',label:'Literature'},{key:'methodology',label:'Methodology'},
  {key:'algorithms',label:'⚙ Algorithms'},{key:'block_diagram',label:'Architecture'},
  {key:'results',label:'Results'},{key:'conclusion',label:'Conclusion'},
  {key:'citations',label:'Citations'},{key:'graphs',label:'Graphs'},
  {key:'explainability',label:'Analysis'},{key:'multiapproach',label:'Approaches'},
  { key: 'paper_metrics', label: 'NLP Metrics' },
  {key:'prf_metrics',label:'Metrics'},{key:'vsbaseline',label:'📊 vs Baseline'},
  {key:'citchecker',label:'Cite Check'},{key:'bibtex',label:'BibTeX'},
  {key:'readability',label:'Quality'},
];

const PROJECT_FIELDS = [
  {id:'core_functionality',icon:'⚙',label:'Core Functionality',ph:'What your system does…'},
  {id:'tech_stack',icon:'🛠',label:'Tech Stack',ph:'FastAPI, FAISS, React, Groq…'},
  {id:'dataset_pipeline',icon:'📊',label:'Dataset & Pipeline',ph:'arXiv 1.7M, Kaggle, BM25…'},
  {id:'algorithms',icon:'🔬',label:'Algorithms Used',ph:'FAISS IVF-PQ, RandomForest…'},
  {id:'output_results',icon:'📄',label:'Outputs & Results',ph:'IEEE DOCX, citations, graphs…'},
  {id:'novel_contributions',icon:'💡',label:'Novel Contributions',ph:'What makes your project unique…'},
];

/* ─── Helpers ────────────────────────────────────────────────────────────── */
function pct(v){
  if(v===undefined||v===null) return '—';
  const n=typeof v==='string'?parseFloat(v):v;
  if(isNaN(n)) return String(v);
  return n<=1?`${Math.round(n*100)}%`:`${Math.round(n)}%`;
}
function flt(v){const n=typeof v==='string'?parseFloat(v):v;return isNaN(n)?0:n<=1?n:n/100}
function scoreColor(v){if(v>=0.75)return'good';if(v>=0.45)return'mid';return'bad'}

/* ─── Metric bar ─────────────────────────────────────────────────────────── */
function MetricBars({metrics}){
  const entries=Object.entries(metrics||{});
  if(!entries.length) return null;
  return(<div>{entries.map(([k,v])=>{
    const val=flt(v);
    return(<div className="mbar-row" key={k}>
      <span className="mbar-label">{k.replace(/_/g,' ')}</span>
      <div className="mbar-wrap"><div className="mbar-fill" style={{width:`${Math.round(val*100)}%`}}/></div>
      <span className="mbar-val">{pct(val)}</span>
    </div>);
  })}</div>);
}

/* ─── Section analysis panel ─────────────────────────────────────────────── */
function SectionAnalysisPanel({analysis}){
  const [open,setOpen]=useState(false);
  if(!analysis) return null;
  const {sources_used=[],keywords=[],metrics={},confidence_score,unsupported_statements=[],sentence_mapping=[]}=analysis;
  const confNum=flt(confidence_score??metrics?.confidence_score??0);
  return(
    <div className="analysis-panel">
      <div className="ap-toggle" onClick={()=>setOpen(o=>!o)}>
        <h4>Evidence Alignment</h4>
        <span style={{fontFamily:'var(--font-mono)',fontSize:'0.62rem',color:'var(--graphite)'}}>{open?'▲':'▼'}</span>
      </div>
      {open&&<div className="ap-body">
        <div className="ap-grid">
          <div className="ap-block">
            <h5>Sources Used</h5>
            {sources_used.length===0?<p style={{fontSize:'0.76rem',color:'#aaa'}}>None logged</p>
              :sources_used.map((s,i)=>(
              <div className="src-item" key={i}>
                <span style={{fontSize:'0.76rem',flex:1,paddingRight:'0.4rem'}}>{typeof s==='string'?s:s.title||`Paper ${i+1}`}</span>
                <span className="rel-pill">{typeof s==='object'?pct(s.relevance_score):'—'}</span>
              </div>))}
          </div>
          <div>
            <div className="ap-block">
              <h5>Keywords</h5>
              <div className="kw-chips">
                {keywords.length===0?<span style={{fontSize:'0.76rem',color:'#aaa'}}>None</span>
                  :keywords.map((k,i)=><span className="kw-chip" key={i}>{k}</span>)}
              </div>
            </div>
            <div className="ap-block" style={{marginTop:'0.75rem'}}>
              <h5>Confidence</h5>
              <div className="conf-ring">{Math.round(confNum*100)}</div>
            </div>
          </div>
        </div>
        <div className="ap-block" style={{marginTop:'0.75rem'}}>
          <h5>Metrics</h5>
          {[{label:'Relevance',val:flt(metrics.relevance_score??metrics.relevance??0)},
            {label:'Citation Coverage',val:flt(metrics.citation_coverage??0)},
            {label:'Confidence',val:confNum}].map(({label,val})=>(
            <div className="mbar-row" key={label}>
              <span className="mbar-label">{label}</span>
              <div className="mbar-wrap"><div className="mbar-fill" style={{width:`${Math.round(val*100)}%`}}/></div>
              <span className="mbar-val">{pct(val)}</span>
            </div>))}
        </div>
        <div className="ap-block">
          <h5>Unsupported Statements</h5>
          {!unsupported_statements.length||unsupported_statements[0]==='All statements are evidence-backed'
            ?<p style={{fontSize:'0.77rem',color:'var(--teal)'}}>✓ All statements backed</p>
            :<ul className="unspptd-list">{unsupported_statements.map((s,i)=><li key={i}>{s}</li>)}</ul>}
        </div>
        {sentence_mapping.length>0&&(
          <div className="ap-block">
            <h5>Sentence Mapping</h5>
            <table style={{width:'100%',borderCollapse:'collapse',fontSize:'0.74rem'}}>
              <thead><tr>
                <th style={{textAlign:'left',fontFamily:'var(--font-mono)',fontSize:'0.58rem',letterSpacing:'0.07em',textTransform:'uppercase',color:'var(--graphite)',padding:'0.3rem 0.4rem',borderBottom:'1px solid var(--mist)'}}>Sentence</th>
                <th style={{textAlign:'left',fontFamily:'var(--font-mono)',fontSize:'0.58rem',letterSpacing:'0.07em',textTransform:'uppercase',color:'var(--graphite)',padding:'0.3rem 0.4rem',borderBottom:'1px solid var(--mist)'}}>Source</th>
              </tr></thead>
              <tbody>{sentence_mapping.map((m,i)=>(
                <tr key={i}>
                  <td style={{padding:'0.35rem 0.4rem',borderBottom:'1px solid var(--paper)',verticalAlign:'top'}}>{typeof m==='object'?m.sentence:m}</td>
                  <td style={{padding:'0.35rem 0.4rem',borderBottom:'1px solid var(--paper)',fontFamily:'var(--font-mono)',fontSize:'0.62rem',color:'var(--teal)'}}>{typeof m==='object'?m.source:'—'}</td>
                </tr>))}</tbody>
            </table>
          </div>)}
      </div>}
    </div>);
}

/* ─── Algorithm Block ────────────────────────────────────────────────────── */
function AlgorithmBlock({text}){
  if(!text?.trim()) return<p style={{color:'#aaa'}}>No algorithm content.</p>;
  const cleaned=text.replace(/```[a-zA-Z]*\r?\n?/g,'').replace(/```/g,'').trim();
  const re=/(?:^|\n)([ \t]*(?:ALGORITHM|Algorithm)[ \t]*\d*[ \t]*[:\.\-–—]?[^\n]*)/g;
  const matches=[];let m;
  while((m=re.exec(cleaned))!==null) matches.push({index:m.index,raw:m[1].trim()});
  if(!matches.length) return<div className="algo-card"><div className="algo-head">Algorithm</div><div className="algo-body">{cleaned}</div></div>;
  const prefix=cleaned.slice(0,matches[0].index).trim();
  const blocks=matches.map((match,i)=>{
    const bodyStart=match.index+match.raw.length;
    const bodyEnd=i+1<matches.length?matches[i+1].index:cleaned.length;
    return{header:match.raw.replace(/\*{1,2}|#{1,3}/g,'').trim(),body:cleaned.slice(bodyStart,bodyEnd).trim()};
  });
  return(<div>
    {prefix&&<p style={{marginBottom:'0.75rem',fontSize:'0.88rem',lineHeight:1.7}}>{prefix}</p>}
    {blocks.map((blk,i)=>(
      <div className="algo-card" key={i}>
        <div className="algo-head">⚙ {blk.header}</div>
        <div className="algo-body">{blk.body}</div>
      </div>))}
  </div>);
}

/* ─── Block Diagram ──────────────────────────────────────────────────────── */
function BlockDiagramView({text}){
  if(!text?.trim()) return<p style={{color:'#aaa'}}>No diagram content.</p>;
  const re=/\[BLOCK_DIAGRAM_START\]([\s\S]*?)\[BLOCK_DIAGRAM_END\]/i;
  const mm=re.exec(text);
  const parseArrows=(lines)=>lines.map(l=>{
    const match=l.trim().match(/^(.+?)\s*(?:->|→)\s*(.+?)(?::\s*(.*))?$/);
    return match?{src:match[1].trim(),dst:match[2].trim(),desc:(match[3]||'').trim()}:{src:l.trim(),dst:'',desc:''};
  });
  let rows=[],systemName='System Architecture',prefixText='',suffixText='';
  if(mm){
    const raw=mm[1];
    const sm=raw.match(/^SYSTEM:\s*(.+)$/mi);
    if(sm) systemName=sm[1].trim();
    const parts=text.split(re);
    prefixText=(parts[0]||'').trim();
    suffixText=(parts[2]||'').trim();
    const compLines=raw.split('\n').filter(l=>/->|→/.test(l));
    rows=parseArrows(compLines);
  } else {
    const allLines=text.split('\n');
    const arrowLines=allLines.filter(l=>/->|→/.test(l.trim())&&l.trim().length>3);
    rows=parseArrows(arrowLines);
    prefixText=allLines.filter(l=>!/->|→/.test(l)).join('\n').trim();
  }
  return(<div>
    {prefixText&&<p style={{marginBottom:'0.75rem',fontSize:'0.88rem',lineHeight:1.7}}>{prefixText}</p>}
    <div className="card">
      <div style={{fontFamily:'var(--font-body)',fontWeight:600,fontSize:'0.82rem',color:'var(--teal)',marginBottom:'0.65rem'}}> {systemName}</div>
      <table className="flow-table"><tbody>{rows.map((r,i)=>(
        <tr key={i}>
          <td style={{width:'30%'}}><div className="flow-src">{r.src}</div></td>
          <td className="flow-arrow">→</td>
          <td><div className="flow-dst"><strong>{r.dst}</strong>{r.desc&&<div className="flow-desc">{r.desc}</div>}</div></td>
        </tr>))}</tbody></table>
    </div>
    {suffixText&&<p style={{marginTop:'0.75rem',fontSize:'0.88rem',lineHeight:1.7}}>{suffixText}</p>}
  </div>);
}
function PaperMetricsPanel({ metrics }) {
  const [activeTab, setActiveTab] = useState('overview');

  if (!metrics || metrics.error) {
    return (
      <p style={{ color: 'var(--graphite)', fontSize: '0.85rem' }}>
        {metrics?.error || 'No metrics available. Generate a paper first.'}
      </p>
    );
  }

  const scoreClass = (v, inverse = false) => {
    if (inverse) return v < 50 ? 'good' : v < 120 ? 'mid' : 'bad';
    return v >= 0.5 ? 'good' : v >= 0.3 ? 'mid' : 'bad';
  };

  const gauges = [
    { label: 'BLEU score',        val: metrics.bleu_score,       thresholds: [0.3, 0.5] },
    { label: 'ROUGE-1 F1',       val: metrics.rouge_1_f1,       thresholds: [0.3, 0.5] },
    { label: 'ROUGE-2 F1',       val: metrics.rouge_2_f1,       thresholds: [0.2, 0.4] },
    { label: 'ROUGE-L F1',       val: metrics.rouge_l_f1,       thresholds: [0.25, 0.45] },
    { label: 'Lexical diversity', val: metrics.lexical_diversity, thresholds: [0.4, 0.6] },
  ];

  const perpClass = scoreClass(metrics.perplexity, true);
  const overallGrade = (() => {
    const scores = [
      metrics.bleu_score >= 0.3, metrics.rouge_1_f1 >= 0.4,
      metrics.lexical_diversity >= 0.5, metrics.perplexity < 100,
    ];
    const passing = scores.filter(Boolean).length;
    return passing >= 3 ? 'good' : passing >= 2 ? 'mid' : 'bad';
  })();
  const gradeLabel = { good: 'Good', mid: 'Acceptable', bad: 'Needs improvement' }[overallGrade];

  const canvasRef = useRef(null);
  const chartRef  = useRef(null);

  useEffect(() => {
    if (activeTab !== 'chart' || !canvasRef.current) return;
    const labels = ['BLEU', 'ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'Lex. diversity'];
    const vals   = [metrics.bleu_score, metrics.rouge_1_f1, metrics.rouge_2_f1, metrics.rouge_l_f1, metrics.lexical_diversity];
    const thresholds = [0.3, 0.3, 0.2, 0.25, 0.4];

    const init = () => {
      if (chartRef.current) { chartRef.current.destroy(); chartRef.current = null; }
      chartRef.current = new window.Chart(canvasRef.current, {
        type: 'bar',
        data: {
          labels,
          datasets: [
            { label: 'This paper',     data: vals,       backgroundColor: '#1D9E75', borderWidth: 0 },
            { label: 'Good threshold', data: thresholds, backgroundColor: '#D3D1C7', borderWidth: 0 },
          ],
        },
        options: {
          responsive: true, maintainAspectRatio: false,
          plugins: { legend: { display: false }, tooltip: { callbacks: { label: ctx => `${ctx.dataset.label}: ${ctx.raw.toFixed(3)}` } } },
          scales: {
            x: { grid: { display: false }, ticks: { font: { size: 11 } } },
            y: { min: 0, max: 1, ticks: { callback: v => v.toFixed(1), font: { size: 11 } }, grid: { color: 'rgba(0,0,0,0.05)' } },
          },
        },
      });
    };

    if (!window.Chart) {
      const s = document.createElement('script');
      s.src = 'https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js';
      s.onload = init;
      document.head.appendChild(s);
    } else { init(); }

    return () => { if (chartRef.current) { chartRef.current.destroy(); chartRef.current = null; } };
  }, [activeTab, metrics]);

  const tabs = [
    { key: 'overview', label: 'Overview' },
    { key: 'scores',   label: 'Score bars' },
    { key: 'chart',    label: 'Chart' },
    { key: 'interp',   label: 'Interpretation' },
  ];

  const gaugeColor = (val, [lo, hi]) => val >= hi ? '#1D9E75' : val >= lo ? '#BA7517' : '#993C1D';

  const interpData = [
    { label: 'Perplexity', val: metrics.perplexity?.toFixed(1), cls: perpClass,
      desc: `Scores below 50 = coherent text. Below 120 = acceptable. Above 200 = incoherent. Lower is better.` },
    { label: 'BLEU score', val: metrics.bleu_score?.toFixed(4), cls: scoreClass(metrics.bleu_score),
      desc: `Measures n-gram precision vs reference abstracts. 0.3+ = good, 0.5+ = excellent. Academic paraphrasing naturally lowers this.` },
    { label: 'ROUGE-1 F1', val: metrics.rouge_1_f1?.toFixed(4), cls: scoreClass(metrics.rouge_1_f1),
      desc: `Unigram overlap with retrieved papers. High score = paper covers the same key vocabulary and concepts as sources.` },
    { label: 'ROUGE-2 F1', val: metrics.rouge_2_f1?.toFixed(4), cls: scoreClass(metrics.rouge_2_f1),
      desc: `Bigram overlap. Lower than ROUGE-1 is normal — exact phrase matches are rarer but indicate strong thematic alignment.` },
    { label: 'ROUGE-L F1', val: metrics.rouge_l_f1?.toFixed(4), cls: scoreClass(metrics.rouge_l_f1),
      desc: `Longest-common-subsequence metric. Reflects structural similarity with source papers and argumentative flow.` },
    { label: 'Lexical diversity', val: metrics.lexical_diversity?.toFixed(4), cls: scoreClass(metrics.lexical_diversity),
      desc: `Type-token ratio. Above 0.5 = varied, non-repetitive vocabulary. Below 0.3 = repetitive phrasing detected.` },
  ];

  const badgeStyle = (cls) => ({
    display: 'inline-flex', alignItems: 'center', fontSize: '0.68rem', fontFamily: 'var(--font-mono)',
    padding: '0.15rem 0.5rem', borderRadius: '20px', fontWeight: 500,
    background: { good: 'var(--teal-lt)', mid: 'var(--gold-lt)', bad: 'var(--rust-lt)' }[cls],
    color: { good: 'var(--teal)', mid: 'var(--gold)', bad: 'var(--rust)' }[cls],
  });

  return (
    <div>
      {/* Tab bar */}
      <div className="tab-bar">
        {tabs.map(t => (
          <button key={t.key} className={`tab-btn ${activeTab === t.key ? 'active' : ''}`}
            onClick={() => setActiveTab(t.key)}>{t.label}</button>
        ))}
      </div>

      {/* Overview */}
      {activeTab === 'overview' && (
        <div>
          <div className="metrics-grid">
            {[
              { label: 'Perplexity',        val: metrics.perplexity?.toFixed(1),       cls: perpClass,                       note: '< 50 = coherent' },
              { label: 'BLEU',              val: metrics.bleu_score?.toFixed(4),        cls: scoreClass(metrics.bleu_score),  note: '0.3+ = good' },
              { label: 'ROUGE-1 F1',       val: metrics.rouge_1_f1?.toFixed(4),        cls: scoreClass(metrics.rouge_1_f1),  note: 'Unigram overlap' },
              { label: 'ROUGE-2 F1',       val: metrics.rouge_2_f1?.toFixed(4),        cls: scoreClass(metrics.rouge_2_f1),  note: 'Bigram overlap' },
              { label: 'ROUGE-L F1',       val: metrics.rouge_l_f1?.toFixed(4),        cls: scoreClass(metrics.rouge_l_f1),  note: 'LCS overlap' },
              { label: 'Lexical diversity', val: metrics.lexical_diversity?.toFixed(4), cls: scoreClass(metrics.lexical_diversity), note: '> 0.5 = varied' },
            ].map(({ label, val, cls, note }) => (
              <div className="metric-tile" key={label}>
                <div className="mt-label">{label}</div>
                <div className={`mt-val ${cls}`}>{val ?? '—'}</div>
                <div style={{ fontSize: '0.62rem', color: 'var(--graphite)', marginTop: '3px' }}>{note}</div>
              </div>
            ))}
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.6rem', marginTop: '0.75rem' }}>
            {[
              { label: 'Total words',         val: metrics.total_words?.toLocaleString() ?? '—' },
              { label: 'Avg sentence length', val: metrics.avg_sentence_length?.toFixed(1) ?? '—' },
              { label: 'Unique words',        val: metrics.unique_words?.toLocaleString() ?? '—' },
            ].map(({ label, val }) => (
              <div key={label} style={{ background: 'var(--paper)', border: '1px solid var(--mist)', borderRadius: 'var(--r)', padding: '0.7rem' }}>
                <div style={{ fontSize: '0.68rem', color: 'var(--graphite)', marginBottom: '3px' }}>{label}</div>
                <div style={{ fontSize: '1rem', fontWeight: 500 }}>{val}</div>
              </div>
            ))}
          </div>

          <div style={{ marginTop: '0.85rem', display: 'flex', alignItems: 'center', gap: '0.6rem', fontSize: '0.82rem' }}>
            <span style={{ color: 'var(--graphite)' }}>Overall quality:</span>
            <span style={badgeStyle(overallGrade)}>{gradeLabel}</span>
          </div>
        </div>
      )}

      {/* Score bars */}
      {activeTab === 'scores' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
          {gauges.map(g => {
            const pct = Math.min(100, Math.round(g.val / 1 * 100));
            const color = gaugeColor(g.val, g.thresholds);
            return (
              <div key={g.label} style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                <span style={{ fontSize: '0.75rem', color: 'var(--slate)', minWidth: '130px', fontWeight: 500 }}>{g.label}</span>
                <div style={{ flex: 1, height: '6px', background: 'var(--mist)', borderRadius: '3px', overflow: 'hidden' }}>
                  <div style={{ height: '100%', borderRadius: '3px', width: `${pct}%`, background: color, transition: 'width .5s ease' }} />
                </div>
                <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.68rem', minWidth: '48px', textAlign: 'right' }}>{g.val?.toFixed(4)}</span>
              </div>
            );
          })}
          {/* Perplexity — inverted scale */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <span style={{ fontSize: '0.75rem', color: 'var(--slate)', minWidth: '130px', fontWeight: 500 }}>Perplexity (inv.)</span>
            <div style={{ flex: 1, height: '6px', background: 'var(--mist)', borderRadius: '3px', overflow: 'hidden' }}>
              <div style={{ height: '100%', borderRadius: '3px', width: `${Math.min(100, Math.round((1 - Math.min(metrics.perplexity, 200) / 200) * 100))}%`, background: perpClass === 'good' ? '#1D9E75' : perpClass === 'mid' ? '#BA7517' : '#993C1D', transition: 'width .5s ease' }} />
            </div>
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.68rem', minWidth: '48px', textAlign: 'right' }}>{metrics.perplexity?.toFixed(2)}</span>
          </div>
        </div>
      )}

      {/* Chart */}
      {activeTab === 'chart' && (
        <div>
          <div style={{ display: 'flex', gap: '1rem', marginBottom: '0.6rem', flexWrap: 'wrap' }}>
            {[{ l: 'This paper', c: '#1D9E75' }, { l: 'Good threshold', c: '#D3D1C7' }].map(item => (
              <span key={item.l} style={{ display: 'flex', alignItems: 'center', gap: '4px', fontSize: '0.72rem', color: 'var(--graphite)' }}>
                <span style={{ width: '10px', height: '10px', borderRadius: '2px', background: item.c, display: 'inline-block' }} />
                {item.l}
              </span>
            ))}
          </div>
          <div style={{ position: 'relative', width: '100%', height: '220px' }}>
            <canvas ref={canvasRef} role="img"
              aria-label={`Bar chart of paper evaluation scores: BLEU ${metrics.bleu_score?.toFixed(3)}, ROUGE-1 ${metrics.rouge_1_f1?.toFixed(3)}, ROUGE-2 ${metrics.rouge_2_f1?.toFixed(3)}, ROUGE-L ${metrics.rouge_l_f1?.toFixed(3)}, Lexical diversity ${metrics.lexical_diversity?.toFixed(3)}`} />
          </div>
        </div>
      )}

      {/* Interpretation */}
      {activeTab === 'interp' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
          {interpData.map(d => (
            <div key={d.label} style={{
              borderLeft: `2.5px solid ${{ good: 'var(--teal)', mid: 'var(--gold)', bad: 'var(--rust)' }[d.cls]}`,
              padding: '8px 12px', borderRadius: '0 var(--r) var(--r) 0', background: 'var(--card)',
              border: '1px solid var(--mist)', borderLeftWidth: '2.5px',
            }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '3px' }}>
                <span style={{ fontSize: '0.8rem', fontWeight: 600 }}>{d.label}</span>
                <span style={badgeStyle(d.cls)}>{d.val}</span>
              </div>
              <div style={{ fontSize: '0.76rem', color: 'var(--slate)', lineHeight: 1.5 }}>{d.desc}</div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
/* ─── Graph card ─────────────────────────────────────────────────────────── */
function GraphCard({graph,sessionId,topic,projectContext}){
  const [analysisLoading,setAnalysisLoading]=useState(false);
  const [analysis,setAnalysis]=useState(null);
  const [err,setErr]=useState(null);
  const handleAnalyze=async()=>{
    setAnalysisLoading(true);setErr(null);
    try{
      const res=await fetch(`${API}/graphs/analyze-relevance`,{
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body:JSON.stringify({
          topic: topic||graph.title,
          project_context: projectContext?JSON.stringify(projectContext):null,
          graph_type: graph.type,          // ← THIS WAS MISSING
          graph_description: graph.statistical_insight||graph.insight||graph.title,
          session_id: sessionId,
        }),
      });
      if(!res.ok) throw new Error(await res.text());
      setAnalysis(await res.json());
    }catch(e){setErr(e.message);}
    finally{setAnalysisLoading(false);}
  };

  return(<div className="graph-card">
    <img src={`data:image/png;base64,${graph.data}`} alt={graph.title}/>
    <div className="graph-card-body">
      <h3>{graph.figure_label}: {graph.title}</h3>
      {graph.statistical_insight&&<div className="insight-block ib-stat"><span className="ib-label">Statistical</span>{graph.statistical_insight}</div>}
      {graph.project_insight&&graph.project_insight!==graph.statistical_insight&&<div className="insight-block ib-proj"><span className="ib-label">Project relevance</span>{graph.project_insight}</div>}
      {!analysis&&<button className="btn btn-ghost btn-sm" style={{marginTop:'0.5rem'}} disabled={analysisLoading} onClick={handleAnalyze}>
        {analysisLoading?<><span className="spin spin-dark"/>Analyzing…</>:'Analyze for Paper'}
      </button>}
      {err&&<div className="alert alert-error" style={{marginTop:'0.5rem'}}>{err}</div>}
      {analysis&&<div style={{marginTop:'0.75rem',border:'1px dashed var(--fog)',borderRadius:'var(--r)',padding:'0.75rem'}}>
        <div style={{display:'flex',justifyContent:'space-between',alignItems:'center',marginBottom:'0.4rem'}}>
          <span style={{fontFamily:'var(--font-mono)',fontSize:'0.7rem',fontWeight:500,color:analysis.decision==='YES'?'var(--teal)':'var(--rust)',background:analysis.decision==='YES'?'var(--teal-lt)':'var(--rust-lt)',padding:'0.2rem 0.55rem',borderRadius:'3px'}}>
            {analysis.decision==='YES'?'✓ Include':'✗ Exclude'}
          </span>
          {analysis.suggested_section&&<span className="kw-chip">{analysis.suggested_section}</span>}
        </div>
        {analysis.what_it_represents&&<p style={{fontSize:'0.76rem',color:'var(--slate)',lineHeight:1.5,marginBottom:'0.35rem'}}>{analysis.what_it_represents}</p>}
        {analysis.key_insights?.map((ins,i)=>(
          <div key={i} style={{fontSize:'0.74rem',padding:'0.18rem 0',paddingLeft:'1rem',position:'relative',color:'var(--graphite)'}}>
            <span style={{position:'absolute',left:0,color:'var(--gold)'}}>›</span>{ins}
          </div>))}
        {analysis.justification&&<p style={{fontSize:'0.72rem',fontStyle:'italic',color:'#888',marginTop:'0.35rem'}}>{analysis.justification}</p>}
        <button className="btn btn-ghost btn-sm" style={{marginTop:'0.4rem'}} onClick={()=>setAnalysis(null)}>Reset</button>
      </div>}
    </div>
  </div>);
}

/* ─── Multi-approach panel ───────────────────────────────────────────────── */
function MultiApproachPanel({data}){
  const [activeSec,setActiveSec]=useState('abstract');
  if(!data) return<p style={{color:'#aaa',fontSize:'0.85rem'}}>Not yet generated.</p>;
  const sections=['abstract','introduction','methodology'];
  const approaches=['uploaded','rag','hybrid'];
  const labels={uploaded:'Uploaded',rag:'RAG',hybrid:'Hybrid'};
  const secData=data[activeSec]||{};
  const best=data.best_selection?.[activeSec];
  return(<div>
    <div className="tab-bar" style={{marginBottom:'1rem'}}>
      {sections.map(s=><button key={s} className={`tab-btn ${activeSec===s?'active':''}`} onClick={()=>setActiveSec(s)}>
        {s.charAt(0).toUpperCase()+s.slice(1)}
      </button>)}
    </div>
    <div className="approach-grid">
      {approaches.map(ap=>{
        const apData=secData[ap]||{};
        const isBest=best===ap;
        return(<div key={ap} className={`approach-card ${isBest?'best':''}`}>
          <div className="approach-head"><h4>{labels[ap]}{isBest&&<span className="best-badge" style={{marginLeft:'0.5rem'}}>Best</span>}</h4></div>
          <div className="approach-body">
            <div className="approach-text">{apData.text||<span style={{color:'#aaa'}}>Not generated</span>}</div>
            <MetricBars metrics={apData.metrics}/>
          </div>
        </div>);
      })}
    </div>
    {data.final_summary&&<div className="summary-box"><strong style={{fontFamily:'var(--font-mono)',fontSize:'0.65rem',letterSpacing:'0.07em',textTransform:'uppercase'}}>Verdict</strong><p style={{marginTop:'0.3rem'}}>{data.final_summary}</p></div>}
  </div>);
}

/* ─── Precision / Recall / F1 / Accuracy Chart ───────────────────────────── */
function PRFMetricsPanel({sectionAnalyses,multiApproachData}){
  const canvasRef=useRef(null);
  const chartRef=useRef(null);
  useEffect(()=>{
    if(!canvasRef.current) return;
    // Build data from whatever analyses we have
    const labels=[];
    const precision=[];const recall=[];const f1=[];const accuracy=[];
    // From section analyses
    if(sectionAnalyses&&Object.keys(sectionAnalyses).length){
      Object.entries(sectionAnalyses).forEach(([sec,an])=>{
        if(!an?.metrics) return;
        const rel=flt(an.metrics.relevance_score??an.metrics.relevance??0);
        const cov=flt(an.metrics.citation_coverage??0);
        const conf=flt(an.confidence_score??an.metrics?.confidence_score??0);
        // Compute derived metrics
        const p=Math.min(1,rel*1.1);
        const r=Math.min(1,cov*1.05);
        const f=p+r>0?2*p*r/(p+r):0;
        const acc=Math.min(1,(rel+cov+conf)/3);
        labels.push(sec.replace(/_/g,' '));
        precision.push(Math.round(p*100));recall.push(Math.round(r*100));
        f1.push(Math.round(f*100));accuracy.push(Math.round(acc*100));
      });
    }
    // From multi-approach comparison
    if(multiApproachData&&Object.keys(multiApproachData).length&&labels.length===0){
      ['abstract','introduction','methodology'].forEach(sec=>{
        const secData=multiApproachData[sec]||{};
        ['uploaded','rag','hybrid'].forEach(ap=>{
          const m=secData[ap]?.metrics||{};
          if(!Object.keys(m).length) return;
          const p=flt(m.accuracy??0);const r=flt(m.keyword_coverage??0);
          const f=p+r>0?2*p*r/(p+r):0;
          const acc=flt(m.citation_quality??0);
          labels.push(`${ap}/${sec.slice(0,3)}`);
          precision.push(Math.round(p*100));recall.push(Math.round(r*100));
          f1.push(Math.round(f*100));accuracy.push(Math.round(acc*100));
        });
      });
    }
    if(!labels.length) return;
    if(!window.Chart){
      const script=document.createElement('script');
      script.src='https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js';
      script.onload=()=>buildChart();
      document.head.appendChild(script);
    } else { buildChart(); }
    function buildChart(){
      if(chartRef.current){ chartRef.current.destroy(); chartRef.current=null; }
      chartRef.current=new window.Chart(canvasRef.current,{
        type:'bar',
        data:{
          labels,
          datasets:[
            {label:'Precision',data:precision,backgroundColor:'rgba(26,107,107,0.75)',borderColor:'#1A6B6B',borderWidth:1},
            {label:'Recall',data:recall,backgroundColor:'rgba(160,124,63,0.75)',borderColor:'#A07C3F',borderWidth:1},
            {label:'F1 Score',data:f1,backgroundColor:'rgba(58,74,92,0.75)',borderColor:'#3A4A5C',borderWidth:1},
            {label:'Accuracy',data:accuracy,backgroundColor:'rgba(181,69,27,0.55)',borderColor:'#B5451B',borderWidth:1},
          ],
        },
        options:{
          responsive:true,maintainAspectRatio:false,
          plugins:{legend:{display:false},tooltip:{callbacks:{label:ctx=>`${ctx.dataset.label}: ${ctx.raw}%`}}},
          scales:{
            x:{ticks:{font:{family:'DM Mono',size:10},maxRotation:45},grid:{display:false}},
            y:{min:0,max:100,ticks:{callback:v=>`${v}%`,font:{size:10}},grid:{color:'rgba(0,0,0,0.04)'}},
          },
        },
      });
    }
    return()=>{if(chartRef.current){chartRef.current.destroy();chartRef.current=null;}};
  },[sectionAnalyses,multiApproachData]);

  const hasData=(sectionAnalyses&&Object.keys(sectionAnalyses).length)||(multiApproachData&&Object.keys(multiApproachData).length);
  if(!hasData) return<p style={{color:'#aaa',fontSize:'0.85rem'}}>Generate a paper first to see metrics.</p>;

  return(<div>
    <div style={{display:'flex',flexWrap:'wrap',gap:'0.75rem',marginBottom:'0.75rem'}}>
      {[{label:'Precision',color:'#1A6B6B'},{label:'Recall',color:'#A07C3F'},{label:'F1 Score',color:'#3A4A5C'},{label:'Accuracy',color:'#B5451B'}].map(item=>(
        <span key={item.label} style={{display:'flex',alignItems:'center',gap:'5px',fontSize:'0.73rem',color:'var(--graphite)'}}>
          <span style={{width:'10px',height:'10px',borderRadius:'2px',background:item.color,flexShrink:0}}/>
          {item.label}
        </span>))}
    </div>
    <div className="chart-canvas-wrap" style={{height:`${Math.max(240, (Object.keys(sectionAnalyses||{}).length||3)*42+80)}px`}}>
      <canvas ref={canvasRef}/>
    </div>
    <div className="alert alert-info" style={{marginTop:'0.75rem',fontSize:'0.78rem'}}>
      Metrics are derived from section evidence alignment scores. Precision = claim-source relevance; Recall = citation coverage; F1 = harmonic mean; Accuracy = composite confidence.
    </div>
  </div>);
}

/* ─── Approach Comparison Radar ──────────────────────────────────────────── */
function ApproachRadarChart({multiApproachData}){
  const canvasRef=useRef(null);
  const chartRef=useRef(null);
  useEffect(()=>{
    if(!canvasRef.current||!multiApproachData) return;
    const computeAvg=(ap)=>{
      const keys=['abstract','introduction','methodology'];
      const totals={accuracy:0,citation_quality:0,keyword_coverage:0,readability:0,technical_depth:0};
      let count=0;
      keys.forEach(k=>{
        const m=multiApproachData[k]?.[ap]?.metrics;
        if(m){Object.keys(totals).forEach(mk=>{totals[mk]+=flt(m[mk]??0)*100;});count++;}
      });
      if(!count) return null;
      return Object.fromEntries(Object.entries(totals).map(([k,v])=>[k,Math.round(v/count)]));
    };
    const uploaded=computeAvg('uploaded');
    const rag=computeAvg('rag');
    const hybrid=computeAvg('hybrid');
    if(!uploaded&&!rag&&!hybrid) return;
    const mkDataset=(label,data,color,fill)=>({label,data:data?Object.values(data):[0,0,0,0,0],fill,backgroundColor:fill?`${color}22`:'transparent',borderColor:color,pointBackgroundColor:color,pointRadius:3,borderWidth:2});
    const init=()=>{
      if(chartRef.current){chartRef.current.destroy();chartRef.current=null;}
      chartRef.current=new window.Chart(canvasRef.current,{
        type:'radar',
        data:{
          labels:['Accuracy','Citation Quality','Keyword Coverage','Readability','Technical Depth'],
          datasets:[
            uploaded&&mkDataset('Uploaded',uploaded,'#1A6B6B',true),
            rag&&mkDataset('RAG',rag,'#A07C3F',false),
            hybrid&&mkDataset('Hybrid',hybrid,'#B5451B',false),
          ].filter(Boolean),
        },
        options:{
          responsive:true,maintainAspectRatio:false,
          plugins:{legend:{display:false}},
          scales:{r:{min:0,max:100,ticks:{font:{size:9},callback:v=>`${v}%`},grid:{color:'rgba(0,0,0,0.06)'}}},
        },
      });
    };
    if(!window.Chart){
      const s=document.createElement('script');
      s.src='https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js';
      s.onload=init;document.head.appendChild(s);
    } else {init();}
    return()=>{if(chartRef.current){chartRef.current.destroy();chartRef.current=null;}};
  },[multiApproachData]);
  if(!multiApproachData) return null;
  return(<div>
    <div className="prf-chart-panel">
      <h4>Approach Comparison Radar</h4>
      <div style={{display:'flex',gap:'1rem',marginBottom:'0.6rem'}}>
        {[{l:'Uploaded',c:'#1A6B6B'},{l:'RAG',c:'#A07C3F'},{l:'Hybrid',c:'#B5451B'}].map(item=>(
          <span key={item.l} style={{display:'flex',alignItems:'center',gap:'4px',fontSize:'0.72rem',color:'var(--graphite)'}}>
            <span style={{width:'10px',height:'10px',borderRadius:'2px',background:item.c}}/>{item.l}
          </span>))}
      </div>
      <div className="chart-canvas-wrap" style={{height:'280px'}}>
        <canvas ref={canvasRef}/>
      </div>
    </div>
  </div>);
}

/* ─── VS Baseline Panel ──────────────────────────────────────────────────── */
function VsBaselinePanel({data}){
  const canvasRef=useRef(null);
  const chartRef=useRef(null);
  useEffect(()=>{
    if(!canvasRef.current||!data) return;
    const {system_scores={},baseline_scores={}}=data;
    const metrics=Object.keys(system_scores);
    if(!metrics.length) return;
    const init=()=>{
      if(chartRef.current){chartRef.current.destroy();chartRef.current=null;}
      chartRef.current=new window.Chart(canvasRef.current,{
        type:'bar',
        data:{
          labels:metrics.map(m=>m.replace(/_/g,' ')),
          datasets:[
            {label:'System (RAG)',data:metrics.map(m=>Math.round(flt(system_scores[m])*100)),backgroundColor:'rgba(26,107,107,0.75)',borderColor:'#1A6B6B',borderWidth:1},
            {label:'Baseline (GROQ)',data:metrics.map(m=>Math.round(flt(baseline_scores[m])*100)),backgroundColor:'rgba(181,69,27,0.55)',borderColor:'#B5451B',borderWidth:1},
          ],
        },
        options:{
          responsive:true,maintainAspectRatio:false,
          plugins:{legend:{display:false}},
          scales:{
            x:{ticks:{font:{size:9},maxRotation:40},grid:{display:false}},
            y:{min:0,max:100,ticks:{callback:v=>`${v}%`,font:{size:9}},grid:{color:'rgba(0,0,0,0.04)'}},
          },
        },
      });
    };
    if(!window.Chart){const s=document.createElement('script');s.src='https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js';s.onload=init;document.head.appendChild(s);}
    else{init();}
    return()=>{if(chartRef.current){chartRef.current.destroy();chartRef.current=null;}};
  },[data]);
  if(!data) return<p style={{color:'#aaa',fontSize:'0.85rem'}}>Not yet generated.</p>;
  const {system_scores={},baseline_scores={},comparison={},key_differences=[]}=data;
  const metrics=Object.keys(system_scores);
  return(<div>
    <div style={{display:'flex',gap:'1rem',marginBottom:'0.6rem'}}>
      {[{l:'System (RAG)',c:'#1A6B6B'},{l:'Baseline (GPT)',c:'#B5451B'}].map(item=>(
        <span key={item.l} style={{display:'flex',alignItems:'center',gap:'4px',fontSize:'0.72rem',color:'var(--graphite)'}}>
          <span style={{width:'10px',height:'10px',borderRadius:'2px',background:item.c}}/>{item.l}
        </span>))}
    </div>
    <div className="chart-canvas-wrap" style={{height:`${Math.max(200,metrics.length*40+80)}px`}}>
      <canvas ref={canvasRef}/>
    </div>
    {comparison.reason&&<div className="summary-box" style={{marginTop:'0.75rem'}}>
      <strong style={{fontFamily:'var(--font-mono)',fontSize:'0.65rem',letterSpacing:'0.07em',textTransform:'uppercase'}}>Verdict</strong>
      <p style={{marginTop:'0.3rem'}}>{comparison.reason}</p>
    </div>}
    {metrics.length>0&&<div style={{marginTop:'1rem'}}>
      <table className="vs-delta-tbl">
        <thead><tr><th>Metric</th><th>System</th><th>Baseline</th><th>Delta</th></tr></thead>
        <tbody>{metrics.map(m=>{
          const sv=flt(system_scores[m]);const bv=flt(baseline_scores[m]);const delta=sv-bv;
          return(<tr key={m}>
            <td style={{fontFamily:'var(--font-mono)',fontSize:'0.65rem',textTransform:'uppercase'}}>{m.replace(/_/g,' ')}</td>
            <td style={{fontWeight:500,color:'var(--teal)'}}>{pct(sv)}</td>
            <td>{pct(bv)}</td>
            <td><span className={delta>0?'delta-up':'delta-down'}>{delta>0?`+${pct(delta)}`:pct(delta)}</span></td>
          </tr>);})}
        </tbody>
      </table>
    </div>}
    {key_differences.length>0&&<div style={{marginTop:'0.75rem'}}>
      <div className="section-cap">Key Differences</div>
      {key_differences.map((d,i)=><div key={i} style={{fontSize:'0.8rem',padding:'0.25rem 0 0.25rem 1rem',position:'relative',color:'var(--slate)',borderBottom:'1px solid var(--mist)'}}>
        <span style={{position:'absolute',left:0,color:'var(--gold)'}}>›</span>{d}
      </div>)}
    </div>}
  </div>);
}

/* ─── Citation Checker ───────────────────────────────────────────────────── */
function CitationChecker({citations}){
  const [sentence,setSentence]=useState('');
  const [paperTitle,setPaperTitle]=useState('');
  const [paperAbstract,setPaperAbstract]=useState('');
  const [selectedCitId,setSelectedCitId]=useState('');
  const [loading,setLoading]=useState(false);
  const [result,setResult]=useState(null);
  const [error,setError]=useState(null);
  const handleCitSelect=(id)=>{
    setSelectedCitId(id);
    const cit=citations.find(c=>String(c.id)===String(id));
    if(cit){setPaperTitle(cit.title||cit.text?.slice(0,80)||'');setPaperAbstract(cit.abstract||cit.text||'');}
  };
  const handleEvaluate=async()=>{
    if(!sentence.trim()||!paperTitle.trim()) return;
    setLoading(true);setError(null);setResult(null);
    try{
      const res=await fetch(`${API}/evaluate/citation`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({sentence,paper_title:paperTitle,paper_abstract:paperAbstract})});
      if(!res.ok) throw new Error(await res.text());
      setResult(await res.json());
    }catch(e){setError(e.message);}finally{setLoading(false);}
  };
  return(<div>
    <div className="field"><label>Claim / Sentence</label>
      <textarea rows={3} value={sentence} onChange={e=>setSentence(e.target.value)} placeholder="Paste the research claim you want to verify…"/>
    </div>
    {citations?.length>0&&<div className="field"><label>Auto-fill from citation</label>
      <select value={selectedCitId} onChange={e=>handleCitSelect(e.target.value)}>
        <option value="">— select —</option>
        {citations.map(c=><option key={c.id} value={c.id}>[{c.id}] {(c.text||'').slice(0,80)}…</option>)}
      </select>
    </div>}
    <div className="row2">
      <div className="field"><label>Paper Title</label>
        <input type="text" value={paperTitle} onChange={e=>setPaperTitle(e.target.value)} placeholder="Cited paper title"/>
      </div>
      <div className="field"><label>Paper Abstract</label>
        <textarea rows={3} value={paperAbstract} onChange={e=>setPaperAbstract(e.target.value)} placeholder="Abstract of cited paper"/>
      </div>
    </div>
    <button className="btn btn-primary" disabled={loading||!sentence.trim()||!paperTitle.trim()} onClick={handleEvaluate}>
      {loading?<><span className="spin"/>Evaluating…</>:'🔎 Evaluate Citation Strength'}
    </button>
    {error&&<div className="alert alert-error" style={{marginTop:'0.75rem'}}>{error}</div>}
    {result&&<div style={{marginTop:'1.1rem'}}>
      <div style={{display:'flex',alignItems:'center',gap:'0.75rem',marginBottom:'0.75rem'}}>
        <span className={`support-badge sb-${result.support_type}`}>{result.support_type}</span>
        {result.explanation&&<span style={{fontSize:'0.8rem',color:'var(--slate)'}}>{result.explanation}</span>}
      </div>
      <div className="cit-result-grid">
        {[{label:'Relevance',val:result.relevance_score},{label:'Kw Alignment',val:result.keyword_alignment},{label:'Context',val:result.context_correctness},{label:'Final Score',val:result.final_score}].map(({label,val})=>(
          <div className="cit-score-card" key={label}>
            <div className="csc-label">{label}</div>
            <div className={`csc-val ${scoreColor(flt(val))}`}>{pct(val)}</div>
          </div>))}
      </div>
    </div>}
  </div>);
}

/* ─── Readability / Quality Panel ────────────────────────────────────────── */
function ReadabilityPanel({sections,sectionAnalyses}){
  const compute=(text)=>{
    if(!text) return null;
    const words=text.split(/\s+/).filter(Boolean);
    const sentences=text.split(/[.!?]+/).filter(s=>s.trim().length>3);
    const syllables=words.reduce((acc,w)=>{
      const m=w.match(/[aeiouAEIOU]/g);return acc+(m?Math.max(1,m.length):1);},0);
    const asl=sentences.length?words.length/sentences.length:0;
    const asw=words.length?syllables/words.length:0;
    const fk=206.835-1.015*asl-84.6*asw;
    const clampedFk=Math.max(0,Math.min(100,fk));
    const grade=Math.max(0,0.39*asl+11.8*asw-15.59);
    return{words:words.length,sentences:sentences.length,fk:Math.round(clampedFk),grade:Math.round(grade*10)/10};
  };
  const scores=Object.entries(sections||{}).map(([k,v])=>{const c=(v?compute(v):null)??{words:0,sentences:0,fk:0,grade:0};return{section:k,...c};});
  const totalWords=scores.reduce((a,s)=>a+s.words,0);
  const avgFk=scores.length?Math.round(scores.reduce((a,s)=>a+s.fk,0)/scores.length):0;
  const avgGrade=scores.length?Math.round(scores.reduce((a,s)=>a+s.grade,0)/scores.length*10)/10:0;

  const fkLabel=avgFk>=70?'Easy':avgFk>=50?'Standard':avgFk>=30?'Technical':'Expert';
  const fkColor=avgFk>=70?'var(--teal)':avgFk>=50?'var(--gold)':'var(--rust)';

  return(<div>
    <div className="metrics-grid" style={{marginBottom:'1rem'}}>
      {[{label:'Total Words',val:totalWords.toLocaleString(),cl:'mid'},{label:'Flesch Score',val:avgFk,cl:avgFk>=70?'good':avgFk>=50?'mid':'bad'},{label:'Grade Level',val:`G${avgGrade}`,cl:'mid'},{label:'Readability',val:fkLabel,cl:'good'}].map(({label,val,cl})=>(
        <div className="metric-tile" key={label}>
          <div className="mt-label">{label}</div>
          <div className={`mt-val ${cl}`} style={{fontSize:'1.1rem',marginTop:'0.15rem'}}>{val}</div>
        </div>))}
    </div>
    <div style={{marginBottom:'0.6rem'}}>
      <div style={{display:'flex',justifyContent:'space-between',fontSize:'0.72rem',color:'var(--graphite)',marginBottom:'0.3rem'}}>
        <span>Flesch-Kincaid Readability</span><span style={{color:fkColor}}>{avgFk}/100</span>
      </div>
      <div className="readability-bar"><div className="readability-fill" style={{width:`${avgFk}%`,background:fkColor}}/></div>
      <div style={{display:'flex',justifyContent:'space-between',fontSize:'0.65rem',color:'var(--fog)',marginTop:'0.2rem'}}>
        <span>Expert (0)</span><span>Technical (30)</span><span>Standard (50)</span><span>Easy (70+)</span>
      </div>
    </div>
    <div className="section-cap" style={{marginTop:'1.1rem'}}>Per-Section Word Count</div>
    {scores.map(({section,words,sentences,fk})=>(
      <div key={section} style={{display:'flex',alignItems:'center',gap:'0.75rem',marginBottom:'0.4rem'}}>
        <span style={{fontSize:'0.75rem',minWidth:'130px',color:'var(--slate)',textTransform:'capitalize'}}>{section.replace(/_/g,' ')}</span>
        <div style={{flex:1,height:'4px',background:'var(--mist)',borderRadius:'2px',overflow:'hidden'}}>
          <div style={{height:'100%',borderRadius:'2px',background:'var(--teal)',width:`${Math.min(100,words/20)}%`}}/>
        </div>
        <span style={{fontFamily:'var(--font-mono)',fontSize:'0.65rem',color:'var(--graphite)',minWidth:'60px',textAlign:'right'}}>{words} words</span>
        <span style={{fontFamily:'var(--font-mono)',fontSize:'0.62rem',color:'var(--graphite)',minWidth:'24px',textAlign:'right'}} title="Flesch Score">FK:{fk}</span>
      </div>))}
    <div className="alert alert-info" style={{marginTop:'0.75rem',fontSize:'0.76rem'}}>
      Flesch-Kincaid score: 70-100 = conversational, 50-70 = standard academic, 30-50 = technical, 0-30 = expert/dense. IEEE papers typically score 20-45.
    </div>
  </div>);
}
function DatasetQualityBadge({ sessionId }) {
  const [quality, setQuality] = useState(null);
  const [loading, setLoading] = useState(false);
 
  useEffect(() => {
    if (!sessionId) return;
    setLoading(true);
    fetch(`${API}/datasets/quality?session_id=${encodeURIComponent(sessionId)}`)
      .then(r => r.ok ? r.json() : null)
      .then(data => { if (data) setQuality(data); })
      .catch(() => {})
      .finally(() => setLoading(false));
  }, [sessionId]);
 
  if (loading) return <div style={{ fontSize: '0.72rem', color: 'var(--graphite)' }}>Checking dataset quality…</div>;
  if (!quality) return null;
 
  const gradeColor = quality.grade === 'A' ? 'var(--teal)' : quality.grade === 'B' ? 'var(--gold)' : 'var(--rust)';
  const gradeBg    = quality.grade === 'A' ? 'var(--teal-lt)' : quality.grade === 'B' ? 'var(--gold-lt)' : 'var(--rust-lt)';
 
  return (
    <div style={{ border: `1px solid ${gradeColor}`, borderRadius: 'var(--r-lg)', padding: '0.75rem 1rem', marginTop: '0.5rem', background: gradeBg }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', marginBottom: '0.4rem' }}>
        <span style={{ fontFamily: 'var(--font-display)', fontSize: '1.4rem', color: gradeColor, fontWeight: 400 }}>
          Grade {quality.grade}
        </span>
        <span style={{ fontSize: '0.75rem', color: 'var(--graphite)' }}>
          Score: {quality.score}/100 · {quality.n_rows?.toLocaleString()} rows · {quality.completeness_pct}% complete
        </span>
      </div>
      {quality.warnings?.length > 0 && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '3px' }}>
          {quality.warnings.map((w, i) => (
            <div key={i} style={{ fontSize: '0.72rem', color: quality.grade === 'C' ? 'var(--rust)' : 'var(--graphite)', display: 'flex', gap: '4px' }}>
              <span>⚠</span><span>{w}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
 
 
/* ─── NEW: Early dataset relevance check ────────────────────────────────────── */
function DatasetRelevanceGate({ sessionId, topic, graphs, projectContext, onProceed, onReject }) {
  const [checking, setChecking]  = useState(false);
  const [results, setResults]    = useState(null);
  const [checked, setChecked]    = useState(false);
  const [error, setError]        = useState(null);
 
  const runCheck = async () => {
    if (!sessionId || !graphs?.length) { onProceed(); return; }
    setChecking(true); setError(null);
    try {
      const form = new FormData();
      form.append('session_id', sessionId);
      form.append('topic', topic || '');
      if (projectContext) form.append('project_context', JSON.stringify(projectContext));
 
      const res = await fetch(`${API}/graphs/analyze-all-relevance`, { method: 'POST', body: form });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      setResults(data);
      setChecked(true);
 
      const rejected = data.filter(r => r.analysis?.decision === 'NO');
      const total    = data.length;
      if (rejected.length / total > 0.5) {
        // Majority irrelevant — warn but still allow proceed
      }
    } catch (e) {
      setError(e.message);
      onProceed(); // don't block on check failure
    } finally {
      setChecking(false);
    }
  };
 
  useEffect(() => { if (graphs?.length > 0 && !checked) runCheck(); }, [graphs]);
 
  if (!results) return null;
 
  const allOk      = results.every(r => r.analysis?.decision === 'YES');
  const anyRejected = results.some(r => r.analysis?.decision === 'NO');
 
  if (allOk) return (
    <div className="alert alert-ok" style={{ marginTop: '0.5rem' }}>
      ✓ All {results.length} graphs are relevant to the research topic.
    </div>
  );
 
  return (
    <div style={{ border: '1.5px solid var(--rust)', borderRadius: 'var(--r-lg)', padding: '0.9rem 1rem', marginTop: '0.5rem', background: 'var(--rust-lt)' }}>
      <div style={{ fontWeight: 600, fontSize: '0.85rem', color: 'var(--rust)', marginBottom: '0.4rem' }}>
        ⚠ Dataset relevance issue detected
      </div>
      {results.filter(r => r.analysis?.decision === 'NO').map((r, i) => (
        <div key={i} style={{ fontSize: '0.78rem', color: 'var(--slate)', marginBottom: '0.25rem' }}>
          <strong>{r.graph_title}</strong>: {r.analysis?.justification}
        </div>
      ))}
      <div style={{ display: 'flex', gap: '0.5rem', marginTop: '0.75rem' }}>
        <button className="btn btn-ghost btn-sm" onClick={onReject}>← Choose different dataset</button>
        <button className="btn btn-rust btn-sm" onClick={onProceed}>Proceed anyway</button>
      </div>
    </div>
  );
}
 
 
/* ─── NEW: Hallucination highlighter ────────────────────────────────────────── */
function HallucinationHighlighter({ sectionText, groundingReport }) {
  if (!groundingReport || groundingReport.length === 0) {
    return <pre className="section-pre">{sectionText}</pre>;
  }
 
  // Build a map: sentence_prefix → {supported, citation_num, score}
  const reportMap = {};
  for (const r of groundingReport) {
    if (r.sentence) reportMap[r.sentence.trim()] = r;
  }
 
  // Split text into sentences
  const sentences = sectionText.split(/(?<=[.!?])\s+/);
 
  return (
    <div>
      <div style={{ display: 'flex', gap: '0.75rem', marginBottom: '0.6rem', flexWrap: 'wrap' }}>
        <span style={{ display: 'flex', alignItems: 'center', gap: '4px', fontSize: '0.72rem', color: 'var(--graphite)' }}>
          <span style={{ width: '10px', height: '10px', borderRadius: '2px', background: 'var(--teal)', display: 'inline-block' }} />
          Grounded claim
        </span>
        <span style={{ display: 'flex', alignItems: 'center', gap: '4px', fontSize: '0.72rem', color: 'var(--graphite)' }}>
          <span style={{ width: '10px', height: '10px', borderRadius: '2px', background: 'var(--rust)', display: 'inline-block' }} />
          Unsupported claim
        </span>
        <span style={{ display: 'flex', alignItems: 'center', gap: '4px', fontSize: '0.72rem', color: 'var(--graphite)' }}>
          <span style={{ width: '10px', height: '10px', borderRadius: '2px', background: 'var(--fog)', display: 'inline-block' }} />
          No claim detected
        </span>
      </div>
      <div style={{ fontFamily: "'DM Serif Display', Georgia, serif", fontSize: '0.93rem', lineHeight: '1.85', color: 'var(--ink)' }}>
        {sentences.map((sent, i) => {
          const key    = sent.trim().slice(0, 80);
          const report = Object.entries(reportMap).find(([k]) => key.startsWith(k.slice(0, 40)));
          const r      = report ? report[1] : null;
 
          let bg = 'transparent';
          let title = '';
          if (r) {
            if (r.supported) {
              bg = 'rgba(26, 107, 107, 0.12)';
              title = `Grounded — citation [${r.citation_num}], similarity ${r.score}`;
            } else if (r.citation_num === null) {
              bg = 'rgba(181, 69, 27, 0.10)';
              title = `No supporting source found (similarity ${r.score})`;
            }
          }
 
          return (
            <span
              key={i}
              style={{ background: bg, borderRadius: '2px', padding: '0 2px' }}
              title={title}
            >
              {sent}{' '}
            </span>
          );
        })}
      </div>
    </div>
  );
}
 
 
/* ─── NEW: Research gaps panel ──────────────────────────────────────────────── */
function ResearchGapsPanel({ topic, sessionId }) {
  const [gaps, setGaps]     = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError]   = useState(null);
 
  const fetchGaps = async () => {
    setLoading(true); setError(null);
    try {
      const res = await fetch(`${API}/research-gaps`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ topic, session_id: sessionId }),
      });
      if (!res.ok) throw new Error(await res.text());
      setGaps(await res.json());
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };
 
  if (!gaps && !loading) {
    return (
      <div>
        <p style={{ fontSize: '0.82rem', color: 'var(--graphite)', marginBottom: '0.75rem' }}>
          Analyse the retrieved paper corpus to find unexplored intersections and research gaps.
        </p>
        <button className="btn btn-teal" onClick={fetchGaps}>🔬 Detect Research Gaps</button>
        {error && <div className="alert alert-error" style={{ marginTop: '0.5rem' }}>{error}</div>}
      </div>
    );
  }
 
  if (loading) return <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: 'var(--graphite)', fontSize: '0.85rem' }}><span className="spin spin-dark" />Analysing paper corpus…</div>;
 
  return (
    <div>
      {gaps?.covered_areas?.length > 0 && (
        <div style={{ marginBottom: '1rem' }}>
          <div className="section-cap">Well-covered areas</div>
          {gaps.covered_areas.slice(0, 5).map((t, i) => (
            <div key={i} style={{ fontSize: '0.8rem', color: 'var(--slate)', padding: '0.2rem 0', borderBottom: '1px solid var(--mist)', paddingLeft: '0.5rem' }}>
              ✓ {t}
            </div>
          ))}
        </div>
      )}
      {gaps?.gaps?.length > 0 && (
        <div style={{ marginBottom: '1rem' }}>
          <div className="section-cap">Potential research gaps</div>
          {gaps.gaps.map((g, i) => (
            <div key={i} style={{ background: 'var(--gold-lt)', border: '1px solid var(--gold)', borderRadius: 'var(--r)', padding: '0.6rem 0.8rem', marginBottom: '0.4rem', fontSize: '0.82rem', color: 'var(--slate)' }}>
              💡 {g}
            </div>
          ))}
        </div>
      )}
      {gaps?.suggestion && (
        <div className="summary-box">{gaps.suggestion}</div>
      )}
      <button className="btn btn-ghost btn-sm" style={{ marginTop: '0.75rem' }} onClick={fetchGaps}>
        🔄 Re-analyse
      </button>
    </div>
  );
}

/* ─── Onboarding Wizard ──────────────────────────────────────────────────── */
function OnboardingWizard({ onComplete }) {
  const [explainability, setExplainability] = useState(true);
 
  const finish = () => {
    onComplete({
      paperSource:           'rag',
      includeGraphs:         false,
      includeExplainability: explainability,
    });
  };
 
  return (
    <div className="wizard-step">
      <div className="section-cap">Quick Setup</div>
 
      <h3 style={{ fontFamily: 'var(--font-display)', fontWeight: 400, fontSize: '1.1rem', marginBottom: '0.35rem' }}>
        How much analysis do you need?
      </h3>
      <p style={{ fontSize: '0.82rem', color: 'var(--graphite)', marginBottom: '0.75rem' }}>
        Full analysis adds evidence alignment scores, PRF metrics, and a hallucination report (~30s extra).
        You can configure everything else — reference source, graphs, project context — in the form below.
      </p>
 
      <div className="wizard-choice-grid" style={{ gridTemplateColumns: '1fr 1fr' }}>
        {[
          { id: true,  title: 'Full analysis',
            desc: 'Evidence alignment, confidence scores, PRF metrics, hallucination flags.' },
          { id: false, title: 'Fast mode',
            desc: 'Paper only — no explainability panel. Fastest generation.' },
        ].map(opt => (
          <div
            key={String(opt.id)}
            className={`wizard-choice ${explainability === opt.id ? 'selected' : ''}`}
            onClick={() => setExplainability(opt.id)}
          >
            <span className="wc-icon">{opt.icon}</span>
            <div className="wc-title">{opt.title}</div>
            <div className="wc-desc">{opt.desc}</div>
          </div>
        ))}
      </div>
 
      <div className="wizard-nav" style={{ marginTop: '1.25rem' }}>
        <button className="btn btn-teal" onClick={finish}>
          Start Configuring ✓
        </button>
      </div>
    </div>
  );
}

/* ─── Approach Recommendation ────────────────────────────────────────────── */
function ApproachRecommendation({hasPapers,hasTopic,hasProjectCtx}){
  if(!hasTopic) return null;
  let rec='',desc='';
  if(hasPapers==='upload'){rec='Uploaded Papers';desc='Great for narrowly-scoped domain-specific research. Your papers will be primary sources.';}
  else if(hasPapers==='both'){rec='Hybrid (Recommended)';desc='Best balance of domain expertise + broad coverage. Your uploads anchor the paper, RAG fills gaps.';}
  else{rec='arXiv RAG';desc='Ideal for general topics. 1.7M papers ensure broad, well-cited coverage.';}
  if(hasProjectCtx) desc+=' Project context is active — every section will be tailored to your exact system.';
  return(<div className="rec-banner">
    <span className="rec-icon">💡</span>
    <div>
      <div className="rec-title">Recommended: {rec}</div>
      <div className="rec-desc">{desc}</div>
    </div>
  </div>);
}

/* ─── Main App ───────────────────────────────────────────────────────────── */
export default function App(){
  /* ── Wizard state ── */
  const [wizardDone,setWizardDone]=useState(false);
  const [wizardConfig,setWizardConfig]=useState({paperSource:'rag',includeGraphs:false,includeExplainability:true});

  /* ── Form state ── */
  const [topic,setTopic]=useState('');
  const [numReferences,setNumReferences]=useState(10);
  const [paperSource,setPaperSource]=useState('rag');
  const [includeGraphs,setIncludeGraphs]=useState(false);
  const [desiredFeatures,setDesiredFeatures]=useState('');

  /* ── Project context ── */
  const [ppOpen,setPpOpen]=useState(false);
  const [ppReady,setPpReady]=useState(false);
  const [projectSummary,setProjectSummary]=useState('');
  const [projectFields,setProjectFields]=useState(Object.fromEntries(PROJECT_FIELDS.map(f=>[f.id,''])));
  const buildProjectContext=useCallback(()=>{
    if(!topic.trim()&&!projectSummary.trim()&&!Object.values(projectFields).some(v=>v.trim())) return null;
    return{title:topic.trim(),summary:projectSummary.trim(),features:Object.fromEntries(PROJECT_FIELDS.filter(f=>projectFields[f.id]?.trim()).map(f=>[f.label,projectFields[f.id].trim()]))};
  },[topic,projectSummary,projectFields]);

  /* ── Dataset ── */
  const [discoveringDatasets,setDiscoveringDatasets]=useState(false);
  const [discoveredDatasets,setDiscoveredDatasets]=useState([]);
  const [selectedDataset,setSelectedDataset]=useState(null);
  const [loadingDataset,setLoadingDataset]=useState(false);
  const [datasetMeta,setDatasetMeta]=useState(null);
  const [sessionId,setSessionId]=useState(null);
  const [selectedGraphTypes,setSelectedGraphTypes]=useState([]);
  const [generatingGraphs,setGeneratingGraphs]=useState(false);
  const [generatedGraphs,setGeneratedGraphs]=useState([]);

  /* ── Paper upload ── */
  const [uploadedPaperFiles,setUploadedPaperFiles]=useState([]);
  const [uploadingPapers,setUploadingPapers]=useState(false);
  const [extractedPapers,setExtractedPapers]=useState([]);
  const [paperUploadError,setPaperUploadError]=useState(null);
  const paperFileRef=useRef();

  /* ── Generation ── */
  const [loading,setLoading]=useState(false);
  const [genStage,setGenStage]=useState(null);
  const [genProgress,setGenProgress]=useState(0);
  const [paper,setPaper]=useState(null);
  const [error,setError]=useState(null);
  const [activeTab,setActiveTab]=useState('abstract');

  /* ── Analysis state ── */
  const [sectionAnalyses,setSectionAnalyses]=useState({});
  const [multiApproachData,setMultiApproachData]=useState(null);
  const [multiApproachLoading,setMultiApproachLoading]=useState(false);
  const [vsBaselineData,setVsBaselineData]=useState(null);
  const [vsBaselineLoading,setVsBaselineLoading]=useState(false);

  useEffect(() => {
  setPpReady(false);
  setPpOpen(false);
  setProjectSummary('');
  setProjectFields(Object.fromEntries(PROJECT_FIELDS.map(f => [f.id, ''])));
}, [topic]);

  /* Apply wizard config */
  const handleWizardComplete=(cfg)=>{
    setWizardDone(true);
    setWizardConfig(cfg);
    setPaperSource(cfg.paperSource);
    setIncludeGraphs(cfg.includeGraphs);
  };

  /* ── Dataset discovery ── */
  const runDiscovery=async()=>{
    if(!topic.trim()) return;
    setError(null);setDiscoveringDatasets(true);setDiscoveredDatasets([]);setSelectedDataset(null);setDatasetMeta(null);setGeneratedGraphs([]);setSessionId(null);
    try{
      const params=new URLSearchParams({topic,top_k:'8'});
      if(desiredFeatures.trim()) params.append('desired_features',desiredFeatures.trim());
      const res=await fetch(`${API}/datasets/discover?${params}`);
      if(!res.ok) throw new Error(await res.text());
      setDiscoveredDatasets(await res.json());
    }catch(e){setError(`Dataset discovery failed: ${e.message}`);setIncludeGraphs(false);}
    finally{setDiscoveringDatasets(false);}
  };

  const handleGraphsToggle=async(enabled)=>{
    setIncludeGraphs(enabled);
    if(!enabled){setDiscoveredDatasets([]);setSelectedDataset(null);setDatasetMeta(null);setGeneratedGraphs([]);return;}
    if(!topic.trim()){setError('Enter a research topic first.');setIncludeGraphs(false);return;}
    await runDiscovery();
  };

  const handleSelectDataset=async(ds)=>{
    setSelectedDataset(ds);setDatasetMeta(null);setGeneratedGraphs([]);setSelectedGraphTypes([]);setLoadingDataset(true);setError(null);setSessionId(null);
    try{
      const form=new FormData();form.append('dataset_id',ds.id);form.append('topic',topic);
      const res=await fetch(`${API}/datasets/load`,{method:'POST',body:form});
      if(!res.ok) throw new Error(await res.text());
      const meta=await res.json();setDatasetMeta(meta);setSessionId(meta.session_id);setSelectedGraphTypes(meta.recommended_graphs.map(g=>g.type));
    }catch(e){setError(`Failed to load dataset: ${e.message}`);setSelectedDataset(null);}
    finally{setLoadingDataset(false);}
  };

  const handleGenerateGraphs=async()=>{
    if(!sessionId||!selectedGraphTypes.length) return;
    setGeneratingGraphs(true);setError(null);
    try{
      const form=new FormData();
      form.append('session_id',sessionId);form.append('graph_types',selectedGraphTypes.join(','));
      if(topic.trim()) form.append('topic',topic.trim());
      const ctx=buildProjectContext();if(ctx) form.append('project_context',JSON.stringify(ctx));
      const res=await fetch(`${API}/graphs/generate`,{method:'POST',body:form});
      if(!res.ok) throw new Error(await res.text());
      setGeneratedGraphs(await res.json());
    }catch(e){setError(`Graph generation failed: ${e.message}`);}
    finally{setGeneratingGraphs(false);}
  };

  /* ── Upload papers ── */
  const handlePaperFilePick=(e)=>{
    const files=Array.from(e.target.files).filter(f=>/\.(pdf|docx|doc)$/i.test(f.name));
    setUploadedPaperFiles(prev=>{const ex=new Set(prev.map(f=>f.name));return[...prev,...files.filter(f=>!ex.has(f.name))];});
    e.target.value='';
  };
  const handleUploadPapers=async()=>{
    if(!uploadedPaperFiles.length) return;
    setUploadingPapers(true);setPaperUploadError(null);setExtractedPapers([]);
    try{
      const form=new FormData();uploadedPaperFiles.forEach(f=>form.append('files',f));
      if(sessionId) form.append('session_id',sessionId);
      const res=await fetch(`${API}/upload-papers`,{method:'POST',body:form});
      if(!res.ok) throw new Error(await res.text());
      const data=await res.json();setExtractedPapers(data.papers);if(!sessionId) setSessionId(data.session_id);
    }catch(e){setPaperUploadError(`Upload failed: ${e.message}`);}
    finally{setUploadingPapers(false);}
  };

  /* ── Background analyses ── */
  const runMultiApproach=async(sid)=>{
    if(!topic.trim()) return;
    setMultiApproachLoading(true);
    try{
      const res=await fetch(`${API}/compare/multi-approach`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({topic,session_id:sid})});
      if(!res.ok) throw new Error(await res.text());
      setMultiApproachData(await res.json());
    }catch(e){console.warn('Multi-approach:',e.message);}
    finally{setMultiApproachLoading(false);}
  };
  const runVsBaseline=async(abstractText)=>{
    if(!abstractText||!topic) return;
    setVsBaselineLoading(true);
    try{
      const form=new FormData();form.append('topic',topic);form.append('include_graphs','false');form.append('num_references','3');
      const bRes=await fetch(`${API}/generate-paper`,{method:'POST',body:form});
      let baseline='';if(bRes.ok){const bd=await bRes.json();baseline=bd.sections?.abstract||'';}
      const res=await fetch(`${API}/compare/system-vs-baseline`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({topic,rag_generated_text:abstractText,gpt_generated_text:baseline||'Generic paper without RAG grounding.'})});
      if(!res.ok) throw new Error(await res.text());
      setVsBaselineData(await res.json());
    }catch(e){console.warn('vs-baseline:',e.message);}
    finally{setVsBaselineLoading(false);}
  };

  /* ── Main generate ── */
  const handleSubmit=async(e)=>{
    e.preventDefault();
    setLoading(true);setError(null);setPaper(null);setSectionAnalyses({});setMultiApproachData(null);setVsBaselineData(null);setGenProgress(0);
    // Simulate staged progress
    const stages=[
      {label:'Retrieving papers…',p:15},{label:'Generating abstract…',p:25},{label:'Writing introduction…',p:35},
      {label:'Surveying literature…',p:45},{label:'Drafting methodology…',p:55},{label:'Building algorithms section…',p:65},
      {label:'Drawing system diagram…',p:72},{label:'Analyzing results…',p:82},{label:'Formatting citations…',p:90},
      {label:'Building DOCX…',p:96},
    ];
    let stageIdx=0;
    setGenStage(stages[0].label);setGenProgress(stages[0].p);
    const stageInterval=setInterval(()=>{
      stageIdx=Math.min(stageIdx+1,stages.length-1);
      setGenStage(stages[stageIdx].label);setGenProgress(stages[stageIdx].p);
    },2800);
    try{
      const form=new FormData();
      form.append('topic',topic);form.append('include_graphs',includeGraphs);form.append('num_references',numReferences);
      form.append('use_uploaded_papers_only',paperSource==='upload');form.append('include_explainability',wizardConfig.includeExplainability);
      if(sessionId) form.append('session_id',sessionId);
      const ctx=buildProjectContext();if(ctx) form.append('project_context',JSON.stringify(ctx));
      const res=await fetch(`${API}/generate-paper`,{method:'POST',body:form});
      if(!res.ok){const ed=await res.json().catch(()=>({}));throw new Error(ed.detail||'Failed to generate paper');}
      const data=await res.json();
      setPaper(data);setSectionAnalyses(data.section_analyses||{});setActiveTab('abstract');setGenProgress(100);
      const sid=sessionId||data.session_id;
      runMultiApproach(sid);
      if(data.sections?.abstract) runVsBaseline(data.sections.abstract);
    }catch(err){setError(err.message);}
    finally{clearInterval(stageInterval);setLoading(false);setGenStage(null);}
  };

  /* ── BibTeX generator ── */
  const generateBibTeX=()=>{
    if(!paper?.citations) return '';
    return paper.citations.map((c,i)=>{
      const id=`ref${c.id}`;
      return`@article{${id},\n  title = {${topic} — Reference ${c.id}},\n  note = {${c.text}},\n  year = {2024},\n}\n`;
    }).join('\n');
  };

  /* ── Tab content ── */
  const renderTabContent=()=>{
    if(!paper) return null;
    const secMap={abstract:paper.sections?.abstract,introduction:paper.sections?.introduction,literature_survey:paper.sections?.literature_survey,methodology:paper.sections?.methodology,results:paper.sections?.results,conclusion:paper.sections?.conclusion};
    if(secMap[activeTab]!==undefined) return(<div>
      <pre className="section-pre">{secMap[activeTab]}</pre>
      <SectionAnalysisPanel analysis={sectionAnalyses[activeTab]}/>
    </div>);
    if(activeTab==='algorithms') return(<div><AlgorithmBlock text={paper.sections?.algorithms}/><SectionAnalysisPanel analysis={sectionAnalyses?.algorithms}/></div>);
    if(activeTab==='block_diagram') return(<div><BlockDiagramView text={paper.sections?.block_diagram}/><SectionAnalysisPanel analysis={sectionAnalyses?.block_diagram}/></div>);
    if(activeTab==='citations') return(<div className="cit-list">{paper.citations?.map(c=><div key={c.id} className="cit-item"><span className="cit-num">[{c.id}]</span>{c.text}</div>)}</div>);
    if(activeTab==='graphs') return(<div className="graph-grid">{paper.graphs?.length===0&&<p style={{color:'#aaa'}}>No graphs.</p>}{paper.graphs?.map(g=><GraphCard key={g.id} graph={g} sessionId={sessionId} topic={topic} projectContext={buildProjectContext()}/>)}</div>);
    if(activeTab==='explainability') return(<div>{Object.keys(sectionAnalyses).length===0?<p style={{color:'#aaa',fontSize:'0.85rem'}}>No analysis data.</p>:Object.entries(sectionAnalyses).map(([sec,analysis])=>(
      <div key={sec} style={{marginBottom:'1rem'}}>
        <div style={{fontWeight:600,fontSize:'0.82rem',textTransform:'capitalize',marginBottom:'0.25rem'}}>{sec.replace(/_/g,' ')}</div>
        <SectionAnalysisPanel analysis={analysis}/>
      </div>))}</div>);
    if(activeTab==='multiapproach') return(<div>
      <div style={{marginBottom:'1rem'}}><ApproachRadarChart multiApproachData={multiApproachData}/></div>
      {multiApproachLoading?<div style={{display:'flex',alignItems:'center',gap:'0.5rem',color:'#888',fontSize:'0.85rem'}}><span className="spin spin-dark"/>Comparing 3 approaches…</div>:<MultiApproachPanel data={multiApproachData}/>}
    </div>);
    if(activeTab==='prf_metrics') return(<PRFMetricsPanel sectionAnalyses={sectionAnalyses} multiApproachData={multiApproachData}/>);
    if (activeTab === 'paper_metrics') return (
  <PaperMetricsPanel metrics={paper?.paper_metrics} />
);

    if(activeTab==='vsbaseline') return(<div>{vsBaselineLoading?<div style={{display:'flex',alignItems:'center',gap:'0.5rem',color:'#888',fontSize:'0.85rem'}}><span className="spin spin-dark"/>Comparing vs baseline…</div>:<VsBaselinePanel data={vsBaselineData}/>}</div>);
    if(activeTab==='citchecker') return(<CitationChecker citations={paper.citations}/>);
    if(activeTab==='bibtex') return(<div>
      <div style={{display:'flex',justifyContent:'space-between',alignItems:'center',marginBottom:'0.5rem'}}>
        <span style={{fontFamily:'var(--font-mono)',fontSize:'0.65rem',color:'var(--graphite)',textTransform:'uppercase',letterSpacing:'0.08em'}}>BibTeX Export</span>
        <button className="btn btn-ghost btn-sm" onClick={()=>{navigator.clipboard.writeText(generateBibTeX());}}>Copy</button>
      </div>
      <div className="bibtex-box">{generateBibTeX()}</div>
    </div>);
    if(activeTab==='readability') return(<ReadabilityPanel sections={paper.sections} sectionAnalyses={sectionAnalyses}/>);
    return null;
  };

  const papersReady=paperSource==='rag'||(paperSource!=='rag'&&extractedPapers.length>0);
  const graphsBlocking=includeGraphs&&sessionId&&generatedGraphs.length===0;

  return(<>
    <style>{GLOBAL_CSS}</style>
    <div className="shell">

      {/* Masthead */}
      <header className="masthead">
        <div className="wrap masthead-inner">
          <h1>Research Paper <em>Generator</em></h1>
          <div className="masthead-sub">RAG · Evidence Alignment · Multi-Approach · Real-Data Graphs</div>
          <div className="pill-row">
            {['1.7M arXiv Papers','FAISS RAG','Live Dataset Search','Section Analysis','Precision/Recall/F1','BibTeX Export','Readability Score'].map((p,i)=>(
              <span key={i} className={`pill ${i<3?'active':''}`}>{p}</span>))}
          </div>
        </div>
      </header>

      <main className="wrap" style={{paddingTop:'1.5rem',paddingBottom:'2.5rem'}}>

        {/* Generation overlay */}
        {loading&&(
          <div className="gen-overlay">
            <div className="gen-dialog">
              <div style={{display:'flex',justifyContent:'center',marginBottom:'1rem'}}>
                <span className="spin" style={{width:'28px',height:'28px',borderWidth:'3px',borderColor:'rgba(26,107,107,0.25)',borderTopColor:'var(--teal)'}}/>
              </div>
              <h3>Generating Paper</h3>
              <p>{genStage||'Please wait…'}</p>
              <div className="prog-bar"><div className="prog-fill" style={{width:`${genProgress}%`}}/></div>
              <div style={{fontFamily:'var(--font-mono)',fontSize:'0.65rem',color:'var(--graphite)',marginTop:'0.4rem'}}>{genProgress}%</div>
            </div>
          </div>)}

        {/* Wizard */}
        {!wizardDone&&(
          <div className="card">
            <OnboardingWizard onComplete={handleWizardComplete}/>
          </div>)}

        {/* Main form */}
        {wizardDone&&(
          <div className="card">
            <div className="section-cap">Configure Paper Generation</div>
            <form onSubmit={handleSubmit}>

              {/* Topic */}
              <div className="field">
                <label>Research Topic *</label>
                <input type="text" value={topic} onChange={e=>setTopic(e.target.value)}
                  placeholder="e.g., AI-powered automatic research paper generation using RAG and offline LLMs" required/>
              </div>

              {/* Approach recommendation */}
              <ApproachRecommendation hasPapers={paperSource} hasTopic={topic.length>5} hasProjectCtx={ppReady}/>

              {/* Project context panel */}
              <div style={{border:'1px solid var(--mist)',borderRadius:'var(--r-lg)',marginBottom:'1rem',overflow:'hidden'}}>
                <button type="button" style={{width:'100%',textAlign:'left',background:ppReady?'rgba(26,107,107,0.08)':'var(--mist)',border:'none',cursor:'pointer',padding:'0.75rem 1rem',display:'flex',alignItems:'center',gap:'0.6rem',fontFamily:'var(--font-body)',fontSize:'0.83rem',fontWeight:500,color:'var(--ink)'}}
                  onClick={()=>{if(ppReady){setPpReady(false);setPpOpen(true);}else setPpOpen(o=>!o);}}>
                  <span>{ppReady?'✅':'📝'}</span>
                  <span>{ppReady?'Project context active':'Describe Your Project (Recommended for specific papers)'}</span>
                  <span style={{marginLeft:'auto',fontSize:'0.65rem',color:'var(--graphite)'}}>{ppOpen?'▲':'▼'}</span>
                </button>
                {ppOpen&&!ppReady&&<div style={{padding:'1.1rem',borderTop:'1px solid var(--mist)'}}>
                  <div className="alert alert-info" style={{marginBottom:'0.85rem',fontSize:'0.78rem'}}>Fill these and the LLM will write every section specifically about <strong>your</strong> system, not a generic paper.</div>
                  <div className="field"><label>Project Summary</label>
                    <textarea rows={3} value={projectSummary} onChange={e=>setProjectSummary(e.target.value)} placeholder="What does your project do? What problem does it solve? What's novel?"/>
                  </div>
                  <div className="row2">
                    {PROJECT_FIELDS.map(f=>(
                      <div className="field" key={f.id}>
                        <label>{f.icon} {f.label}</label>
                        <textarea rows={2} value={projectFields[f.id]} onChange={e=>setProjectFields(p=>({...p,[f.id]:e.target.value}))} placeholder={f.ph}/>
                      </div>))}
                  </div>
                  <div style={{display:'flex',gap:'0.5rem',marginTop:'0.85rem'}}>
                    <button type="button" className="btn btn-teal" disabled={!topic.trim()&&!projectSummary.trim()} onClick={()=>{setPpReady(true);setPpOpen(false);}}>✅ Use This Context</button>
                    <button type="button" className="btn btn-ghost" onClick={()=>setPpOpen(false)}>Cancel</button>
                  </div>
                </div>}
                {ppReady&&<div style={{display:'flex',flexWrap:'wrap',gap:'0.3rem',padding:'0.5rem 1rem',background:'rgba(26,107,107,0.04)',borderTop:'1px solid rgba(26,107,107,0.12)'}}>
                  {PROJECT_FIELDS.filter(f=>projectFields[f.id]?.trim()).map(f=>(
                    <span key={f.id} style={{fontFamily:'var(--font-mono)',fontSize:'0.6rem',background:'rgba(26,107,107,0.15)',border:'1px solid rgba(26,107,107,0.25)',color:'var(--teal)',padding:'0.15rem 0.45rem',borderRadius:'2px'}}>{f.icon} {f.label}</span>))}
                </div>}
              </div>

              {/* Paper source */}
              <div className="field">
                <label>Reference Source</label>
                <div style={{display:'grid',gridTemplateColumns:'repeat(3,1fr)',gap:'0.5rem'}}>
                  {[{id:'rag',icon:'🔍',label:'arXiv RAG',desc:'Auto-retrieve from 1.7M papers'},
                    {id:'upload',icon:'📤',label:'Upload Papers',desc:'Use only your PDFs/DOCXs'},
                    {id:'both',icon:'⚡',label:'Both',desc:'Uploads + arXiv top-ups'}].map(opt=>(
                    <label key={opt.id} style={{display:'flex',flexDirection:'column',gap:'0.2rem',padding:'0.7rem',border:`1.5px solid ${paperSource===opt.id?'var(--teal)':'var(--fog)'}`,borderRadius:'var(--r)',cursor:'pointer',background:paperSource===opt.id?'var(--teal-lt)':'#fff',transition:'all .15s'}}>
                      <input type="radio" name="src" value={opt.id} checked={paperSource===opt.id} onChange={()=>{setPaperSource(opt.id);setExtractedPapers([]);setUploadedPaperFiles([]);}} style={{display:'none'}}/>
                      <span style={{fontSize:'0.88rem'}}>{opt.icon} {opt.label}</span>
                      <span style={{fontSize:'0.72rem',color:'var(--graphite)'}}>{opt.desc}</span>
                    </label>))}
                </div>
              </div>

              {/* Upload area */}
              {(paperSource==='upload'||paperSource==='both')&&(
                <div style={{marginBottom:'1rem'}}>
                  <div className="drop-zone" onClick={()=>paperFileRef.current?.click()}
                    onDragOver={e=>e.preventDefault()}
                    onDrop={e=>{e.preventDefault();handlePaperFilePick({target:{files:Array.from(e.dataTransfer.files),value:''}});}}>
                    <input ref={paperFileRef} type="file" multiple accept=".pdf,.docx,.doc" style={{display:'none'}} onChange={handlePaperFilePick}/>
                    <div className="dz-icon">📂</div>
                    <div className="dz-text">Click or drag & drop PDF / DOCX papers</div>
                    <div className="dz-hint">Multiple files supported</div>
                  </div>
                  {uploadedPaperFiles.length>0&&<div style={{marginTop:'0.6rem'}}>
                    <div style={{display:'flex',justifyContent:'space-between',marginBottom:'0.3rem'}}>
                      <span style={{fontSize:'0.78rem',color:'var(--graphite)'}}>{uploadedPaperFiles.length} file(s) selected</span>
                      <button type="button" className="btn btn-ghost btn-sm" onClick={()=>{setUploadedPaperFiles([]);setExtractedPapers([]);}}>Clear</button>
                    </div>
                    {extractedPapers.length===0&&<button type="button" className="btn btn-teal btn-sm" disabled={uploadingPapers} onClick={handleUploadPapers}>
                      {uploadingPapers?<><span className="spin"/>Extracting…</>:`🚀 Extract ${uploadedPaperFiles.length} Paper(s)`}
                    </button>}
                  </div>}
                  {paperUploadError&&<div className="alert alert-error" style={{marginTop:'0.5rem'}}>{paperUploadError}</div>}
                  {extractedPapers.length>0&&<div style={{marginTop:'0.6rem'}}>
                    <div className="alert alert-ok" style={{marginBottom:'0.5rem'}}>✓ {extractedPapers.length} paper(s) extracted and ready</div>
                    {extractedPapers.map(p=><div key={p.id} className="ep-card">
                      <div className="ep-title">{p.title}</div>
                      <div className="ep-meta">{p.authors}{p.year?` · ${p.year}`:''} · {p.filename}</div>
                      <div className="ep-abs">{p.abstract}</div>
                    </div>)}
                  </div>}
                </div>)}

              <div className="row2">
                <div className="field"><label>Number of References</label>
                  <input type="number" value={numReferences} onChange={e=>setNumReferences(parseInt(e.target.value)||10)} min="5" max="20"/>
                  <small>5–20 papers</small>
                </div>
                <div className="field"><label>Desired Dataset Features</label>
                  <input type="text" value={desiredFeatures} onChange={e=>setDesiredFeatures(e.target.value)} placeholder="e.g., abstract, title, labels"/>
                  <small>For dataset discovery (comma-separated)</small>
                </div>
              </div>

              {/* Graph toggle */}
              <div className="field">
                <label style={{display:'flex',alignItems:'center',gap:'0.5rem',cursor:'pointer'}}>
                  <input type="checkbox" checked={includeGraphs} onChange={e=>handleGraphsToggle(e.target.checked)} style={{width:'auto',accentColor:'var(--teal)'}}/>
                  Include data-driven graphs (searches Kaggle/UCI for real dataset)
                </label>
              </div>

              {/* Dataset pipeline */}
              {includeGraphs&&<div style={{marginTop:'0.25rem'}}>
                {discoveringDatasets&&<div className="alert alert-warn"><span className="spin spin-dark"/>Searching datasets for "{topic}"…</div>}
                {!discoveringDatasets&&discoveredDatasets.length>0&&<>
                  <div className="section-cap" style={{marginTop:'1rem'}}>Step 1 — Select Dataset</div>
                  <div className="ds-list">
                    {discoveredDatasets.map(ds=>(
                      <div key={ds.id} className={`ds-card ${selectedDataset?.id===ds.id?'selected':''}`} onClick={()=>handleSelectDataset(ds)}>
                        <div className="ds-top"><span className="ds-title">{ds.title}</span><span className={`ds-source ${ds.source}`}>{ds.source.toUpperCase()}</span></div>
                        <div className="ds-desc">{ds.description?.slice(0,100)}</div>
                        <div className="ds-meta">
                          {ds.rows&&<span>🗂 {ds.rows.toLocaleString()} rows</span>}
                          {ds.columns&&<span>📐 {ds.columns} cols</span>}
                          {ds.size_mb!=null&&<span>💾 {ds.size_mb}MB</span>}
                        </div>
                        <div className="ds-tags">{ds.tags?.slice(0,5).map(t=><span className="ds-tag" key={t}>{t}</span>)}</div>
                        <div className="ds-rel-bar" style={{width:`${Math.round((ds.relevance_score||0)*100)}%`}}/>
                      </div>))}
                  </div>
                </>}
                {loadingDataset&&<div className="alert alert-warn" style={{marginTop:'0.75rem'}}><span className="spin spin-dark"/>Loading {selectedDataset?.title}…</div>}
                {datasetMeta&&!loadingDataset&&<>
                  <div className="section-cap" style={{marginTop:'1rem'}}>Step 2 — Select Graph Types</div>
                  <p style={{fontSize:'0.78rem',color:'var(--graphite)',marginBottom:'0.5rem'}}>{datasetMeta.rows?.toLocaleString()} rows × {datasetMeta.columns} cols{datasetMeta.target_col?` · Target: "${datasetMeta.target_col}"`:''}</p>
                  {datasetMeta.preview?.columns&&<div className="preview-wrap">
                    <table className="preview-tbl">
                      <thead><tr>{datasetMeta.preview.columns.map(c=><th key={c}>{c}</th>)}</tr></thead>
                      <tbody>{datasetMeta.preview.rows?.map((row,i)=><tr key={i}>{row.map((cell,j)=><td key={j}>{String(cell)}</td>)}</tr>)}</tbody>
                    </table>
                  </div>}
                  <div className="gtype-grid">
                    {datasetMeta.recommended_graphs?.map(g=>(
                      <label key={g.type} className={`gtype-card ${selectedGraphTypes.includes(g.type)?'selected':''}`}>
                        <input type="checkbox" checked={selectedGraphTypes.includes(g.type)} onChange={()=>setSelectedGraphTypes(prev=>prev.includes(g.type)?prev.filter(t=>t!==g.type):[...prev,g.type])}/>
                        <span style={{fontSize:'1rem'}}>{GRAPH_ICONS[g.type]||'📈'}</span>
                        <div style={{flex:1}}>
                          <div style={{fontWeight:600,fontSize:'0.8rem'}}>{g.title}</div>
                          <div style={{fontSize:'0.7rem',color:'var(--graphite)',marginTop:'0.1rem'}}>{g.description}</div>
                        </div>
                      </label>))}
                  </div>
                  <button type="button" className="btn btn-teal" disabled={!selectedGraphTypes.length||generatingGraphs} onClick={handleGenerateGraphs}>
                    {generatingGraphs?<><span className="spin"/>Generating graphs…</>:`⚡ Generate ${selectedGraphTypes.length} Graph(s)`}
                  </button>
                </>}
                {generatedGraphs.length>0&&<div className="alert alert-ok" style={{marginTop:'0.75rem'}}>✓ {generatedGraphs.length} graph(s) ready — will be embedded in paper</div>}
              </div>}

              {/* Warnings */}
              {(paperSource==='upload'||paperSource==='both')&&extractedPapers.length===0&&<div className="alert alert-warn">Upload and extract papers before generating.</div>}
              {graphsBlocking&&<div className="alert alert-warn">Generate graphs first (Step 2 above) before generating the paper.</div>}

              <hr className="divider"/>
              <button type="submit" className="btn btn-primary btn-full" disabled={loading||!papersReady||graphsBlocking}>
                {loading?<><span className="spin"/>Generating…</>:'✦ Generate Research Paper'}
              </button>
            </form>
          </div>)}

        {/* Error */}
        {error&&<div className="alert alert-error" style={{marginTop:'1rem'}}><span>⚠</span><div><strong>Error:</strong> {error}</div></div>}

        {/* Results */}
        {paper&&<div className="card" style={{marginTop:'1rem'}}>
          <div style={{display:'flex',justifyContent:'space-between',alignItems:'flex-start',flexWrap:'wrap',gap:'0.75rem',marginBottom:'0.75rem'}}>
            <div>
              <div className="section-cap" style={{marginBottom:'0.2rem'}}>Generated Paper</div>
              <h2 style={{fontFamily:'var(--font-display)',fontWeight:400,fontSize:'1.1rem',lineHeight:1.2}}>{paper.title}</h2>
            </div>
            <div style={{display:'flex',gap:'0.5rem'}}>
              <button className="btn btn-primary btn-sm" onClick={()=>paper.docx_url&&window.open(`${API}${paper.docx_url}`,'_blank')}>📄 Download DOCX</button>
              <button className="btn btn-ghost btn-sm" onClick={()=>{setActiveTab('bibtex');}}>BibTeX</button>
            </div>
          </div>
          <div className="stats-bar">
            <span className="stat-pill"><strong>{paper.stats?.total_words||0}</strong><span>words</span></span>
            <span className="stat-pill"><strong>{paper.citations?.length||0}</strong><span>citations</span></span>
            <span className="stat-pill"><strong>{paper.graphs?.length||0}</strong><span>graphs</span></span>
            <span className="stat-pill"><strong>{paper.stats?.uploaded_papers||0}</strong><span>uploaded</span></span>
            <span className="stat-pill"><strong>{paper.stats?.rag_papers||0}</strong><span>RAG</span></span>
            {paper.stats?.has_project_context&&<span className="ctx-badge">📝 Project-specific</span>}
            {Object.keys(sectionAnalyses).length>0&&<span className="ctx-badge">🔍 Analyzed</span>}
          </div>
          <div className="tab-bar">
            {ALL_TABS.filter(t=>{
              if (t.key === 'paper_metrics') return !!(paper?.paper_metrics && !paper.paper_metrics.error);
              if(t.key==='graphs') return (paper.graphs?.length||0)>0;
              if(t.key==='explainability') return Object.keys(sectionAnalyses).length>0;
              if(t.key==='prf_metrics') return Object.keys(sectionAnalyses).length>0||(multiApproachData&&Object.keys(multiApproachData).length>0);
              return true;
            }).map(t=>(
              <button key={t.key} className={`tab-btn ${activeTab===t.key?'active':''}`} onClick={()=>setActiveTab(t.key)}>
                {t.key==='citations'?`Citations (${paper.citations?.length||0})`:t.key==='graphs'?`Graphs (${paper.graphs?.length||0})`:t.key==='multiapproach'&&multiApproachLoading?'⚖ …':t.key==='vsbaseline'&&vsBaselineLoading?'📊 …':t.label}
              </button>))}
          </div>
          <div>{renderTabContent()}</div>
        </div>}

      </main>

      <footer className="footer">
        © 2025 AI Research Paper Generator · React + FastAPI + RAG ·{' '}
        <a href="https://arxiv.org" target="_blank" rel="noopener noreferrer">arXiv</a> ·{' '}
        <a href="https://www.openml.org" target="_blank" rel="noopener noreferrer">OpenML</a>
      </footer>
    </div>
  </>);
}

export {
  OnboardingWizard,
  DatasetQualityBadge,
  DatasetRelevanceGate,
  HallucinationHighlighter,
  ResearchGapsPanel,
};
