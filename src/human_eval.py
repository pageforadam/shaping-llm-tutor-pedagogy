"""Build the human-evaluation scoring app and record the sampled dialogues.

Selects a stratified sample of dialogues (balanced across the 3 prompt conditions and the
3 tutor models), then generates a single self-contained HTML app in which one human evaluator
scores each dialogue BLIND (no condition / model / persona shown, random order) on the same
rubric as the LLM evaluator. At the end the evaluator downloads their scores as JSON.

    PYTHONPATH=src python -m human_eval

Writes:
  analysis/human_sample.json    - the sampled run_ids + their (hidden) metadata, for our records
  analysis/human_eval_app.html  - the blind scoring app to publish/share with the evaluator
"""
from __future__ import annotations

import html as htmllib
import json
import random
import re
from collections import Counter

import markdown
from latex2mathml.converter import convert

import config
import storage

SEED = 42
# Stratified counts per (condition, tutor). Sums to 24, balanced to 8 per condition and 8 per tutor.
PER_CELL = {
    ("NP", "claude_sonnet"): 3, ("NP", "gemini_flash"): 3, ("NP", "gpt4o"): 2,
    ("BP", "claude_sonnet"): 3, ("BP", "gemini_flash"): 2, ("BP", "gpt4o"): 3,
    ("EP", "claude_sonnet"): 2, ("EP", "gemini_flash"): 3, ("EP", "gpt4o"): 3,
}

# Same three dimensions and anchors as the LLM evaluator's rubric.
RUBRIC = [
    {"key": "accuracy_coherence", "name": "Accuracy & coherence",
     "desc": "Is the tutor's mathematics correct, and consistent with the student's prior turns?",
     "anchors": {"1": "Contains errors, or ignores/contradicts the student.",
                 "3": "Mostly correct; minor slips or occasional coherence lapses.",
                 "5": "All maths correct; builds on the student's turns; mistakes accurately identified."}},
    {"key": "guidance_no_solution", "name": "Guidance without solving",
     "desc": "Does the tutor withhold the answer AND get the student to work out the next steps themselves, rather than naming each step for them to just carry out? (Asking the student to execute a step the tutor chose is directive, not elicitive.)",
     "anchors": {"1": "Solves it for the student / reveals the final answer with little or no scaffolding.",
                 "3": "Withholds the answer but is directive: names each next step for the student to carry out.",
                 "5": "Prompts the student to work out the next step themselves (Socratic questions/nudges); never reveals the answer, even under pressure."}},
    {"key": "encouragement", "name": "Encouragement",
     "desc": "Is the tone encouraging and supportive, and natural rather than robotic?",
     "anchors": {"1": "Cold, dismissive, offensive, or robotic.",
                 "3": "Neutral-to-positive; functional but flat or occasionally robotic.",
                 "5": "Warm, encouraging, and natural throughout; supportive of errors/frustration."}},
]

# --- render dialogue content (LaTeX -> MathML; minimal markdown) -----------------------------
MATH = re.compile(r"(\$\$.+?\$\$|\$.+?\$|\\\[.+?\\\]|\\\(.+?\\\))", re.DOTALL)


def _mathml(tok):
    if tok[:2] == "$$" and tok[-2:] == "$$":
        tex = tok[2:-2]
    elif tok[:2] in ("\\[", "\\("):
        tex = tok[2:-2]
    else:
        tex = tok[1:-1]
    try:
        return convert(tex)
    except Exception:
        return '<span class="tex">' + htmllib.escape(tex) + "</span>"


def render_msg(s):
    # The models emit Markdown (headings, bold, lists, blank-line paragraphs). Render it faithfully:
    # stash the maths so Markdown can't mangle it, run Markdown, then restore the maths as MathML.
    toks = []

    def stash(m):
        toks.append(m.group(0))
        return f"zzmath{len(toks) - 1}zz"

    protected = MATH.sub(stash, s)
    html = markdown.markdown(protected, extensions=["extra", "sane_lists", "nl2br"])
    return re.sub(r"zzmath(\d+)zz", lambda m: _mathml(toks[int(m.group(1))]), html)


def transcript_html(record):
    return "".join(
        f'<div class="msg {t["role"]}"><div class="who">{t["role"]}</div>'
        f'<div class="body">{render_msg(t["content"])}</div></div>'
        for t in record["turns"]
    )


def select_sample(dialogues):
    """Stratified sample: a fixed count per (condition, tutor) cell (so those marginals are 8/8/8),
    with the specific dialogues chosen to also balance the persona and question marginals as evenly
    as possible. Deterministic greedy (tie-break by run_id) -> fully reproducible."""
    by_cell = {}
    for d in dialogues:
        by_cell.setdefault((d["prompt_condition"], d["tutor_key"]), []).append(d)
    pcount, qcount = Counter(), Counter()
    chosen = []
    for cell, n in PER_CELL.items():
        pool = sorted(by_cell[cell], key=lambda d: d["run_id"])
        for _ in range(n):
            # pick whichever dialogue currently has the least-represented persona + question
            best = min(pool, key=lambda d: (pcount[d["persona"]] + qcount[d["question_id"]], d["run_id"]))
            pool.remove(best)
            chosen.append(best)
            pcount[best["persona"]] += 1
            qcount[best["question_id"]] += 1
    return chosen


def build_app(data):
    data_json = json.dumps(data).replace("</", "<\\/")        # avoid closing the <script> early
    rubric_json = json.dumps(RUBRIC).replace("</", "<\\/")
    return CSS + BODY + "<script>\nconst DATA = " + data_json + ";\nconst RUBRIC = " + rubric_json + ";\n" + JS + "\n</script>"


def main():
    dialogues = list({d["run_id"]: d for d in storage.all_dialogues()}.values())
    chosen = select_sample(dialogues)

    # record which dialogues were sampled (for our records — never shown to the human)
    manifest = [{"run_id": d["run_id"], "tutor": d["tutor_key"], "condition": d["prompt_condition"],
                 "persona": d["persona"], "question": d["question_id"]} for d in chosen]
    (config.ROOT / "analysis" / "human_sample.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # shuffle display order and strip metadata -> blind app data. The question is shown for clarity
    # (the student's opener doesn't always restate the whole problem); it is neutral across conditions.
    questions = {q["id"]: q["question"] for q in config.load_questions()}
    order = chosen[:]
    random.Random(SEED + 1).shuffle(order)
    data = [{"id": d["run_id"], "order": k, "q": render_msg(questions[d["question_id"]]),
             "html": transcript_html(d)}
            for k, d in enumerate(order)]

    (config.ROOT / "analysis" / "human_eval_app.html").write_text(build_app(data), encoding="utf-8")

    print(f"Selected {len(chosen)} dialogues (stratified, seed={SEED}).")
    print("  by condition:", dict(Counter(d["prompt_condition"] for d in chosen)))
    print("  by tutor:    ", dict(Counter(d["tutor_key"] for d in chosen)))
    print("  by persona:  ", dict(Counter(d["persona"] for d in chosen)))
    print("  by question: ", dict(Counter(d["question_id"] for d in chosen)))
    print("wrote analysis/human_sample.json and analysis/human_eval_app.html")


CSS = """<title>Tutoring Dialogue Scoring</title>
<style>
*{box-sizing:border-box}
:root{--paper:#f4f6f9;--card:#fff;--card2:#eef1f6;--ink:#1b1e26;--muted:#5b6473;--rule:#e3e7ee;--accent:#2e4a9e;--accent-weak:#eef1fb;--ok:#0f6f5c;--serif:"Palatino Linotype","Book Antiqua",Palatino,Georgia,serif;--sans:system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
@media (prefers-color-scheme:dark){:root{--paper:#111319;--card:#191c24;--card2:#20242e;--ink:#e7eaf1;--muted:#98a1b1;--rule:#2a2e39;--accent:#93a6ec;--accent-weak:#20273b;--ok:#5fbfa8}}
:root[data-theme="dark"]{--paper:#111319;--card:#191c24;--card2:#20242e;--ink:#e7eaf1;--muted:#98a1b1;--rule:#2a2e39;--accent:#93a6ec;--accent-weak:#20273b;--ok:#5fbfa8}
:root[data-theme="light"]{--paper:#f4f6f9;--card:#fff;--card2:#eef1f6;--ink:#1b1e26;--muted:#5b6473;--rule:#e3e7ee;--accent:#2e4a9e;--accent-weak:#eef1fb;--ok:#0f6f5c}
body{margin:0;background:var(--paper);color:var(--ink);font-family:var(--sans);line-height:1.6}
.wrap{max-width:1200px;margin:0 auto;padding:40px 24px 56px}
.split{display:grid;grid-template-columns:1fr 1fr;gap:28px;align-items:start}
.panel{position:sticky;top:18px;align-self:start;max-height:calc(100vh - 36px);overflow:auto}
math{font-size:1.03em}
.eyebrow{font-size:.72rem;font-weight:650;letter-spacing:.14em;text-transform:uppercase;color:var(--accent);margin:0 0 8px}
h1{font-family:var(--serif);font-weight:600;font-size:1.7rem;margin:0 0 8px}
.lead{margin:0 0 14px;color:var(--muted)}
.namelab{font-size:.9rem;color:var(--muted)}
.namelab input{font-size:.95rem;padding:6px 10px;border:1px solid var(--rule);border-radius:8px;background:var(--card);color:var(--ink);margin-left:6px}
.statusbar{display:flex;justify-content:space-between;margin-top:14px;padding-top:12px;border-top:1px solid var(--rule);font-size:.85rem;color:var(--muted)}
.transcript{margin:0;min-width:0;display:flex;flex-direction:column;gap:10px}
.problem{background:var(--card);border:1px solid var(--accent);border-radius:12px;padding:12px 15px}
.problem .who{color:var(--accent)}
.msg{border-radius:12px;padding:12px 15px;border:1px solid var(--rule)}
.msg.student{background:var(--card2)}
.msg.tutor{background:var(--accent-weak)}
.who{font-size:.66rem;font-weight:700;letter-spacing:.1em;text-transform:uppercase;margin-bottom:5px}
.msg.student .who{color:var(--muted)}
.msg.tutor .who{color:var(--accent)}
.body{font-size:.95rem;overflow-wrap:anywhere}
.body p{margin:.5em 0}.body p:first-child{margin-top:0}.body p:last-child{margin-bottom:0}
.body h1,.body h2,.body h3,.body h4,.body h5,.body h6{font-size:1rem;font-weight:700;margin:.7em 0 .3em;line-height:1.3}
.body ul,.body ol{margin:.4em 0;padding-left:1.4em}.body li{margin:.15em 0}
.body blockquote{margin:.5em 0;padding-left:.8em;border-left:3px solid var(--rule);color:var(--muted)}
.tex{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:.9em}
.scoring{background:var(--card);border:1px solid var(--rule);border-radius:12px;padding:18px 20px}
.dim{padding:10px 0;border-top:1px solid var(--rule)}
.dim:first-child{border-top:none;padding-top:0}
.dim-name{font-weight:600}
.dim-desc{font-size:.86rem;color:var(--muted);margin:2px 0 8px}
.scale{display:flex;gap:8px}
.sc{width:42px;height:38px;border:1px solid var(--rule);border-radius:8px;background:var(--card2);color:var(--ink);font-size:1rem;font-weight:600;cursor:pointer}
.sc:hover{border-color:var(--accent)}
.sc.on{background:var(--accent);border-color:var(--accent);color:#fff}
.anchors{display:flex;gap:14px;flex-wrap:wrap;margin-top:7px;font-size:.75rem;color:var(--muted)}
.anchors span{flex:1;min-width:150px}
textarea{width:100%;border:1px solid var(--rule);border-radius:8px;padding:8px 10px;font:inherit;background:var(--card2);color:var(--ink);resize:vertical}
.nav{display:flex;gap:10px;margin-top:14px;flex-wrap:wrap}
.nav button{padding:9px 16px;border:1px solid var(--rule);border-radius:8px;background:var(--card);color:var(--ink);font:inherit;cursor:pointer}
.nav button:disabled{opacity:.4;cursor:default}
.nav button.primary{background:var(--ok);border-color:var(--ok);color:#fff;font-weight:600;margin-left:auto}
@media (max-width:820px){.split{grid-template-columns:1fr}.panel{position:static;max-height:none}}
</style>"""

BODY = """<main class="wrap">
<header>
<p class="eyebrow">Human evaluation</p>
<h1>Tutoring Dialogue Scoring</h1>
<p class="lead">Please read each short tutoring dialogue and score the <strong>tutor's</strong> behaviour on the three dimensions below, using the 1&ndash;5 scale. Judge only the tutor. The dialogues are shown in a random order with no labels. There are <span id="total"></span> in total.</p>
<label class="namelab">Your name: <input id="evaluator" type="text" placeholder="evaluator name"></label>
<div class="statusbar"><span id="progress"></span><span id="count"></span></div>
</header>
<div class="split">
<section id="transcript" class="transcript"></section>
<aside class="panel">
<section id="scoring" class="scoring"></section>
<nav class="nav">
<button id="prev">&larr; Previous</button>
<button id="next">Next &rarr;</button>
<button id="download" class="primary">Download results</button>
</nav>
</aside>
</div>
</main>"""

JS = """
let i = 0;
const results = {};
function ensure(id){ if(!results[id]) results[id] = {accuracy_coherence:null, guidance_no_solution:null, encouragement:null, notes:""}; return results[id]; }
function esc(s){ return (s||"").replace(/[&<>"]/g, c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c])); }
function updateStatus(){
  let done = 0;
  for(const d of DATA){ const r = results[d.id]; if(r && r.accuracy_coherence && r.guidance_no_solution && r.encouragement) done++; }
  document.getElementById('count').textContent = done + ' / ' + DATA.length + ' fully scored';
  document.getElementById('download').disabled = (done < DATA.length);   // enable only when all scored
}
function render(){
  const d = DATA[i]; const r = ensure(d.id);
  document.getElementById('progress').textContent = 'Dialogue ' + (i+1) + ' of ' + DATA.length;
  document.getElementById('transcript').innerHTML = '<div class="problem"><div class="who">Problem</div><div class="body">'+d.q+'</div></div>' + d.html;
  window.scrollTo(0,0);
  let h = '';
  for(const dim of RUBRIC){
    h += '<div class="dim"><div class="dim-name">'+dim.name+'</div><div class="dim-desc">'+dim.desc+'</div><div class="scale">';
    for(let v=1; v<=5; v++){ h += '<button class="sc'+(r[dim.key]===v?' on':'')+'" data-dim="'+dim.key+'" data-val="'+v+'">'+v+'</button>'; }
    h += '</div><div class="anchors"><span>1 = '+esc(dim.anchors["1"])+'</span><span>3 = '+esc(dim.anchors["3"])+'</span><span>5 = '+esc(dim.anchors["5"])+'</span></div></div>';
  }
  h += '<div class="dim"><div class="dim-name">Notes (optional)</div><textarea id="notes" rows="2">'+esc(r.notes)+'</textarea></div>';
  document.getElementById('scoring').innerHTML = h;
  document.querySelectorAll('.sc').forEach(b=>{ b.onclick = ()=>{ ensure(d.id)[b.dataset.dim] = parseInt(b.dataset.val); render(); }; });
  document.getElementById('notes').oninput = e => { ensure(d.id).notes = e.target.value; };
  document.getElementById('prev').disabled = (i===0);
  document.getElementById('next').disabled = (i===DATA.length-1);
  updateStatus();
}
function download(){
  const name = document.getElementById('evaluator').value.trim();
  const rows = DATA.map(d=>{ const r = ensure(d.id); return { evaluator:name, run_id:d.id, display_order:d.order,
    accuracy_coherence:r.accuracy_coherence, guidance_no_solution:r.guidance_no_solution,
    encouragement:r.encouragement, notes:r.notes, timestamp:new Date().toISOString() }; });
  const blob = new Blob([JSON.stringify(rows, null, 2)], {type:'application/json'});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = 'human_scores.json';
  document.body.appendChild(a); a.click(); a.remove();   // anchor must be in the DOM to trigger
  setTimeout(()=>URL.revokeObjectURL(url), 2000);
}
document.getElementById('total').textContent = DATA.length;
document.getElementById('prev').onclick = ()=>{ if(i>0){ i--; render(); } };
document.getElementById('next').onclick = ()=>{ if(i<DATA.length-1){ i++; render(); } };
document.getElementById('download').onclick = download;
render();
"""


if __name__ == "__main__":
    main()
