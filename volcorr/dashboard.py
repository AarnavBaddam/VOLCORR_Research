"""Builds dashboard/index.html: one self-contained file with the stats output embedded verbatim.
No server, no network, no external scripts."""
import html
import json
import re

from .common import DASH_DIR, DATA_DIR, PACKAGES, RESULTS_DIR, ROOT, load_cfg

OUT = DASH_DIR / "index.html"


def md_to_html(text):
    """Just enough Markdown for DEVIATIONS.md: headings, pipe tables, paragraphs, bold, code."""
    def inline(s):
        s = html.escape(s)
        s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
        return re.sub(r"`(.+?)`", r"<code>\1</code>", s)
    out, rows, para = [], [], []

    def flush():
        if rows:
            head, body = rows[0], rows[2:]
            out.append("<div class='scroll'><table><thead><tr>" + "".join(f"<th>{inline(c)}</th>" for c in head) +
                       "</tr></thead><tbody>" + "".join("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>"
                                                         for r in body) + "</tbody></table></div>")
            rows.clear()
        if para:
            out.append("<p>" + inline(" ".join(para)) + "</p>")
            para.clear()
    for line in text.splitlines():
        if line.startswith("|"):
            if para:
                flush()
            rows.append([c.strip() for c in line.strip().strip("|").split("|")])
        elif line.startswith("#"):
            flush()
            n = len(line) - len(line.lstrip("#"))
            out.append(f"<h{n + 2}>{inline(line.lstrip('# '))}</h{n + 2}>")
        elif not line.strip():
            flush()
        else:
            if rows:
                flush()
            para.append(line.strip())
    flush()
    return "\n".join(out)


def locked_versions():
    lock = (ROOT / "uv.lock").read_text(encoding="utf-8")
    found = dict(re.findall(r'\[\[package\]\]\nname = "([^"]+)"\nversion = "([^"]+)"', lock))
    return {p: found.get(p) for p in PACKAGES}


def main():
    cfg = load_cfg()
    stats_dir = RESULTS_DIR / "stats"

    def load(p):
        return json.loads(p.read_text(encoding="utf-8"))
    dropped = (RESULTS_DIR / "dropped_days.csv").read_text().strip().splitlines()[1:]
    data = {
        "stats": load(stats_dir / "stats.json"),
        "robustness": load(stats_dir / "robustness.json"),
        "manifest": load(DATA_DIR / "raw" / "MANIFEST.json"),
        "data_report": load(DATA_DIR / "clean" / "data_report.json"),
        "revisions": {k: {"repo": v["repo"], "revision": v["revision"]} for k, v in cfg["foundation"]["models"].items()},
        "versions": locked_versions(),
        "dropped_days": dropped,
        "deviations_html": md_to_html((ROOT / "DEVIATIONS.md").read_text(encoding="utf-8")),
    }
    blob = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(TEMPLATE.replace("__DATA__", blob), encoding="utf-8", newline="\n")
    print("wrote", OUT, f"({OUT.stat().st_size / 1024:.0f} KB)")


TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Error correlation: foundation vs classical volatility models</title>
<style>
:root{color-scheme:light;--bg:#f4f3f0;--surface:#fcfcfb;--text:#0b0b0b;--text2:#52514e;--muted:#7a7972;--grid:#e3e2dd;--line:#cfcec8;
--f:#2a78d6;--c:#eb6834;--e:#1baf7a;--good:#0ca30c;--warn:#fab219;--crit:#d03b3b;--shade:#00000014;--hi:#fff7e0}
@media (prefers-color-scheme:dark){:root{color-scheme:dark;--bg:#111110;--surface:#1a1a19;--text:#f0efec;--text2:#c3c2b7;--muted:#8f8e86;--grid:#2e2e2b;--line:#44443f;
--f:#3987e5;--c:#d95926;--e:#199e70;--shade:#ffffff1c;--hi:#3a3320}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
header{padding:20px 24px 0;max-width:1200px;margin:auto}
h1{font-size:20px;margin:0 0 2px}h2{font-size:16px;margin:0 0 4px}h3{font-size:14px;margin:18px 0 6px}h4{font-size:13px;margin:14px 0 4px}
.sub{color:var(--text2);margin:0 0 14px}
nav{display:flex;flex-wrap:wrap;gap:4px;border-bottom:1px solid var(--line)}
nav button{font:inherit;background:none;border:0;border-bottom:3px solid transparent;color:var(--text2);padding:8px 12px;cursor:pointer}
nav button[aria-selected=true]{color:var(--text);border-color:var(--text);font-weight:600}
nav button:focus-visible,.seg button:focus-visible{outline:2px solid var(--f);outline-offset:2px}
main{max-width:1200px;margin:auto;padding:18px 24px 60px}
section{display:none}section.on{display:block}
.card{background:var(--surface);border:1px solid var(--grid);border-radius:8px;padding:16px;margin-bottom:16px}
.grid{display:grid;gap:16px;grid-template-columns:repeat(auto-fit,minmax(260px,1fr))}
.grid2{display:grid;gap:16px;grid-template-columns:repeat(auto-fit,minmax(420px,1fr))}
.hyp .id{font-weight:700;color:var(--text2)}.hyp .est{font-size:30px;font-weight:650;line-height:1.2;margin:6px 0 0}
.hyp p{margin:4px 0;color:var(--text2)}.hyp dl{display:grid;grid-template-columns:auto 1fr;gap:2px 10px;margin:8px 0}
.hyp dt{color:var(--muted)}.hyp dd{margin:0;font-variant-numeric:tabular-nums}
.badge{display:inline-flex;align-items:center;gap:6px;border:1px solid var(--line);border-radius:999px;padding:2px 10px;font-weight:600;font-size:13px}
.badge i{width:10px;height:10px;border-radius:50%;display:inline-block}
.banner{border-left:4px solid var(--warn);background:var(--hi);padding:10px 14px;border-radius:4px;margin-bottom:16px}
.banner.ok{border-color:var(--good);background:var(--surface)}
.tag{font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:.04em;border:1px solid var(--line);color:var(--text2);border-radius:4px;padding:1px 6px;margin-left:8px;vertical-align:middle}
.note{color:var(--text2);font-size:13px;margin:4px 0 10px}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;font-size:13px}
th,td{text-align:left;padding:5px 8px;border-bottom:1px solid var(--grid);vertical-align:top}
th{color:var(--text2);font-weight:600;white-space:nowrap}td.n,th.n{text-align:right}
tr.hi td{background:var(--hi);font-weight:600}tr.grp td{font-weight:700;background:var(--bg)}
.scroll{overflow-x:auto}code{font:12px ui-monospace,Consolas,monospace;word-break:break-all}
svg{display:block;width:100%;height:auto}svg text{fill:var(--text2);font-size:12px}svg .lab{fill:var(--text)}
.legend{display:flex;flex-wrap:wrap;gap:14px;color:var(--text2);font-size:13px;margin:4px 0 8px}
.legend i{display:inline-block;width:12px;height:12px;border-radius:3px;margin-right:6px;vertical-align:-1px}
.seg{display:inline-flex;border:1px solid var(--line);border-radius:6px;overflow:hidden;margin-bottom:10px}
.seg button{font:inherit;background:none;border:0;padding:4px 12px;color:var(--text2);cursor:pointer}
.seg button[aria-pressed=true]{background:var(--text);color:var(--surface)}
#tip{position:fixed;pointer-events:none;background:var(--text);color:var(--surface);padding:6px 9px;border-radius:5px;font-size:12px;max-width:320px;z-index:9;line-height:1.4}
ul.concl{margin:6px 0 0;padding-left:20px}ul.concl li{margin:4px 0}
</style>
</head>
<body>
<header>
<h1>Do foundation models make more correlated volatility forecast errors than classical models?</h1>
<p class="sub" id="sub"></p>
<nav role="tablist" id="nav"></nav>
</header>
<main id="main"></main>
<div id="tip" hidden></div>
<script id="data" type="application/json">__DATA__</script>
<script>
const D=JSON.parse(document.getElementById('data').textContent),S=D.stats,R=D.robustness,M=S.meta;
const NAMES={chronos_bolt_base:'Chronos-Bolt',timesfm_2_0_500m:'TimesFM 2.0',moirai_1_1_r_base:'Moirai 1.1-R',garch:'GARCH(1,1)',gjr:'GJR-GARCH',ewma:'EWMA',har:'log-HAR-RV',ens_foundation:'Foundation ensemble',ens_classical:'Classical ensemble',ens_all:'All-model ensemble'};
const famOf=s=>{const i=M.models.indexOf(s);return i<0?'e':i<3?'f':'c'};
const FAM={f:'Foundation',c:'Classical',e:'Ensemble'};
const col=k=>`var(--${k})`;
const f=(x,d=3)=>x==null?'n/a':(x!==0&&Math.abs(x)<1e-4)?x.toExponential(2):x.toFixed(d);
const pmin=1/M.valid_draws;
const fp=p=>p==null?'n/a':p<pmin?'<'+pmin.toFixed(4):p.toFixed(4);
const ci=o=>o&&o.est!=null?`[${f(o.lo)}, ${f(o.hi)}]`:'n/a';
const esc=s=>String(s).replace(/&/g,'&amp;').replace(/"/g,'&quot;').replace(/</g,'&lt;');
const EXP='<span class="tag">exploratory</span>';
const dark=matchMedia('(prefers-color-scheme:dark)').matches;

/* ---------- chart helpers ---------- */
function scale(d0,d1,r0,r1){return v=>r0+(v-d0)/(d1-d0)*(r1-r0)}
function ticks(a,b,n=5){const st=Math.pow(10,Math.floor(Math.log10((b-a)/n)));const m=[1,2,5,10].find(k=>(b-a)/(st*k)<=n)*st;const out=[];for(let v=Math.ceil(a/m)*m;v<=b+1e-12;v+=m)out.push(+v.toFixed(10));return out}
function dec(a,b){return (b-a)<0.05?3:(b-a)<0.5?2:(b-a)<5?1:0}
// dot-and-interval chart: rows [{label, est, lo, hi, color, tip}], optional group headings {group:'...'}
function intervals(rows,{zero=false,unit='',digits=3}={}){
  const vals=rows.filter(r=>r.est!=null).flatMap(r=>[r.lo??r.est,r.hi??r.est,r.est]);if(zero)vals.push(0);
  let a=Math.min(...vals),b=Math.max(...vals);const pad=(b-a||1)*0.12;a-=pad;b+=pad;
  const W=760,L=190,Rm=170,rh=26,H=rows.length*rh+34,x=scale(a,b,L,W-Rm);let s='';
  for(const t of ticks(a,b))s+=`<line x1="${x(t)}" x2="${x(t)}" y1="6" y2="${H-26}" stroke="var(--grid)"/><text x="${x(t)}" y="${H-10}" text-anchor="middle">${t.toFixed(dec(a,b))}${unit}</text>`;
  if(a<0&&b>0)s+=`<line x1="${x(0)}" x2="${x(0)}" y1="6" y2="${H-26}" stroke="var(--muted)"/>`;
  rows.forEach((r,i)=>{const y=i*rh+20;
    if(r.group){s+=`<text x="0" y="${y+4}" class="lab" font-weight="600">${esc(r.group)}</text>`;return}
    s+=`<text x="${L-12}" y="${y+4}" text-anchor="end" class="lab">${esc(r.label)}</text>`;
    if(r.est==null){s+=`<text x="${L+6}" y="${y+4}">not estimable</text>`;return}
    const tip=esc(`<b>${r.label}</b><br>estimate ${f(r.est,digits)}${r.lo!=null?`<br>95% CI [${f(r.lo,digits)}, ${f(r.hi,digits)}]`:''}${r.tip?'<br>'+r.tip:''}`);
    if(r.lo!=null)s+=`<line x1="${x(r.lo)}" x2="${x(r.hi)}" y1="${y}" y2="${y}" stroke="${r.color}" stroke-width="2" stroke-linecap="round"/>`;
    s+=`<circle cx="${x(r.est)}" cy="${y}" r="5" fill="${r.color}" stroke="var(--surface)" stroke-width="2"/>`;
    s+=`<text x="${W-Rm+10}" y="${y+4}" class="lab">${f(r.est,digits)}${r.lo!=null?` <tspan fill="var(--muted)">[${f(r.lo,digits)}, ${f(r.hi,digits)}]</tspan>`:''}</text>`;
    s+=`<rect x="${L}" y="${y-rh/2}" width="${W-L}" height="${rh}" fill="transparent" data-tip="${tip}"/>`});
  return `<svg viewBox="0 0 ${W} ${H}" role="img">${s}</svg>`}
// horizontal bars from zero: rows [{label, v, color, tip}]
function bars(rows,{digits=3}={}){
  const W=760,L=190,Rm=70,rh=28,H=rows.length*rh+30,b=Math.max(...rows.map(r=>r.v))*1.05,x=scale(0,b,L,W-Rm);let s='';
  for(const t of ticks(0,b))s+=`<line x1="${x(t)}" x2="${x(t)}" y1="4" y2="${H-24}" stroke="var(--grid)"/><text x="${x(t)}" y="${H-8}" text-anchor="middle">${t.toFixed(dec(0,b))}</text>`;
  rows.forEach((r,i)=>{const y=i*rh+8,w=x(r.v)-L;
    s+=`<text x="${L-12}" y="${y+13}" text-anchor="end" class="lab">${esc(r.label)}</text>`;
    s+=`<path d="M${L},${y}h${w-4}a4,4 0 0 1 4,4v10a4,4 0 0 1 -4,4h${-(w-4)}z" fill="${r.color}" data-tip="${esc(`<b>${r.label}</b><br>${r.tip||f(r.v,digits)}`)}"/>`;
    s+=`<text x="${x(r.v)+6}" y="${y+13}" class="lab">${f(r.v,digits)}</text>`});
  return `<svg viewBox="0 0 ${W} ${H}" role="img">${s}</svg>`}
function ramp(t){ // sequential single hue (blue); low recedes toward the surface in either mode
  const lo=dark?[13,54,107]:[205,226,251],hi=dark?[158,197,244]:[13,54,107];t=Math.max(0,Math.min(1,t));
  return `rgb(${lo.map((c,i)=>Math.round(c+(hi[i]-c)*t)).join(',')})`}
function heatmap(m,labels){
  const n=m.length,c=58,L=120,T=96,gap=n===7?8:0,W=L+n*c+gap+50,H=T+n*c+gap+6;let s='';
  const flat=m.flat().filter((v,i)=>i%(n+1)),lo=Math.floor(Math.min(...flat)*10)/10;
  const off=i=>i*c+(n===7&&i>=3?gap:0);
  labels.forEach((l,i)=>{s+=`<text x="${L-8}" y="${T+off(i)+c/2+4}" text-anchor="end" class="lab">${l}</text>`;
    s+=`<text transform="translate(${L+off(i)+c/2+4},${T-8}) rotate(-40)" class="lab">${l}</text>`});
  for(let i=0;i<n;i++)for(let j=0;j<n;j++){const v=m[i][j],t=(v-lo)/(1-lo),x=L+off(j),y=T+off(i);
    const kind=i===j?'':(i<3)===(j<3)?(i<3?'within foundation':'within classical'):'cross-family';
    s+=`<rect x="${x+1}" y="${y+1}" width="${c-2}" height="${c-2}" rx="3" fill="${i===j?'var(--grid)':ramp(t)}" data-tip="${esc(`<b>${labels[i]} × ${labels[j]}</b><br>Spearman ${f(v)}${kind?'<br>'+kind:''}`)}"/>`;
    if(i!==j)s+=`<text x="${x+c/2}" y="${y+c/2+4}" text-anchor="middle" style="fill:${(t>0.5)!==dark?'#fff':'#0b0b0b'};pointer-events:none">${v.toFixed(2)}</text>`}
  return `<svg viewBox="0 0 ${W} ${H}" style="max-width:${W}px" role="img">${s}</svg><p class="note">Colour scale: ${lo.toFixed(1)} (light) to 1.0 (dark${dark?'→bright':''}). The gap separates the foundation block (top left) from the classical block (bottom right); off-diagonal blocks are cross-family.</p>`}
// line chart with shaded bands and crosshair: series [{name,color,values}]
function lines(id,dates,series,{bands,marker,yfmt=2,height=280}={}){
  const W=1100,L=46,Rm=150,T=10,B=28,H=height,all=series.flatMap(s=>s.values).filter(v=>v!=null);
  let a=Math.min(...all),b=Math.max(...all);const pad=(b-a)*0.08;a-=pad;b+=pad;
  const i0=series[0].values.findIndex(v=>v!=null),n=dates.length,x=scale(i0,n-1,L,W-Rm),y=scale(a,b,H-B,T);let s='';
  if(bands){let st=-1;for(let i=i0;i<=n;i++){const on=i<n&&bands[i];if(on&&st<0)st=i;if(!on&&st>=0){s+=`<rect x="${x(st-0.5)}" y="${T}" width="${Math.max(x(i-0.5)-x(st-0.5),1.5)}" height="${H-B-T}" fill="var(--shade)"/>`;st=-1}}}
  for(const t of ticks(a,b,4))s+=`<line x1="${L}" x2="${W-Rm}" y1="${y(t)}" y2="${y(t)}" stroke="var(--grid)"/><text x="${L-6}" y="${y(t)+4}" text-anchor="end">${t.toFixed(yfmt)}</text>`;
  let last='';dates.forEach((d,i)=>{const ym=d.slice(0,7);if(i>=i0&&ym!==last&&(+d.slice(5,7))%3===1){s+=`<text x="${x(i)}" y="${H-8}" text-anchor="middle">${ym}</text>`}last=ym});
  if(marker){const mi=dates.findIndex(d=>d>=marker.date);s+=`<line x1="${x(mi)}" x2="${x(mi)}" y1="${T}" y2="${H-B}" stroke="var(--muted)" stroke-dasharray="4 3"/><text x="${x(mi)+5}" y="${T+11}">${marker.label}</text>`}
  series.forEach(se=>{let d='',pen=false;se.values.forEach((v,i)=>{if(v==null){pen=false;return}d+=(pen?'L':'M')+x(i).toFixed(1)+','+y(v).toFixed(1);pen=true});
    s+=`<path d="${d}" fill="none" stroke="${se.color}" stroke-width="2" stroke-linejoin="round"/>`;
    const lv=se.values[n-1];s+=`<text x="${W-Rm+8}" y="${y(lv)+4}" class="lab">${se.name} ${lv.toFixed(yfmt)}</text>`});
  s+=`<line id="${id}-x" y1="${T}" y2="${H-B}" stroke="var(--muted)" visibility="hidden"/><rect id="${id}-o" x="${L}" y="${T}" width="${W-Rm-L}" height="${H-B-T}" fill="transparent"/>`;
  setTimeout(()=>{const o=document.getElementById(id+'-o'),cx=document.getElementById(id+'-x'),tip=document.getElementById('tip');if(!o)return;
    o.addEventListener('mousemove',e=>{const r=o.getBoundingClientRect(),i=Math.max(i0,Math.min(n-1,Math.round(i0+(e.clientX-r.left)/r.width*(n-1-i0))));
      cx.setAttribute('x1',x(i));cx.setAttribute('x2',x(i));cx.setAttribute('visibility','visible');
      tip.innerHTML=`<b>${dates[i]}</b>${bands?(bands[i]?' · stress day':' · calm day'):''}<br>`+series.map(se=>`${se.name}: ${se.values[i]==null?'n/a':se.values[i].toFixed(yfmt)}`).join('<br>');
      tip.hidden=false;place(e)});
    o.addEventListener('mouseleave',()=>{cx.setAttribute('visibility','hidden');tip.hidden=true})});
  return `<svg viewBox="0 0 ${W} ${H}" role="img">${s}</svg>`}
function place(e){const tip=document.getElementById('tip');tip.style.left=Math.min(e.clientX+14,innerWidth-tip.offsetWidth-8)+'px';tip.style.top=Math.min(e.clientY+14,innerHeight-tip.offsetHeight-8)+'px'}
document.addEventListener('mousemove',e=>{const t=e.target.closest&&e.target.closest('[data-tip]'),tip=document.getElementById('tip');if(!t){if(!e.target.id||!e.target.id.endsWith('-o'))tip.hidden=true;return}tip.innerHTML=t.dataset.tip;tip.hidden=false;place(e)});
const legend=items=>`<div class="legend">${items.map(([c,l])=>`<span><i style="background:${c}"></i>${l}</span>`).join('')}</div>`;
const famLegend=legend([[col('f'),'Foundation models'],[col('c'),'Classical models']]);
function table(head,rows,cls=[]){return `<div class="scroll"><table><thead><tr>${head.map((h,i)=>`<th class="${cls[i]||''}">${h}</th>`).join('')}</tr></thead><tbody>${rows.map(r=>r.grp?`<tr class="grp"><td colspan="${head.length}">${r.grp}</td></tr>`:`<tr class="${r.hi?'hi':''}">${(r.cells||r).map((c,i)=>`<td class="${cls[i]||''}">${c}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`}
const card=(title,body,note='')=>`<div class="card"><h2>${title}</h2>${note?`<p class="note">${note}</p>`:''}${body}</div>`;

/* ---------- pages ---------- */
const C=S.components,H=Object.fromEntries(S.hypotheses.map(h=>[h.id,h]));
const VERD={supported:['var(--good)','✓ Supported'],'not supported':['var(--muted)','✕ Not supported'],inconclusive:['var(--warn)','! Inconclusive']};
const stressMsg=`${M.n_stress} stress days (minimum ${M.min_stress_days}) in ${M.n_episodes} episodes (minimum ${M.min_episodes})`;
const inconc=M.stress_minimum_met?'':' Stress and calm estimates are shown for transparency; the pre-registered stress minimum is not met, so no stress conclusion is drawn.';

function overview(){
  const cards=S.hypotheses.map(h=>{const [c,l]=VERD[h.verdict];
    const mem=h.members?`<p style="margin-top:8px"><b>Member − ensemble QLIKE</b></p>`+Object.entries(h.members).map(([k,v])=>`<p style="font-variant-numeric:tabular-nums">${NAMES[k]}: ${f(v.est,4)} ${ci2(v,4)}, p ${fp(v.p)}</p>`).join(''):'';
    return `<div class="card hyp"><span class="id">${h.id}</span> <span class="badge"><i style="background:${c}"></i>${l}</span>
    <p>${h.hypothesis}</p><div class="est">${f(h.est,h.id==='H4'?4:3)}</div>
    <dl><dt>${h.id==='H4'?'Smallest QLIKE gain':'Estimate'} 95% CI</dt><dd>[${f(h.lo,h.id==='H4'?4:3)}, ${f(h.hi,h.id==='H4'?4:3)}]</dd><dt>p (bootstrap)</dt><dd>${fp(h.p)}</dd><dt>p (Holm-adjusted)</dt><dd>${fp(h.p_holm)}</dd></dl>${mem}
    ${h.note?`<p><b>${h.note}</b></p>`:''}</div>`}).join('');
  return `<div class="banner ${M.stress_minimum_met?'ok':''}"><b>${M.stress_minimum_met?'✓ Stress minimum met':'! Stress minimum not met'}:</b> ${stressMsg}. ${M.stress_minimum_met?'':'H2 and H3 are reported as inconclusive because of insufficient stress. The rule was not changed.'}</div>
  <div class="grid">${cards}</div>
  ${card('Pre-registered conclusion',`<ul class="concl">${S.conclusion.map(c=>`<li>${c}</li>`).join('')}</ul>`,'Read from the decision table fixed in the pre-registration, using the Holm-adjusted results above.')}
  ${card('How these numbers were produced',table(['Item','Value'],[
    ['Primary window (forecast origins)',`${M.window.start} to ${M.window.end}, ${M.n_days} trading days, ${M.n_assets} assets`],
    ['Horizon and target','1-day Garman-Klass variance, recalibrated forecasts'],
    ['Stress days / calm days',`${M.n_stress} / ${M.n_calm}`],
    ['Inference',`Stationary bootstrap, ${M.n_boot.toLocaleString()} resamples, mean block ${M.mean_block}, seed ${M.seed}`],
    ['Draws discarded (fewer than 20 stress days)',`${M.discarded_draws.toLocaleString()} of ${M.n_boot.toLocaleString()}`],
    ['Multiple testing','Holm across H1 to H4 at 5%']]))}`}
const ci2=(o,d=3)=>o&&o.est!=null?`[${f(o.lo,d)}, ${f(o.hi,d)}]`:'';

function corrPage(){
  const reg=[['all','All days'],['stress','Stress days'],['calm','Calm days']];
  const rows=reg.flatMap(([k,l])=>[{group:`${l} (${k==='all'?M.n_days:k==='stress'?M.n_stress:M.n_calm})`},
    {label:'Foundation ρ_F',...C['rho_F_'+k],color:col('f'),tip:'mean of 3 within-family pairs'},{label:'Classical ρ_C',...C['rho_C_'+k],color:col('c'),tip:'mean of 6 within-family pairs'}]);
  const ne=reg.flatMap(([k,l])=>[{group:l},{label:'Foundation (max 3)',...C['neff_F_'+k],color:col('f')},{label:'Classical (max 4)',...C['neff_C_'+k],color:col('c')}]);
  const labs=M.models.map(m=>NAMES[m]);
  return card('Within-family error correlation: foundation vs classical',famLegend+intervals(rows)+
    table(['Statistic','Estimate','95% CI','p (bootstrap)'],[['H1: ρ_F − ρ_C, all days',f(H.H1.est),ci(H.H1),fp(H.H1.p)],['H2: (ρ_F stress − calm) − (ρ_C stress − calm)',f(H.H2.est),ci(H.H2),fp(H.H2.p)]],['','n','n','n']),
    'Spearman correlation of log forecast errors, pairwise within each family, per asset, averaged across pairs and then across the 10 assets. Dots are estimates, bars are 95% bootstrap intervals.'+inconc)+
  card('Pairwise error correlations, all seven models',`<div class="seg" id="hmseg">${reg.map(([k,l],i)=>`<button aria-pressed="${i===0}" data-k="${k}">${l}</button>`).join('')}</div><div id="hm">${heatmap(S.corr.all,labs)}</div>`,
    'Mean across assets of the per-asset Spearman matrix. Hover a cell for the pair.')+
  card('Effective number of independent models'+EXP,famLegend+intervals(ne,{digits:3}),'(Σλ)² / Σλ² of each family\'s error correlation matrix. A value near 1 means the family behaves as one model. The families differ in size, so compare each to its own maximum.'+inconc)}

function ensPage(){
  const q=S.qlike.map(r=>({label:NAMES[r.series],v:r.qlike_all,color:col(famOf(r.series)),tip:`QLIKE all ${f(r.qlike_all,4)}<br>stress ${f(r.qlike_stress,4)}<br>calm ${f(r.qlike_calm,4)}`}));
  const dreg=[['all','All days'],['stress','Stress days'],['calm','Calm days']].map(([k,l])=>({label:l,est:C['D_'+k].est*100,lo:C['D_'+k].lo*100,hi:C['D_'+k].hi*100,color:col('e')}));
  const mem=Object.entries(H.H4.members),dmh=S.dm.h4;
  return card('QLIKE by model and ensemble',legend([[col('f'),'Foundation models'],[col('c'),'Classical models'],[col('e'),'Equal-weight ensembles']])+bars(q)+
    table(['Series','Family','QLIKE all','QLIKE stress','QLIKE calm','MSE all'],S.qlike.map(r=>[NAMES[r.series],FAM[famOf(r.series)],f(r.qlike_all,4),f(r.qlike_stress,4),f(r.qlike_calm,4),r.mse_all.toExponential(3)]),['','','n','n','n','n']),
    'Mean daily loss on the primary window; daily loss is the equal-weight mean across the 10 assets. Lower is better. MSE is secondary and shown for comparison only.')+
  card('H4: foundation ensemble vs each member',table(['Member','QLIKE member − ensemble','95% CI','p (bootstrap)','DM statistic (NW, bandwidth 5)','DM p'],
    mem.map(([k,v])=>[NAMES[k],f(v.est,4),ci2(v,4),fp(v.p),f(dmh[k].stat,2),f(dmh[k].p,4)]),['','n','n','n','n','n']),
    `Positive means the ensemble is better. H4 needs all three positive; its p-value is the largest of the three (${fp(H.H4.p)}, Holm-adjusted ${fp(H.H4.p_holm)}). The Diebold-Mariano test is the pre-registered check.`)+
  card('Diversification benefit D by regime',intervals(dreg,{zero:true,unit:'%',digits:2})+
    table(['Statistic','Estimate','95% CI','p (bootstrap)'],[['H3: D(calm) − D(stress)',f(H.H3.est,4),ci2(H.H3,4),fp(H.H3.p)]],['','n','n','n']),
    'D = 1 − QLIKE(foundation ensemble) / mean QLIKE of its three members, in percent.'+inconc)+
  `<div class="grid2">${card('Model Confidence Set, 10% level'+EXP,table(['Series','In the set','MCS p-value'],M.series.map(s=>[NAMES[s],S.mcs.included.includes(s)?'✓ included':'– excluded',f(S.mcs.pvalues[s],4)]),['','','n']),'Hansen, Lunde and Nason (2011) on daily QLIKE over the seven models and three ensembles.')}
  ${card('Ensembles vs log-HAR-RV'+EXP,table(['Ensemble','QLIKE ensemble − HAR-RV','DM statistic','DM p'],Object.entries(S.dm.vs_har).map(([k,v])=>[NAMES[k],f(v.mean_diff,4),f(v.stat,2),f(v.p,4)]),['','n','n','n']),'Negative means the ensemble beats HAR-RV, the baseline to beat.')}</div>`}

function tsPage(){
  const r=S.rolling,bands=r.stress.map(Boolean);
  return card(`Rolling ${r.window}-day within-family error correlation`+EXP,legend([[col('f'),'Foundation ρ_F'],[col('c'),'Classical ρ_C'],['var(--shade)','Stress day (primary rule)']])+
    lines('roll',r.dates,[{name:'Foundation',color:col('f'),values:r.rho_F},{name:'Classical',color:col('c'),values:r.rho_C}],{bands,marker:{date:r.primary_start,label:'primary window starts'}}),
    `Trailing ${r.window} trading days, averaged across assets. Descriptive only: the window length is not pre-registered. Dates before ${r.primary_start} are in the secondary window (possibly contaminated).`)+
  card('VIX close and stress flags',legend([['var(--text2)','VIX close'],['var(--shade)','Stress day: VIX above its trailing 252-day 80th percentile']])+
    lines('vix',r.dates,[{name:'VIX',color:'var(--text2)',values:r.vix.map((v,i)=>r.rho_F[i]==null?null:v)}],{bands,marker:{date:r.primary_start,label:'primary window starts'},yfmt:1,height:220}),
    `Primary window: ${stressMsg}.`)}

function hcell(o,d=3){return !o||o.est==null?'–':`${f(o.est,d)} <span style="color:var(--muted)">${ci2(o,d)}</span><br><span style="color:var(--muted)">p ${fp(o.p)}</span>`}
function robPage(){
  const ba=S.by_asset,rows=[];let last='';
  for(const r of R.rows){if(r.check!==last&&r.check!=='Primary result'){rows.push({grp:r.check});last=r.check}
    rows.push({hi:r.check==='Primary result',cells:[(r.check==='Primary result'?'Primary result: ':'')+r.variant+(r.note?`<br><span style="color:var(--muted)">${r.note}</span>`:'')+(r.n_stress!=null?`<br><span style="color:var(--muted)">${r.n_stress} stress days, ${r.n_episodes} episodes</span>`:''),hcell(r.H1),hcell(r.H2),hcell(r.H3,4),hcell(r.H4,4)]})}
  return card('By asset: within-family error correlation'+EXP,famLegend.replace('Foundation models','ρ_F − ρ_C (positive: foundation more correlated)').replace(/<span><i style="background:var\(--c\)".*?<\/span>/,'')+
    intervals(ba.map(b=>({label:`${b.asset} · ${b.asset_class}`,...b.diff,color:col('f')})),{zero:true})+
    table(['Asset','Class','ρ_F','ρ_C','ρ_F − ρ_C','95% CI',...M.series.map(s=>'QLIKE '+NAMES[s])],ba.map(b=>[b.asset,b.asset_class,f(b.rho_F),f(b.rho_C),f(b.diff.est),ci(b.diff),...M.series.map(s=>f(b.qlike[s],3))]),['','','n','n','n','n',...M.series.map(()=>'n')]),
    'Per-asset version of the H1 statistic, all days of the primary window.')+
  card('Robustness checks vs the primary result'+EXP,table(['Check','H1: ρ_F − ρ_C','H2: stress gap','H3: D(calm) − D(stress)','H4: smallest member − ensemble'],rows),
    'Each cell: estimate, 95% bootstrap interval, unadjusted two-sided p. A dash means the check does not change that statistic. Robustness checks are exploratory and cannot overturn a confirmatory result. H4 shows the smallest of the three member differences and the largest p.'+inconc)}

function secPage(){
  const s=S.secondary,p=S.primary_summary,ks=Object.keys(s.qlike_ratio_vs_har);
  return `<div class="banner"><b>! Possibly contaminated, diagnostic only.</b> Secondary-window dates may be inside the foundation models' pretraining data. Nothing on this page is confirmatory.</div>`+
  card('Memorization diagnostic: foundation / HAR-RV QLIKE ratio'+EXP,table(['Model','Secondary window (possibly contaminated)','Primary window (clean)','Secondary − primary'],ks.map(k=>[NAMES[k],f(s.qlike_ratio_vs_har[k]),f(p.qlike_ratio_vs_har[k]),f(s.qlike_ratio_vs_har[k]-p.qlike_ratio_vs_har[k])]),['','n','n','n']),
    'A ratio below 1 means the model beats log-HAR-RV. A clearly lower ratio in the secondary window than in the clean window would be a sign of memorization.')+
  card('Error correlation in both windows'+EXP,table(['Window','Origins','Days','ρ_F','ρ_C','ρ_F − ρ_C'],[['Secondary (possibly contaminated)',`${s.first} to ${s.last}`,s.n_days,f(s.rho_F),f(s.rho_C),f(s.rho_F-s.rho_C)],['Primary (clean)',`${p.first} to ${p.last}`,p.n_days,f(p.rho_F),f(p.rho_C),f(p.rho_F-p.rho_C)]],['','','n','n','n','n'])+
    `<h3>Secondary-window pairwise correlations</h3>`+heatmap(s.corr,M.models.map(m=>NAMES[m])),
    `Pre-registered secondary window: ${s.requested.start} to ${s.requested.end}. Recalibrated forecasts exist only from ${s.first}: the sample starts in 2015 and the classical models and the recalibration each need 1,000 prior days (see the deviation log, item C1).`)+
  card('Crisis windows'+EXP,table(['Window','Days available','ρ_F','ρ_C'],S.crisis.map(c=>[`${c.start} to ${c.end}`,c.n_days,c.rho_F==null?'not available':f(c.rho_F),c.rho_C==null?'not available':f(c.rho_C)]),['','n','n','n']),
    'Secondary-window origins only; possibly contaminated. Windows before the first recalibrated origin are not available. Very short windows give noisy correlations.')}

function intPage(){
  const dr=D.data_report,mf=D.manifest.files;
  return card('Raw data manifest (SHA-256)',table(['File','Ticker','Rows','First','Last','Downloaded (UTC)','SHA-256'],Object.entries(mf).map(([k,v])=>[k,v.ticker,v.rows,v.first,v.last,v.downloaded_utc,`<code>${v.sha256}</code>`]),['','','n']),'Source: yfinance '+Object.values(mf)[0].yfinance+', auto_adjust off. Raw files are never overwritten; every run re-verifies these hashes.')+
  card('Foundation model checkpoints',table(['Model','Hugging Face repository','Pinned revision'],Object.entries(D.revisions).map(([k,v])=>[NAMES[k],v.repo,`<code>${v.revision}</code>`])))+
  `<div class="grid2">${card('Floored and dropped observations',table(['Asset','Bars','GK values floored at 1e-8'],Object.keys(dr.panel_rows_per_asset).map(a=>[a,dr.panel_rows_per_asset[a],dr.floored_per_asset[a]]),['','n','n'])+
    `<p class="note">Bars dropped as missing or invalid: ${dr.dropped_bars.length?dr.dropped_bars.map(d=>`${d.asset} ${d.reason} ${d.n}`).join(', '):'0'}. VIX days filled forward: ${dr.vix_days_filled_forward}. Asset-days dropped because a model failed twice: ${D.dropped_days.length}.</p>`)}
  ${card('Bootstrap and software',table(['Item','Value'],[['Resamples',M.n_boot.toLocaleString()],['Draws discarded (fewer than 20 stress days)',M.discarded_draws.toLocaleString()],['Valid draws',M.valid_draws.toLocaleString()],['Seed',M.seed],...Object.entries(D.versions).map(([k,v])=>[k,v])]),'Package versions from uv.lock.')}</div>`+
  card('Deviation and interpretation log',D.deviations_html)}

const PAGES=[['Overview',overview],['Foundation vs classical',corrPage],['Ensembling',ensPage],['Time series',tsPage],['Per asset and robustness',robPage],['Secondary window',secPage],['Integrity',intPage]];
document.getElementById('sub').textContent=`Pre-registration v1.0 (frozen 10 Nov 2026) · primary window ${M.window.start} to ${M.window.end} · ${M.n_days} days · ${M.n_assets} ETFs`;
const nav=document.getElementById('nav'),main=document.getElementById('main');
PAGES.forEach(([name,fn],i)=>{const b=document.createElement('button');b.textContent=`${i+1}. ${name}`;b.setAttribute('role','tab');nav.appendChild(b);
  const s=document.createElement('section');s.innerHTML=fn();main.appendChild(s);
  b.onclick=()=>{[...nav.children].forEach((x,j)=>{x.setAttribute('aria-selected',j===i);main.children[j].classList.toggle('on',j===i)});location.hash=i+1}});
nav.children[Math.max(0,Math.min(PAGES.length,+location.hash.slice(1)||1)-1)].click();
document.getElementById('hmseg').addEventListener('click',e=>{const k=e.target.dataset.k;if(!k)return;[...e.currentTarget.children].forEach(b=>b.setAttribute('aria-pressed',b===e.target));document.getElementById('hm').innerHTML=heatmap(S.corr[k],M.models.map(m=>NAMES[m]))});
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()
