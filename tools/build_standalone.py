"""발표자료를 단일 HTML 파일로 번들(업로드용). 슬라이드 11장 인라인 +
이미지/영상 base64 임베드 + style.css 인라인 + 키보드/버튼 네비.
사용: python -m tools.build_standalone
출력: ~/Downloads/RoadVision_발표.html
"""
from __future__ import annotations
import base64, os, re, mimetypes

ROOT = os.path.abspath(".")
SLIDES = "presentation/slides"
ASSETS_BASE = "presentation"  # ../assets/ → presentation/assets/

def data_uri(relpath_from_presentation: str) -> str:
    # relpath like 'assets/clips/demo_ldw.mp4'
    full = os.path.join(ASSETS_BASE, relpath_from_presentation)
    mime = mimetypes.guess_type(full)[0] or "application/octet-stream"
    with open(full, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{b64}"

def main():
    css = open("presentation/style.css", encoding="utf-8").read()

    slides_html = []
    for i in range(1, 12):
        p = f"{SLIDES}/slide-{i:02d}.html"
        if not os.path.exists(p):
            continue
        html = open(p, encoding="utf-8").read()
        # <body>...</body> 추출
        body = re.search(r"<body>(.*?)</body>", html, re.S).group(1)
        # NOTES 주석 제거
        body = re.sub(r"<!--\s*NOTES:.*?-->", "", body, flags=re.S)
        # 자산 참조 → data URI (../assets/... )
        for m in set(re.findall(r"\.\./(assets/[A-Za-z0-9_/]+\.(?:png|mp4|jpg|jpeg))", body)):
            body = body.replace(f"../{m}", data_uri(m))
        slides_html.append(f'<div class="deckslide" data-idx="{i}">{body.strip()}</div>')

    n = len(slides_html)
    deck = "\n".join(slides_html)

    out = f"""<!DOCTYPE html>
<html lang="ko"><head><meta charset="UTF-8">
<title>RoadVision — 발표자료</title>
<style>
{css}
*,*::before,*::after{{box-sizing:border-box;}}
html,body{{margin:0;height:100%;background:#111;overflow:hidden;font-family:system-ui,sans-serif;}}
#stage{{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%) scale(var(--scale,1));
  transform-origin:center center;width:1280px;height:720px;overflow:hidden;background:#fff;}}
#deck>.deckslide{{display:none;width:1280px;height:720px;}}
#deck>.deckslide.active{{display:block;}}
#deck .slide{{box-shadow:none;}}
#controls{{position:fixed;bottom:18px;left:50%;transform:translateX(-50%);display:flex;align-items:center;
  gap:16px;background:rgba(0,0,0,.55);border-radius:32px;padding:8px 20px;color:#fff;font-size:13px;
  user-select:none;z-index:100;transition:opacity .35s;}}
#controls.idle{{opacity:0;pointer-events:none;}}
#controls button{{background:rgba(255,255,255,.15);border:1px solid rgba(255,255,255,.3);color:#fff;
  border-radius:6px;padding:5px 14px;font-size:14px;cursor:pointer;}}
#controls button:disabled{{opacity:.35;}}
#ind{{min-width:64px;text-align:center;font-weight:700;}}
</style></head>
<body>
<div id="stage"><div id="deck">
{deck}
</div></div>
<div id="controls">
  <button id="prev">← 이전</button><span id="ind">1 / {n}</span><button id="next">다음 →</button>
  <span style="opacity:.5;font-size:11px;margin-left:8px;">← / → 키 · F 전체화면</span>
</div>
<script>
const N={n}; let cur=1;
const slides=[...document.querySelectorAll('.deckslide')];
const ind=document.getElementById('ind');
const stage=document.getElementById('stage');
function show(k){{ if(k<1||k>N) return; cur=k;
  slides.forEach(s=>s.classList.toggle('active', +s.dataset.idx===cur));
  ind.textContent=cur+' / '+N;
  document.getElementById('prev').disabled=cur===1;
  document.getElementById('next').disabled=cur===N;
  // 활성 슬라이드 영상만 재생
  document.querySelectorAll('video').forEach(v=>{{ try{{v.pause();}}catch(e){{}} }});
  slides[cur-1].querySelectorAll('video').forEach(v=>{{ v.currentTime=0; v.play().catch(()=>{{}}); }});
}}
document.getElementById('prev').onclick=()=>show(cur-1);
document.getElementById('next').onclick=()=>show(cur+1);
window.addEventListener('keydown',e=>{{
  if(e.key==='ArrowRight'||e.key===' '){{e.preventDefault();show(cur+1);}}
  if(e.key==='ArrowLeft'){{e.preventDefault();show(cur-1);}}
  if(e.key==='f'||e.key==='F'){{ if(!document.fullscreenElement)document.documentElement.requestFullscreen().catch(()=>{{}}); else document.exitFullscreen().catch(()=>{{}}); }}
}});
function rescale(){{ stage.style.setProperty('--scale', Math.min(innerWidth/1280, innerHeight/720)); }}
addEventListener('resize',rescale); rescale();
// 컨트롤 자동숨김
const ctr=document.getElementById('controls'); let t=null;
function showCtr(){{ ctr.classList.remove('idle'); clearTimeout(t); t=setTimeout(()=>ctr.classList.add('idle'),2500); }}
addEventListener('mousemove',showCtr); addEventListener('keydown',showCtr);
show(1); showCtr();
</script>
</body></html>"""

    dst = os.path.expanduser("~/Downloads/RoadVision_발표.html")
    with open(dst, "w", encoding="utf-8") as f:
        f.write(out)
    print("단일 HTML:", dst, f"({os.path.getsize(dst)/1e6:.1f} MB, 슬라이드 {n}장)")

if __name__ == "__main__":
    main()
