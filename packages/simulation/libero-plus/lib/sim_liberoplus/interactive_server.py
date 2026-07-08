# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Interactive LIBERO demo (model-agnostic, chunk-replay).

Command-driven showcase: the sim sits IDLE on a scene until you send an instruction;
then it resets the env + policy and runs that one task to completion (or until Stop),
streaming the composed agentview|wrist view to the browser as MJPEG and saving a debug
MP4 (executed command banner on top). Randomize swaps to a new random scene.

The policy is chosen at runtime via POLICY_FACTORY=module:function (default: the built-in
RandomPolicy shipped with this simulator). Any model that implements sim_liberoplus.Policy can
drive this demo. Stdlib http.server only.

Env: SUITE, TASK_ID, SEED, PORT (8080), OUT_DIR (/outputs), VIEW_RES (512),
VIDEO_RES (640), POLICY_FACTORY.  View: ssh -L 8080:localhost:8080 <host>.
"""
import json
import os
import random
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from sim_liberoplus.envutil import env_int, env_str
from sim_liberoplus.libero_env import SUITES, get_libero_image
from sim_liberoplus.policy import load_policy
from sim_liberoplus.render import banner_frame, compose_view, encode_jpeg, save_mp4
from sim_liberoplus.scene import build_scene

SUITE = env_str("SUITE", "libero_object")
TASK_ID = env_int("TASK_ID", 0)
SEED = env_int("SEED", 1000)
PORT = env_int("PORT", 8080)
VIEW_RES = env_int("VIEW_RES", 512)
VIDEO_RES = env_int("VIDEO_RES", 640)
OUT_DIR = env_str("OUT_DIR", "/sim_outputs")

STATE = {
    "mode": "loading", "instruction": "", "scene_task": "",
    "suite": SUITE, "task_id": TASK_ID, "objects": [],
    "step": 0, "success": False, "status": "starting", "frame": None, "video_url": "",
}
LOCK = threading.Lock()
PENDING = {"action": None, "instruction": ""}
EVENT = threading.Event()
STOP = {"flag": False}


def _set_frame(rgb):
    with LOCK:
        STATE["frame"] = encode_jpeg(rgb)


def engine_thread():
    os.makedirs(os.path.join(OUT_DIR, "interactive"), exist_ok=True)
    with LOCK:
        STATE["status"] = "loading policy ..."
    policy = load_policy()

    def show_idle(sc):
        obs = sc.reset()
        with LOCK:
            STATE.update(mode="idle", status="idle - send an instruction", step=0,
                         success=False, suite=sc.suite, task_id=sc.task_id,
                         scene_task=sc.description, objects=sc.objects,
                         instruction="", video_url="")
        STATE.pop("_video_path", None)
        _set_frame(compose_view(get_libero_image(obs), height=VIEW_RES))

    scene = build_scene(SUITE, TASK_ID, seed=SEED)
    show_idle(scene)

    def run_command(sc, instruction):
        with LOCK:
            STATE.update(mode="running", instruction=instruction, step=0, success=False,
                         status=f"running: {instruction}", video_url="")
        STOP["flag"] = False
        frames = []

        def on_frame(imgs, step, holding):
            rgb = compose_view(imgs, height=VIEW_RES)
            _set_frame(rgb)
            frames.append(banner_frame(compose_view(imgs), instruction, VIDEO_RES))
            with LOCK:
                STATE["step"] = step

        from sim_liberoplus.rollout import run_episode

        success, _ = run_episode(sc, policy, instruction, on_frame=on_frame,
                                 should_stop=lambda: STOP["flag"])

        url = ""
        if frames:
            ts = datetime.now().strftime("%H%M%S")
            name = f"interactive/{ts}_{sc.suite}_{sc.task_id}_{'ok' if success else 'run'}.mp4"
            path = os.path.join(OUT_DIR, name)
            try:
                save_mp4(frames, path, fps=20)
                url = "/video?ts=" + ts
                with LOCK:
                    STATE["_video_path"] = path
            except Exception as e:  # noqa: BLE001
                print("video save failed:", e, flush=True)

        with LOCK:
            STATE.update(mode="idle", success=success, video_url=url,
                         status=("success" if success else ("stopped" if STOP["flag"] else "done")))
        show_idle(sc)

    while True:
        EVENT.wait()
        EVENT.clear()
        with LOCK:
            action, instruction = PENDING["action"], PENDING["instruction"]
            PENDING["action"] = None
        if action == "randomize":
            STOP["flag"] = True
            suite = random.choice(SUITES)
            tid = random.randint(0, 9)
            with LOCK:
                STATE["status"] = f"loading scene {suite}/{tid} ..."
            try:
                new_scene = build_scene(suite, tid, seed=SEED)
            except Exception as e:  # noqa: BLE001
                with LOCK:
                    STATE["status"] = f"scene build failed: {e}"
                continue
            scene.close()
            scene = new_scene
            show_idle(scene)
        elif action == "run":
            run_command(scene, instruction)


PAGE = b"""<!doctype html><html><head><meta charset=utf-8>
<title>FastWAM sim - LIBERO (live)</title>
<style>
 body{background:#0f1012;color:#e8e8ea;font-family:system-ui,sans-serif;margin:0;padding:20px}
 .wrap{max-width:820px;margin:0 auto}
 h1{font-size:18px;font-weight:600;margin:0 0 12px}
 img{width:100%;max-width:760px;height:auto;border-radius:10px;background:#000;display:block}
 .row{display:flex;gap:8px;margin-top:14px}
 input{flex:1;padding:11px;border-radius:8px;border:1px solid #333;background:#1b1b1f;color:#eee;font-size:15px}
 button{padding:11px 16px;border-radius:8px;border:0;color:#fff;font-size:15px;cursor:pointer}
 .send{background:#3b82f6}.stop{background:#ef4444}.rand{background:#8b5cf6}
 .meta{margin-top:12px;font-size:13px;color:#9aa3ad}
 .panel{margin-top:12px;background:#16171b;border:1px solid #26272c;border-radius:10px;padding:12px;font-size:13px}
 .chip{display:inline-block;background:#23252b;border-radius:14px;padding:3px 10px;margin:3px 4px 0 0;color:#cdd3da}
 a{color:#7aa2ff} video{width:100%;max-width:760px;border-radius:10px;margin-top:10px;background:#000}
</style></head><body><div class=wrap>
<h1>FastWAM simulator - LIBERO (live)</h1>
<img src="/stream" alt="sim">
<div class=row>
 <input id=cmd placeholder="type an instruction, then Send (resets the scene and runs it)">
 <button class=send onclick=send()>Send</button>
 <button class=stop onclick=stop()>Stop</button>
 <button class=rand onclick=rnd()>Randomize</button>
</div>
<div class=meta id=meta>status: loading...</div>
<div class=panel><b id=scene>scene</b><div id=objs></div></div>
<div id=vidwrap></div>
</div>
<script>
async function send(){const v=document.getElementById('cmd').value;if(!v)return;
 await fetch('/command',{method:'POST',headers:{'Content-Type':'application/x-www-form-urlencoded'},body:'instruction='+encodeURIComponent(v)});}
async function stop(){await fetch('/stop',{method:'POST'});}
async function rnd(){await fetch('/randomize',{method:'POST'});}
document.getElementById('cmd').addEventListener('keydown',e=>{if(e.key==='Enter')send();});
let lastVid='';
async function poll(){
 try{const s=await(await fetch('/status')).json();
  document.getElementById('meta').textContent='['+s.mode+'] '+s.status+' | step '+s.step+(s.instruction?' | running: "'+s.instruction+'"':'');
  document.getElementById('scene').textContent='Scene: '+s.suite+' / task '+s.task_id+' - "'+s.scene_task+'"';
  document.getElementById('objs').innerHTML='Objects: '+(s.objects||[]).map(o=>'<span class=chip>'+o+'</span>').join('');
  const box=document.getElementById('cmd');
  if(s.mode==='idle'&&!box.value&&s.scene_task)box.value=s.scene_task;
  if(s.video_url&&s.video_url!==lastVid){lastVid=s.video_url;
   document.getElementById('vidwrap').innerHTML='<div style=\"margin-top:8px;font-size:13px;color:#9aa3ad\">last run video:</div><video controls autoplay loop src=\"'+s.video_url+'\"></video>';}
 }catch(e){}
 setTimeout(poll,800);
}
poll();
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, ctype, body):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/":
            self._send(200, "text/html; charset=utf-8", PAGE)
        elif path == "/status":
            with LOCK:
                s = {k: STATE[k] for k in ("mode", "status", "instruction", "scene_task",
                                           "suite", "task_id", "objects", "step",
                                           "success", "video_url")}
            self._send(200, "application/json", json.dumps(s).encode())
        elif path == "/video":
            with LOCK:
                p = STATE.get("_video_path")
            if p and os.path.exists(p):
                with open(p, "rb") as f:
                    self._send(200, "video/mp4", f.read())
            else:
                self._send(404, "text/plain", b"no video")
        elif path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while True:
                    with LOCK:
                        frame = STATE["frame"]
                    if frame:
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                        self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode())
                        self.wfile.write(frame)
                        self.wfile.write(b"\r\n")
                    time.sleep(0.06)
            except (BrokenPipeError, ConnectionResetError):
                pass
        else:
            self._send(404, "text/plain", b"not found")

    def do_POST(self):
        path = urlparse(self.path).path
        if path == "/command":
            n = int(self.headers.get("Content-Length", "0"))
            instr = parse_qs(self.rfile.read(n).decode()).get("instruction", [""])[0].strip()
            if instr:
                STOP["flag"] = True
                with LOCK:
                    PENDING["action"], PENDING["instruction"] = "run", instr
                EVENT.set()
            self.send_response(204)
            self.end_headers()
        elif path == "/stop":
            STOP["flag"] = True
            self.send_response(204)
            self.end_headers()
        elif path == "/randomize":
            STOP["flag"] = True
            with LOCK:
                PENDING["action"] = "randomize"
            EVENT.set()
            self.send_response(204)
            self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()


def main():
    threading.Thread(target=engine_thread, daemon=True).start()
    srv = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print(f"interactive demo on http://0.0.0.0:{PORT}  (ssh -L {PORT}:localhost:{PORT} <host>)", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
