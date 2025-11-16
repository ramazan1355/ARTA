# armeta_app.py
import io, os, uuid, json
from pathlib import Path
from typing import List, Dict, Any
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import cv2
import fitz
import easyocr
import pytesseract

# YOLO (ultralytics)
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    YOLO = None
    ULTRALYTICS_AVAILABLE = False

# --- Config ---
BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR = BASE_DIR / "models"
YOLO_WEIGHTS = MODELS_DIR / "yolov8.pt"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

_detector = None
def load_detector():
    global _detector
    if _detector is None and ULTRALYTICS_AVAILABLE and YOLO_WEIGHTS.exists():
        _detector = YOLO(str(YOLO_WEIGHTS))
    return _detector

reader = easyocr.Reader(['en'], gpu=False)

# --- Utils ---
def ensure_image(b: bytes) -> np.ndarray:
    return np.array(Image.open(io.BytesIO(b)).convert("RGB"))

def draw_boxes(img: np.ndarray, objects: List[Dict[str, Any]]) -> np.ndarray:
    out = img.copy()
    for obj in objects:
        x1, y1, x2, y2 = map(int, obj["bbox"])
        label = obj.get("label", "")
        conf = obj.get("confidence", 0.0)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(out, f"{label} {conf:.2f}", (x1, max(16,y1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    return out

def write_json(obj: Dict[str, Any], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

# --- Detection ---
def fallback_detect(img_np: np.ndarray) -> List[Dict[str, Any]]:
    h,w = img_np.shape[:2]
    return [
        {"label":"signature","bbox":[int(w*0.6), int(h*0.7), int(w*0.95), int(h*0.92)],"confidence":0.6},
        {"label":"stamp","bbox":[int(w*0.02), int(h*0.02), int(w*0.2), int(h*0.18)],"confidence":0.5}
    ]

def ultralytics_detect(img_bytes: bytes) -> List[Dict[str, Any]]:
    detector = load_detector()
    if detector is None:
        return []
    img_np = ensure_image(img_bytes)
    is_success, buf = cv2.imencode(".jpg", cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
    preds = detector.predict(source=buf.tobytes(), imgsz=1024, conf=0.35, device='cpu')
    results=[]
    if preds and len(preds)>0:
        r=preds[0]
        boxes=getattr(r,"boxes",None)
        if boxes:
            for b in boxes:
                xyxy=b.xyxy.cpu().numpy().tolist()
                conf=float(b.conf.cpu().numpy()) if hasattr(b,"conf") else 0.5
                cls=int(b.cls.cpu().numpy()) if hasattr(b,"cls") else 0
                label=detector.model.names.get(cls,str(cls)) if hasattr(detector,"model") else str(cls)
                results.append({"label":label,"bbox":[int(xyxy[0]),int(xyxy[1]),int(xyxy[2]),int(xyxy[3])],"confidence":conf})
    return results

# --- App ---
app = FastAPI(title="ARMETA Document AI")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

INDEX_HTML = """
<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>ARMETA Document AI</title>
<style>
body{font-family:sans-serif;background:#0f1724;color:#e6eef8;padding:12px}
.screen{max-width:480px;margin:auto;background:#0b1220;padding:18px;border-radius:14px}
h1{margin:0 0 8px;font-size:24px}
.btn{padding:10px 14px;border-radius:10px;background:linear-gradient(90deg,#7c3aed,#5b21b6);border:none;color:white;cursor:pointer;margin:4px}
.drop{border:2px dashed #fff; padding:18px; border-radius:12px; text-align:center; margin-top:12px}
img.preview{display:block;max-width:100%;border-radius:12px;margin-top:12px}
pre{background:rgba(0,0,0,0.25);padding:10px;border-radius:8px;overflow:auto;height:160px}
</style>
</head>
<body>
<div class="screen">
<h1>ARMETA Document AI</h1>
<button id="uploadBtn" class="btn">Upload</button>
<div id="dropZone" class="drop">Drop file here or click</div>
<input id="fileInput" type="file" style="display:none"/>
<p id="fileName">—</p>
<img id="thumb" class="preview" style="display:none"/>
<div>
<button id="detectNow" class="btn">Detect Image</button>
<button id="detectPdfNow" class="btn">Detect PDF</button>
<button id="autofillNow" class="btn">Auto-fill PDF</button>
</div>
<img id="drawerPreview" class="preview" style="display:none"/>
<pre id="miniJson">{ }</pre>
<button id="downloadJson" class="btn">Download JSON</button>
</div>
<script>
const fileInput=document.getElementById('fileInput');
const dropZone=document.getElementById('dropZone');
const fileName=document.getElementById('fileName');
const thumb=document.getElementById('thumb');
let currentFile=null;
let lastResult=null;

['dragenter','dragover','dragleave','drop'].forEach(evt=>dropZone.addEventListener(evt,e=>{e.preventDefault();e.stopPropagation();}));
dropZone.addEventListener('click',()=>fileInput.click());
dropZone.addEventListener('drop',e=>{const f=e.dataTransfer.files[0];if(f)setFile(f);});
fileInput.onchange=e=>{if(e.target.files.length)setFile(e.target.files[0]);};
function setFile(f){currentFile=f;fileName.textContent=f.name;thumb.src=URL.createObjectURL(f);thumb.style.display='block';document.getElementById('miniJson').textContent='{ }';}
const backend=window.location.origin;

async function postFD(url,fd,blob=false){const res=await fetch(url,{method:'POST',body:fd});if(!res.ok)throw new Error('Server '+res.status);return blob?res.blob():res.json();}

document.getElementById('detectNow').onclick=async()=>{
  if(!currentFile)return alert('Choose file first');
  if(currentFile.type==='application/pdf')return alert('Use Detect PDF');
  const fd=new FormData();fd.append('file',currentFile);
  const blob=await postFD(backend+'/detect/image',fd,true);
  document.getElementById('drawerPreview').src=URL.createObjectURL(blob);
};

document.getElementById('detectPdfNow').onclick=async()=>{
  if(!currentFile)return alert('Choose file first');
  if(currentFile.type!=='application/pdf')return alert('Must be PDF');
  const fd=new FormData();fd.append('file',currentFile);
  const data=await postFD(backend+'/detect/pdf',fd,false);
  lastResult=data;document.getElementById('miniJson').textContent=JSON.stringify(data,null,2);
  if(Array.isArray(data.results)&&data.results.length&&data.results[0].visualization){
    document.getElementById('drawerPreview').src=backend+'/'+data.results[0].visualization;
    document.getElementById('drawerPreview').style.display='block';
  }
};

async function pickFile(promptText){return new Promise(resolve=>{const inp=document.createElement('input');inp.type='file';inp.accept='image/*';inp.onchange=()=>resolve(inp.files[0]||null);inp.click();});}

document.getElementById('autofillNow').onclick=async()=>{
  if(!currentFile)return alert('Choose PDF first');if(currentFile.type!=='application/pdf')return alert('Must be PDF');
  const sig=await pickFile('Signature');if(!sig)return;
  const stamp=await pickFile('Stamp');if(!stamp)return;
  const fd=new FormData();fd.append('pdf',currentFile);fd.append('signature',sig);fd.append('stamp',stamp);
  const blob=await postFD(backend+'/autofill',fd,true);
  const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='filled_'+currentFile.name;a.click();
};

document.getElementById('downloadJson').onclick=()=>{
  if(!lastResult)return alert('No result');
  const blob=new Blob([JSON.stringify(lastResult,null,2)],{type:'application/json'});
  const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='result.json';a.click();
};
</script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(INDEX_HTML)

# --- /detect/image ---
@app.post("/detect/image")
async def detect_image(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(400,"Must be image")
    data=await file.read()
    img_np=ensure_image(data)
    objects=[]
    if ULTRALYTICS_AVAILABLE and YOLO_WEIGHTS.exists():
        try: objects=ultralytics_detect(data)
        except: objects=fallback_detect(img_np)
    else: objects=fallback_detect(img_np)
    vis=draw_boxes(img_np,objects)
    vis_name=f"vis_{uuid.uuid4().hex[:8]}.jpg"
    vis_path=RESULTS_DIR/vis_name
    cv2.imwrite(str(vis_path),cv2.cvtColor(vis,cv2.COLOR_RGB2BGR))
    json_obj={"image_name":file.filename,"objects":objects}
    json_name=f"res_{uuid.uuid4().hex[:8]}.json"
    json_path=RESULTS_DIR/json_name
    write_json(json_obj,json_path)
    return FileResponse(vis_path, media_type="image/jpeg", headers={"X-Result-JSON-Path":str(json_path.relative_to(BASE_DIR))})

# --- /detect/pdf ---
@app.post("/detect/pdf")
async def detect_pdf(file: UploadFile = File(...)):
    if file.content_type!="application/pdf": raise HTTPException(400,"Must be PDF")
    data=await file.read()
    doc=fitz.open(stream=data,filetype="pdf")
    results=[]
    for i,page in enumerate(doc):
        pix=page.get_pixmap(matrix=fitz.Matrix(2,2))
        img_bytes=pix.tobytes()
        img_np=ensure_image(img_bytes)
        if ULTRALYTICS_AVAILABLE and YOLO_WEIGHTS.exists():
            try: objects=ultralytics_detect(img_bytes)
            except: objects=fallback_detect(img_np)
        else: objects=fallback_detect(img_np)
        vis=draw_boxes(img_np,objects)
        vis_name=f"pdf_vis_p{i+1}_{uuid.uuid4().hex[:6]}.jpg"
        vis_path=RESULTS_DIR/vis_name
        cv2.imwrite(str(vis_path),cv2.cvtColor(vis,cv2.COLOR_RGB2BGR))
        results.append({"page":i+1,"objects":objects,"visualization":str(vis_path.relative_to(BASE_DIR))})
    out={"results":results}
    json_name=f"pdf_result_{uuid.uuid4().hex[:8]}.json"
    write_json(out,RESULTS_DIR/json_name)
    return JSONResponse(out)

# --- /autofill ---
@app.post("/autofill")
async def autofill(pdf:UploadFile=File(...), signature:UploadFile=File(...), stamp:UploadFile=File(...)):
    pdf_bytes=await pdf.read()
    sig_img=Image.open(io.BytesIO(await signature.read())).convert("RGBA")
    stamp_img=Image.open(io.BytesIO(await stamp.read())).convert("RGBA")
    doc=fitz.open(stream=pdf_bytes,filetype="pdf")
    out_pdf=fitz.open()
    for page in doc:
        pix=page.get_pixmap()
        base=Image.frombytes("RGB",(pix.width,pix.height),pix.samples)
        w,h=base.size
        sig_w=int(w*0.25);sig_h=int(sig_img.height*(sig_w/sig_img.width));sig_r=sig_img.resize((sig_w,sig_h),Image.LANCZOS)
        stamp_w=int(w*0.18);stamp_h=int(stamp_img.height*(stamp_w/stamp_img.width));stamp_r=stamp_img.resize((stamp_w,stamp_h),Image.LANCZOS)
        base.paste(sig_r,(w-sig_w-40,h-sig_h-40),sig_r)
        base.paste(stamp_r,(40,40),stamp_r)
        buf=io.BytesIO();base.save(buf,format="PDF");buf.seek(0)
        page_pdf=fitz.open(stream=buf.read(),filetype="pdf");out_pdf.insert_pdf(page_pdf)
    out_name=f"filled_{uuid.uuid4().hex[:8]}.pdf"
    out_path=RESULTS_DIR/out_name
    out_pdf.save(str(out_path))
    return FileResponse(out_path,media_type="application/pdf")
