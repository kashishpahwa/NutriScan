"""
NutriScan — AI-powered Food & Nutrition Tracker
Run:  streamlit run food101_app.py

Put the .streamlit/config.toml file next to this script.
The theme (dark background, blue accent, clean font) is set there.
"""

import json, re
from pathlib import Path
from itertools import product as iterproduct

import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

# ─────────────────────────────────────────────────────────────
DEVICE = torch.device(
    "cuda"  if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
ROOT_DIR           = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = ROOT_DIR / "utils" / "best_model.pth"
DEFAULT_NUTRITION  = ROOT_DIR / "utils" / "nutrition_values.json"

NUTRIENT_KEYS = ["calories_kcal","carbs_g","protein_g","fat_g","fiber_g","sugar_g","sodium_mg"]
DEFAULT_GOALS = {
    "calories_kcal":2000,"carbs_g":275,"protein_g":50,
    "fat_g":78,"fiber_g":28,"sugar_g":50,"sodium_mg":2300,
}
DISPLAY_NAMES = {
    "calories_kcal":"Calories","carbs_g":"Carbs","protein_g":"Protein",
    "fat_g":"Fat","fiber_g":"Fiber","sugar_g":"Sugar","sodium_mg":"Sodium",
}
UNITS = {
    "calories_kcal":"kcal","carbs_g":"g","protein_g":"g","fat_g":"g",
    "fiber_g":"g","sugar_g":"g","sodium_mg":"mg",
}
ICONS = {
    "calories_kcal":"🔥","carbs_g":"🌾","protein_g":"💪",
    "fat_g":"🧈","fiber_g":"🥦","sugar_g":"🍬","sodium_mg":"🧂",
}

GRID_SIZES        = [(2,2),(3,3)]
CONF_THRESHOLD    = 0.20
DEFAULT_SERVING_G = 150
OTHERS_LABEL      = "✏️ Other (enter manually)"
OTHERS_CLASS      = "__others__"


# ══════════════════════════════════════════════════════════════
#  MINIMAL CSS  — layout only, no color overrides
#  Colors come from .streamlit/config.toml
# ══════════════════════════════════════════════════════════════
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* Apply Inter globally */
html, body, [class*="css"], [data-testid="stAppViewContainer"],
[data-testid="stSidebar"], button, input, select, textarea {
    font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
}

/* Hide Streamlit branding */
#MainMenu, footer, header,
[data-testid="stDecoration"],
[data-testid="stToolbar"] { display:none !important; }

.block-container { padding: 1.5rem 2.5rem 4rem !important; max-width:1300px; }

/* ── Nutrient grid ── */
.nut-grid {
    display: grid;
    grid-template-columns: repeat(7,1fr);
    gap: 8px; margin: 0.8rem 0;
}
.nut-cell {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 10px;
    padding: 10px 6px; text-align: center;
}
.nut-icon { font-size: 1.1rem; line-height:1; }
.nut-val  { font-size:0.8rem; font-weight:600; color:#e8e8e8; margin-top:5px; }
.nut-name { font-size:0.58rem; color:#666; text-transform:uppercase;
            letter-spacing:0.5px; margin-top:2px; }

/* ── Progress bars ── */
.prog-row {
    display:flex; justify-content:space-between;
    font-size:0.75rem; color:#999; margin-bottom:4px;
}
.prog-name { font-weight:500; color:#ccc; }
.prog-track {
    background: rgba(255,255,255,0.07);
    border-radius:100px; height:6px;
    overflow:hidden; margin-bottom:12px;
}
.prog-fill { height:6px; border-radius:100px; }
.fill-blue   { background:#4f8ef7; }
.fill-amber  { background:#f59e0b; }
.fill-red    { background:#ef4444; }

/* ── Drop zone ── */
.dropzone {
    border: 1.5px dashed rgba(255,255,255,0.1);
    border-radius: 12px; padding: 3rem 2rem;
    text-align: center; color: #444;
}

/* ── Step list ── */
.step {
    display:flex; gap:0.85rem; align-items:flex-start;
    padding:0.6rem 0;
    border-bottom:1px solid rgba(255,255,255,0.05);
}
.step:last-child { border:none; }
.step-n {
    background:rgba(79,142,247,0.15); color:#4f8ef7;
    border-radius:50%; width:22px; height:22px; min-width:22px;
    display:flex; align-items:center; justify-content:center;
    font-size:0.68rem; font-weight:700;
}
.step-t { font-size:0.82rem; color:#888; line-height:1.55; padding-top:2px; }
.step-t b { color:#e8e8e8; }

/* ── Badge ── */
.badge {
    display:inline-block; font-size:0.67rem; font-weight:500;
    padding:2px 8px; border-radius:20px;
    background:rgba(79,142,247,0.12); color:#7aaaf7;
    border:1px solid rgba(79,142,247,0.2);
}
.badge-manual {
    background:rgba(245,158,11,0.1); color:#d4a44a;
    border-color:rgba(245,158,11,0.2);
}

/* ── Metric tweaks ── */
div[data-testid="stMetricValue"] {
    font-size:1rem !important; font-weight:600 !important;
}
div[data-testid="stMetricLabel"] {
    font-size:0.67rem !important; text-transform:uppercase;
    letter-spacing:0.5px; opacity:0.5;
}
div[data-testid="stMetricDelta"] { display:none !important; }

/* ── Dataframe ── */
div[data-testid="stDataFrame"] iframe {
    border-radius:8px;
}

/* ── Buttons ── */
.stButton > button {
    font-family:'Inter',sans-serif !important;
    font-weight:500 !important;
    border-radius:8px !important;
    font-size:0.82rem !important;
}

/* ── Remove top padding on expanders ── */
div[data-testid="stExpander"] { border-radius:10px !important; }
</style>
"""


# ══════════════════════════════════════════════════════════════
#  MODEL
# ══════════════════════════════════════════════════════════════

def build_model(name, n):
    if name == "resnet50":
        m = models.resnet50(weights=None)
        m.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(m.fc.in_features, n))
    elif name in ("efficientnet","efficientnet_b0"):
        m = models.efficientnet_b0(weights=None)
        m.classifier = nn.Sequential(nn.Dropout(0.4), nn.Linear(m.classifier[1].in_features, n))
    elif name == "efficientnet_b3":
        m = models.efficientnet_b3(weights=None)
        m.classifier = nn.Sequential(nn.Dropout(0.5), nn.Linear(m.classifier[1].in_features, n))
    elif name == "vit":
        m = models.vit_b_16(weights=None)
        m.heads.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(m.heads.head.in_features, n))
    else:
        raise ValueError(name)
    return m

@st.cache_resource
def load_model(path):
    ckpt = torch.load(path, map_location=DEVICE)
    m    = build_model(ckpt["model_name"], len(ckpt["classes"])).to(DEVICE)
    m.load_state_dict(ckpt["state_dict"])
    m.eval()
    return m, ckpt["classes"], int(ckpt.get("image_size", 224))

@st.cache_data
def load_json(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


# ══════════════════════════════════════════════════════════════
#  INFERENCE
# ══════════════════════════════════════════════════════════════

def get_tfm(sz):
    return transforms.Compose([
        transforms.Resize(int(sz*1.14)), transforms.CenterCrop(sz),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

@torch.no_grad()
def predict_single(img, model, classes, sz, top_k=5):
    x = get_tfm(sz)(img.convert("RGB")).unsqueeze(0).to(DEVICE)
    p = torch.softmax(model(x), dim=1)
    tp,ti = p.topk(top_k,dim=1)
    return [{"class_id":classes[i.item()],
              "label":classes[i.item()].replace("_"," ").title(),
              "score":float(v.item())} for v,i in zip(tp[0],ti[0])]

def crop_grid(img, rows, cols):
    W,H = img.size; cw,ch = W//cols, H//rows
    return [img.crop((c*cw,r*ch,(c+1)*cw,(r+1)*ch))
            for r,c in iterproduct(range(rows),range(cols))]

@torch.no_grad()
def batch_pred(imgs, model, classes, sz):
    t = get_tfm(sz)
    b = torch.stack([t(i.convert("RGB")) for i in imgs]).to(DEVICE)
    p = torch.softmax(model(b),dim=1)
    tp,ti = p.topk(1,dim=1)
    return [{"class_id":classes[ti[n][0].item()],
              "label":classes[ti[n][0].item()].replace("_"," ").title(),
              "score":float(tp[n][0].item())} for n in range(len(imgs))]

def detect_thali(img, model, classes, sz, conf=CONF_THRESHOLD):
    dc,da = {},{}
    for rows,cols in GRID_SIZES:
        ca = 1.0/(rows*cols)
        for pred in batch_pred(crop_grid(img,rows,cols), model, classes, sz):
            if pred["score"] >= conf:
                cid = pred["class_id"]
                dc[cid] = max(dc.get(cid,0), pred["score"])
                da[cid] = da.get(cid,0) + ca*pred["score"]
    if not da: return []
    tot = sum(da.values())
    return [{"class_id":cid,"label":cid.replace("_"," ").title(),
              "confidence":round(dc[cid]*100,1),"area_pct":round(v/tot*100,1)}
            for cid,v in sorted(da.items(),key=lambda x:-x[1])]


# ══════════════════════════════════════════════════════════════
#  NUTRITION
# ══════════════════════════════════════════════════════════════

def ref_g(nut):
    if "serving_grams" in nut:
        try: return float(nut["serving_grams"])
        except: pass
    ms = re.findall(r"(\d+(?:\.\d+)?)\s*(?:g|ml)\b",
                    str(nut.get("serving","")), re.I)
    return float(ms[-1]) if ms else float(DEFAULT_SERVING_G)

def scale_nut(nut, grams):
    r = grams/ref_g(nut) if ref_g(nut) else 0
    return {k: float(nut.get(k,0))*r for k in NUTRIENT_KEYS}

def sum_nut(items):
    t = {k:0.0 for k in NUTRIENT_KEYS}
    for item in items:
        for k in NUTRIENT_KEYS: t[k] += float(item.get(k,0))
    return t

def fmt(k,v):
    return f"{v:.0f} {UNITS[k]}" if k in ("calories_kcal","sodium_mg") \
           else f"{v:.1f} {UNITS[k]}"

def label_list(classes):
    return sorted(c.replace("_"," ").title() for c in classes)

def to_cid(label): return label.lower().replace(" ","_")


# ══════════════════════════════════════════════════════════════
#  COMPONENTS
# ══════════════════════════════════════════════════════════════

def nut_grid(vals):
    cells = "".join(
        f'<div class="nut-cell">'
        f'<div class="nut-icon">{ICONS[k]}</div>'
        f'<div class="nut-val">{fmt(k,vals[k])}</div>'
        f'<div class="nut-name">{DISPLAY_NAMES[k]}</div>'
        f'</div>' for k in NUTRIENT_KEYS
    )
    st.markdown(f'<div class="nut-grid">{cells}</div>', unsafe_allow_html=True)

def progress_bars(vals, goals):
    html = ""
    for k in NUTRIENT_KEYS:
        g   = goals[k]; v = vals[k]
        pct = min(v/g,1.0)*100 if g else 0
        cls = "fill-blue" if pct < 75 else ("fill-amber" if pct < 100 else "fill-red")
        html += (
            f'<div class="prog-row">'
            f'<span class="prog-name">{ICONS[k]} {DISPLAY_NAMES[k]}</span>'
            f'<span>{fmt(k,v)}'
            f'<span style="color:#444"> / {fmt(k,g)}</span></span>'
            f'</div>'
            f'<div class="prog-track">'
            f'<div class="prog-fill {cls}" style="width:{pct:.1f}%"></div>'
            f'</div>'
        )
    st.markdown(html, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  SESSION STATE
# ══════════════════════════════════════════════════════════════

def init_state():
    defaults = {
        "thali_items":[],"thali_next_id":0,
        "daily_log":[],"daily_goals":dict(DEFAULT_GOALS),
        "last_upload_key":None,
    }
    for k,v in DEFAULT_GOALS.items():
        defaults[f"goal_{k}"] = v
    for k,v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

def add_item(label,cid,grams,source="manual",
             confidence=None,area_pct=None,custom_nut=None):
    st.session_state.thali_items.append({
        "id":st.session_state.thali_next_id,
        "label":label,"class_id":cid,"grams":grams,
        "source":source,"confidence":confidence,
        "area_pct":area_pct,"custom_nut":custom_nut,
    })
    st.session_state.thali_next_id += 1

def seed_detection(dishes, total_g):
    st.session_state.thali_items  = []
    st.session_state.thali_next_id = 0
    for d in dishes:
        add_item(d["label"],d["class_id"],
                 max(round(total_g*d["area_pct"]/100),10),
                 source="detected",
                 confidence=d["confidence"],area_pct=d["area_pct"])


# ══════════════════════════════════════════════════════════════
#  OTHERS FORM
# ══════════════════════════════════════════════════════════════

def others_form(prefix):
    name = st.text_input("Dish name", placeholder="e.g. Mango Lassi", key=f"{prefix}_name")
    c1,c2,c3,c4 = st.columns(4)
    cal  = c1.number_input("Calories (kcal)", 0, 5000, 200, 10, key=f"{prefix}_cal")
    carb = c2.number_input("Carbs (g)",       0, 500,  30,  1,  key=f"{prefix}_carb")
    prot = c3.number_input("Protein (g)",     0, 300,  5,   1,  key=f"{prefix}_prot")
    fat  = c4.number_input("Fat (g)",         0, 300,  8,   1,  key=f"{prefix}_fat")
    c5,c6,c7 = st.columns(3)
    fib  = c5.number_input("Fiber (g)",   0, 100,  2,   1,  key=f"{prefix}_fib")
    sug  = c6.number_input("Sugar (g)",   0, 300,  5,   1,  key=f"{prefix}_sug")
    sod  = c7.number_input("Sodium (mg)", 0, 5000, 200, 50, key=f"{prefix}_sod")
    ref  = st.number_input("Values above are per (g)", 5, 2000, 100, 10, key=f"{prefix}_ref")
    return {"dish_name":name or "Custom dish","serving_grams":ref,
            "calories_kcal":cal,"carbs_g":carb,"protein_g":prot,"fat_g":fat,
            "fiber_g":fib,"sugar_g":sug,"sodium_mg":sod}


# ══════════════════════════════════════════════════════════════
#  DISH EDITOR
# ══════════════════════════════════════════════════════════════

def dish_editor(classes, nutrition_db):
    labels    = label_list(classes)
    full_list = labels + [OTHERS_LABEL]

    if st.session_state.thali_items:
        st.markdown("**Dishes**")
        h = st.columns([3,2,3,1])
        for col,txt in zip(h,["Dish","Quantity (g)","Source",""]):
            col.caption(txt.upper())

        for item in list(st.session_state.thali_items):
            cols = st.columns([3,2,3,1])

            cur = (full_list.index(OTHERS_LABEL)
                   if item["class_id"] == OTHERS_CLASS
                   else (labels.index(item["label"]) if item["label"] in labels else 0))

            new_label = cols[0].selectbox(
                "dish", full_list, index=cur,
                key=f"lbl_{item['id']}", label_visibility="collapsed")

            if new_label == OTHERS_LABEL:
                item["class_id"] = OTHERS_CLASS
                item["label"]    = "Custom Dish"
            elif new_label != item["label"]:
                item["label"]    = new_label
                item["class_id"] = to_cid(new_label)
                item["custom_nut"] = None

            item["grams"] = cols[1].number_input(
                "g", 5, 2000, int(item["grams"]), 10,
                key=f"g_{item['id']}", label_visibility="collapsed")

            if item["source"] == "detected":
                cols[2].markdown(
                    f'<span class="badge">📷 {item["confidence"]}% conf · {item["area_pct"]}% plate</span>',
                    unsafe_allow_html=True)
            else:
                cols[2].markdown(
                    '<span class="badge badge-manual">✋ manual</span>',
                    unsafe_allow_html=True)

            if cols[3].button("✕", key=f"del_{item['id']}"):
                st.session_state.thali_items = [
                    x for x in st.session_state.thali_items if x["id"] != item["id"]]
                st.rerun()

            if item["class_id"] == OTHERS_CLASS:
                with st.expander("Enter nutrition info", expanded=not bool(item.get("custom_nut"))):
                    item["custom_nut"] = others_form(f"edit_{item['id']}")

        st.divider()

    # Add row
    st.markdown("**Add a dish**")
    ac = st.columns([3,2,1])
    pick  = ac[0].selectbox("pick",["— select —"]+full_list,
                             key="add_pick",label_visibility="collapsed")
    grams = ac[1].number_input("g",5,2000,150,10,
                                key="add_g",label_visibility="collapsed")
    add_cn = None
    if pick == OTHERS_LABEL:
        with st.expander("Enter nutrition info", expanded=True):
            add_cn = others_form("add_oth")

    if ac[2].button("Add", use_container_width=True):
        if pick == "— select —":
            st.warning("Select a dish first.")
        elif pick == OTHERS_LABEL:
            n = add_cn["dish_name"] if add_cn else "Custom dish"
            add_item(n, OTHERS_CLASS, grams, source="manual", custom_nut=add_cn)
            st.rerun()
        else:
            add_item(pick, to_cid(pick), grams, source="manual")
            st.rerun()

    # Nutrition
    log_entries = []
    for item in st.session_state.thali_items:
        if item["class_id"] == OTHERS_CLASS and item.get("custom_nut"):
            cn    = item["custom_nut"]
            rg    = float(cn.get("serving_grams", DEFAULT_SERVING_G))
            ratio = item["grams"]/rg if rg else 1.0
            sc    = {k:float(cn.get(k,0))*ratio for k in NUTRIENT_KEYS}
            lbl   = cn.get("dish_name", item["label"])
        else:
            nut   = nutrition_db.get(item["class_id"],{})
            sc    = scale_nut(nut, item["grams"])
            lbl   = item["label"]
        log_entries.append({"Dish":lbl,"Grams":item["grams"],"Servings":1.0, **sc})

    if log_entries:
        st.markdown("**Nutrition breakdown**")
        st.dataframe(
            [{"Dish":e["Dish"],"Qty":f"{e['Grams']} g",
              "Cal":fmt("calories_kcal",e["calories_kcal"]),
              "Carbs":fmt("carbs_g",e["carbs_g"]),
              "Protein":fmt("protein_g",e["protein_g"]),
              "Fat":fmt("fat_g",e["fat_g"]),
              "Fiber":fmt("fiber_g",e["fiber_g"]),
              "Sugar":fmt("sugar_g",e["sugar_g"]),
              "Sodium":fmt("sodium_mg",e["sodium_mg"])} for e in log_entries],
            use_container_width=True, hide_index=True)

        combined = sum_nut(log_entries)
        st.markdown("**Meal total**")
        nut_grid(combined)
        return log_entries, combined

    return [], sum_nut([])


# ══════════════════════════════════════════════════════════════
#  SINGLE DISH
# ══════════════════════════════════════════════════════════════

def single_dish_ui(results, classes, nutrition_db):
    labels    = label_list(classes)
    full_list = labels + [OTHERS_LABEL]
    top       = results[0]

    st.success(f"**{top['label']}** — {top['score']:.0%} confident")

    c1,c2 = st.columns([2,2])
    override = c1.selectbox("Correct if wrong",["— keep —"]+full_list,
                             index=0,key="s_override")
    cn = None
    if override == OTHERS_LABEL:
        with st.expander("Enter nutrition info", expanded=True):
            cn = others_form("s_oth")
        actual = cn.get("dish_name","Custom Dish"); actual_cid = OTHERS_CLASS
    else:
        actual = override if override != "— keep —" else top["label"]
        actual_cid = to_cid(actual)

    nut = nutrition_db.get(actual_cid,{}) if actual_cid != OTHERS_CLASS else {}
    rg  = float(cn["serving_grams"]) if cn else (ref_g(nut) if nut else DEFAULT_SERVING_G)

    grams = c2.number_input(f"Quantity (g) — ref. portion ≈ {rg:.0f} g",
                             5,2000,int(rg),10,key="s_grams")

    if actual_cid != OTHERS_CLASS:
        st.dataframe(
            [{"Dish":r["label"],"Confidence":f"{r['score']*100:.1f}%",
              "Cal":fmt("calories_kcal",scale_nut(nutrition_db.get(r["class_id"],{}),
                        ref_g(nutrition_db.get(r["class_id"],{})))["calories_kcal"]),
              "Carbs":fmt("carbs_g",scale_nut(nutrition_db.get(r["class_id"],{}),
                          ref_g(nutrition_db.get(r["class_id"],{})))["carbs_g"]),
              "Protein":fmt("protein_g",scale_nut(nutrition_db.get(r["class_id"],{}),
                            ref_g(nutrition_db.get(r["class_id"],{})))["protein_g"])}
             for r in results],
            use_container_width=True, hide_index=True)

    if cn:
        ratio = grams/rg if rg else 1.0
        sc = {k:float(cn.get(k,0))*ratio for k in NUTRIENT_KEYS}
    else:
        sc = scale_nut(nut, grams)

    st.markdown(f"**{actual} · {grams} g**")
    nut_grid(sc)
    return {"Dish":actual,"Grams":grams,"Servings":1.0, **sc}


# ══════════════════════════════════════════════════════════════
#  DAILY LOG
# ══════════════════════════════════════════════════════════════

def daily_log_ui():
    st.markdown("### Today's Progress")
    vals  = sum_nut(st.session_state.daily_log)
    goals = st.session_state.daily_goals
    progress_bars(vals, goals)

    if st.session_state.daily_log:
        st.markdown("**Logged meals**")
        rows = []
        for item in st.session_state.daily_log:
            row = {"Dish":item["Dish"],"Qty":f"{item.get('Grams','—')} g"}
            row.update({DISPLAY_NAMES[k]:fmt(k,item[k]) for k in NUTRIENT_KEYS})
            rows.append(row)
        st.dataframe(rows, use_container_width=True, hide_index=True)
        if st.button("Clear log", type="secondary"):
            st.session_state.daily_log = []; st.rerun()
    else:
        st.caption("Nothing logged yet today.")


# ══════════════════════════════════════════════════════════════
#  GOALS EDITOR  — reset works because widget key == session_state key
# ══════════════════════════════════════════════════════════════

def goals_editor():
    with st.expander("🎯 Customise Daily Goals"):
        st.caption("Blue = on track · Amber = nearing limit · Red = exceeded")

        # Reset BEFORE widgets render so they pick up the new values
        if st.button("Reset to defaults", type="secondary", key="reset_btn"):
            for k,v in DEFAULT_GOALS.items():
                st.session_state[f"goal_{k}"] = v
            st.session_state.daily_goals = dict(DEFAULT_GOALS)
            st.rerun()

        c1,c2 = st.columns(2)
        st.session_state.daily_goals["calories_kcal"] = c1.number_input(
            "Calories (kcal)", 500, 6000, step=50, key="goal_calories_kcal")
        st.session_state.daily_goals["carbs_g"] = c2.number_input(
            "Carbs (g)", 0, 600, step=5, key="goal_carbs_g")
        c3,c4 = st.columns(2)
        st.session_state.daily_goals["protein_g"] = c3.number_input(
            "Protein (g)", 0, 300, step=5, key="goal_protein_g")
        st.session_state.daily_goals["fat_g"] = c4.number_input(
            "Fat (g)", 0, 300, step=5, key="goal_fat_g")
        c5,c6,c7 = st.columns(3)
        st.session_state.daily_goals["fiber_g"] = c5.number_input(
            "Fiber (g)", 0, 100, step=1, key="goal_fiber_g")
        st.session_state.daily_goals["sugar_g"] = c6.number_input(
            "Sugar (g)", 0, 200, step=5, key="goal_sugar_g")
        st.session_state.daily_goals["sodium_mg"] = c7.number_input(
            "Sodium (mg)", 0, 5000, step=100, key="goal_sodium_mg")


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

def main():
    st.set_page_config(page_title="NutriScan", page_icon="🥗", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    init_state()

    if not DEFAULT_CHECKPOINT.exists():
        st.error("Model not found — run `python food101_train.py` first."); st.stop()
    nutrition_db = load_json(DEFAULT_NUTRITION)
    if not nutrition_db:
        st.error("Nutrition DB missing — check `utils/nutrition_values.json`."); st.stop()

    model, classes, image_size = load_model(DEFAULT_CHECKPOINT)

    # ── Sidebar ──────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 🥗 NutriScan")
        st.caption("AI food recognition & nutrition tracking")
        st.divider()

        mode = st.radio("Mode", [
            "📷  Single dish",
            "🍱  Thali / multi-dish",
            "✋  Manual entry",
        ], label_visibility="collapsed")
        mode = mode.split("  ",1)[1]

        if "Thali" in mode:
            st.divider()
            conf_thresh = st.slider(
                "Detection sensitivity", 0.05, 0.60, CONF_THRESHOLD, 0.05,
                help="Lower = more dishes found. Higher = confident matches only.")
            total_grams = st.number_input(
                "Estimated total plate weight (g)", 100, 3000, 600, 50)

        st.divider()
        st.caption("Recognises 182 food categories including Indian and international cuisines. "
                   "Nutrition values are estimates and vary by recipe and brand.")

    # ── Page title ───────────────────────────────────────────
    st.markdown("# 🥗 NutriScan")
    st.caption("Snap a photo of your meal — get instant nutrition insights and track your day.")
    st.divider()

    goals_editor()
    st.markdown("<br>", unsafe_allow_html=True)

    # ── Main ─────────────────────────────────────────────────
    if mode == "Manual entry":
        st.markdown("### Manual Meal Log")
        st.caption("Select each dish and quantity. Choose **Other** if it's not in the list.")
        log_entries, _ = dish_editor(classes, nutrition_db)
        if log_entries:
            if st.button("➕ Add to today's log", type="primary"):
                st.session_state.daily_log.extend(log_entries)
                st.success(f"✅ {len(log_entries)} dish(es) added.")
                st.session_state.thali_items = []; st.rerun()

    else:
        left, right = st.columns([1,1], gap="large")

        with left:
            st.markdown("### Upload Photo")
            uploaded = st.file_uploader("photo", type=["jpg","jpeg","png","webp"],
                                        label_visibility="collapsed")
            image = Image.open(uploaded).convert("RGB") if uploaded else None
            if image:
                st.image(image, use_container_width=True)
            else:
                st.markdown(
                    '<div class="dropzone">'
                    '<div style="font-size:2rem;opacity:0.3">📷</div>'
                    '<div style="font-size:0.82rem;margin-top:0.5rem;color:#555">'
                    'Drop your food photo here<br>'
                    '<span style="font-size:0.7rem;color:#3a3a3a">JPG · PNG · WEBP</span>'
                    '</div></div>',
                    unsafe_allow_html=True)

        with right:
            if not image:
                st.markdown("### How it works")
                st.markdown(
                    '<div>'
                    '<div class="step"><div class="step-n">1</div>'
                    '<div class="step-t"><b>Upload</b> a clear photo of your meal</div></div>'
                    '<div class="step"><div class="step-n">2</div>'
                    '<div class="step-t"><b>Review</b> predictions and correct any mistakes</div></div>'
                    '<div class="step"><div class="step-n">3</div>'
                    '<div class="step-t"><b>Adjust</b> gram quantities per dish</div></div>'
                    '<div class="step"><div class="step-n">4</div>'
                    '<div class="step-t"><b>Log it</b> to track your daily nutrition goals</div></div>'
                    '</div>',
                    unsafe_allow_html=True)
                st.caption("💡 For thali, switch to Thali mode in the sidebar.")

            elif "Thali" in mode:
                uk = f"{uploaded.name}_{uploaded.size}"
                if st.session_state.last_upload_key != uk:
                    with st.spinner("Detecting dishes…"):
                        dishes = detect_thali(image, model, classes, image_size, conf=conf_thresh)
                    seed_detection(dishes, total_grams)
                    st.session_state.last_upload_key = uk
                    if not dishes:
                        st.warning("No dishes detected. Lower sensitivity or try Single dish mode.")

                log_entries, _ = dish_editor(classes, nutrition_db)
                if log_entries:
                    if st.button("➕ Add all to today's log", type="primary"):
                        st.session_state.daily_log.extend(log_entries)
                        st.success(f"✅ {len(log_entries)} dish(es) added.")
                        st.session_state.thali_items = []; st.rerun()

            else:
                results   = predict_single(image, model, classes, image_size)
                log_entry = single_dish_ui(results, classes, nutrition_db)
                if st.button("➕ Add to today's log", type="primary"):
                    st.session_state.daily_log.append(log_entry)
                    st.success(f"✅ {log_entry['Dish']} added."); st.rerun()

    st.divider()
    daily_log_ui()


if __name__ == "__main__":
    main()