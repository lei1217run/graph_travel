import os
import json
import re
import time
import logging
import uuid
from openai import OpenAI
try:
    from xiyou.config import load_settings, resolve_paths, list_target_books
except ImportError:
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from xiyou.config import load_settings

settings = load_settings()
logging.basicConfig(level=logging.INFO, format="%(message)s")
API_KEY = settings["llm"]["api_key"]
BASE_URL = settings["llm"]["base_url"]
MODEL_NAME = settings["llm"]["model_name"]
SYSTEM_PROMPT = settings["llm"].get("system_prompt", "你是一个输出 JSON 的助手。")
LLM_STREAM = settings["llm"].get("stream", False)
LLM_THINKING = settings["llm"].get("thinking", False)
_corpora = settings.get("corpora", {})
_default_book = _corpora.get("default") or (next(iter(_corpora.get("items", {}).keys())) if isinstance(_corpora.get("items", {}), dict) and _corpora.get("items", {}) else "xiyouji")
_paths_single = resolve_paths(settings, _default_book)
INPUT_DIR = _paths_single["chapters_dir"]
RESULT_DIR = _paths_single["results_dir"]
LIMIT_COUNT = settings["run"]["limit_count"]
TIMEOUT_MS = settings["run"]["timeout_ms"]
NAMING_MODE = settings["naming"]["mode"]
PROMPTS_CFG = settings.get("prompts", {})
if not API_KEY or not BASE_URL:
    raise ValueError("缺少LLM配置")
os.makedirs(RESULT_DIR, exist_ok=True)


# ================= 核心逻辑 =================

def get_client():
    return OpenAI(api_key=API_KEY, base_url=BASE_URL)

def _estimate_dialogue_ratio(text: str) -> float:
    lines = [l for l in text.splitlines() if l.strip()]
    dialogue = sum(1 for l in lines if re.search(r'[“”"\']', l))
    return dialogue / max(1, len(lines))

def _choose_template(text: str) -> dict:
    selected_id = PROMPTS_CFG.get("selected", "relations_plus")
    mode = PROMPTS_CFG.get("selection_mode", "static")
    if mode == "dynamic":
        sel = PROMPTS_CFG.get("selectors", {})
        length_th = sel.get("length_threshold", 8000)
        dialog_th = sel.get("dialogue_ratio_threshold", 0.50)
        if len(text) > length_th and _estimate_dialogue_ratio(text) > dialog_th:
            override_to = PROMPTS_CFG.get("dynamic", {}).get("override_to", "events_relations")
            selected_id = override_to
    for t in PROMPTS_CFG.get("templates", []):
        if t.get("id") == selected_id:
            return t
    return {"id": selected_id, "entity_types": ["Person","Location","Item","Spell","Organization"], "relation_types": ["师徒","敌对","位于","持有","使用"], "instructions": "仅输出JSON列表"}

def _build_prompt(tpl: dict, text: str) -> str:
    entity_types = ", ".join(tpl.get("entity_types", []))
    relation_types = ", ".join(tpl.get("relation_types", []))
    instructions = tpl.get("instructions", "仅输出JSON")
    vars = tpl.get("variables", {})
    lang = vars.get("language", "zh")
    book = vars.get("book", "")
    syn = vars.get("synonyms_map", {})
    alias_rules = vars.get("alias_rules", [])
    syn_lines = "; ".join([f"{k}->{v}" for k, v in syn.items()])
    alias_text = "; ".join(alias_rules)
    schema = tpl.get("output_schema", "")
    max_tokens = tpl.get("max_tokens", 0)
    rel_cfg = settings.get("relations", {})
    allowed = ", ".join(rel_cfg.get("allowed", []))
    precedence = ", ".join(rel_cfg.get("precedence", []))
    header = (
        f"你是一个知识图谱提取专家。请用{lang}输出严格的JSON。\n"
        f"语料：{book}\n"
        f"输出结构：{schema}\n"
        f"输出长度提示：{max_tokens}\n"
        f"实体类型：{entity_types}\n"
        f"关系类型：{relation_types}\n"
        f"闭集关系：{allowed}\n"
        f"关系优先级：{precedence}\n"
        f"同义词合并：{syn_lines}\n"
        f"别名规则：{alias_text}\n"
        f"{instructions}\n"
        f"当一条证据同时支持多个关系时，请分别输出多条记录，不进行覆盖。\n"
        f"仅从闭集关系中选择，若不确定请忽略。\n"
        f"保留 evidence 原文片段，并标注触发词类别（如：姓氏/法名/封号/别名）。\n"
        f"若实体类型无法确定，仍输出关系但勿猜测类型。\n\n"
    )
    return header + f"【文本内容】：\n{text}"

def extract_chapter(client, text, filename, tpl: dict):
    try:
        prompt = _build_prompt(tpl, text)
        messages=[
            {"role": "system", "content": tpl.get("variables", {}).get("system_prompt", SYSTEM_PROMPT)},
            {"role": "user", "content": prompt}
        ]
        extra = {"enable_thinking": True} if LLM_THINKING else None
        if LLM_STREAM:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.1,
                timeout=TIMEOUT_MS / 1000.0,
                stream=True,
                extra_body=extra
            )
            chunks = []
            for chunk in completion:
                try:
                    delta = chunk.choices[0].delta
                    if hasattr(delta, "content") and delta.content:
                        chunks.append(delta.content)
                except Exception:
                    pass
            content = "".join(chunks)
        else:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.1,
                timeout=TIMEOUT_MS / 1000.0,
                extra_body=extra
            )
            content = response.choices[0].message.content
        content = content.replace("```json", "").replace("```", "").strip()
        try:
            return json.loads(content)
        except Exception:
            s = content.strip()
            i = s.find("{"); j = s.rfind("}")
            if i != -1 and j != -1:
                try:
                    return json.loads(s[i:j+1])
                except Exception:
                    pass
            i = s.find("["); j = s.rfind("]")
            if i != -1 and j != -1:
                return json.loads(s[i:j+1])
            raise
    except Exception as e:
        logging.error(json.dumps({"stage": "extract", "event": "error", "file": filename, "template_id": tpl.get("id", "relations_plus"), "error_type": type(e).__name__, "error_message": str(e)}, ensure_ascii=False))
        return None


def _list_chapter_files(input_dir):
    items = []
    for name in os.listdir(input_dir):
        if not name.endswith(".txt"):
            continue
        m = re.match(r"^(?:chapter_(\d{3})|(\d{3})_.*)\.txt$", name)
        if not m:
            continue
        num = int((m.group(1) or m.group(2)))
        items.append((num, os.path.join(input_dir, name)))
    items.sort(key=lambda x: x[0])
    return [p for _, p in items]

def process_book(book_key: str, input_dir: str, result_dir: str, client):
    files = _list_chapter_files(input_dir)
    files_to_process = files[:LIMIT_COUNT] if LIMIT_COUNT else files
    total = len(files_to_process)
    for i, file_path in enumerate(files_to_process):
        name = os.path.basename(file_path)
        m = re.match(r"^(?:chapter_\d{3}|(\d{3})_.*)\.txt$", name)
        cid = int((m.group(1))) if m and m.group(1) else (int(re.match(r"^(\d{3})_", name).group(1)) if re.match(r"^(\d{3})_", name) else i + 1)
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        title = None
        tm = re.match(r"^(?:chapter_\d{3}|(\d{3})_(.+))\.txt$", name)
        if tm and tm.group(2):
            title = tm.group(2)
        if not title:
            title = next((l for l in text.splitlines() if l.strip()), "")
        tpl = _choose_template(text)
        tpl_vars = dict(tpl.get("variables", {}))
        tpl_vars["book"] = book_key
        tpl["variables"] = tpl_vars
        template_id = tpl.get("id", "relations_plus")
        basename_id = settings.get("results", {}).get("basename_template_id", "relations_plus")
        result_name = f"result_{cid:03d}.json" if template_id == basename_id else f"result_{cid:03d}__tpl-{template_id}.json"
        if settings.get("results", {}).get("use_model_suffix", False):
            base, ext = os.path.splitext(result_name)
            result_name = f"{base}__model-{MODEL_NAME.replace(' ', '_').replace('/', '_')}{ext}"
        result_path = os.path.join(result_dir, result_name)
        overwrite = settings.get("results", {}).get("overwrite", False)
        if os.path.exists(result_path) and not overwrite:
            logging.info(json.dumps({"stage": "extract", "event": "skip", "index": i + 1, "total": total, "file": name, "chapter_id": f"{cid:03d}", "chapter_title": title, "template_id": template_id, "model_name": MODEL_NAME}, ensure_ascii=False))
            continue
        if os.path.exists(result_path) and overwrite:
            logging.info(json.dumps({"stage": "extract", "event": "overwrite", "index": i + 1, "total": total, "file": name, "target": result_name, "chapter_id": f"{cid:03d}", "chapter_title": title, "template_id": template_id, "model_name": MODEL_NAME}, ensure_ascii=False))
        request_id = f"req_{uuid.uuid4().hex[:8]}"
        logging.info(json.dumps({"stage": "extract", "event": "start", "index": i + 1, "total": total, "file": name, "chapter_id": f"{cid:03d}", "chapter_title": title, "model_name": MODEL_NAME, "request_id": request_id}, ensure_ascii=False))
        logging.info(json.dumps({"stage": "extract", "event": "template_selected", "template_id": template_id, "chapter_id": f"{cid:03d}", "chapter_title": title, "model_name": MODEL_NAME, "request_id": request_id}, ensure_ascii=False))
        t0 = time.time()
        data = extract_chapter(client, text, name, tpl)
        if data:
            bk_name = None
            items = settings.get("corpora", {}).get("items", {})
            if isinstance(items, dict):
                info = items.get(book_key) or {}
                bk_name = info.get("name") or info.get("folder")
            if not bk_name:
                bk_name = book_key
            meta = {"book_key": book_key, "book_name": bk_name, "chapter_id": f"{cid:03d}", "chapter_title": title, "model_name": MODEL_NAME, "template_id": template_id, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            payload = None
            if isinstance(data, dict):
                data["meta"] = meta
                payload = data
            else:
                payload = {"relations": data, "meta": meta}
            with open(result_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            dt = int((time.time() - t0) * 1000)
            rel_count = len((payload or {}).get("relations", [])) if isinstance(payload, dict) else 0
            evt_count = len((payload or {}).get("events", [])) if isinstance(payload, dict) else 0
            total_count = rel_count + evt_count
            logging.info(json.dumps({"stage": "extract", "event": "saved", "file": result_name, "duration_ms": dt, "template_id": template_id, "relations_count": rel_count, "events_count": evt_count, "records_total": total_count, "chapter_id": f"{cid:03d}", "chapter_title": title, "model_name": MODEL_NAME, "request_id": request_id}, ensure_ascii=False))
        else:
            logging.error(json.dumps({"stage": "extract", "event": "fail", "file": name, "template_id": template_id, "chapter_id": f"{cid:03d}", "chapter_title": title, "model_name": MODEL_NAME, "request_id": request_id}, ensure_ascii=False))
    logging.info(json.dumps({"stage": "extract", "event": "done"}, ensure_ascii=False))

def main():
    client = get_client()
    mode = settings.get("corpora", {}).get("mode", "single")
    if mode == "batch":
        books = list_target_books(settings)
        for bk in books:
            p = resolve_paths(settings, bk)
            os.makedirs(p["results_dir"], exist_ok=True)
            logging.info(json.dumps({"stage": "extract", "event": "book_start", "book_key": bk, "input_dir": p.get("chapters_dir"), "result_dir": p.get("results_dir")}, ensure_ascii=False))
            process_book(bk, p["chapters_dir"], p["results_dir"], client)
    else:
        os.makedirs(RESULT_DIR, exist_ok=True)
        process_book(_default_book, INPUT_DIR, RESULT_DIR, client)


if __name__ == "__main__":
    main()
