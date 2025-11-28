import os
import json
import time
import re
from typing import List, Dict, Optional, Tuple, Set
from pydantic import BaseModel
from .config import load_settings, resolve_paths

class CompareChapterQuery(BaseModel):
    book: str
    chapter_id: str
    templates: List[str]
    model_name: Optional[str] = None
    confidence_min: float = 0.0
    include_events: bool = True

class CompareMetrics(BaseModel):
    relations_count_by_template: Dict[str, int]
    events_count_by_template: Dict[str, int]
    relations_union_count: int
    relations_intersection_count: int
    relations_jaccard: float
    events_union_count: int
    events_intersection_count: int
    events_jaccard: float
    by_relation_type: Dict[str, Dict[str, int]]

class CompareBatchQuery(BaseModel):
    book: str
    chapter_ids: Optional[List[str]] = None
    chapter_range: Optional[List[str]] = None
    templates: List[str]
    model_name: Optional[str] = None
    confidence_min: float = 0.0
    include_events: bool = True

def _merge_synonyms(settings: dict, tpl_ids: List[str]) -> Dict[str, str]:
    m: Dict[str, str] = {}
    for t in settings.get("prompts", {}).get("templates", []):
        if t.get("id") in tpl_ids:
            for k, v in (t.get("variables", {}).get("synonyms_map", {}) or {}).items():
                m[k] = v
    return m

def _collect_alias_rules(settings: dict, tpl_ids: List[str]) -> Dict[str, List]:
    strip_tokens: List[str] = []
    regex_rules: List[Tuple[re.Pattern, str]] = []
    for t in settings.get("prompts", {}).get("templates", []):
        if t.get("id") in tpl_ids:
            rules = (t.get("variables", {}).get("alias_rules", []) or [])
            for r in rules:
                if isinstance(r, str):
                    s = r.strip()
                    if s.startswith("strip:"):
                        tok = s.split(":", 1)[1].strip()
                        if tok:
                            strip_tokens.append(tok)
                    elif s.startswith("regex:"):
                        expr = s.split(":", 1)[1]
                        if "=" in expr:
                            pat, rep = expr.split("=", 1)
                            try:
                                regex_rules.append((re.compile(pat.strip()), rep))
                            except Exception:
                                pass
    return {"strip_tokens": strip_tokens, "regex_rules": regex_rules}

def _canonical(name: str, syn: Dict[str, str], alias: Dict[str, List]) -> str:
    n = name.strip()
    for pat, rep in alias.get("regex_rules", []):
        try:
            n = pat.sub(rep, n)
        except Exception:
            pass
    n = syn.get(n, n)
    for tok in alias.get("strip_tokens", []):
        if n.startswith(tok):
            n = n[len(tok):]
        if n.endswith(tok):
            n = n[:-len(tok)]
    return n.strip()

def _read_result(path: str) -> Optional[Dict]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        s = f.read().strip()
        if not s:
            return None
        try:
            return json.loads(s)
        except:
            return None

def _result_path(base_dir: str, cid: str, default_tpl: str, tpl_id: Optional[str], model_name: Optional[str]) -> str:
    name = f"result_{cid}.json" if not tpl_id or tpl_id == default_tpl else f"result_{cid}__tpl-{tpl_id}.json"
    if model_name:
        base, ext = os.path.splitext(name)
        name = f"{base}__model-{model_name.replace(' ', '_').replace('/', '_')}{ext}"
    return os.path.join(base_dir, name)

def _candidate_paths(base_dir: str, cid: str, default_tpl: str, tpl_id: Optional[str], settings: dict, model_name_hint: Optional[str]) -> List[str]:
    paths: List[str] = []
    name_tpl = tpl_id if tpl_id and tpl_id != default_tpl else None
    use_suffix = settings.get("results", {}).get("use_model_suffix", False)
    lm_name = model_name_hint or settings.get("llm", {}).get("model_name")
    base = f"result_{cid}.json" if not name_tpl else f"result_{cid}__tpl-{name_tpl}.json"
    paths.append(os.path.join(base_dir, base))
    if use_suffix and lm_name:
        b, e = os.path.splitext(base)
        paths.append(os.path.join(base_dir, f"{b}__model-{lm_name.replace(' ', '_').replace('/', '_')}{e}"))
    return paths

def _parse_relations(data, syn: Dict[str, str], alias: Dict[str, List], confidence_min: float) -> Tuple[List[Dict], Set[str], Dict[str, int]]:
    items: List[Dict] = []
    keys: Set[str] = set()
    by_type: Dict[str, int] = {}
    seq = []
    if isinstance(data, list):
        seq = data
    elif isinstance(data, dict):
        seq = data.get("relations", [])
    for r in seq:
        head = _canonical(str(r.get("head", "")), syn, alias)
        tail = _canonical(str(r.get("tail", "")), syn, alias)
        head_type = str(r.get("head_type", ""))
        tail_type = str(r.get("tail_type", ""))
        relation = str(r.get("relation", ""))
        conf = r.get("confidence", None)
        if conf is not None and conf < confidence_min:
            continue
        q = r.get("qualifiers", {}) or {}
        ev = r.get("evidence", None)
        key = "|".join([head, head_type, relation, tail, tail_type])
        keys.add(key)
        items.append({"head": head, "head_type": head_type, "relation": relation, "tail": tail, "tail_type": tail_type, "confidence": conf, "qualifiers": q, "evidence": ev})
        by_type[relation] = by_type.get(relation, 0) + 1
    return items, keys, by_type

def _parse_events(data, syn: Dict[str, str], alias: Dict[str, List], confidence_min: float) -> Tuple[List[Dict], Set[str]]:
    items: List[Dict] = []
    keys: Set[str] = set()
    seq = []
    if isinstance(data, dict):
        seq = data.get("events", [])
    for e in seq:
        et = str(e.get("event_type", ""))
        parts = e.get("participants", {}) or {}
        norm_parts: Dict[str, List[str]] = {}
        for role, names in parts.items():
            norm_parts[role] = [ _canonical(str(x), syn, alias) for x in (names or []) ]
        ev = e.get("evidence", None)
        conf = e.get("confidence", None)
        if conf is not None and conf < confidence_min:
            continue
        role_sets = ",".join([ f"{role}:{','.join(norm_parts.get(role, []))}" for role in sorted(norm_parts.keys()) ])
        key = "|".join([et, role_sets])
        keys.add(key)
        items.append({"event_type": et, "participants": norm_parts, "evidence": ev, "confidence": conf})
    return items, keys

def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    u = len(a | b)
    i = len(a & b)
    return i / u if u else 0.0

def compare_chapter(query: CompareChapterQuery) -> Dict:
    settings = load_settings()
    p = resolve_paths(settings, query.book)
    base_dir = p["results_dir"]
    default_tpl = settings.get("prompts", {}).get("selected", "relations_plus")
    cid = query.chapter_id
    tpl_ids = query.templates
    syn = _merge_synonyms(settings, tpl_ids)
    alias = _collect_alias_rules(settings, tpl_ids)
    rel_counts: Dict[str, int] = {}
    evt_counts: Dict[str, int] = {}
    rel_sets: Dict[str, Set[str]] = {}
    evt_sets: Dict[str, Set[str]] = {}
    by_rel_type_all: Dict[str, Dict[str, int]] = {}
    for tpl in tpl_ids:
        files = _candidate_paths(base_dir, cid, default_tpl, tpl, settings, query.model_name)
        data = None
        for fp in files:
            data = _read_result(fp)
            if data:
                break
        rel_items, rel_keys, by_rel_type = _parse_relations(data or [], syn, alias, query.confidence_min)
        evt_items, evt_keys = _parse_events(data or {}, syn, alias, query.confidence_min) if query.include_events else ([], set())
        rel_counts[tpl] = len(rel_items)
        evt_counts[tpl] = len(evt_items)
        rel_sets[tpl] = rel_keys
        evt_sets[tpl] = evt_keys
        by_rel_type_all[tpl] = by_rel_type
    rel_union = set().union(*rel_sets.values()) if rel_sets else set()
    evt_union = set().union(*evt_sets.values()) if evt_sets else set()
    rel_inter = set.intersection(*rel_sets.values()) if rel_sets and len(rel_sets) > 1 else set()
    evt_inter = set.intersection(*evt_sets.values()) if evt_sets and len(evt_sets) > 1 else set()
    rel_j = 0.0
    evt_j = 0.0
    if len(tpl_ids) == 2:
        rel_j = _jaccard(rel_sets[tpl_ids[0]], rel_sets[tpl_ids[1]])
        evt_j = _jaccard(evt_sets[tpl_ids[0]], evt_sets[tpl_ids[1]])
    diffs_rel_unique: Dict[str, List[str]] = {}
    diffs_evt_unique: Dict[str, List[str]] = {}
    for tpl in tpl_ids:
        diffs_rel_unique[tpl] = list(rel_sets[tpl] - rel_inter)
        diffs_evt_unique[tpl] = list(evt_sets[tpl] - evt_inter)
    by_relation_type: Dict[str, Dict[str, int]] = {}
    types = set()
    for tpl in tpl_ids:
        types |= set(by_rel_type_all[tpl].keys())
    for t in types:
        by_relation_type[t] = {}
        for tpl in tpl_ids:
            by_relation_type[t][tpl] = by_rel_type_all[tpl].get(t, 0)
        by_relation_type[t]["union"] = sum([ by_rel_type_all[tpl].get(t, 0) for tpl in tpl_ids ])
        by_relation_type[t]["intersection"] = min([ by_rel_type_all[tpl].get(t, 0) for tpl in tpl_ids ]) if tpl_ids else 0
    samples_rel = [{"key": k} for k in list(rel_union)[:10]]
    samples_evt = [{"key": k} for k in list(evt_union)[:10]]
    meta = {
        "book": query.book,
        "chapter_id": cid,
        "templates": ",".join(tpl_ids),
        "model_name": query.model_name or "",
        "filters": json.dumps({"confidence_min": query.confidence_min, "include_events": query.include_events}, ensure_ascii=False),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }
    metrics = CompareMetrics(
        relations_count_by_template=rel_counts,
        events_count_by_template=evt_counts,
        relations_union_count=len(rel_union),
        relations_intersection_count=len(rel_inter),
        relations_jaccard=rel_j,
        events_union_count=len(evt_union),
        events_intersection_count=len(evt_inter),
        events_jaccard=evt_j,
        by_relation_type=by_relation_type
    ).dict()
    diffs = {
        "relations_unique": diffs_rel_unique,
        "relations_intersection": list(rel_inter),
        "events_unique": diffs_evt_unique,
        "events_intersection": list(evt_inter)
    }
    samples = {"relations": samples_rel, "events": samples_evt}
    return {"meta": meta, "metrics": metrics, "diffs": diffs, "samples": samples}

def compare_overview(book: str) -> Dict:
    settings = load_settings()
    p = resolve_paths(settings, book)
    base_dir = p["results_dir"]
    default_tpl = settings.get("prompts", {}).get("selected", "relations_plus")
    tpl_ids = [t.get("id") for t in settings.get("prompts", {}).get("templates", [])]
    files = [f for f in os.listdir(base_dir) if f.endswith(".json")]
    chapters = {}
    models = set()
    for name in files:
        m = None
        if name.startswith("result_"):
            if "__tpl-" in name:
                m = name.split("result_")[1].split("__tpl-")
                cid = m[0]
                rest = m[1]
                tpl = rest.split(".json")[0]
                if "__model-" in tpl:
                    tpl, mod = tpl.split("__model-")
                    models.add(mod)
                chapters.setdefault(cid, {"templates": set(), "files": []})
                chapters[cid]["templates"].add(tpl)
                chapters[cid]["files"].append(name)
            else:
                cid_full = name.split("result_")[1].split(".json")[0]
                cid = cid_full.split("__model-")[0]
                tpl = default_tpl
                chapters.setdefault(cid, {"templates": set(), "files": []})
                chapters[cid]["templates"].add(tpl)
                chapters[cid]["files"].append(name)
                if "__model-" in cid_full:
                    models.add(cid_full.split("__model-")[1])
    overview = {
        "meta": {"book": book, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
        "templates": tpl_ids,
        "models": sorted(list(models)),
        "chapters_total": len(chapters),
        "by_chapter": {cid: {"templates_present": sorted(list(info["templates"])), "files": info["files"]} for cid, info in chapters.items()}
    }
    return overview

def compare_batch(payload: CompareBatchQuery) -> Dict:
    settings = load_settings()
    p = resolve_paths(settings, payload.book)
    base_dir = p["results_dir"]
    default_tpl = settings.get("prompts", {}).get("selected", "relations_plus")
    ids = payload.chapter_ids or []
    if not ids and payload.chapter_range and len(payload.chapter_range) == 2:
        try:
            start = int(payload.chapter_range[0])
            end = int(payload.chapter_range[1])
            ids = [f"{i:03d}" for i in range(start, end + 1)]
        except:
            ids = []
    per = []
    rel_sum = {tpl: 0 for tpl in payload.templates}
    evt_sum = {tpl: 0 for tpl in payload.templates}
    for cid in ids:
        q = CompareChapterQuery(book=payload.book, chapter_id=cid, templates=payload.templates, model_name=payload.model_name, confidence_min=payload.confidence_min, include_events=payload.include_events)
        res = compare_chapter(q)
        per.append({"chapter_id": cid, "metrics": res["metrics"], "diffs": res["diffs"], "samples": res["samples"]})
        m = res["metrics"]
        for tpl, cnt in m["relations_count_by_template"].items():
            rel_sum[tpl] = rel_sum.get(tpl, 0) + cnt
        for tpl, cnt in m.get("events_count_by_template", {}).items():
            evt_sum[tpl] = evt_sum.get(tpl, 0) + cnt
    meta = {"book": payload.book, "chapter_ids": ids, "templates": ",".join(payload.templates), "model_name": payload.model_name or "", "filters": json.dumps({"confidence_min": payload.confidence_min, "include_events": payload.include_events}, ensure_ascii=False), "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    summary = {"relations_count_by_template": rel_sum, "events_count_by_template": evt_sum}
    return {"meta": meta, "summary": summary, "per_chapter": per}

class CompareBatchBooksQuery(BaseModel):
    books: List[str]
    chapter_ids: Optional[List[str]] = None
    chapter_range: Optional[List[str]] = None
    templates: List[str]
    model_name: Optional[str] = None
    confidence_min: float = 0.0
    include_events: bool = True

def compare_overview_multi(books: List[str]) -> Dict:
    res = {}
    for bk in books:
        res[bk] = compare_overview(bk)
    return {"books": res}

def compare_batch_books(payload: CompareBatchBooksQuery) -> Dict:
    books = payload.books or []
    out = {}
    rel_total = {tpl: 0 for tpl in payload.templates}
    evt_total = {tpl: 0 for tpl in payload.templates}
    for bk in books:
        q = CompareBatchQuery(book=bk, chapter_ids=payload.chapter_ids, chapter_range=payload.chapter_range, templates=payload.templates, model_name=payload.model_name, confidence_min=payload.confidence_min, include_events=payload.include_events)
        r = compare_batch(q)
        out[bk] = r
        s = r.get("summary", {})
        for tpl, cnt in (s.get("relations_count_by_template", {}) or {}).items():
            rel_total[tpl] = rel_total.get(tpl, 0) + cnt
        for tpl, cnt in (s.get("events_count_by_template", {}) or {}).items():
            evt_total[tpl] = evt_total.get(tpl, 0) + cnt
    meta = {"books": books, "templates": ",".join(payload.templates), "model_name": payload.model_name or "", "filters": json.dumps({"confidence_min": payload.confidence_min, "include_events": payload.include_events}, ensure_ascii=False), "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    grand = {"relations_count_by_template": rel_total, "events_count_by_template": evt_total}
    return {"meta": meta, "summary_total": grand, "per_book": out}
