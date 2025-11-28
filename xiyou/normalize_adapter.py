import re
from typing import Any, Dict, List, Tuple

TYPE_ENUM = {"Person","Location","Item","Spell","Organization","Deity"}

def parse_alias_rules(rules: List[str]) -> Dict[str, List]:
    strip_tokens: List[str] = []
    regex_rules: List[Tuple[object, str]] = []
    for s in rules or []:
        s = str(s).strip()
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
                except:
                    pass
    return {"strip_tokens": strip_tokens, "regex_rules": regex_rules}

def canonical_original(name: str, alias: Dict[str, List]) -> str:
    n = str(name or "").strip()
    for pat, rep in alias.get("regex_rules", []):
        try:
            n = pat.sub(rep, n)
        except:
            pass
    return n.strip()

def canonical_norm(name: str, synonyms: Dict[str, str], alias: Dict[str, List]) -> str:
    n = str(name or "").strip()
    for pat, rep in alias.get("regex_rules", []):
        try:
            n = pat.sub(rep, n)
        except:
            pass
    n = synonyms.get(n, n)
    for tok in alias.get("strip_tokens", []):
        if n.startswith(tok):
            n = n[len(tok):]
        if n.endswith(tok):
            n = n[:-len(tok)]
    return n.strip()

def _relations_from_obj(obj: Any) -> List[Dict]:
    if isinstance(obj, dict):
        if "relations" in obj:
            return obj.get("relations") or []
        if "relation_with_meta" in obj:
            return obj.get("relation_with_meta") or []
    if isinstance(obj, list):
        return obj
    return []

def normalize_relations(obj: Any, syn: Dict[str, str], alias: Dict[str, List], alias_relations: List[str]) -> List[Dict]:
    rels = _relations_from_obj(obj)
    out: List[Dict] = []
    for r in rels or []:
        raw_head = r.get("head") or r.get("subject") or ""
        raw_tail = r.get("tail") or r.get("object") or ""
        raw_type = r.get("type") or r.get("relation") or ""
        if not raw_head or not raw_tail or not raw_type:
            continue
        rt = str(raw_type)
        if rt in set(alias_relations or []):
            ah = canonical_original(raw_head, alias)
            at = canonical_original(raw_tail, alias)
            alias_side = None
            canonical_side = None
            if at in syn:
                alias_side = at
                canonical_side = syn.get(at)
            elif ah in syn:
                alias_side = ah
                canonical_side = syn.get(ah)
            else:
                alias_side = at
                canonical_side = ah
            item = {
                "head": alias_side,
                "relation": "ALIAS",
                "tail": canonical_norm(canonical_side, syn, alias),
                "confidence": r.get("confidence"),
                "evidence": r.get("evidence"),
                "qualifiers": r.get("qualifiers") or {"alias_source": "evidence_trigger"},
            }
            out.append(item)
            continue
        h = canonical_norm(raw_head, syn, alias)
        t = canonical_norm(raw_tail, syn, alias)
        if h in TYPE_ENUM or t in TYPE_ENUM:
            continue
        item = {
            "head": h,
            "relation": str(rt),
            "tail": t,
            "confidence": r.get("confidence"),
            "evidence": r.get("evidence"),
            "qualifiers": r.get("qualifiers") or {},
        }
        out.append(item)
    return out

def normalize_events(obj: Any, syn: Dict[str, str], alias: Dict[str, List]) -> List[Dict]:
    evs = []
    if isinstance(obj, dict):
        evs = obj.get("events") or obj.get("event_plus_relation") or []
    out: List[Dict] = []
    for e in evs or []:
        et = str(e.get("event_type") or e.get("type") or "")
        tm = e.get("time")
        loc = e.get("location")
        parts = e.get("participants") or {}
        if isinstance(parts, list):
            d: Dict[str, List[str]] = {}
            for p in parts:
                role = str(p.get("role") or "")
                ent = canonical_norm(str(p.get("entity") or ""), syn, alias)
                if role:
                    d.setdefault(role, []).append(ent)
            parts = d
        else:
            d: Dict[str, List[str]] = {}
            for role, names in parts.items():
                d[role] = [ canonical_norm(str(x), syn, alias) for x in (names or []) ]
            parts = d
        out.append({"event_type": et, "time": tm, "location": loc, "participants": parts, "evidence": e.get("evidence"), "confidence": e.get("confidence")})
    return out

def collect_entities(relations: List[Dict], entities_hint: List[Dict], syn: Dict[str, str], alias: Dict[str, List]) -> List[Dict]:
    seen = set()
    out: List[Dict] = []
    for r in relations or []:
        h = r["head"]
        t = r["tail"]
        if h and h not in seen:
            seen.add(h); out.append({"name": h})
        if t and t not in seen:
            seen.add(t); out.append({"name": t})
    for e in entities_hint or []:
        n = canonical_norm(e.get("name",""), syn, alias)
        if n and n not in seen:
            seen.add(n); out.append({"name": n})
    return out

def normalize_output(raw: Any, tpl_id: str, synonyms_map: Dict[str, str], alias_rules: List[str], alias_relations: List[str] = None) -> Dict:
    alias = parse_alias_rules(alias_rules or [])
    relations = normalize_relations(raw, synonyms_map or {}, alias, alias_relations or [])
    entities_hint = raw.get("entities") if isinstance(raw, dict) else []
    events = normalize_events(raw, synonyms_map or {}, alias)
    entities = collect_entities(relations, entities_hint or [], synonyms_map or {}, alias)
    meta = raw.get("meta") if isinstance(raw, dict) else {}
    extra_alias: List[Dict] = []
    for a, c in (synonyms_map or {}).items():
        extra_alias.append({"head": a, "relation": "ALIAS", "tail": c, "confidence": 0.99, "evidence": None, "qualifiers": {"alias_source": "synonyms_map"}})
    relations = relations + extra_alias
    return {"entities": entities, "relations": relations, "events": events, "meta": meta}

def calibrate_relations(relations: List[Dict], settings: Dict) -> List[Dict]:
    cfg = settings.get("relations", {})
    patterns = cfg.get("patterns", [])
    allowed = set(cfg.get("allowed", []))
    out: List[Dict] = []
    for r in relations or []:
        ev = str(r.get("evidence") or "")
        added = False
        for p in patterns:
            rel = p.get("relation")
            inc = p.get("include") or []
            exc = p.get("exclude") or []
            if inc and any(k in ev for k in inc) and not any(x in ev for x in exc):
                nr = dict(r)
                nr["relation"] = rel
                out.append(nr)
                added = True
        if not added:
            out.append(r)
    if allowed:
        out = [x for x in out if str(x.get("relation")) in allowed or not allowed]
    return out

def fuse_relations(relations: List[Dict], settings: Dict) -> List[Dict]:
    fusion = settings.get("fusion", {})
    group_by = fusion.get("group_by", "head_tail")
    evidence_merge = fusion.get("evidence_merge", "union")
    confidence_merge = fusion.get("confidence_merge", "max")
    chapter_strategy = fusion.get("chapter_strategy", "first")
    precedence = settings.get("relations", {}).get("precedence", [])

    groups: Dict[str, List[Dict]] = {}
    for r in relations or []:
        if group_by == "head_tail":
            k = f"{r.get('head')}|{r.get('tail')}"
        else:
            key_fmt = fusion.get("key_format", "{head}|{tail}|{relation}")
            k = key_fmt.format(head=r.get("head"), tail=r.get("tail"), relation=r.get("relation"))
        groups.setdefault(k, []).append(r)

    out: List[Dict] = []
    for k, items in groups.items():
        # 选择唯一关系：按 precedence 优先级，其次按 confidence 最大
        def rel_rank(x: Dict) -> tuple:
            rel = x.get("relation")
            pri = precedence.index(rel) if rel in precedence else len(precedence)
            conf = x.get("confidence") or 0.0
            return (pri, -conf)
        winner = sorted(items, key=rel_rank)[0]
        head = winner.get("head")
        tail = winner.get("tail")
        relation = winner.get("relation")
        # 合并证据
        ev_set = set()
        for it in items:
            ev = it.get("evidence")
            if evidence_merge == "union" and isinstance(ev, str):
                ev_set.add(ev)
        evidence = " | ".join(sorted(ev_set))
        # 合并置信度
        confidences = [it.get("confidence") or 0.0 for it in items]
        confidence = max(confidences) if confidence_merge == "max" else (sum(confidences)/len(confidences) if confidences else 0.0)
        # 合并章节
        chapters = [it.get("chapter_id") for it in items if it.get("chapter_id")]
        chapter_id = chapters[0] if (chapter_strategy == "first" and chapters) else None
        # 结果
        out.append({
            "head": head,
            "tail": tail,
            "relation": relation,
            "confidence": confidence,
            "evidence": evidence,
            "qualifiers": {"chapters": chapters},
            "chapter_id": chapter_id,
        })
    return out
