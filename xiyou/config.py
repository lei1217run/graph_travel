import os
import re

def _load_yaml(path):
    try:
        import yaml
    except Exception:
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return None

def _load_toml(path):
    try:
        import tomllib
    except Exception:
        return None
    try:
        with open(path, "rb") as f:
            return tomllib.load(f) or {}
    except Exception:
        return None

def load_settings():
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass
    cwd = os.getcwd()
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data = {}
    for name in ("settings.yaml", "settings.yml"):
        for root in (cwd, base_dir):
            p = os.path.join(root, name)
            if os.path.exists(p):
                y = _load_yaml(p)
                if y:
                    data.update(y)
                    break
        if data:
            break
    if not data:
        for root in (cwd, base_dir):
            p = os.path.join(root, "settings.toml")
            if os.path.exists(p):
                t = _load_toml(p)
                if t:
                    data.update(t)
                    break
    llm = data.get("llm", {})
    corpora = data.get("corpora", {})
    run = data.get("run", {})
    naming = data.get("naming", {})
    neo4j = data.get("neo4j", {})
    prompts = data.get("prompts", {})
    results = data.get("results", {})
    def _env_or(value: str | None, env_key: str, default: str | None = None, prefer_env: bool = True):
        env_val = os.environ.get(env_key)
        if prefer_env and env_val:
            return env_val
        if isinstance(value, str):
            m = re.fullmatch(r"\$\{([A-Z0-9_]+)\}", value.strip())
            if m:
                v = os.environ.get(m.group(1))
                if v:
                    return v
                return default
        return value if value is not None else default
    llm["api_key"] = _env_or(llm.get("api_key"), "LLM_API_KEY", default=None, prefer_env=True)
    llm["base_url"] = _env_or(llm.get("base_url"), "LLM_BASE_URL", default=llm.get("base_url"), prefer_env=False)
    llm["model_name"] = _env_or(llm.get("model_name"), "LLM_MODEL_NAME", default=llm.get("model_name", "qwen-plus"), prefer_env=False)
    corpora.setdefault("mode", "single")
    corpora.setdefault("base_dir", "data")
    corpora.setdefault("items", {})
    if corpora.get("mode") == "single" and not corpora.get("default"):
        items = corpora.get("items", {})
        if isinstance(items, dict) and items:
            corpora["default"] = next(iter(items.keys()))
    run.setdefault("limit_count", 3)
    run.setdefault("timeout_ms", 20000)
    naming.setdefault("mode", os.environ.get("NAMING_MODE", "TITLE_PREFIXED"))
    prompts.setdefault("selected", "relations_plus")
    selectors = prompts.get("selectors", {})
    selectors.setdefault("length_threshold", 8000)
    selectors.setdefault("dialogue_ratio_threshold", 0.50)
    prompts["selectors"] = selectors
    prompts.setdefault("selection_mode", "static")
    prompts.setdefault("dynamic", {"override_to": "events_relations"})
    templates = prompts.get("templates", [])
    prompts["templates"] = templates
    results.setdefault("use_model_suffix", False)
    results.setdefault("overwrite", False)
    results.setdefault("basename_template_id", "relations_plus")
    neo4j["uri"] = _env_or(neo4j.get("uri"), "NEO4J_URI", default=neo4j.get("uri"), prefer_env=False)
    neo4j["user"] = _env_or(neo4j.get("user"), "NEO4J_USER", default=neo4j.get("user"), prefer_env=False)
    neo4j["password"] = _env_or(neo4j.get("password"), "NEO4J_PASSWORD", default=None, prefer_env=True)
    return {"llm": llm, "corpora": corpora, "run": run, "naming": naming, "neo4j": neo4j, "prompts": prompts, "results": results}

def resolve_paths(cfg: dict, book_key: str) -> dict:
    corp = cfg.get("corpora", {})
    items = corp.get("items", {})
    base_dir = corp.get("base_dir") or "data"
    base_dir = base_dir if os.path.isabs(base_dir) else os.path.normpath(os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")), base_dir))
    info = items.get(book_key, {})
    folder = info.get("folder") or info.get("name") or book_key
    raw_dir = os.path.join(base_dir, folder)
    chapters_dir = raw_dir
    results_dir = os.path.join(raw_dir, "results")
    raw_file = os.path.join(raw_dir, info.get("raw_file") or f"{book_key}.txt")
    return {"raw_dir": raw_dir, "chapters_dir": chapters_dir, "results_dir": results_dir, "raw_file": raw_file}

def list_target_books(cfg: dict) -> list:
    corp = cfg.get("corpora", {})
    mode = corp.get("mode", "single")
    if mode == "single":
        d = corp.get("default")
        return [d] if d else []
    incl = corp.get("include")
    if incl:
        return incl
    items = corp.get("items", {})
    if isinstance(items, dict) and items:
        return list(items.keys())
    return []

