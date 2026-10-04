#!/usr/bin/env python3
"""CPU-only real-tokenizer probe; does not qualify generated answer quality."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vllm_apple.rag import prepare_rag


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must be new")
    import transformers
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True,
                                             trust_remote_code=False)

    def count(request):
        return len(tokenizer.apply_chat_template(request["messages"], tokenize=True,
                                                 add_generation_prompt=True, return_dict=False))

    cases = []
    model_limit = json.loads((args.model / "config.json").read_text())["max_position_embeddings"]
    for language, question, text in (
        ("en", "Which languages are supported?", "English, Japanese and Simplified Chinese."),
        ("ja", "対応する言語は？", "英語、日本語、简体中文に対応します。"),
        ("zh", "支持哪些语言？", "支持英语、日语和简体中文。"),
    ):
        payload = dict(question=question, language=language,
                       documents=[dict(id="manual", title="Manual", text=text)])
        initial = prepare_rag(payload, system_role=False, max_tokens=128)
        limit = count(initial["request"]) + 128
        assert limit <= model_limit
        payload["documents"].append(dict(id="oversized", title="Long", text=text * 500))
        fitted = prepare_rag(payload, system_role=False, max_tokens=128,
                             max_source_bytes=262144, context_tokens=limit, token_counter=count)
        rejected = False
        try:
            prepare_rag(payload, system_role=False, max_tokens=128, context_tokens=128,
                        token_counter=count)
        except ValueError:
            rejected = True
        passed = (fitted["omitted_document_ids"] == ["oversized"]
                  and fitted["prompt_tokens"] + 128 == limit and rejected
                  and fitted["token_budget_verified"])
        cases.append(dict(language=language, passed=passed,
                          prompt_tokens=fitted["prompt_tokens"], context_tokens=limit,
                          omitted_document_ids=fitted["omitted_document_ids"],
                          base_overflow_rejected=rejected))
    files = [args.model / name for name in ("config.json", "tokenizer_config.json",
                                            "tokenizer.json", "special_tokens_map.json")]
    report = dict(schema_version=1, scope="CPU real tokenizer/context fitting only",
                  generation_quality_qualified=False, transformers=transformers.__version__,
                  model_context_limit=model_limit, source_sha256={
                      str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in (Path("vllm_apple/rag.py"), *files)},
                  cases=cases, passed=all(c["passed"] for c in cases))
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
