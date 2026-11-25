import zipfile

from speaker_transcriber.context import (
    build_profile,
    build_prompt,
    build_vocabulary,
    correct_text,
    extract_text_from_file,
)


def _make_project(tmp_path):
    proj = tmp_path / "phd"
    proj.mkdir()
    (proj / "thesis.md").write_text(
        "# Thesis\nWe study retrieval augmented generation with BERT fine-tuning. "
        "The budget report and the slides are due. The the the and of in.\n"
    )
    (proj / "notes.tex").write_text("\\section{Intro} We compare LLM baselines and RAG pipelines, RAG again.\n")
    (proj / "refs.bib").write_text("@article{x, title={Retrieval Augmented Generation}, author={Alice Bob}}\n")
    sub = proj / ".git"
    sub.mkdir()
    (sub / "ignored.txt").write_text("secretterm shouldnotappear\n")
    return proj


def test_build_vocabulary_extracts_terms_skips_stopwords_and_hidden(tmp_path):
    proj = _make_project(tmp_path)
    terms = build_vocabulary(proj, min_freq=1)
    low = [t.lower() for t in terms]
    assert "retrieval" in low
    assert "augmented" in low
    assert "bert" in low  # acronym, freq 1 is fine
    assert "llm" in low
    assert "rag" in low
    assert "budget" in low
    assert "the" not in low and "and" not in low
    assert "shouldnotappear" not in low
    # default min_freq=2 still keeps acronyms but drops hapax words
    default_terms = [t.lower() for t in build_vocabulary(proj)]
    assert "bert" in default_terms
    assert "budget" not in default_terms


def test_build_vocabulary_respects_max_terms(tmp_path):
    proj = tmp_path / "big"
    proj.mkdir()
    (proj / "a.txt").write_text(" ".join(f"term{i}" for i in range(500)) + " term0")
    terms = build_vocabulary(proj, max_terms=50)
    assert len(terms) <= 50


def test_build_prompt_contains_terms_and_is_capped():
    terms = [f"term{i}" for i in range(200)]
    prompt = build_prompt(terms, max_chars=200)
    assert "term0" in prompt
    assert len(prompt) <= 200


def test_extract_text_from_docx(tmp_path):
    docx = tmp_path / "doc.docx"
    with zipfile.ZipFile(docx, "w") as z:
        z.writestr("word/document.xml", "<w:p><w:r><w:t>Hello thesis world</w:t></w:r></w:p>")
    text = extract_text_from_file(docx)
    assert "Hello thesis world" in text


def test_correct_text_fixes_misheard_and_casing():
    terms = ["retrieval", "augmented", "generation", "budget", "BERT"]
    out = correct_text("we used retrieval augmented jenneration, with a big budget.", terms, threshold=0.85)
    assert "generation," in out  # fuzzy fix, punctuation preserved
    assert "budget." in out
    assert correct_text("the bert model", terms) == "the BERT model"
    assert correct_text("hello world", terms) == "hello world"
    # conservative default: does not rewrite near-miss common words
    assert correct_text("the model is good", ["models"]) == "the model is good"


def test_build_profile(tmp_path):
    proj = _make_project(tmp_path)
    profile = build_profile(proj)
    assert profile.source_dir == str(proj)
    assert profile.terms
    assert profile.prompt
    d = profile.to_dict()
    assert d["num_terms"] == len(profile.terms) and d["terms"] == profile.terms


def test_build_profile_missing_folder(tmp_path):
    try:
        build_profile(tmp_path / "nope")
    except ValueError:
        return
    raise AssertionError("expected ValueError")
