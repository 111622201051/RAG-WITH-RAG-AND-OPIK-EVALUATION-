import streamlit as st, os, shutil, warnings, json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from datasets import Dataset
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")
load_dotenv(dotenv_path=Path(__file__).parent.parent / '.env')

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_community.vectorstores import FAISS
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

# ── Opik Integration ──────────────────────────────────────────────────────────
import opik
from opik import Opik
from opik.evaluation import evaluate as opik_evaluate
from opik.evaluation.metrics import (
    Hallucination, AnswerRelevance, ContextPrecision, Moderation
)

# Configure Opik — reads OPIK_API_KEY & OPIK_WORKSPACE from .env automatically
# You can also call: opik.configure(api_key="YOUR_KEY", workspace="YOUR_WORKSPACE")
opik.configure(use_local=False)   # set use_local=True if self-hosting Opik

OPIK_PROJECT = os.getenv("OPIK_PROJECT_NAME", "RAG-Eval-App")

def log_to_opik(eval_data: list[dict], run_label: str = "eval-run"):
    try:
        import opik
        from opik import Opik
        from opik.integrations.langchain import OpikTracer

        client = Opik()
        ds_name = f"{OPIK_PROJECT}-{run_label}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        opik_ds = client.get_or_create_dataset(ds_name)

        # Insert items into Opik dataset
        items = []
        for d in eval_data:
            ctx = d["contexts"] if isinstance(d["contexts"], str) else "\n".join(d["contexts"])
            items.append({
                "input":        d["question"],
                "output":       d["answer"],
                "context":      ctx,
                "ground_truth": d.get("ground_truth", ""),
            })
        opik_ds.insert(items)

        # Log each Q&A as a TRACE (this gives you token cost tracking!)
        for item in items:
            tracer = OpikTracer(project_name=OPIK_PROJECT)
            trace = client.trace(
                name        = "rag-qa",
                project_name= OPIK_PROJECT,
                input       = {"question": item["input"]},
                output      = {"answer":   item["output"]},
                metadata    = {"context":  item["context"][:500],
                               "ground_truth": item["ground_truth"]},
            )
            trace.end()

        st.success(f"✅ {len(items)} traces logged to Opik!")
        return client

    except Exception as e:
        st.warning(f"⚠️ Opik logging failed: {e}")
        return None

        # Define a simple task function for Opik evaluation
        def rag_task(item: dict) -> dict:
            return {
                "input":   item["input"],
                "output":  next(
                    (d["answer"] for d in eval_data if d["question"] == item["input"]),
                    ""
                ),
                "context": item["context"],
            }

        # Metrics to compute inside Opik
        metrics = [
            Hallucination(),
            AnswerRelevance(),
            Moderation(),
        ]
        # ContextPrecision needs a reference — add only if ground truths exist
        if any(d.get("ground_truth", "").strip() for d in eval_data):
            metrics.append(ContextPrecision())

        experiment = opik_evaluate(
            dataset          = opik_ds,
            task             = rag_task,
            scoring_metrics  = metrics,
            experiment_name  = ds_name,
            project_name     = OPIK_PROJECT,
            nb_samples       = len(items),
        )
        return experiment
    except Exception as e:
        st.warning(f"⚠️ Opik logging failed: {e}")
        return None

# ── Original App Setup ────────────────────────────────────────────────────────
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

AZ_EP  = os.getenv("AZURE_OPENAI_ENDPOINT")
AZ_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZ_DEP = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
AZ_VER = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")

DEFAULTS = dict(
    messages=[], chat_eval_data=[], auto_eval_data=[],
    chat_ground_truths=[], ragas_results=None,
    trulens_results=None,
    trust_history=[], retriever=None,
    opik_experiment_url=None,      # ← NEW: stores last Opik experiment URL
)
for k, v in DEFAULTS.items():
    if k not in st.session_state: st.session_state[k] = v

def get_all_eval_data():
    return st.session_state.chat_eval_data + st.session_state.auto_eval_data

# ── Cached Resources ──────────────────────────────────────────────────────────
@st.cache_resource
def get_llm():
    return AzureChatOpenAI(deployment_name=AZ_DEP, azure_endpoint=AZ_EP,
        api_key=AZ_KEY, openai_api_version=AZ_VER, temperature=0, max_retries=2, timeout=25)

@st.cache_resource
def get_emb():
    return AzureOpenAIEmbeddings(deployment="text-embedding-ada-002",
        azure_endpoint=AZ_EP, api_key=AZ_KEY, openai_api_version=AZ_VER, max_retries=2)

@st.cache_resource
def build_retriever(file_key):
    docs = []
    for path, ext in file_key:
        loaders = {".pdf": PyPDFLoader, ".docx": Docx2txtLoader, ".txt": TextLoader}
        if ext in loaders: docs.extend(loaders[ext](path).load())
    texts = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50).split_documents(docs)
    return FAISS.from_documents(texts, get_emb()).as_retriever(search_kwargs={"k": 3})

# ── Firewall ──────────────────────────────────────────────────────────────────
SAFE_TERMS = ["resume","cv","whose","who is","who's","name","cgpa","gpa","grade",
              "college","university","project","skill","experience","certification",
              "education","achievement","internship","work","profile","summary",
              "what","when","where","how","which","describe","explain","list"]
UNSAFE_TERMS = ["how to make a bomb","make explosives","build a weapon","create malware",
                "write ransomware","hack into","exploit vulnerability","ddos attack",
                "phishing email","poison someone","kill someone","child porn",
                "bypass security","jailbreak","ignore your instructions",
                "ignore previous instructions","act as dan","override system prompt"]

def firewall_check(text):
    low = text.lower().strip()
    if any(s in low for s in SAFE_TERMS):    return True, ""
    if any(u in low for u in UNSAFE_TERMS):
        return False, "🚨 Blocked: This request is not permitted. Please ask document questions."
    if len(low.split()) <= 6:                return True, ""
    try:
        r = get_llm().invoke(
            "Does this ask for weapons, malware, or real-world illegal harm? "
            "Answer YES or NO only. Document/resume/education questions = NO.\n"
            f"Input: {text}"
        )
        bad = r.content.strip().upper().startswith("YES")
        return not bad, ("🚨 Blocked." if bad else "")
    except:
        return True, ""

# ── Core Q&A ──────────────────────────────────────────────────────────────────
def fmt_docs(docs): return "\n\n".join(d.page_content for d in docs)

def ask(question, retriever, history=""):
    from opik.integrations.langchain import OpikTracer

    docs   = retriever.invoke(question)
    ctx    = fmt_docs(docs)
    hist   = f"Conversation so far:\n{history}\n\n" if history else ""
    prompt = (f"{hist}Answer ONLY from context. Be concise (2-3 sentences). "
              f"If not found, say 'Not found in document.'\n\n"
              f"Context:\n{ctx}\n\nQuestion: {question}\nAnswer:")

    # ← Opik tracer tracks tokens + cost automatically
    tracer = OpikTracer(project_name=OPIK_PROJECT)
    ans    = get_llm().invoke(prompt, config={"callbacks": [tracer]}).content.strip()
    return ans, ctx, docs

# ── Auto Question Generation ──────────────────────────────────────────────────
def generate_qa(retriever, n, mode):
    ctx = fmt_docs(retriever.invoke("overview summary introduction"))
    llm = get_llm()
    results = []

    if mode == "single":
        raw = llm.invoke(
            f"Generate exactly {n} completely independent questions on DIFFERENT topics.\n"
            f"Vary types: factual, conceptual, comparative, descriptive.\n"
            f"Return ONLY a numbered list.\nContext: {ctx[:2500]}"
        ).content.strip()
        questions = [l.strip().lstrip("0123456789.)- ").strip()
                     for l in raw.split("\n") if len(l.strip()) > 10][:n]
        if not questions:
            st.error("Could not generate questions — try a longer document."); return []

        prog = st.progress(0, f"Answering {len(questions)} questions in parallel…")
        def answer_worker(idx_q):
            idx, q = idx_q
            a, c, _ = ask(q, retriever)
            return idx, q, a, c
        raw_res = []
        with ThreadPoolExecutor(max_workers=3) as ex:
            futs = {ex.submit(answer_worker, (i, q)): i for i, q in enumerate(questions)}
            done = 0
            for f in as_completed(futs):
                raw_res.append(f.result())
                done += 1
                prog.progress(done / len(questions), f"Answered {done}/{len(questions)}")
        prog.empty()
        results = [(q, a, c) for _, q, a, c in sorted(raw_res, key=lambda x: x[0])]

    else:  # multi-turn
        first_q = llm.invoke(
            f"Generate ONE clear opening question about this document. Return ONLY the question.\n"
            f"Context:\n{ctx[:1200]}"
        ).content.strip().lstrip("0123456789.)- ")
        history = ""
        prog = st.progress(0, f"Building {n}-turn conversation…")
        for i in range(n):
            q = first_q if i == 0 else llm.invoke(
                f"Based on this conversation, write the NEXT logical follow-up question.\n"
                f"Must build on the last answer. Return ONLY the question.\n\n"
                f"Conversation:\n{history}"
            ).content.strip().lstrip("0123456789.)- ")
            a, c, _ = ask(q, retriever, history)
            results.append((q, a, c))
            history += f"Q{i+1}: {q}\nA{i+1}: {a}\n\n"
            prog.progress((i+1)/n, f"Turn {i+1}/{n}: {q[:50]}…")
        prog.empty()
    return results

# ── RAGAS ─────────────────────────────────────────────────────────────────────
def run_ragas():
    data = get_all_eval_data()
    if not data: st.warning("No Q&A data yet!"); return None
    llm_w = LangchainLLMWrapper(AzureChatOpenAI(deployment_name=AZ_DEP, azure_endpoint=AZ_EP,
        api_key=AZ_KEY, openai_api_version=AZ_VER, temperature=0))
    emb_w = LangchainEmbeddingsWrapper(AzureOpenAIEmbeddings(deployment="text-embedding-ada-002",
        azure_endpoint=AZ_EP, api_key=AZ_KEY, openai_api_version=AZ_VER))
    use_gt  = any(g.strip() for g in st.session_state.chat_ground_truths)
    metrics = [faithfulness, answer_relevancy]
    if use_gt:
        from ragas.metrics import context_precision, context_recall, answer_correctness
        metrics += [context_precision, context_recall, answer_correctness]
    for m in metrics:
        if hasattr(m, 'llm'): m.llm = llm_w
        if hasattr(m, 'embeddings'): m.embeddings = emb_w
    ctxs = [[c] if isinstance(c, str) else c for c in [e["contexts"] for e in data]]
    ds = {"question": [e["question"] for e in data],
          "answer":   [e["answer"]   for e in data], "contexts": ctxs}
    if use_gt:
        gts = list(st.session_state.chat_ground_truths) + [""] * len(st.session_state.auto_eval_data)
        ds["ground_truth"] = (gts + [""] * len(data))[:len(data)]
    try:
        with st.spinner("Running RAGAS evaluation…"):
            return evaluate(Dataset.from_dict(ds), metrics=metrics)
    except Exception as e:
        st.error(f"RAGAS error: {e}"); return None

# ── TruLens (direct LLM scoring) ─────────────────────────────────────────────
SCORE_PROMPT = """\
Score this RAG output. Return ONLY valid JSON, no explanation.

Question: {question}
Context: {context}
Answer: {answer}

- groundedness (0-1): Every claim in Answer is supported by Context?
- context_relevance (0-1): Context is relevant to the Question?
- answer_relevance (0-1): Answer directly addresses the Question?

JSON only: {{"groundedness":<float>,"context_relevance":<float>,"answer_relevance":<float>}}"""

def _score_item(item):
    q   = item["question"]
    ans = item["answer"]
    ctx = (item["contexts"] if isinstance(item["contexts"], str)
           else "\n".join(item["contexts"]))[:1200]
    try:
        raw = get_llm().invoke(SCORE_PROMPT.format(question=q, context=ctx, answer=ans)).content.strip()
        s = raw.find("{"); e = raw.rfind("}") + 1
        sc = json.loads(raw[s:e]) if s != -1 and e > s else {}
        clamp = lambda v: round(min(1.0, max(0.0, float(v))), 3)
        return {"Question": q, "Answer": ans,
                "Groundedness":      clamp(sc.get("groundedness",      0)),
                "Context Relevance": clamp(sc.get("context_relevance", 0)),
                "Answer Relevance":  clamp(sc.get("answer_relevance",  0))}
    except:
        return {"Question": q, "Answer": ans,
                "Groundedness": 0.0, "Context Relevance": 0.0, "Answer Relevance": 0.0}

def run_trulens():
    data = get_all_eval_data()
    if not data: st.warning("No Q&A data yet!"); return None
    prog    = st.progress(0, "TruLens scoring…")
    results = [None] * len(data)
    def score_worker(i_item):
        i, item = i_item
        return i, _score_item(item)
    with ThreadPoolExecutor(max_workers=3) as ex:
        futs = {ex.submit(score_worker, (i, item)): i for i, item in enumerate(data)}
        done = 0
        for f in as_completed(futs):
            i, row = f.result()
            results[i] = row
            done += 1
            prog.progress(done / len(data), f"Scored {done}/{len(data)}")
    prog.empty()
    return pd.DataFrame(results)

# ── Display Functions ─────────────────────────────────────────────────────────
def show_ragas(result):
    if not result: st.info("▶️ Run Evaluation (sidebar) to see RAGAS results."); return
    df   = result.to_pandas()
    nums = df.select_dtypes(include=["number"]).mean().to_dict()
    cols = st.columns(min(4, len(nums)))
    for i, (k, v) in enumerate(nums.items()):
        cols[i%len(cols)].metric(
            f"{'🟢' if v>=.8 else '🟡' if v>=.6 else '🔴'} {k.replace('_',' ').title()}", f"{v:.3f}")
    w  = {"faithfulness":.35,"answer_relevancy":.25,"context_precision":.20,"context_recall":.20}
    ts = round(sum(nums.get(m,0)*wt for m,wt in w.items()), 3)
    c1, c2 = st.columns(2)
    c1.metric("🛡 Trust Score", ts)
    c2.metric("⚠️ Hallucination Risk", round(1 - nums.get("faithfulness", 1), 3))
    st.session_state.trust_history.append(ts)
    t1, t2 = st.tabs(["📊 Bar", "🎯 Radar"])
    with t1:
        fig = go.Figure(go.Bar(x=list(nums), y=list(nums.values()),
            text=[f"{v:.3f}" for v in nums.values()], textposition="auto",
            marker_color=["green" if v>=.8 else "orange" if v>=.6 else "red" for v in nums.values()]))
        fig.update_layout(yaxis_range=[0,1], height=280)
        st.plotly_chart(fig, use_container_width=True)
    with t2:
        vals, cats = list(nums.values()), list(nums)
        fig = go.Figure(go.Scatterpolar(r=vals+vals[:1], theta=cats+cats[:1], fill="toself"))
        fig.update_layout(polar=dict(radialaxis=dict(range=[0,1])), height=280)
        st.plotly_chart(fig, use_container_width=True)
    if len(st.session_state.trust_history) > 1:
        tdf = pd.DataFrame({"Run": range(1, len(st.session_state.trust_history)+1),
                            "Trust Score": st.session_state.trust_history})
        st.plotly_chart(px.line(tdf, x="Run", y="Trust Score", markers=True), use_container_width=True)

def show_trulens(df):
    if df is None or df.empty: st.info("▶️ Run Evaluation (sidebar) to see TruLens results."); return
    sc = [c for c in ["Groundedness","Context Relevance","Answer Relevance"] if c in df.columns]
    with st.expander("📋 Per-Question Results", expanded=True):
        st.dataframe(df[[c for c in ["Question","Answer"]+sc if c in df.columns]].round(3),
                     use_container_width=True)
    avg = df[sc].mean()
    cols = st.columns(len(avg))
    for i, (k, v) in enumerate(avg.items()):
        cols[i].metric(f"{'🟢' if v>=.8 else '🟡' if v>=.6 else '🔴'} {k}", f"{v:.3f}")
    st.metric("🧩 Composite Score", round(avg.mean(),3),
              delta="🎯 Target met" if avg.mean()>=.75 else "⚠️ Below target")
    cdf = pd.DataFrame({"Metric": avg.index.tolist(), "Score": avg.values.tolist()})
    fig = px.bar(cdf, x="Metric", y="Score", text="Score", color="Score",
                 color_continuous_scale="RdYlGn", range_y=[0,1])
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    st.plotly_chart(fig, use_container_width=True)
    if len(df) > 1:
        mdf = df[sc].reset_index().melt(id_vars="index", var_name="Metric", value_name="Score")
        mdf["Q"] = mdf["index"].apply(lambda x: f"Q{x+1}")
        st.plotly_chart(px.line(mdf, x="Q", y="Score", color="Metric", markers=True,
            range_y=[0,1], title="Per-Question Score Trends"), use_container_width=True)

def show_compare():
    rs, ts = {}, {}
    if st.session_state.ragas_results:
        rs = st.session_state.ragas_results.to_pandas().select_dtypes(include=["number"]).mean().to_dict()
    if st.session_state.trulens_results is not None and not st.session_state.trulens_results.empty:
        sc = [c for c in ["Groundedness","Context Relevance","Answer Relevance"]
              if c in st.session_state.trulens_results.columns]
        ts = st.session_state.trulens_results[sc].mean().to_dict() if sc else {}

    if not rs and not ts:
        st.info("Run at least two evaluations to compare."); return

    keys = list(set(list(rs)+list(ts)))
    cdf  = pd.DataFrame({"Metric": keys,
                          "RAGAS":   [rs.get(k)  for k in keys],
                          "TruLens": [ts.get(k)  for k in keys]})
    st.dataframe(cdf.round(3), use_container_width=True)
    fig = go.Figure([
        go.Bar(name="RAGAS",   x=cdf["Metric"], y=cdf["RAGAS"]),
        go.Bar(name="TruLens", x=cdf["Metric"], y=cdf["TruLens"]),
    ])
    fig.update_layout(barmode="group", yaxis_range=[0,1], title="Framework Comparison: RAGAS vs TruLens")
    st.plotly_chart(fig, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════════
# UI
# ════════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="RAG Eval", page_icon="🧠", layout="wide")
st.title("🧠 Document Q&A + RAGAS, TruLens & Opik Evaluation")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    enable_ragas   = st.checkbox("🔍 RAGAS",   value=True)
    enable_trulens = st.checkbox("🧪 TruLens", value=True)
    enable_opik    = st.checkbox("🌐 Opik (cloud)", value=True)   # ← NEW

    st.divider()
    use_gt = st.checkbox("✏️ Ground Truth (chat Q only)", value=False)
    st.divider()

    total = len(get_all_eval_data())
    st.caption(f"📊 Chat: {len(st.session_state.chat_eval_data)} | Auto: {len(st.session_state.auto_eval_data)} | Total: {total}")

    if st.button("▶️ Run Evaluation", type="primary", use_container_width=True):
        if not get_all_eval_data():
            st.warning("Ask questions or auto-generate first!")
        else:
            if enable_ragas:
                st.session_state.ragas_results   = run_ragas()
            if enable_trulens:
                st.session_state.trulens_results = run_trulens()

            # ── Opik: send ALL eval data to comet.com/opik ──────────────────
            if enable_opik:
                with st.spinner("📡 Sending results to Opik platform…"):
                    data = get_all_eval_data()

                    # Attach ground truths if available
                    enriched = []
                    for i, d in enumerate(data):
                        entry = dict(d)
                        if i < len(st.session_state.chat_ground_truths):
                            entry["ground_truth"] = st.session_state.chat_ground_truths[i]
                        enriched.append(entry)

                    experiment = log_to_opik(enriched, run_label="streamlit")
                    if experiment:
                        # Try to extract a URL from the experiment object
                        exp_url = getattr(experiment, "url", None) or \
                                  getattr(experiment, "experiment_url", None)
                        if exp_url:
                            st.session_state.opik_experiment_url = exp_url
                        st.success("✅ Evaluation logged to Opik!")
                    # ─────────────────────────────────────────────────────────

            st.rerun()

    # Show Opik link if available
    if st.session_state.opik_experiment_url:
        st.markdown(f"[🔗 View in Opik →]({st.session_state.opik_experiment_url})")

    if st.button("🗑️ Clear All", use_container_width=True):
        for k, v in DEFAULTS.items(): st.session_state[k] = v
        shutil.rmtree(UPLOAD_FOLDER, ignore_errors=True)
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        build_retriever.clear(); st.rerun()

    # Ground Truth — ONLY chat questions
    if use_gt and st.session_state.chat_eval_data:
        st.divider()
        st.markdown("### 🎯 Ground Truths")
        st.caption("Only manually asked chat questions appear here.")
        chat_data = st.session_state.chat_eval_data
        total_c   = len(chat_data)
        while len(st.session_state.chat_ground_truths) < total_c:
            st.session_state.chat_ground_truths.append("")
        done = sum(1 for g in st.session_state.chat_ground_truths if g.strip())
        st.progress(done/total_c, f"{done}/{total_c} provided")
        for i, item in enumerate(chat_data):
            cur = st.session_state.chat_ground_truths[i]
            with st.expander(f"{'✅' if cur.strip() else '❌'} Q{i+1}: {item['question'][:40]}…"):
                st.caption(f"🤖 {item['answer'][:80]}")
                val = st.text_area("Ground Truth", value=cur, key=f"gt_{i}", height=60)
                if st.button("💾 Save", key=f"sv_{i}"):
                    st.session_state.chat_ground_truths[i] = val.strip()
                    st.toast(f"Q{i+1} saved!")

# ── Main Tabs ─────────────────────────────────────────────────────────────────
st.markdown("---")
tab_ragas, tab_tru, tab_cmp, tab_gen = st.tabs(
    ["🔍 RAGAS", "🧪 TruLens", "⚖️ Compare", "🤖 Auto Generate"])

with tab_ragas:
    show_ragas(st.session_state.ragas_results)

with tab_tru:
    show_trulens(st.session_state.trulens_results)

with tab_cmp:
    show_compare()
    c1, c2 = st.columns(2)
    with c1:
        if st.session_state.ragas_results:
            st.download_button("📊 RAGAS CSV",
                st.session_state.ragas_results.to_pandas().to_csv(index=False), "ragas.csv")
    with c2:
        if st.session_state.trulens_results is not None and not st.session_state.trulens_results.empty:
            st.download_button("🧪 TruLens CSV",
                st.session_state.trulens_results.to_csv(index=False), "trulens.csv", "text/csv")

with tab_gen:
    st.markdown("### 🤖 Automatic Question Generation")
    st.caption("Auto Q&A is used for evaluation only — never appears in the Ground Truth sidebar.")
    if not st.session_state.retriever:
        st.info("⬆️ Upload a document below first.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### 🔵 Single Turn")
            st.caption("Each question is **independent** — different topic, no relation to each other.")
            sn = st.number_input("Number of questions", 1, 30, 5, key="sn")
            if st.button("▶️ Generate Single Turn", key="gs", use_container_width=True):
                with st.spinner(f"Generating {sn} independent questions…"):
                    res = generate_qa(st.session_state.retriever, int(sn), "single")
                if res:
                    for q, a, c in res:
                        st.session_state.auto_eval_data.append({"question": q, "answer": a, "contexts": c})
                    st.success(f"✅ {len(res)} Q&A pairs added!")
                    st.dataframe(pd.DataFrame([{"#": i+1, "Question": r[0], "Answer": r[1][:100]+"…"}
                        for i, r in enumerate(res)]), use_container_width=True, hide_index=True)
        with c2:
            st.markdown("#### 🟣 Multi Turn")
            st.caption("Each question **builds on the previous answer** — connected conversation chain.")
            mn = st.number_input("Number of turns", 2, 20, 5, key="mn")
            if st.button("▶️ Generate Multi Turn", key="gm", use_container_width=True):
                with st.spinner(f"Building {mn}-turn conversation…"):
                    res = generate_qa(st.session_state.retriever, int(mn), "multi")
                if res:
                    for q, a, c in res:
                        st.session_state.auto_eval_data.append({"question": q, "answer": a, "contexts": c})
                    st.success(f"✅ {len(res)}-turn chain added!")
                    st.dataframe(pd.DataFrame([{"Turn": i+1, "Question": r[0], "Answer": r[1][:100]+"…"}
                        for i, r in enumerate(res)]), use_container_width=True, hide_index=True)

        if st.session_state.auto_eval_data:
            st.markdown(f"---\n**📋 Auto-generated: {len(st.session_state.auto_eval_data)} Q&A pairs → click ▶️ Run Evaluation in sidebar**")
            with st.expander("View auto-generated entries"):
                st.dataframe(pd.DataFrame([
                    {"#": i+1, "Q": d["question"], "A": d["answer"][:80]+"…"}
                    for i, d in enumerate(st.session_state.auto_eval_data)
                ]), use_container_width=True, hide_index=True)
            if st.button("🗑️ Clear Auto-Generated Only", key="clr_auto"):
                st.session_state.auto_eval_data = []; st.rerun()

# ── Chat ──────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 💬 Document Q&A Chat")
files = st.file_uploader("📁 Upload PDF / DOCX / TXT", type=["pdf","docx","txt"], accept_multiple_files=True)

if files:
    saved = []
    for f in files:
        path = os.path.join(UPLOAD_FOLDER, f.name)
        with open(path, "wb") as fp: fp.write(f.getbuffer())
        saved.append((path, os.path.splitext(f.name)[-1].lower()))
    retriever = build_retriever(tuple(saved))
    st.session_state.retriever = retriever
    st.success(f"✅ {len(files)} document(s) ready!")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]): st.markdown(msg["content"])

    if prompt := st.chat_input("Ask anything about your document…"):
        is_safe, reason = firewall_check(prompt)
        if not is_safe:
            st.warning(reason)
        else:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"): st.markdown(prompt)
            try:
                ans, ctx, docs = ask(prompt, retriever)
                with st.chat_message("assistant"):
                    st.markdown(ans)
                    for src in set(d.metadata.get("source","?") for d in docs):
                        st.caption(f"📄 {os.path.basename(src)}")
                st.session_state.messages.append({"role": "assistant", "content": ans})
                st.session_state.chat_eval_data.append({"question": prompt, "answer": ans, "contexts": ctx})
                while len(st.session_state.chat_ground_truths) < len(st.session_state.chat_eval_data):
                    st.session_state.chat_ground_truths.append("")
                st.toast("📝 Saved for evaluation")
            except Exception as e:
                st.error(f"Error: {e}")

st.caption(
    "pip install streamlit langchain-openai langchain-community ragas datasets "
    "plotly python-dotenv opik"
)