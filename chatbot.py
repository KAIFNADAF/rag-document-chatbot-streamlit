import os
import tempfile
import streamlit as st
import re
from dataclasses import dataclass
import json
from datetime import datetime
import concurrent.futures
import time
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    Docx2txtLoader,
    UnstructuredFileLoader,
    DirectoryLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="wide")

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.subheader("How to use")
st.sidebar.info("Upload → Index → Ask questions. Switch modes for announcements.")

st.sidebar.subheader("System")
if st.session_state.get("indexed"):
    st.sidebar.success("Ready ✅ (indexed)")
else:
    st.sidebar.warning("Not indexed yet")

st.sidebar.write(f"Docs: {st.session_state.get('doc_count', 0)}")
st.sidebar.write(f"Chunks: {st.session_state.get('chunk_count', 0)}")

st.sidebar.divider()
mode = st.sidebar.radio(
    "Mode",
    ["Chat", "Announcement Generator", "Segmented Distribution"],
    index=0,
)

st.sidebar.caption("Note: Segmented mode can be slow on local models—use fewer segments/variants.")
if st.sidebar.button("Reset (clear index + chat)"):
    st.session_state.chat_history = []
    st.session_state.indexed = False
    st.session_state.vector_store = None
    st.session_state.doc_count = 0
    st.session_state.chunk_count = 0
    st.experimental_rerun()

# -----------------------------
# Main title
# -----------------------------
st.title("RAG-Based Chatbot with File Uploads")

status = st.container(border=True)
with status:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mode", mode)
    c2.metric("Indexed", "✅ Yes" if st.session_state.get("indexed") else "❌ No")
    c3.metric("Docs", st.session_state.get("doc_count", 0))
    c4.metric("Chunks", st.session_state.get("chunk_count", 0))

# -----------------------------
# Session state (must be BEFORE generator st.stop)
# -----------------------------
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "indexed" not in st.session_state:
    st.session_state.indexed = False
if "doc_count" not in st.session_state:
    st.session_state.doc_count = 0
if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0

# -----------------------------
# Helpers
# -----------------------------
def get_loader(file_path: str):
    extension = os.path.splitext(file_path)[1].lower()
    if extension == ".pdf":
        return PyPDFLoader(file_path, extract_images=False)
    elif extension == ".txt":
        return TextLoader(file_path, encoding="utf-8")
    elif extension == ".csv":
        return CSVLoader(file_path)
    elif extension == ".docx":
        return Docx2txtLoader(file_path)
    else:
        return UnstructuredFileLoader(file_path, mode="elements", strategy="fast")

def extract_section(text: str, header: str, next_headers=None) -> str:
    """
    Extracts a section from the model output based on headers like:
    'LONG VERSION:' ... until next header.

    Returns empty string if not found.
    """
    if next_headers is None:
        next_headers = ["SUBJECT:", "LONG VERSION:", "SHORT VERSION:", "SIGNAGE LINE:"]

    # Normalize common formatting variants (e.g., **LONG VERSION:**)
    norm = text.replace("**", "")
    header = header.replace("**", "")

    # Find section start
    start_match = re.search(rf"(?im)^\s*{re.escape(header)}\s*$", norm)
    if not start_match:
        # Sometimes header is on same line: "LONG VERSION: blah"
        start_match = re.search(rf"(?im)^\s*{re.escape(header)}\s*:", norm)
        if not start_match:
            # Handle "LONG VERSION:" with extra spaces or colon variants
            start_match = re.search(rf"(?im)^\s*{re.escape(header.rstrip(':'))}\s*:\s*$", norm)
            if not start_match:
                return ""

    start_idx = start_match.end()

    # Find next header after start
    # Build regex for any next header except the current one
    candidates = [h for h in next_headers if h.strip().lower() != header.strip().lower()]
    next_pat = "|".join([re.escape(h.replace("**", "")) for h in candidates])

    end_match = re.search(rf"(?im)^\s*(?:{next_pat})\s*$", norm[start_idx:])
    if end_match:
        end_idx = start_idx + end_match.start()
        return norm[start_idx:end_idx].strip()

    return norm[start_idx:].strip()

def compress_base_announcement(base_text: str, max_chars: int = 2200) -> str:
    base_text = base_text or ""
    # Prefer LONG VERSION if present
    lv = extract_section(base_text, "LONG VERSION:")
    if lv.strip():
        return trim_text("LONG VERSION:\n" + lv.strip(), max_chars=max_chars)
    # otherwise just trim whole thing
    return trim_text(base_text, max_chars=max_chars)

def trim_text(text: str, max_chars: int = 3500) -> str:
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    # keep the start (usually definitions) + end (often exceptions/steps)
    head = text[: int(max_chars * 0.75)]
    tail = text[-int(max_chars * 0.25) :]
    return head + "\n...\n" + tail

def safe_llm_invoke(llm, prompt: str, timeout_s: int = 120) -> str:
    """
    Run llm.invoke(prompt) with a hard timeout so Streamlit doesn't hang forever
    if Ollama stalls.
    """
    def _call():
        resp = llm.invoke(prompt)
        return resp.content if hasattr(resp, "content") else str(resp)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(_call)
        try:
            return fut.result(timeout=timeout_s)
        except concurrent.futures.TimeoutError:
            return f"ERROR: LLM timed out after {timeout_s}s. Try fewer segments/variants or shorter base text."
        except Exception as e:
            return f"ERROR: LLM call failed: {e}"

        
def safe_llm_invoke_with_retry(llm, prompt: str, timeout_s: int = 120) -> str:
    text = safe_llm_invoke(llm, prompt, timeout_s=timeout_s)

    if text.startswith("ERROR: LLM timed out"):
        # Retry once with trimmed prompt
        shorter_prompt = trim_text(prompt, max_chars=6000)
        text = safe_llm_invoke(llm, shorter_prompt, timeout_s=timeout_s)

    return text

def index_documents(uploaded_files):
    if not uploaded_files:
        st.warning("No files uploaded.")
        return None

    with tempfile.TemporaryDirectory() as temp_dir:
        # Save uploaded files to temp dir
        for uploaded_file in uploaded_files:
            file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

        # Load docs
        loader = DirectoryLoader(
            temp_dir, glob="**/*", loader_cls=get_loader, show_progress=True
        )
        try:
            docs = loader.load()
            st.session_state.doc_count = len(docs)
        except Exception as e:
            st.error(f"Error loading documents: {str(e)}")
            st.info(
                "If this is a Poppler-related error, ensure Poppler is installed and in PATH. "
                "Alternatively, try uploading non-PDF files."
            )
            return None

        if not docs:
            st.error("No documents loaded.")
            return None

        # Split docs
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=300,
            add_start_index=True,
        )
        chunks = text_splitter.split_documents(docs)
        st.session_state.chunk_count = len(chunks)

        # Embed + FAISS
        try:
            embeddings = OllamaEmbeddings(model="nomic-embed-text")
            vector_store = FAISS.from_documents(chunks, embeddings)
            return vector_store
        except Exception as e:
            st.error(f"Error creating vector store: {str(e)}")
            return None


def get_grounding_context(query: str, k: int = 4) -> str:
    """
    Retrieve top-k chunks from the indexed vector store and return as a single string.
    """
    if not st.session_state.get("indexed") or st.session_state.get("vector_store") is None:
        return ""

    try:
        docs = st.session_state.vector_store.similarity_search(query, k=k)
        joined = "\n\n---\n\n".join([d.page_content for d in docs])
        return trim_text(joined, max_chars=3500) 
    
    except Exception as e:
        st.warning(f"Grounding retrieval failed: {e}")
        return ""
@dataclass
class EngagementResult:
    score: int
    reading_time_sec: int
    word_count: int
    avg_sentence_len: float
    has_cta: bool
    bullets: int
    reasons: list


def _count_words(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def _split_sentences(text: str) -> list:
    # Simple sentence splitter
    sents = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s for s in sents if s]


def _count_bullets(text: str) -> int:
    # Counts common bullet markers
    return len(re.findall(r"^\s*[-*•]\s+", text, flags=re.MULTILINE))


def _has_cta(text: str) -> bool:
    # Simple CTA heuristics
    cta_patterns = [
        r"\bplease\b",
        r"\breach out\b",
        r"\bcontact\b",
        r"\blet us know\b",
        r"\baction\b",
        r"\bcomplete\b",
        r"\bfill\b",
        r"\brespond\b",
        r"\bjoin\b",
        r"\bcheck\b",
        r"\bread\b",
        r"\breview\b",
        r"\bconfirm\b",
    ]
    return any(re.search(p, text, flags=re.IGNORECASE) for p in cta_patterns)


def analyze_engagement(text: str, channel: str) -> EngagementResult:
    # We analyze the whole output, but scoring focuses on LONG VERSION-ish content quality.
    # If the model didn't follow format, still gives a reasonable score.

    word_count = _count_words(text)
    reading_time_sec = int((word_count / 200) * 60)  # 200 wpm
    sentences = _split_sentences(text)
    avg_sentence_len = (word_count / max(1, len(sentences)))

    bullets = _count_bullets(text)
    has_cta = _has_cta(text)

    score = 60  # baseline
    reasons = []

    # Length preferences by channel (rough, defensible heuristics)
    if channel == "Email":
        if 120 <= word_count <= 260:
            score += 10; reasons.append("Good length for email (+10)")
        elif word_count < 90:
            score -= 8; reasons.append("A bit too short for email (-8)")
        elif word_count > 320:
            score -= 12; reasons.append("A bit too long for email (-12)")
    elif channel == "Slack":
        if 60 <= word_count <= 160:
            score += 10; reasons.append("Good length for Slack (+10)")
        elif word_count > 220:
            score -= 12; reasons.append("Too long for Slack (-12)")
        if bullets >= 2:
            score += 8; reasons.append("Good use of bullets for Slack (+8)")
        else:
            score -= 5; reasons.append("Slack messages work better with bullets (-5)")
    elif channel == "Digital signage":
        # Signage should be short. If user returns a full long version, penalize.
        if word_count <= 60:
            score += 10; reasons.append("Concise enough for signage (+10)")
        else:
            score -= 18; reasons.append("Too long for signage (-18)")
    elif channel == "Intranet post":
        if 140 <= word_count <= 320:
            score += 10; reasons.append("Good length for intranet (+10)")
        elif word_count > 400:
            score -= 12; reasons.append("Too long for intranet (-12)")

    # Sentence length: aim for clarity
    if avg_sentence_len <= 18:
        score += 6; reasons.append("Clear sentence length (+6)")
    elif avg_sentence_len >= 28:
        score -= 10; reasons.append("Sentences are quite long (-10)")

    # CTA
    if has_cta:
        score += 8; reasons.append("Clear call-to-action (+8)")
    else:
        score -= 6; reasons.append("Missing a clear call-to-action (-6)")


    # Clamp score
    score = max(0, min(100, score))

    return EngagementResult(
        score=score,
        reading_time_sec=reading_time_sec,
        word_count=word_count,
        avg_sentence_len=round(avg_sentence_len, 1),
        has_cta=has_cta,
        bullets=bullets,
        reasons=reasons,
    )
def constraint_penalty(text: str, channel: str) -> tuple[int, list]:
    reasons = []
    penalty = 0

    lv = extract_section(text, "LONG VERSION:") or text
    signage = extract_section(text, "SIGNAGE LINE:")

    wc = _count_words(lv)
    bullets = _count_bullets(lv)

    if channel == "Slack":
        if wc > 120:
            penalty += 20; reasons.append("Slack LONG VERSION > 120 words (-20)")
        if bullets < 3 or bullets > 5:
            penalty += 15; reasons.append("Slack needs 3–5 bullets (-15)")
        if bullets == 0:
            penalty += 20; reasons.append("Slack LONG VERSION must be bullets only (-20)")

    if channel == "Digital signage":
        if signage:
            if _count_words(signage) > 20:
                penalty += 15; reasons.append("SIGNAGE LINE > 20 words (-15)")
        if wc > 60:
            penalty += 20; reasons.append("Signage LONG VERSION > 60 words (-20)")

    return penalty, reasons

def build_announcement_prompt(topic, key_points, audience, tone, channel, context: str = "", variant_hint: str = ""):
    key_points_clean = "\n".join([f"- {kp.strip()}" for kp in key_points.split("\n") if kp.strip()])

    context_block = ""
    if context.strip():
        context_block = f"""
COMPANY CONTEXT (use for company-specific details ONLY):
{context}

Rules for using context:
- Only state company-specific facts if they appear in the context above.
- If context is missing a required detail, explicitly say what’s missing and keep wording neutral.
- If any checklist item is not found in context, do NOT guess it; omit it or state it is not specified.

Coverage checklist (include if present in context):
- Effective date
- Office attendance requirements (by location/team if specified)
- Exceptions / manager approval
- Core working hours / flexible hours
- Remote work equipment / support
- Compliance consequences (if any)
- Communication expectations (e.g., Slack/email reachability)
"""

    # Channel differentiation rules (stronger, structural)
    channel_rules = ""
    if channel == "Email":
        channel_rules = """
Channel requirements (Email):
- Use a professional greeting (e.g., "Hi everyone," or "Dear all,").
- Use short paragraphs and/or bullets for key requirements.
- Include a clear call-to-action (who to contact or what to do next).
- End with a simple sign-off (e.g., "Thanks," and a placeholder name).
"""
    elif channel == "Slack":
        channel_rules = """
Channel requirements (Slack):
- LONG VERSION must be <= 120 words.
- No formal greeting or sign-off.
- Use ONLY bullet points in LONG VERSION (no paragraphs).
- Include exactly 3–5 bullets.
- Each bullet must be one sentence max.
- End with ONE short CTA line (e.g., "Action: Review the policy here: [link]. Questions: [HR email]").
- SUBJECT should be a short Slack headline (<= 8 words) or repeat the topic.
"""
    elif channel == "Digital signage":
        channel_rules = """
Channel requirements (Digital signage):
- SIGNAGE LINE must be <= 20 words.
- LONG VERSION should be extremely short (max 60 words) and action-oriented.
- No greeting, no sign-off.
"""
    elif channel == "Intranet post":
        channel_rules = """
Channel requirements (Intranet post):
- Slightly more detail than email is okay, but still scannable.
- Include a short summary + bullet list of requirements.
- Include a call-to-action and "Where to find more info" placeholder.
"""

    # Variant hint for A/B (light but meaningful differences)
    variant_block = ""
    if variant_hint.strip():
        variant_block = f"\nVariant guidance: {variant_hint}\n"

    return f"""
You are an Internal Communications Copilot for an enterprise company.

Task: Draft an internal announcement.

Topic: {topic}
Audience: {audience}
Tone: {tone}
Primary channel: {channel}

Key points to include:
{key_points_clean if key_points_clean else "- (none provided)"}

{context_block}

General rules:
- Do NOT invent company-specific policies or facts.
- Be concise, clear, and non-fluffy.
- Avoid emojis unless tone is "Friendly" and channel is "Slack".
- Use Irish/UK English spelling (e.g., organise, programme).
{variant_block}
{channel_rules}

Return EXACTLY in this format:

SUBJECT:
<one subject line>

LONG VERSION:
<announcement suitable for the selected channel>

SHORT VERSION:
<a short version (1-3 sentences) suitable for a quick post>

SIGNAGE LINE:
<one single-line version suitable for digital signage / intranet banner>
""".strip()


def render_announcement_generator(llm):
    st.subheader("Announcement Generator")

    colA, colB = st.columns(2)

    with colA:
        topic = st.text_input(
            "Topic / Title",
            placeholder="e.g., New hybrid working guidelines",
            key="ann_topic",
        )

        audience = st.selectbox(
            "Audience",
            [
                "All staff",
                "Managers",
                "Engineering",
                "Sales",
                "Customer Support",
                "Remote teams",
            ],
            key="ann_audience",
        )

    with colB:
        tone = st.selectbox(
            "Tone",
            ["Executive", "Formal", "Friendly", "Urgent"],
            key="ann_tone",
        )

        channel = st.selectbox(
            "Channel",
            ["Email", "Slack", "Digital signage", "Intranet post"],
            key="ann_channel",
        )

    key_points = st.text_area(
        "Key points (one per line)",
        placeholder="e.g.\nPolicy starts April 1\n2 days in office for Dublin team\nManagers to confirm schedules",
        key="ann_key_points",
    )

    # Grounding controls
    can_ground = st.session_state.get("indexed") and st.session_state.get("vector_store") is not None
    use_grounding = st.checkbox(
        "Ground in indexed documents (policy/company context)",
        value=False,
        disabled=not can_ground,
        help="Index documents in Chat mode first, then return here.",
    )
    k = st.slider("Context chunks (k)", 2, 8, 4, 1, disabled=not use_grounding)
    show_context = st.checkbox("Show retrieved context (debug)", value=False, disabled=not use_grounding)

    # A/B toggle
    do_ab = st.checkbox("Generate A/B variants (compare predicted engagement)", value=True)

    generate = st.button("Generate", type="primary", disabled=(not topic.strip()))

    if generate:
        retrieval_query = f"{topic}\n{key_points}"
        context = get_grounding_context(retrieval_query, k=k) if use_grounding else ""

        variants = [
            ("Variant A", "More direct and concise. Use strong, clear headings and bullets."),
        ]

        if do_ab:
            variants.append(
                ("Variant B", "More empathetic and supportive. Emphasise benefits and support channels.")
            )
        
        results = []

        

        with st.spinner("Generating..."):
            for name, hint in variants:
                prompt = build_announcement_prompt(
                    topic=topic,
                    key_points=key_points,
                    audience=audience,
                    tone=tone,
                    channel=channel,
                    context=context,
                    variant_hint=hint,
                )

                text = safe_llm_invoke_with_retry(llm, prompt, timeout_s=120)
                # 🔹 Extract LONG VERSION only for scoring
                long_version = extract_section(text, "LONG VERSION:")
                # 🔹 Score only LONG VERSION (fallback to full text if extraction fails)
                score = analyze_engagement(long_version if long_version else text, channel=channel)
                penalty, penalty_reasons = constraint_penalty(text, channel)
                final_score = max(0, score.score - penalty)
                # overwrite score for ranking
                score.score = final_score
                score.reasons.extend(penalty_reasons)
                results.append((name, text, score))

        if use_grounding and show_context:
            with st.expander("Retrieved Context (Debug)"):
                st.write(context if context.strip() else "(No context retrieved)")

        for name, text, score in results:
            st.markdown(f"## {name}")
            st.metric("Predicted engagement score", f"{score.score}/100")
            # 🔹 Optional debug section
            long_version = extract_section(text, "LONG VERSION:")
            if long_version:
                with st.expander(f"{name} — Scored content (LONG VERSION)"):
                    st.write(long_version)

            cols = st.columns(4)
            cols[0].metric("Word count", score.word_count)
            cols[1].metric("Reading time", f"{max(1, score.reading_time_sec)}s")
            cols[2].metric("Avg sentence length", score.avg_sentence_len)
            cols[3].metric("CTA present", "Yes" if score.has_cta else "No")

            with st.expander("Why this score?"):
                for r in score.reasons:
                    st.write(f"- {r}")

            st.markdown("### Output")
            st.code(text, language="markdown")

        if do_ab and len(results) == 2:
            winner = max(results, key=lambda x: x[2].score)
            st.success(f"Suggested winner: **{winner[0]}** ({winner[2].score}/100)")

def build_segment_adaptation_prompt(
    base_announcement: str,
    segment: str,
    channel: str,
    tone: str,
    audience_notes: str = "",
    context: str = ""
) -> str:
    """
    Takes a base announcement (already drafted) and adapts it for a specific audience segment.
    Keeps channel constraints and optional grounding context.
    """
    context_block = ""
    if context.strip():
        context_block = f"""
COMPANY CONTEXT (use for company-specific details ONLY):
{context}

Rules for using context:
- Only state company-specific facts if they appear in the context above.
- If a specific detail is not in context, keep it neutral (do not guess).
"""

    channel_rules = ""
    if channel == "Email":
        channel_rules = """
Channel requirements (Email):
- Professional greeting + simple sign-off.
- 120–260 words in LONG VERSION if possible.
- Use bullets for requirements.
- End with one clear CTA line.
"""
    elif channel == "Slack":
        channel_rules = """
Channel requirements (Slack):
- LONG VERSION must be <= 120 words.
- No greeting or sign-off.
- Use ONLY bullet points in LONG VERSION (no paragraphs).
- Include exactly 3–5 bullets.
- Each bullet must be one sentence max.
- End with ONE short CTA line (include [link] + [HR/People Ops contact]).
"""
    elif channel == "Digital signage":
        channel_rules = """
Channel requirements (Digital signage):
- SIGNAGE LINE must be <= 20 words.
- LONG VERSION should be <= 60 words, action-oriented.
- No greeting, no sign-off.
"""
    elif channel == "Intranet post":
        channel_rules = """
Channel requirements (Intranet post):
- 180–320 words is acceptable.
- Include short summary + bullets + 'Where to find more info'.
- End with one clear CTA line.
"""

    return f"""
You are an enterprise Internal Communications Copilot.

Task:
Adapt the BASE ANNOUNCEMENT for the audience segment below, while preserving factual accuracy.

Audience segment: {segment}
Tone: {tone}
Primary channel: {channel}

Audience notes (if any):
{audience_notes if audience_notes.strip() else "- None"}

BASE ANNOUNCEMENT:
{base_announcement}

{context_block}

Rules:
- Keep all policy facts consistent with the base announcement and/or context.
- Do not add new policy rules that are not present in the base announcement or context.
- Make the message more relevant to the segment (what they need to know / do).
- If the segment is "Managers", include manager responsibilities (approvals, SLA, next steps) if relevant.
- If the segment is "Contractors", keep it short and include "where applicable" language if policy is unclear.
- Avoid personal data and avoid language implying surveillance (e.g., "we know who has not complied").

{channel_rules}

Return EXACTLY in this format:

SUBJECT:
<subject line>

LONG VERSION:
<adapted announcement for the chosen channel>

SHORT VERSION:
<1–3 sentence summary>

SIGNAGE LINE:
<one single-line banner/signage version>
""".strip()


def render_segmented_distribution(llm):
    st.header("Segmented Distribution")

    st.caption(
        "Create audience-specific variants from one base announcement, score them, and export."
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        channel = st.selectbox("Channel", ["Email", "Slack", "Digital signage", "Intranet post"])
    with col2:
        tone = st.selectbox("Tone", ["Executive", "Formal", "Friendly", "Urgent"])

    segments_master = [
        "All staff",
        "Managers",
        "Engineering",
        "Sales",
        "Customer Support",
        "Remote teams",
        "Dublin office",
        "Regional offices",
        "Contractors",
        "New starters",
    ]

    selected_segments = st.multiselect(
        "Select audience segments",
        segments_master,
        default=["All staff", "Managers"]
    )

    audience_notes = st.text_area(
        "Optional notes (constraints, nuance, internal context)",
        placeholder="e.g., Managers must respond within 5 working days. Avoid implying monitoring."
    )

    st.markdown("### Base announcement")
    base_announcement = st.text_area(
        "Paste the base announcement here (full text).",
        height=220,
        placeholder="Paste your generated announcement (including SUBJECT/LONG/SHORT/SIGNAGE), or just the main body."
    )

    # Optional grounding
    can_ground = st.session_state.get("indexed") and st.session_state.get("vector_store") is not None
    use_grounding = st.checkbox(
        "Ground adaptations in indexed documents",
        value=False,
        disabled=not can_ground,
        help="Index documents in Chat mode first."
    )
    k = st.slider("Context chunks (k)", 2, 10, 6, 1, disabled=not use_grounding)
    show_context = st.checkbox("Show retrieved context (debug)", value=False, disabled=not use_grounding)

    # A/B variants per segment
    do_ab = st.checkbox("Generate A/B variants per segment", value=False)
    max_segments = st.slider("Max segments to generate (safety)", 1, 10, min(6, max(1, len(selected_segments) or 1)))
    if "cancel_gen" not in st.session_state:
        st.session_state.cancel_gen = False

    cancel = st.button("Cancel generation", type="secondary")
    if cancel:
        st.session_state.cancel_gen = True
        st.warning("Cancelling... (will stop after the current model call returns)")
        
    generate = st.button(
        "Generate segment variants",
        type="primary",
        disabled=(not base_announcement.strip() or not selected_segments)
    )

    if not generate:
        return
    st.session_state.cancel_gen = False
    base_compact = compress_base_announcement(base_announcement, max_chars=2200)

    # Build retrieval query from base announcement + notes
    retrieval_query = f"{base_announcement}\n{audience_notes}\n{tone}\n{channel}"
    context = get_grounding_context(retrieval_query, k=k) if use_grounding else ""

    if use_grounding and show_context:
        with st.expander("Retrieved Context (Debug)"):
            st.write(context if context.strip() else "(No context retrieved)")

    # Limit segments generated (avoid long runs)
    segments_to_run = selected_segments[:max_segments]

    results_rows = []
    export_payload = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "channel": channel,
        "tone": tone,
        "segments": [],
    }

    total_calls = len(segments_to_run) * (2 if do_ab else 1)
    done_calls = 0
    progress = st.progress(0)
    status = st.empty()

    with st.spinner("Generating variants..."):
        for seg in segments_to_run:
            if st.session_state.cancel_gen:
                break

            variants = [("A", "More direct and concise.")]
            if do_ab:
                variants.append(("B", "More supportive and empathetic."))

            seg_variants = []

            for vname, vhint in variants:
                if st.session_state.cancel_gen:
                    break

                status.write(f"Generating {seg} — Variant {vname} ({done_calls+1}/{total_calls})")
                base_compact = compress_base_announcement(base_announcement, max_chars=2200)

                prompt = build_segment_adaptation_prompt(
                    base_announcement=base_compact,
                    segment=seg,
                    channel=channel,
                    tone=tone,
                    audience_notes=audience_notes + (f"\nVariant guidance: {vhint}" if vhint else ""),
                    context=context
                )

                text = safe_llm_invoke_with_retry(llm, prompt, timeout_s=120)

                done_calls += 1
                progress.progress(min(1.0, done_calls / max(1, total_calls)))

                long_version = extract_section(text, "LONG VERSION:")
                score = analyze_engagement(long_version if long_version else text, channel=channel)
                penalty, penalty_reasons = constraint_penalty(text, channel)
                final_score = max(0, score.score - penalty)
                score.score = final_score
                score.reasons.extend(penalty_reasons)

                seg_variants.append({
                    "variant": vname,
                    "score": score.score,
                    "word_count": score.word_count,
                    "reading_time_sec": score.reading_time_sec,
                    "avg_sentence_len": score.avg_sentence_len,
                    "cta_present": score.has_cta,
                    "text": text,
                })

            # pick winner
            winner = max(seg_variants, key=lambda x: x["score"])
            export_payload["segments"].append({
                "segment": seg,
                "winner": winner,
                "all_variants": seg_variants
            })

            # table rows
            for item in seg_variants:
                results_rows.append({
                    "Segment": seg,
                    "Variant": item["variant"],
                    "Score": item["score"],
                    "Words": item["word_count"],
                    "CTA": "Yes" if item["cta_present"] else "No",
                    "Preview": (extract_section(item["text"], "SUBJECT:")[:80] + "...") if extract_section(item["text"], "SUBJECT:") else item["text"][:80] + "..."
                })

    st.success(f"Generated variants for {len(segments_to_run)} segment(s).")

    # Show results table
    st.markdown("### Results")
    st.dataframe(results_rows, width="stretch")

    # Show winners
    st.markdown("### Winners (best score per segment)")
    for seg_obj in export_payload["segments"]:
        seg = seg_obj["segment"]
        win = seg_obj["winner"]
        st.subheader(seg)
        st.metric("Winner", f"Variant {win['variant']} — {win['score']}/100")
        with st.expander("Winner output"):
            st.code(win["text"], language="markdown")

    # Export buttons
    st.markdown("### Export")
    export_json = json.dumps(export_payload, indent=2).encode("utf-8")
    st.download_button(
        "Download JSON",
        data=export_json,
        file_name="segmented_distribution.json",
        mime="application/json",
    )

    # simple CSV export (flat: segment, winner variant, score)
    csv_lines = ["segment,winner_variant,score,words,cta_present"]
    for seg_obj in export_payload["segments"]:
        w = seg_obj["winner"]
        csv_lines.append(f"{seg_obj['segment']},{w['variant']},{w['score']},{w['word_count']},{w['cta_present']}")
    export_csv = ("\n".join(csv_lines)).encode("utf-8")
    st.download_button(
        "Download summary CSV",
        data=export_csv,
        file_name="segmented_distribution_summary.csv",
        mime="text/csv",
    )
# -----------------------------
# LLM init + mode gate
# -----------------------------
llm = ChatOllama(model="llama3.1:8b", temperature=0.4)

if mode == "Announcement Generator":
    render_announcement_generator(llm)
    st.stop()

if mode == "Segmented Distribution":
    render_segmented_distribution(llm)
    st.stop()
# -----------------------------
# Chat mode UI (upload + index + chat)
# -----------------------------
tab_upload, tab_chat, tab_debug = st.tabs(["📄 Upload & Index", "💬 Chat", "🧪 Debug"])

with tab_upload:
    with st.expander("Quick start", expanded=not st.session_state.get("indexed", False)):
        st.markdown(
            """
1) Upload a policy pack (TXT/DOCX recommended)  
2) Click **Index Uploaded Files**  
3) Ask questions in the chat box (try: “What’s the hybrid policy effective date?”)  
"""
        )

    uploaded_files = st.file_uploader(
        "Upload your files (optional)",
        accept_multiple_files=True,
        type=["pdf", "txt", "csv", "docx", "md", "html", "json", "htm"],
    )

    if st.button("Index Uploaded Files") and uploaded_files:
        with st.status("Indexing documents…", expanded=True) as status:
            st.session_state.vector_store = index_documents(uploaded_files)
            if st.session_state.vector_store:
                st.session_state.indexed = True
                status.update(label="Indexing complete ✅", state="complete")
                st.toast("Documents indexed successfully", icon="✅")
                st.success(
                    f"Loaded {st.session_state.doc_count} documents, split into {st.session_state.chunk_count} chunks."
                )
            else:
                status.update(label="Indexing failed ❌", state="error")
                st.error("Indexing failed. Check the error messages above.")

    if st.session_state.indexed:
        st.info(
            f"Indexed {st.session_state.doc_count} documents, "
            f"{st.session_state.chunk_count} chunks. You can now query your data."
        )

# -----------------------------
# General chat chain
# -----------------------------
general_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful AI assistant. Answer the user's question concisely and accurately. "
            "If you don't know the answer, say so.",
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
general_chat_chain = general_prompt | llm | StrOutputParser()

# -----------------------------
# RAG chain (only if indexed)
# -----------------------------
conversational_rag_chain = None
if st.session_state.indexed:
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, just reformulate it if needed."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    retriever = st.session_state.vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": 10}
    )

    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    qa_system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the provided context to answer the question as accurately as possible. "
        "If the context doesn't contain enough information, provide a brief answer based on what is available "
        "or say you need more details."
        "\n\n{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    qa_chain = create_stuff_documents_chain(llm, qa_prompt)
    conversational_rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

# -----------------------------
# Chat UI
# -----------------------------
with tab_chat:
    st.subheader("Chat with your data" if st.session_state.indexed else "General Chat")

    for message in st.session_state.chat_history:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(message.content)

    if prompt := st.chat_input("Ask a question" + (" about your documents" if st.session_state.indexed else "")):
        with st.chat_message("user"):
            st.markdown(prompt)

        st.session_state.chat_history.append(HumanMessage(content=prompt))

        with st.spinner("Thinking..."):
            try:
                if st.session_state.indexed and conversational_rag_chain is not None:
                    response = conversational_rag_chain.invoke(
                        {"input": prompt, "chat_history": st.session_state.chat_history}
                    )
                    answer = response["answer"]

                    if response.get("context"):
                        with st.expander("Retrieved Context (Debug)"):
                            st.write("\n".join([doc.page_content for doc in response["context"]]))
                else:
                    answer = general_chat_chain.invoke(
                        {"input": prompt, "chat_history": st.session_state.chat_history}
                    )

                with st.chat_message("assistant"):
                    st.markdown(answer)

                st.session_state.chat_history.append(AIMessage(content=answer))
            except Exception as e:
                error_message = f"Error processing query: {str(e)}"
                with st.chat_message("assistant"):
                    st.markdown(error_message)
                st.session_state.chat_history.append(AIMessage(content=error_message))

    if st.button("Reset Chat History"):
        st.session_state.chat_history = []
        if st.session_state.indexed:
            st.session_state.indexed = False
            st.session_state.vector_store = None
            st.session_state.doc_count = 0
            st.session_state.chunk_count = 0
        st.experimental_rerun()

with tab_debug:
    st.write("Session state snapshot:")
    st.json(
        {
            "mode": mode,
            "indexed": st.session_state.get("indexed"),
            "doc_count": st.session_state.get("doc_count"),
            "chunk_count": st.session_state.get("chunk_count"),
            "has_vector_store": st.session_state.get("vector_store") is not None,
            "chat_history_len": len(st.session_state.get("chat_history", [])),
        }
    )