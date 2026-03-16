import os
import json
import time
import streamlit as st
from google import genai
from google.genai import types
from tavily import TavilyClient
from groq import Groq

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG  (first Streamlit call — required before all else)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Live AI Debate Arena",
    page_icon="⚔️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Clear stale cache on every cold boot
st.cache_data.clear()

# ─────────────────────────────────────────────────────────────────────────────
# PERSONAS
# ─────────────────────────────────────────────────────────────────────────────
PERSONALITIES = {
    "The Scholar": {
        "emoji": "🎓",
        "color": "#4A90D9",
        "description": "Academic · data-driven · formal",
        "system": (
            "You are The Scholar — a meticulous academic debater. "
            "You cite data, reference studies, use formal language, and build structured arguments. "
            "You never use slang, always back claims with evidence, and speak in a measured, authoritative tone. "
            "Begin every argument with a formal thesis statement."
        ),
    },
    "The Joker": {
        "emoji": "🃏",
        "color": "#E74C3C",
        "description": "Sarcastic · roasts opponents · witty",
        "system": (
            "You are The Joker — a razor-sharp satirical debater. "
            "You use biting sarcasm, theatrical wit, and merciless roasts to dismantle opponents. "
            "Every joke contains a real argument. Roast your opponent's weakest point first, then land the actual argument. "
            "Punchy, memorable one-liners. No bullet points — just talk."
        ),
    },
    "The Robot": {
        "emoji": "🤖",
        "color": "#27AE60",
        "description": "Cold logic · data-only · no emotion",
        "system": (
            "You are The Robot — a purely logical debate engine. "
            "Process arguments as boolean logic and probability trees. "
            "Flag emotional appeals as LOGICAL_FALLACY[type]. "
            "Respond only with data, probability estimates, causal chains, and cold facts. "
            "Speak in a slightly mechanical, detached tone. Use precise numerical claims whenever possible."
        ),
    },
    "The Lawyer": {
        "emoji": "⚖️",
        "color": "#8E44AD",
        "description": "Aggressive · finds flaws · nitpicks",
        "system": (
            "You are The Lawyer — an aggressive, detail-obsessed litigator. "
            "Identify every flaw, contradiction, and hidden assumption in your opponent's case. "
            "Object loudly, demand precision, build airtight arguments. "
            "Speak as if cross-examining a witness: direct, confrontational, relentless. "
            "Open with 'Objection.' or 'Let the record show...' when appropriate."
        ),
    },
    "The Dreamer": {
        "emoji": "🌟",
        "color": "#F39C12",
        "description": "Idealistic · ethics-focused · visionary",
        "system": (
            "You are The Dreamer — an idealistic, ethics-first visionary. "
            "Argue for the greater good, human dignity, and long-term societal flourishing. "
            "Paint vivid pictures of better futures and ground every argument in moral philosophy. "
            "Speak with passion and hope. Appeal to shared values. Make people feel the stakes."
        ),
    },
    "The Skeptic": {
        "emoji": "🔍",
        "color": "#95A5A6",
        "description": "Questions everything · demands proof",
        "system": (
            "You are The Skeptic — a relentless critical thinker who questions everything. "
            "Demand extraordinary evidence for extraordinary claims. "
            "Expose hidden assumptions, challenge consensus, probe every weak point. "
            "Use phrases like 'But where is the proof?', 'That assumes...', or 'Have you considered...'. "
            "Apply the same skepticism to your own side — intellectual honesty above all."
        ),
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
GLOBAL_CSS = """
<style>
/* Main container padding */
.stApp {
    padding: 3rem 10% !important;
}
/* Chat message breathing room */
.stChatMessage {
    margin-bottom: 30px !important;
}
/* Fact sheet breathing room */
.fact-sheet {
    margin-bottom: 30px !important;
}
.arena-title {
    text-align: center;
    font-size: 2.8rem;
    font-weight: 900;
    background: linear-gradient(90deg, #E74C3C 0%, #8E44AD 50%, #4A90D9 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.1rem;
}
.arena-subtitle {
    text-align: center;
    color: #6b7280;
    font-size: 0.92rem;
    margin-bottom: 1.8rem;
}
.fact-card {
    background: #0f172a;
    border-radius: 12px;
    padding: 18px;
    border: 1px solid #1e293b;
    min-height: 220px;
    height: 100%;
    margin-bottom: 30px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.45), 0 1.5px 6px rgba(0,0,0,0.3);
}
.fact-card pre {
    text-align: left;
    line-height: 1.6;
    white-space: pre-wrap;
    font-family: inherit;
    font-size: 0.85rem;
    color: #d1d5db;
}
.research-card {
    background: #111827;
    border-radius: 12px;
    padding: 18px;
    border: 1px solid #1e293b;
    min-height: 220px;
    height: 100%;
    margin-bottom: 30px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.45), 0 1.5px 6px rgba(0,0,0,0.3);
}
.research-card pre {
    text-align: left;
    line-height: 1.6;
    white-space: pre-wrap;
    font-family: inherit;
    font-size: 0.78rem;
    color: #6b7280;
}
.round-banner {
    text-align: center;
    font-size: 1.3rem;
    font-weight: 800;
    color: #F39C12;
    padding: 10px 0;
    border-top: 1px solid #2d2d2d;
    border-bottom: 1px solid #2d2d2d;
    margin: 24px 0 16px 0;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}
.verdict-card {
    background: linear-gradient(135deg, #0f172a, #111827);
    border: 2px solid #E74C3C;
    border-radius: 12px;
    padding: 30px 34px;
    margin-top: 20px;
    margin-bottom: 30px;
    box-shadow: 0 8px 32px rgba(231,76,60,0.18), 0 2px 8px rgba(0,0,0,0.4);
}
.fatal-flaw-box {
    background: #1a0404;
    border-left: 4px solid #E74C3C;
    border-radius: 6px;
    padding: 12px 16px;
    margin: 14px 0;
    color: #fca5a5;
    font-style: italic;
    font-size: 0.97rem;
    line-height: 1.6;
}
.thinking-tag {
    display: inline-block;
    background: #1e293b;
    color: #475569;
    border-radius: 4px;
    padding: 2px 9px;
    font-size: 0.72rem;
    margin-bottom: 5px;
    letter-spacing: 0.03em;
}
.engine-tag {
    display: inline-block;
    background: #0d1b2a;
    color: #38bdf8;
    border-radius: 4px;
    padding: 2px 9px;
    font-size: 0.70rem;
    margin-bottom: 5px;
    margin-left: 4px;
    letter-spacing: 0.03em;
    border: 1px solid #1e3a5f;
}
.stance-for     { color: #4ade80; font-weight: 700; }
.stance-against { color: #f87171; font-weight: 700; }
</style>
"""


# ─────────────────────────────────────────────────────────────────────────────
# API CLIENTS
# ─────────────────────────────────────────────────────────────────────────────
def get_gemini_client():
    key = os.environ.get("GEMINI_API_KEY", "")
    if not key:
        st.error(
            "❌ **GEMINI_API_KEY** not found. Add it to Replit Secrets and restart."
        )
        st.stop()
    try:
        return genai.Client(api_key=key)
    except Exception as exc:
        st.error(f"❌ Gemini init failed: {exc}")
        st.stop()


def get_tavily_client():
    key = os.environ.get("TAVILY_API_KEY", "")
    if not key:
        st.warning("⚠️ TAVILY_API_KEY missing — debate will use Gemini knowledge only.")
        return None
    try:
        return TavilyClient(api_key=key)
    except Exception as exc:
        st.warning(f"⚠️ Tavily init failed ({exc}) — using Gemini knowledge only.")
        return None


def get_groq_client():
    key = os.environ.get("GROQ_API_KEY", "")
    if not key:
        st.error("❌ **GROQ_API_KEY** not found. Add it to Replit Secrets and restart.")
        st.stop()
    try:
        return Groq(api_key=key)
    except Exception as exc:
        st.error(f"❌ Groq init failed: {exc}")
        st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# RESEARCH PHASE  — Gemini 2.5 Flash, 2 debaters only, 2-second gap
# ─────────────────────────────────────────────────────────────────────────────
def research_topic(
    topic: str,
    d1_name: str,
    d2_name: str,
    progress_bar,
    status_text,
) -> tuple[str, dict]:
    gemini = get_gemini_client()

    # Tavily web search
    status_text.text("🔍  Searching the web for live facts via Tavily…")
    progress_bar.progress(5)
    raw_facts = ""

    tavily = get_tavily_client()
    if tavily:
        try:
            results = tavily.search(
                query=topic,
                search_depth="advanced",
                max_results=5,
                include_answer=True,
            )
            if results.get("answer"):
                raw_facts += f"Summary: {results['answer']}\n\n"
            for r in results.get("results", [])[:3]:
                raw_facts += (
                    f"Source: {r.get('title', 'N/A')}\n"
                    f"Excerpt: {r.get('content', '')[:400]}\n\n"
                )
        except Exception as exc:
            raw_facts = f"[Tavily error: {exc}. Using general knowledge.]"
    else:
        raw_facts = f"No live web data. Use general knowledge about: {topic}"

    progress_bar.progress(20)

    # Brief only the two active debaters
    active = [(d1_name, PERSONALITIES[d1_name]), (d2_name, PERSONALITIES[d2_name])]
    briefs: dict = {}

    for idx, (name, persona) in enumerate(active):
        status_text.text(
            f"🧠  Gemini briefing {persona['emoji']} {name} ({idx + 1}/2)…"
        )
        prompt = (
            f"You are a ruthless debate coach briefing {name} before a live public debate.\n\n"
            f'DEBATE TOPIC: "{topic}"\n\n'
            f"LIVE RESEARCH:\n{raw_facts}\n\n"
            f"{name}'s PERSONA: {persona['description']}\n\n"
            "Write a SECRET TACTICAL BRIEF (bullet points) covering:\n"
            "• 2-3 KEY FACTS from the research to weaponize\n"
            "• The STRONGEST ANGLE given this persona\n"
            "• One KILLER ARGUMENT tailored to their voice\n"
            "• One VULNERABILITY to watch for\n\n"
            "Sharp, actionable, max 130 words."
        )
        try:
            resp = gemini.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.7),
            )
            briefs[name] = resp.text.strip()
        except Exception as exc:
            briefs[name] = (
                f"Brief unavailable ({exc}). Stay in character using general knowledge."
            )

        progress_bar.progress(20 + (idx + 1) * 35)

        if idx == 0:
            status_text.text("⏳  2-second quota guard before briefing debater 2…")
            time.sleep(2)

    progress_bar.progress(100)
    status_text.text("✅  Research complete — debaters are briefed!")
    time.sleep(0.4)
    return raw_facts, briefs


# ─────────────────────────────────────────────────────────────────────────────
# DEBATE ENGINE — Groq Llama-3 70B, single call: DRAFT / CRITIQUE / FINAL
# ─────────────────────────────────────────────────────────────────────────────
def hidden_thought_process(
    groq_client,
    persona_name: str,
    persona: dict,
    stance: str,
    topic: str,
    brief: str,
    chat_history: list,
    round_num: int,
    is_rebuttal: bool,
    opponent_name: str,
) -> str:
    history_snippet = "\n".join(
        f"[{m['role']}]: {m['content']}" for m in chat_history[-6:]
    )
    if is_rebuttal:
        context = (
            f"ROUND {round_num} — REBUTTAL against {opponent_name}.\n"
            f"Recent debate:\n{history_snippet}"
        )
    else:
        context = f"ROUND {round_num} — Opening argument."
        if history_snippet:
            context += f"\nRecent debate:\n{history_snippet}"

    stance_label = (
        "FOR (strongly supporting)"
        if stance == "FOR"
        else "AGAINST (strongly opposing)"
    )

    prompt = f"""You are {persona_name} in a live public debate. You are arguing {stance_label} the topic below.

TOPIC: "{topic}"
YOUR SECRET BRIEF: {brief}
{context}

YOUR PERSONA VOICE: {persona["system"]}

You MUST produce all three labeled sections in order. Use the exact labels shown.

DRAFT:
Write your raw first-draft argument in your persona's voice. 3-5 sentences. Stay in character.

CRITIQUE:
Critically evaluate your own draft. List 2-3 specific flaws or improvements (logic, persona voice, memorability).

FINAL:
Rewrite the argument from scratch, fixing every flaw from CRITIQUE. This is what the audience hears.
Stay 100% in character as {persona_name}. Argue clearly {stance_label} the topic.
3-5 sentences. No headers, no bullet points — just speak naturally."""

    try:
        resp = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.85,
            max_tokens=600,
        )
        raw = resp.choices[0].message.content or ""
    except Exception as exc:
        return f"[{persona_name} encountered a technical issue: {exc}]"

    # Parse out the FINAL section
    if "FINAL:" in raw:
        final_text = raw.split("FINAL:", 1)[-1].strip()
        # Remove any stray section tags that may have leaked
        for tag in ("DRAFT:", "CRITIQUE:"):
            if tag in final_text:
                final_text = final_text.split(tag, 1)[0].strip()
    else:
        # Fallback: return the last third of the response
        lines = [ln.strip() for ln in raw.strip().splitlines() if ln.strip()]
        final_text = " ".join(lines[len(lines) // 2 :]) if lines else raw.strip()

    return final_text if final_text else raw.strip()


# ─────────────────────────────────────────────────────────────────────────────
# WORD-BY-WORD STREAMING  — 0.01 s per word for a fast data-stream feel
# ─────────────────────────────────────────────────────────────────────────────
def word_stream(text: str, delay: float = 0.1):
    words = text.split(" ")
    for i, word in enumerate(words):
        yield word + (" " if i < len(words) - 1 else "")
        time.sleep(delay)


# ─────────────────────────────────────────────────────────────────────────────
# JUDGE — Groq Llama-3 70B, raw JSON output only
# ─────────────────────────────────────────────────────────────────────────────
def judge_debate(
    groq_client,
    topic: str,
    d1_name: str,
    d2_name: str,
    chat_history: list,
) -> dict:
    transcript = "\n".join(f"[{m['role']}]: {m['content']}" for m in chat_history)

    prompt = (
        f'You are an impartial debate judge evaluating a debate on: "{topic}"\n\n'
        f"DEBATER A: {d1_name}\n"
        f"DEBATER B: {d2_name}\n\n"
        f"FULL TRANSCRIPT:\n{transcript}\n\n"
        "Score each debater 0-100 on: argument strength, evidence, persona authenticity, rebuttal quality, persuasiveness.\n"
        "Identify the single most damaging mistake as Fatal_Flaw.\n\n"
        "CRITICAL INSTRUCTION: Output ONLY a single raw valid JSON object. "
        "No markdown. No code blocks. No backticks. No intro text. No explanation. "
        "Just the JSON object itself, starting with { and ending with }.\n\n"
        "Required JSON keys:\n"
        f'  "Winner": "{d1_name}", "{d2_name}", or "Draw"\n'
        f'  "Score_A": integer 0-100 for {d1_name}\n'
        f'  "Score_B": integer 0-100 for {d2_name}\n'
        '  "Fatal_Flaw": one sentence identifying the worst argument or mistake\n'
        '  "Verdict": 2-3 sentences explaining your judgment\n\n'
        "Output the JSON object now:"
    )

    try:
        resp = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=400,
        )
        raw = resp.choices[0].message.content or ""
        # Strip any accidental markdown fences
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1].lstrip("json").strip()
        return json.loads(raw)
    except Exception as exc:
        return {
            "Winner": "Draw",
            "Score_A": 50,
            "Score_B": 50,
            "Fatal_Flaw": f"Judge error: {exc}",
            "Verdict": "Both debaters argued passionately. A formal verdict could not be rendered.",
        }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

    # Header
    st.markdown(
        '<div class="arena-title">⚔️ Live AI Debate Arena</div>', unsafe_allow_html=True
    )
    st.markdown(
        '<div class="arena-subtitle">'
        "Real-time AI persona battles &nbsp;·&nbsp; "
        '<strong style="color:#38bdf8;">Groq Llama-3 70B</strong> debate engine &nbsp;·&nbsp; '
        '<strong style="color:#a78bfa;">Gemini 2.5 Flash</strong> research &nbsp;·&nbsp; '
        "Hidden Thought (Draft → Critique → Final)"
        "</div>",
        unsafe_allow_html=True,
    )

    # Session state defaults
    defaults = {
        "debate_done": False,
        "chat_history": [],
        "fact_sheets": {},
        "raw_facts": "",
        "briefs": {},
        "verdict": None,
        "saved_d1": "",
        "saved_d2": "",
        "saved_s1": "FOR",
        "saved_s2": "AGAINST",
        "saved_topic": "",
        "saved_rounds": 3,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # Sidebar
    with st.sidebar:
        st.markdown("## 🎮 Arena Controls")
        st.markdown("---")

        topic = st.text_input(
            "🗣️ Debate Topic",
            value="Is artificial intelligence a net positive for humanity?",
            help="Any topic — political, scientific, philosophical, or absurd.",
        )

        names = list(PERSONALITIES.keys())
        display = [f"{p['emoji']} {n}" for n, p in PERSONALITIES.items()]

        st.markdown("### 🥊 Debater 1")
        d1_idx = st.selectbox(
            "Persona",
            range(len(names)),
            format_func=lambda i: display[i],
            index=0,
            key="sel_d1",
        )
        stance1 = st.radio(
            "Stance", ["FOR", "AGAINST"], index=0, horizontal=True, key="s1"
        )

        st.markdown("### 🥊 Debater 2")
        d2_idx = st.selectbox(
            "Persona",
            range(len(names)),
            format_func=lambda i: display[i],
            index=1,
            key="sel_d2",
        )
        stance2 = st.radio(
            "Stance", ["FOR", "AGAINST"], index=1, horizontal=True, key="s2"
        )

        same_persona = d1_idx == d2_idx
        same_stance = stance1 == stance2

        if same_persona:
            st.warning("⚠️ Choose two different personas!")
        if same_stance:
            st.info("💡 Tip: opposite stances create better conflict!")

        rounds = st.slider("🔄 Rounds", min_value=1, max_value=5, value=3)

        st.markdown("---")
        st.markdown("### 📋 Persona Guide")
        for name, p in PERSONALITIES.items():
            st.markdown(f"**{p['emoji']} {name}** — {p['description']}")

        st.markdown("---")
        start_btn = st.button(
            "⚔️  START DEBATE",
            type="primary",
            use_container_width=True,
            disabled=same_persona,
        )

    d1_name = names[d1_idx]
    d2_name = names[d2_idx]
    d1 = PERSONALITIES[d1_name]
    d2 = PERSONALITIES[d2_name]

    # Idle state
    if (
        not start_btn
        and not st.session_state.debate_done
        and not st.session_state.chat_history
    ):
        _, mid, _ = st.columns([1, 2, 1])
        with mid:
            st.markdown(
                """<div style="text-align:center;padding:72px 16px;color:#4b5563;">
                <div style="font-size:4.5rem;">⚔️</div>
                <h3 style="color:#6b7280;">Configure your debate in the sidebar</h3>
                <p>Pick two AI personas, set your topic and number of rounds,<br>
                then hit <strong>START DEBATE</strong> to watch the battle unfold live.</p>
                <p style="font-size:0.82rem;color:#374151;margin-top:12px;">
                🧠 Research: Gemini 2.5 Flash &nbsp;·&nbsp;
                ⚡ Debate + Judge: Groq Llama-3 70B
                </p>
                </div>""",
                unsafe_allow_html=True,
            )
        return

    # ─────────────────────────────────────────────────────────────────────────
    # NEW DEBATE
    # ─────────────────────────────────────────────────────────────────  �──(��────
    if start_btn and not same_persona:
        st.session_state.debate_done = False
        st.session_state.chat_history = []
        st.session_state.fact_sheets = {}
        st.session_state.raw_facts = ""
        st.session_state.briefs = {}
        st.session_state.verdict = None
        st.session_state.saved_d1 = d1_name
        st.session_state.saved_d2 = d2_name
        st.session_state.saved_s1 = stance1
        st.session_state.saved_s2 = stance2
        st.session_state.saved_topic = topic
        st.session_state.saved_rounds = rounds

        # Research phase (Gemini)
        st.markdown("## 🔬 Research Phase")
        prog = st.progress(0)
        status = st.empty()
        raw_facts, briefs = research_topic(topic, d1_name, d2_name, prog, status)
        st.session_state.briefs = briefs
        st.session_state.raw_facts = raw_facts
        st.session_state.fact_sheets = {
            d1_name: briefs.get(d1_name, "No brief generated."),
            d2_name: briefs.get(d2_name, "No brief generated."),
        }
        prog.empty()
        status.empty()
        st.success(
            "✅  Research complete — the debaters are briefed. Let the battle begin!"
        )
        time.sleep(0.5)

        # 15-second cool-down with live countdown
        cooldown = st.empty()
        for i in range(15, 0, -1):
            cooldown.info(
                f"⏳ Debaters are studying their briefs... (Cooling down API — {i}s)"
            )
            time.sleep(1)
        cooldown.empty()

        groq_client = get_groq_client()

        # 3-column Tactical Fact Sheets
        st.markdown("---")
        st.markdown("## 🎯 Tactical Fact Sheets")
        col_a, col_mid, col_b = st.columns([2, 1.5, 2])

        with col_a:
            s1_badge = (
                '<span class="stance-for">▲ FOR</span>'
                if stance1 == "FOR"
                else '<span class="stance-against">▼ AGAINST</span>'
            )
            st.markdown(
                f"""<div class="fact-card fact-sheet">
                <h3 style="color:{d1["color"]};">{d1["emoji"]} {d1_name} {s1_badge}</h3>
                <p style="color:#6b7280;font-size:0.82rem;margin-bottom:8px;">{d1["description"]}</p>
                <hr style="border-color:{d1["color"]}33;margin:6px 0;">
                <pre>{st.session_state.fact_sheets[d1_name]}</pre>
                </div>""",
                unsafe_allow_html=True,
            )

        with col_mid:
            truncated = raw_facts[:800] + ("…" if len(raw_facts) > 800 else "")
            st.markdown(
                f"""<div class="research-card fact-sheet">
                <h4 style="color:#94a3b8;text-align:center;">🌐 Live Research</h4>
                <hr style="border-color:#2d2d2d;margin:6px 0;">
                <pre>{truncated}</pre>
                </div>""",
                unsafe_allow_html=True,
            )

        with col_b:
            s2_badge = (
                '<span class="stance-for">▲ FOR</span>'
                if stance2 == "FOR"
                else '<span class="stance-against">▼ AGAINST</span>'
            )
            st.markdown(
                f"""<div class="fact-card fact-sheet">
                <h3 style="color:{d2["color"]};">{d2["emoji"]} {d2_name} {s2_badge}</h3>
                <p style="color:#6b7280;font-size:0.82rem;margin-bottom:8px;">{d2["description"]}</p>
                <hr style="border-color:{d2["color"]}33;margin:6px 0;">
                <pre>{st.session_state.fact_sheets[d2_name]}</pre>
                </div>""",
                unsafe_allow_html=True,
            )

        # Live Arena
        st.markdown("---")
        st.markdown("## 🏟️ Live Arena")

        for rnd in range(1, rounds + 1):
            st.markdown(
                f'<div class="round-banner">⚔️&nbsp; Round {rnd} of {rounds}</div>',
                unsafe_allow_html=True,
            )

            # Debater 1 turn
            with st.chat_message("user", avatar=d1["emoji"]):
                st.markdown(
                    f"**{d1_name}** "
                    f"<span style='color:#6b7280;font-size:0.8rem;'>[{stance1}]</span>"
                    f'&nbsp;<span class="thinking-tag">💭 Draft→Critique→Final</span>'
                    f'&nbsp;<span class="engine-tag">⚡ Groq Llama-3</span>',
                    unsafe_allow_html=True,
                )
                t1 = st.write_stream(
                    word_stream(
                        hidden_thought_process(
                            groq_client,
                            d1_name,
                            d1,
                            stance1,
                            topic,
                            st.session_state.briefs.get(d1_name, ""),
                            st.session_state.chat_history,
                            rnd,
                            is_rebuttal=(rnd > 1),
                            opponent_name=d2_name,
                        )
                    )
                )
            st.session_state.chat_history.append({"role": d1_name, "content": t1 or ""})
            time.sleep(0.2)

            # Debater 2 turn
            with st.chat_message("assistant", avatar=d2["emoji"]):
                st.markdown(
                    f"**{d2_name}** "
                    f"<span style='color:#6b7280;font-size:0.8rem;'>[{stance2}]</span>"
                    f'&nbsp;<span class="thinking-tag">💭 Draft→Critique→Final</span>'
                    f'&nbsp;<span class="engine-tag">⚡ Groq Llama-3</span>',
                    unsafe_allow_html=True,
                )
                t2 = st.write_stream(
                    word_stream(
                        hidden_thought_process(
                            groq_client,
                            d2_name,
                            d2,
                            stance2,
                            topic,
                            st.session_state.briefs.get(d2_name, ""),
                            st.session_state.chat_history,
                            rnd,
                            is_rebuttal=True,
                            opponent_name=d1_name,
                        )
                    )
                )
            st.session_state.chat_history.append({"role": d2_name, "content": t2 or ""})
            time.sleep(0.2)

        # Judge (Groq)
        st.markdown("---")
        st.markdown("## 🏛️ The Judge Is Deliberating…")
        with st.spinner("Groq Llama-3 70B is analysing the full transcript…"):
            verdict = judge_debate(
                groq_client, topic, d1_name, d2_name, st.session_state.chat_history
            )
        st.session_state.verdict = verdict
        st.session_state.debate_done = True
        st.rerun()

    # ─────────────────────────────────────────────────────────────────────────
    # COMPLETED DEBATE — replay results
    # ─────────────────────────────────────────────────────────────────────────
    if st.session_state.debate_done and st.session_state.chat_history:
        s_d1 = st.session_state.saved_d1
        s_d2 = st.session_state.saved_d2
        st1 = st.session_state.saved_s1
        st2 = st.session_state.saved_s2
        s_topic = st.session_state.saved_topic
        p1 = PERSONALITIES[s_d1]
        p2 = PERSONALITIES[s_d2]

        # Fact sheets
        fs = st.session_state.fact_sheets
        rf = st.session_state.raw_facts
        if fs:
            st.markdown("## 🎯 Tactical Fact Sheets")
            col_a, col_mid, col_b = st.columns([2, 1.5, 2])

            with col_a:
                sb1 = (
                    '<span class="stance-for">▲ FOR</span>'
                    if st1 == "FOR"
                    else '<span class="stance-against">▼ AGAINST</span>'
                )
                st.markdown(
                    f"""<div class="fact-card fact-sheet">
                    <h3 style="color:{p1["color"]};">{p1["emoji"]} {s_d1} {sb1}</h3>
                    <p style="color:#6b7280;font-size:0.82rem;margin-bottom:8px;">{p1["description"]}</p>
                    <hr style="border-color:{p1["color"]}33;margin:6px 0;">
                    <pre>{fs.get(s_d1, "")}</pre>
                    </div>""",
                    unsafe_allow_html=True,
                )

            with col_mid:
                trunc = rf[:800] + ("…" if len(rf) > 800 else "")
                st.markdown(
                    f"""<div class="research-card fact-sheet">
                    <h4 style="color:#94a3b8;text-align:center;">🌐 Live Research</h4>
                    <hr style="border-color:#2d2d2d;margin:6px 0;">
                    <pre>{trunc}</pre>
                    </div>""",
                    unsafe_allow_html=True,
                )

            with col_b:
                sb2 = (
                    '<span class="stance-for">▲ FOR</span>'
                    if st2 == "FOR"
                    else '<span class="stance-against">▼ AGAINST</span>'
                )
                st.markdown(
                    f"""<div class="fact-card fact-sheet">
                    <h3 style="color:{p2["color"]};">{p2["emoji"]} {s_d2} {sb2}</h3>
                    <p style="color:#6b7280;font-size:0.82rem;margin-bottom:8px;">{p2["description"]}</p>
                    <hr style="border-color:{p2["color"]}33;margin:6px 0;">
                    <pre>{fs.get(s_d2, "")}</pre>
                    </div>""",
                    unsafe_allow_html=True,
                )

        # Full transcript
        st.markdown("---")
        st.markdown("## 🏟️ Full Debate Transcript")
        for msg in st.session_state.chat_history:
            role, content = msg["role"], msg["content"]
            if role == s_d1:
                with st.chat_message("user", avatar=p1["emoji"]):
                    s_lbl = st1
                    st.markdown(
                        f"**{role}** <span style='color:#6b7280;font-size:0.8rem;'>[{s_lbl}]</span>",
                        unsafe_allow_html=True,
                    )
                    st.write(content)
            else:
                with st.chat_message("assistant", avatar=p2["emoji"]):
                    s_lbl = st2
                    st.markdown(
                        f"**{role}** <span style='color:#6b7280;font-size:0.8rem;'>[{s_lbl}]</span>",
                        unsafe_allow_html=True,
                    )
                    st.write(content)

        # Verdict card
        v = st.session_state.verdict
        if v:
            winner = v.get("Winner", "Draw")
            score_a = v.get("Score_A", 50)
            score_b = v.get("Score_B", 50)
            fatal_flaw = v.get("Fatal_Flaw", "")
            verdict_tx = v.get("Verdict", "")

            if winner == s_d1:
                wcolor, wemoji = p1["color"], p1["emoji"]
            elif winner == s_d2:
                wcolor, wemoji = p2["color"], p2["emoji"]
            else:
                wcolor, wemoji = "#F39C12", "🤝"

            st.markdown("---")
            st.markdown("## 🏛️ Judge's Verdict")
            st.markdown(
                f"""<div class="verdict-card">
                <h2 style="text-align:center;color:{wcolor};margin-bottom:4px;">{wemoji} {winner} Wins!</h2>
                <p style="text-align:center;color:#9ca3af;font-style:italic;font-size:1rem;margin-bottom:14px;">"{verdict_tx}"</p>
                <div class="fatal-flaw-box">
                    <strong style="color:#f87171;">💀 Fatal Flaw Identified:</strong><br>
                    {fatal_flaw}
                </div>
                <p style="text-align:center;color:#475569;font-size:0.78rem;margin-top:10px;">
                    Judged by ⚡ Groq Llama-3 70B
                </p>
                </div>""",
                unsafe_allow_html=True,
            )

            st.markdown("### 📊 Final Scoreboard")
            c1, c2, c3 = st.columns([2, 1, 2])
            with c1:
                st.metric(
                    f"{p1['emoji']} {s_d1}",
                    f"{score_a} / 100",
                    delta="🏆 Winner" if winner == s_d1 else None,
                )
                st.progress(score_a / 100)
            with c2:
                st.markdown(
                    '<div style="text-align:center;padding-top:26px;font-size:1.5rem;'
                    'color:#E74C3C;font-weight:900;">VS</div>',
                    unsafe_allow_html=True,
                )
            with c3:
                st.metric(
                    f"{p2['emoji']} {s_d2}",
                    f"{score_b} / 100",
                    delta="🏆 Winner" if winner == s_d2 else None,
                )
                st.progress(score_b / 100)

        st.markdown("---")
        if st.button("🔄  New Debate", type="secondary", use_container_width=False):
            for k, v_def in defaults.items():
                st.session_state[k] = (
                    []
                    if isinstance(v_def, list)
                    else {}
                    if isinstance(v_def, dict)
                    else v_def
                )
            st.rerun()


if __name__ == "__main__":
    main()
