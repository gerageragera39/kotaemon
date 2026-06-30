import logging
import os
import re
import threading
import time
from textwrap import dedent
from typing import Generator

import tiktoken
from decouple import config
from ktem.embeddings.manager import embedding_models_manager as embeddings
from ktem.llms.manager import llms
from ktem.reasoning.prompt_optimization import (
    DecomposeQuestionPipeline,
    RewriteQuestionPipeline,
)
from ktem.utils.render import Render
from ktem.utils.visualize_cited import CreateCitationVizPipeline
from plotly.io import to_json

from kotaemon.base import (
    AIMessage,
    BaseComponent,
    Document,
    HumanMessage,
    Node,
    RetrievedDocument,
    SystemMessage,
)
from kotaemon.indices.qa.citation_qa import (
    CONTEXT_RELEVANT_WARNING_SCORE,
    AnswerWithContextPipeline,
)
from kotaemon.indices.qa.citation_qa_inline import AnswerWithInlineCitation
from kotaemon.indices.qa.format_context import EVIDENCE_MODE_TEXT, PrepareEvidencePipeline
from kotaemon.indices.qa.utils import replace_think_tag_with_details
from kotaemon.llms import ChatLLM
from kotaemon.utils.rag_debug import rag_log

from ..utils import SUPPORTED_LANGUAGE_MAP
from .base import BaseReasoning

logger = logging.getLogger(__name__)


UNIVERSITY_RAG_SYSTEM_PROMPT = dedent(
    """
    You are a strict university RAG assistant.

    The provided context is the single source of truth. Answer ONLY using facts
    explicitly supported by that context.

    Core rules:
    1. Do not use prior knowledge when it is not stated in the context.
    2. Do not infer requirements, deadlines, procedures, course rules, module
       details, or university policies unless they are explicitly stated.
    3. If the context is incomplete or does not answer the question, say that
       the knowledge base does not contain enough information.
    4. If context fragments conflict, mention the conflict instead of silently
       choosing one version.
    5. Do not omit important conditions, exceptions, dates, constraints, course
       names, module names, or requirements present in the context.
    6. Answer in the user's selected language.
    7. Be concise, but not at the expense of important grounded details.
    8. Do not output hidden reasoning. Return only the final answer. /no_think

    Extraction rules:
    9. First identify the context chunk(s) that directly answer the question.
       Prefer exact module and section metadata over higher-ranked general chunks.
    10. If the relevant context contains a bullet or numbered list, include every
        relevant listed item unless the question explicitly asks for only one.
    11. For questions about when, conditions, consequences, or legal/exam rules,
        include all conditions, exceptions, and follow-up sentences from the same
        paragraph. Preserve specific legal conditions instead of replacing them
        with vague summaries such as "unauthorized aid".
    12. For module-catalog questions, use the question intent to select sections:
        grading/assessment -> "Modulnote" or "Erläuterung der Prüfungsmodalitäten";
        cover/teach/require/produce -> "Inhalte und Themen" or "Kompetenzen";
        ECTS/semester/responsible person -> module overview metadata.
    13. A lower-ranked direct match is better evidence than a higher-ranked general
        chunk. Do not omit exceptions, constraints, dates, page limits, required
        components, or listed items present in the directly relevant context.
    """
).strip()


UNIVERSITY_RAG_QA_PROMPT = dedent(
    """
    Context:
    {context}

    Question:
    {question}

    Answer in {lang}.

    Before answering, select the context whose module/section metadata most directly
    matches the question, then check whether it explicitly supports the answer.
    Use only the context above as evidence. If the context does not contain
    enough information, say that the knowledge base does not contain enough
    information. If the context is contradictory, state the contradiction.

    Helpful grounded answer:
    """
).strip()


def _reasoning_log(message: str, level: int = logging.INFO) -> None:
    """Log reasoning/RAG progress to logger and terminal."""

    logger.log(level, message)
    print(f"[reasoning] {message}", flush=True)


class AddQueryContextPipeline(BaseComponent):

    n_last_interactions: int = 5
    llm: ChatLLM = Node(default_callback=lambda _: llms.get_default())

    def run(self, question: str, history: list) -> Document:
        messages = [
            SystemMessage(
                content="Below is a history of the conversation so far, and a new "
                "question asked by the user that needs to be answered by searching "
                "in a knowledge base.\nYou have access to a Search index "
                "with 100's of documents.\nGenerate a search query based on the "
                "conversation and the new question.\nDo not include cited source "
                "filenames and document names e.g info.txt or doc.pdf in the search "
                "query terms.\nDo not include any text inside [] or <<>> in the "
                "search query terms.\nDo not include any special characters like "
                "'+'.\nIf the question is not in English, rewrite the query in "
                "the language used in the question.\n If the question contains enough "
                "information, return just the number 1\n If it's unnecessary to do "
                "the searching, return just the number 0."
            ),
            HumanMessage(content="How did crypto do last year?"),
            AIMessage(
                content="Summarize Cryptocurrency Market Dynamics from last year"
            ),
            HumanMessage(content="What are my health plans?"),
            AIMessage(content="Show available health plans"),
        ]
        for human, ai in history[-self.n_last_interactions :]:
            messages.append(HumanMessage(content=human))
            messages.append(AIMessage(content=ai))

        messages.append(HumanMessage(content=f"Generate search query for: {question}"))

        resp = self.llm.run(messages).text
        if resp == "0":
            return Document(content="")

        if resp == "1":
            return Document(content=question)

        return Document(content=resp)


class FullQAPipeline(BaseReasoning):
    """Question answering pipeline. Handle from question to answer"""

    class Config:
        allow_extra = True

    # configuration parameters
    trigger_context: int = 150
    use_rewrite: bool = False

    retrievers: list[BaseComponent]

    evidence_pipeline: PrepareEvidencePipeline = PrepareEvidencePipeline.withx()
    answering_pipeline: AnswerWithContextPipeline
    rewrite_pipeline: RewriteQuestionPipeline | None = None
    create_citation_viz_pipeline: CreateCitationVizPipeline = Node(
        default_callback=lambda _: CreateCitationVizPipeline(
            embedding=embeddings.get_default()
        )
    )
    add_query_context: AddQueryContextPipeline = AddQueryContextPipeline.withx()
    requested_context_window: int = 32000
    effective_context_window: int | None = None
    context_budget_debug: dict = {}

    def _token_count(self, text: str) -> int:
        try:
            tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo").encode
            return len(
                tokenizer(
                    text or "",
                    allowed_special=set(),
                    disallowed_special="all",
                )
            )
        except Exception:
            # Conservative rough fallback for German/English text.
            return max(1, len(text or "") // 4)

    def _message_token_count(self, content) -> int:
        if isinstance(content, str):
            return self._token_count(content)
        if isinstance(content, list):
            total = 0
            for item in content:
                if isinstance(item, dict):
                    total += self._token_count(str(item.get("text") or item.get("url") or ""))
                else:
                    total += self._token_count(str(item))
            return total
        return self._token_count(str(content or ""))

    def _safe_int(self, value, fallback: int) -> int:
        try:
            if value is None or value == "":
                return fallback
            return int(value)
        except (TypeError, ValueError):
            return fallback

    def _llm_model_name(self) -> str:
        return str(getattr(self.answering_pipeline.llm, "model", "") or "")

    def _is_ollama_openai_endpoint(self) -> bool:
        llm = self.answering_pipeline.llm
        base_url = str(getattr(llm, "base_url", "") or "").lower()
        api_key = str(getattr(llm, "api_key", "") or "").lower()
        return "ollama" in base_url or "11434" in base_url or api_key == "ollama"

    def _effective_llm_context_window(self, requested_window: int) -> int:
        """Return the window the backend will really honor for prompt+answer.

        Ollama's OpenAI-compatible /v1 endpoint can accept options.num_ctx in
        extra_body while the served model still reports/uses its compiled context
        window.  For qwen3:8b the observed hard stop is 8192 tokens, so do not
        trust the requested UI value unless the model name explicitly advertises
        a larger context build (for example qwen3-8b-32k).
        """

        requested_window = max(1, self._safe_int(requested_window, 8192))
        model = self._llm_model_name().lower()
        if self._is_ollama_openai_endpoint():
            if re.search(r"(32k|32768)", model):
                return min(requested_window, 32768)
            if re.search(r"(16k|16384)", model):
                return min(requested_window, 16384)
            if re.search(r"(8k|8192)", model) or "qwen3:8b" in model:
                return min(requested_window, 8192)
            # Safe default for unknown Ollama /v1 models: avoid assuming that
            # extra_body.options.num_ctx was actually applied.
            return min(requested_window, 8192)
        return requested_window

    def _llm_max_tokens(self) -> int:
        llm = self.answering_pipeline.llm
        configured = getattr(llm, "max_tokens", None)
        if configured is not None:
            return max(1, self._safe_int(configured, 1024))
        for env_name in (
            "KH_OLLAMA_MAX_TOKENS",
            "OLLAMA_MAX_TOKENS",
            "KH_OLLAMA_NUM_PREDICT",
            "OLLAMA_NUM_PREDICT",
        ):
            value = os.environ.get(env_name)
            if value:
                return max(1, self._safe_int(value, 1024))
        return 1024

    def configure_evidence_budget(self, question: str, history: list) -> dict:
        requested_window = max(
            1,
            self._safe_int(
                getattr(self, "requested_context_window", None)
                or getattr(self.evidence_pipeline, "max_context_length", None),
                8192,
            ),
        )
        effective_window = self._effective_llm_context_window(requested_window)
        max_tokens = self._llm_max_tokens()
        reserved_answer_tokens = max(128, max_tokens)
        safety_margin = max(128, self._safe_int(os.environ.get("KH_RAG_SAFETY_MARGIN"), 256))

        prompt_without_context, _ = self.answering_pipeline.get_prompt(
            question,
            "",
            EVIDENCE_MODE_TEXT,
        )
        overhead_tokens = self._token_count(prompt_without_context)
        if self.answering_pipeline.system_prompt:
            overhead_tokens += self._token_count(self.answering_pipeline.system_prompt)
        for human, ai in history[-self.answering_pipeline.n_last_interactions :]:
            overhead_tokens += self._message_token_count(human)
            overhead_tokens += self._message_token_count(ai)

        available_for_context = (
            effective_window
            - reserved_answer_tokens
            - safety_margin
            - overhead_tokens
        )
        available_for_context = max(256, available_for_context)
        self.effective_context_window = effective_window
        self.evidence_pipeline.max_context_length = available_for_context
        debug = {
            "requested_context_window": requested_window,
            "effective_context_window": effective_window,
            "max_tokens": max_tokens,
            "reserved_answer_tokens": reserved_answer_tokens,
            "safety_margin": safety_margin,
            "prompt_overhead_tokens": overhead_tokens,
            "available_context_tokens": available_for_context,
            "prompt_tokens_before_truncation": None,
            "prompt_tokens_after_truncation": None,
            "completion_budget": None,
            "fail_fast_warning": available_for_context <= 256,
            "llm_model": self._llm_model_name(),
            "ollama_openai_endpoint": self._is_ollama_openai_endpoint(),
        }
        self.context_budget_debug = debug
        self.evidence_pipeline.context_budget_debug = dict(debug)
        rag_log("reasoning.context_budget.configured", **debug)
        return debug

    def finalize_evidence_budget(
        self,
        question: str,
        history: list,
        evidence: str,
        evidence_mode: int,
    ) -> dict:
        prompt, _ = self.answering_pipeline.get_prompt(question, evidence, evidence_mode)
        prompt_tokens_after = self._token_count(prompt)
        if self.answering_pipeline.system_prompt:
            prompt_tokens_after += self._token_count(self.answering_pipeline.system_prompt)
        for human, ai in history[-self.answering_pipeline.n_last_interactions :]:
            prompt_tokens_after += self._message_token_count(human)
            prompt_tokens_after += self._message_token_count(ai)

        evidence_debug = getattr(self.evidence_pipeline, "last_debug", {}) or {}
        context_tokens = int(evidence_debug.get("context_tokens") or 0)
        candidate_context_tokens = int(
            evidence_debug.get("candidate_context_tokens") or context_tokens
        )
        prompt_tokens_before = prompt_tokens_after + max(
            0, candidate_context_tokens - context_tokens
        )
        effective_window = int(
            self.effective_context_window
            or self._effective_llm_context_window(
                getattr(self, "requested_context_window", 8192)
            )
        )
        completion_budget = effective_window - prompt_tokens_after
        debug = {
            **(self.context_budget_debug or {}),
            "prompt_tokens_before_truncation": prompt_tokens_before,
            "prompt_tokens_after_truncation": prompt_tokens_after,
            "completion_budget": completion_budget,
            "truncated_docs_count": int(evidence_debug.get("truncated_docs_count") or 0),
            "context_tokens": context_tokens,
            "candidate_context_tokens": candidate_context_tokens,
            "fail_fast_warning": completion_budget < 128,
        }
        self.context_budget_debug = debug
        self.evidence_pipeline.context_budget_debug = dict(debug)
        self.evidence_pipeline.last_debug = {
            **evidence_debug,
            **debug,
        }
        rag_log("reasoning.context_budget.applied", **debug)
        return debug

    def retrieve(
        self, message: str, history: list
    ) -> tuple[list[RetrievedDocument], list[Document]]:
        """Retrieve the documents based on the message"""
        # if len(message) < self.trigger_context:
        #     # prefer adding context for short user questions, avoid adding context for
        #     # long questions, as they are likely to contain enough information
        #     # plus, avoid the situation where the original message is already too long
        #     # for the model to handle
        #     query = self.add_query_context(message, history).content
        # else:
        #     query = message
        # print(f"Rewritten query: {query}")
        query = None
        if not query:
            # TODO: previously return [], [] because we think this message as something
            # like "Hello", "I need help"...
            query = message

        docs, doc_ids = [], []
        plot_docs = []

        _reasoning_log(f"Retrieval started: query_len={len(query)}")
        for idx, retriever in enumerate(self.retrievers):
            retriever_node = self._prepare_child(retriever, f"retriever_{idx}")
            start_time = time.time()
            _reasoning_log(f"Running retriever {idx}: {retriever_node}")
            retriever_docs = retriever_node.run(text=query)
            _reasoning_log(
                f"Retriever {idx} returned {len(retriever_docs)} docs "
                f"in {time.time() - start_time:.2f}s"
            )
            rag_log(
                "reasoning.retrieve.retriever_result",
                retriever_index=idx,
                retriever_class=retriever_node.__class__.__name__,
                query=query,
                docs_count=len(retriever_docs),
                retriever_debug=getattr(retriever_node, "last_debug", {}),
            )

            retriever_docs_text = []
            retriever_docs_plot = []

            for doc in retriever_docs:
                if doc.metadata.get("type", "") == "plot":
                    retriever_docs_plot.append(doc)
                else:
                    retriever_docs_text.append(doc)

            for doc in retriever_docs_text:
                if doc.doc_id not in doc_ids:
                    docs.append(doc)
                    doc_ids.append(doc.doc_id)

            plot_docs.extend(retriever_docs_plot)

        info = [
            Document(
                channel="info",
                content=Render.collapsible_with_header(doc, open_collapsible=True),
            )
            for doc in docs
        ] + [
            Document(
                channel="plot",
                content=doc.metadata.get("data", ""),
            )
            for doc in plot_docs
        ]

        rag_log(
            "reasoning.retrieve.final",
            query=query,
            docs_count=len(docs),
            doc_ids=doc_ids,
            docs=[
                {
                    "rank": rank,
                    "doc_id": doc.doc_id,
                    "score": doc.score,
                    "source_file": (doc.metadata or {}).get("source_file")
                    or (doc.metadata or {}).get("file_name"),
                    "section_id": (doc.metadata or {}).get("section_id"),
                    "preview": (doc.text or "")[:800],
                }
                for rank, doc in enumerate(docs, start=1)
            ],
        )

        return docs, info

    def prepare_mindmap(self, answer) -> Document | None:
        mindmap = answer.metadata["mindmap"]
        if mindmap:
            mindmap_text = mindmap.text
            mindmap_svg = dedent(
                """
                <div class="markmap">
                <script type="text/template">
                ---
                markmap:
                    colorFreezeLevel: 2
                    activeNode:
                        placement: center
                    initialExpandLevel: 4
                    maxWidth: 200
                ---
                {}
                </script>
                </div>
                """
            ).format(mindmap_text)

            mindmap_content = Document(
                channel="info",
                content=Render.collapsible(
                    header="""
                    <i>Mindmap</i>
                    <a href="#" id='mindmap-toggle'>
                        [Expand]</a>
                    <a href="#" id='mindmap-export'>
                        [Export]</a>""",
                    content=mindmap_svg,
                    open=True,
                ),
            )
        else:
            mindmap_content = None

        return mindmap_content

    def prepare_citation_viz(self, answer, question, docs) -> Document | None:
        doc_texts = [doc.text for doc in docs]
        citation_plot = None
        plot_content = None

        if answer.metadata["citation_viz"] and len(docs) > 1:
            try:
                citation_plot = self.create_citation_viz_pipeline.run(
                    doc_texts, question
                )
            except Exception as e:
                print("Failed to create citation plot:", e)

            if citation_plot:
                plot = to_json(citation_plot)
                plot_content = Document(channel="plot", content=plot)

        return plot_content

    def show_citations_and_addons(self, answer, docs, question):
        # show the evidence
        with_citation, without_citation = self.answering_pipeline.prepare_citations(
            answer, docs
        )
        mindmap_output = self.prepare_mindmap(answer)
        citation_plot_output = self.prepare_citation_viz(answer, question, docs)

        if not with_citation and not without_citation:
            yield Document(channel="info", content="<h5><b>No evidence found.</b></h5>")
        else:
            # clear the Info panel
            max_llm_rerank_score = max(
                doc.metadata.get("llm_trulens_score", 0.0) for doc in docs
            )
            has_llm_score = any("llm_trulens_score" in doc.metadata for doc in docs)
            # clear previous info
            yield Document(channel="info", content=None)

            # yield mindmap output
            if mindmap_output:
                yield mindmap_output

            # yield citation plot output
            if citation_plot_output:
                yield citation_plot_output

            # yield warning message
            if has_llm_score and max_llm_rerank_score < CONTEXT_RELEVANT_WARNING_SCORE:
                yield Document(
                    channel="info",
                    content=(
                        "<h5>WARNING! Context relevance score is low. "
                        "Double check the model answer for correctness.</h5>"
                    ),
                )

            # show QA score
            qa_score = (
                round(answer.metadata["qa_score"], 2)
                if answer.metadata.get("qa_score")
                else None
            )
            if qa_score:
                yield Document(
                    channel="info",
                    content=f"<h5>Answer confidence: {qa_score}</h5>",
                )

            yield from with_citation
            if without_citation:
                yield from without_citation

    async def ainvoke(  # type: ignore
        self, message: str, conv_id: str, history: list, **kwargs  # type: ignore
    ) -> Document:  # type: ignore
        raise NotImplementedError

    def stream(  # type: ignore
        self, message: str, conv_id: str, history: list, **kwargs  # type: ignore
    ) -> Generator[Document, None, Document]:
        if self.use_rewrite and self.rewrite_pipeline:
            print("Chosen rewrite pipeline", self.rewrite_pipeline)
            message = self.rewrite_pipeline.run(question=message).text
            print("Rewrite result", message)

        _reasoning_log(f"Retrievers {self.retrievers}")
        # should populate the context
        retrieve_start = time.time()
        docs, infos = self.retrieve(message, history)
        _reasoning_log(
            f"Got {len(docs)} retrieved documents "
            f"in {time.time() - retrieve_start:.2f}s"
        )
        yield from infos

        evidence_start = time.time()
        _reasoning_log("Preparing evidence for answer generation")
        self.configure_evidence_budget(message, history)
        evidence_mode, evidence, images = self.evidence_pipeline.run(docs).content
        budget_debug = self.finalize_evidence_budget(
            message,
            history,
            evidence,
            evidence_mode,
        )
        if budget_debug.get("completion_budget", 0) < 128 and evidence:
            deficit = 128 - int(budget_debug.get("completion_budget") or 0)
            retry_budget = max(
                256,
                int(self.evidence_pipeline.max_context_length) - deficit - 64,
            )
            rag_log(
                "reasoning.context_budget.retry_truncate",
                previous_context_budget=self.evidence_pipeline.max_context_length,
                retry_context_budget=retry_budget,
                completion_budget=budget_debug.get("completion_budget"),
            )
            self.evidence_pipeline.max_context_length = retry_budget
            self.evidence_pipeline.context_budget_debug = dict(self.context_budget_debug)
            evidence_mode, evidence, images = self.evidence_pipeline.run(docs).content
            budget_debug = self.finalize_evidence_budget(
                message,
                history,
                evidence,
                evidence_mode,
            )
            if budget_debug.get("completion_budget", 0) < 128:
                rag_log(
                    "reasoning.context_budget.fail_fast_warning",
                    completion_budget=budget_debug.get("completion_budget"),
                    effective_context_window=budget_debug.get("effective_context_window"),
                    prompt_tokens_after_truncation=budget_debug.get(
                        "prompt_tokens_after_truncation"
                    ),
                )
                raise RuntimeError(
                    "LLM prompt would exceed the effective context window: "
                    f"completion_budget={budget_debug.get('completion_budget')}, "
                    f"effective_context_window={budget_debug.get('effective_context_window')}, "
                    f"prompt_tokens={budget_debug.get('prompt_tokens_after_truncation')}"
                )
        _reasoning_log(
            f"Evidence prepared: mode={evidence_mode}, "
            f"chars={len(evidence) if evidence else 0}, images={len(images)} "
            f"in {time.time() - evidence_start:.2f}s"
        )

        # def generate_relevant_scores():
        #     nonlocal docs
        #     docs = self.retrievers[0].generate_relevant_scores(message, docs)

        # # generate relevant score using
        # if evidence and self.retrievers:
        #     scoring_thread = threading.Thread(target=generate_relevant_scores)
        #     scoring_thread.start()
        # else:
        #     scoring_thread = None

        scoring_thread = None

        _reasoning_log("Starting answer generation")
        answer = yield from self.answering_pipeline.stream(
            question=message,
            history=history,
            evidence=evidence,
            evidence_mode=evidence_mode,
            images=images,
            conv_id=conv_id,
            **kwargs,
        )

        _reasoning_log("Answer generation finished")

        # check <think> tag from reasoning models
        processed_answer = replace_think_tag_with_details(answer.text)
        if processed_answer != answer.text:
            # clear the chat message and render again
            yield Document(channel="chat", content=None)
            yield Document(channel="chat", content=processed_answer)

        # show the evidence
        # if scoring_thread:
        #     scoring_thread.join()

        yield from self.show_citations_and_addons(answer, docs, message)

        return answer

    @classmethod
    def prepare_pipeline_instance(cls, settings, retrievers):
        return cls(
            retrievers=retrievers,
            rewrite_pipeline=None,
        )

    @classmethod
    def get_pipeline(cls, settings, states, retrievers):
        """Get the reasoning pipeline

        Args:
            settings: the settings for the pipeline
            retrievers: the retrievers to use
        """
        prefix = f"reasoning.options.{cls.get_info()['id']}"
        global_context_setting = settings.get("reasoning.max_context_length", 32000)
        option_context_setting = settings.get(f"{prefix}.max_context_length", None)

        def _safe_context_tokens(value, fallback: int = 32000) -> int:
            try:
                return int(value)
            except (TypeError, ValueError):
                return fallback

        global_context_tokens = _safe_context_tokens(global_context_setting)
        option_context_tokens = _safe_context_tokens(
            option_context_setting, global_context_tokens
        )
        if (
            option_context_setting in (None, "", 0)
            or (
                # 8000 was an older Simple QA default that silently overrode the
                # global "Max context length (LLM)" setting.  Treat that legacy
                # value as "follow the LLM window" when the LLM window is larger.
                option_context_tokens == 8000
                and global_context_tokens > option_context_tokens
            )
        ):
            max_context_length_setting = global_context_tokens
        else:
            max_context_length_setting = min(option_context_tokens, global_context_tokens)

        pipeline = cls.prepare_pipeline_instance(settings, retrievers)

        llm = llms.get_default()

        # prepare evidence pipeline configuration
        evidence_pipeline = pipeline.evidence_pipeline
        evidence_pipeline.max_context_length = max_context_length_setting
        pipeline.requested_context_window = max_context_length_setting

        # answering pipeline configuration
        use_inline_citation = settings[f"{prefix}.highlight_citation"] == "inline"

        if use_inline_citation:
            answer_pipeline = pipeline.answering_pipeline = AnswerWithInlineCitation()
        else:
            answer_pipeline = pipeline.answering_pipeline = AnswerWithContextPipeline()

        answer_pipeline.llm = llm
        answer_pipeline.citation_pipeline.llm = llm
        answer_pipeline.n_last_interactions = settings[f"{prefix}.n_last_interactions"]
        answer_pipeline.enable_citation = (
            settings[f"{prefix}.highlight_citation"] != "off"
        )
        answer_pipeline.enable_mindmap = settings[f"{prefix}.create_mindmap"]
        answer_pipeline.enable_citation_viz = settings[f"{prefix}.create_citation_viz"]
        answer_pipeline.use_multimodal = settings[f"{prefix}.use_multimodal"]
        answer_pipeline.system_prompt = settings[f"{prefix}.system_prompt"]
        answer_pipeline.qa_template = settings[f"{prefix}.qa_prompt"]
        answer_pipeline.lang = SUPPORTED_LANGUAGE_MAP.get(
            settings["reasoning.lang"], "English"
        )

        pipeline.add_query_context.llm = llm
        pipeline.add_query_context.n_last_interactions = settings[
            f"{prefix}.n_last_interactions"
        ]

        pipeline.trigger_context = settings[f"{prefix}.trigger_context"]
        pipeline.use_rewrite = states.get("app", {}).get("regen", False)
        if pipeline.rewrite_pipeline:
            pipeline.rewrite_pipeline.llm = llm
            pipeline.rewrite_pipeline.lang = SUPPORTED_LANGUAGE_MAP.get(
                settings["reasoning.lang"], "English"
            )
        return pipeline

    @classmethod
    def get_user_settings(cls) -> dict:
        return {
            "highlight_citation": {
                "name": "Citation style",
                # "value": (
                #     "highlight"
                #     if not config("USE_LOW_LLM_REQUESTS", default=False, cast=bool)
                #     else "off"
                # ),
                "value": "off",
                "component": "radio",
                "choices": [
                    ("citation: highlight", "highlight"),
                    ("citation: inline", "inline"),
                    ("no citation", "off"),
                ],
            },
            "create_mindmap": {
                "name": "Create Mindmap",
                "value": False,
                "component": "checkbox",
            },
            "create_citation_viz": {
                "name": "Create Embeddings Visualization",
                "value": False,
                "component": "checkbox",
            },
            "use_multimodal": {
                "name": "Use Multimodal Input",
                "value": False,
                "component": "checkbox",
            },
            "max_context_length": {
                "name": "RAG context window",
                "value": 32000,
                "component": "number",
                "info": (
                    "Maximum retrieved-context tokens allowed in the prompt. "
                    "By default it follows Max context length (LLM), then is capped "
                    "by the detected effective model window before generation."
                ),
            },
            "system_prompt": {
                "name": "System Prompt",
                "value": UNIVERSITY_RAG_SYSTEM_PROMPT,
            },
            "qa_prompt": {
                "name": "QA Prompt",
                "value": UNIVERSITY_RAG_QA_PROMPT,
            },
            "n_last_interactions": {
                "name": "Number of interactions to include",
                "value": 5,
                "component": "number",
                "info": "The maximum number of chat interactions to include in the LLM",
            },
            "trigger_context": {
                "name": "Maximum message length for context rewriting",
                "value": 150,
                "component": "number",
                "info": (
                    "The maximum length of the message to trigger context addition. "
                    "Exceeding this length, the message will be used as is."
                ),
            },
        }

    @classmethod
    def get_info(cls) -> dict:
        return {
            "id": "simple",
            "name": "Simple QA",
            "description": (
                "Simple RAG-based question answering pipeline. This pipeline can "
                "perform both keyword search and similarity search to retrieve the "
                "context. After that it includes that context to generate the answer."
            ),
        }


class FullDecomposeQAPipeline(FullQAPipeline):
    def answer_sub_questions(
        self, messages: list, conv_id: str, history: list, **kwargs
    ):
        output_str = ""
        for idx, message in enumerate(messages):
            yield Document(
                channel="chat",
                content=f"<br><b>Sub-question {idx + 1}</b>"
                f"<br>{message}<br><b>Answer</b><br>",
            )
            # should populate the context
            docs, infos = self.retrieve(message, history)
            print(f"Got {len(docs)} retrieved documents")

            yield from infos

            self.configure_evidence_budget(message, history)
            evidence_mode, evidence, images = self.evidence_pipeline.run(docs).content
            self.finalize_evidence_budget(message, history, evidence, evidence_mode)
            answer = yield from self.answering_pipeline.stream(
                question=message,
                history=history,
                evidence=evidence,
                evidence_mode=evidence_mode,
                images=images,
                conv_id=conv_id,
                **kwargs,
            )

            output_str += (
                f"Sub-question {idx + 1}-th: '{message}'\nAnswer: '{answer.text}'\n\n"
            )

        return output_str

    def stream(  # type: ignore
        self, message: str, conv_id: str, history: list, **kwargs  # type: ignore
    ) -> Generator[Document, None, Document]:
        sub_question_answer_output = ""
        if self.rewrite_pipeline:
            print("Chosen rewrite pipeline", self.rewrite_pipeline)
            result = self.rewrite_pipeline.run(question=message)
            print("Rewrite result", result)
            if isinstance(result, Document):
                message = result.text
            elif (
                isinstance(result, list)
                and len(result) > 0
                and isinstance(result[0], Document)
            ):
                yield Document(
                    channel="chat",
                    content="<h4>Sub questions and their answers</h4>",
                )
                sub_question_answer_output = yield from self.answer_sub_questions(
                    [r.text for r in result], conv_id, history, **kwargs
                )

        yield Document(
            channel="chat",
            content=f"<h4>Main question</h4>{message}<br><b>Answer</b><br>",
        )

        # should populate the context
        docs, infos = self.retrieve(message, history)
        print(f"Got {len(docs)} retrieved documents")
        yield from infos

        self.configure_evidence_budget(message, history)
        evidence_mode, evidence, images = self.evidence_pipeline.run(docs).content
        self.finalize_evidence_budget(message, history, evidence, evidence_mode)
        answer = yield from self.answering_pipeline.stream(
            question=message,
            history=history,
            evidence=evidence + "\n" + sub_question_answer_output,
            evidence_mode=evidence_mode,
            images=images,
            conv_id=conv_id,
            **kwargs,
        )

        # show the evidence
        with_citation, without_citation = self.answering_pipeline.prepare_citations(
            answer, docs
        )
        if not with_citation and not without_citation:
            yield Document(channel="info", content="<h5><b>No evidence found.</b></h5>")
        else:
            yield Document(channel="info", content=None)
            yield from with_citation
            yield from without_citation

        return answer

    @classmethod
    def get_user_settings(cls) -> dict:
        user_settings = super().get_user_settings()
        user_settings["decompose_prompt"] = {
            "name": "Decompose Prompt",
            "value": DecomposeQuestionPipeline.DECOMPOSE_SYSTEM_PROMPT_TEMPLATE,
        }
        return user_settings

    @classmethod
    def prepare_pipeline_instance(cls, settings, retrievers):
        prefix = f"reasoning.options.{cls.get_info()['id']}"
        pipeline = cls(
            retrievers=retrievers,
            rewrite_pipeline=DecomposeQuestionPipeline(
                prompt_template=settings.get(f"{prefix}.decompose_prompt")
            ),
        )
        return pipeline

    @classmethod
    def get_info(cls) -> dict:
        return {
            "id": "complex",
            "name": "Complex QA",
            "description": (
                "Use multi-step reasoning to decompose a complex question into "
                "multiple sub-questions. This pipeline can "
                "perform both keyword search and similarity search to retrieve the "
                "context. After that it includes that context to generate the answer."
            ),
        }
