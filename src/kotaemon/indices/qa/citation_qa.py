import threading
from collections import defaultdict
from typing import Generator

import numpy as np
from decouple import config
from theflow.settings import settings as flowsettings

from kotaemon.base import (
    AIMessage,
    BaseComponent,
    Document,
    HumanMessage,
    Node,
    SystemMessage,
)
from kotaemon.llms import ChatLLM, PromptTemplate
from kotaemon.utils.rag_debug import rag_log

from .citation import CitationPipeline
from .format_context import (
    EVIDENCE_MODE_FIGURE,
    EVIDENCE_MODE_TABLE,
    EVIDENCE_MODE_TEXT,
)
from .utils import find_text

try:
    from ktem.llms.manager import llms
    from ktem.reasoning.prompt_optimization.mindmap import CreateMindmapPipeline
    from ktem.utils.render import Render
except ImportError:
    raise ImportError("Please install `ktem` to use this component")

MAX_IMAGES = 10
CITATION_TIMEOUT = 5.0
CONTEXT_RELEVANT_WARNING_SCORE = config(
    "CONTEXT_RELEVANT_WARNING_SCORE", 0.3, cast=float
)

DEFAULT_QA_TEXT_PROMPT = (
    "Use the following pieces of context to answer the question at the end in detail with clear explanation. "  # noqa: E501
    "If you don't know the answer, just say that you don't know, don't try to "
    "make up an answer. Give answer in "
    "{lang}.\n\n"
    "{context}\n"
    "Question: {question}\n"
    "Helpful Answer:"
)

DEFAULT_QA_TABLE_PROMPT = (
    "Use the given context: texts, tables, and figures below to answer the question, "
    "then provide answer with clear explanation."
    "If you don't know the answer, just say that you don't know, "
    "don't try to make up an answer. Give answer in {lang}.\n\n"
    "Context:\n"
    "{context}\n"
    "Question: {question}\n"
    "Helpful Answer:"
)  # noqa

DEFAULT_QA_CHATBOT_PROMPT = (
    "Pick the most suitable chatbot scenarios to answer the question at the end, "
    "output the provided answer text. If you don't know the answer, "
    "just say that you don't know. Keep the answer as concise as possible. "
    "Give answer in {lang}.\n\n"
    "Context:\n"
    "{context}\n"
    "Question: {question}\n"
    "Answer:"
)  # noqa

DEFAULT_QA_FIGURE_PROMPT = (
    "Use the given context: texts, tables, and figures below to answer the question. "
    "If you don't know the answer, just say that you don't know. "
    "Give answer in {lang}.\n\n"
    "Context: \n"
    "{context}\n"
    "Question: {question}\n"
    "Answer: "
)  # noqa


class AnswerWithContextPipeline(BaseComponent):
    """Answer the question based on the evidence

    Args:
        llm: the language model to generate the answer
        citation_pipeline: generates citation from the evidence
        qa_template: the prompt template for LLM to generate answer (refer to
            evidence_mode)
        qa_table_template: the prompt template for LLM to generate answer for table
            (refer to evidence_mode)
        qa_chatbot_template: the prompt template for LLM to generate answer for
            pre-made scenarios (refer to evidence_mode)
        lang: the language of the answer. Currently support English and Japanese
    """

    llm: ChatLLM = Node(default_callback=lambda _: llms.get_default())
    vlm_endpoint: str = getattr(flowsettings, "KH_VLM_ENDPOINT", "")
    use_multimodal: bool = getattr(flowsettings, "KH_REASONINGS_USE_MULTIMODAL", True)
    citation_pipeline: CitationPipeline = Node(
        default_callback=lambda _: CitationPipeline(llm=llms.get_default())
    )
    create_mindmap_pipeline: CreateMindmapPipeline = Node(
        default_callback=lambda _: CreateMindmapPipeline(llm=llms.get_default())
    )

    qa_template: str = DEFAULT_QA_TEXT_PROMPT
    qa_table_template: str = DEFAULT_QA_TABLE_PROMPT
    qa_chatbot_template: str = DEFAULT_QA_CHATBOT_PROMPT
    qa_figure_template: str = DEFAULT_QA_FIGURE_PROMPT

    enable_citation: bool = False
    enable_mindmap: bool = False
    enable_citation_viz: bool = False

    system_prompt: str = ""
    lang: str = "English"  # support English and Japanese
    n_last_interactions: int = 5

    def get_prompt(self, question, evidence, evidence_mode: int):
        """Prepare the prompt and other information for LLM"""
        if evidence_mode == EVIDENCE_MODE_TEXT:
            prompt_template = PromptTemplate(self.qa_template)
        elif evidence_mode == EVIDENCE_MODE_TABLE:
            prompt_template = PromptTemplate(self.qa_table_template)
        elif evidence_mode == EVIDENCE_MODE_FIGURE:
            if self.use_multimodal:
                prompt_template = PromptTemplate(self.qa_figure_template)
            else:
                prompt_template = PromptTemplate(self.qa_template)
        else:
            prompt_template = PromptTemplate(self.qa_chatbot_template)

        prompt = prompt_template.populate(
            context=evidence,
            question=question,
            lang=self.lang,
        )

        return prompt, evidence

    def run(
        self, question: str, evidence: str, evidence_mode: int = 0, **kwargs
    ) -> Document:
        return self.invoke(question, evidence, evidence_mode, **kwargs)

    def invoke(
        self,
        question: str,
        evidence: str,
        evidence_mode: int = 0,
        images: list[str] = [],
        **kwargs,
    ) -> Document:
        raise NotImplementedError

    async def ainvoke(  # type: ignore
        self,
        question: str,
        evidence: str,
        evidence_mode: int = 0,
        images: list[str] = [],
        **kwargs,
    ) -> Document:
        """Answer the question based on the evidence

        In addition to the question and the evidence, this method also take into
        account evidence_mode. The evidence_mode tells which kind of evidence is.
        The kind of evidence affects:
            1. How the evidence is represented.
            2. The prompt to generate the answer.

        By default, the evidence_mode is 0, which means the evidence is plain text with
        no particular semantic representation. The evidence_mode can be:
            1. "table": There will be HTML markup telling that there is a table
                within the evidence.
            2. "chatbot": There will be HTML markup telling that there is a chatbot.
                This chatbot is a scenario, extracted from an Excel file, where each
                row corresponds to an interaction.

        Args:
            question: the original question posed by user
            evidence: the text that contain relevant information to answer the question
                (determined by retrieval pipeline)
            evidence_mode: the mode of evidence, 0 for text, 1 for table, 2 for chatbot
        """
        raise NotImplementedError

    def stream(  # type: ignore
        self,
        question: str,
        evidence: str,
        evidence_mode: int = 0,
        images: list[str] = [],
        **kwargs,
    ) -> Generator[Document, None, Document]:
        history = kwargs.get("history", [])
        print(f"Got {len(images)} images")
        # check if evidence exists, use QA prompt
        if evidence:
            prompt, evidence = self.get_prompt(question, evidence, evidence_mode)
        else:
            prompt = question
        rag_log(
            "qa.answer.start",
            question=question,
            evidence_chars=len(evidence or ""),
            evidence_preview=(evidence or "")[:2000],
            evidence_mode=evidence_mode,
            images_count=len(images),
            history_turns=len(history),
            llm_class=self.llm.__class__.__name__ if self.llm else None,
            llm_model=getattr(self.llm, "model", None),
        )

        # retrieve the citation
        citation = None
        mindmap = None

        def citation_call():
            nonlocal citation
            citation = self.citation_pipeline.run(context=evidence, question=question)

        def mindmap_call():
            nonlocal mindmap
            mindmap = self.create_mindmap_pipeline.run(
                context=evidence, question=question
            )

        citation_thread = None
        mindmap_thread = None

        # execute function call in thread
        if evidence:
            if self.enable_citation:
                citation_thread = threading.Thread(target=citation_call)
                citation_thread.start()

            if self.enable_mindmap:
                mindmap_thread = threading.Thread(target=mindmap_call)
                mindmap_thread.start()

        output = ""
        logprobs = []
        fallback_attempted = False

        messages = []
        if self.system_prompt:
            messages.append(SystemMessage(content=self.system_prompt))

        for human, ai in history[-self.n_last_interactions :]:
            messages.append(HumanMessage(content=human))
            messages.append(AIMessage(content=ai))

        def _model_name() -> str:
            return str(getattr(self.llm, "model", "") or "").lower()

        def _should_force_no_think() -> bool:
            model = _model_name()
            return "qwen3" in model or "qwen" in model

        def _with_no_think(text: str) -> str:
            if not _should_force_no_think() or "/no_think" in text:
                return text
            return (
                f"{text}\n\n"
                "Do not output hidden reasoning. Return the final answer only. "
                "/no_think"
            )

        prompt = _with_no_think(prompt)
        if self.system_prompt and _should_force_no_think() and "/no_think" not in self.system_prompt:
            messages[0] = SystemMessage(
                content=(
                    f"{messages[0].content}\n"
                    "Do not use chain-of-thought. Return only the final answer. /no_think"
                )
            )

        if self.use_multimodal and evidence_mode == EVIDENCE_MODE_FIGURE:
            # create image message:
            messages.append(
                HumanMessage(
                    content=[
                        {"type": "text", "text": prompt},
                    ]
                    + [
                        {
                            "type": "image_url",
                            "image_url": {"url": image},
                        }
                        for image in images[:MAX_IMAGES]
                    ],
                )
            )
        else:
            # append main prompt
            messages.append(HumanMessage(content=prompt))
        rag_log(
            "qa.answer.messages_ready",
            question=question,
            prompt_chars=len(prompt or ""),
            prompt_preview=(prompt or "")[:3000],
            message_count=len(messages),
            message_roles=[message.__class__.__name__ for message in messages],
            no_think_forced=_should_force_no_think(),
        )

        def _non_streaming_fallback(reason: str, retry_no_think: bool = False) -> str:
            """Run a plain completion when an OpenAI-compatible stream is empty.

            Some local servers finish a streaming request without text (or with
            an obviously truncated single article such as "The") while the same
            prompt succeeds through a normal request.  Without this fallback the
            UI renders "Generate nothing" even though retrieval and evidence are
            already present.
            """

            fallback_messages = messages
            if retry_no_think and messages:
                fallback_messages = list(messages)
                last = fallback_messages[-1]
                if isinstance(last, HumanMessage):
                    fallback_messages[-1] = HumanMessage(
                        content=_with_no_think(
                            f"{last.content}\n\n"
                            "The previous attempt produced no final answer. "
                            "Answer now in one concise paragraph using only the context."
                        )
                    )
            print(f"Streaming produced {reason}; falling back to normal processing")
            rag_log(
                "qa.answer.fallback.start",
                reason=reason,
                retry_no_think=retry_no_think,
                current_output_chars=len(output or ""),
                fallback_message_count=len(fallback_messages),
            )
            fallback = self.llm.run(fallback_messages)
            reasoning = (
                getattr(fallback, "additional_kwargs", {}) or {}
            ).get("reasoning_content")
            if reasoning and not (getattr(fallback, "text", None) or getattr(fallback, "content", None)):
                print("Non-streaming fallback returned reasoning_content without final content")
            fallback_text = (
                getattr(fallback, "text", None)
                or getattr(fallback, "content", None)
                or ""
            )
            rag_log(
                "qa.answer.fallback.end",
                reason=reason,
                retry_no_think=retry_no_think,
                fallback_output_chars=len(fallback_text),
                fallback_output_preview=fallback_text[:1200],
                reasoning_chars=len(str(reasoning or "")),
                completion_tokens=getattr(fallback, "completion_tokens", None),
                total_tokens=getattr(fallback, "total_tokens", None),
            )
            return fallback_text

        def _looks_incomplete_generation(text: str) -> bool:
            stripped = " ".join((text or "").strip().split())
            if not stripped:
                return True
            lowered = stripped.lower().strip(" .,:;!?")
            if lowered in {
                "a",
                "an",
                "the",
                "die",
                "der",
                "das",
                "um",
                "based",
                "auf",
                "im",
                "in",
            }:
                return True
            words = stripped.split()
            return len(words) <= 4 and stripped[-1:] not in ".!?…"

        try:
            # try streaming first
            print("Trying LLM streaming")
            rag_log("qa.answer.stream.start", message_count=len(messages))
            for out_msg in self.llm.stream(messages):
                text_delta = out_msg.text or ""
                output += text_delta
                logprobs += out_msg.logprobs
                if text_delta:
                    yield Document(channel="chat", content=text_delta)
            rag_log(
                "qa.answer.stream.end",
                output_chars=len(output),
                output_preview=output[:1200],
                incomplete=_looks_incomplete_generation(output),
            )
        except NotImplementedError:
            fallback_attempted = True
            rag_log("qa.answer.stream.unsupported")
            output = _non_streaming_fallback("unsupported")
            if _looks_incomplete_generation(output):
                output = _non_streaming_fallback("unsupported and empty", retry_no_think=True)
            if output:
                yield Document(channel="chat", content=output)
        except Exception as exc:
            print(f"Streaming failed after {len(output)} chars: {exc!r}")
            rag_log(
                "qa.answer.stream.error",
                error=repr(exc),
                partial_output_chars=len(output),
                partial_output_preview=output[:1200],
            )
            if _looks_incomplete_generation(output):
                if output:
                    yield Document(channel="chat", content=None)
                fallback_attempted = True
                try:
                    output = _non_streaming_fallback(f"stream error: {exc!r}")
                    if _looks_incomplete_generation(output):
                        output = _non_streaming_fallback(
                            f"stream error retry: {exc!r}",
                            retry_no_think=True,
                        )
                    if output:
                        yield Document(channel="chat", content=output)
                except Exception as fallback_exc:
                    raise RuntimeError(
                        "LLM generation failed during streaming and fallback"
                    ) from fallback_exc
            else:
                raise

        if (
            not fallback_attempted
            and _looks_incomplete_generation(output)
        ):
            rag_log(
                "qa.answer.incomplete_after_stream",
                output_chars=len(output),
                output_preview=output[:1200],
            )
            if output:
                # Clear the partial one-word stream before replacing it with the
                # non-streaming answer.
                yield Document(channel="chat", content=None)
            fallback_attempted = True
            fallback_output = _non_streaming_fallback("empty or incomplete text")
            if _looks_incomplete_generation(fallback_output):
                fallback_output = _non_streaming_fallback(
                    "empty or incomplete text after fallback",
                    retry_no_think=True,
                )
            if fallback_output:
                output = fallback_output
                yield Document(channel="chat", content=output)
        if logprobs:
            qa_score = np.exp(np.average(logprobs))
        else:
            qa_score = None
        rag_log(
            "qa.answer.final",
            output_chars=len(output),
            output_preview=output[:2000],
            fallback_attempted=fallback_attempted,
            qa_score=qa_score,
        )

        if citation_thread:
            citation_thread.join(timeout=CITATION_TIMEOUT)
        if mindmap_thread:
            mindmap_thread.join(timeout=CITATION_TIMEOUT)

        answer = Document(
            text=output,
            metadata={
                "citation_viz": self.enable_citation_viz,
                "mindmap": mindmap,
                "citation": citation,
                "qa_score": qa_score,
            },
        )

        return answer

    def match_evidence_with_context(self, answer, docs) -> dict[str, list[dict]]:
        """Match the evidence with the context"""
        spans: dict[str, list[dict]] = defaultdict(list)

        if not answer.metadata["citation"]:
            return spans

        evidences = answer.metadata["citation"].evidences
        for quote in evidences:
            matched_excerpts = []
            for doc in docs:
                matches = find_text(quote, doc.text)

                for start, end in matches:
                    if "|" not in doc.text[start:end]:
                        spans[doc.doc_id].append(
                            {
                                "start": start,
                                "end": end,
                            }
                        )
                        matched_excerpts.append(doc.text[start:end])

            # print("Matched citation:", quote, matched_excerpts),
        return spans

    def prepare_citations(self, answer, docs) -> tuple[list[Document], list[Document]]:
        """Prepare the citations to show on the UI"""
        with_citation, without_citation = [], []
        has_llm_score = any("llm_trulens_score" in doc.metadata for doc in docs)

        spans = self.match_evidence_with_context(answer, docs)
        id2docs = {doc.doc_id: doc for doc in docs}
        doc_order = {doc.doc_id: idx for idx, doc in enumerate(docs)}
        not_detected = [doc.doc_id for doc in docs if doc.doc_id not in spans]

        def _score(value, default: float = 0.0) -> float:
            try:
                return float(value)
            except (TypeError, ValueError):
                return default

        def _vector_score(doc) -> float | None:
            # -1.0 is the sentinel for full-text-only hits. Vector-only retrieval
            # carries the actual similarity in doc.score, so use it as the fallback
            # when LLM/reranker scores are missing or tied at 0.
            score = _score(
                doc.metadata.get("retrieval_score", getattr(doc, "score", None)),
                default=-1.0,
            )
            return None if score == -1.0 else score

        def _ranking_key(doc_id: str) -> tuple:
            doc = id2docs[doc_id]
            llm_score = doc.metadata.get("llm_trulens_score")
            reranking_score = doc.metadata.get("reranking_score")
            vector_score = _vector_score(doc)
            return (
                1 if llm_score is not None else 0,
                _score(llm_score),
                1 if reranking_score is not None else 0,
                _score(reranking_score),
                1 if vector_score is not None else 0,
                _score(vector_score),
                -doc_order[doc_id],
            )

        def _display_score(doc) -> float:
            llm_score = doc.metadata.get("llm_trulens_score")
            if llm_score is not None and _score(llm_score) > 0:
                return _score(llm_score)

            reranking_score = doc.metadata.get("reranking_score")
            if reranking_score is not None and _score(reranking_score) > 0:
                return _score(reranking_score)

            return _score(_vector_score(doc))

        # render highlight spans
        for _id, ss in spans.items():
            if not ss:
                if _id not in not_detected:
                    not_detected.append(_id)
                continue
            cur_doc = id2docs[_id]
            highlight_text = ""

            ss = sorted(ss, key=lambda x: x["start"])
            last_end = 0
            text = cur_doc.text[: ss[0]["start"]]

            for idx, span in enumerate(ss):
                # prevent overlapping between span
                span_start = max(last_end, span["start"])
                span_end = max(last_end, span["end"])

                to_highlight = cur_doc.text[span_start:span_end]
                last_end = span_end

                # append to highlight on PDF viewer
                highlight_text += (" " if highlight_text else "") + to_highlight

                span_idx = span.get("idx", None)
                if span_idx is not None:
                    to_highlight = f"【{span_idx}】" + to_highlight

                text += Render.highlight(
                    to_highlight,
                    elem_id=str(span_idx) if span_idx is not None else None,
                )
                if idx < len(ss) - 1:
                    text += cur_doc.text[span["end"] : ss[idx + 1]["start"]]

            text += cur_doc.text[ss[-1]["end"] :]
            # add to display list
            with_citation.append(
                Document(
                    channel="info",
                    content=Render.collapsible_with_header_score(
                        cur_doc,
                        override_text=text,
                        highlight_text=highlight_text,
                        open_collapsible=True,
                    ),
                )
            )

        print("Got {} cited docs".format(len(with_citation)))

        sorted_not_detected = sorted(
            not_detected,
            key=_ranking_key,
            reverse=True,
        )

        for id_ in sorted_not_detected:
            doc = id2docs[id_]
            doc_score = _display_score(doc)
            is_open = not has_llm_score or (
                doc_score
                > CONTEXT_RELEVANT_WARNING_SCORE
                # and len(with_citation) == 0
            )
            without_citation.append(
                Document(
                    channel="info",
                    content=Render.collapsible_with_header_score(
                        doc, open_collapsible=is_open
                    ),
                )
            )
        return with_citation, without_citation
