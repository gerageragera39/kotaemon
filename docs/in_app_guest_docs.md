# About KURAGa

**KURAGa** is the **KU Retrieval-Augmented Guide Assistant** — a university-document RAG chatbot for KU Digital Projects.

It helps you ask questions over documents that project admins have indexed. KURAGa is a course project and is **not an official KU service**.

## What can I ask about?

Ask about information that is likely contained in the indexed university/programme documents, for example requirements, module descriptions, procedures, deadlines, or policy text.

As a guest, your chat is automatically scoped to **Search All** admin-indexed documents. You cannot upload files or choose only one document.

## How to ask good questions

- Include the programme, topic, deadline, or rule you care about.
- Ask one focused question at a time.
- Request evidence if you need to verify a detail.
- If the answer is vague, ask a follow-up question with more context.

## What do citations and evidence mean?

KURAGa retrieves passages from indexed documents before generating an answer. Citations/evidence show the passages it used. Use them to verify the response.

A citation does **not** guarantee the answer is correct. The model can misunderstand a source, miss a better source, or combine information incorrectly.

## Limitations

- KURAGa only knows the documents admins indexed.
- Indexed documents may be incomplete or outdated.
- Retrieval can miss relevant passages.
- Generated answers can be wrong.
- For official academic, legal, or administrative decisions, consult official KU documents and staff.

## Privacy and deployment note

This repository is intended for local/course deployment. Do not enter sensitive personal information unless your deployment has been reviewed and approved for that data.

## Attribution

KURAGa is built on [Cinnamon/kotaemon](https://github.com/Cinnamon/kotaemon), Apache-2.0, with substantial KU Digital Projects modifications. Internal package names remain `kotaemon` and `ktem` for compatibility.
