# Synthetic Shipping Knowledge Base (RAG Demo)
**Updated:** 2025-08-18

This repository contains **fictional** documentation for three placeholder carriers — *Shipping Company A*, *Shipping Company B*, and *Shipping Company C* — plus shared references.  
It’s designed to seed a Retrieval-Augmented Generation (RAG) system during MVP/prototyping. All names, numbers, policies, and examples are **made up for demonstration**.

## Contents
- `company_a_faq.md` — Parcel-focused carrier with Express/Economy services.
- `company_b_faq.md` — Consumer-focused cross‑border shipper with DDP defaults.
- `company_c_faq.md` — Heavy/air cargo oriented carrier with special handling.
- `tracking_status_reference.md` — Canonical status dictionary + synonyms.
- `common_policies.md` — Shared compliance, data privacy, and safety notes.

## Usage Notes
- Clearly label UI and docs with: “**Demo — synthetic data, not affiliated with any brand**.”
- Do **not** use real logos, trademarks, or imply endorsement.
- When you later swap in real docs, **respect each site’s Terms**, rate limits, and robots.txt.
- Store this dataset in your RAG vector store and chunk by headings (`##`/`###`), keeping Q&A pairs intact.
- You may freely copy/modify these files for demos (CC0 / public-domain equivalent).

