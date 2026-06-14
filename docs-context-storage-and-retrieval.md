# บทบาทของ Context: การบันทึกและการค้นคืน (Sparse & Dense)

เอกสารนี้สรุปว่า **context** ที่ generate จาก LLM (ตามแนวทาง [Contextual Retrieval ของ Anthropic](https://www.anthropic.com/engineering/contextual-retrieval)) ถูก **เก็บที่ไหน** และ **ถูกค้นอย่างไร** ในโปรเจกต์นี้ — พร้อมเทียบกับ paper ต้นฉบับ

อ้างอิงโค้ดจริงใน [`authenticRAG.py`](./authenticRAG.py)

---

## 1) Context คืออะไรในระบบนี้

ก่อน index แต่ละ chunk เราเรียก LLM (Qwen ผ่าน OpenRouter) ให้สร้างข้อความสั้น ๆ (50–100 tokens) อธิบายว่า chunk นี้อยู่ตรงไหน/พูดถึงเรื่องอะไรในเอกสารใหญ่ เพื่อเสริมบริบทให้ chunk สามารถถูกค้นเจอได้ดีขึ้น

- Prompt อยู่ที่ [`get_context_prompt()` บรรทัด 165–178](./authenticRAG.py#L165)
- Generate ที่ [`generate_context()` บรรทัด 180–184](./authenticRAG.py#L180) — `max_tokens=100`, `temperature=0.1`

---

## 2) การบันทึก (Indexing)

โค้ดทำที่ [`add_documents_with_context()` บรรทัด 186–262](./authenticRAG.py#L186) — บันทึกลง OpenSearch **2 index แยกกัน**

### 2.1 Vector index (`anthropic-vector-index`) — สำหรับ dense search

```python
contextualized_content = f"{doc.page_content}\n\n{context}"   # chunk + context
embedding = self.encoder.embed_query(contextualized_content)  # embed รวม
```

ฟิลด์ที่เก็บ:

| ฟิลด์ | เนื้อหา | หมายเหตุ |
|---|---|---|
| `embedding` | เวกเตอร์ของ `chunk + context` รวมกัน (1024 มิติ จาก BAAI/bge-m3) | **context "ละลาย" อยู่ในเวกเตอร์** |
| `content` | chunk ดิบ (markdown) | ใช้ดึงคืนเป็น human-readable ตอน retrieve |
| `doc_id`, `chunk_id` | id และเลขลำดับ chunk | metadata |

**สำคัญ:** ตัว `context` ที่ LLM สร้าง **ไม่ได้เก็บเป็น text แยก** ใน vector index — มันอยู่แต่ในเวกเตอร์เท่านั้น ดึงคืนเป็นข้อความไม่ได้

### 2.2 BM25 index (`anthropic-bm25-index`) — สำหรับ sparse search

ฟิลด์ที่เก็บ ([บรรทัด 234–240](./authenticRAG.py#L234)):

| ฟิลด์ | เนื้อหา |
|---|---|
| `content` | chunk ดิบ |
| `contextualized_content` | **เฉพาะ context อย่างเดียว** (ไม่รวม chunk) |
| `doc_id`, `chunk_id`, `original_index` | metadata |

**ต่างจาก paper:** paper บอกให้ `prepend` context เข้าไปใน chunk **ก่อน** สร้าง BM25 index (เก็บเป็นฟิลด์เดียวที่รวมแล้ว) แต่โค้ดนี้เลือก**เก็บแยกฟิลด์** แล้วใช้ `multi_match` ตอนค้นแทน

---

## 3) การค้นคืน (Retrieval)

### 3.1 Dense search — [`dense_search()` บรรทัด 282–301](./authenticRAG.py#L282)

```python
query_embedding = self.encoder.embed_query(query)   # embed query ดิบ ๆ
"knn": {"embedding": {"vector": query_embedding, "k": k}}
```

**กลไก:**
1. embed query (ไม่ได้เสริม context ให้ query)
2. kNN กับฟิลด์ `embedding` ใน vector index
3. เพราะเวกเตอร์ใน index มี context ผสมอยู่ → ระยะห่างเวกเตอร์สะท้อนทั้ง chunk และ context
4. ผลคือ query ที่ตรงกับความหมายของ context (แม้ไม่ตรงกับ chunk ดิบ) ก็ยัง match ได้

**Context ถูกใช้ค้นโดย "อ้อม" ผ่านระยะห่างเวกเตอร์**

**สิ่งที่ retrieve คืนต่อ 1 hit:**

| ตำแหน่ง | ค่า | Human-readable? |
|---|---|---|
| `hit["_id"]` | เช่น `"doc_0"` | ✅ |
| `hit["_score"]` | คะแนน similarity | ✅ |
| `hit["_source"]["content"]` | **chunk ดิบ (ข้อความอ่านได้)** | ✅ ใช่ |
| `hit["_source"]["embedding"]` | เวกเตอร์ 1024 มิติ | ❌ ไม่ใช่ข้อความ |
| `hit["_source"]["doc_id"]`, `chunk_id` | id, ลำดับ | ✅ |

> **Context ที่ LLM สร้างไม่ติดกลับมาใน dense search** เพราะไม่ได้เก็บเป็น field ในเวกเตอร์ index

### 3.2 Sparse search — [`sparse_search()` บรรทัด 264–280](./authenticRAG.py#L264)

```python
"multi_match": {
    "query": query,
    "fields": ["content", "contextualized_content"],   # ค้น 2 ฟิลด์
    "type": "best_fields"
}
```

**กลไก:**
1. ไม่มี embedding — ใช้ keyword matching ของ BM25
2. query match กับ**ทั้ง 2 ฟิลด์** พร้อมกัน: chunk ดิบ + context
3. ใช้ `best_fields` = หยิบคะแนนสูงสุดจากฟิลด์ใดฟิลด์หนึ่ง
4. ถ้าคำใน query โผล่ใน context แม้ไม่โผล่ใน chunk → ยังถูก match

**Context ถูกใช้ค้นโดย "ตรง" ผ่านการ match keyword**

**สิ่งที่ retrieve คืนต่อ 1 hit:**

- `_id`, `_score`
- `_source["content"]` (chunk ดิบ)
- `_source["contextualized_content"]` (**context ที่ LLM generate** — ดึงคืนเป็น text ได้!)
- `_source["doc_id"]`, `chunk_id`, `original_index`

> **ถ้าอยากเห็น context ที่ LLM สร้าง ต้องดึงจาก BM25 index เท่านั้น** vector index ไม่มี

### 3.3 Hybrid search + RRF — [`hybrid_search()` บรรทัด 321–336](./authenticRAG.py#L321)

รัน sparse + dense ขนาน แล้วรวมด้วย Reciprocal Rank Fusion (k=60)
- Chunk ที่ถูกดึงเจอจาก**ทั้ง 2 ทาง** จะถูกดันขึ้นบนสุด
- Chunk ที่ context ช่วยให้ถูกค้นเจอ (จากทางใดทางหนึ่ง) ก็มีโอกาสติดอันดับ

---

## 4) สรุปบทบาทของ Context

| ประเด็น | Dense (vector) | Sparse (BM25) |
|---|---|---|
| Context ถูก embed? | ✅ ใช่ (รวมกับ chunk) | ❌ ไม่ — เก็บเป็น text |
| เก็บที่ไหน | ฟิลด์ `embedding` (ในรูปเวกเตอร์) | ฟิลด์ `contextualized_content` (ในรูป text) |
| ตอนค้นถูกใช้ไหม | ✅ โดยอ้อมผ่านระยะห่างเวกเตอร์ | ✅ โดยตรงผ่าน keyword matching |
| Query ถูกเสริม context? | ❌ ไม่ — embed query ดิบ | ❌ ไม่ — match query ดิบ |
| Context ดึงคืนเป็น text ได้? | ❌ ไม่ได้ (อยู่ในเวกเตอร์) | ✅ ได้ (ฟิลด์ `contextualized_content`) |

**ใจความ:** Context ไม่ได้ถูก inject เข้า query, ไม่ถูกส่งเข้า LLM ตอน retrieve, ไม่ทำ reranking — มันแค่ทำหน้าที่เดียวคือ **เพิ่มโอกาสที่ chunk จะถูกค้นเจอ** ผ่าน 2 ทาง: ดันเวกเตอร์ให้ใกล้ query (dense) + เพิ่มคำให้ BM25 match (sparse)

---

## 5) เทียบกับ Paper ต้นฉบับของ Anthropic

| ประเด็น | Anthropic paper | โค้ดนี้ | ตรงกัน? |
|---|---|---|---|
| Prompt สร้าง context | ใช้คำเดียวกัน | ✅ ใช้คำต่อคำ | ✅ |
| ความยาว context | 50–100 tokens | `max_tokens=100` | ✅ |
| Embedding | embed(`context + chunk`) | embed(`chunk + context`) | 🟡 ลำดับสลับ แต่ embed ตัวรวมเหมือนกัน |
| BM25 index | prepend context เข้า chunk **ก่อน** index เป็นฟิลด์เดียว | เก็บ**แยก 2 ฟิลด์** (`content` + `contextualized_content`) | ❌ ต่าง |
| Sparse retrieval | match query กับฟิลด์รวม | match query กับ 2 ฟิลด์ด้วย `multi_match` `best_fields` | ❌ ต่าง (ผลใกล้เคียงแต่กลไก scoring/IDF ต่าง) |
| Dense retrieval | embed query → kNN เทียบเวกเตอร์รวม | เหมือนกัน | ✅ |
| Hybrid + RRF | ใช้ rank fusion | ใช้ RRF k=60 | ✅ |
| Reranking | บทความเสนอเพิ่ม rerank (ลด failure 67%) | **ยังไม่มี** ในโค้ด | ❌ ขาด |

**บทสรุป:** โค้ดนี้คือ implementation ของ Contextual Retrieval ตามแนวทาง Anthropic อย่างชัดเจน — โดยเฉพาะฝั่ง **dense (embedding) ตรงกับ paper** ส่วนฝั่ง **sparse (BM25) เบี่ยงไปจาก paper เล็กน้อย** (เก็บ context แยกฟิลด์แทน prepend) และยัง**ไม่มี reranking** ซึ่งเป็น optimization ที่ paper บอกว่าให้ผลดีที่สุด

---

## 6) ตัวอย่างเชิงรูปธรรม

สมมุติ chunk เดิม:
> "The company's revenue grew by 3% over the previous quarter."

LLM generate context:
> "This chunk is from an SEC filing on ACME corp's performance in Q2 2023; the previous quarter's revenue was $314 million."

### ใน Vector index:
- `embedding` = vec("The company's revenue grew by 3% over the previous quarter.\n\nThis chunk is from an SEC filing on ACME corp's performance in Q2 2023; the previous quarter's revenue was $314 million.")
- `content` = "The company's revenue grew by 3% over the previous quarter."
- context ตัวข้อความ → **ไม่ได้เก็บแยก**

### ใน BM25 index:
- `content` = "The company's revenue grew by 3% over the previous quarter."
- `contextualized_content` = "This chunk is from an SEC filing on ACME corp's performance in Q2 2023; the previous quarter's revenue was $314 million."

### ตอนค้นด้วย query `"ACME Q2 2023 revenue"`:
- **Dense:** query embed → kNN เทียบเวกเตอร์ที่มี "ACME"/"Q2 2023" อยู่ในส่วน context → similarity สูง → ดึง chunk ขึ้น
- **Sparse:** query มีคำ "ACME", "Q2", "2023" → match ที่ฟิลด์ `contextualized_content` (chunk ดิบไม่มีคำเหล่านี้เลย) → BM25 ให้คะแนน → ดึง chunk ขึ้น
- **Hybrid + RRF:** ทั้งสองทางเจอ → chunk ติดอันดับสูงสุด

→ ในตัวอย่างนี้ ถ้าไม่มี context, chunk จะ**ไม่ถูกค้นเจอเลย** ทั้ง dense (เวกเตอร์ห่าง) และ sparse (ไม่มี keyword ตรง)

นี่คือเหตุผลที่ Contextual Retrieval ช่วยลด retrieval failure rate ได้ 35–49%
