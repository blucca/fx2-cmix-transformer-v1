# Fixed text for the paired language-loss probe

Credit: **National Institute of Standards and Technology**, *NIST AI 100-1:
Artificial Intelligence Risk Management Framework (AI RMF 1.0)*, January 2023.
[Official PDF](https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf) ·
[DOI](https://doi.org/10.6028/NIST.AI.100-1) ·
[NIST copyright/disclaimer](https://www.nist.gov/disclaimer).

The NIST permission statement reads:

> With the exception of material marked as copyrighted, information presented
> on NIST sites are considered public information and may be distributed or
> copied. Use of appropriate byline/photo/image credits is requested.

`fixed-language-text-2048.txt` contains selected **government narrative** from
its Executive Summary and sections 1.2.1–1.2.3. Printed pages 1, 2, 3, 5, 6 and
7 correspond to physical PDF pages 6, 7, 8, 10, 11 and 12. Graphics, captions,
and the separately attributed ISO/OECD definition passages are excluded.

The original extraction used `pdftotext -layout`. Page headers/footers were
removed, lower-case words broken at a line-ending hyphen were joined, and
whitespace was collapsed to one space. The separately attributed definition
blocks were removed before the experiment.

The first text contained 1,935 tokenizer tokens, below the frozen 2,048-target
minimum. **Before any language-model inference or NLL observation**, the
remaining government paragraphs in section 1.2.2 and the first paragraph of
1.2.3 were appended from the same cached PDF, after its attributed ISO
passage. The final fixed text is **12,626 UTF-8 bytes / 2,285 available tokens**
with the pinned companion GPT-2 tokenizer. No source, quantizer or model was
selected using a measured language result.

The probe scores the first **2,048 tokens** as two independent 1,024-target
blocks. Each block starts with BOS token 50256 as an unscored condition, and
each model carries its own KV history. The rest of the text remains in the
file so the same first 2,048 tokens can be reproduced. Results are a paired
loss measurement for this one fixed government document.
