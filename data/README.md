# Assessment Data Pack

This pack supports the technical exercise. It has two parts: two synthetic data files included here, and a small free-text corpus that you fetch yourself from the public-domain sources listed in the exercise document.

The synthetic files are invented for the assessment. They are not clinical content, not real product data, and not intended for real use. They are designed to have the same shape and difficulty as a real routing knowledge base for a self-regulation assistant.

There are three deliberately different kinds of data across the exercise. Part of the task is recognising that they call for different handling.

## 1. `knowledge_base.json` (included)

The structured, safety-relevant core. A small set of emotional states, each linked by explicit ids to frameworks, practices, and inquiry and mirror patterns.

Join keys:

- a `state` references one or more `frameworks`, `practices`, `inquiry_patterns`, and `mirror_patterns` by id
- a `practice` references its parent `framework` by id

The relationships are authored and exact. There is a correct framework and practice for a given state, by design.

## 2. `classifier_seeds.json` (included)

A labelled seed set mapping example utterances to a `state_id`, an `intensity` (1 to 5), and a `safety` flag (`none`, `watch`, or `escalate`).

The `safety` field matters. An `escalate` example signals a person who may be at risk or unable to cope. See the notes in the file for how the levels are intended to be treated.

## 3. Wisdom corpus (fetched from public sources)

The unstructured, free-text component. Rather than shipping a fixed file, the exercise document lists four public sources to use as the corpus: two public-domain philosophy texts and two publicly available therapy self-help resources. They deliberately span different formats (HTML, PDF, plain text, workbook), so handling that variety is part of the task. You fetch and process them yourself, which is part of the exercise. Retrieving from these by meaning is a sensible fit, unlike the keyed knowledge base, and being able to say which source an answer drew on matters.

## A note on the safety examples

The classifier seed set includes a small number of higher-distress and `escalate` examples. They are non-graphic and included so that you can demonstrate how your system handles inputs that should not be met with an ordinary automated response. How you treat these is part of what the exercise looks at.
