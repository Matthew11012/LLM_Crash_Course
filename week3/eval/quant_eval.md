# Quantization Evaluation Report

## Setup
- Base model: GPT-2
- Adapters: LoRA (week2)
- Baselines: FP16 vs Int8 (bitsandbytes)
- Dataset: Wikitext-2 (validation, 100 samples for perplexity)
- Prompts: 14 hand-picked prompts (see appendix)

---

## 1. Perplexity
- FP16: 96.12
- Int8: 95.84
- % Δ: -0.29%

**Criteria:** ≤10–15% increase → [Pass]

---

## 2. Embedding Similarity
- Mean cosine similarity: 0.915

---

## 3. Human Evaluation
- Num of prompts judged “acceptable” for Int8: 10 / 14
- Acceptance rate: 71.4%

**Notes on errors / degradation:**
- Repeating sentences in prompt "Imagine a world where gravity is half as strong. Describe how daily life would change."
- Shorter and less clear answer in prompt "Give step-by-step instructions for tying a shoelace."

---

## 4. Conclusion
- ✅ Int8 quality within acceptable range? [Yes]

---

## Appendix
### Prompts
"Explain why the sky is blue"

"List the steps for making a peanut butter and jelly sandwich"
    
"Write a short story about a robot who learns to paint."
    
"Compare cats and dogs as pets in a few sentences."
    
"Summarize the causes of the French Revolution in 3 sentences."
    
"Write a haiku about winter mornings.:"
    
"Explain how photosynthesis works to a 5th grader."
    
"Translate this sentence into Spanish: Knowledge is power."
    
"What are the advantages and disadvantages of electric cars?"
   
"Give step-by-step instructions for tying a shoelace."
    
"Imagine a world where gravity is half as strong. Describe how daily life would change."
    
"Write a dialogue between a doctor and a patient with a cold."
    
"List the first 5 prime numbers and explain why 4 is not prime."
    
"Compose a short motivational message for someone taking an exam."
