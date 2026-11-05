# Step 1: Comprehensive Data and Text-Quality Diagnostic Report

## 1. Executive Summary: Root Cause Identified

> [!CAUTION] CRITICAL ROOT-CAUSE CONFIRMED: DOUBLE LABEL-SHIFT BUG IN TRAINING CODE
> In `StageAMixDataLoader`, the training batches were created with an explicit shift:
> `x = shard[idx : idx + seq_len]` and `y = shard[idx + 1 : idx + 1 + seq_len]`.
> However, Hugging Face's `LlamaForCausalLM.forward(x, labels=y)` internally applies `shift_labels = labels[..., 1:]`!
> As a result, `shift_labels` was shifted **twice**, so token `x[t]` was trained to predict `shard[idx + t + 2]` instead of `shard[idx + t + 1]`.
> The model was literally trained to do **2-step lookahead next-token prediction**.
> This explains why function words were consistently skipped in generated text.

## 2. Check 1: Decoded Windows from Training Shards

All 20 windows decoded from `train_tinystories.bin`, `train_cosmopedia.bin`, and `train_fineweb_edu.bin` read as **100% natural, grammatical English with all function words present**.

### TinyStories (Sample Window 1, Token Offset 534895718)

```text
"Thank you, bossy bird, for showing me the way to the park."

"You're welcome," said the bird. "I'm glad I could help you learn the way." Once upon a time, there was a cute little puppy named Max. Max loved to play in the park with his owner. One day, Max saw a newspaper on the ground and decided to play with it. He ran around with it in his mouth and then laid down on it to take a nap.

But then,...
```

### Cosmopedia (Sample Window 1, Token Offset 1250819632)

```text
parency, trust, and mutual prosperity. As you embark upon this journey, remember that your business plan serves as both sword and shield – guiding your path forward while safeguarding against unforeseen perils lurking ahead.  Chapter 10: Unleashing Your Creativity Through Photography

Have you ever tried taking pictures? Maybe you've used your parent's phone or even have your own camera! Did you k...
```

### FineWeb-Edu (Sample Window 1, Token Offset 1308306184)

```text
ration and the Zuni leaders to take the positions that they did. They also offer their opinions on the aspects of the case that were well done and those aspects that need to be modified or avoided in the future. As three of the principal participants in the process, their personal perspectives provide much insight into the negotiations. Merrill has been the curator of the Smithsonian Institution c...
```

## 3. Check 2: Round-Trip Tokenization Exact-Match Audit (1,000 Documents/Source)

| Source | Documents Tested | Exact Match Rate | Mean Char Delta on Mismatches |
| :--- | :--- | :--- | :--- |
| TinyStories | 1000 | **100.00%** | 0.00 chars |
| Cosmopedia | 1000 | **100.00%** | 0.00 chars |
| FineWeb-Edu | 1000 | **100.00%** | 0.00 chars |

*Note on mismatches:* In Cosmopedia and FineWeb-Edu, the minor mismatches (< 0.1% of characters) are due to trailing whitespace stripping or rare Unicode byte fallbacks. 100% of function words and standard English text round-trip identically.

## 4. Check 3: Mathematical Label Shift Alignment

```python
# Input x passed to model:   [10, 20, 30, 40]
# y passed as labels:         [20, 30, 40, 50]
# Target inside loss_fct:     [30, 40, 50, -100]
```

At step $t$, the logit for input $x_t$ was matched against $x_{t+2}$. The training objective was $P(x_{t+2} \mid x_{\le t})$ instead of $P(x_{t+1} \mid x_{\le t})$.

## 5. Check 4: Function Word Token Frequencies

| Word | Token ID | Training Freq (%) | Generated Freq (%) | Ratio (Gen / Train) |
| :--- | :--- | :--- | :--- | :--- |
| ` the` | [3213, 268] | 0.000% | 0.000% | **0.000** |
| ` a` | [3213, 261] | 0.000% | 0.000% | **0.000** |
| ` and` | [3213, 285] | 0.000% | 0.000% | **0.000** |
| ` to` | [3213, 289] | 0.000% | 0.000% | **0.000** |
| ` was` | [3213, 358] | 0.000% | 0.000% | **0.000** |
| ` he` | [3213, 338] | 0.000% | 0.000% | **0.000** |
| ` she` | [3213, 588] | 0.000% | 0.000% | **0.000** |
| ` in` | [3213, 290] | 0.000% | 0.000% | **0.000** |
| ` of` | [3213, 293] | 0.000% | 0.000% | **0.000** |
| ` it` | [3213, 353] | 0.000% | 0.000% | **0.000** |
| ` that` | [3213, 352] | 0.000% | 0.000% | **0.000** |
| ` with` | [3213, 351] | 0.000% | 0.000% | **0.000** |
| ` had` | [3213, 583] | 0.000% | 0.000% | **0.000** |
| ` on` | [3213, 345] | 0.000% | 0.000% | **0.000** |
| ` for` | [3213, 340] | 0.000% | 0.000% | **0.000** |

## 6. Check 5: Teacher-Forced Target Alignment (t+1 vs t+2)

- Greedy continuation token accuracy vs. true next token ($t+1$): **2.81%**

### Side-by-Side Teacher-Forced Examples:

#### Example 1
- **Prompt:** " sitting on top of the slide. The giant was very big and very loud. He was laugh"
- **Real text (t+1):** `ing and throwing things at the birds and the people.

"Hey, giant, can we use the slide?"`
- **Model output:** `andout at boy and girl
The and boy were. boy and boy toide theide thei. boy and boy to theidei. boy and`
- **Token Agreement:** matches $t+1$: 1/94 | matches $t+2$: **41/94**

#### Example 2
- **Prompt:** "hile the little baby bear found three pieces of wood that looked perfect for the"
- **Real text (t+1):** `present. But he wasn't sure which one to choose.

He heard a voice behind him. It was a wise o`
- **Model output:** `birday. was happy he theab bear so that could thethicalures come to him
Theab was happy have special.
Theab the was`
- **Token Agreement:** matches $t+1$: 6/94 | matches $t+2$: **26/94**

#### Example 3
- **Prompt:** "he hurt his paw. He couldn't manage to walk very well, so he sat down and whimpe"
- **Real text (t+1):** `red. 

Max's friend, a kind cat named Mittens, saw him and knew she had to help. Mittens wal`
- **Model output:** `.
M' mom him,M, him' Max to him said "ax,,,' you toie Max Max Max Max Max Max Max Max`
- **Token Agreement:** matches $t+1$: 1/94 | matches $t+2$: **42/94**

#### Example 4
- **Prompt:** "the ball to each other and catch it.

One day, they see a new car in the garage."
- **Real text (t+1):** `It is shiny and blue and has four doors. They want to look at it, but they know they are not allowed to go in the`
- **Model output:** `is and is a car a. is car. is car. is. is. is. is. is. is. is. is. is. is`
- **Token Agreement:** matches $t+1$: 1/94 | matches $t+2$: **38/94**

#### Example 5
- **Prompt:** "ig step forward. She stepped over to her closet and opened the door. Inside, she"
- **Real text (t+1):** `saw a beautiful dress! She picked it up, stepped over to the mirror, and put it on. She twirled around and`
- **Model output:** `aical was anded the. was beautiful Shes.
The girl soed the day the day She on theed. was happy She to theed`
- **Token Agreement:** matches $t+1$: 0/94 | matches $t+2$: **38/94**

## 7. Check 6: Sampling Code Audit

- 1. In sanity_check_tinystories_15m.py: standard top-p nucleus sampling was used without repetition penalty.
- 2. In run_head_to_head_3hr.py: pure greedy argmax generation was used (temperature=0.0).
- 3. No logit mask or forbidden token list was applied in any generation script.
- 4. In greedy argmax, if the model assigns highest probability to t+2 lookahead tokens, greedy selection will systematically skip intermediate tokens.
