# What We Are Doing, In Plain English

No jargon. If you read only one document, read this one.

---

## 0. Where we are right now

**Phase 0 is done and it passed.** The go/no-go answered `FULL-PROGRAM` — the
best of five possible outcomes. In plain terms:

- Two copies of the same model, trained separately, **agree about which images
  are hard** (0.72). So the thing we're measuring is real, not noise.
- A ResNet and a WideResNet agree **~95% as much as our measurement precision
  allows** (T = 0.946). Compute-need looks like a property of the image.
- Our measurement carries **substantial information that existing "difficulty"
  scores do not** (adding it nearly doubles predictive power). It isn't an old
  idea renamed.
- **One of our own predictions was wrong**, which is the interesting part. We
  expected the three compute dials to be measuring one underlying thing. They
  aren't — depth and resolution are moderately related, precision is nearly
  independent. That means the field's habit of studying only depth and
  generalising is not safe.

Numbers and caveats: `08_PHASE0_RESULTS.md`.

---

## 1. The question, in one paragraph

Modern neural networks can be made to "think harder" on some inputs than others. A photo of a clearly-lit golden retriever is easy — the network could answer after a fraction of its layers. A blurry photo of a dog that looks a bit like a wolf is hard — it needs everything the network has.

Systems that do this are called **adaptive inference**, and they save real compute. But there's a question underneath them that, as far as we can tell, **nobody has actually measured**:

> When an input needs a lot of computation — is that because of the *input*, or because of the *model*?

Put another way: if a big ResNet finds an image hard, will a small Vision Transformer *also* find it hard? Or is "hard" something each model decides for itself?

Everybody in this field has assumed the first answer. Nobody has checked.

**That's the whole project.** We define a way to measure "how much computation does this specific image need", measure it across many different networks, and see whether the answers agree.

---

## 2. Why this matters (why it's publishable)

Today, when a network decides whether to think harder, it asks itself: *"am I confident yet?"*

There's a known problem with that. Small networks are **badly calibrated** — they're confidently wrong a lot. So we're asking the least reliable narrator in the room to make the decision.

The obvious fix: have a big, well-calibrated teacher network tell the small one "this image is hard, spend more on it." That only works **if difficulty is a property of the image**. If it isn't, the teacher's advice is worthless.

So:

- **If compute-need transfers between models** → the teacher-guided approach is justified, and we build it (that's our method, MSC-KD).
- **If it doesn't transfer** → a whole growing line of research rests on a false assumption, and saying so clearly *is the paper*. Arguably a better one.

**We win either way.** That is the deliberate design of this project, and it's why it's worth doing.

---

## 3. What is MSC? (the core idea)

**MSC = Minimum Sufficient Compute.** For one image, it's the answer to:

> *What is the smallest amount of computation at which this network's answer stops changing?*

### An analogy

Imagine reading a book to answer a question about it.

- Some questions you can answer after 20% of the book.
- Some need 60%.
- Some need the whole thing.

MSC is "what fraction of the book did I need?" — a number between 0 and 1 for each question.

### Why a *fraction* and not "number of layers"

Because we want to compare a ResNet to a Vision Transformer, and they have completely different numbers of layers and completely different costs. Saying "ResNet needed 12 layers, ViT needed 6 layers" tells you nothing.

Saying **"ResNet needed 40% of its full cost, ViT needed 40% of its full cost"** is a real comparison. This normalisation is the single most important design choice in the project — it's what makes the question askable at all.

### The subtle bit: "stops changing", not "first gets it right"

Here's something people get wrong. If you check a network at 20%, 40%, 60%, 80%, 100% of its compute, you might see:

| Compute | Answer |
|---|---|
| 20% | cat |
| 40% | **dog** ← correct |
| 60% | cat |
| 80% | dog |
| 100% | dog |

The naive reading is "it only needed 40%!" But that's an **accident**. It got the right answer at 40%, then *lost* it at 60%, then found it again. That's a coincidence, not a property of the image.

So we require the answer to be **settled**: correct at that budget *and every larger budget*. Here the real MSC is 80%, not 40%.

We call this **stable sufficiency**, and it's the reason our definition is more defensible than the obvious one.

### The "too hard for anyone" pile

Some images the full network *itself* isn't confident about. For those, MSC is trivially 100% and the number means nothing.

We put those in a separate pile and report them separately. If we quietly mixed them in, every model would agree on them (they're all "100%"), which would make every model look like it agrees with every other model — **a completely fake result**. Excluding them is essential honesty.

---

## 4. The four ways we reduce compute

We measure MSC along three different "dials" (a fourth is deferred):

| Dial | What we turn down | Analogy |
|---|---|---|
| **Depth** | Stop after fewer layers | Read fewer chapters |
| **Resolution** | Feed a smaller image | Read with blurry glasses |
| **Precision** | Use fewer bits per number | Read a low-quality photocopy |
| *(Width — deferred)* | *Use fewer channels* | *Read a condensed edition* |

Everyone else in this field picks **one** dial (usually depth) and calls it "compute". We measure three, on the same images, with the same model.

That lets us ask something genuinely new: **are these the same thing?** If an image needs more depth, does it also need more resolution? Or are they unrelated?

Nobody knows. We'll find out. Either answer is a result.

---

## 5. The five questions

| | Question | Plain English | Why we care |
|---|---|---|---|
| **Q1** | Is MSC stable? | Train the *same* model twice with different random starts. Do they agree on which images are hard? | If they don't, our measurement is just noise and nothing else matters. **This is the reality check.** |
| **Q2** | Is compute-need one thing? | Do depth, resolution and precision agree on which images are expensive? | Nobody has ever asked. Either answer is publishable. |
| **Q3** | Does it transfer? | Does a ResNet agree with a ViT about which images are hard? | **The main question.** |
| **Q4** | Is this actually new? | Is MSC just "difficulty" — which people already measure — under a new name? | If yes, we're not contributing a new idea. We test this head-on rather than hoping nobody asks. |
| **Q5** | Does the method work? | Can a teacher's compute-advice beat a student's own confidence? | The application. Deliberately last. |

### The noise ceiling — the most important technical idea here

Suppose ResNet and ViT agree 60% of the time about which images are hard. Is that a lot?

**You cannot know without a reference point.**

- If the *same model trained twice* agrees 95% of the time, then 60% across models is a big drop → architecture matters a lot.
- If the *same model trained twice* only agrees 62%, then 60% across models is almost perfect → architecture barely matters.

Same number, opposite conclusions.

So Q1 isn't a side experiment — it's the **denominator**. Every transfer number gets divided by it. Existing papers in the adjacent literature report raw correlations without this, which is why their numbers are hard to interpret. Fixing that is one of our contributions.

This is also why we train **two seeds of every architecture**. It looks like redundancy. It isn't.

---

## 6. What each notebook does

Sixteen notebooks. Run them in order. Each one is self-contained and resumable.

### Setup

| Notebook | Plain English | Time |
|---|---|---|
| **NB00 — Setup & Verify** | Checks everything before you waste GPU hours. Can I reach HuggingFace? Can I *write* to it? Is the dataset there? Do all 15 model types actually build and train? Then it kills a training run halfway and restarts it, to prove resuming works. | 15 min |

**Why NB00 matters:** every check corresponds to a failure you'd otherwise discover three hours in. The resume test especially — a resume that *looks* fine but scrambles the random-number sequence would silently ruin Q1, and Q1 is the denominator of everything.

### Phase 0 — the go/no-go

| Notebook | Plain English | Time |
|---|---|---|
| **NB01 — Phase 0 Train** | Train just 4 models (2 architectures × 2 random seeds). | ~12 GPU-h |
| **NB02 — Phase 0 Measure** | Attach "early exit" points to each trained model, run every image through every compute setting, and measure everything about the finished model — confusion matrix, calibration, latency, energy per image. | ~2 h |
| **NB03 — Phase 0 Decision** | Compute Q1, Q3, Q4 on this small sample. Print a verdict. | 10 min |

**Why Phase 0 exists:** the full project is ~1,200 GPU-hours. Phase 0 is 12. If MSC turns out to be noise, we find out in a week instead of two months. NB03 prints one of five verdicts and tells you plainly what to do next. **Three of the five still lead to a paper.**

**Stop after NB03 and actually look at the answer before continuing.**

### Phase 1 — the atlas

| Notebook | Plain English | Time |
|---|---|---|
| **NB04 — Train ResNets** | 5 ResNet variants × 3 seeds. | ~25 GPU-h |
| **NB05 — Train WideResNets & VGGs** | 5 more architectures × 3 seeds. | ~25 GPU-h |
| **NB06 — Train Mobile nets** | MobileNetV2, ShuffleNetV2. | ~15 GPU-h |
| **NB07 — Train Modern nets** | ConvNeXt, Vision Transformer, MLP-Mixer. Different training recipe — these need AdamW, not SGD. | ~45 GPU-h |
| **NB08 — Measure everything** | Run the compute sweep on all 45 trained models. | ~25 GPU-h |

**Why split training across four notebooks:** so you can run them on different accounts simultaneously, and so a crash in one doesn't cost you the others. NB07 is separated because transformers need a completely different recipe — SGD makes them fail to learn at all.

**Why ViT and Mixer are non-negotiable:** they're the ones with genuinely different "thinking styles" from a ResNet. Our whole prediction for Q3 is that transfer drops when you cross that boundary. Without them, Q3 is just "do ResNets agree with other ResNets", which is much less interesting.

### Analysis — no GPU needed

| Notebook | Plain English | Time |
|---|---|---|
| **NB09 — Q1 Noise ceilings** | For each architecture: how much do two random seeds agree? | 5 min |
| **NB10 — Q2 Are the dials the same?** | Do depth, resolution and precision agree? | 5 min |
| **NB11 — Q3 Transfer** | The big one. Every architecture pair, corrected for noise. Produces the heatmap. | 15 min |
| **NB12 — Q4 Is it new?** | Is MSC explained away by existing difficulty scores? | 20 min |

**Turn the GPU off for these.** They're pure arithmetic on the tables we already saved, and a GPU session burns your weekly quota for nothing.

### The method

| Notebook | Plain English | Time |
|---|---|---|
| **NB13 — Train MSC-KD** | Teach a small model to predict how much compute each image needs, using a big model's answers. | ~120 GPU-h |
| **NB14 — Compare** | Head-to-head against what the field currently does. | ~5 h |

**Only run these if Q3 said transfer works.** If it didn't, the paper is the negative result and this is wasted compute.

### Wrap-up

| Notebook | Plain English | Time |
|---|---|---|
| **NB15 — Paper outputs** | All tables, all figures, energy totals, and a record of which run produced which number. | 10 min |

---

## 7. Running many accounts at once — how the splitting works

This is the part you asked about, and it's the same trick as your image-generation notebooks.

### The problem

Phase 1 is 45 training runs, about 110 GPU-hours. On one account that's roughly two weeks of calendar time. You have six accounts. How do you split the work so that **no two accounts train the same model** and **nothing gets forgotten** — without the accounts being able to talk to each other?

### The trick: decide by arithmetic, not by negotiation

Every notebook has two lines at the top:

```python
NUM_WORKERS = 1     # DEFAULT: this one account does everything
WORKER_ID   = 0     # <<< 0 on account 1, 1 on account 2, ...
```

**The default is 1**, so out of the box one account runs each notebook end to
end. Nothing is lost by that except time. Every notebook also tells you, right
at the top, how many models it contains and roughly how long it will take at 1,
2, 4 or 6 accounts — so you can decide per notebook.

Every account computes the **same** assignment from the **same** list of jobs. Each one keeps only the jobs assigned to its own number.

Because every account runs identical code on identical input, they reach identical conclusions — **without ever communicating**.

This gives you three guarantees for free:

- **No overlap.** A job is assigned to exactly one worker.
- **No gaps.** Every job is assigned to someone.
- **Restart-proof.** The assignment depends only on the job's name — not on when you started, not on who crashed, not on how far anyone else has got.

Your NB05 notebook did this by hashing the image ID. Same idea.

### One improvement over the original

Your image pipeline had ~10,000 jobs of roughly equal size. Hashing splits that beautifully.

Our situation is different: **45 jobs of very unequal size.** A small ResNet takes about an hour; a Vision Transformer takes about six. And 45 items hashed into 6 buckets comes out badly:

```
hash split:      [11, 7, 4, 10, 3, 10] runs per worker    → 4.91x imbalance
```

One account works 33 hours while another finishes in 9 and sits idle. Since the phase isn't done until the *slowest* worker is done, that imbalance directly costs you days.

So we changed the assignment rule. We estimate each job's cost, sort from most expensive to least, and hand each one to whichever worker currently has the least work queued. Result:

```
cost-balanced:   [7, 7, 7, 8, 8, 8] runs per worker       → 1.02x imbalance
```

**Every worker finishes within ~2% of the same time.** For Phase 1 that's roughly 33 hours of wall-clock down to about 19.

It gets better as you go: once real timings exist in the logs, the estimates are replaced with measured seconds-per-epoch, so the scheduler self-corrects.

### What if an account dies?

Each notebook writes a heartbeat every 30 minutes. If a worker goes quiet for over 2 hours, other workers — **once they've finished their own share** — pick up its abandoned jobs automatically.

Own work first, always. So two *live* workers never fight over the same job.

### Just want to run on one account?

That is already the default. Everything works, it just takes longer:

| | 1 account | 6 accounts |
|---|---|---|
| Phase 0 (NB01–03) | ~12 h | ~3 h |
| Atlas (NB04–08) | ~116 h | ~20 h |
| Whole project | ~163 h | ~28 h |

The pipeline pauses cleanly at 8.5 hours and resumes when you start a fresh
session, so a 25-hour notebook is three sessions, not a problem.

---

## 8. What gets saved (everything)

Your instruction was *"save every single detail — we only train once."* That's exactly right, so here's what's recorded **every epoch**, for every run:

**Did it learn?**
train/val loss, train/val accuracy, top-5 accuracy, F1, precision, recall, loss min/max/std

**Was the optimiser healthy?**
learning rate (per parameter group), gradient norm mean/max/95th-percentile, how often gradient clipping kicked in, total weight norm, update-to-weight ratio, AMP loss scale, count of NaN/Inf batches

**Where did the time go?**
epoch time, training vs evaluation time, **time waiting for data vs time computing**, per-step timing at the 50th/90th/99th percentile, throughput, samples seen

**Was the hardware the bottleneck?**
VRAM allocated / reserved / peak / total, GPU utilisation, GPU temperature, SM clock, CPU %, RAM used, free disk

**What did it cost?**
energy in joules and kWh (per epoch and cumulative), CO₂, mean and peak power draw, number of power samples

**Which run was this?**
run ID, worker ID, session ID, hostname, timestamp, architecture, seed, batch size

**How well-calibrated is it?** *(added beyond the original list)*
ECE, MCE, NLL, Brier, mean confidence — because Q5's whole claim is *"small
models are overconfident, so their own confidence is a poor guide"*. Measuring
calibration every epoch turns that from an assertion into evidence.

That's **171 columns per epoch**, plus separate raw streams: **power at 10 Hz per
GPU**, **system stats every second per GPU**, and a **downsampled per-step trace**
so you can see a slowdown *within* an epoch.

And once training finishes, **91 more columns** of final evaluation: confusion
matrix, per-class scores, latency at three batch sizes, inference energy, model
size in three precisions, compression ratios.

Why so much? Because six months from now, when a reviewer asks "why was ViT-Tiny slower per epoch than ConvNeXt despite fewer FLOPs?", you can answer it from the logs. Without `dataload_frac` you'd be guessing, and the session is long gone.

### Saved to HuggingFace, one repository

**`Shanmuk4622/msc-cifar100`** — everything, with one folder per run:

```
runs/{run_id}/
├── metrics/       the CSVs — epochs.csv (171 columns), final.csv, confusion
│                  matrix, per-class table, calibration, latency benchmark
├── telemetry/     raw streams — power at 10 Hz, system at 1 Hz, step traces
├── per_sample/    the per-image measurement tables
├── checkpoints/   model weights
└── env/           exact software and hardware this ran on
```

One repo rather than two, for a reason worth knowing: HuggingFace's write limit
is counted **per user**, not per repository. Two repos meant two commits every
push cycle and double the rate-limit consumption for no benefit.

It's a *dataset* repo rather than a model repo because HuggingFace shows CSV and
Parquet previews for datasets — so every metrics table is browsable in your
browser without downloading anything.

---

## 9. How saving and resuming actually work

### The push schedule

Pushing to HuggingFace on every epoch would hit the rate limit (~128 commits/hour) and get you throttled. So:

| When | What |
|---|---|
| Every 30 minutes | Checkpoint, logs, heartbeat |
| Every 10 epochs | Everything |
| New best accuracy | Best checkpoint (suppressed if <3 epochs since last push) |
| Stage finishes | Everything, including measurement tables |
| **You press Stop** | **Everything, immediately, blocking until confirmed** |
| Kaggle kills the session | Same |
| Something crashes | Same, plus a "failed" marker |
| 8.5 hours elapsed | Everything, marks itself "paused" before Kaggle intervenes |

Three protections against the rate limit:

1. **One commit per push.** Twenty files pushed together = one commit, not twenty.
2. **A budget.** Hard cap of 20 commits/hour per account. Six accounts × 20 = 120, under the 128 limit. When the cap is reached, the uploader *waits* rather than failing.
3. **Skip unchanged files.** Config files don't change; re-sending them is free.

If a rate-limit error arrives anyway, we read the "try again in N seconds" from the response and wait exactly that long.

### Resuming

Kill the session whenever. To resume: **open a fresh session, run all cells.** That's it.

What happens under the hood:

1. Download only the relevant files (never the whole repo — that would blow the 20 GB disk)
2. Rebuild progress from the actual training logs rather than trusting a status file — a session that died mid-push leaves those disagreeing, and the log is the honest one
3. Spot "broken stubs" — runs marked finished that clearly aren't — and reset them
4. Load the checkpoint and **verify the settings haven't changed** since it started
5. Trim the log so resumed epochs don't get double-counted
6. Restore the model, optimiser, scheduler, AMP scaler, **all four random-number generators**, cumulative energy and elapsed time

Point 6 matters more than it looks. Without the random-number state, a resumed run sees images in a different order than an uninterrupted one would. Then "same model, different seed" stops meaning what Q1 needs it to mean — and Q1 is the denominator of the whole project. NB00 tests this explicitly.

If the settings *have* changed since the run started, **it refuses to resume and says so.** That's deliberate. Silently continuing under changed settings is how you end up with numbers that don't reproduce and no idea why.

---

## 10. Where the disk space goes

Kaggle gives you two storage areas, and mixing them up is how sessions die at hour six.

| Location | Size | We use it for |
|---|---|---|
| `/kaggle/working` | 20 GB | **Only** files waiting to be uploaded |
| `/kaggle/temp` | ~1 TB | Datasets, caches, scratch |

Datasets go to `/kaggle/temp`. Always.

And once a run finishes, we **check that HuggingFace actually has the files** (by listing the repo — not by assuming an upload that didn't error succeeded), then delete the local copy. That's what keeps 20 GB sufficient for a 45-run atlas.

---

## 11. How to read the results

### Q1 — noise ceiling

| Value | Meaning |
|---|---|
| ≥ 0.6 | Good. MSC is a real, stable property. |
| 0.4–0.6 | Marginal. Use fewer, more separated compute settings and retry. |
| < 0.4 | MSC is mostly noise. Stop and switch direction. |

### Q3 — transfer (the number that matters)

We report **T**, which is the raw agreement divided by the noise ceiling.

| T | Meaning |
|---|---|
| ≈ 1.0 | Perfect transfer — as good as measurement allows. Compute-need is a property of the *image*. |
| 0.7–0.9 | Strong transfer. The teacher-guided method is justified. |
| 0.5–0.7 | Partial. Something shared, something architecture-specific. |
| < 0.5 | **Poor transfer — and this is the strongest possible result.** It means the field's assumption is wrong. |

Our prediction: within-family (ResNet→ResNet) high, across-CNN moderate, CNN→Transformer low.

### Q4 — is it new?

| ΔR² | Meaning |
|---|---|
| ≥ 0.05 | MSC carries information beyond existing difficulty scores. New object. |
| 0.02–0.05 | Marginal. |
| < 0.02 | MSC ≈ difficulty renamed. **Still publishable** — reframe as "cheap difficulty scores are enough", which is a useful engineering finding. |

### The sanity check you must not skip

Every analysis notebook runs a **shuffled control**: it deliberately scrambles one model's answers and re-measures agreement. That must come out at approximately **zero**.

If it doesn't, there's a bug — almost certainly the two tables' rows aren't lined up. This matters enormously, because misaligned tables produce numbers that look completely reasonable and are completely fictional. The notebooks `assert` on this and stop.

---

## 12. What went wrong along the way (and why that's good)

Six real bugs were found by *running* this, not by reading it. Worth knowing
because they shape why some parts look the way they do:

| What | Why it mattered |
|---|---|
| The rate limiter counted per repository | HuggingFace counts per **user**. Two repos meant double the budget was being used. Fixed by one shared counter — and by dropping to one repo. |
| The shared run-log was losing entries | HuggingFace can't append to a file, so each worker rewriting it erased the others. Two runs were training and the log showed one. Now each worker writes its own file. |
| You couldn't resume your own run | The "someone else is working on this" check didn't check *who*. So after a session paused, your own account was locked out of it for two hours. |
| One model had duplicate compute settings | `resnet8x4` has only 3 layers-groups, so asking for 5 distinct exit points produced three identical ones — which makes "the smallest sufficient budget" meaningless. |
| Two models couldn't run at smaller image sizes | A Vision Transformer needs its position encodings resized; an MLP-Mixer genuinely cannot do it at all. Both are now handled explicitly and the limitation is recorded. |
| The resume test wasn't testing resume | It trained a short run and asked for more epochs — which is a *completion*, not an *interruption*. It never touched the code that matters. Now it kills the run for real and compares the loss curves across the seam. |

The last one is the important lesson: **a test that can't fail for the right
reason is worse than no test**, because it manufactures confidence.

Full detail in `07_REPLICATION_PLAYBOOK.md` §13.

---

## 13. If you only remember five things

1. **We're measuring whether "this image is hard" is a fact about the image or about the model.** Nobody has checked, and both answers are publishable.

2. **Q1 (two seeds of the same model) is the denominator, not a side experiment.** Without it, no transfer number can be interpreted.

3. **Run NB00 first and don't skip the resume test.** It's the cheapest place to catch the expensive mistakes.

4. **Stop after NB03 and read the verdict** before spending the other 1,180 GPU-hours.

5. **Set `WORKER_ID` differently on every account.** That one line is what makes six accounts six times faster instead of six times redundant.

---

## Where to go next

- **To run it:** `04_NOTEBOOK_RUNBOOK.md` — exact steps, settings, troubleshooting
- **What gets saved:** `06_DATA_SCHEMA.md` — the repo structure and every column
- **For design decisions:** `03_IMPLEMENTATION_PLAN.md` — what was built and why
- **For the science:** `00_RESEARCH_PROTOCOL.md` — the formal version of this document
- **To rebuild this elsewhere:** `07_REPLICATION_PLAYBOOK.md` — project-agnostic
