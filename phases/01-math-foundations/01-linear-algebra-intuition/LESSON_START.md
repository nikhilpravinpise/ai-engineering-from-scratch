# 🎯 First Lesson: Linear Algebra Intuition

Your first lesson is ready to go. Here's exactly what to do.

## Step 1: Understand What You're Building (15 min)

Open this file in your editor:
```
phases/01-math-foundations/01-linear-algebra-intuition/docs/en.md
```

Read from start to **"The Problem"** section. This tells you:
- Why you need linear algebra
- What vectors and matrices really are
- How they connect to AI

**Key concepts to grasp:**
- Vectors = points in space (geometrically)
- Matrices = linear transformations
- Dot product = measure of similarity
- Rank = number of independent dimensions

---

## Step 2: See What You're Implementing (10 min)

Your implementation file is here:
```
phases/01-math-foundations/01-linear-algebra-intuition/code/vectors.py
```

Open it. You'll see a `Vector` class with:
- `__add__()` — add two vectors
- `dot()` — compute dot product
- `magnitude()` — vector length
- `normalize()` — unit vector
- `project_onto()` — vector projection

**Also look for:**
- `is_independent()` function
- Matrix operations
- Gram-Schmidt process (if included)

These are partially or fully written. Your job is to **complete any missing implementations** and make sure all tests pass.

---

## Step 3: Verify Tests Exist (5 min)

Tests tell you what your code must do:
```
cd phases/01-math-foundations/01-linear-algebra-intuition
```

Look for test files:
```
code/tests/test_vectors.py
```

These tests define success. Every test must pass.

---

## Step 4: Run the Tests First (5 min)

See what's failing:
```bash
cd phases/01-math-foundations/01-linear-algebra-intuition
python -m pytest code/tests/ -v
```

or if using unittest:
```bash
python -m unittest discover code/tests -v
```

The output shows which tests are failing. **Don't fix code randomly.** Let the tests guide you.

---

## Step 5: Implement & Fix (45 min)

For each failing test:

1. **Read the test** — understand what it expects
2. **Read the lesson** — find the math explanation
3. **Write the code** — implement the math
4. **Run the test** — verify it passes
5. **Move to next test** — repeat

**Example workflow:**

```python
# In code/vectors.py

# If this test fails:
# def test_dot_product():
#     v1 = Vector([1, 2, 3])
#     v2 = Vector([4, 5, 6])
#     assert v1.dot(v2) == 32

# Then implement:
def dot(self, other):
    return sum(a * b for a, b in zip(self.components, other.components))
    # 1*4 + 2*5 + 3*6 = 32 ✓
```

**Pro tip:** One test at a time. Green light = done, not "kinda working."

---

## Step 6: Run All Tests Until They Pass (5 min)

```bash
python -m pytest code/tests/ -v
```

All tests should show `PASSED`. You're done when:
- ✅ All tests pass
- ✅ No import errors
- ✅ Your code runs without hanging

---

## Step 7: Compare with the Framework (10 min)

After passing all tests, look at how NumPy does the same thing:

```python
import numpy as np

# Your implementation:
v1 = Vector([1, 2, 3])
dot_result = v1.dot(Vector([4, 5, 6]))

# NumPy's way:
v1_np = np.array([1, 2, 3])
v2_np = np.array([4, 5, 6])
dot_result_np = np.dot(v1_np, v2_np)

# Are they the same? Yes!
```

This reinforces that frameworks aren't magic — they just do the same math more efficiently.

---

## Step 8: Take the Quiz (10 min)

Open the quiz after completing the lesson:
```
phases/01-math-foundations/01-linear-algebra-intuition/quiz.json
```

This file has 6 questions (1 pre-lesson, 3 check, 2 post-lesson). 

**Take the post-lesson quiz** (questions 5-6) now that you've learned.

---

## Step 9: Save Your Artifact

Your implementation is an artifact. Keep it:
- Use it as reference in later lessons
- Copy it to your personal projects
- Share it to show you understand the concepts

The `/outputs/` folder in this lesson might contain example uses.

---

## Timeline for Lesson 1

| Step | Time | What | Done? |
|------|------|------|-------|
| 1. Read narrative | 15 min | Understand concepts | ⬜ |
| 2. Review existing code | 10 min | See what you're building | ⬜ |
| 3. Check tests | 5 min | Know what "done" means | ⬜ |
| 4. Run tests | 5 min | See failures | ⬜ |
| 5. Code & fix | 45 min | Implement & iterate | ⬜ |
| 6. All green | 5 min | Verify everything works | ⬜ |
| 7. Compare frameworks | 10 min | See NumPy equivalent | ⬜ |
| 8. Take quiz | 10 min | Test your understanding | ⬜ |
| 9. Archive artifact | 5 min | Save your work | ⬜ |
| **TOTAL** | **105 min** | **Full lesson** | — |

---

## If You Get Stuck

### "A test is failing and I don't know why"
1. Read the test name — it tells you what to check
2. Read the lesson docs — the answer is usually there
3. Print values: `print(f"Got {actual}, expected {expected}")`
4. Adjust your code and rerun

### "I don't understand the math"
1. Reread the lesson narrative (slower this time)
2. Draw it on paper or in your notebook
3. Try a simple example by hand
4. Then code it

### "The code runs but tests fail"
1. Check edge cases (empty vectors, single element, etc.)
2. Verify your formula matches the lesson math
3. Add print statements to debug
4. Ask: "What did the test expect vs. what did I give?"

### "I finished fast and want more"
Great! Move to lesson 02 next, or try:
- Optimize your code for speed
- Extend it with more vector operations
- Visualize vectors with matplotlib

---

## Ready?

Start here:
```bash
cd c:\Dev\ai-engineering-from-scratch\phases\01-math-foundations\01-linear-algebra-intuition
code docs/en.md
```

Read for 15 minutes, then run tests. You've got this! 💪

---

**Next:** Once you finish Lesson 01 (all tests pass + quiz done), move to [Lesson 02: Vectors & Matrices Operations](../02-vectors-matrices-operations/)

---

Questions? Check:
- The lesson narrative (docs/en.md)
- The quiz (quiz.json) for common misconceptions
- Your progress tracker (PROGRESS.md in the root)
