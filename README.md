# Vector Norms Analysis Dashboard

https://github.com/user-attachments/assets/4bdbf0f8-8ff3-4f17-96d8-97062bd6a2b9

## Why I Built This

I was learning about vector norms and got confused about **when to use which norm**. All the theory was there, but I couldn't see the practical differences. So I created this interactive dashboard to actually **see** and **compare** how different norms behave in real applications.

## What It Does

Interactive dashboard showing **L1, L2, and L∞ norms** across 5 different scenarios:

### 1. **Interactive Visualizer**
- Play with vector components
- See how each norm changes
- Visual comparison of norm "balls"

### 2. **Recommendation Systems**
- Which norm finds better similar users?
- Compare similarity scores
- **Result:** L2 usually wins

### 3. **Fraud Detection**
- Which norm catches anomalies better?
- ROC curves comparison
- **Result:** Depends on data, but L2 often best

### 4. **Machine Learning Regularization**
- L1 vs L2 regularization effects
- Feature selection comparison
- **Key:** L1 = sparse, L2 = shrinks all

### 5. **Performance Benchmark**
- Speed comparison of different norms
- **Result:** L∞ fastest, L2 slowest

## Quick Start

```bash
pip install streamlit numpy pandas plotly scikit-learn matplotlib seaborn
streamlit run norms_dashboard.py
```

## When to Use What

| Norm | Best For | Why |
|------|----------|-----|
| **L1** | Sparse solutions, outlier-robust | Sets features to zero |
| **L2** | General purpose, smooth solutions | Balanced, differentiable |
| **L∞** | Minimax problems, fast computation | Focuses on largest component |

## Key Insights I Learned

- **L2 is usually the go-to** for most applications
- **L1 for feature selection** when you want sparse models
- **L∞ for speed** when you need fast computation
- **Context matters** - test different norms for your specific problem

## Files

- `norms_dashboard.py` - Main dashboard
- `README.md` - This file

That's it! Now I actually understand when to use which norm instead of just memorizing formulas.
