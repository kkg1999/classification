# Active Feature Acquisition Pipeline (`tcga_active_fs_4.py`)

Notation follows the usual feature-selection convention (Guyon & Elisseeff 2003;
Chandrashekar & Sahin 2014): rounded nodes are inputs/outputs, rectangles are
processing steps, diamonds are decisions, and the feedback arcs are the wrapper loop.

```mermaid
flowchart TD
    A([TCGA cohort<br/>expression X, labels y]) --> B[Stratified split<br/>80% train / 20% test]
    B --> C[Train-only ANOVA pre-filter<br/>top 10,000 genes]

    subgraph ACQ [Active acquisition — training data only]
        direction TB
        A0[Cold start:<br/>random panel of 5 genes] --> A1[Baseline CV score<br/>5-fold stratified, fixed folds, macro F1]
        A1 --> Q{Panel ceiling, empty pool<br/>or round budget reached?}
        Q -- no --> B1[Query: sample 200 candidates,<br/>unscreened genes first]
        B1 --> B2[Cheap screening model<br/>-> candidate importances]
        B2 --> B3[Shortlist top 20]
        B3 --> C1[Paired fold-wise gain<br/>of panel + candidate]
        C1 --> C2{gain − λ·SE gain > 0 ?}
        C2 -- yes --> C3[Acquire gene,<br/>update panel scores]
        C2 -- no --> C4[Reject gene]
        C3 --> C5{Round budget of 5 genes<br/>or shortlist exhausted?}
        C4 --> C5
        C5 -- no --> C1
        C5 -- yes --> E1[Log round<br/>cost vs CV curve]
        E1 --> E2{10 barren rounds<br/>AND ≥ 50% pool screened?}
        E2 -- no --> Q
    end

    C --> A0
    Q -- yes --> P([Final gene panel])
    E2 -- yes --> P

    P --> F[Held-out evaluation<br/>accuracy, macro P / R / F1]
    C --> G[Size-matched baselines<br/>ANOVA top-k, random panels, full pool]
    G --> F
    P --> H[Classifier transfer<br/>RF, linear SVM, MLP]
    H --> F
    F --> I{More of the<br/>10 outer splits?}
    I -- yes --> B
    I -- no --> J[Aggregate: mean ± sd, panel stability,<br/>binomial recurrence test with BH-FDR]
    J --> K([Reported tables + best panel<br/>chosen by selection-time CV])
```

**Key design points**

- All selection happens on training data only; the test split is touched once, for reporting.
- Folds are fixed by a single seed for the whole run, so candidate comparisons are *paired*.
- The acceptance gate `gain − λ·SE(gain) > 0` rejects genes whose benefit is inside CV noise.
- Patience cannot trigger before half the pool has been screened, so a cold start cannot stop the search early.
- The per-round history *is* the cost-vs-accuracy curve; no separate panel-size sweep or second argmax is taken.
- The best panel is picked by selection-time CV score, keeping the reported test metric unbiased.
