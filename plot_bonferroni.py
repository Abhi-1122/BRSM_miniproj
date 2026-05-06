import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import json, os

os.makedirs("output", exist_ok=True)

df = pd.read_csv("GLMM_PostHoc_REC.csv")
bon = df[df["correction"] == "bonferroni"].copy()
bon["condition"] = pd.Categorical(bon["condition"], categories=["Both", "Task-Only", "Item-Only"], ordered=True)
bon = bon.sort_values("condition")

# CI from z (already have SE, df=Inf so z-based CI)
bon["ci_lo"] = bon["estimate"] - 1.96 * bon["SE"]
bon["ci_hi"] = bon["estimate"] + 1.96 * bon["SE"]

# color by significance
def sig_color(p):
    if p < 0.01:   return "#2166AC"
    elif p < 0.05: return "#74ADD1"
    else:          return "#AAAAAA"

bon["color"] = bon["p.value"].apply(sig_color)

fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

y_pos = np.arange(len(bon))
labels = bon["condition"].values

for i, (_, row) in enumerate(bon.iterrows()):
    ax.plot([row["ci_lo"], row["ci_hi"]], [i, i],
            color=row["color"], linewidth=2.5, solid_capstyle="round")
    ax.scatter(row["estimate"], i, color=row["color"], s=120, zorder=5)
    # p-value annotation
    p_str = f"p = {row['p.value']:.3f}" if row["p.value"] >= 0.001 else "p < .001"
    sig = "**" if row["p.value"] < 0.01 else ("*" if row["p.value"] < 0.05 else "ns")
    d_val = row["cohen's d"]
    ax.text(row["ci_hi"] + 0.01, i, f"  {sig}  {p_str}  (d = {d_val:.2f})",
            va="center", fontsize=11, color="#333")

ax.axvline(0, color="#999", linewidth=1.2, linestyle="--")
ax.grid(True, which="major", axis="both", linestyle=":", linewidth=0.8, color="#D0D0D0", alpha=0.9)
ax.set_axisbelow(True)
ax.set_yticks(y_pos)
ax.set_yticklabels(labels, fontsize=13)
ax.set_xlabel("Post vs Mid Contrast (Log-Odds)", fontsize=12)
ax.set_title("Post-Boundary REC Impairment Scales with Boundary Strength\n(Bonferroni-corrected Post vs Mid contrasts per condition)", 
             fontsize=12, fontweight="bold")
ax.set_xlim(-0.55, 0.35)
ax.spines[["top","right"]].set_visible(False)

# legend
from matplotlib.patches import Patch
legend_els = [Patch(fc="#2166AC", label="p < .01"),
              Patch(fc="#74ADD1", label="p < .05"),
              Patch(fc="#AAAAAA", label="ns")]
ax.legend(handles=legend_els, fontsize=10, loc="lower right", frameon=True, framealpha=0.9)

plt.tight_layout()
plt.savefig("output/fig_bonferroni_forest.png", dpi=300, bbox_inches="tight")
plt.close()

with open("output/fig_bonferroni_forest.png.meta.json", "w") as f:
    json.dump({"caption": "Bonferroni Post vs Mid REC Contrasts by Condition",
               "description": "Forest plot showing post-boundary REC impairment scales with boundary strength: Both > Task-Only > Item-Only"}, f)

print("Done")