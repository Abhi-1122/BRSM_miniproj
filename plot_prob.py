import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import json, os

os.makedirs("output", exist_ok=True)

COLORS = {"Item-Only": "#4878CF", "Task-Only": "#E8762B", "Both": "#54A868"}
ANNO_FONTSIZE = 16

# --- Load CSVs ---
rec = pd.read_csv("GLMM_REC.csv")
ldi = pd.read_csv("GLMM_LDI.csv")

rec_se = rec.set_index("term")["SE_log odds"].to_dict()
ldi_se = ldi.set_index("term")["SE_log odds"].to_dict()
rec_b  = rec.set_index("term")["log odds"].to_dict()
ldi_b  = ldi.set_index("term")["log odds"].to_dict()

def logistic(z):
    return 1.0 / (1.0 + np.exp(-z))

# ============================================================
# REC PLOT  (x-order: Post, Mid, Pre)
# ============================================================
# coefficients
r_int   = rec_b["(Intercept)"]
r_post  = rec_b["pos_post"]
r_pre   = rec_b["pos_pre"]
r_task  = rec_b["grp_task"]
r_both  = rec_b["grp_both"]
r_pt    = rec_b["pos_post:grp_task"]
r_pb    = rec_b["pos_post:grp_both"]
r_pret  = rec_b["grp_task:pos_pre"]
r_preb  = rec_b["grp_both:pos_pre"]

# SE
se_int  = rec_se["(Intercept)"]
se_post = rec_se["pos_post"]
se_pre  = rec_se["pos_pre"]
se_task = rec_se["grp_task"]
se_both = rec_se["grp_both"]
se_pt   = rec_se["pos_post:grp_task"]
se_pb   = rec_se["pos_post:grp_both"]
se_pret = rec_se["grp_task:pos_pre"]
se_preb = rec_se["grp_both:pos_pre"]

# log-odds per [Post, Mid, Pre]
logit = {
    "Item-Only": np.array([r_int + r_post, r_int, r_int + r_pre]),
    "Task-Only": np.array([r_int+r_post+r_task+r_pt, r_int+r_task, r_int+r_pre+r_task+r_pret]),
    "Both":      np.array([r_int+r_post+r_both+r_pb, r_int+r_both, r_int+r_pre+r_both+r_preb]),
}
se_lo = {
    "Item-Only": np.array([np.sqrt(se_int**2+se_post**2), se_int, np.sqrt(se_int**2+se_pre**2)]),
    "Task-Only": np.array([np.sqrt(se_int**2+se_post**2+se_task**2+se_pt**2),
                           np.sqrt(se_int**2+se_task**2),
                           np.sqrt(se_int**2+se_pre**2+se_task**2+se_pret**2)]),
    "Both":      np.array([np.sqrt(se_int**2+se_post**2+se_both**2+se_pb**2),
                           np.sqrt(se_int**2+se_both**2),
                           np.sqrt(se_int**2+se_pre**2+se_both**2+se_preb**2)]),
}

x = np.array([0, 1, 2])
xlabels = ["Post", "Mid", "Pre"]

fig, ax = plt.subplots(figsize=(12, 8))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

ax.axvspan(-0.5, 0.5, alpha=0.07, color="gray")  # highlight Post region

for grp in ["Item-Only", "Task-Only", "Both"]:
    p  = logistic(logit[grp])
    lo = logistic(logit[grp] - 1.96*se_lo[grp])
    hi = logistic(logit[grp] + 1.96*se_lo[grp])
    ax.plot(x, p, marker="o", linewidth=2.5, color=COLORS[grp], label=grp, markersize=8)
    ax.fill_between(x, lo, hi, color=COLORS[grp], alpha=0.15)

# annotations at Post (x=0)
p_task_post = logistic(logit["Task-Only"][0])
p_both_post = logistic(logit["Both"][0])
ax.annotate("Post × Task-Only: OR=0.74, p=.021",
            xy=(0, p_task_post), xytext=(0.35, p_task_post+0.06),
            arrowprops=dict(arrowstyle="->", lw=1.2, color="#444"),
            fontsize=ANNO_FONTSIZE, fontweight="bold", color="#333")
ax.annotate("Post × Both: OR=0.67, p=.002",
            xy=(0, p_both_post), xytext=(0.35, p_both_post-0.09),
            arrowprops=dict(arrowstyle="->", lw=1.2, color="#444"),
            fontsize=ANNO_FONTSIZE, fontweight="bold", color="#333")

ax.set_xticks(x)
ax.set_xticklabels(xlabels, fontsize=14)
ax.set_ylabel("Predicted P(Old | Target) — GLMM OR estimate", fontsize=13)
ax.set_xlabel("Boundary Position", fontsize=13)
ax.set_title("GLMM-Predicted Recognition Probability (REC) by Position and Group", fontsize=14, fontweight="bold")
ax.legend(frameon=True, fontsize=12)
ax.spines[["top","right"]].set_visible(False)
ax.tick_params(axis='y', labelsize=12)
plt.tight_layout()
plt.savefig("output/fig_REC_predicted_prob.png", dpi=300, bbox_inches="tight")
plt.close()

# ============================================================
# LDI PLOT  (x-order: Post, Mid, Pre)
# ============================================================
l_int  = ldi_b["(Intercept)"]
l_pre  = ldi_b["pos_pre"]
l_task = ldi_b["grp_task"]
l_both = ldi_b["grp_both"]
l_pret = ldi_b["pos_pre:grp_task"]
l_preb = ldi_b["pos_pre:grp_both"]

sl_int  = ldi_se["(Intercept)"]
sl_pre  = ldi_se["pos_pre"]
sl_task = ldi_se["grp_task"]
sl_both = ldi_se["grp_both"]
sl_pret = ldi_se["pos_pre:grp_task"]
sl_preb = ldi_se["pos_pre:grp_both"]

# LDI model has no pos_post term — Post = baseline intercept + group
logit_ldi = {
    "Item-Only": np.array([l_int,        l_int,        l_int+l_pre]),
    "Task-Only": np.array([l_int+l_task, l_int+l_task, l_int+l_pre+l_task+l_pret]),
    "Both":      np.array([l_int+l_both, l_int+l_both, l_int+l_pre+l_both+l_preb]),
}
se_ldi = {
    "Item-Only": np.array([sl_int, sl_int, np.sqrt(sl_int**2+sl_pre**2)]),
    "Task-Only": np.array([np.sqrt(sl_int**2+sl_task**2),
                           np.sqrt(sl_int**2+sl_task**2),
                           np.sqrt(sl_int**2+sl_pre**2+sl_task**2+sl_pret**2)]),
    "Both":      np.array([np.sqrt(sl_int**2+sl_both**2),
                           np.sqrt(sl_int**2+sl_both**2),
                           np.sqrt(sl_int**2+sl_pre**2+sl_both**2+sl_preb**2)]),
}

fig2, ax2 = plt.subplots(figsize=(12, 8))
fig2.patch.set_facecolor("white")
ax2.set_facecolor("white")

ax2.axvspan(1.5, 2.5, alpha=0.07, color="teal")  # highlight Pre region

for grp in ["Item-Only", "Task-Only", "Both"]:
    p  = logistic(logit_ldi[grp])
    lo = logistic(logit_ldi[grp] - 1.96*se_ldi[grp])
    hi = logistic(logit_ldi[grp] + 1.96*se_ldi[grp])
    ax2.plot(x, p, marker="o", linewidth=2.5, color=COLORS[grp], label=grp, markersize=8)
    ax2.fill_between(x, lo, hi, color=COLORS[grp], alpha=0.15)

# annotations at Pre (x=2), mirroring REC style with interaction OR and p values
p_task_pre = logistic(logit_ldi["Task-Only"][2])
p_both_pre = logistic(logit_ldi["Both"][2])
ax2.annotate("Pre × Task-Only: OR=1.10, p=.357",
             xy=(2, p_task_pre), xytext=(1.15, p_task_pre+0.07),
             arrowprops=dict(arrowstyle="->", lw=1.2, color="#444"),
             fontsize=ANNO_FONTSIZE, fontweight="bold", color="#333")
ax2.annotate("Pre × Both: OR=0.94, p=.547",
             xy=(2, p_both_pre), xytext=(1.2, p_both_pre-0.10),
             arrowprops=dict(arrowstyle="->", lw=1.2, color="#444"),
             fontsize=ANNO_FONTSIZE, fontweight="bold", color="#333")

ax2.set_xticks(x)
ax2.set_xticklabels(xlabels, fontsize=14)
ax2.set_ylabel("Predicted P(Similar | Lure) — GLMM OR estimate", fontsize=13)
ax2.set_xlabel("Boundary Position", fontsize=13)
ax2.set_title("GLMM-Predicted Lure Discrimination (LDI) by Position and Group", fontsize=14, fontweight="bold")
ax2.legend(frameon=True, fontsize=12)
ax2.spines[["top","right"]].set_visible(False)
ax2.tick_params(axis='y', labelsize=12)
plt.tight_layout()
plt.savefig("output/fig_LDI_predicted_prob.png", dpi=300, bbox_inches="tight")
plt.close()

# metadata
for fname, cap, desc in [
    ("output/fig_REC_predicted_prob.png",
     "GLMM-Predicted REC by Position and Group",
     "Line plot with 95% CI bands showing post-boundary REC drop for Both and Task-Only groups"),
    ("output/fig_LDI_predicted_prob.png",
     "GLMM-Predicted LDI by Position and Group",
     "Line plot with 95% CI bands showing flat LDI across positions; Both group lower overall"),
]:
    with open(fname+".meta.json","w") as f:
        json.dump({"caption": cap, "description": desc}, f)

print("Done")