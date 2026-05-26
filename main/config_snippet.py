# config.py  — add this line alongside your existing MASK_HSV setting
# ──────────────────────────────────────────────────────────────────────────────

MASK_HSV = [(0, 96, 78), (80, 255, 220)]   # your existing HSV range (keep as-is)

# Hybrid segmentation frame-skip interval (default used when no CLI flag or
# environment variable is set).
#   0  → pure HSV only, no K-Means/GrabCut (fastest, original behaviour)
#   1  → run full hybrid pipeline every frame (best mask quality, slowest)
#   3  → hybrid every 3rd frame, HSV fill in-between (recommended balance)
#   5+ → even faster; mask refreshes less often
HYBRID_INTERVAL = 3
 


##0 = maximum skipping (hybrid never runs)
##1 = zero skipping (hybrid runs every single frame)
##5 = run every 5th frame (4 frames skipped in between)